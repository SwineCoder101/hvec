use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use hvec_bench::judge::{self, JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT};
use hvec_connect::{ChatRequest, Message};
use hvec_store::RunRow;

use crate::cli::{JudgeArgs, JudgeCommand};
use crate::context::Ctx;

pub async fn run(config: &Option<PathBuf>, cmd: JudgeCommand) -> Result<()> {
    match cmd {
        JudgeCommand::Run(args) => judge_cmd(config, args).await,
        JudgeCommand::Agreement { batch, set, json } => agreement_cmd(config, batch.as_deref(), set.as_deref(), json),
    }
}

async fn judge_cmd(config: &Option<PathBuf>, args: JudgeArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let judge_name = ctx.chat_profile_name(args.judge.as_deref()).to_owned();
    let judge_model = hvec_connect::chat_model(&ctx.config, &judge_name)?;

    let rows = store.runs_by_kinds(&["bench", "judge"], Some(&args.batch))?;
    let already: std::collections::HashSet<String> = rows
        .iter()
        .filter(|r| r.kind == "judge")
        .filter_map(|r| r.metadata["run_id"].as_str().map(str::to_owned))
        .collect();
    let mut targets: Vec<&RunRow> = rows
        .iter()
        .filter(|r| r.kind == "bench" && r.metadata["error"].is_null())
        .filter(|r| args.rejudge || !already.contains(&r.id))
        .collect();
    if let Some(set) = &args.set {
        targets.retain(|r| r.metadata["question_set"].as_str() == Some(set));
    }
    if let Some(n) = args.limit {
        targets.truncate(n);
    }
    if targets.is_empty() {
        bail!(
            "no unjudged bench runs in batch {} ({} already judged); pass --rejudge to grade them again",
            args.batch,
            already.len()
        );
    }
    eprintln!(
        "judging {} runs from batch {} with {} ({}), prompt {JUDGE_PROMPT_VERSION}",
        targets.len(),
        args.batch,
        judge_name,
        judge_model.model_id()
    );

    let mut failures = 0usize;
    for (i, r) in targets.iter().enumerate() {
        let question = r.metadata["question"].as_str().unwrap_or("");
        let answer = r.metadata["answer"].as_str().unwrap_or("");
        let gold: Vec<String> = r.metadata["answers"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
            .unwrap_or_default();
        let chunk_ids: Vec<i64> = r.metrics["retrieved"]
            .as_array()
            .map(|a| a.iter().filter_map(|c| c["chunk_id"].as_i64()).collect())
            .unwrap_or_default();
        let passages: Vec<String> = store.chunk_texts(&chunk_ids)?.into_iter().map(|(_, t)| t).collect();
        if passages.len() != chunk_ids.len() {
            eprintln!(
                "warning: run {} retrieved {} chunks but only {} still exist; faithfulness is judged against those",
                &r.id[..8],
                chunk_ids.len(),
                passages.len()
            );
        }
        let prompt = judge::build_judge_prompt(question, &gold, &passages, answer);

        let mut row = RunRow::new("judge");
        row.collection = r.collection.clone();
        row.chat_profile = Some(judge_name.clone());
        row.chat_model = Some(judge_model.model_id().to_owned());
        row.embed_profile = r.embed_profile.clone();
        row.embed_model = r.embed_model.clone();
        row.codec = r.codec.clone();
        row.metric = r.metric.clone();
        row.metadata = serde_json::json!({
            "batch": args.batch,
            "run_id": r.id,
            "question_set": r.metadata["question_set"],
            "question_id": r.metadata["question_id"],
            "answered_by": r.chat_model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "has_gold": !gold.is_empty(),
            "passages": passages.len(),
        });

        let t = Instant::now();
        let outcome = judge_model
            .complete(ChatRequest {
                system: Some(JUDGE_SYSTEM_PROMPT.to_owned()),
                messages: vec![Message::user(prompt)],
                max_tokens: Some(args.max_tokens),
            })
            .await
            .context("judge model")
            .and_then(|resp| {
                let v = judge::parse_verdict(&resp.text).with_context(|| format!("judge reply: {}", resp.text))?;
                Ok((resp, v))
            });
        let ms = t.elapsed().as_millis() as u64;
        match outcome {
            Ok((resp, verdict)) => {
                row.metadata["judge_model_served"] = serde_json::json!(resp.model);
                row.metadata["reason"] = serde_json::json!(verdict.reason);
                row.metrics = serde_json::json!({
                    "correct": verdict.correct,
                    "faithful": verdict.faithful,
                    "generation_ms": ms,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                });
                let cheap = r.metrics["contains"].as_bool() == Some(true);
                eprintln!(
                    "[{}/{}] {} · {} → correct {}, faithful {}{}",
                    i + 1,
                    targets.len(),
                    r.collection.as_deref().unwrap_or("-"),
                    r.metadata["question_id"].as_str().unwrap_or("?"),
                    verdict.correct.map_or("n/a".to_owned(), |b| b.to_string()),
                    verdict.faithful,
                    match verdict.correct {
                        Some(c) if c != cheap => format!(" (contains said {cheap})"),
                        _ => String::new(),
                    }
                );
                store.insert_run(&row)?;
            }
            Err(e) => {
                failures += 1;
                eprintln!("[{}/{}] {} → ERROR {e:#}", i + 1, targets.len(), &r.id[..8]);
                if args.fail_fast {
                    bail!("stopping after first error (--fail-fast)");
                }
            }
        }
    }
    eprintln!(
        "\n{} verdicts recorded{}",
        targets.len() - failures,
        if failures > 0 {
            format!(", {failures} errors (not recorded; rerun to retry)")
        } else {
            String::new()
        }
    );
    let rows = store.runs_by_kinds(&["bench", "judge"], Some(&args.batch))?;
    print_agreement(&judge::agreement(&rows));
    eprintln!("\nreport with: hvec report --batch {}", args.batch);
    Ok(())
}

fn agreement_cmd(config: &Option<PathBuf>, batch: Option<&str>, set: Option<&str>, json: bool) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let mut rows = store.runs_by_kinds(&["bench", "judge"], batch)?;
    if let Some(set) = set {
        rows.retain(|r| r.metadata["question_set"].as_str() == Some(set));
    }
    let a = judge::agreement(&rows);
    if json {
        println!("{}", serde_json::to_string_pretty(&a)?);
        return Ok(());
    }
    if a.n == 0 {
        println!("no judged runs with gold answers; run `hvec judge run --batch <id>` first");
        return Ok(());
    }
    print_agreement(&a);
    Ok(())
}

fn print_agreement(a: &judge::Agreement) {
    if a.n == 0 {
        return;
    }
    println!("judge against the cheap rules, {} judged runs with gold answers:", a.n);
    println!(
        "  correct by judge {:.3} · exact {:.3} · contains {:.3}",
        a.judge_true as f64 / a.n as f64,
        a.exact_true as f64 / a.n as f64,
        a.contains_true as f64 / a.n as f64
    );
    println!(
        "  agreement with exact {:.3}, with contains {:.3}",
        a.agree_exact, a.agree_contains
    );
    println!(
        "  judge correct but contains wrong: {} (paraphrases the rule missed) · contains right but judge wrong: {} (substring false positives)",
        a.judge_only, a.contains_only
    );
}
