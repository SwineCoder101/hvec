//! End-to-end tests of the benchmark pipeline through the real binary:
//! ingest -> retrieve -> generate -> run log, against a mock model server.

mod support;

use support::{Env, assert_ok, stderr, stdout};

const REFUNDS: &str = "Refund policy. Customers may request a refund within 30 days of purchase. \
Refunds are issued to the original payment method. The refund window is thirty days.";
const SHIPPING: &str = "Shipping policy. Orders ship within two business days. Standard shipping \
takes three to five business days. Express shipping is available at checkout.";

/// Count data rows of a given kind in `runs list` output, skipping the header.
fn count_runs(listing: &str, kind: &str) -> usize {
    listing
        .lines()
        .skip(1)
        .filter(|l| l.split_whitespace().nth(2) == Some(kind))
        .count()
}

fn ingest_two_docs(env: &Env) {
    let a = env.write_doc("refunds.md", REFUNDS);
    let b = env.write_doc("shipping.md", SHIPPING);
    let out = env.run(&[
        "ingest",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--collection",
        "docs",
    ]);
    assert_ok(&out, "ingest");
    assert!(
        stdout(&out).contains("ingested 2 chunks from 2 files"),
        "{}",
        stdout(&out)
    );
}

#[test]
fn ingest_creates_collection_bound_to_embedder() {
    let env = Env::new();
    ingest_two_docs(&env);
    let out = env.run(&["collections", "list"]);
    assert_ok(&out, "collections list");
    let s = stdout(&out);
    assert!(
        s.contains("docs") && s.contains("128") && s.contains("f32") && s.contains("hash-128"),
        "{s}"
    );
}

#[test]
fn query_retrieves_relevant_chunk_and_records_run() {
    let env = Env::new();
    ingest_two_docs(&env);

    let out = env.run(&[
        "query",
        "what is the refund window?",
        "--collection",
        "docs",
        "-k",
        "1",
        "--json",
    ]);
    assert_ok(&out, "query");
    let row: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("json run row");

    assert_eq!(row["kind"], "query");
    assert_eq!(row["codec"], "f32");
    assert_eq!(row["metric"], "cosine");
    assert_eq!(row["embed_model"], "hash-128");
    assert_eq!(row["chat_profile"], "mock-anthropic");
    assert_eq!(row["chat_model"], "claude-opus-5-5-served");
    assert_eq!(row["metadata"]["prompt_version"], "v1");
    assert_eq!(row["metadata"]["dimension"], 128);
    assert_eq!(row["metadata"]["answer"], "MOCK-ANTHROPIC: 1 msgs");

    let m = &row["metrics"];
    assert_eq!(m["k"], 1);
    assert_eq!(m["input_tokens"], 123);
    assert_eq!(m["output_tokens"], 7);
    let retrieved = m["retrieved"].as_array().expect("retrieved");
    assert_eq!(retrieved.len(), 1);
    assert!(
        retrieved[0]["source"].as_str().unwrap().ends_with("refunds.md"),
        "{retrieved:?}"
    );
    assert!(retrieved[0]["score"].as_f64().unwrap() > 0.0);

    // The run is in the log and retrievable by id prefix.
    let id = row["id"].as_str().unwrap();
    let list = env.run(&["runs", "list"]);
    assert_ok(&list, "runs list");
    assert!(stdout(&list).contains(&id[..8]));
    let show = env.run(&["runs", "show", &id[..8]]);
    assert_ok(&show, "runs show");
    let back: serde_json::Value = serde_json::from_str(&stdout(&show)).unwrap();
    assert_eq!(back["id"], id);
}

#[test]
fn anthropic_wire_format_is_correct() {
    let env = Env::new();
    ingest_two_docs(&env);
    assert_ok(
        &env.run(&["query", "refund?", "--collection", "docs", "-k", "2"]),
        "query",
    );

    let reqs = env.server.anthropic_requests();
    assert_eq!(reqs.len(), 1);
    let r = &reqs[0];
    assert_eq!(r.headers.get("x-api-key").map(String::as_str), Some("test-key-123"));
    assert_eq!(
        r.headers.get("anthropic-version").map(String::as_str),
        Some("2023-06-01")
    );
    assert_eq!(
        r.headers.get("anthropic-beta").map(String::as_str),
        Some("server-side-fallback-2026-07-01")
    );
    assert_eq!(r.body["model"], "claude-opus-5-5");
    assert_eq!(r.body["fallbacks"], "default");
    assert_eq!(r.body["output_config"]["effort"], "medium");
    assert_eq!(r.body["max_tokens"], 4096);
    assert!(
        r.body["system"]
            .as_str()
            .unwrap()
            .starts_with("You answer questions using only")
    );
    let user = r.body["messages"][0]["content"].as_str().unwrap();
    assert!(user.starts_with("Context passages:"), "{user}");
    assert!(
        user.contains("[1]") && user.contains("[2]"),
        "two passages expected: {user}"
    );
    assert!(user.ends_with("Question: refund?"), "{user}");
}

#[test]
fn openai_compatible_wire_format_is_correct() {
    let env = Env::new();
    ingest_two_docs(&env);
    let out = env.run(&[
        "query",
        "shipping time?",
        "--collection",
        "docs",
        "--chat",
        "mock-openai",
        "-k",
        "1",
        "--json",
    ]);
    assert_ok(&out, "query openai");
    let row: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
    assert_eq!(row["chat_model"], "fake-model");
    assert_eq!(row["metrics"]["input_tokens"], 45);
    assert!(
        row["metrics"]["retrieved"][0]["source"]
            .as_str()
            .unwrap()
            .ends_with("shipping.md")
    );

    let reqs = env.server.openai_requests();
    assert_eq!(reqs.len(), 1);
    let roles: Vec<&str> = reqs[0].body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user"]);
    assert!(
        !reqs[0].headers.contains_key("authorization"),
        "no api_key_env configured, so no bearer header"
    );
}

#[test]
fn refusal_is_reported_not_recorded() {
    let env = Env::new();
    ingest_two_docs(&env);
    let out = env.run(&["query", "PLEASE-REFUSE this", "--collection", "docs"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("declined"), "{}", stderr(&out));
    let list = env.run(&["runs", "list"]);
    assert!(stdout(&list).contains("no runs yet"), "{}", stdout(&list));
}

#[test]
fn chat_session_persists_and_resumes_with_history() {
    let env = Env::new();
    ingest_two_docs(&env);

    let out = env.run_with_stdin(
        &["-s", "s1", "-c", "docs", "-k", "1"],
        "first question\nsecond question\n/quit\n",
    );
    assert_ok(&out, "chat first run");
    let so = stdout(&out);
    assert!(
        so.contains("MOCK-ANTHROPIC: 1 msgs") && so.contains("MOCK-ANTHROPIC: 3 msgs"),
        "{so}"
    );

    let out = env.run_with_stdin(&["chat", "-s", "s1", "-c", "docs", "-k", "1"], "third\n");
    assert_ok(&out, "chat resume");
    assert!(stderr(&out).contains("s1 (4 messages)"), "{}", stderr(&out));
    assert!(stdout(&out).contains("MOCK-ANTHROPIC: 5 msgs"), "{}", stdout(&out));

    let show = env.run(&["sessions", "show", "s1"]);
    assert_ok(&show, "sessions show");
    assert_eq!(stdout(&show).matches("] user\n").count(), 3);
    assert_eq!(stdout(&show).matches("] assistant\n").count(), 3);

    // Every chat turn is a benchmark run too.
    let list = env.run(&["runs", "list"]);
    assert_eq!(count_runs(&stdout(&list), "chat"), 3, "{}", stdout(&list));
}

#[test]
fn shell_commands_switch_model_and_toggle_retrieval() {
    let env = Env::new();
    ingest_two_docs(&env);
    let script = "/use none\nplain hello\n/use docs\n/model mock-openai\nrefund?\n/status\n/quit\n";
    let out = env.run_with_stdin(&["-s", "s2"], script);
    assert_ok(&out, "shell script");
    let so = stdout(&out);
    let se = stderr(&out);
    assert!(
        so.contains("MOCK-ANTHROPIC: 1 msgs"),
        "plain chat went to default profile: {so}"
    );
    assert!(
        // system prompt + 2 turns of history + the new user turn
        so.contains("MOCK-OPENAI: 4 msgs"),
        "rag turn went to switched profile with history: {so}"
    );
    assert!(se.contains("not recorded"), "plain chat must not be recorded: {se}");
    assert!(se.contains("chat       mock-openai"), "{se}");

    let plain = env.server.anthropic_requests();
    assert_eq!(plain.len(), 1);
    assert!(plain[0].body["system"].as_str().unwrap().contains("concise assistant"));
    assert_eq!(count_runs(&stdout(&env.run(&["runs", "list"])), "chat"), 1);
}

#[test]
fn mixing_embedders_in_a_collection_is_refused() {
    let env = Env::new();
    ingest_two_docs(&env);
    // Add a second hash profile with a different dimension and try to ingest into the same collection.
    let cfg = std::fs::read_to_string(&env.config_path).unwrap()
        + "\n[embed.hash2]\nprovider = \"hash\"\nmodel = \"hash-64\"\ndimension = 64\n";
    std::fs::write(&env.config_path, cfg).unwrap();
    let c = env.write_doc("c.md", "more text here");
    let out = env.run(&[
        "ingest",
        c.to_str().unwrap(),
        "--collection",
        "docs",
        "--embedder",
        "hash2",
    ]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("refusing to mix"), "{}", stderr(&out));
}

#[test]
fn bench_recall_measures_codecs_against_f32_and_records_runs() {
    let env = Env::new();
    // A few more documents so top-k is meaningful.
    for i in 0..6 {
        env.write_doc(
            &format!("extra{i}.md"),
            &format!("Topic {i}: notes about subject number {i} and its details, item {i}."),
        );
    }
    let dir = env.dir.path().to_str().unwrap().to_owned();
    env.write_doc("refunds.md", REFUNDS);
    env.write_doc("shipping.md", SHIPPING);
    assert_ok(&env.run(&["ingest", &dir, "--collection", "docs"]), "ingest dir");

    let out = env.run(&[
        "bench",
        "recall",
        "--collection",
        "docs",
        "--codecs",
        "int8,binary,binary-centred",
        "-k",
        "3",
        "--sample",
        "4",
        "--json",
    ]);
    assert_ok(&out, "bench recall");
    let reports: Vec<serde_json::Value> = serde_json::from_str(&stdout(&out)).expect("json reports");
    let names: Vec<&str> = reports.iter().map(|r| r["codec"].as_str().unwrap()).collect();
    assert_eq!(names, ["f32", "int8", "binary", "binary-centred"]);
    assert_eq!(reports[0]["recall_at_k"], 1.0);
    assert_eq!(reports[0]["mean_abs_score_error"], 0.0);
    assert_eq!(reports[0]["queries"], 4);
    assert_eq!(reports[0]["vectors"], 8);
    assert!(
        reports[1]["recall_at_k"].as_f64().unwrap() >= 0.5,
        "int8: {}",
        reports[1]
    );
    assert_eq!(reports[1]["bytes_per_vector"], 136);
    assert_eq!(reports[2]["bytes_per_vector"], 16);
    assert_eq!(reports[2]["compression_ratio"], 32.0);
    // The trained codec is fitted on the collection and costs no extra bytes per vector.
    assert_eq!(reports[3]["bytes_per_vector"], 16);
    assert_eq!(reports[3]["compression_ratio"], 32.0);
    assert!(reports[3]["recall_at_k"].as_f64().unwrap() > 0.0);

    let list = env.run(&["runs", "list"]);
    assert_eq!(count_runs(&stdout(&list), "recall"), 4, "{}", stdout(&list));

    // A query file is embedded with the collection's embedder.
    let qf = env.write_doc("queries.txt", "refund window\nshipping time\n");
    let out = env.run(&[
        "bench",
        "recall",
        "--collection",
        "docs",
        "--queries",
        qf.to_str().unwrap(),
        "--no-record",
        "--json",
    ]);
    assert_ok(&out, "bench recall with query file");
    let reports: Vec<serde_json::Value> = serde_json::from_str(&stdout(&out)).unwrap();
    assert_eq!(reports[0]["queries"], 2);
    assert_eq!(
        count_runs(&stdout(&env.run(&["runs", "list"])), "recall"),
        4,
        "--no-record must not add runs"
    );
}

#[test]
fn bench_recall_refuses_non_f32_collection() {
    let env = Env::new();
    let a = env.write_doc("a.md", REFUNDS);
    assert_ok(
        &env.run(&["ingest", a.to_str().unwrap(), "--collection", "d8", "--codec", "int8"]),
        "ingest int8",
    );
    let out = env.run(&["bench", "recall", "--collection", "d8"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("needs an f32 collection"), "{}", stderr(&out));
}

#[test]
fn query_against_int8_and_binary_collections_records_the_codec() {
    let env = Env::new();
    let a = env.write_doc("refunds.md", REFUNDS);
    let b = env.write_doc("shipping.md", SHIPPING);
    for codec in ["int8", "binary"] {
        let col = format!("docs-{codec}");
        assert_ok(
            &env.run(&[
                "ingest",
                a.to_str().unwrap(),
                b.to_str().unwrap(),
                "--collection",
                &col,
                "--codec",
                codec,
            ]),
            "ingest",
        );
        let out = env.run(&[
            "query",
            "what is the refund window?",
            "--collection",
            &col,
            "-k",
            "1",
            "--json",
        ]);
        assert_ok(&out, "query");
        let row: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
        assert_eq!(row["codec"], codec);
        assert!(
            row["metrics"]["retrieved"][0]["source"]
                .as_str()
                .unwrap()
                .ends_with("refunds.md"),
            "{codec}: {row}"
        );
    }
    let list = stdout(&env.run(&["collections", "list"]));
    assert!(list.contains("int8") && list.contains("binary"), "{list}");
}

#[test]
fn binary_centred_collection_is_fitted_once_and_reused_on_append() {
    let env = Env::new();
    let a = env.write_doc("refunds.md", REFUNDS);
    let b = env.write_doc("shipping.md", SHIPPING);
    let c = env.write_doc(
        "warranty.md",
        "Warranty: every device is covered against manufacturing defects for two years from purchase.",
    );
    // The first ingest fits the mean on everything it sees.
    let out = env.run(&[
        "ingest",
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--collection",
        "centred",
        "--codec",
        "binary-centred",
    ]);
    assert_ok(&out, "ingest fitted");
    assert!(stdout(&out).contains("codec binary-centred"), "{}", stdout(&out));
    assert!(stderr(&out).contains("fitted binary-centred on"), "{}", stderr(&out));
    let list = stdout(&env.run(&["collections", "list"]));
    assert!(list.contains("binary-centred"), "{list}");

    // Appending reuses the stored parameters instead of refitting, so the
    // vectors already stored stay comparable with the new ones.
    let out = env.run(&[
        "ingest",
        c.to_str().unwrap(),
        "--collection",
        "centred",
        "--codec",
        "binary-centred",
    ]);
    assert_ok(&out, "ingest append");
    assert!(!stderr(&out).contains("fitted binary-centred"), "{}", stderr(&out));
    assert!(stdout(&out).contains("3 chunks total"), "{}", stdout(&out));

    for (question, source) in [
        ("what is the refund window?", "refunds.md"),
        ("how long does shipping take?", "shipping.md"),
        ("how long is the warranty?", "warranty.md"),
    ] {
        let out = env.run(&["query", question, "--collection", "centred", "-k", "1", "--json"]);
        assert_ok(&out, "query");
        let row: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
        assert_eq!(row["codec"], "binary-centred");
        assert!(
            row["metrics"]["retrieved"][0]["source"]
                .as_str()
                .unwrap()
                .ends_with(source),
            "{question}: {row}"
        );
    }

    // A different codec into the same collection is refused.
    let out = env.run(&[
        "ingest",
        a.to_str().unwrap(),
        "--collection",
        "centred",
        "--codec",
        "binary",
    ]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("stores binary-centred vectors"),
        "{}",
        stderr(&out)
    );

    // bench recall needs f32 ground truth, so a fitted collection is refused too.
    let out = env.run(&["bench", "recall", "--collection", "centred"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("needs an f32 collection"), "{}", stderr(&out));
}

#[test]
fn bench_run_scores_answers_and_report_aggregates_cells() {
    let env = Env::new();
    let a = env.write_doc("refunds.md", REFUNDS);
    let b = env.write_doc("shipping.md", SHIPPING);
    for (col, codec) in [("qa-f32", "f32"), ("qa-bin", "binary")] {
        assert_ok(
            &env.run(&[
                "ingest",
                a.to_str().unwrap(),
                b.to_str().unwrap(),
                "--collection",
                col,
                "--codec",
                codec,
            ]),
            "ingest",
        );
    }
    // The mock answers "<x>" for "say exactly <x>", so q1 is correct, q2 is wrong, q3 has no gold.
    let qs = env.write_doc(
        "refund-qa.jsonl",
        concat!(
            "# refund questions\n",
            "{\"id\":\"q1\",\"question\":\"refund window? say exactly Thirty days.\",\"answers\":[\"30 days\",\"thirty days\"],\"source\":\"refunds.md\"}\n",
            "{\"id\":\"q2\",\"question\":\"shipping time? say exactly next week\",\"answers\":[\"three to five business days\"],\"source\":\"shipping.md\"}\n",
            "{\"id\":\"q3\",\"question\":\"anything? say exactly whatever\"}\n",
        ),
    );
    let out = env.run(&[
        "bench",
        "run",
        "--questions",
        qs.to_str().unwrap(),
        "--collections",
        "qa-f32,qa-bin",
        "--chat",
        "mock-anthropic,mock-openai",
        "-k",
        "1",
        "--label",
        "exp-test",
    ]);
    assert_ok(&out, "bench run");
    let se = stderr(&out);
    assert!(
        se.contains("3 questions × 2 collections × 2 chat profiles = 12 runs"),
        "{se}"
    );
    assert!(se.contains("12 runs recorded"), "{se}");
    let batch = se
        .lines()
        .find_map(|l| l.strip_prefix("batch "))
        .and_then(|l| l.split(':').next())
        .expect("batch id")
        .to_owned();

    // 12 bench rows; the summary printed by `bench run` has 4 cells.
    assert_eq!(
        count_runs(&stdout(&env.run(&["runs", "list", "-n", "50"])), "bench"),
        12
    );
    let cells_text = stdout(&out);
    assert_eq!(
        cells_text
            .lines()
            .skip(1)
            .filter(|l| l.starts_with("refund-qa"))
            .count(),
        4,
        "{cells_text}"
    );

    // Report as JSON: every cell saw 3 questions, exact 1/3 (q1), contains 1/3, source hit 1/2 (q1 yes, q2 no; q3 unhinted).
    let out = env.run(&["report", "--batch", &batch, "--json"]);
    assert_ok(&out, "report");
    let cells: Vec<serde_json::Value> = serde_json::from_str(&stdout(&out)).unwrap();
    assert_eq!(cells.len(), 4);
    for c in &cells {
        assert_eq!(c["n"], 3, "{c}");
        assert_eq!(c["errors"], 0);
        assert!((c["exact"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-9, "{c}");
        assert!((c["contains"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-9, "{c}");
        assert_eq!(c["key"]["question_set"], "refund-qa");
    }
    let f32_cells: Vec<&serde_json::Value> = cells.iter().filter(|c| c["key"]["codec"] == "f32").collect();
    assert_eq!(f32_cells.len(), 2);
    assert_eq!(
        f32_cells[0]["source_hit"], 1.0,
        "f32 retrieval finds both hinted sources: {}",
        f32_cells[0]
    );
    let models: Vec<&str> = cells.iter().map(|c| c["key"]["chat_model"].as_str().unwrap()).collect();
    assert!(
        models.contains(&"claude-opus-5-5") && models.contains(&"fake-model"),
        "cells group by configured model: {models:?}"
    );
    let opus = cells
        .iter()
        .find(|c| c["key"]["chat_model"] == "claude-opus-5-5")
        .unwrap();
    assert_eq!(
        opus["served_other"], 3,
        "the mock reports a '-served' suffix, so every run counts as rerouted"
    );

    // Filters.
    let out = env.run(&["report", "--batch", "nope"]);
    assert!(stdout(&out).contains("no bench runs match"));
    let out = env.run(&["report", "--set", "refund-qa"]);
    assert_eq!(
        stdout(&out).lines().filter(|l| l.starts_with("refund-qa")).count(),
        4,
        "{}",
        stdout(&out)
    );
    assert!(
        stdout(&out).contains("6 run(s) were served by a different model"),
        "{}",
        stdout(&out)
    );

    // A run row carries the question, gold answers, answer and scores.
    let one = env.run(&["runs", "list", "-n", "1"]);
    let id = stdout(&one)
        .lines()
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned();
    let row: serde_json::Value = serde_json::from_str(&stdout(&env.run(&["runs", "show", &id]))).unwrap();
    assert_eq!(row["kind"], "bench");
    assert_eq!(row["metadata"]["label"], "exp-test");
    assert_eq!(row["metadata"]["batch"], batch);
    assert!(row["metrics"]["exact"].is_boolean() && row["metrics"]["contains"].is_boolean());
}

#[test]
fn bench_run_records_model_errors_and_continues() {
    let env = Env::new();
    let a = env.write_doc("refunds.md", REFUNDS);
    assert_ok(
        &env.run(&["ingest", a.to_str().unwrap(), "--collection", "qa"]),
        "ingest",
    );
    let qs = env.write_doc("qs.jsonl", "{\"id\":\"bad\",\"question\":\"PLEASE-REFUSE this\",\"answers\":[\"x\"]}\n{\"id\":\"good\",\"question\":\"say exactly x\",\"answers\":[\"x\"]}\n");
    let out = env.run(&[
        "bench",
        "run",
        "--questions",
        qs.to_str().unwrap(),
        "--collections",
        "qa",
    ]);
    assert_ok(&out, "bench run with a refusal");
    assert!(stderr(&out).contains("1 errors"), "{}", stderr(&out));
    let cells: Vec<serde_json::Value> = serde_json::from_str(&stdout(&env.run(&["report", "--json"]))).unwrap();
    assert_eq!(cells[0]["n"], 2);
    assert_eq!(cells[0]["errors"], 1);
    assert_eq!(cells[0]["exact"], 1.0, "the good run is the only one scored");

    let out = env.run(&[
        "bench",
        "run",
        "--questions",
        qs.to_str().unwrap(),
        "--collections",
        "qa",
        "--fail-fast",
    ]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("--fail-fast"));
}
