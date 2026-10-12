use anyhow::{Context as _, Result};
use hvec_bench::beir;
use hvec_bench::questions::{self, Question};

use crate::cli::{ImportBeirArgs, QuestionsCommand};

pub fn run(cmd: QuestionsCommand) -> Result<()> {
    match cmd {
        QuestionsCommand::ImportBeir(args) => import_beir(&args),
        QuestionsCommand::Check { path } => {
            let qs = questions::load(&path)?;
            describe(&qs);
            Ok(())
        }
    }
}

fn import_beir(args: &ImportBeirArgs) -> Result<()> {
    let mut qs = beir::import(&args.dir, &args.split)
        .with_context(|| format!("importing BEIR dataset at {}", args.dir.display()))?;
    let labelled = match &args.scifact_claims {
        Some(claims) => beir::apply_scifact_labels(&mut qs, claims)
            .with_context(|| format!("applying SciFact labels from {}", claims.display()))?,
        None => 0,
    };
    if let Some(n) = args.limit {
        qs.truncate(n);
    }
    let judgments: usize = qs.iter().map(|q| q.relevant.len()).sum();
    let mut header = format!(
        "# Imported from BEIR `{}` ({} split) by `hvec questions import-beir`: {} questions, {} judgments",
        args.dir.file_name().and_then(|s| s.to_str()).unwrap_or("dataset"),
        args.split,
        qs.len(),
        judgments
    );
    if args.scifact_claims.is_some() {
        header.push_str(&format!(", {labelled} with supports/refutes gold answers"));
    }
    header.push('\n');
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&args.out, header + &questions::to_jsonl(&qs))
        .with_context(|| format!("writing {}", args.out.display()))?;
    eprintln!("wrote {} questions to {}", qs.len(), args.out.display());
    if let Some(dir) = &args.corpus_out {
        let n = beir::write_corpus(&args.dir, dir)?;
        eprintln!("wrote {n} documents to {}", dir.display());
    }
    describe(&qs);
    Ok(())
}

fn describe(qs: &[Question]) {
    let with_answers = qs.iter().filter(|q| !q.answers.is_empty()).count();
    let with_source = qs.iter().filter(|q| q.source.is_some()).count();
    let with_relevant = qs.iter().filter(|q| !q.relevant.is_empty()).count();
    let with_query = qs.iter().filter(|q| q.query.is_some()).count();
    let judgments: usize = qs.iter().map(|q| q.relevant.len()).sum();
    println!("{} questions", qs.len());
    println!("  {with_answers} with gold answers (exact / contains scoring)");
    println!("  {with_source} with a `source` hint (source-hit scoring)");
    println!("  {with_relevant} with relevance judgments, {judgments} judgments in all (nDCG, document recall)");
    if with_query > 0 {
        println!("  {with_query} with a separate retrieval `query`");
    }
    if let Some(q) = qs.first() {
        println!("first: {} · {}", q.id, q.question);
    }
}
