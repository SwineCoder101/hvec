use std::path::PathBuf;

use anyhow::Result;

use crate::cli::{CollectionsCommand, RunsCommand, SessionsCommand};
use crate::context::Ctx;

pub fn embedders() -> Result<()> {
    #[cfg(feature = "local")]
    {
        println!("{:<28} {:<44} {:>5}  description", "name", "huggingface repo", "dims");
        for m in hvec_connect::local::supported_models() {
            println!("{:<28} {:<44} {:>5}  {}", m.name, m.code, m.dimension, m.description);
        }
        println!("\nUse either column as `model` in an [embed.*] profile.");
        Ok(())
    }
    #[cfg(not(feature = "local"))]
    {
        anyhow::bail!("this build has the `local` feature disabled; only remote embedders are available")
    }
}

pub fn collections(config: &Option<PathBuf>, cmd: CollectionsCommand) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    match cmd {
        CollectionsCommand::List => {
            let cols = store.list_collections()?;
            if cols.is_empty() {
                println!("no collections yet; run `hvec ingest <paths> --collection <name>`");
                return Ok(());
            }
            println!(
                "{:<20} {:>8} {:>5}  {:<8} {:<8} embedder",
                "name", "chunks", "dims", "codec", "metric"
            );
            for c in cols {
                println!(
                    "{:<20} {:>8} {:>5}  {:<8} {:<8} {} ({})",
                    c.name,
                    c.chunk_count,
                    c.dimension,
                    c.codec,
                    c.metric.name(),
                    c.embed_profile,
                    c.embed_model
                );
            }
        }
        CollectionsCommand::Delete { name } => {
            store.delete_collection(&name)?;
            println!("deleted collection {name}");
        }
    }
    Ok(())
}

pub fn sessions(config: &Option<PathBuf>, cmd: SessionsCommand) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    match cmd {
        SessionsCommand::List => {
            let sessions = store.list_sessions()?;
            if sessions.is_empty() {
                println!("no sessions yet; run `hvec chat --session <name> --collection <name>`");
                return Ok(());
            }
            println!(
                "{:<20} {:>8}  {:<20} {:<12} created",
                "name", "messages", "collection", "chat"
            );
            for s in sessions {
                println!(
                    "{:<20} {:>8}  {:<20} {:<12} {}",
                    s.name,
                    s.message_count,
                    s.collection.as_deref().unwrap_or("-"),
                    s.chat_profile,
                    s.created_at
                );
            }
        }
        SessionsCommand::Show { name } => {
            store
                .get_session(&name)?
                .ok_or_else(|| hvec_store::StoreError::NoSuchSession(name.clone()))?;
            for m in store.messages(&name)? {
                println!("[{}] {}\n{}\n", m.created_at, m.role, m.content.trim());
            }
        }
        SessionsCommand::Delete { name } => {
            store.delete_session(&name)?;
            println!("deleted session {name}");
        }
    }
    Ok(())
}

pub fn runs(config: &Option<PathBuf>, cmd: RunsCommand) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    match cmd {
        RunsCommand::List { limit } => {
            let runs = store.list_runs(limit)?;
            if runs.is_empty() {
                println!("no runs yet; run `hvec query`");
                return Ok(());
            }
            println!(
                "{:<8} {:<24} {:<6} {:<16} {:<8} {:<22} {:>6} {:>6} {:>6}",
                "id", "created", "kind", "collection", "codec", "chat model", "ret ms", "gen ms", "out tok"
            );
            for r in runs {
                let m = &r.metrics;
                println!(
                    "{:<8} {:<24} {:<6} {:<16} {:<8} {:<22} {:>6} {:>6} {:>6}",
                    &r.id[..8],
                    r.created_at,
                    r.kind,
                    r.collection.as_deref().unwrap_or("-"),
                    r.codec.as_deref().unwrap_or("-"),
                    r.chat_model.as_deref().unwrap_or("-"),
                    m["retrieval_ms"].as_u64().unwrap_or(0),
                    m["generation_ms"].as_u64().unwrap_or(0),
                    m["output_tokens"].as_u64().unwrap_or(0),
                );
            }
        }
        RunsCommand::Show { id } => {
            let run = store.get_run(&id)?;
            println!("{}", serde_json::to_string_pretty(&run)?);
        }
    }
    Ok(())
}
