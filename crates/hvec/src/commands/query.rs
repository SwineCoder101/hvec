use std::path::PathBuf;

use anyhow::Result;
use hvec_bench::QueryRecord;

use crate::cli::QueryArgs;
use crate::commands::rag;
use crate::context::Ctx;

pub async fn run(config: &Option<PathBuf>, args: QueryArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let collection = store.require_collection(&args.collection)?;
    let chat_name = ctx.chat_profile_name(args.chat.as_deref()).to_owned();

    let embedder = hvec_connect::embedder(&ctx.config, &collection.embed_profile).await?;
    let chat = hvec_connect::chat_model(&ctx.config, &chat_name)?;
    let codec = collection.codec()?;

    let pipeline = rag::Pipeline {
        store: &store,
        collection: &collection,
        codec: codec.as_ref(),
        embedder,
        chat: chat.clone(),
    };
    let outcome = pipeline.answer(Vec::new(), &args.question, args.k).await?;

    let record = QueryRecord {
        kind: "query",
        collection: collection.name.clone(),
        chat_profile: chat_name,
        chat_model_configured: chat.model_id().to_owned(),
        chat_model_reported: outcome.response.model.clone(),
        embed_profile: collection.embed_profile.clone(),
        embed_model: collection.embed_model.clone(),
        dimension: collection.dimension,
        codec: collection.codec.clone(),
        metric: collection.metric.name().to_owned(),
        session: None,
        question: args.question.clone(),
        answer: outcome.response.text.clone(),
        stop_reason: outcome.response.stop_reason.clone(),
        metrics: outcome.metrics.clone(),
    };
    let row = record.into_row();
    if !args.no_record {
        store.insert_run(&row)?;
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&row)?);
        return Ok(());
    }
    if args.show_context {
        rag::print_context(&outcome.hits);
    }
    println!("{}", outcome.response.text.trim());
    let m = &outcome.metrics;
    eprintln!(
        "\n[{} | {} | k={} | embed {}ms, retrieve {}ms, generate {}ms | {} in / {} out tokens{}]",
        outcome.response.model,
        collection.codec,
        m.k,
        m.embed_ms,
        m.retrieval_ms,
        m.generation_ms,
        m.input_tokens,
        m.output_tokens,
        if args.no_record { "" } else { " | recorded" }
    );
    Ok(())
}
