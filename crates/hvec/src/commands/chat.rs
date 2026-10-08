use std::io::Write as _;
use std::path::PathBuf;

use anyhow::Result;
use hvec_bench::QueryRecord;
use hvec_connect::{Message, Role};
use hvec_core::codec_by_name;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::cli::ChatArgs;
use crate::commands::rag;
use crate::context::Ctx;

pub async fn run(config: &Option<PathBuf>, args: ChatArgs) -> Result<()> {
    let ctx = Ctx::load(config.as_deref())?;
    let store = ctx.open_store()?;
    let collection = store.require_collection(&args.collection)?;
    let chat_name = ctx.chat_profile_name(args.chat.as_deref()).to_owned();

    let embedder = hvec_connect::embedder(&ctx.config, &collection.embed_profile).await?;
    let chat = hvec_connect::chat_model(&ctx.config, &chat_name)?;
    let codec = codec_by_name(&collection.codec, collection.dimension)?;
    let session = store.ensure_session(&args.session, Some(&collection.name), &chat_name)?;
    let pipeline = rag::Pipeline {
        store: &store,
        collection: &collection,
        codec: codec.as_ref(),
        embedder,
        chat: chat.clone(),
    };

    // Replay the stored transcript as plain turns. Retrieved context is not
    // stored with the history, so earlier passages do not bloat later prompts.
    let mut history: Vec<Message> = store
        .messages(&session.name)?
        .into_iter()
        .filter_map(|m| {
            Role::parse(&m.role).map(|role| Message {
                role,
                content: m.content,
            })
        })
        .collect();

    eprintln!(
        "session `{}` ({} prior messages) | collection `{}` | model {} | codec {} | k={}. Ctrl-D or /quit to exit.",
        session.name,
        history.len(),
        collection.name,
        chat.model_id(),
        collection.codec,
        args.k
    );

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        print!("you> ");
        std::io::stdout().flush()?;
        let Some(line) = lines.next_line().await? else { break };
        let question = line.trim();
        if question.is_empty() {
            continue;
        }
        if question == "/quit" || question == "/exit" {
            break;
        }

        let outcome = pipeline.answer(history.clone(), question, args.k).await?;

        store.append_message(&session.name, "user", question)?;
        store.append_message(&session.name, "assistant", &outcome.response.text)?;
        history.push(Message::user(question));
        history.push(Message::assistant(outcome.response.text.clone()));

        let record = QueryRecord {
            kind: "chat",
            collection: collection.name.clone(),
            chat_profile: chat_name.clone(),
            chat_model_configured: chat.model_id().to_owned(),
            chat_model_reported: outcome.response.model.clone(),
            embed_profile: collection.embed_profile.clone(),
            embed_model: collection.embed_model.clone(),
            dimension: collection.dimension,
            codec: collection.codec.clone(),
            metric: collection.metric.name().to_owned(),
            session: Some(session.name.clone()),
            question: question.to_owned(),
            answer: outcome.response.text.clone(),
            stop_reason: outcome.response.stop_reason.clone(),
            metrics: outcome.metrics.clone(),
        };
        store.insert_run(&record.into_row())?;

        println!("\n{}\n", outcome.response.text.trim());
        let m = &outcome.metrics;
        eprintln!(
            "[retrieve {}ms, generate {}ms | {} in / {} out tokens]\n",
            m.retrieval_ms, m.generation_ms, m.input_tokens, m.output_tokens
        );
    }
    eprintln!("saved session `{}`", session.name);
    Ok(())
}
