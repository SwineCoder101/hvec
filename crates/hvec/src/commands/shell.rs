//! The interactive shell behind bare `hvec`.
//!
//! Plain text is a chat turn. Lines starting with `/` are commands. The
//! shell owns one SQLite connection, one session, an optional collection,
//! and lazily built model connectors. It runs on a blocking thread and
//! drives async work through a runtime handle so line editing can block.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use hvec_bench::QueryRecord;
use hvec_connect::{ChatModel, ChatRequest, Embedder, Message, Role};
use hvec_core::{Codec, codec_by_name};
use hvec_store::{Collection, Session, Store};
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, config::Configurer};
use tokio::runtime::Handle;

use crate::cli::{IngestArgs, ShellArgs};
use crate::commands::rag::{self, Pipeline};
use crate::context::Ctx;

const PLAIN_SYSTEM_PROMPT: &str = "You are a concise assistant running inside the hvec command-line tool.";

pub async fn run(config: &Option<PathBuf>, args: ShellArgs) -> Result<()> {
    let ctx = Ctx::load_or_init(config.as_deref())?;
    let handle = Handle::current();
    tokio::task::spawn_blocking(move || Shell::start(handle, ctx, args)?.run_loop())
        .await
        .context("shell thread panicked")?
}

struct Shell {
    handle: Handle,
    ctx: Ctx,
    store: Store,
    session: Session,
    history: Vec<Message>,
    collection: Option<Collection>,
    codec: Option<Box<dyn Codec>>,
    chat_name: String,
    chat: Option<Arc<dyn ChatModel>>,
    embedders: HashMap<String, Arc<dyn Embedder>>,
    k: usize,
    show_context: bool,
    editor: DefaultEditor,
    history_path: PathBuf,
}

enum Flow {
    Continue,
    Quit,
}

impl Shell {
    fn start(handle: Handle, ctx: Ctx, args: ShellArgs) -> Result<Self> {
        let store = ctx.open_store()?;
        let chat_name = ctx.chat_profile_name(args.chat.as_deref()).to_owned();

        let collection = match args.collection.as_deref() {
            Some(name) => Some(store.require_collection(name)?),
            None => {
                let mut all = store.list_collections()?;
                if all.len() == 1 { Some(all.remove(0)) } else { None }
            }
        };
        let codec = collection
            .as_ref()
            .map(|c| codec_by_name(&c.codec, c.dimension))
            .transpose()?;

        let session_name = args
            .session
            .unwrap_or_else(|| chrono::Local::now().format("%Y%m%d-%H%M%S").to_string());
        let session = store.ensure_session(&session_name, collection.as_ref().map(|c| c.name.as_str()), &chat_name)?;
        let history = load_history(&store, &session.name)?;

        let mut editor = DefaultEditor::new()?;
        editor.set_auto_add_history(true);
        let history_path = ctx.config.db_path().with_file_name("shell-history.txt");
        let _ = editor.load_history(&history_path);

        let shell = Self {
            handle,
            ctx,
            store,
            session,
            history,
            collection,
            codec,
            chat_name,
            chat: None,
            embedders: HashMap::new(),
            k: args.k,
            show_context: false,
            editor,
            history_path,
        };
        shell.banner();
        Ok(shell)
    }

    fn banner(&self) {
        eprintln!(
            "hvec {} | config {} | db {}",
            env!("CARGO_PKG_VERSION"),
            self.ctx.config_path.display(),
            self.ctx.config.db_path().display()
        );
        self.status();
        eprintln!("Type a question, or /help for commands. Ctrl-D or /quit to exit.\n");
    }

    fn status(&self) {
        let col = match &self.collection {
            Some(c) => format!(
                "{} ({} chunks, {} dims, codec {}, {})",
                c.name, c.chunk_count, c.dimension, c.codec, c.embed_model
            ),
            None => "none (plain chat, no retrieval; use /use <collection>)".to_owned(),
        };
        let model = self
            .ctx
            .config
            .chat
            .get(&self.chat_name)
            .map(|p| p.model.as_str())
            .unwrap_or("?");
        eprintln!("session    {} ({} messages)", self.session.name, self.history.len());
        eprintln!("collection {col}");
        eprintln!(
            "chat       {} ({model}) | k={} | show-context={}",
            self.chat_name, self.k, self.show_context
        );
    }

    fn run_loop(mut self) -> Result<()> {
        loop {
            let prompt = match &self.collection {
                Some(c) => format!("hvec[{}]> ", c.name),
                None => "hvec> ".to_owned(),
            };
            let line = match self.editor.readline(&prompt) {
                Ok(l) => l,
                Err(ReadlineError::Interrupted) => {
                    eprintln!("(Ctrl-C) use /quit or Ctrl-D to exit");
                    continue;
                }
                Err(ReadlineError::Eof) => break,
                Err(e) => return Err(e.into()),
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let flow = if let Some(cmd) = line.strip_prefix('/') {
                self.command(cmd)
            } else {
                self.turn(line)
            };
            match flow {
                Ok(Flow::Continue) => {}
                Ok(Flow::Quit) => break,
                Err(e) => eprintln!("error: {e:#}"),
            }
        }
        let _ = self.editor.save_history(&self.history_path);
        eprintln!(
            "saved session `{}`. Resume with: hvec -s {}",
            self.session.name, self.session.name
        );
        Ok(())
    }

    fn command(&mut self, cmd: &str) -> Result<Flow> {
        let mut parts = cmd.split_whitespace();
        let name = parts.next().unwrap_or("");
        let rest: Vec<&str> = parts.collect();
        match name {
            "quit" | "exit" | "q" => return Ok(Flow::Quit),
            "help" | "?" => print_help(),
            "status" => self.status(),
            "collections" => crate::commands::list::collections(
                &Some(self.ctx.config_path.clone()),
                crate::cli::CollectionsCommand::List,
            )?,
            "sessions" => {
                crate::commands::list::sessions(&Some(self.ctx.config_path.clone()), crate::cli::SessionsCommand::List)?
            }
            "runs" => {
                let limit = rest.first().and_then(|s| s.parse().ok()).unwrap_or(10);
                crate::commands::list::runs(
                    &Some(self.ctx.config_path.clone()),
                    crate::cli::RunsCommand::List { limit },
                )?;
            }
            "embedders" => crate::commands::list::embedders()?,
            "use" => match rest.first() {
                Some(&"none") => {
                    self.collection = None;
                    self.codec = None;
                    eprintln!("retrieval off; plain chat");
                }
                Some(name) => {
                    let c = self.store.require_collection(name)?;
                    self.codec = Some(codec_by_name(&c.codec, c.dimension)?);
                    eprintln!("using collection `{}` ({} chunks)", c.name, c.chunk_count);
                    self.collection = Some(c);
                }
                None => eprintln!("usage: /use <collection> | /use none"),
            },
            "model" | "chat" => match rest.first() {
                Some(name) => {
                    self.ctx.config.chat_profile(name)?;
                    self.chat_name = (*name).to_owned();
                    self.chat = None;
                    eprintln!("chat profile set to `{name}`");
                }
                None => {
                    for (n, p) in &self.ctx.config.chat {
                        eprintln!(
                            "{}{:<16} {:?} {}",
                            if *n == self.chat_name { "* " } else { "  " },
                            n,
                            p.provider,
                            p.model
                        );
                    }
                }
            },
            "session" => match rest.first() {
                Some(name) => {
                    self.session = self.store.ensure_session(
                        name,
                        self.collection.as_ref().map(|c| c.name.as_str()),
                        &self.chat_name,
                    )?;
                    self.history = load_history(&self.store, &self.session.name)?;
                    eprintln!(
                        "session `{}` ({} prior messages)",
                        self.session.name,
                        self.history.len()
                    );
                }
                None => eprintln!("usage: /session <name>"),
            },
            "k" => match rest.first().and_then(|s| s.parse::<usize>().ok()) {
                Some(k) if k > 0 => {
                    self.k = k;
                    eprintln!("k = {k}");
                }
                _ => eprintln!("usage: /k <positive integer>"),
            },
            "context" => {
                self.show_context = !self.show_context;
                eprintln!("show-context = {}", self.show_context);
            }
            "ingest" => {
                let Some(col) = self.collection.as_ref().map(|c| c.name.clone()) else {
                    return Err(anyhow!(
                        "no collection selected; run /use <name> first (it is created on ingest if missing)"
                    ));
                };
                if rest.is_empty() {
                    eprintln!("usage: /ingest <path> [<path>...]");
                    return Ok(Flow::Continue);
                }
                self.ingest(&col, rest.iter().map(PathBuf::from).collect())?;
            }
            "new" => {
                let name = rest
                    .first()
                    .map(|s| (*s).to_owned())
                    .ok_or_else(|| anyhow!("usage: /new <collection> <path> [<path>...]"))?;
                let paths: Vec<PathBuf> = rest.iter().skip(1).map(PathBuf::from).collect();
                if paths.is_empty() {
                    return Err(anyhow!("usage: /new <collection> <path> [<path>...]"));
                }
                self.ingest(&name, paths)?;
            }
            other => eprintln!("unknown command /{other}; try /help"),
        }
        Ok(Flow::Continue)
    }

    fn ingest(&mut self, collection: &str, paths: Vec<PathBuf>) -> Result<()> {
        let args = IngestArgs {
            paths,
            collection: collection.to_owned(),
            embedder: None,
            codec: self
                .collection
                .as_ref()
                .map(|c| c.codec.clone())
                .unwrap_or_else(|| "f32".to_owned()),
            chunk_words: 200,
            overlap_words: 40,
            ext: ["md", "txt", "markdown", "rst"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            batch: 32,
        };
        let cfg = Some(self.ctx.config_path.clone());
        self.handle.block_on(crate::commands::ingest::run(&cfg, args))?;
        let c = self.store.require_collection(collection)?;
        self.codec = Some(codec_by_name(&c.codec, c.dimension)?);
        self.collection = Some(c);
        Ok(())
    }

    fn chat_model(&mut self) -> Result<Arc<dyn ChatModel>> {
        if let Some(c) = &self.chat {
            return Ok(Arc::clone(c));
        }
        let model = hvec_connect::chat_model(&self.ctx.config, &self.chat_name)?;
        self.chat = Some(Arc::clone(&model));
        Ok(model)
    }

    fn embedder_for(&mut self, profile: &str) -> Result<Arc<dyn Embedder>> {
        if let Some(e) = self.embedders.get(profile) {
            return Ok(Arc::clone(e));
        }
        let e = self
            .handle
            .block_on(hvec_connect::embedder(&self.ctx.config, profile))?;
        self.embedders.insert(profile.to_owned(), Arc::clone(&e));
        Ok(e)
    }

    fn turn(&mut self, question: &str) -> Result<Flow> {
        let chat = self.chat_model()?;
        let history = self.history.clone();

        let (answer, summary) = match self.collection.clone() {
            Some(collection) => {
                let embedder = self.embedder_for(&collection.embed_profile)?;
                let codec = self.codec.as_deref().ok_or_else(|| anyhow!("codec not loaded"))?;
                let pipeline = Pipeline {
                    store: &self.store,
                    collection: &collection,
                    codec,
                    embedder,
                    chat: Arc::clone(&chat),
                };
                let outcome = self.handle.block_on(pipeline.answer(history, question, self.k))?;
                if self.show_context {
                    rag::print_context(&outcome.hits);
                }
                let m = &outcome.metrics;
                let summary = format!(
                    "[{} | {} | k={} | embed {}ms, retrieve {}ms, generate {}ms | {} in / {} out tokens | recorded]",
                    outcome.response.model,
                    collection.codec,
                    m.k,
                    m.embed_ms,
                    m.retrieval_ms,
                    m.generation_ms,
                    m.input_tokens,
                    m.output_tokens
                );
                let record = QueryRecord {
                    kind: "chat",
                    collection: collection.name.clone(),
                    chat_profile: self.chat_name.clone(),
                    chat_model_configured: chat.model_id().to_owned(),
                    chat_model_reported: outcome.response.model.clone(),
                    embed_profile: collection.embed_profile.clone(),
                    embed_model: collection.embed_model.clone(),
                    dimension: collection.dimension,
                    codec: collection.codec.clone(),
                    metric: collection.metric.name().to_owned(),
                    session: Some(self.session.name.clone()),
                    question: question.to_owned(),
                    answer: outcome.response.text.clone(),
                    stop_reason: outcome.response.stop_reason.clone(),
                    metrics: outcome.metrics.clone(),
                };
                self.store.insert_run(&record.into_row())?;
                (outcome.response.text, summary)
            }
            None => {
                let mut messages = history;
                messages.push(Message::user(question));
                let t = std::time::Instant::now();
                let response = self.handle.block_on(chat.complete(ChatRequest {
                    system: Some(PLAIN_SYSTEM_PROMPT.to_owned()),
                    messages,
                    max_tokens: None,
                }))?;
                let summary = format!(
                    "[{} | no retrieval | {}ms | {} in / {} out tokens | not recorded]",
                    response.model,
                    t.elapsed().as_millis(),
                    response.usage.input_tokens,
                    response.usage.output_tokens
                );
                (response.text, summary)
            }
        };

        self.store.append_message(&self.session.name, "user", question)?;
        self.store.append_message(&self.session.name, "assistant", &answer)?;
        self.history.push(Message::user(question));
        self.history.push(Message::assistant(answer.clone()));

        println!("\n{}\n", answer.trim());
        eprintln!("{summary}\n");
        Ok(Flow::Continue)
    }
}

fn load_history(store: &Store, session: &str) -> Result<Vec<Message>> {
    Ok(store
        .messages(session)?
        .into_iter()
        .filter_map(|m| {
            Role::parse(&m.role).map(|role| Message {
                role,
                content: m.content,
            })
        })
        .collect())
}

fn print_help() {
    println!(
        "\
Plain text        ask a question (retrieval on when a collection is selected)
/use <name>       select a collection; /use none for plain chat
/new <name> <paths...>   create a collection by ingesting files or directories
/ingest <paths...>       add more files to the current collection
/model [name]     list chat profiles, or switch to one
/session <name>   switch to (or create) a session
/k <n>            passages to retrieve per turn
/context          toggle printing retrieved passages
/status           show session, collection and model
/collections  /sessions  /runs [n]  /embedders
/quit             exit (Ctrl-D also works)"
    );
}
