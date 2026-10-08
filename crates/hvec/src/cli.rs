//! Command-line surface. Kept in one file so `hvec --help` is easy to audit.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "hvec", version, about, long_about = None)]
pub struct Cli {
    /// Path to the config file. Defaults to $HVEC_CONFIG or the platform config dir.
    #[arg(long, global = true, env = "HVEC_CONFIG")]
    pub config: Option<PathBuf>,

    /// Increase log verbosity. Repeat for more.
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Create, inspect or locate the config file.
    #[command(subcommand)]
    Config(ConfigCommand),
    /// List local embedding models this build can download.
    Embedders,
    /// Manage vector collections.
    #[command(subcommand)]
    Collections(CollectionsCommand),
    /// Inspect saved chat sessions.
    #[command(subcommand)]
    Sessions(SessionsCommand),
    /// Inspect the run log.
    #[command(subcommand)]
    Runs(RunsCommand),
    /// Chunk, embed and store documents in a collection.
    Ingest(IngestArgs),
    /// Ask one question against a collection and record the run.
    Query(QueryArgs),
    /// Interactive multi-turn chat with retrieval, persisted as a session.
    Chat(ChatArgs),
}

#[derive(Debug, Clone, Copy, Subcommand)]
pub enum ConfigCommand {
    /// Write a commented starter config.
    Init {
        /// Overwrite an existing file.
        #[arg(long)]
        force: bool,
    },
    /// Print the resolved config.
    Show,
    /// Print the config file path.
    Path,
}

#[derive(Debug, Subcommand)]
pub enum CollectionsCommand {
    /// List collections with their embedder, codec and chunk count.
    List,
    /// Delete a collection and all its chunks.
    Delete { name: String },
}

#[derive(Debug, Subcommand)]
pub enum SessionsCommand {
    /// List sessions.
    List,
    /// Print a session's transcript.
    Show { name: String },
    /// Delete a session and its messages.
    Delete { name: String },
}

#[derive(Debug, Subcommand)]
pub enum RunsCommand {
    /// List recent runs, newest first.
    List {
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: usize,
    },
    /// Print one run as JSON. Accepts an id prefix.
    Show { id: String },
}

#[derive(Debug, Args)]
pub struct IngestArgs {
    /// Files or directories to ingest. Directories are walked recursively.
    #[arg(required = true)]
    pub paths: Vec<PathBuf>,
    /// Collection to write into. Created if missing.
    #[arg(short, long)]
    pub collection: String,
    /// Embedding profile. Defaults to `default_embedder` in config.
    #[arg(short, long)]
    pub embedder: Option<String>,
    /// Codec used to store vectors.
    #[arg(long, default_value = "f32")]
    pub codec: String,
    /// Words per chunk.
    #[arg(long, default_value_t = 200)]
    pub chunk_words: usize,
    /// Overlapping words between consecutive chunks.
    #[arg(long, default_value_t = 40)]
    pub overlap_words: usize,
    /// File extensions to include when walking directories.
    #[arg(long, value_delimiter = ',', default_value = "md,txt,markdown,rst")]
    pub ext: Vec<String>,
    /// Embedding batch size.
    #[arg(long, default_value_t = 32)]
    pub batch: usize,
}

#[derive(Debug, Args)]
pub struct QueryArgs {
    /// The question.
    pub question: String,
    /// Collection to retrieve from.
    #[arg(short, long)]
    pub collection: String,
    /// Chat profile. Defaults to `default_chat` in config.
    #[arg(long)]
    pub chat: Option<String>,
    /// Number of passages to retrieve.
    #[arg(short, long, default_value_t = 5)]
    pub k: usize,
    /// Print the retrieved passages before the answer.
    #[arg(long)]
    pub show_context: bool,
    /// Do not write this run to the run log.
    #[arg(long)]
    pub no_record: bool,
    /// Print the full run record as JSON instead of a summary.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct ChatArgs {
    /// Session name. Created if missing; resumed otherwise.
    #[arg(short, long)]
    pub session: String,
    /// Collection to retrieve from.
    #[arg(short, long)]
    pub collection: String,
    /// Chat profile. Defaults to `default_chat` in config.
    #[arg(long)]
    pub chat: Option<String>,
    /// Number of passages to retrieve per turn.
    #[arg(short, long, default_value_t = 5)]
    pub k: usize,
}
