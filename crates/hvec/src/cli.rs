//! Command-line surface. Kept in one file so `hvec --help` is easy to audit.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "hvec", version, about, long_about = None, args_conflicts_with_subcommands = true)]
#[command(after_help = "Run `hvec` with no subcommand to open the interactive shell.")]
pub struct Cli {
    /// Path to the config file. Defaults to $HVEC_CONFIG or the platform config dir.
    #[arg(long, global = true, env = "HVEC_CONFIG")]
    pub config: Option<PathBuf>,

    /// Increase log verbosity. Repeat for more.
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Options for the interactive shell, used when no subcommand is given.
    #[command(flatten)]
    pub shell: ShellArgs,

    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Flags for the interactive shell (bare `hvec`) and the `chat` alias.
#[derive(Debug, Clone, Args)]
pub struct ShellArgs {
    /// Session to resume or create. Defaults to a new timestamped session.
    #[arg(short, long)]
    pub session: Option<String>,
    /// Collection to retrieve from. Defaults to the only collection if exactly one exists.
    #[arg(short, long)]
    pub collection: Option<String>,
    /// Chat profile. Defaults to `default_chat` in config.
    #[arg(long)]
    pub chat: Option<String>,
    /// Number of passages to retrieve per turn.
    #[arg(short, long, default_value_t = 5)]
    pub k: usize,
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
    /// Benchmarks. `bench recall` needs no chat model.
    #[command(subcommand)]
    Bench(BenchCommand),
    /// Ask one question against a collection and record the run.
    Query(QueryArgs),
    /// Open the interactive shell. Same as running `hvec` with no subcommand.
    Chat(ShellArgs),
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

#[derive(Debug, Subcommand)]
pub enum BenchCommand {
    /// Re-encode an f32 collection with other codecs and measure retrieval against the exact ranking.
    Recall(RecallArgs),
}

#[derive(Debug, Args)]
pub struct RecallArgs {
    /// An f32 collection to use as ground truth.
    #[arg(short, long)]
    pub collection: String,
    /// Codecs to evaluate.
    #[arg(long, value_delimiter = ',', default_value = "int8,binary")]
    pub codecs: Vec<String>,
    /// Neighbours per query.
    #[arg(short, long, default_value_t = 10)]
    pub k: usize,
    /// How many stored vectors to use as self-queries (each excludes itself).
    #[arg(long, default_value_t = 100)]
    pub sample: usize,
    /// A text file with one query per line, embedded with the collection's embedder, instead of self-queries.
    #[arg(long)]
    pub queries: Option<PathBuf>,
    /// Do not write the results to the run log.
    #[arg(long)]
    pub no_record: bool,
    /// Print the reports as JSON instead of a table.
    #[arg(long)]
    pub json: bool,
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
    /// Codec used to store vectors: f32, int8 or binary.
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
