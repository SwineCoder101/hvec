//! `hvec`: benchmark homomorphic vector compression against real RAG pipelines.

mod cli;
mod commands;
mod context;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    let Some(command) = cli.command else {
        return commands::shell::run(&cli.config, cli.shell).await;
    };
    match command {
        Command::Config(cmd) => commands::config::run(&cli.config, cmd),
        Command::Embedders => commands::list::embedders(),
        Command::Collections(cmd) => commands::list::collections(&cli.config, cmd),
        Command::Sessions(cmd) => commands::list::sessions(&cli.config, cmd),
        Command::Runs(cmd) => commands::list::runs(&cli.config, cmd),
        Command::Ingest(args) => commands::ingest::run(&cli.config, args).await,
        Command::Query(args) => commands::query::run(&cli.config, args).await,
        Command::Chat(args) => commands::shell::run(&cli.config, args).await,
    }
}

fn init_tracing(verbose: u8) {
    let default = match verbose {
        0 => "warn",
        1 => "info,hvec=debug",
        _ => "debug",
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();
}
