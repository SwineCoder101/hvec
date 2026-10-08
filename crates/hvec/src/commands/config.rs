use std::path::Path;

use anyhow::{Result, bail};
use hvec_connect::Config;

use crate::cli::ConfigCommand;
use crate::context::Ctx;

pub fn run(config_override: &Option<std::path::PathBuf>, cmd: ConfigCommand) -> Result<()> {
    let path = config_override.clone().unwrap_or_else(Config::default_path);
    match cmd {
        ConfigCommand::Init { force } => init(&path, force),
        ConfigCommand::Path => {
            println!("{}", path.display());
            Ok(())
        }
        ConfigCommand::Show => {
            let ctx = Ctx::load(Some(&path))?;
            println!("# {}", ctx.config_path.display());
            println!("# database: {}", ctx.config.db_path().display());
            print!("{}", toml::to_string_pretty(&ctx.config)?);
            Ok(())
        }
    }
}

fn init(path: &Path, force: bool) -> Result<()> {
    if path.exists() && !force {
        bail!("{} already exists; pass --force to overwrite", path.display());
    }
    Config::write_starter(path)?;
    println!("wrote {}", path.display());
    println!("next: export ANTHROPIC_API_KEY (or edit the file to use another provider), then `hvec ingest`");
    Ok(())
}
