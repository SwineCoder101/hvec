//! Shared setup: load config, open the store.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use hvec_connect::Config;
use hvec_store::Store;

pub struct Ctx {
    pub config_path: PathBuf,
    pub config: Config,
}

impl Ctx {
    pub fn load(config_override: Option<&Path>) -> Result<Self> {
        let config_path = config_override
            .map(Path::to_path_buf)
            .unwrap_or_else(Config::default_path);
        let config = Config::load(&config_path)?;
        Ok(Self { config_path, config })
    }

    pub fn open_store(&self) -> Result<Store> {
        let path = self.config.db_path();
        Store::open(&path).with_context(|| format!("opening database at {}", path.display()))
    }

    pub fn chat_profile_name<'a>(&'a self, flag: Option<&'a str>) -> &'a str {
        flag.unwrap_or(&self.config.default_chat)
    }

    pub fn embed_profile_name<'a>(&'a self, flag: Option<&'a str>) -> &'a str {
        flag.unwrap_or(&self.config.default_embedder)
    }

    pub fn metric(&self) -> Result<hvec_core::Metric> {
        hvec_core::Metric::parse(&self.config.metric)
            .with_context(|| format!("metric in {}", self.config_path.display()))
    }
}
