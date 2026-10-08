//! Test support: an in-process HTTP server that speaks enough of the
//! Anthropic Messages API and the OpenAI chat API to drive the pipeline,
//! and records every request it receives.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct Recorded {
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: serde_json::Value,
}

pub struct MockServer {
    pub base_url: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
}

impl MockServer {
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let log = Arc::clone(&log);
                std::thread::spawn(move || handle(stream, &log));
            }
        });
        Self {
            base_url: format!("http://{addr}"),
            requests,
        }
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().expect("lock").clone()
    }

    pub fn anthropic_requests(&self) -> Vec<Recorded> {
        self.requests()
            .into_iter()
            .filter(|r| r.path.ends_with("/v1/messages"))
            .collect()
    }

    pub fn openai_requests(&self) -> Vec<Recorded> {
        self.requests()
            .into_iter()
            .filter(|r| r.path.ends_with("/chat/completions"))
            .collect()
    }
}

fn handle(mut stream: TcpStream, log: &Mutex<Vec<Recorded>>) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("/").to_owned();
    let mut headers = HashMap::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_owned());
        }
    }
    let len: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut raw = vec![0u8; len];
    reader.read_exact(&mut raw).expect("body");
    let body: serde_json::Value = serde_json::from_slice(&raw).unwrap_or(serde_json::Value::Null);

    let user_text = last_user_text(&body);
    let (status, payload) = if path.ends_with("/v1/messages") {
        if user_text.contains("PLEASE-REFUSE") {
            (
                200,
                serde_json::json!({
                    "content": [], "model": body["model"], "stop_reason": "refusal",
                    "stop_details": {"type": "refusal", "category": "test"},
                    "usage": {"input_tokens": 1, "output_tokens": 0}
                }),
            )
        } else {
            let n = body["messages"].as_array().map_or(0, Vec::len);
            (
                200,
                serde_json::json!({
                    "content": [{"type": "text", "text": format!("MOCK-ANTHROPIC: {n} msgs")}],
                    "model": format!("{}-served", body["model"].as_str().unwrap_or("?")),
                    "stop_reason": "end_turn", "stop_details": null,
                    "usage": {"input_tokens": 123, "output_tokens": 7}
                }),
            )
        }
    } else if path.ends_with("/chat/completions") {
        let n = body["messages"].as_array().map_or(0, Vec::len);
        (
            200,
            serde_json::json!({
                "model": body["model"],
                "choices": [{"message": {"role": "assistant", "content": format!("MOCK-OPENAI: {n} msgs")}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 45, "completion_tokens": 6}
            }),
        )
    } else {
        (404, serde_json::json!({"error": "no route"}))
    };

    log.lock().expect("lock").push(Recorded { path, headers, body });

    let data = payload.to_string();
    let reason = if status == 200 { "OK" } else { "Not Found" };
    let _ = write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{data}",
        data.len()
    );
    let _ = stream.flush();
}

fn last_user_text(body: &serde_json::Value) -> String {
    body["messages"]
        .as_array()
        .and_then(|m| m.iter().rev().find(|x| x["role"] == "user"))
        .and_then(|m| m["content"].as_str())
        .unwrap_or("")
        .to_owned()
}

/// A temp workspace with a config pointing at the mock server and a hash embedder.
pub struct Env {
    pub dir: tempfile::TempDir,
    pub config_path: std::path::PathBuf,
    pub server: MockServer,
}

impl Env {
    pub fn new() -> Self {
        let server = MockServer::start();
        let dir = tempfile::tempdir().expect("tempdir");
        let config_path = dir.path().join("config.toml");
        let db = dir.path().join("hvec.db");
        std::fs::write(
            &config_path,
            format!(
                r#"
db_path = "{db}"
default_chat = "mock-anthropic"
default_embedder = "hash"
metric = "cosine"

[chat.mock-anthropic]
provider = "anthropic"
model = "claude-opus-5-5"
base_url = "{base}"
api_key_env = "HVEC_TEST_KEY"
effort = "medium"

[chat.mock-openai]
provider = "openai"
model = "fake-model"
base_url = "{base}/v1"

[embed.hash]
provider = "hash"
model = "hash-128"
dimension = 128
"#,
                db = db.display(),
                base = server.base_url
            ),
        )
        .expect("write config");
        Self {
            dir,
            config_path,
            server,
        }
    }

    pub fn write_doc(&self, name: &str, text: &str) -> std::path::PathBuf {
        let p = self.dir.path().join(name);
        std::fs::write(&p, text).expect("write doc");
        p
    }

    pub fn hvec(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_hvec"));
        c.env("HVEC_CONFIG", &self.config_path)
            .env("HVEC_TEST_KEY", "test-key-123")
            .env_remove("RUST_LOG");
        c
    }

    pub fn run(&self, args: &[&str]) -> Output {
        self.hvec().args(args).output().expect("run hvec")
    }

    pub fn run_with_stdin(&self, args: &[&str], stdin: &str) -> Output {
        let mut child = self
            .hvec()
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn hvec");
        child
            .stdin
            .take()
            .expect("stdin")
            .write_all(stdin.as_bytes())
            .expect("write stdin");
        child.wait_with_output().expect("wait")
    }
}

pub fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

pub fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

pub fn assert_ok(o: &Output, what: &str) {
    assert!(
        o.status.success(),
        "{what} failed\nstdout:\n{}\nstderr:\n{}",
        stdout(o),
        stderr(o)
    );
}

pub fn exists(p: &Path) -> bool {
    p.exists()
}
