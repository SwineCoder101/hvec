//! Persistent chat sessions.

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::{Result, Store, StoreError};

/// A named conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    pub name: String,
    pub collection: Option<String>,
    pub chat_profile: String,
    pub created_at: String,
    pub message_count: u64,
}

/// A stored turn. Role is `user` or `assistant`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredMessage {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub created_at: String,
}

impl Store {
    /// Create the session if it does not exist, then return it.
    pub fn ensure_session(&self, name: &str, collection: Option<&str>, chat_profile: &str) -> Result<Session> {
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions (name, collection, chat_profile, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![name, collection, chat_profile, Self::now()],
        )?;
        self.get_session(name)?
            .ok_or_else(|| StoreError::NoSuchSession(name.to_owned()))
    }

    pub fn get_session(&self, name: &str) -> Result<Option<Session>> {
        self.conn
            .query_row(
                "SELECT s.name, s.collection, s.chat_profile, s.created_at,
                        (SELECT COUNT(*) FROM messages WHERE session = s.name)
                 FROM sessions s WHERE s.name = ?1",
                params![name],
                row_to_session,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn list_sessions(&self) -> Result<Vec<Session>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.name, s.collection, s.chat_profile, s.created_at,
                    (SELECT COUNT(*) FROM messages WHERE session = s.name)
             FROM sessions s ORDER BY s.created_at",
        )?;
        let rows = stmt.query_map([], row_to_session)?;
        rows.collect::<std::result::Result<_, _>>().map_err(Into::into)
    }

    pub fn delete_session(&self, name: &str) -> Result<()> {
        let n = self
            .conn
            .execute("DELETE FROM sessions WHERE name = ?1", params![name])?;
        if n == 0 {
            return Err(StoreError::NoSuchSession(name.to_owned()));
        }
        Ok(())
    }

    pub fn append_message(&self, session: &str, role: &str, content: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO messages (session, role, content, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![session, role, content, Self::now()],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn messages(&self, session: &str) -> Result<Vec<StoredMessage>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT id, role, content, created_at FROM messages WHERE session = ?1 ORDER BY id")?;
        let rows = stmt.query_map(params![session], |row| {
            Ok(StoredMessage {
                id: row.get(0)?,
                role: row.get(1)?,
                content: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        rows.collect::<std::result::Result<_, _>>().map_err(Into::into)
    }
}

fn row_to_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<Session> {
    Ok(Session {
        name: row.get(0)?,
        collection: row.get(1)?,
        chat_profile: row.get(2)?,
        created_at: row.get(3)?,
        message_count: row.get::<_, i64>(4)? as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_round_trip_in_order() {
        let store = Store::open_in_memory().unwrap();
        store.ensure_session("s1", Some("docs"), "anthropic").unwrap();
        store.append_message("s1", "user", "hi").unwrap();
        store.append_message("s1", "assistant", "hello").unwrap();
        let msgs = store.messages("s1").unwrap();
        assert_eq!(
            msgs.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            ["user", "assistant"]
        );
        assert_eq!(store.get_session("s1").unwrap().unwrap().message_count, 2);
    }
}
