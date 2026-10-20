//! Bounded retries with exponential backoff for provider HTTP calls.
//!
//! A connector decides per reply whether to try again; this module holds the
//! shared policy: which statuses are worth retrying, how long to wait, and how
//! the attempt history is attached to a final error.

use std::time::Duration;

use reqwest::header::{HeaderMap, RETRY_AFTER};

use crate::ConnectError;
use crate::traits::Attempt;

/// Attempts per request when a profile does not say otherwise.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 3;
/// Wait before the second attempt; doubles on each further one.
const BASE_DELAY: Duration = Duration::from_secs(1);
/// Ceiling on the computed backoff.
const MAX_DELAY: Duration = Duration::from_secs(30);
/// Ceiling on a server-supplied `retry-after`.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);
/// Jitter added to the computed backoff, spread over this many milliseconds.
const JITTER_MS: u64 = 250;

/// HTTP statuses that mean "try the same request again": timeouts, conflicts,
/// rate limits, overload and other server-side failures.
#[must_use]
pub fn retryable_status(status: u16) -> bool {
    matches!(status, 408 | 409 | 429 | 500..=599)
}

/// Transport failures are retried unless the request itself could not be built.
#[must_use]
pub fn retryable_transport(err: &reqwest::Error) -> bool {
    !err.is_builder()
}

/// A `retry-after` header given in seconds, capped. The HTTP-date form is ignored.
#[must_use]
pub fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let secs: u64 = headers.get(RETRY_AFTER)?.to_str().ok()?.trim().parse().ok()?;
    Some(Duration::from_secs(secs).min(MAX_RETRY_AFTER))
}

/// How long to wait after attempt `attempt` (1-based) failed. A server-supplied
/// `retry_after` wins; otherwise exponential backoff from [`BASE_DELAY`] with a
/// little jitter so parallel runs do not retry in lockstep.
#[must_use]
pub fn backoff(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(after) = retry_after {
        return after;
    }
    let exp = attempt.saturating_sub(1).min(16);
    let base = BASE_DELAY.saturating_mul(1 << exp).min(MAX_DELAY);
    base + Duration::from_millis(jitter_ms())
}

fn jitter_ms() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    u64::from(nanos) % JITTER_MS
}

/// Attach the attempt history to the error that ended a request. A request
/// that failed on its first and only attempt comes back unchanged.
#[must_use]
pub fn after_retries(provider: &'static str, err: ConnectError, attempts: Vec<Attempt>) -> ConnectError {
    if attempts.len() > 1 {
        ConnectError::AfterRetries {
            provider,
            attempts,
            source: Box::new(err),
        }
    } else {
        err
    }
}

/// Display an error with its source chain on one line.
#[must_use]
pub fn error_chain(err: &dyn std::error::Error) -> String {
    let mut text = err.to_string();
    let mut cur = err.source();
    while let Some(e) = cur {
        text.push_str(": ");
        text.push_str(&e.to_string());
        cur = e.source();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::AttemptOutcome;

    fn attempt(number: u32) -> Attempt {
        Attempt {
            number,
            status: Some(529),
            body: "overloaded".into(),
            elapsed_ms: 1,
            backoff_ms: None,
            outcome: AttemptOutcome::Failed,
        }
    }

    #[test]
    fn statuses() {
        for s in [408, 409, 429, 500, 502, 529] {
            assert!(retryable_status(s), "{s}");
        }
        for s in [200, 400, 401, 403, 404, 422] {
            assert!(!retryable_status(s), "{s}");
        }
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let one = backoff(1, None);
        let two = backoff(2, None);
        let far = backoff(40, None);
        assert!(one >= Duration::from_secs(1) && one < Duration::from_millis(1250));
        assert!(two >= Duration::from_secs(2) && two < Duration::from_millis(2250));
        assert!(far >= MAX_DELAY && far < MAX_DELAY + Duration::from_millis(JITTER_MS));
    }

    #[test]
    fn retry_after_header_wins() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, "7".parse().unwrap());
        assert_eq!(retry_after(&headers), Some(Duration::from_secs(7)));
        assert_eq!(backoff(1, retry_after(&headers)), Duration::from_secs(7));

        headers.insert(RETRY_AFTER, "99999".parse().unwrap());
        assert_eq!(retry_after(&headers), Some(MAX_RETRY_AFTER));

        headers.insert(RETRY_AFTER, "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap());
        assert_eq!(retry_after(&headers), None);
    }

    #[test]
    fn single_attempt_error_is_unchanged() {
        let err = ConnectError::Api {
            provider: "x",
            status: 400,
            body: "bad".into(),
        };
        assert!(matches!(
            after_retries("x", err, vec![attempt(1)]),
            ConnectError::Api { status: 400, .. }
        ));
    }

    #[test]
    fn several_attempts_are_wrapped() {
        let err = ConnectError::Api {
            provider: "x",
            status: 529,
            body: "overloaded".into(),
        };
        let wrapped = after_retries("x", err, vec![attempt(1), attempt(2)]);
        let ConnectError::AfterRetries { attempts, source, .. } = &wrapped else {
            panic!("expected AfterRetries, got {wrapped:?}");
        };
        assert_eq!(attempts.len(), 2);
        assert!(matches!(**source, ConnectError::Api { status: 529, .. }));
        let text = wrapped.to_string();
        assert!(text.contains("after 2 attempts"), "{text}");
        assert!(text.contains("attempt 1: HTTP 529"), "{text}");
    }
}
