//! A typed query record survives the trip through the generic run log.

use hvec_bench::{QueryMetrics, QueryRecord, RetrievedChunk};
use hvec_store::Store;

#[test]
fn query_record_round_trips_through_store() {
    let store = Store::open_in_memory().unwrap();
    let record = QueryRecord {
        kind: "query",
        collection: "docs".into(),
        chat_profile: "anthropic".into(),
        chat_model_configured: "claude-opus-5-5".into(),
        chat_model_reported: "claude-opus-5-5-served".into(),
        embed_profile: "local".into(),
        embed_model: "bge-small".into(),
        dimension: 384,
        codec: "f32".into(),
        metric: "cosine".into(),
        session: None,
        payload: "none".into(),
        question: "q?".into(),
        answer: "a.".into(),
        stop_reason: Some("end_turn".into()),
        attempts: serde_json::json!([
            {"number": 1, "status": 529, "body": "overloaded", "elapsed_ms": 40, "backoff_ms": 1000, "outcome": "retried"},
            {"number": 2, "status": 200, "body": "{}", "elapsed_ms": 900, "backoff_ms": null, "outcome": "succeeded"}
        ]),
        metrics: QueryMetrics {
            k: 3,
            embed_ms: 5,
            retrieval_ms: 1,
            generation_ms: 900,
            input_tokens: 1200,
            output_tokens: 80,
            retrieved: vec![RetrievedChunk {
                chunk_id: 7,
                source: "a.md".into(),
                ordinal: 2,
                score: 0.81,
            }],
            payload_ms: 0,
            context_chars_before: 10,
            context_chars_after: 10,
            context_words_before: 2,
            context_words_after: 2,
        },
    };
    let row = record.into_row();
    store.insert_run(&row).unwrap();

    let back = store.get_run(&row.id).unwrap();
    assert_eq!(back, row);
    assert_eq!(back.codec.as_deref(), Some("f32"));
    assert_eq!(back.chat_model.as_deref(), Some("claude-opus-5-5-served"));
    assert_eq!(back.metadata["chat_model_configured"], "claude-opus-5-5");
    assert_eq!(back.metadata["dimension"], 384);
    let m: QueryMetrics = serde_json::from_value(back.metrics).unwrap();
    assert_eq!(m.k, 3);
    assert_eq!(m.retrieved[0].chunk_id, 7);
    assert_eq!(back.metadata["payload"], "none");
    assert_eq!(back.metadata["attempts"].as_array().map(Vec::len), Some(2));
    assert_eq!(back.metadata["attempts"][0]["status"], 529);
    assert_eq!(back.metadata["attempts"][1]["outcome"], "succeeded");
    // Rows written before the payload stage existed still deserialise.
    let old: QueryMetrics = serde_json::from_str(
        r#"{"k":1,"embed_ms":0,"retrieval_ms":0,"generation_ms":0,"input_tokens":0,"output_tokens":0,"retrieved":[]}"#,
    )
    .unwrap();
    assert_eq!(old.payload_ms, 0);
}
