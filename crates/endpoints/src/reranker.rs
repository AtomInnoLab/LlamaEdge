//! Define types for the `reranker` endpoint.

use crate::common::Usage;
use serde::{Deserialize, Serialize};

/// Creates a reranker request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerRequest {
    /// ID of the model to use.
    pub model: String,
    /// The input text to rerank.
    pub query: String,
    /// The list of documents to rerank.
    pub documents: Vec<String>,
    /// The number of results to return.
    pub top_n: Option<usize>,
}

/// Defines the reranker response.
#[derive(Debug, Serialize, Deserialize)]
pub struct RerankerResponse {
    pub object: String,
    pub results: Vec<RerankerObject>,
    pub model: String,
    pub usage: Usage,
}

/// Represents a reranked document.
#[derive(Debug, Serialize, Deserialize)]
pub struct RerankerObject {
    pub index: u64,
    pub relevance_score: f64,
}
