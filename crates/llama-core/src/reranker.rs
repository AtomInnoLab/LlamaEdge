
//! Define APIs for reranking.

use crate::{
    error::{BackendError, LlamaCoreError},
    metadata::ggml::GgmlMetadata,
    running_mode,
    utils::{get_output_buffer, get_token_info_by_graph},
    Graph, RunningMode, RERANKER_GRAPHS, OUTPUT_TENSOR,
};
use endpoints::{
    common::Usage,
    reranker::{RerankerObject, RerankerRequest, RerankerResponse},
};

pub async fn reranker(
    reranker_request: &RerankerRequest,
) -> Result<RerankerResponse, LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Reranking documents");

    let running_mode = running_mode()?;
    if running_mode != RunningMode::Reranker && running_mode != RunningMode::ChatEmbeddingReranker {
        let err_msg = format!("Reranking is not supported in the {} mode.", running_mode);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        return Err(LlamaCoreError::Operation(err_msg));
    }

    let model_name = &reranker_request.model;

    // get the reranker graph
    let reranker_graphs = match RERANKER_GRAPHS.get() {
        Some(reranker_graphs) => reranker_graphs,
        None => {
            let err_msg = "No reranker model is available.";

            #[cfg(feature = "logging")]
            error!(target: "stdout", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let mut reranker_graphs = reranker_graphs.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `RERANKER_GRAPHS`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "stdout", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    let graph = match reranker_graphs.contains_key(model_name) {
        true => reranker_graphs.get_mut(model_name).unwrap(),
        false => match reranker_graphs.iter_mut().next()     {
            Some((_, graph)) => graph,
            None => {
                let err_msg = "There is no model available in the reranker graphs.";

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                return Err(LlamaCoreError::Operation(err_msg.into()));
            }
        },
    };

    // check if the `reranking` option of metadata is enabled
    if !graph.metadata.reranking {
        graph.metadata.reranking = true;
        graph.update_metadata()?;
    }

    let (data, usage) = compute_reranking(graph, &reranker_request.query, &reranker_request.documents)?;

    let reranker_response = RerankerResponse {
        object: String::from("list"),
        results: data,
        model: graph.name().to_owned(),
        usage,
    };

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Reranking documents for query {} completed.", reranker_request.query);

    Ok(reranker_response)
}


fn compute_reranking(
    graph: &mut Graph<GgmlMetadata>, 
    query: &str,
    documents: &[String]
) -> Result<(Vec<RerankerObject>, Usage), LlamaCoreError> {
    #[cfg(feature = "logging")]
    info!(target: "stdout", "Reranking {} documents for query {}", documents.len(), query);

    // compute reranking
    let mut reranked_documents: Vec<RerankerObject> = Vec::new();
    let mut usage = Usage::default();
    for (idx, document) in documents.iter().enumerate() {
        // set input
        let tensor_data = document.as_bytes().to_vec();
        graph
            .set_input(0, wasmedge_wasi_nn::TensorType::U8, &[1], &tensor_data)
            .map_err(|e| {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                LlamaCoreError::Backend(BackendError::SetInput(err_msg))
            })?;
        
        #[cfg(feature = "logging")]
        info!(target: "stdout", "Reranking document {}", idx + 1);

        match graph.compute() {
            Ok(_) => {
                // Retrieve the output.
                let output_buffer = get_output_buffer(graph, OUTPUT_TENSOR)?;

                // convert inference result to string
                let output = std::str::from_utf8(&output_buffer[..]).map_err(|e| {
                    let err_msg = format!(
                        "Failed to decode the buffer of the inference result to a utf-8 string. Reason: {}",
                        e
                    );

                    #[cfg(feature = "logging")]
                    error!(target: "stdout", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

                info!(target: "stdout", "Output: {}", output);

                let reranker_object = RerankerObject {
                    index: idx as u64,
                    relevance_score: 1.0,
                };

                reranked_documents.push(reranker_object);

                let token_info = get_token_info_by_graph(graph)?;
                usage.prompt_tokens += token_info.prompt_tokens;
                usage.completion_tokens += token_info.completion_tokens;
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
            }
            Err(e) => {
                let err_msg = e.to_string();

                #[cfg(feature = "logging")]
                error!(target: "stdout", "{}", &err_msg);

                return Err(LlamaCoreError::Backend(BackendError::Compute(err_msg)));
            }
        }   
    }

    #[cfg(feature = "logging")]
    info!(target: "stdout", "Reranking documents for query {} completed.", query);


    Ok((reranked_documents, usage))
}
