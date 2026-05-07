use axum::{routing::post, Json, Router, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::backends::llama_cpp::LlamaBackend;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
    pub debug: Option<bool>,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub response: String,
}

// All three models now go through llama.cpp behind a single Mutex.
// model_name decides whether the request runs through the orchestrator
// (Phi-3 / Mistral) or hits the model directly (Qwen).
pub struct AppState {
    pub backend: Mutex<LlamaBackend>,
    pub model_name: String,
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let debug = payload.debug.unwrap_or(false);
    println!("API Hit ({}) [debug={}]: {}", state.model_name, debug, payload.prompt);

    let mut backend = state.backend.lock().await;

    if state.model_name == "Qwen" {
        // Qwen — direct, no agents
        let response = backend.generate(&payload.prompt).unwrap_or_default();
        Json(ChatResponse { response })
    } else {
        // MULTI-AGENT PIPELINE TEMPORARILY DISABLED
        // DIRECT INFERENCE MODE ENABLED
        // ORCHESTRATOR BYPASSED FOR PERFORMANCE
        //
        // Phi-3 / Mistral now use direct llama.cpp inference, same as Qwen.
        // The `trace` field is retained for future re-enable support.
        let response = backend.generate(&payload.prompt).unwrap_or_default();
        let _ = debug; // retained for API compatibility while trace is bypassed
        Json(ChatResponse { response})
    }
}

pub async fn start_api_llama(backend: LlamaBackend, model_name: &str) {
    let shared_state = Arc::new(AppState {
        backend: Mutex::new(backend),
        model_name: model_name.to_string(),
    });

    let app = Router::new()
        .route("/chat", post(chat_handler))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Lore API online: http://localhost:3000/chat");
    axum::serve(listener, app).await.unwrap();
}
