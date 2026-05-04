use axum::{routing::post, Json, Router, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::Device;
use tokenizers::Tokenizer;
use crate::Model; // Refers to the trait in main.rs

#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub response: String,
}

pub struct AppState {
    pub model: Mutex<Box<dyn Model + Send>>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub eos_tokens: Vec<u32>, // Added to fix argument mismatch
    pub model_name: String,   // Added to fix argument mismatch[cite: 2]
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    println!("📥 API Hit ({}): {}", state.model_name, payload.prompt);

    // For now, returning an echo. In the next step, we can move 
    // the chat_loop logic here for real generation.
    Json(ChatResponse {
        response: format!("{} Server ready. Echo: {}", state.model_name, payload.prompt),
    })
}

pub async fn start_api(
    model: Box<dyn Model + Send>,
    tokenizer: Tokenizer,
    device: Device,
    eos_tokens: Vec<u32>, // Argument #4[cite: 2]
    model_name: &str,     // Argument #5[cite: 2]
) {
    let shared_state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
        device,
        eos_tokens,
        model_name: model_name.to_string(),
    });

    let app = Router::new()
        .route("/chat", post(chat_handler))
        .with_state(shared_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("🚀 Lore API (Axum 0.8) Online: http://localhost:3000/chat");
    
    axum::serve(listener, app).await.unwrap();
}