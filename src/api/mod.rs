use axum::{routing::post, Json, Router, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::Device;
use tokenizers::Tokenizer;
use crate::Model;
use crate::agents::{AgentStep, InferenceContext, orchestrator::Orchestrator};

#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
    pub debug: Option<bool>, // if true, trace is included in response
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub response: String,
    pub trace: Option<Vec<AgentStep>>, // None unless debug = true
}

pub struct AppState {
    pub model: Mutex<Box<dyn Model + Send>>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub eos_tokens: Vec<u32>,
    pub model_name: String,
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let debug = payload.debug.unwrap_or(false);
    println!("API Hit ({}) [debug={}]: {}", state.model_name, debug, payload.prompt);

    let mut model = state.model.lock().await;

    let ctx = InferenceContext {
        tokenizer: &state.tokenizer,
        device: &state.device,
        eos_tokens: &state.eos_tokens,
        model_name: &state.model_name,
    };

    let mut orchestrator = Orchestrator::new();
    let trace = orchestrator.run(&mut **model, &ctx, payload.prompt).unwrap();

    Json(ChatResponse {
        response: trace.final_output,
        trace: if debug { Some(trace.steps) } else { None },
    })
}

pub async fn start_api(
    model: Box<dyn Model + Send>,
    tokenizer: Tokenizer,
    device: Device,
    eos_tokens: Vec<u32>,
    model_name: &str,
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
    println!("Lore API online: http://localhost:3000/chat");

    axum::serve(listener, app).await.unwrap();
}
