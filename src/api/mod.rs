use axum::{routing::post, Json, Router, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use anyhow::{Error as E, Result};
use crate::Model;

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
    pub eos_tokens: Vec<u32>,
    pub model_name: String,
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    println!("API Hit ({}): {}", state.model_name, payload.prompt);

    let formatted = match state.model_name.as_str() {
        "Phi-3"   => format!("<|user|>\n{}<|end|>\n<|assistant|>", payload.prompt),
        "Mistral" => format!("<s>[INST] {} [/INST]", payload.prompt),
        _         => format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", payload.prompt),
    };

    let tokens = state.tokenizer.encode(formatted, true).unwrap();
    let prompt_tokens = tokens.get_ids().to_vec();

    let mut model = state.model.lock().await;
    let mut tokens_to_process = prompt_tokens.clone();
    let mut generated: Vec<u32> = Vec::new();
    let mut total_pos = 0;

    for _ in 0..500 {
        let input_tensor = Tensor::new(tokens_to_process.as_slice(), &state.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let logits = model.forward(&input_tensor, total_pos).unwrap();
        total_pos += tokens_to_process.len();

        let next_token = get_next_token(&logits).unwrap();

        if state.eos_tokens.contains(&next_token) { break; }

        generated.push(next_token);
        tokens_to_process = vec![next_token];
    }

    let response_text = state.tokenizer.decode(&generated, true).unwrap_or_default();

    Json(ChatResponse { response: response_text })
}

fn get_next_token(logits: &Tensor) -> Result<u32> {
    let shape = logits.dims();
    let last_row = match shape.len() {
        3 => logits.get(0)?.get(shape[1] - 1)?,
        2 => logits.get(shape[0] - 1)?,
        _ => logits.clone(),
    };
    Ok(last_row.argmax(0)?.to_scalar::<u32>()?)
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
