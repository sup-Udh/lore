use axum::{routing::post, Json, Router, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use anyhow::Result;
use crate::{Model, TokenOutputStream};

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
    let prompt_ids = tokens.get_ids().to_vec();

    let mut model = state.model.lock().await;

    // OPT 2: total_pos reset per request — each API call is stateless
    let mut total_pos: usize = 0;
    let mut last_token: u32 = 0;

    // OPT 3: incremental decode via TokenOutputStream — no bulk Vec<u32> + decode at end
    let mut decoder = TokenOutputStream::new(state.tokenizer.clone());

    for step in 0..500usize {
        // OPT 4: first pass feeds full prompt; after that feed one token using
        // from_ref to avoid a heap Vec alloc on every step
        let ids: &[u32] = if step == 0 { &prompt_ids } else { std::slice::from_ref(&last_token) };

        let input_tensor = Tensor::new(ids, &state.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let logits = model.forward(&input_tensor, total_pos).unwrap();
        total_pos += ids.len();

        let next_token = get_next_token(&logits).unwrap();
        if state.eos_tokens.contains(&next_token) { break; }

        last_token = next_token;
        decoder.next_token(next_token).unwrap();
    }

    let response_text = decoder.into_text().unwrap_or_default();

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
