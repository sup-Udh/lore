pub mod orchestrator;
pub mod research;
pub mod compare;

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use anyhow::{Error as E, Result};
use crate::{Model, TokenOutputStream};

pub struct InferenceContext<'a> {
    pub tokenizer: &'a Tokenizer,
    pub device: &'a Device,
    pub eos_tokens: &'a [u32],
    pub model_name: &'a str,
}

// Core inference engine shared by all agents.
// Each call is independent: total_pos resets to 0 so the KV cache
// is always written from position 0 for the new prompt.
pub fn generate(
    model: &mut dyn Model,
    ctx: &InferenceContext,
    system: &str,
    input: &str,
) -> Result<String> {
    let formatted = match ctx.model_name {
        "Phi-3"   => format!("<|system|>\n{}<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>", system, input),
        "Mistral" => format!("<s>[INST] {}\n\n{} [/INST]", system, input),
        _         => format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", system, input),
    };

    let tokens = ctx.tokenizer.encode(formatted, true).map_err(E::msg)?;
    let prompt_ids = tokens.get_ids().to_vec();

    let mut total_pos: usize = 0;
    let mut last_token: u32 = 0;
    let mut decoder = TokenOutputStream::new(ctx.tokenizer.clone());

    for step in 0..500usize {
        let ids: &[u32] = if step == 0 { &prompt_ids } else { std::slice::from_ref(&last_token) };
        let input_tensor = Tensor::new(ids, ctx.device)?.unsqueeze(0)?;
        let logits = model.forward(&input_tensor, total_pos)?;
        total_pos += ids.len();

        let next_token = get_next_token(&logits)?;
        if ctx.eos_tokens.contains(&next_token) { break; }

        last_token = next_token;
        decoder.next_token(next_token)?;
    }

    decoder.into_text()
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
