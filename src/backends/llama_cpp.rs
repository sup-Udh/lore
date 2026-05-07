// LLAMA.CPP MIGRATION (all models)
// PERSISTENT SESSION
// BACKEND ABSTRACTION
//
// Single LlamaModel + LlamaContext per backend instance.  ModelKind controls
// per-model chat-template formatting; the inference loop is identical for all.

use anyhow::{Context, Result};
use colored::*;
use std::num::NonZeroU32;
use std::sync::OnceLock;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend as LlamaCppBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

// PERSISTENT SESSION — global llama.cpp init exactly once.
static BACKEND: OnceLock<LlamaCppBackend> = OnceLock::new();

fn get_backend() -> Result<&'static LlamaCppBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let b = LlamaCppBackend::init().context("LlamaBackend::init failed")?;
    let _ = BACKEND.set(b);
    Ok(BACKEND.get().expect("backend just set"))
}

// Per-model differences — only the chat template and AddBos policy differ.
#[derive(Clone, Copy, Debug)]
pub enum ModelKind {
    Qwen,
    Phi3,
    Mistral,
}

impl ModelKind {
    fn add_bos(&self) -> AddBos {
        match self {
            // Qwen and Phi-3 use ChatML-style templates that don't need BOS.
            ModelKind::Qwen | ModelKind::Phi3 => AddBos::Never,
            // Mistral's [INST] template includes its own <s>; we still let
            // llama.cpp handle BOS to match the GGUF metadata.
            ModelKind::Mistral => AddBos::Always,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            ModelKind::Qwen => "Qwen",
            ModelKind::Phi3 => "Phi-3",
            ModelKind::Mistral => "Mistral",
        }
    }
}

pub struct LlamaBackend {
    // 'static via Box::leak — model lives for the program's lifetime.
    model: &'static LlamaModel,
    ctx: LlamaContext<'static>,
    n_ctx: i32,
    max_new_tokens: i32,
    kind: ModelKind,
}

// SAFETY: serialized externally by tokio::sync::Mutex around the backend.
unsafe impl Send for LlamaBackend {}

impl LlamaBackend {
    pub fn new(model_path: &str, kind: ModelKind) -> Result<Self> {
        let t0 = Instant::now();
        println!(
            "{} initializing backend for {} (one-time)...",
            "[llama.cpp]".cyan().bold(),
            kind.label()
        );

        let backend = get_backend()?;
        let model_params = LlamaModelParams::default();

        println!("{} loading GGUF: {}", "[llama.cpp]".cyan().bold(), model_path);
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .with_context(|| format!("failed to load {} GGUF via llama.cpp", kind.label()))?;
        let model: &'static LlamaModel = Box::leak(Box::new(model));

        let n_ctx_value: u32 = 2048;
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx_value));
        let ctx = model
            .new_context(backend, ctx_params)
            .with_context(|| format!("failed to create llama.cpp context for {}", kind.label()))?;

        println!(
            "{} {} ready (n_ctx={}, load_time={:.2}s)",
            "[llama.cpp]".cyan().bold(),
            kind.label(),
            n_ctx_value,
            t0.elapsed().as_secs_f32()
        );

        Ok(Self {
            model,
            ctx,
            n_ctx: n_ctx_value as i32,
            max_new_tokens: 500,
            kind,
        })
    }

    fn format_prompt(&self, system: &str, user: &str) -> String {
        match self.kind {
            ModelKind::Qwen => format!(
                "<|im_start|>system\n{}<|im_end|>\n\
                 <|im_start|>user\n{}<|im_end|>\n\
                 <|im_start|>assistant\n",
                system, user
            ),
            ModelKind::Phi3 => format!(
                "<|system|>\n{}<|end|>\n\
                 <|user|>\n{}<|end|>\n\
                 <|assistant|>",
                system, user
            ),
            ModelKind::Mistral => {
                // Mistral instruct format: system goes inside the first [INST].
                format!("[INST] {}\n\n{} [/INST]", system, user)
            }
        }
    }

    // Default helper for callers that just want a single-shot answer.
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        self.generate_with_system("You are a helpful assistant.", prompt)
    }

    // Used by the orchestrator agents (each agent has its own system prompt).
    pub fn generate_with_system(&mut self, system: &str, user: &str) -> Result<String> {
        let t0 = Instant::now();
        println!(
            "{} generate() called ({})",
            "[llama.cpp]".cyan().bold(),
            self.kind.label()
        );

        let formatted = self.format_prompt(system, user);

        let tokens_list = self
            .model
            .str_to_token(&formatted, self.kind.add_bos())
            .context("tokenization failed")?;

        println!(
            "{} prompt = {} tokens, max_new = {}",
            "[llama.cpp]".cyan().bold(),
            tokens_list.len(),
            self.max_new_tokens
        );

        // PERSISTENT SESSION — wipe KV cache without freeing buffers.
        self.ctx.clear_kv_cache();

        let n_prompt = tokens_list.len() as i32;
        if n_prompt + self.max_new_tokens > self.n_ctx {
            anyhow::bail!(
                "prompt ({} tokens) + max generation ({}) exceeds context ({})",
                n_prompt,
                self.max_new_tokens,
                self.n_ctx
            );
        }

        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in tokens_list.iter().enumerate() {
            let is_last = i as i32 == last_index;
            batch.add(*token, i as i32, &[0], is_last)?;
        }
        self.ctx.decode(&mut batch)?;

        let mut sampler = LlamaSampler::greedy();
        let mut n_cur = n_prompt;
        let mut output = String::new();

        for _ in 0..self.max_new_tokens {
            let new_token_id = sampler.sample(&self.ctx, batch.n_tokens() - 1);
            sampler.accept(new_token_id);

            if self.model.is_eog_token(new_token_id) {
                break;
            }

            #[allow(deprecated)]
            let piece = self
                .model
                .token_to_str(new_token_id, Special::Tokenize)
                .unwrap_or_default();
            output.push_str(&piece);

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true)?;
            n_cur += 1;
            self.ctx.decode(&mut batch)?;
        }

        let n_generated = n_cur - tokens_list.len() as i32;
        let elapsed = t0.elapsed().as_secs_f32();
        let tps = if elapsed > 0.0 {
            n_generated as f32 / elapsed
        } else {
            0.0
        };
        println!(
            "{} done: {} tokens in {:.2}s ({:.1} tok/s)",
            "[llama.cpp]".cyan().bold(),
            n_generated,
            elapsed,
            tps
        );

        Ok(output)
    }
}
