use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen2::ModelWeights as QwenWeights;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3Weights;
use tokenizers::Tokenizer; 
use std::io::Write;

fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("--- 🤖 Running Qwen 2.5 with KV Caching ---");
    run_qwen(&device)?;

    println!("\n------------------------------------------\n");

    println!("--- 🤖 Running Phi-3 with KV Caching ---");
    run_phi3(&device)?;

    Ok(())
}

fn run_phi3(device: &Device) -> Result<()> {
    let tokenizer = Tokenizer::from_file("models/phi3_tokenizer.json").map_err(E::msg)?;
    let model_path = "models/phi3-mini-4k-instruct-q4.gguf";
    
    let mut file = std::fs::File::open(model_path)
        .map_err(|_| E::msg(format!("Could not find Phi-3 model file at {}", model_path)))?;

    let content = gguf_file::Content::read(&mut file)?;
    let mut model = Phi3Weights::from_gguf(false, content, &mut file, &device)?;

    let prompt = "<|user|>\n write me a simple html and css page with a h1 and then css to bring it to to the center build the entire boilderplate <|end|>\n<|assistant|>";
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let prompt_tokens = tokens.get_ids();

    println!("Prompt: Coding Task\nResponse: ");

    // 🧠 1. PREFILL PHASE
    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?; 

    let mut next_token = get_next_token(&logits)?;

    print!("{}", tokenizer.decode(&[next_token], true).map_err(E::msg)?);
    std::io::stdout().flush()?;

    // 🔁 2. GENERATION LOOP
    for i in 0..200 {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let pos = prompt_tokens.len() + i;
        let logits = model.forward(&input, pos)?;

        next_token = get_next_token(&logits)?;

        if next_token == 32000 || next_token == 32007 { break; }

        let word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{}", word);
        std::io::stdout().flush()?;
    }

    println!("\n\nPhi-3 Done!");
    Ok(())
}

fn run_qwen(device: &Device) -> Result<()> {
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").map_err(E::msg)?;
    let model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";

    let mut file = std::fs::File::open(model_path)
        .map_err(|_| E::msg(format!("Could not find Qwen model file at {}", model_path)))?;

    let content = gguf_file::Content::read(&mut file)?;
    let mut model = QwenWeights::from_gguf(content, &mut file, &device)?;

    let prompt = "explain what llm models are in simple terms";
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let prompt_tokens = tokens.get_ids();

    println!("Prompt: {}\nResponse: ", prompt);

    // 🧠 1. PREFILL PHASE
    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;

    let mut next_token = get_next_token(&logits)?;

    print!("{}", tokenizer.decode(&[next_token], true).map_err(E::msg)?);
    std::io::stdout().flush()?;

    // 🔁 2. GENERATION LOOP
    for i in 0..100 {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let pos = prompt_tokens.len() + i;
        let logits = model.forward(&input, pos)?;

        next_token = get_next_token(&logits)?;

        if next_token == 151643 || next_token == 151645 { break; }

        let word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{}", word);
        std::io::stdout().flush()?;
    }

    println!("\n\nQwen Done!");
    Ok(())
}

// 🛠️ HELPER: Extract the last token from logits safely
fn get_next_token(logits: &Tensor) -> Result<u32> {
    let shape = logits.dims();
    let last_row = match shape.len() {
        3 => logits.get(0)?.get(shape[1] - 1)?, // Handle [batch, seq, vocab]
        2 => logits.get(shape[0] - 1)?,         // Handle [seq, vocab]
        _ => logits.clone(),
    };
    let next_id = last_row.argmax(0)?.to_scalar::<u32>()?;
    Ok(next_id)
}