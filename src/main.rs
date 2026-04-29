use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen2::ModelWeights as QwenWeights;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3Weights;
use tokenizers::{Tokenizer}; 
use std::io::Write;



fn main() -> Result<()> {
    let device = Device::Cpu;

    println!("--- 🤖 Running Qwen 2.5 ---");
    run_qwen(&device)?;

    println!("\n--- 🤖 Running Phi 3 ---");
    run_phi3(&device)?;

    Ok(())
}





fn run_phi3(device: &Device) -> Result<()> {
    // --------------------------------------------------
    // 📦 LOAD PHI-3 TOKENIZER & MODEL
    // --------------------------------------------------
    let tokenizer = Tokenizer::from_file("models/phi3_tokenizer.json")
        .map_err(E::msg)?;

    let model_path = "models/phi3-mini-4k-instruct-q4.gguf";
    let mut file = std::fs::File::open(model_path)
        .map_err(|_| E::msg(format!("Could not find Phi-3 model file at {}", model_path)))?;

    let content = gguf_file::Content::read(&mut file)?;
// Change this line in run_phi3:
    let mut model = Phi3Weights::from_gguf(false, content, &mut file, &device)?;
    // --------------------------------------------------
    // 💬 PREPARE CODING PROMPT
    // --------------------------------------------------
    // Phi-3 Chat Template: <|user|>\n{prompt}<|end|>\n<|assistant|>
    let prompt = "<|user|>\nWrite a simple Rust function to calculate the Fibonacci sequence.<|end|>\n<|assistant|>";

    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let prompt_tokens = tokens.get_ids();

    println!("Prompt: Coding Task\nResponse: ");

    // --------------------------------------------------
    // 🧠 GENERATION (PREFILL & LOOP)
    // --------------------------------------------------
    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;

    let mut next_token = {
        let shape = logits.dims();
        let last_row = match shape.len() {
            3 => logits.get(0)?.get(shape[1] - 1)?,
            2 => logits.get(shape[0] - 1)?,
            _ => logits.clone(),
        };
        last_row.argmax(0)?.to_scalar::<u32>()?
    };

    print!("{}", tokenizer.decode(&[next_token], true).map_err(E::msg)?);
    std::io::stdout().flush()?;

    for i in 0..200 {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let pos = prompt_tokens.len() + i;
        let logits = model.forward(&input, pos)?;

        let next = {
            let shape = logits.dims();
            let last_logits = match shape.len() {
                3 => logits.get(0)?.get(0)?,
                2 => logits.get(0)?,
                _ => logits.squeeze(0)?,
            };
            last_logits.argmax(0)?.to_scalar::<u32>()?
        };

        next_token = next;

        // Stop tokens for Phi-3 (typically 32000 for <|endoftext|> or 32007 for <|end|>)
        if next_token == 32000 || next_token == 32007 {
            break;
        }

        let word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{}", word);
        std::io::stdout().flush()?;
    }

    println!("\n\nPhi-3 Done!");
    Ok(())
}

// Keep your existing run_qwen function below...




fn run_qwen(device: &Device) -> Result<()> {
    // Now that 'use tokenizers::Tokenizer' is added above, this will work
    let tokenizer = Tokenizer::from_file("models/tokenizer.json")
        .map_err(E::msg)?;

    // --------------------------------------------------
    // 📦 LOAD GGUF MODEL
    // --------------------------------------------------
    let model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";

    let mut file = std::fs::File::open(model_path)
        .map_err(|_| E::msg(format!("Could not find model file at {}", model_path)))?;

    let content = gguf_file::Content::read(&mut file)?;
    
    // We use QwenWeights because of the 'as QwenWeights' alias in the imports
    let mut model = QwenWeights::from_gguf(content, &mut file, &device)?;

    // --------------------------------------------------
    // 💬 PREPARE PROMPT
    // --------------------------------------------------
    let prompt = "explain what llm models are in simple terms";

    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let prompt_tokens = tokens.get_ids();

    println!("Prompt: {}\nResponse: ", prompt);

    // --------------------------------------------------
    // 🧠 PREFILL (UNDERSTAND FULL PROMPT)
    // --------------------------------------------------
    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;

    // --------------------------------------------------
    // 🎯 GET FIRST GENERATED TOKEN
    // --------------------------------------------------
    let mut next_token = {
        let shape = logits.dims();

        let last_row = match shape.len() {
            3 => logits.get(0)?.get(shape[1] - 1)?,
            2 => logits.get(shape[0] - 1)?,
            _ => logits.clone(),
        };

        last_row.argmax(0)?.to_scalar::<u32>()?
    };

    let first_word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
    print!("{}", first_word);
    std::io::stdout().flush()?;

    // --------------------------------------------------
    // 🔁 GENERATION LOOP (CORE LLM LOGIC)
    // --------------------------------------------------
    for i in 0..100 {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let pos = prompt_tokens.len() + i;
        let logits = model.forward(&input, pos)?;

        let next = {
            let shape = logits.dims();
            let last_logits = match shape.len() {
                3 => logits.get(0)?.get(0)?,
                2 => logits.get(0)?,
                _ => logits.squeeze(0)?,
            };
            last_logits.argmax(0)?.to_scalar::<u32>()?
        };

        next_token = next;

        // Stop tokens for Qwen 2.5
        if next_token == 151643 || next_token == 151645 {
            break;
        }

        let word = tokenizer.decode(&[next_token], true).map_err(E::msg)?; // decoding the token back to normal text and then prompting back.

        print!("{}", word);
        std::io::stdout().flush()?;
    }


    println!("\n\nDone!");


    Ok(())
}