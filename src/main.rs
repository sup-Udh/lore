use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;
use std::io::Write;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Tokenizer
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").map_err(E::msg)?;

    // 2. Open GGUF file
    let model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    let mut file = std::fs::File::open(model_path)
        .map_err(|_| E::msg(format!("Could not find model file at {}", model_path)))?;
    
    // 3. Load GGUF Content and Model
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)?;

    // 4. Prepare Prompt
    let prompt = "why is the sky blue? Answer in a concise manner.";
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let prompt_tokens = tokens.get_ids();

    println!("Prompt: {}\nResponse: ", prompt);

    // 5. Initial Step: Process the whole prompt
    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    
    // FIX: Instead of dims3, we check how many dimensions we actually have
    let mut next_token = {
        let shape = logits.dims();
        let last_row = match shape.len() {
            3 => logits.get(0)?.get(shape[1] - 1)?, // Handle [1, seq, vocab]
            2 => logits.get(shape[0] - 1)?,         // Handle [seq, vocab]
            _ => logits.clone(),                   // Fallback
        };
        last_row.argmax(0)?.to_scalar::<u32>()?
    };

    let first_word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
    print!("{}", first_word);
    std::io::stdout().flush()?;

    // 6. Generation Loop
    for i in 0..100 {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let pos = prompt_tokens.len() + i;
        let logits = model.forward(&input, pos)?;
        
        // FIX: Extract token from the very end of the tensor safely
        next_token = {
            let shape = logits.dims();
            let last_logits = match shape.len() {
                3 => logits.get(0)?.get(0)?, // [1, 1, vocab]
                2 => logits.get(0)?,         // [1, vocab]
                _ => logits.squeeze(0)?,     // Fallback
            };
            last_logits.argmax(0)?.to_scalar::<u32>()?
        };

        if next_token == 151643 || next_token == 151645 { break; }

        let word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{}", word);
        std::io::stdout().flush()?; 
    }

    println!("\n\nDone!");
    Ok(())
}