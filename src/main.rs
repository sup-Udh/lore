use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;
use std::io::Write;

fn main() -> Result<()> {
    // --------------------------------------------------
    // 🔧 1. DEVICE SETUP
    // --------------------------------------------------

    let device = Device::Cpu;


    // --------------------------------------------------
    // 🔤 2. LOAD TOKENIZER
    // --------------------------------------------------

    let tokenizer = Tokenizer::from_file("models/tokenizer.json")
        .map_err(E::msg)?;


    // --------------------------------------------------
    // 📦 3. LOAD GGUF MODEL
    // --------------------------------------------------

    let model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";

    let mut file = std::fs::File::open(model_path)
        .map_err(|_| E::msg(format!("Could not find model file at {}", model_path)))?;

    let content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)?;


    // --------------------------------------------------
    // 💬 4. PREPARE PROMPT
    // --------------------------------------------------
    // Convert user input into token IDs
    // Example:
    // "hello" -> [840, 20772, ...]
    let prompt = "explain what llm models are in simple terms";

    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let prompt_tokens = tokens.get_ids();

    println!("Prompt: {}\nResponse: ", prompt);


    // --------------------------------------------------
    // 🧠 5. PREFILL (UNDERSTAND FULL PROMPT)
    // --------------------------------------------------
    // First pass: feed entire sentence to the model
    // This builds context (like reading the whole question)
    //
    // Shape transformation:
    // [seq_len] -> [1, seq_len] (batch dimension added)
    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;

    // offset = 0 → start of sequence
    let logits = model.forward(&input, 0)?;


    // --------------------------------------------------
    // 🎯 6. GET FIRST GENERATED TOKEN
    // --------------------------------------------------
    // Model outputs logits = probabilities for next token
    //
    // Shape could be:
    // [1, seq, vocab] OR [seq, vocab]
    //
    // We always want:
    // 👉 last token prediction (end of sequence)
    let mut next_token = {
        let shape = logits.dims();

        let last_row = match shape.len() {
            // [batch, seq, vocab]
            3 => logits.get(0)?.get(shape[1] - 1)?,

            // [seq, vocab]
            2 => logits.get(shape[0] - 1)?,

            // fallback safety
            _ => logits.clone(),
        };

        // Pick most probable token (greedy decoding)
        last_row.argmax(0)?.to_scalar::<u32>()?
    };


    // Decode and print first generated token
    let first_word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
    print!("{}", first_word);
    std::io::stdout().flush()?;


    // --------------------------------------------------
    // 🔁 7. GENERATION LOOP (CORE LLM LOGIC)
    // --------------------------------------------------
    // Now we generate tokens one-by-one
    //
    // Important idea:
    // - First step = full prompt (context building)
    // - Next steps = single token input (efficient decoding)
    //
    // Model remembers previous tokens internally
    for i in 0..100 {

        // Feed ONLY last generated token
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;

        // Position in sequence (VERY IMPORTANT)
        // Tells model where we are in context
        let pos = prompt_tokens.len() + i;

        let logits = model.forward(&input, pos)?;


        // --------------------------------------------------
        // 🎯 EXTRACT NEXT TOKEN
        // --------------------------------------------------
        // Shape during generation is usually:
        // [1, 1, vocab] OR [1, vocab]
        let next = {
            let shape = logits.dims();

            let last_logits = match shape.len() {
                // [1, 1, vocab]
                3 => logits.get(0)?.get(0)?,

                // [1, vocab]
                2 => logits.get(0)?,

                // fallback
                _ => logits.squeeze(0)?,
            };

            last_logits.argmax(0)?.to_scalar::<u32>()?
        };

        next_token = next;


        // --------------------------------------------------
        // 🛑 STOP CONDITION
        // --------------------------------------------------
        // These token IDs represent end-of-sequence
        // When model "decides it's done"
        if next_token == 151643 || next_token == 151645 {
            break;
        }


        // --------------------------------------------------
        // 🖨️ STREAM OUTPUT
        // --------------------------------------------------
        // Decode token → word and print immediately
        // Gives real-time typing effect
        let word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{}", word);
        std::io::stdout().flush()?;
    }


    // --------------------------------------------------
    // ✅ DONE
    // --------------------------------------------------
    println!("\n\nDone!");
    Ok(())
}