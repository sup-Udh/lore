use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen2::ModelWeights as QwenWeights;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3Weights;
use tokenizers::Tokenizer; 
use std::io::Write;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "lore", about = "Local LLM CLI", version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a chat with a specific model
    Chat {
        #[arg(short, long, value_enum, default_value = "qwen")]
        model: ModelChoice,

        #[arg(short, long)]
        prompt: String,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ModelChoice {
    Qwen,
    Phi3,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = Device::Cpu;

    match cli.command {
        Commands::Chat { model, prompt } => {
            match model {
                ModelChoice::Qwen => run_qwen(&device, &prompt)?,
                ModelChoice::Phi3 => {
                    // Wrap Phi-3 prompt in its specific instruct template
                    let instruct_prompt = format!("<|user|>\n{}<|end|>\n<|assistant|>", prompt);
                    run_phi3(&device, &instruct_prompt)?;
                }
            }
        }
    }

    Ok(())
}

fn run_phi3(device: &Device, prompt: &str) -> Result<()> {
    let tokenizer = Tokenizer::from_file("models/phi3_tokenizer.json").map_err(E::msg)?;
    let model_path = "models/phi3-mini-4k-instruct-q4.gguf";
    
    let mut file = std::fs::File::open(model_path)
        .map_err(|_| E::msg(format!("Could not find Phi-3 model file at {}", model_path)))?;

    let content = gguf_file::Content::read(&mut file)?;
    let mut model = Phi3Weights::from_gguf(false, content, &mut file, &device)?;

    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let prompt_tokens = tokens.get_ids();

    println!("--- 🤖 Lore (Phi-3) ---\n");

    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?; 

    let mut next_token = get_next_token(&logits)?;
    print!("{}", tokenizer.decode(&[next_token], true).map_err(E::msg)?);
    std::io::stdout().flush()?;

    for i in 0..500 {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let pos = prompt_tokens.len() + i;
        let logits = model.forward(&input, pos)?;
        next_token = get_next_token(&logits)?;

        if next_token == 32000 || next_token == 32007 { break; }

        let word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{}", word);
        std::io::stdout().flush()?;
    }

    println!("\n");
    Ok(())
}

fn run_qwen(device: &Device, prompt: &str) -> Result<()> {
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").map_err(E::msg)?;
    let model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";

    let mut file = std::fs::File::open(model_path)
        .map_err(|_| E::msg(format!("Could not find Qwen model file at {}", model_path)))?;

    let content = gguf_file::Content::read(&mut file)?;
    let mut model = QwenWeights::from_gguf(content, &mut file, &device)?;

    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let prompt_tokens = tokens.get_ids();

    println!("--- 🤖 Lore (Qwen 2.5) ---\n");

    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;

    let mut next_token = get_next_token(&logits)?;
    print!("{}", tokenizer.decode(&[next_token], true).map_err(E::msg)?);
    std::io::stdout().flush()?;

    for i in 0..500 {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let pos = prompt_tokens.len() + i;
        let logits = model.forward(&input, pos)?;
        next_token = get_next_token(&logits)?;

        if next_token == 151643 || next_token == 151645 { break; }

        let word = tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        print!("{}", word);
        std::io::stdout().flush()?;
    }

    println!("\n");
    Ok(())
}

fn get_next_token(logits: &Tensor) -> Result<u32> {
    let shape = logits.dims();
    let last_row = match shape.len() {
        3 => logits.get(0)?.get(shape[1] - 1)?,
        2 => logits.get(shape[0] - 1)?,
        _ => logits.clone(),
    };
    let next_id = last_row.argmax(0)?.to_scalar::<u32>()?;
    Ok(next_id)
}