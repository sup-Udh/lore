use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen2::ModelWeights as QwenWeights;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3Weights;
use candle_transformers::models::quantized_llama::ModelWeights as MistralWeights;
use tokenizers::Tokenizer; 
use std::io::{self, Write};
use clap::{Parser, Subcommand, ValueEnum};
use colored::*;

#[derive(Parser)]
#[command(name = "lore", about = "Local LLM CLI", version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Chat {
        #[arg(short, long, value_enum, default_value = "qwen")]
        model: ModelChoice,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ModelChoice {
    Qwen,
    Phi3,
    Mistral,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = Device::Cpu;

    println!("{}", r#"
    ██╗      ██████╗ ██████╗ ███████╗
    ██║     ██╔═══██╗██╔══██╗██╔════╝
    ██║     ██║   ██║██████╔╝█████╗  
    ██║     ██║   ██║██╔══██╗██╔══╝  
    ███████╗╚██████╔╝██║  ██║███████╗
    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
    "# .cyan().bold());
    println!("{}", "--- Local Intelligence Engine Initialized ---".black().on_white());

    match cli.command {
        Commands::Chat { model } => {
            match model {
                ModelChoice::Qwen => run_chat_qwen(&device)?,
                ModelChoice::Mistral => run_chat_mistral(&device)?,
                ModelChoice::Phi3 => run_chat_phi3(&device)?,
            }
        }
    }

    Ok(())
}

fn run_chat_qwen(device: &Device) -> Result<()> {
    println!("{}", "Loading Qwen 2.5...".yellow());
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").map_err(E::msg)?;
    let model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = QwenWeights::from_gguf(content, &mut file, &device)?;

    chat_loop(device, tokenizer, |t, p| model.forward(t, p), vec![151643, 151645], "Qwen")
}

fn run_chat_phi3(device: &Device) -> Result<()> {
    println!("{}", "Loading Phi-3...".yellow());
    let tokenizer = Tokenizer::from_file("models/phi3_tokenizer.json").map_err(E::msg)?;
    let model_path = "models/phi3-mini-4k-instruct-q4.gguf";
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = Phi3Weights::from_gguf(false, content, &mut file, &device)?;

    chat_loop(device, tokenizer, |t, p| model.forward(t, p), vec![32000, 32007], "Phi-3")
}

fn run_chat_mistral(device: &Device) -> Result<()> {
    println!("{}", "Loading Mistral 7B v0.3...".yellow());
    let tokenizer = Tokenizer::from_file("models/mistral_tokenizer.json").map_err(E::msg)?;
    // Ensure the filename matches exactly what is in your models folder
    let model_path = "models/mistral-7b-v0.3.gguf"; 
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = MistralWeights::from_gguf(content, &mut file, &device)?;

    chat_loop(device, tokenizer, |t, p| model.forward(t, p), vec![2, 28723], "Mistral")
}

fn chat_loop<F>(
    device: &Device, 
    tokenizer: Tokenizer, 
    mut forward: F, 
    eos_tokens: Vec<u32>,
    model_name: &str
) -> Result<()> 
where F: FnMut(&Tensor, usize) -> candle_core::Result<Tensor> 
{
    println!("{} Mode Active. Type 'exit' to quit.", model_name.green());
    let mut total_pos = 0;

    loop {
        print!("\n{} > ", "You".blue().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "exit" { break; }
        if input.is_empty() { continue; }

        let formatted_input = match model_name {
            "Phi-3" => format!("<|user|>\n{}<|end|>\n<|assistant|>", input),
            "Mistral" => format!("<s>[INST] {} [/INST]", input),
            _ => format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", input),
        };

        let tokens = tokenizer.encode(formatted_input, true).map_err(E::msg)?;
        let prompt_tokens = tokens.get_ids();
        
        print!("\n{}: ", model_name.purple().bold());
        
        let mut tokens_to_process = prompt_tokens.to_vec();
        
        // Use a persistent decoder state to handle spacing correctly
        let mut decoder = TokenOutputStream::new(tokenizer.clone());

        for _ in 0..500 {
            let input_tensor = Tensor::new(tokens_to_process.as_slice(), device)?.unsqueeze(0)?;
            let logits = forward(&input_tensor, total_pos)?;
            total_pos += tokens_to_process.len();
            
            let next_token = get_next_token(&logits)?;
            
            if eos_tokens.contains(&next_token) { break; }

            // Fix Spacing: Use a streaming decoder approach
            if let Some(t) = decoder.next_token(next_token)? {
                print!("{}", t);
                io::stdout().flush()?;
            }

            tokens_to_process = vec![next_token];
        }
        println!();
    }
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

/// Helper struct to handle the "missing space" issue in token streaming
struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
}

impl TokenOutputStream {
    fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer, tokens: Vec::new(), prev_index: 0 }
    }

    fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        self.tokens.push(token);
        let full_text = self.tokenizer.decode(&self.tokens, true).map_err(E::msg)?;
        let readable_text = &full_text[self.prev_index..];
        
        if readable_text.is_empty() {
            return Ok(None);
        }

        self.prev_index = full_text.len();
        Ok(Some(readable_text.to_string()))
    }
}