mod api;
mod agents;

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

pub trait Model {
    fn forward(&mut self, input: &Tensor, pos: usize) -> candle_core::Result<Tensor>;
}

struct QwenModel(QwenWeights);
impl Model for QwenModel {
    fn forward(&mut self, input: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        self.0.forward(input, pos)
    }
}

struct Phi3Model(Phi3Weights);
impl Model for Phi3Model {
    fn forward(&mut self, input: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        self.0.forward(input, pos)
    }
}

struct MistralModel(MistralWeights);
impl Model for MistralModel {
    fn forward(&mut self, input: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        self.0.forward(input, pos)
    }
}

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
    Serve {
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

#[tokio::main]
async fn main() -> Result<()> {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // OPTIMIZATION: Maximize CPU usage via Rayon thread pool
    std::env::set_var("RAYON_NUM_THREADS", num_cpus.to_string());
    // Some BLAS/linear-algebra backends (used by candle matmul) respect OMP_NUM_THREADS
    std::env::set_var("OMP_NUM_THREADS", num_cpus.to_string());

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
    println!("Threads: {}", num_cpus);

    match cli.command {
        Commands::Chat { model } => {
            match model {
                ModelChoice::Qwen    => run_chat_qwen(&device)?,
                ModelChoice::Mistral => run_chat_mistral(&device)?,
                ModelChoice::Phi3    => run_chat_phi3(&device)?,
            }
        }
        Commands::Serve { model } => {
            match model {
                ModelChoice::Qwen    => run_serve_qwen(&device).await?,
                ModelChoice::Mistral => run_serve_mistral(&device).await?,
                ModelChoice::Phi3    => run_serve_phi3(&device).await?,
            }
        }
    }

    Ok(())
}

// ── Chat runners ──────────────────────────────────────────────────────────────
// Each runner loads the model into its wrapper and passes &mut dyn Model
// to chat_loop so the orchestrator can borrow it without owning it.

fn run_chat_qwen(device: &Device) -> Result<()> {
    println!("{}", "Loading Qwen 2.5...".yellow());
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").map_err(E::msg)?;
    let model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    // OPTIMIZATION: Using memory-mapped GGUF loading to avoid full file copy and reduce RAM usage
    let file = std::fs::File::open(model_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    let content = gguf_file::Content::read(&mut cursor)?;
    let mut model = QwenModel(QwenWeights::from_gguf(content, &mut cursor, device)?);
    chat_loop(device, tokenizer, &mut model, vec![151643, 151645], "Qwen")
}

fn run_chat_phi3(device: &Device) -> Result<()> {
    println!("{}", "Loading Phi-3...".yellow());
    let tokenizer = Tokenizer::from_file("models/phi3_tokenizer.json").map_err(E::msg)?;
    let model_path = "models/phi3-mini-4k-instruct-q4.gguf";
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = Phi3Model(Phi3Weights::from_gguf(false, content, &mut file, device)?);
    chat_loop(device, tokenizer, &mut model, vec![32000, 32007], "Phi-3")
}

fn run_chat_mistral(device: &Device) -> Result<()> {
    println!("{}", "Loading Mistral 7B v0.3...".yellow());
    let tokenizer = Tokenizer::from_file("models/mistral_tokenizer.json").map_err(E::msg)?;
    let model_path = "models/mistral-7b-v0.3.gguf";
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut model = MistralModel(MistralWeights::from_gguf(content, &mut file, device)?);
    chat_loop(device, tokenizer, &mut model, vec![2, 28723], "Mistral")
}

// ── Serve runners ─────────────────────────────────────────────────────────────

async fn run_serve_qwen(device: &Device) -> Result<()> {
    println!("{}", "Loading Qwen 2.5 for API server...".yellow());
    let tokenizer = Tokenizer::from_file("models/tokenizer.json").map_err(E::msg)?;
    let model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    // OPTIMIZATION: Using memory-mapped GGUF loading to avoid full file copy and reduce RAM usage
    let file = std::fs::File::open(model_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    let content = gguf_file::Content::read(&mut cursor)?;
    let model = QwenWeights::from_gguf(content, &mut cursor, device)?;
    api::start_api(Box::new(QwenModel(model)), tokenizer, device.clone(), vec![151643, 151645], "Qwen").await;
    Ok(())
}

async fn run_serve_phi3(device: &Device) -> Result<()> {
    println!("{}", "Loading Phi-3 for API server...".yellow());
    let tokenizer = Tokenizer::from_file("models/phi3_tokenizer.json").map_err(E::msg)?;
    let model_path = "models/phi3-mini-4k-instruct-q4.gguf";
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let model = Phi3Weights::from_gguf(false, content, &mut file, device)?;
    api::start_api(Box::new(Phi3Model(model)), tokenizer, device.clone(), vec![32000, 32007], "Phi-3").await;
    Ok(())
}

async fn run_serve_mistral(device: &Device) -> Result<()> {
    println!("{}", "Loading Mistral 7B v0.3 for API server...".yellow());
    let tokenizer = Tokenizer::from_file("models/mistral_tokenizer.json").map_err(E::msg)?;
    let model_path = "models/mistral-7b-v0.3.gguf";
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let model = MistralWeights::from_gguf(content, &mut file, device)?;
    api::start_api(Box::new(MistralModel(model)), tokenizer, device.clone(), vec![2, 28723], "Mistral").await;
    Ok(())
}

// ── Chat loop ─────────────────────────────────────────────────────────────────

fn chat_loop(
    device: &Device,
    tokenizer: Tokenizer,
    model: &mut dyn Model,
    eos_tokens: Vec<u32>,
    model_name: &str,
) -> Result<()> {
    use agents::{InferenceContext, generate, orchestrator::Orchestrator};

    println!("{} Mode Active. Type 'exit' to quit.", model_name.green());

    let ctx = InferenceContext {
        tokenizer: &tokenizer,
        device,
        eos_tokens: &eos_tokens,
        model_name,
    };

    // Orchestrator is ZST-backed so creating it is free even if Qwen never uses it
    let mut orchestrator = Orchestrator::new();

    loop {
        print!("\n{} > ", "You".blue().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "exit" { break; }
        if input.is_empty() { continue; }

        if model_name == "Qwen" {
            // Direct single-pass inference — no agents, no overhead
            let response = generate(model, &ctx, "You are a helpful assistant.", input)?;
            println!("\n{}: {}", model_name.purple().bold(), response);
        } else {
            // Multi-agent pipeline for Phi-3 and Mistral
            let trace = orchestrator.run(model, &ctx, input.to_string())?;
            for (i, step) in trace.steps.iter().enumerate() {
                println!("\n{}", format!("[Step {}] {}", i + 1, step.agent).yellow().bold());
                println!("{}", step.output.trim().dimmed());
            }
            println!("\n{}: {}", model_name.purple().bold(), trace.final_output);
        }
    }
    Ok(())
}


