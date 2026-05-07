mod api;
mod agents;
mod backends;

use anyhow::Result;
use std::io::{self, Write};
use clap::{Parser, Subcommand, ValueEnum};
use colored::*;

use backends::llama_cpp::{LlamaBackend, ModelKind};

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
    // Rayon/OpenMP thread pools — used by some matmul backends llama.cpp
    // links against (BLAS, OpenBLAS, etc.).
    std::env::set_var("RAYON_NUM_THREADS", num_cpus.to_string());
    std::env::set_var("OMP_NUM_THREADS", num_cpus.to_string());

    let cli = Cli::parse();

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
        Commands::Chat { model } => match model {
            ModelChoice::Qwen    => run_chat(ModelKind::Qwen,    "models/qwen2.5-1.5b-instruct-q4_k_m.gguf", "Qwen")?,
            ModelChoice::Phi3    => run_chat(ModelKind::Phi3,    "models/phi3-mini-4k-instruct-q4.gguf",      "Phi-3")?,
            ModelChoice::Mistral => run_chat(ModelKind::Mistral, "models/mistral-7b-v0.3.gguf",                "Mistral")?,
        },
        Commands::Serve { model } => match model {
            ModelChoice::Qwen    => run_serve(ModelKind::Qwen,    "models/qwen2.5-1.5b-instruct-q4_k_m.gguf", "Qwen").await?,
            ModelChoice::Phi3    => run_serve(ModelKind::Phi3,    "models/phi3-mini-4k-instruct-q4.gguf",      "Phi-3").await?,
            ModelChoice::Mistral => run_serve(ModelKind::Mistral, "models/mistral-7b-v0.3.gguf",                "Mistral").await?,
        },
    }

    Ok(())
}

fn run_chat(kind: ModelKind, model_path: &str, model_name: &str) -> Result<()> {
    println!("{}", format!("Loading {} (llama.cpp backend)...", model_name).yellow());
    let mut backend = LlamaBackend::new(model_path, kind)?;
    chat_loop(&mut backend, model_name)
}

async fn run_serve(kind: ModelKind, model_path: &str, model_name: &str) -> Result<()> {
    println!("{}", format!("Loading {} for API server (llama.cpp backend)...", model_name).yellow());
    let backend = LlamaBackend::new(model_path, kind)?;
    api::start_api_llama(backend, model_name).await;
    Ok(())
}

fn chat_loop(backend: &mut LlamaBackend, model_name: &str) -> Result<()> {
    println!("{} Mode Active. Type 'exit' to quit.", model_name.green());

    loop {
        print!("\n{} > ", "You".blue().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "exit" { break; }
        if input.is_empty() { continue; }

        if model_name == "Qwen" {
            // REAL-TIME TOKEN STREAMING
            // STREAM TOKENS DIRECTLY TO TERMINAL
            // LOW-LATENCY INFERENCE OUTPUT
            //
            // Qwen — direct, no agents
            let mut out = io::stdout();
            write!(out, "\n{}: ", model_name.purple().bold())?;
            out.flush()?;
            backend.generate_stream(input, |chunk| {
                let _ = write!(out, "{}", chunk);
                let _ = out.flush();
            })?;
            writeln!(out)?;
        } else {
            // MULTI-AGENT PIPELINE TEMPORARILY DISABLED
            // DIRECT INFERENCE MODE ENABLED
            // ORCHESTRATOR BYPASSED FOR PERFORMANCE
            //
            // Phi-3 / Mistral now follow the same direct llama.cpp inference path as Qwen.
            // All agent/orchestrator code remains in-tree for later re-enabling.
            // REAL-TIME TOKEN STREAMING
            // STREAM TOKENS DIRECTLY TO TERMINAL
            // LOW-LATENCY INFERENCE OUTPUT
            let mut out = io::stdout();
            write!(out, "\n{}: ", model_name.purple().bold())?;
            out.flush()?;
            backend.generate_stream(input, |chunk| {
                let _ = write!(out, "{}", chunk);
                let _ = out.flush();
            })?;
            writeln!(out)?;
        }
    }
    Ok(())
}
