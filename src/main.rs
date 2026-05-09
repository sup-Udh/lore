mod api;
mod backends;
mod scanner;
mod summarizer;


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
    Open {
        path: String,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ModelChoice {
    Qwen,
    Phi3,
    Mistral,
    #[value(name = "deepseek", alias = "deep-seek")]
    DeepSeek,
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
            ModelChoice::Qwen    => run_chat(ModelKind::Qwen,    "models/qwen2.5-7b-instruct-q4_k_m.gguf", "Qwen")?,
            ModelChoice::Phi3    => run_chat(ModelKind::Phi3,    "models/phi3-mini-4k-instruct-q4.gguf",      "Phi-3")?,
            ModelChoice::Mistral => run_chat(ModelKind::Mistral, "models/mistral-7b-v0.3.gguf",                "Mistral")?,
            ModelChoice::DeepSeek => run_chat(ModelKind::DeepSeek, "models/deepseek-r1-distill-qwen-32b.gguf", "DeepSeek")?,
        },
        Commands::Serve { model } => match model {
            ModelChoice::Qwen    => run_serve(ModelKind::Qwen,    "models/qwen2.5-1.5b-instruct-q4_k_m.gguf", "Qwen").await?,
            ModelChoice::Phi3    => run_serve(ModelKind::Phi3,    "models/phi3-mini-4k-instruct-q4.gguf",      "Phi-3").await?,
            ModelChoice::Mistral => run_serve(ModelKind::Mistral, "models/mistral-7b-v0.3.gguf",                "Mistral").await?,
            ModelChoice::DeepSeek => run_serve(ModelKind::DeepSeek, "models/deepseek-r1-distill-qwen-32b.gguf", "DeepSeek").await?,
        },

        Commands::Open { path } => {
            use std::path::Path;

            use scanner::filesystem::scan_project;
            use scanner::lore_dir::initialize_lore_directory;
        
            use summarizer::selector::select_important_files;
            use summarizer::summarizer::summarize_file;
            use summarizer::writer::write_summary;
        
            use backends::llama_cpp::{LlamaBackend, ModelKind};
        
            println!("\n[LORE] Opening project...\n");
        
            let root = Path::new(&path);
        
            // PHASE 1 — SCAN REPOSITORY
            println!("[LORE] Scanning repository...");
            let project_map = scan_project(root)?;
        
            // CREATE .lore/
            initialize_lore_directory(root, &project_map)?;
        
            println!(
                "[LORE] Scan complete. Files discovered: {}\n",
                project_map.files.len()
            );
        
            // PHI3 REPOSITORY ANALYSIS BACKEND
            let mut phi3_backend = LlamaBackend::new(
                "models/qwen2.5-7b-instruct-q4_k_m.gguf",
                ModelKind::Qwen,
            )?;
        
            println!("[LORE] Selecting important files...");
        
            // AI FILE SELECTION
            let important_files = select_important_files(
                &mut phi3_backend,
                &project_map,
            )?;
        
            println!(
                "[LORE] Phi3 selected {} important files.\n",
                important_files.len()
            );

            // PHASE 3 — SEQUENTIAL SUMMARIZATION
            let total = important_files.len();
            if total == 0 {
                println!("[LORE] No important files selected.");
                return Ok(());
            }

            // Best-effort attempt to reduce llama.cpp native logging noise by default.
            // (If you want full llama.cpp logs, set LORE_DEBUG=1.)
            std::env::set_var("LLAMA_LOG_LEVEL", "0");

            println!("[LORE] Summarizing files sequentially...\n");

            // Sequential processing
            for (i, file) in important_files.iter().enumerate() {
                println!("[LORE] Summarizing {}/{}: {}", i + 1, total, file.path);
                
                let full_path = root.join(&file.path);
                let contents = std::fs::read_to_string(&full_path).unwrap_or_default();
                
                // Use a stable, unique output name based on relative path to avoid overwriting
                // summaries for common basenames like `mod.rs`.
                let mut safe = file
                    .path
                    .replace('\\', "__")
                    .replace('/', "__")
                    .replace(':', "_");
                if safe.len() > 180 {
                    safe.truncate(180);
                }
                let output_name = format!("{}.md", safe);
                
                // Summarize the file
                let summary = summarize_file(&mut phi3_backend, &file.path, &contents)?;
                
                // Persist the summary
                write_summary(root, &output_name, &summary)?;
                
                println!("[LORE] Persisted summary: {}", output_name);
            }

            println!("\n[LORE] Repository summaries generated successfully.");
        }

           
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
