mod cli;
mod core;
mod models;

use clap::{Parser, Subcommand};


#[derive(Parser)]
#[command(name="lore")]
#[command(about="Run llms locally with ease")]

struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a model
    Run {
        /// The model to run
        model: String,
    } , 
    Features,
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Run { model } => {
            cli::run(model);
        }
        Commands::Features => {
            cli::features_command();
        }
     
    }
}