// main root module of the cli functions.

use crate::core::runner;
pub mod feature;

pub fn run(model: String) {
    println!("Running model: {}", model);

    let prompt = "Explain Rust simply";

    match runner::run_model(&model, prompt) {
        Ok(output) => println!("AI: {}", output),
        Err(e) => eprintln!("Error: {}", e),
    }
}

pub fn features_command() {
    feature::features();   // 👈 call the function
}