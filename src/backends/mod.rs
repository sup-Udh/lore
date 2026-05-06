use anyhow::Result;

pub trait backend {
    fn generate(&mut self, prompt: &str) -> Result<String>;

}

pub mod llama_cpp;

