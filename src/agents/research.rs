use anyhow::Result;
use crate::backends::llama_cpp::LlamaBackend;

pub struct ResearchAgent;

impl ResearchAgent {
    pub fn new() -> Self { Self }

    pub fn run(&mut self, backend: &mut LlamaBackend, input: &str) -> Result<String> {
        let system = "You are a research assistant. \
            Read the user input carefully, extract all key concepts, \
            expand on relevant details, and produce a thorough factual analysis. \
            Be precise and comprehensive.";
        backend.generate_with_system(system, input)
    }
}
