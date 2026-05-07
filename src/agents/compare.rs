use anyhow::Result;
use crate::backends::llama_cpp::LlamaBackend;

pub struct CompareAgent;

impl CompareAgent {
    pub fn new() -> Self { Self }

    pub fn run(&mut self, backend: &mut LlamaBackend, input: &str) -> Result<String> {
        let system = "You are an analytical assistant. \
            You will receive research notes. \
            Your job is to review them, remove redundancy, \
            and produce a clear, concise, well-structured final answer.";
        backend.generate_with_system(system, input)
    }
}
