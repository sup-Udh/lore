// Agents now run on top of LlamaBackend (llama.cpp).
// The Candle-based generate() / InferenceContext / get_next_token are gone —
// each agent calls backend.generate_with_system(system, input) directly.

pub mod orchestrator;
pub mod research;
pub mod compare;

use serde::Serialize;

#[derive(Serialize, Debug, Clone)]
pub struct AgentStep {
    pub agent: String,
    pub input: String,
    pub output: String,
}

#[derive(Serialize, Debug)]
pub struct Trace {
    pub steps: Vec<AgentStep>,
    pub final_output: String,
}
