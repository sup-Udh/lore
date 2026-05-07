use anyhow::Result;
use colored::*;
use super::{AgentStep, Trace, research::ResearchAgent, compare::CompareAgent};
use crate::backends::llama_cpp::LlamaBackend;

pub struct Orchestrator {
    research: ResearchAgent,
    compare: CompareAgent,
}

impl Orchestrator {
    pub fn new() -> Self {
        Self {
            research: ResearchAgent::new(),
            compare: CompareAgent::new(),
        }
    }

    pub fn run(&mut self, backend: &mut LlamaBackend, input: String) -> Result<Trace> {
        let mut trace = Trace { steps: Vec::new(), final_output: String::new() };

        println!("{}", "[Research Agent] thinking...".dimmed());
        let research_out = self.research.run(backend, &input)?;
        trace.steps.push(AgentStep {
            agent: "ResearchAgent".to_string(),
            input,
            output: research_out.clone(),
        });

        println!("{}", "[Compare Agent] refining...".dimmed());
        let final_out = self.compare.run(backend, &research_out)?;
        trace.steps.push(AgentStep {
            agent: "CompareAgent".to_string(),
            input: research_out,
            output: final_out.clone(),
        });

        trace.final_output = final_out;
        Ok(trace)
    }
}
