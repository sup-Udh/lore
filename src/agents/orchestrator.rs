use anyhow::Result;
use colored::*;
use crate::Model;
use super::{InferenceContext, research::ResearchAgent, compare::CompareAgent};

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

    pub fn run(&mut self, model: &mut dyn Model, ctx: &InferenceContext, input: String) -> Result<String> {
        println!("{}", "[Research Agent] thinking...".dimmed());
        let research_out = self.research.run(model, ctx, input)?;

        println!("{}", "[Compare Agent] refining...".dimmed());
        let final_out = self.compare.run(model, ctx, research_out)?;

        Ok(final_out)
    }
}
