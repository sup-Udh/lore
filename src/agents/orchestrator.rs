use anyhow::Result;
use colored::*;
use crate::Model;
use super::{AgentStep, InferenceContext, Trace, research::ResearchAgent, compare::CompareAgent};

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

    pub fn run(&mut self, model: &mut dyn Model, ctx: &InferenceContext, input: String) -> Result<Trace> {
        let mut trace = Trace { steps: Vec::new(), final_output: String::new() };

        println!("{}", "[Research Agent] thinking...".dimmed());
        let research_out = self.research.run(model, ctx, input.clone())?;
        // OPTIMIZATION: Removed unnecessary cloning in hot path — move `input`
        // into the step instead of cloning it a second time.
        trace.steps.push(AgentStep {
            agent: "ResearchAgent".to_string(),
            input,
            output: research_out.clone(),
        });

        println!("{}", "[Compare Agent] refining...".dimmed());
        // research_out moved here — no extra clone needed for agent input
        let final_out = self.compare.run(model, ctx, research_out)?;
        trace.steps.push(AgentStep {
            agent: "CompareAgent".to_string(),
            input: trace.steps[0].output.clone(),
            output: final_out.clone(),
        });

        trace.final_output = final_out;
        Ok(trace)
    }
}
