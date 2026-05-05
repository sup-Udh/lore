use anyhow::Result;
use crate::Model;
use super::{InferenceContext, generate};

pub struct ResearchAgent;

impl ResearchAgent {
    pub fn new() -> Self { Self }

    pub fn run(&mut self, model: &mut dyn Model, ctx: &InferenceContext, input: String) -> Result<String> {
        let system = "You are a research assistant. \
            Read the user input carefully, extract all key concepts, \
            expand on relevant details, and produce a thorough factual analysis. \
            Be precise and comprehensive.";
        generate(model, ctx, system, &input)
    }
}
