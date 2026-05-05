use anyhow::Result;
use crate::Model;
use super::{InferenceContext, generate};

pub struct CompareAgent;

impl CompareAgent {
    pub fn new() -> Self { Self }

    pub fn run(&mut self, model: &mut dyn Model, ctx: &InferenceContext, input: String) -> Result<String> {
        let system = "You are an analytical assistant. \
            You will receive research notes. \
            Your job is to review them, remove redundancy, \
            and produce a clear, concise, well-structured final answer.";
        generate(model, ctx, system, &input)
    }
}
