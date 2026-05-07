use anyhow::Result;

use crate::backends::llama_cpp::LlamaBackend;

// Generates detailed engineering summaries
// for important repository files.
pub fn summarize_file(
    backend: &mut LlamaBackend,
    path: &str,
    contents: &str,
) -> Result<String> {

    // IMPORTANT:
    //
    // Keep prompts concise.
    //
    // Long prompts reduce:
    // - generation quality
    // - inference speed
    // - available context for code
    let prompt = format!(
r#"
Explain this repository file.

File:
{}

Tasks:
- explain purpose
- explain role in architecture
- summarize important logic
- describe engineering responsibility

Code:
{}
"#,
        path,
        contents
    );

    let summary = backend.generate(&prompt)?;

    Ok(summary)
}