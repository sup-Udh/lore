use anyhow::Result;

use crate::backends::llama_cpp::LlamaBackend;

// Generates detailed engineering summaries
// for important repository files.
pub fn summarize_file(
    backend: &mut LlamaBackend,
    path: &str,
    contents: &str,
) -> Result<String> {

    let prompt = format!(
r#"
You are Lore's software analysis engine.

Your task:
- explain the purpose of this file
- explain its role in the architecture
- summarize important logic
- explain engineering responsibilities
- describe relationships to the repository

Keep explanations concise but technically useful.

File:
{}

Code:
{}
"#,
        path,
        contents
    );

    let summary = backend.generate(&prompt)?;

    Ok(summary)
}