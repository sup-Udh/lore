use anyhow::Result;

use crate::backends::llama_cpp::LlamaBackend;

const SYSTEM_PROMPT: &str =
    "You are a precise repository compression engine.";

const SMALL_FILE_LINES: usize = 120;
const LARGE_FILE_LINES: usize = 400;

const SMALL_SUMMARY_TOKENS: i32 = 64;
const CHUNK_SUMMARY_TOKENS: i32 = 48;

const SAFETY_MARGIN: i32 = 64;

fn build_summary_prompt(path: &str, compressed_code: &str) -> String {
    format!(
        r#"
Summarize this repository file.

Return ONLY:
- purpose
- key symbols
- dependencies

Keep concise.
Maximum 80 tokens.

File:
{}

Code structure:
{}
"#,
        path,
        compressed_code
    )
}

fn approximate_tokens(text: &str) -> usize {
    // Fast approximation.
    // Good enough for chunking.
    text.len() / 4
}

fn extract_structure(contents: &str) -> String {
    let mut important = Vec::new();

    for line in contents.lines() {
        let trimmed = line.trim();

        // Imports
        if trimmed.starts_with("use ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("#include")
        {
            important.push(trimmed.to_string());
            continue;
        }

        // Rust
        if trimmed.starts_with("pub struct ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("pub trait ")
            || trimmed.starts_with("trait ")
            || trimmed.starts_with("impl ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
        {
            important.push(trimmed.to_string());
            continue;
        }

        // JS/TS
        if trimmed.starts_with("export ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("function ")
            || trimmed.starts_with("async function ")
            || trimmed.contains("=>")
        {
            important.push(trimmed.to_string());
            continue;
        }

        // Comments
        if trimmed.starts_with("//")
            || trimmed.starts_with("///")
            || trimmed.starts_with("/*")
        {
            important.push(trimmed.to_string());
        }
    }

    if important.is_empty() {
        // fallback
        contents
            .lines()
            .take(80)
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        important.join("\n")
    }
}

fn chunk_structure(
    compressed: &str,
    target_tokens: usize,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in compressed.lines() {
        let candidate = if current.is_empty() {
            line.to_string()
        } else {
            format!("{}\n{}", current, line)
        };

        if approximate_tokens(&candidate) <= target_tokens {
            current = candidate;
        } else {
            if !current.is_empty() {
                chunks.push(current);
            }

            current = line.to_string();
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

fn summarize_chunk(
    backend: &mut LlamaBackend,
    path: &str,
    chunk: &str,
) -> Result<String> {
    let prompt = build_summary_prompt(path, chunk);

    backend.generate_with_system_limits(
        SYSTEM_PROMPT,
        &prompt,
        CHUNK_SUMMARY_TOKENS,
    )
}

// Fast repository-oriented summarization.
pub fn summarize_file(
    backend: &mut LlamaBackend,
    path: &str,
    contents: &str,
) -> Result<String> {

    // --------------------------------------------------
    // Tiny files
    // --------------------------------------------------

    let line_count = contents.lines().count();

    if line_count <= SMALL_FILE_LINES {
        let compressed = extract_structure(contents);

        let prompt = build_summary_prompt(path, &compressed);

        return backend.generate_with_system_limits(
            SYSTEM_PROMPT,
            &prompt,
            SMALL_SUMMARY_TOKENS,
        );
    }

    // --------------------------------------------------
    // Large files
    // --------------------------------------------------

    let compressed = extract_structure(contents);

    let target_prompt_tokens = backend
        .n_ctx()
        .saturating_sub(CHUNK_SUMMARY_TOKENS)
        .saturating_sub(SAFETY_MARGIN) as usize;

    let chunks = chunk_structure(
        &compressed,
        target_prompt_tokens,
    );

    // If compression reduced enough,
    // avoid chunk pipeline entirely.
    if chunks.len() == 1 {
        let prompt = build_summary_prompt(path, &chunks[0]);

        return backend.generate_with_system_limits(
            SYSTEM_PROMPT,
            &prompt,
            SMALL_SUMMARY_TOKENS,
        );
    }

    // --------------------------------------------------
    // Chunk summaries
    // --------------------------------------------------

    let mut final_summary = String::new();

    for chunk in chunks {
        let summary = summarize_chunk(
            backend,
            path,
            &chunk,
        )?;

        final_summary.push_str(summary.trim());
        final_summary.push('\n');
    }

    Ok(final_summary)
}