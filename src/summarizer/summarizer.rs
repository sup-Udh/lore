use anyhow::Result;

use crate::backends::llama_cpp::LlamaBackend;

fn build_file_prompt(path: &str, code: &str) -> String {
    format!(
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
        path, code
    )
}

fn build_chunk_prompt(path: &str, chunk_idx: usize, chunk_total: usize, code: &str) -> String {
    format!(
        r#"
You are summarizing a large repository file in chunks.

File:
{}

Chunk: {}/{}

Tasks:
- summarize what this chunk does
- list key types/functions/constants and their roles
- note any dependencies on other modules
- keep it concise but specific

Code chunk:
{}
"#,
        path,
        chunk_idx + 1,
        chunk_total,
        code
    )
}

fn build_merge_prompt(path: &str, chunk_summaries: &str) -> String {
    format!(
        r#"
You are combining chunk summaries into one cohesive engineering summary.

File:
{}

Chunk summaries:
{}

Write a single, cohesive summary with:
- purpose
- architecture role
- important logic
- key APIs (functions/types) and how they interact

Avoid repetition.
"#,
        path, chunk_summaries
    )
}

fn chunk_by_token_budget(
    backend: &LlamaBackend,
    path: &str,
    contents: &str,
    target_prompt_tokens: usize,
) -> Result<Vec<String>> {
    // Greedy line-based packing into chunks, with exact token counting against
    // the real chat template via backend tokenizer.
    let lines: Vec<&str> = contents.lines().collect();
    if lines.is_empty() {
        return Ok(vec![String::new()]);
    }

    let system = "You are a precise software engineering assistant.";

    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();

    for line in lines {
        let candidate = if cur.is_empty() {
            line.to_string()
        } else {
            format!("{}\n{}", cur, line)
        };

        // Use the one-shot prompt format for budget estimation.
        let user = build_file_prompt(path, &candidate);
        let n = backend.count_tokens_with_system(system, &user)?;

        if n <= target_prompt_tokens {
            cur = candidate;
            continue;
        }

        if !cur.is_empty() {
            chunks.push(cur);
            cur = String::new();
        }

        // If a single line doesn't fit (pathological), hard-truncate.
        let mut hard = line.to_string();
        if hard.len() > 1_024 {
            hard.truncate(1_024);
        }
        loop {
            let user = build_file_prompt(path, &hard);
            let n = backend.count_tokens_with_system(system, &user)?;
            if n <= target_prompt_tokens || hard.len() <= 128 {
                chunks.push(hard);
                break;
            }
            hard.truncate(hard.len().saturating_sub(128));
        }
    }

    if !cur.is_empty() {
        chunks.push(cur);
    }

    Ok(chunks)
}

// Generates detailed engineering summaries
// for important repository files.
pub fn summarize_file(
    backend: &mut LlamaBackend,
    path: &str,
    contents: &str,
) -> Result<String> {
    // Token-safe summarization:
    // - if content fits, do one-shot summary
    // - otherwise chunk + summarize each chunk + merge
    let system = "You are a precise software engineering assistant.";

    // One-shot attempt with a moderate output budget.
    let one_shot_max_new = 280;
    let one_shot_prompt = build_file_prompt(path, contents);
    let one_shot_tokens = backend.count_tokens_with_system(system, &one_shot_prompt)?;
    if (one_shot_tokens as i32) + one_shot_max_new <= backend.n_ctx() {
        return backend.generate_with_system_limits(system, &one_shot_prompt, one_shot_max_new);
    }

    // Chunking budget: shrink output budget to reclaim prompt space.
    let chunk_max_new = 160;
    let safety_margin: i32 = 64;
    let target_prompt_tokens: usize = (backend
        .n_ctx()
        .saturating_sub(chunk_max_new)
        .saturating_sub(safety_margin)) as usize;

    let chunks = chunk_by_token_budget(backend, path, contents, target_prompt_tokens)?;

    if chunks.len() == 1 {
        let prompt = build_file_prompt(path, &chunks[0]);
        return backend.generate_with_system_limits(system, &prompt, one_shot_max_new);
    }

    // Summarize each chunk.
    let mut chunk_summaries = String::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let prompt = build_chunk_prompt(path, i, chunks.len(), chunk);
        let summary = backend.generate_with_system_limits(system, &prompt, chunk_max_new)?;
        chunk_summaries.push_str(&format!(
            "\n--- Chunk {}/{} ---\n{}\n",
            i + 1,
            chunks.len(),
            summary.trim()
        ));
    }

    // Merge step.
    let merge_max_new = 360;
    let mut compact = chunk_summaries;
    loop {
        let merge_prompt = build_merge_prompt(path, &compact);
        let merge_tokens = backend.count_tokens_with_system(system, &merge_prompt)?;
        if (merge_tokens as i32) + merge_max_new <= backend.n_ctx() || compact.len() <= 2_000 {
            return backend.generate_with_system_limits(system, &merge_prompt, merge_max_new);
        }
        // Too many chunks: trim oldest tail and retry.
        compact.truncate(compact.len().saturating_sub(2_000));
    }
}