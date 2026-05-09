use anyhow::Result;
// revert back
use crate::backends::llama_cpp::LlamaBackend;
use crate::scanner::project_map::{ProjectFile, ProjectMap};

fn build_repo_outline(project_map: &ProjectMap) -> String {
    // Keep this deterministic + compact (no full JSON, no huge lists).
    // We bias toward signals useful for picking entrypoints/config/runtime files.
    let mut ext_counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for f in &project_map.files {
        *ext_counts.entry(f.extension.as_str()).or_insert(0) += 1;
    }

    let mut top_exts: Vec<(String, usize)> = ext_counts
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    top_exts.sort_by(|a, b| b.1.cmp(&a.1));
    top_exts.truncate(12);

    let exts = top_exts
        .into_iter()
        .map(|(ext, n)| {
            if ext.is_empty() {
                format!("(no_ext): {}", n)
            } else {
                format!("{}.{}", ext, n)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "Root: {}\nFiles: {}\nTop extensions: {}\n",
        project_map.root,
        project_map.files.len(),
        exts
    )
}

// Uses Phi3 to intelligently determine
// which files are most important.
//
// IMPORTANT:
//
// We intentionally compress repository data
// before sending it to Phi3 because:
//
// AI repository systems are heavily constrained
// by context windows.
pub fn select_important_files(
    backend: &mut LlamaBackend,
    project_map: &ProjectMap,
) -> Result<Vec<ProjectFile>> {

    // VERY IMPORTANT:
    //
    // Limit repository preview size aggressively.
    //
    // Large repositories can easily overflow
    // Phi3's context window.
    let file_preview = project_map
        .files
        .iter()
        .take(120)
        .map(|f| f.path.clone())
        .collect::<Vec<_>>()
        .join("\n");

    let outline = build_repo_outline(project_map);

    let prompt = format!(
r#"
You are Lore's repository analysis engine.

Repository outline:
{}

Project languages:
{:?}

Project frameworks:
{:?}

Repository files:
{}

Select the MOST important files.

Prioritize:
- entrypoints
- APIs
- orchestration
- runtime logic
- architecture
- infrastructure
- configuration

Avoid:
- generated files
- binaries
- build artifacts

Return ONLY file paths.
One path per line.
"#,
        outline,
        project_map.languages,
        project_map.frameworks,
        file_preview
    );

    // Keep output short: we just need paths.
    let response = backend.generate_with_limits(&prompt, 220)?;

    let important_paths: Vec<String> = response
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    let selected = project_map
        .files
        .iter()
        .filter(|file| {
            important_paths
                .iter()
                .any(|important| {
                    file.path.contains(important)
                })
        })
        .cloned()
        .collect();

    Ok(selected)
}