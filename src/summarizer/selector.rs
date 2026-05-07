use anyhow::Result;

use crate::backends::llama_cpp::LlamaBackend;
use crate::scanner::project_map::{ProjectFile, ProjectMap};

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
        .take(50)
        .map(|f| f.path.clone())
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
r#"
You are Lore's repository analysis engine.

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
        project_map.languages,
        project_map.frameworks,
        file_preview
    );

    let response = backend.generate(&prompt)?;

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