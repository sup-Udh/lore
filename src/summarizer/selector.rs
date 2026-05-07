use anyhow::Result;

use crate::backends::llama_cpp::LlamaBackend;
use crate::scanner::project_map::{ProjectFile, ProjectMap};

// Uses Phi3 to intelligently select the most
// important repository files.
//
// Phi3 reads the generated project map FIRST,
// then determines which files matter most.
pub fn select_important_files(
    backend: &mut LlamaBackend,
    project_map: &ProjectMap,
) -> Result<Vec<ProjectFile>> {

    let file_preview = project_map
        .files
        .iter()
        .take(200)
        .map(|f| format!("{} ({})", f.path, f.size))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
r#"
You are Lore's repository analysis engine.

Project Root:
{}

Languages:
{:?}

Frameworks:
{:?}

Files:
{}

Your task:
- determine the MOST important repository files
- prioritize:
  - entrypoints
  - APIs
  - orchestration
  - infrastructure
  - model logic
  - architecture
  - routing
  - runtime systems

Avoid:
- generated files
- binaries
- build artifacts
- temporary files

Return ONLY important file paths.
One path per line.
"#,
        project_map.root,
        project_map.languages,
        project_map.frameworks,
        file_preview
    );

    let response = backend.generate(&prompt)?;

    let important_paths: Vec<String> = response
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let selected = project_map
        .files
        .iter()
        .filter(|file| {
            important_paths
                .iter()
                .any(|p| file.path.contains(p))
        })
        .cloned()
        .collect();

    Ok(selected)
}