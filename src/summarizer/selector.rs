use anyhow::Result;
use crate::backends::Backend;
use crate::scanner::project_map::{ProjectFile, ProjectMap}


pub fn select_important_files(
    backend: &mut dyn Backend,
    project_map: &ProjectMap,
) -> Result<Vec<ProjectFile>> {

    // Compress file list for prompt efficiency.
    let file_preview = project_map
        .files
        .iter()
        .take(200)
        .map(|f| format!("{} ({})", f.path, f.size))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
r#"
You are Lore's repository analysis agent.

Below is metadata about a software project.

Project Root:
{}

Languages:
{:?}

Frameworks:
{:?}

Files:
{}

Your task:
- determine which files are MOST important
- prioritize:
  - architecture
  - runtime logic
  - entrypoints
  - APIs
  - model orchestration
  - routing
  - infrastructure
  - configuration

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
                .any(|path| file.path.contains(path))
        })
        .cloned()
        .collect();

    Ok(selected)
}