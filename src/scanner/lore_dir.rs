use anyhow::Result;
use std::path::Path;

use super::project_map::ProjectMap;

// Creating Lore's persistent memory directory.

#[derive(serde::Serialize)]
struct CompactProjectMap<'a> {
    root: &'a str,
    languages: &'a [String],
    frameworks: &'a [String],
    file_count: usize,
    // Keep only a small preview of paths to avoid ballooning token usage later.
    file_preview: Vec<&'a str>,
}

pub fn initialize_lore_directory(
    root: &Path,
    project_map: &ProjectMap,
) -> Result<()> {
    let lore_dir = root.join(".lore");

    // Create .lore/
    std::fs::create_dir_all(&lore_dir)?;

    // Create future-proof subfolders.
    std::fs::create_dir_all(lore_dir.join("memory"))?;
    std::fs::create_dir_all(lore_dir.join("summaries"))?;
    std::fs::create_dir_all(lore_dir.join("sessions"))?;

    // Save project metadata.
    let json = serde_json::to_string_pretty(project_map)?;

    std::fs::write(
        lore_dir.join("project.json"),
        json,
    )?;

    // Save a compact version intended for model prompts / quick previews.
    let compact = CompactProjectMap {
        root: &project_map.root,
        languages: &project_map.languages,
        frameworks: &project_map.frameworks,
        file_count: project_map.files.len(),
        file_preview: project_map
            .files
            .iter()
            .take(200)
            .map(|f| f.path.as_str())
            .collect(),
    };
    let compact_json = serde_json::to_string_pretty(&compact)?;
    std::fs::write(lore_dir.join("project_compact.json"), compact_json)?;

    Ok(())
}