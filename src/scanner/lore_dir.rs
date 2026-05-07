use anyhow::Result;
use std::path::Path;

use super::project_map::ProjectMap;

// Creating Lore's persistent memory directory.

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

    Ok(())
}