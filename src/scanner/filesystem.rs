use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;

use super::detector::{detect_frameworks, detect_language};
use super::ignore::should_ignore;
use super::project_map::{ProjectFile, ProjectMap};

// Main project scanner.
//
// Responsibilities:
// - recursively walk filesystem
// - ignore junk folders
// - collect metadata
// - detect languages/frameworks
// - build ProjectMap
pub fn scan_project(root: &Path) -> Result<ProjectMap> {

    let mut files = Vec::new();
    let mut languages = HashSet::new();

    walk_directory(
        root,
        root,
        &mut files,
        &mut languages,
    )?;

    let frameworks = detect_frameworks(root);

    Ok(ProjectMap {
        root: root.display().to_string(),
        languages: languages.into_iter().collect(),
        frameworks,
        files,
    })
}

// Recursive filesystem traversal.
//
// root = original project root
// dir  = current recursive directory
fn walk_directory(
    root: &Path,
    dir: &Path,
    files: &mut Vec<ProjectFile>,
    languages: &mut HashSet<String>,
) -> Result<()> {

    // Skip ignored directories.
    if should_ignore(dir) {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir)? {

        let entry = entry?;
        let path: PathBuf = entry.path();

        if should_ignore(&path) {
            continue;
        }

        if path.is_dir() {

            // Recursive traversal.
            walk_directory(
                root,
                &path,
                files,
                languages,
            )?;

        } else {

            let metadata = std::fs::metadata(&path)?;

            let extension = path
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();

            // Lightweight language detection.
            if let Some(language) = detect_language(&path) {
                languages.insert(language);
            }

            // IMPORTANT:
            // Store RELATIVE paths instead of gigantic absolute Windows paths.
            //
            // This massively reduces token usage later.
            let relative_path = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();

            files.push(ProjectFile {
                path: relative_path,
                extension,
                size: metadata.len(),
            });
        }
    }

    Ok(())
}