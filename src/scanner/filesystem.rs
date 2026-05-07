use std::collections::HashSet;
use super::ignore::should_ignore;
use super::project_map::{ProjectFile, ProjectMap};

// Main recursive project scanner.
//
// Responsibilities:
// - recursively walk filesystem
// - ignore junk directories
// - detect languages
// - collect metadata
// - build ProjectMap
pub fn scan_project(root: &Path) -> Result<ProjectMap> {
    let mut files = Vec::new();
    let mut languages = HashSet::new();

    walk_directory(root, &mut files, &mut languages)?;

    let frameworks = detect_frameworks(root);

    Ok(ProjectMap {
        root: root.display().to_string(),
        languages: languages.into_iter().collect(),
        frameworks,
        files,
    })
}

// Recursive filesystem traversal.
fn walk_directory(
    dir: &Path,
    files: &mut Vec<ProjectFile>,
    languages: &mut HashSet<String>,
) -> Result<()> {
    // Ignore unwanted folders.
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
            walk_directory(&path, files, languages)?;
        } else {
            let metadata = std::fs::metadata(&path)?;

            let extension = path
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default();

            // Detect language.
            if let Some(language) = detect_language(&path) {
                languages.insert(language);
            }

            files.push(ProjectFile {
                path: path.display().to_string(),
                extension,
                size: metadata.len(),
            });
        }
    }

    Ok(())
}