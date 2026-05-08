use std::path::Path;


// add other things later to be deleeted v1 currently

const IGNORED_DIRS: &[&str] = &[
    "target",
    "node_modules",
    ".git",
    ".lore",
    "dist",
    "build",
    ".next",
    "coverage",
    "vendor",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    ".cargo",
    ".gradle",
    ".mypy_cache",
    ".pytest_cache",
    ".gitignore",
    "package-lock.json"
];

// Checks whether a path should be ignored.
pub fn should_ignore(path: &Path) -> bool {
    path.components()
        .any(|component| {
            let part = component.as_os_str().to_string_lossy();
            IGNORED_DIRS.contains(&part.as_ref())
        })
}