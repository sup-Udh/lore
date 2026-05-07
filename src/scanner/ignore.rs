use std::path::Path;
 
// can be updated later on
 const IGNORED_DIRS: &[&str] = &[
    "target",
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    "coverage",
    "vendor",
];


// Checks whether a path should be ignored.
pub fn should_ignore(path: &Path) -> bool {
    path.components()
        .any(|component| {
            let part = component.as_os_str().to_string_lossy();
            IGNORED_DIRS.contains(&part.as_ref())
        })
}