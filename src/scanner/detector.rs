use std::collections::HashSet;
pub fn detect_language(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_string_lossy().to_lowercase();

    let lang = match ext.as_str() {
        "rs" => "Rust",
        "ts" => "TypeScript",
        "tsx" => "React/TypeScript",
        "js" => "JavaScript",
        "jsx" => "React/JavaScript",
        "py" => "Python",
        "go" => "Go",
        "java" => "Java",
        "cpp" | "cc" | "cxx" => "C++",
        "c" => "C",
        "cs" => "C#",
        "php" => "PHP",
        "swift" => "Swift",
        "kt" => "Kotlin",
        _ => return None,
    };

    Some(lang.to_string())
}

// Detect frameworks/dependencies from important files.
//
// In Phase 1 we use simple heuristics.
// Later:
// - dependency graph parsing
// - package lock parsing
// - AST scanning
// can be added.
pub fn detect_frameworks(root: &Path) -> Vec<String> {
    let mut frameworks = HashSet::new();

    let cargo = root.join("Cargo.toml");

    if cargo.exists() {
        frameworks.insert("Cargo".to_string());

        if let Ok(contents) = std::fs::read_to_string(&cargo) {
            if contents.contains("axum") {
                frameworks.insert("Axum".to_string());
            }

            if contents.contains("tokio") {
                frameworks.insert("Tokio".to_string());
            }

            if contents.contains("candle") {
                frameworks.insert("Candle".to_string());
            }

            if contents.contains("llama-cpp") {
                frameworks.insert("llama.cpp".to_string());
            }
        }
    }

    if root.join("package.json").exists() {
        frameworks.insert("Node.js".to_string());
    }

    if root.join("docker-compose.yml").exists() {
        frameworks.insert("Docker".to_string());
    }

    frameworks.into_iter().collect()
}