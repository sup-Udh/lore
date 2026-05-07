use serde::{Serialize, Deserialize};

// Represents one discovered file in the project.
//
// This will later become VERY important for:
// - embeddings
// - retrieval
// - semantic search
// - agent memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFile {
    pub path: String,
    pub extension: String,
    pub size: u64,
}

// Core project metadata structure.
//
// This becomes Lore's internal understanding
// of the repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMap {
    pub root: String,
    pub languages: Vec<String>,
    pub frameworks: Vec<String>,
    pub files: Vec<ProjectFile>,
}