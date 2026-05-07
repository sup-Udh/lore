use serde::{Serialize, Deserialize};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFile {
    pub path: String,
    pub extension: String,
    pub size: u64,
}

// main functioning of how things to be kept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMap {
    pub root: String,
    pub languages: Vec<String>,
    pub frameworks: Vec<String>,
    pub files: Vec<ProjectFile>,
}