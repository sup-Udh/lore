use serde:: {Serialize, Deserialize};

pub struct ProjectFile {
    pub path: String,
    pub extension: String,
    pub size: u64,
}


pub struct ProjectMap {
    pub root: string, 
    pub languages: Vec<String>,
    pub frameworks: Vec<String>,
    pub files: Vec<ProjectFile>,
}