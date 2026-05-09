use anyhow::Result;
use std::path::Path;


// .lore/summaries/ -> file directory
pub fn write_summary(
    root: &Path,
    file_name: &str,
    summary: &str,
) -> Result<()> {

    let summary_dir = root
        .join(".lore")
        .join("summaries");

    std::fs::create_dir_all(&summary_dir)?;

    let output_file = summary_dir.join(file_name);

    std::fs::write(&output_file, summary)?;

    Ok(())
}