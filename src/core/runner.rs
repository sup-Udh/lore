use anyhow::Result;
use candle_core::Device;
use hf_hub::api::sync::Api;


pub fn run_model(model: &str, prompt: &str) -> Result<String> {

    println!("Loading the model: {}", model);

    let device = Device::Cpu;

    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2.5-0.5B".to_string());
     // Download model file
    let model_path = repo.get("model.safetensors")?;

    println!("Model downloaded at: {:?}", model_path);

    // For now we stop here (next step = actual inference)
    Ok(format!(
        "[MODEL LOADED] Prompt: {}",
        prompt
    ))



}