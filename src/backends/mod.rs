// BACKEND ABSTRACTION
// New module that hosts non-Candle inference backends.
// Currently houses the llama.cpp-backed Qwen path.
// KEEP PHI3/MISTRAL ON CANDLE — they don't go through this module.
pub mod llama_cpp;
