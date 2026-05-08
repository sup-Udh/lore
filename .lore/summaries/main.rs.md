 In the `src\main.rs` file of a Rust command-line tool designed for interacting with a Local Language Model (LLM) engine, the code is organized into three main modules: `api`, `backends`, and `runtime`. The `api` module defines the CLI interface, including subcommands and arguments, while the `backends` module provides the `LlamaBackend` struct for the LLM engine backend. The `runtime` module contains the core logic for the tool.

The `api` module defines the command-line interface (CLI) for the tool, including subcommands and their associated arguments. The `backends` module provides the `LlamaBackend` struct, which represents the backend for the LLM engine. The `runtime` module contains the core logic for the tool.

The code determines the number of workers to use based on the number of available CPU cores, ensuring at least one worker and adjusting the number to either 2 or 4. This is achieved using the `const CPU_CORES: usize = std::os::hardware_supports_hyper_count();` constant and the `const MIN_WORKERS: usize = 1;` constant.
 Writes a single, cohesive summary with:
- purpose: The code is designed to summarize a repository's files in parallel using multiple worker threads.
- architecture role: The `api` module defines the CLI interface, the `backends` module provides the LLM engine backend, and the `runtime` module contains the core logic.
- important logic: The code determines the number of workers based on available CPU cores, reduces global thread po