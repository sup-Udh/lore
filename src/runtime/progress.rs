use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum ProgressEvent {
    Phase(&'static str),
    WorkerStarted { worker_id: usize },
    WorkerFileStart { worker_id: usize, path: String },
    WorkerFileDone { worker_id: usize, path: String },
    Persisted { path: String },
    Completed { done: usize, total: usize },
    Error { worker_id: Option<usize>, path: Option<String>, message: String },
}

pub struct ProgressRenderer {
    total: usize,
    done: usize,
    last_print: Instant,
    min_interval: Duration,
}

impl ProgressRenderer {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            done: 0,
            last_print: Instant::now()
                .checked_sub(Duration::from_secs(60))
                .unwrap_or_else(Instant::now),
            min_interval: Duration::from_millis(120),
        }
    }

    pub fn handle(&mut self, ev: ProgressEvent) {
        match &ev {
            ProgressEvent::WorkerFileDone { .. } | ProgressEvent::Persisted { .. } => {
                // Avoid double counting: only count completion once per file, using WorkerFileDone.
            }
            _ => {}
        }

        // Rate-limit noisy events while keeping it "alive".
        let now = Instant::now();
        let should_print = matches!(
            ev,
            ProgressEvent::Phase(_)
                | ProgressEvent::Completed { .. }
                | ProgressEvent::Error { .. }
                | ProgressEvent::WorkerStarted { .. }
        ) || now.duration_since(self.last_print) >= self.min_interval;

        if !should_print {
            return;
        }

        self.last_print = now;

        match ev {
            ProgressEvent::Phase(name) => {
                println!("[LORE] {}...", name);
            }
            ProgressEvent::WorkerStarted { worker_id } => {
                println!("[LORE] Worker-{} online", worker_id);
            }
            ProgressEvent::WorkerFileStart { worker_id, path } => {
                println!("[LORE] Worker-{} summarizing {}", worker_id, path);
            }
            ProgressEvent::WorkerFileDone { worker_id, path } => {
                self.done += 1;
                println!(
                    "[LORE] Worker-{} finished {} ({} / {})",
                    worker_id, path, self.done, self.total
                );
            }
            ProgressEvent::Persisted { path } => {
                println!("[LORE] Persisted summary: {}", path);
            }
            ProgressEvent::Completed { done, total } => {
                println!("[LORE] Completed {}/{} files", done, total);
            }
            ProgressEvent::Error {
                worker_id,
                path,
                message,
            } => match (worker_id, path) {
                (Some(w), Some(p)) => println!("[LORE] Worker-{} error on {}: {}", w, p, message),
                (Some(w), None) => println!("[LORE] Worker-{} error: {}", w, message),
                _ => println!("[LORE] Error: {}", message),
            },
        }
    }
}

