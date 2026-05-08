use std::sync::mpsc::{Receiver, Sender};

use crate::backends::llama_cpp::{LlamaBackend, ModelKind};
use crate::runtime::progress::ProgressEvent;
use crate::summarizer::summarizer::summarize_file;

#[derive(Debug, Clone)]
pub struct SummaryTask {
    pub path: String,
    pub contents: String,
    pub output_name: String,
}

#[derive(Debug, Clone)]
pub struct SummaryResult {
    pub path: String,
    pub output_name: String,
    pub summary: String,
}

pub fn worker_loop(
    worker_id: usize,
    model_path: String,
    kind: ModelKind,
    rx: Receiver<Option<SummaryTask>>,
    result_tx: Sender<Result<SummaryResult, (String, String)>>,
    progress_tx: Sender<ProgressEvent>,
) {
    let _ = progress_tx.send(ProgressEvent::WorkerStarted { worker_id });

    let mut backend = match LlamaBackend::new(&model_path, kind) {
        Ok(b) => b,
        Err(e) => {
            let _ = progress_tx.send(ProgressEvent::Error {
                worker_id: Some(worker_id),
                path: None,
                message: format!("backend init failed: {e}"),
            });
            return;
        }
    };

    while let Ok(msg) = rx.recv() {
        let Some(task) = msg else { break };

        let _ = progress_tx.send(ProgressEvent::WorkerFileStart {
            worker_id,
            path: task.path.clone(),
        });

        match summarize_file(&mut backend, &task.path, &task.contents) {
            Ok(summary) => {
                let _ = progress_tx.send(ProgressEvent::WorkerFileDone {
                    worker_id,
                    path: task.path.clone(),
                });
                let _ = result_tx.send(Ok(SummaryResult {
                    path: task.path,
                    output_name: task.output_name,
                    summary,
                }));
            }
            Err(e) => {
                let msg = format!("{e}");
                let _ = progress_tx.send(ProgressEvent::Error {
                    worker_id: Some(worker_id),
                    path: Some(task.path.clone()),
                    message: msg.clone(),
                });
                let _ = result_tx.send(Err((task.path, msg)));
            }
        }
    }
}

