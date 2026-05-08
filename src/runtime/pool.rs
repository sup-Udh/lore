use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use crate::backends::llama_cpp::ModelKind;
use crate::runtime::progress::ProgressEvent;
use crate::runtime::worker::{worker_loop, SummaryResult, SummaryTask};

pub struct SummarizationPool {
    task_txs: Vec<Sender<Option<SummaryTask>>>,
    pub result_rx: Receiver<Result<SummaryResult, (String, String)>>,
    progress_rx: Option<Receiver<ProgressEvent>>,
    progress_tx: Sender<ProgressEvent>,
    join_handles: Vec<thread::JoinHandle<()>>,
    rr: usize,
}

impl SummarizationPool {
    pub fn new(model_path: &str, kind: ModelKind, workers: usize) -> Self {
        let (result_tx, result_rx) = mpsc::channel();
        let (progress_tx, progress_rx) = mpsc::channel();

        let mut task_txs = Vec::new();
        let mut join_handles = Vec::new();

        for worker_id in 0..workers {
            let (tx, rx) = mpsc::channel::<Option<SummaryTask>>();
            task_txs.push(tx);

            let model_path = model_path.to_string();
            let result_tx = result_tx.clone();
            let progress_tx = progress_tx.clone();

            let handle = thread::spawn(move || {
                worker_loop(worker_id + 1, model_path, kind, rx, result_tx, progress_tx);
            });
            join_handles.push(handle);
        }

        Self {
            task_txs,
            result_rx,
            progress_rx: Some(progress_rx),
            progress_tx,
            join_handles,
            rr: 0,
        }
    }

    pub fn take_progress_rx(&mut self) -> Receiver<ProgressEvent> {
        self.progress_rx
            .take()
            .expect("progress_rx already taken")
    }

    pub fn progress_sender(&self) -> Sender<ProgressEvent> {
        self.progress_tx.clone()
    }

    pub fn submit(&mut self, task: SummaryTask) {
        let idx = self.rr % self.task_txs.len();
        self.rr = self.rr.wrapping_add(1);
        let _ = self.task_txs[idx].send(Some(task));
    }

    pub fn shutdown(self) {
        for tx in &self.task_txs {
            let _ = tx.send(None);
        }
        for h in self.join_handles {
            let _ = h.join();
        }
    }
}

