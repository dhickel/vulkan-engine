use std::sync::mpsc;
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A simple bounded thread pool for asset loading.
///
/// Pre-spawns `num_threads` worker threads that pull jobs from a shared
/// channel. When all senders are dropped (including clones), the channel
/// disconnects and workers exit.
pub(crate) struct BoundedThreadPool {
    sender: Option<mpsc::Sender<Job>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl BoundedThreadPool {
    pub(crate) fn new(num_threads: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = std::sync::Arc::new(std::sync::Mutex::new(receiver));

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let receiver = std::sync::Arc::clone(&receiver);
            workers.push(thread::spawn(move || loop {
                let job = {
                    let receiver = receiver.lock().expect("thread pool mutex poisoned");
                    match receiver.recv() {
                        Ok(job) => job,
                        Err(_) => return, // Channel closed, exit
                    }
                };
                job();
            }));
        }

        Self {
            sender: Some(sender),
            workers,
        }
    }

    /// Submit a job to the thread pool.
    pub(crate) fn execute<F>(&self, job: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(sender) = &self.sender {
            let _ = sender.send(Box::new(job));
        }
    }
}

impl Drop for BoundedThreadPool {
    fn drop(&mut self) {
        // Disconnect the queue before joining so idle workers leave recv(). This also ensures
        // renderer-owned worker jobs release Vulkan cache Arcs before backend teardown.
        self.sender.take();
        for worker in self.workers.drain(..) {
            if worker.join().is_err() {
                log::error!("asset worker panicked before thread-pool shutdown");
            }
        }
    }
}
