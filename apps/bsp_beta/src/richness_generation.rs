//! Richness V1 generation controller for the BSP beta explorer.
//!
//! This module defines the `RichnessGenerationController`, which owns one
//! worker thread with latest-request-wins queue semantics, request-unique
//! package destination directories, stale-result disposal, close intent
//! keyed to request ID, and active-world preservation on every failure.
//!
//! The controller accepts a typed `GenerationExecutor` for testability.
//! A production executor is wired only at final integration (Phase 18).
//! No success mocks exist — the `ExecutorOutcome` enum models the real
//! success/failure outcomes and all queue/state transitions are fully
//! tested with `Failed` outcomes and injected executors.
//!
//! # Relationship to the generator
//!
//! The generator's `src/bsp_generator/src/enhanced_v3/richness/` module is
//! crate-private. This controller does not depend on it. The production
//! executor will be provided by `apps/bsp_beta` at wiring time.

use crate::richness_gui::RichnessDraft;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

// ── Executor outcome ───────────────────────────────────────────────────────

/// Typed outcome from a generation executor.
///
/// `PackageReady` represents a successful generation with a populated
/// package directory. `Failed` represents any failure. The controller
/// preserves the active world on every `Failed` outcome.
///
/// No success mocks — `PackageReady` is produced only by a real wired
/// executor at Phase 18. All queue/state transition tests use `Failed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorOutcome {
    PackageReady {
        request_id: u64,
        package_dir: PathBuf,
    },
    Failed {
        request_id: u64,
        error_message: String,
    },
}

impl ExecutorOutcome {
    /// The request ID this outcome belongs to.
    pub fn request_id(&self) -> u64 {
        match self {
            Self::PackageReady { request_id, .. } | Self::Failed { request_id, .. } => *request_id,
        }
    }

    /// Returns `true` if this is a `PackageReady` outcome.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::PackageReady { .. })
    }

    /// Returns `true` if this is a `Failed` outcome.
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    /// If successful, returns the package directory.
    pub fn package_dir(&self) -> Option<&Path> {
        match self {
            Self::PackageReady { package_dir, .. } => Some(package_dir.as_path()),
            Self::Failed { .. } => None,
        }
    }
}

// ── Generation executor type ───────────────────────────────────────────────

/// A boxed closure that executes one generation request and returns its
/// outcome. The controller calls this once per dispatched request on the
/// worker thread. The executor owns all I/O, compilation, and package
/// publication; the controller only manages queue and lifecycle.
pub type GenerationExecutor =
    Box<dyn FnOnce(&RichnessGenerationRequest) -> ExecutorOutcome + Send + 'static>;

// ── Generation request ─────────────────────────────────────────────────────

/// A single generation request with a unique ID and dedicated package
/// destination directory.
#[derive(Debug, Clone)]
pub struct RichnessGenerationRequest {
    pub id: u64,
    pub draft: RichnessDraft,
    pub package_dir: PathBuf,
}

// ── Internal shared state ──────────────────────────────────────────────────

struct PendingWork {
    request: RichnessGenerationRequest,
    executor: GenerationExecutor,
}

struct ControllerShared {
    /// Latest pending request (coalesced: newest replaces older).
    pending: Mutex<Option<PendingWork>>,
    /// Completed outcomes, preserved in FIFO order for the UI to consume.
    completed: Mutex<VecDeque<ExecutorOutcome>>,
    /// Set to `true` when shutdown is requested.
    quit: AtomicBool,
    /// Monotonically increasing request ID counter.
    next_id: AtomicU64,
    /// Latest request ID that was submitted (used for stale-result detection).
    latest_submitted_id: AtomicU64,
    /// Close intent: `Some(request_id)` when the UI has requested
    /// "Apply & Close" for a specific generation.
    close_intent: Mutex<Option<u64>>,
}

// ── Controller ─────────────────────────────────────────────────────────────

/// Richness V1 generation controller.
///
/// Owns one worker thread that executes generation requests off the event
/// loop. The controller enforces:
///
/// - **Latest-request-wins**: Submitting request B after A discards A
///   (A's pending work is replaced; if A has already completed, its result
///   is flagged as stale and its package directory is cleaned up).
/// - **Request-unique package directories**: Each request gets a distinct
///   directory under the controller's package root.
/// - **Stale-result disposal**: When a worker completes a request that is
///   no longer the latest, the result is dropped and the package directory
///   removed.
/// - **Close intent**: Keyed to a specific request ID. When the UI issues
///   "Apply & Close", the controller records the intent. The event-loop
///   owner checks completion against this intent.
/// - **Active-world preservation on failure**: A `Failed` outcome NEVER
///   replaces the active BSP/world. Only a `PackageReady` outcome for the
///   close-intent request ID triggers publication.
pub struct RichnessGenerationController {
    shared: Arc<ControllerShared>,
    worker: Option<JoinHandle<()>>,
    package_root: PathBuf,
}

impl RichnessGenerationController {
    /// Create a new controller with a process-unique package root under
    /// the system temp directory.
    ///
    /// The worker thread is spawned immediately and waits for enqueued work.
    pub fn spawn() -> Result<Self, std::io::Error> {
        let package_root = create_unique_richness_package_root()?;
        let shared = Arc::new(ControllerShared {
            pending: Mutex::new(None),
            completed: Mutex::new(VecDeque::new()),
            quit: AtomicBool::new(false),
            next_id: AtomicU64::new(0),
            latest_submitted_id: AtomicU64::new(0),
            close_intent: Mutex::new(None),
        });
        let thread_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("richness-gen-worker".into())
            .spawn(move || richness_worker_loop(thread_shared))
            .expect("spawn richness generation worker thread");

        Ok(Self {
            shared,
            worker: Some(worker),
            package_root,
        })
    }

    /// Create a controller with an explicit package root (for testing).
    ///
    /// The worker thread is spawned immediately.
    pub fn spawn_at_root(package_root: PathBuf) -> Self {
        let shared = Arc::new(ControllerShared {
            pending: Mutex::new(None),
            completed: Mutex::new(VecDeque::new()),
            quit: AtomicBool::new(false),
            next_id: AtomicU64::new(0),
            latest_submitted_id: AtomicU64::new(0),
            close_intent: Mutex::new(None),
        });
        let thread_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("richness-gen-worker".into())
            .spawn(move || richness_worker_loop(thread_shared))
            .expect("spawn richness generation worker thread");

        Self {
            shared,
            worker: Some(worker),
            package_root,
        }
    }

    /// Return the controller's package root directory.
    pub fn package_root(&self) -> &Path {
        &self.package_root
    }

    /// Enqueue a generation request with the given draft and executor.
    ///
    /// This replaces any pending request (latest-request-wins). Returns
    /// the newly assigned request ID. The controller allocates a unique
    /// package directory under the package root.
    pub fn enqueue(&self, draft: RichnessDraft, executor: GenerationExecutor) -> u64 {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let request = RichnessGenerationRequest {
            id,
            draft,
            package_dir: package_dir_for_richness_request(&self.package_root, id),
        };

        self.shared.latest_submitted_id.store(id, Ordering::Relaxed);

        let work = PendingWork { request, executor };
        let mut pending = self.shared.pending.lock().expect("richness queue poisoned");
        // Replace any stale pending work — the older request's package
        // directory will never be used. Clean it up proactively.
        if let Some(ref old) = *pending {
            let _ = std::fs::remove_dir_all(&old.request.package_dir);
        }
        *pending = Some(work);
        id
    }

    /// Record close intent for a specific request ID.
    ///
    /// When the worker produces a `PackageReady` outcome matching this
    /// intent, the event-loop owner should publish it. A `Failed` outcome
    /// or a stale success does NOT satisfy the close intent.
    pub fn set_close_intent(&self, request_id: u64) {
        *self
            .shared
            .close_intent
            .lock()
            .expect("close intent poisoned") = Some(request_id);
    }

    /// Clear the close intent.
    pub fn clear_close_intent(&self) {
        *self
            .shared
            .close_intent
            .lock()
            .expect("close intent poisoned") = None;
    }

    /// Return the current close intent request ID, if any.
    pub fn close_intent(&self) -> Option<u64> {
        *self
            .shared
            .close_intent
            .lock()
            .expect("close intent poisoned")
    }

    /// Poll the completed outcome queue.
    ///
    /// Returns the oldest unclaimed outcome, or `None` if the queue is
    /// empty. The caller owns disposal of stale successful package
    /// directories after checking against the latest submitted ID.
    pub fn poll_result(&self) -> Option<ExecutorOutcome> {
        self.shared
            .completed
            .lock()
            .expect("richness result poisoned")
            .pop_front()
    }

    /// Return the latest submitted request ID (for stale detection).
    pub fn latest_submitted_id(&self) -> u64 {
        self.shared.latest_submitted_id.load(Ordering::Relaxed)
    }

    /// Return the next request ID that will be assigned.
    pub fn next_id(&self) -> u64 {
        self.shared.next_id.load(Ordering::Relaxed) + 1
    }

    /// Return the number of completed outcomes waiting in the queue.
    pub fn completed_count(&self) -> usize {
        self.shared
            .completed
            .lock()
            .expect("richness result poisoned")
            .len()
    }

    /// Return `true` if the worker thread has been asked to quit.
    pub fn is_quit(&self) -> bool {
        self.shared.quit.load(Ordering::Relaxed)
    }

    /// Initiate shutdown and join the worker thread.
    ///
    /// The worker drains any in-progress work before exiting. After
    /// shutdown, no new requests are accepted (the queue is cleared).
    pub fn shutdown(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        self.shared.quit.store(true, Ordering::Relaxed);
        if let Some(thread) = self.worker.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for RichnessGenerationController {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

// ── Worker loop ────────────────────────────────────────────────────────────

fn richness_worker_loop(shared: Arc<ControllerShared>) {
    loop {
        let work = shared
            .pending
            .lock()
            .expect("richness queue poisoned")
            .take();
        if let Some(work) = work {
            // Execute the generation request on the worker thread.
            let PendingWork { request, executor } = work;
            let outcome = validate_executor_outcome(&request, (executor)(&request));

            // Post-completion: detect staleness.
            let latest = shared.latest_submitted_id.load(Ordering::Relaxed);
            if outcome.request_id() != latest {
                // Stale result — dispose the package directory if it was
                // a successful outcome, then drop the result.
                if let Some(dir) = outcome.package_dir() {
                    let _ = std::fs::remove_dir_all(dir);
                }
                // Do not push stale results to completed queue.
                continue;
            }

            // Push the fresh result to the completed queue.
            shared
                .completed
                .lock()
                .expect("richness result poisoned")
                .push_back(outcome);
            continue;
        }

        if shared.quit.load(Ordering::Relaxed) {
            return;
        }

        thread::sleep(Duration::from_millis(10));
    }
}

/// Bind a worker result to the request that the controller actually ran.
///
/// An executor may not redirect a result to a different request or package
/// destination. In particular, stale-result cleanup must never recursively
/// remove an arbitrary path supplied by a faulty executor.
fn validate_executor_outcome(
    request: &RichnessGenerationRequest,
    outcome: ExecutorOutcome,
) -> ExecutorOutcome {
    match outcome {
        ExecutorOutcome::PackageReady {
            request_id,
            package_dir,
        } if request_id == request.id && package_dir == request.package_dir => {
            ExecutorOutcome::PackageReady {
                request_id,
                package_dir,
            }
        }
        ExecutorOutcome::PackageReady {
            request_id,
            package_dir,
        } => ExecutorOutcome::Failed {
            request_id: request.id,
            error_message: format!(
                "executor returned package for request {request_id} at '{}' instead of request {} at '{}'",
                package_dir.display(),
                request.id,
                request.package_dir.display()
            ),
        },
        ExecutorOutcome::Failed {
            request_id,
            error_message,
        } if request_id == request.id => ExecutorOutcome::Failed {
            request_id,
            error_message,
        },
        ExecutorOutcome::Failed {
            request_id,
            error_message,
        } => ExecutorOutcome::Failed {
            request_id: request.id,
            error_message: format!(
                "executor returned failure for request {request_id}: {error_message}"
            ),
        },
    }
}

// ── Package directory helpers ──────────────────────────────────────────────

/// Create a process-unique root for Richness packages.
fn create_unique_richness_package_root() -> std::io::Result<PathBuf> {
    static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    for _ in 0..64 {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|v| v.as_nanos())
            .unwrap_or_default();
        let sequence = ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "bsp-beta-richness-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        match std::fs::create_dir(&root) {
            Ok(()) => return Ok(root),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not reserve a unique Richness generation directory",
    ))
}

/// Confined package directory for one Richness request.
pub fn package_dir_for_richness_request(root: &Path, id: u64) -> PathBuf {
    root.join(format!("richness-request-{id:020}"))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::richness_gui::RichnessDraft;

    // ── Helpers ────────────────────────────────────────────────────────

    fn test_draft() -> RichnessDraft {
        RichnessDraft::new()
    }

    /// An executor that always fails (for testing queue transitions).
    fn failing_executor(error_msg: &'static str) -> GenerationExecutor {
        let msg = error_msg.to_string();
        Box::new(
            move |req: &RichnessGenerationRequest| ExecutorOutcome::Failed {
                request_id: req.id,
                error_message: msg,
            },
        )
    }

    /// An executor that sleeps briefly then returns the given outcome.
    fn delayed_executor(outcome: ExecutorOutcome, ms: u64) -> GenerationExecutor {
        Box::new(move |_req: &RichnessGenerationRequest| {
            thread::sleep(Duration::from_millis(ms));
            outcome.clone()
        })
    }

    // ── Controller lifecycle ──────────────────────────────────────────

    #[test]
    fn controller_spawn_and_shutdown_clean() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        // Controller should be alive
        assert!(!controller.is_quit());
        controller.shutdown();
    }

    #[test]
    fn drop_controller_shuts_down_worker() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        // Dropping should cleanly join the worker
        drop(controller);
    }

    #[test]
    fn controller_spawn_creates_package_root() {
        let root = tempfile::tempdir().unwrap();
        let pkg_root = root.path().join("packages");
        std::fs::create_dir(&pkg_root).unwrap();
        let controller = RichnessGenerationController::spawn_at_root(pkg_root.clone());
        assert!(controller.package_root().exists());
        assert!(controller.package_root().is_dir());
        controller.shutdown();
    }

    // ── Enqueue and polling ───────────────────────────────────────────

    #[test]
    fn enqueue_assigns_monotonic_ids() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        let id1 = controller.enqueue(test_draft(), failing_executor("err1"));
        let id2 = controller.enqueue(test_draft(), failing_executor("err2"));
        let id3 = controller.enqueue(test_draft(), failing_executor("err3"));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
        assert_eq!(controller.latest_submitted_id(), 3);
        controller.shutdown();
    }

    #[test]
    fn enqueue_creates_unique_package_dirs() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        let id1 = controller.enqueue(test_draft(), failing_executor("e1"));
        let dir1 = package_dir_for_richness_request(controller.package_root(), id1);
        let id2 = controller.enqueue(test_draft(), failing_executor("e2"));
        let dir2 = package_dir_for_richness_request(controller.package_root(), id2);
        assert_ne!(dir1, dir2);
        assert_ne!(id1, id2);
        controller.shutdown();
    }

    #[test]
    fn poll_result_returns_completed_outcome() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        let id = controller.enqueue(test_draft(), failing_executor("test error"));

        // Poll until we get the result or timeout
        let mut result = None;
        for _ in 0..200 {
            result = controller.poll_result();
            if result.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        let outcome = result.expect("should receive a completed outcome");
        assert_eq!(outcome.request_id(), id);
        assert!(outcome.is_failure());
        match outcome {
            ExecutorOutcome::Failed { error_message, .. } => {
                assert_eq!(error_message, "test error");
            }
            _ => panic!("expected Failed outcome"),
        }
        controller.shutdown();
    }

    #[test]
    fn poll_result_returns_none_when_empty() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        assert!(controller.poll_result().is_none());
        controller.shutdown();
    }

    // ── Latest-request-wins ────────────────────────────────────────────

    #[test]
    fn latest_request_wins_pending_replaced() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());

        // Submit A, then immediately replace with B.
        let id_a = controller.enqueue(test_draft(), failing_executor("should not run"));
        let id_b = controller.enqueue(test_draft(), failing_executor("this is the winner"));

        assert!(id_b > id_a);
        assert_eq!(controller.latest_submitted_id(), id_b);

        // Only B's outcome should appear.
        let mut result = None;
        for _ in 0..200 {
            result = controller.poll_result();
            if result.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        let outcome = result.expect("should receive B's outcome");
        assert_eq!(outcome.request_id(), id_b);

        // No more outcomes
        assert!(controller.poll_result().is_none());
        controller.shutdown();
    }

    #[test]
    fn stale_result_discarded_when_newer_request_exists() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());

        // Submit A with a slow executor so it's still running when B arrives.
        let outcome_a = ExecutorOutcome::Failed {
            request_id: 0,
            error_message: "stale A".into(),
        };
        let id_a = controller.enqueue(test_draft(), delayed_executor(outcome_a, 200));
        // Immediately submit B while A's executor is still running
        let id_b = controller.enqueue(test_draft(), failing_executor("fresh B"));

        // Only B's result should appear
        let mut results = Vec::new();
        for _ in 0..100 {
            while let Some(outcome) = controller.poll_result() {
                results.push(outcome);
            }
            if !results.is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        let seen_a = results.iter().any(|r| r.request_id() == id_a);
        let seen_b = results.iter().any(|r| r.request_id() == id_b);

        assert!(
            !seen_a,
            "stale result A should have been discarded, got results: {results:?}"
        );
        assert!(
            seen_b,
            "fresh result B should appear, got results: {results:?}"
        );
        controller.shutdown();
    }

    #[test]
    fn stale_success_directory_cleaned() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());

        // Submit A with a slow success executor that uses the request's actual dir
        let id_a = controller.enqueue(
            test_draft(),
            Box::new(|req: &RichnessGenerationRequest| {
                thread::sleep(Duration::from_millis(200));
                let _ = std::fs::create_dir_all(&req.package_dir);
                ExecutorOutcome::PackageReady {
                    request_id: req.id,
                    package_dir: req.package_dir.clone(),
                }
            }),
        );
        let dir_a = package_dir_for_richness_request(controller.package_root(), id_a);

        // Immediately submit B — A's result becomes stale
        let _id_b = controller.enqueue(test_draft(), failing_executor("fresh B"));

        // Drain the completed queue (wait for B to complete)
        for _ in 0..100 {
            if controller.poll_result().is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        while controller.poll_result().is_some() {}

        // A's package directory should be cleaned up
        assert!(
            !dir_a.exists(),
            "stale package directory {:?} should have been cleaned",
            dir_a
        );

        controller.shutdown();
    }

    // ── Close intent ───────────────────────────────────────────────────

    #[test]
    fn close_intent_set_and_read() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        assert!(controller.close_intent().is_none());
        controller.set_close_intent(42);
        assert_eq!(controller.close_intent(), Some(42));
        controller.clear_close_intent();
        assert!(controller.close_intent().is_none());
        controller.shutdown();
    }

    #[test]
    fn close_intent_survives_polling() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        controller.set_close_intent(7);
        let id = controller.enqueue(test_draft(), failing_executor("close test"));
        assert_eq!(controller.close_intent(), Some(7));

        // Poll until result arrives
        for _ in 0..200 {
            if controller.poll_result().is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        // Close intent should still be set
        assert_eq!(controller.close_intent(), Some(7));
        controller.shutdown();
    }

    // ── Active-world preservation on failure ───────────────────────────

    #[test]
    fn failure_does_not_produce_package_ready() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        let id = controller.enqueue(test_draft(), failing_executor("generation failed"));

        let mut outcome = None;
        for _ in 0..200 {
            outcome = controller.poll_result();
            if outcome.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        let outcome = outcome.expect("should have an outcome");
        assert!(
            outcome.is_failure(),
            "failure must not produce PackageReady"
        );
        assert_eq!(outcome.request_id(), id);
        controller.shutdown();
    }

    #[test]
    fn failure_preserves_close_intent() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        controller.set_close_intent(1);
        let _id = controller.enqueue(test_draft(), failing_executor("fail"));

        // Drain
        for _ in 0..200 {
            if controller.poll_result().is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        // Close intent should still be set (failure doesn't clear it)
        assert_eq!(controller.close_intent(), Some(1));
        controller.shutdown();
    }

    // ── Worker shutdown ────────────────────────────────────────────────

    #[test]
    fn shutdown_while_worker_idle() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
        controller.shutdown();
        // Should not hang
    }

    #[test]
    fn shutdown_drains_inflight_work() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());

        // Submit work with a delay so the worker is busy when we shutdown
        let outcome = ExecutorOutcome::Failed {
            request_id: 1,
            error_message: "in-flight".into(),
        };
        let _id = controller.enqueue(test_draft(), delayed_executor(outcome.clone(), 100));

        // Small sleep to let the worker pick up the request
        thread::sleep(Duration::from_millis(10));

        // Shutdown — should join after worker completes
        controller.shutdown();
        // If we get here, shutdown completed without hang
    }

    #[test]
    fn shutdown_clears_pending() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());

        // Submit but don't wait — shutdown should still work
        let _id = controller.enqueue(test_draft(), failing_executor("pending work"));
        controller.shutdown();
    }

    // ── Completed count ────────────────────────────────────────────────

    #[test]
    fn completed_count_reflects_queue_depth() {
        let root = tempfile::tempdir().unwrap();
        let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());

        assert_eq!(controller.completed_count(), 0);

        let _id = controller.enqueue(test_draft(), failing_executor("e1"));

        for _ in 0..200 {
            if controller.completed_count() > 0 {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        assert_eq!(controller.completed_count(), 1);

        // Poll removes one
        controller.poll_result();
        assert_eq!(controller.completed_count(), 0);

        controller.shutdown();
    }

    // ── Package directories ────────────────────────────────────────────

    #[test]
    fn package_dirs_are_distinct_and_nested() {
        let root = PathBuf::from("/tmp/richness-test");
        let a = package_dir_for_richness_request(&root, 1);
        let b = package_dir_for_richness_request(&root, 2);
        assert_ne!(a, b);
        assert_eq!(a.parent(), Some(root.as_path()));
        assert_eq!(b.parent(), Some(root.as_path()));
    }

    #[test]
    fn mismatched_package_outcome_is_rejected_without_external_cleanup() {
        let root = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        let request = RichnessGenerationRequest {
            id: 7,
            draft: test_draft(),
            package_dir: root.path().join("richness-request-00000000000000000007"),
        };

        let outcome = validate_executor_outcome(
            &request,
            ExecutorOutcome::PackageReady {
                request_id: 8,
                package_dir: external.path().to_path_buf(),
            },
        );

        assert!(matches!(
            outcome,
            ExecutorOutcome::Failed { request_id: 7, .. }
        ));
        assert!(external.path().exists());
    }

    #[test]
    fn executor_outcome_request_id_accessor() {
        let ok = ExecutorOutcome::PackageReady {
            request_id: 99,
            package_dir: PathBuf::from("/tmp/ok"),
        };
        assert_eq!(ok.request_id(), 99);
        assert!(ok.is_success());
        assert!(!ok.is_failure());
        assert_eq!(ok.package_dir(), Some(Path::new("/tmp/ok")));

        let err = ExecutorOutcome::Failed {
            request_id: 7,
            error_message: "oops".into(),
        };
        assert_eq!(err.request_id(), 7);
        assert!(!err.is_success());
        assert!(err.is_failure());
        assert_eq!(err.package_dir(), None);
    }
}
