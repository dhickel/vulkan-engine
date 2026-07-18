//! File watcher for hot-reloading assets at runtime.
//!
//! Uses the `notify` crate to monitor asset directories. When a file changes,
//! the watcher invalidates the corresponding entry in the `AssetRegistry`
//! so the next access triggers a fresh load.
//!
//! Future-facing hot-reload feature; dead code allowed.
#![allow(dead_code)]

use crate::data::asset_registry::AssetRegistry;
use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

/// Watches asset directories and invalidates registry entries on change.
pub struct FileWatcher {
    _watcher: notify::INotifyWatcher,
    rx: Receiver<notify::Result<Event>>,
    registry: Arc<Mutex<AssetRegistry>>,
    watched_dirs: Vec<PathBuf>,
}

impl FileWatcher {
    /// Create a new file watcher monitoring the given directories.
    /// Changes invalidate entries in the provided registry.
    pub fn new(dirs: Vec<PathBuf>, registry: Arc<Mutex<AssetRegistry>>) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel();

        let mut watcher = notify::INotifyWatcher::new(tx, notify::Config::default())
            .map_err(|e| format!("failed to create file watcher: {e}"))?;

        for dir in &dirs {
            watcher
                .watch(dir, RecursiveMode::Recursive)
                .map_err(|e| format!("failed to watch {}: {e}", dir.display()))?;
        }

        Ok(Self {
            _watcher: watcher,
            rx,
            registry,
            watched_dirs: dirs,
        })
    }

    /// Check for file change events without blocking.
    /// Call this once per frame. Returns the number of changed files processed.
    pub fn poll(&self) -> usize {
        let mut count = 0;
        while let Ok(event) = self.rx.try_recv() {
            if let Ok(event) = event {
                if self.is_modify_event(&event) {
                    for path in &event.paths {
                        if let Ok(mut reg) = self.registry.lock() {
                            reg.invalidate_path(path);
                            log::info!("hot-reload: invalidated {}", path.display());
                        }
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn is_modify_event(&self, event: &Event) -> bool {
        matches!(
            event.kind,
            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
        )
    }
}
