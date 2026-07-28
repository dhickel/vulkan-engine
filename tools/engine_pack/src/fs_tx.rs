//! Filesystem transaction primitives for fail-closed tooling.
//!
//! Provides staging, safe traversal, atomic rename, rollback journal, and
//! symlink-free containment checks for `engine_pack` commands.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Component, Path, PathBuf};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum FsTxError {
    Io {
        path: PathBuf,
        message: String,
    },
    StagingFailed {
        staging_path: PathBuf,
        message: String,
    },
    PublicationFailed {
        target: PathBuf,
        staging: PathBuf,
        message: String,
    },
    RollbackFailed {
        original: PathBuf,
        staging: PathBuf,
        message: String,
    },
    RecoveryRequired {
        journal_path: PathBuf,
        failed_operations: Vec<String>,
        message: String,
    },
    ExistingTarget(PathBuf),
    SymlinkRejected {
        path: PathBuf,
        reason: String,
    },
    RootEscape {
        candidate: PathBuf,
        root: PathBuf,
    },
    InvalidEntryPath(String),
    DuplicateDestination(PathBuf),
    CollisionDetected {
        path: PathBuf,
        existing: PathBuf,
    },
    MissingInput(PathBuf),
    CrossRootCanonical {
        candidate: PathBuf,
        resolved: PathBuf,
        root: PathBuf,
    },
    UnsupportedPlatform {
        operation: String,
        reason: String,
    },
    ValidationBeforePublish {
        staging: PathBuf,
        diagnostics: Vec<String>,
    },
    StagingArtifactInvariant {
        staging: PathBuf,
        message: String,
    },
    PreExistingDestination {
        target: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for FsTxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "I/O error at '{}': {message}", path.display()),
            Self::StagingFailed {
                staging_path,
                message,
            } => {
                write!(
                    f,
                    "staging failed at '{}': {message}",
                    staging_path.display()
                )
            }
            Self::PublicationFailed {
                target,
                staging,
                message,
            } => write!(
                f,
                "publication failed: target='{}' staging='{}': {message}",
                target.display(),
                staging.display()
            ),
            Self::RollbackFailed {
                original,
                staging,
                message,
            } => write!(
                f,
                "rollback failed: original='{}' staging='{}': {message}",
                original.display(),
                staging.display()
            ),
            Self::RecoveryRequired {
                journal_path,
                failed_operations,
                message,
            } => {
                write!(
                    f,
                    "recovery required: journal='{}' failed_ops=[{}]: {message}",
                    journal_path.display(),
                    failed_operations.join(", ")
                )
            }
            Self::ExistingTarget(path) => {
                write!(f, "target already exists: '{}'", path.display())
            }
            Self::SymlinkRejected { path, reason } => {
                write!(f, "symlink rejected at '{}': {reason}", path.display())
            }
            Self::RootEscape { candidate, root } => {
                write!(
                    f,
                    "path '{}' escapes root '{}'",
                    candidate.display(),
                    root.display()
                )
            }
            Self::InvalidEntryPath(msg) => write!(f, "invalid entry path: {msg}"),
            Self::DuplicateDestination(path) => {
                write!(f, "duplicate destination: '{}'", path.display())
            }
            Self::CollisionDetected { path, existing } => {
                write!(
                    f,
                    "collision: '{}' already staged at '{}'",
                    path.display(),
                    existing.display()
                )
            }
            Self::MissingInput(path) => {
                write!(f, "missing input file: '{}'", path.display())
            }
            Self::CrossRootCanonical {
                candidate,
                resolved,
                root,
            } => write!(
                f,
                "canonical path '{}' resolves outside root '{}' (candidate: '{}')",
                resolved.display(),
                root.display(),
                candidate.display()
            ),
            Self::UnsupportedPlatform { operation, reason } => {
                write!(f, "unsupported platform for '{operation}': {reason}")
            }
            Self::ValidationBeforePublish {
                staging,
                diagnostics,
            } => {
                write!(
                    f,
                    "validation of staging directory '{}' failed: [{}]",
                    staging.display(),
                    diagnostics.join(", ")
                )
            }
            Self::StagingArtifactInvariant { staging, message } => {
                write!(
                    f,
                    "staging artifact invariant violation in '{}': {message}",
                    staging.display()
                )
            }
            Self::PreExistingDestination { target, message } => {
                write!(
                    f,
                    "pre-existing destination '{}': {message}",
                    target.display()
                )
            }
        }
    }
}

impl std::error::Error for FsTxError {}

// ---------------------------------------------------------------------------
// Plan entry
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntryType {
    File,
    Directory,
}

/// A single entry in a publication plan.
#[derive(Clone, Debug)]
pub struct PlanEntry {
    /// Absolute source path on disk.
    pub source: PathBuf,
    /// Project-relative logical destination (canonical, `/`-separated).
    pub destination: PathBuf,
    /// Whether this is a file or a directory entry.
    pub entry_type: EntryType,
    /// Human label for the operation.
    pub label: String,
}

// ---------------------------------------------------------------------------
// Rollback journal
// ---------------------------------------------------------------------------

/// Records prior state so a failed multi-step publication can be undone.
#[derive(Clone, Debug)]
struct JournalEntry {
    /// The target path that was modified.
    target: PathBuf,
    /// Path to a backup copy of the original file before modification.
    backup: Option<PathBuf>,
    /// Whether the target existed before this step.
    existed_before: bool,
}

/// A rollback journal that records the prior state of files as they are mutated.
#[derive(Clone, Debug, Default)]
pub struct RollbackJournal {
    entries: Vec<JournalEntry>,
}

impl RollbackJournal {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a file at `target` will be overwritten; back it up first.
    pub fn record_backup(&mut self, target: &Path) -> Result<(), FsTxError> {
        let metadata = match fs::symlink_metadata(target) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(FsTxError::SymlinkRejected {
                    path: target.to_path_buf(),
                    reason: "rollback backup target is a symlink".to_string(),
                });
            }
            Ok(metadata) => Some(metadata),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
            Err(err) => {
                return Err(FsTxError::Io {
                    path: target.to_path_buf(),
                    message: format!("inspect backup target failed: {err}"),
                });
            }
        };
        let existed_before = metadata.is_some();
        let backup = if let Some(metadata) = metadata {
            if !metadata.is_file() {
                return Err(FsTxError::InvalidEntryPath(format!(
                    "backup target is not a regular file: '{}'",
                    target.display()
                )));
            }
            let backup_path = temp_backup_path(target)?;
            fs::copy(target, &backup_path).map_err(|err| FsTxError::Io {
                path: target.to_path_buf(),
                message: format!("backup copy failed: {err}"),
            })?;
            Some(backup_path)
        } else {
            None
        };
        self.entries.push(JournalEntry {
            target: target.to_path_buf(),
            backup,
            existed_before,
        });
        Ok(())
    }

    /// Record that a new file will be created at `target` (not previously existing).
    pub fn record_creation(&mut self, target: &Path) {
        self.entries.push(JournalEntry {
            target: target.to_path_buf(),
            backup: None,
            existed_before: false,
        });
    }

    /// Mark the transaction successful and remove backup files.
    pub fn commit(self) -> Result<(), FsTxError> {
        let mut failed = Vec::new();
        for entry in self.entries {
            if let Some(backup) = entry.backup {
                if fs::symlink_metadata(&backup).is_ok() {
                    if let Err(err) = fs::remove_file(&backup) {
                        failed.push(format!("{}: remove backup failed: {err}", backup.display()));
                    }
                }
            }
        }
        if failed.is_empty() {
            Ok(())
        } else {
            Err(FsTxError::RecoveryRequired {
                journal_path: PathBuf::from("<in-memory journal>"),
                failed_operations: failed,
                message: "publication succeeded but backup cleanup failed".to_string(),
            })
        }
    }

    /// Attempt to roll back all recorded entries in reverse order.
    ///
    /// Returns `Ok(())` if all entries were rolled back successfully.
    /// Returns `Err(FsTxError::RecoveryRequired)` if some rollback steps failed.
    pub fn rollback(self) -> Result<(), FsTxError> {
        let mut failed = Vec::new();
        for entry in self.entries.into_iter().rev() {
            if let Err(err) = rollback_entry(&entry) {
                failed.push(format!("{}: {err}", entry.target.display()));
            }
        }
        if failed.is_empty() {
            Ok(())
        } else {
            Err(FsTxError::RecoveryRequired {
                journal_path: PathBuf::from("<in-memory journal>"),
                failed_operations: failed,
                message: "rollback incomplete; manual recovery may be needed".to_string(),
            })
        }
    }
}

fn rollback_entry(entry: &JournalEntry) -> Result<(), FsTxError> {
    if let Some(backup) = &entry.backup {
        // Restore original from backup
        fs::rename(backup, &entry.target).map_err(|err| FsTxError::RollbackFailed {
            original: backup.clone(),
            staging: entry.target.clone(),
            message: format!("restore rename failed: {err}"),
        })?;
    } else if !entry.existed_before {
        // Remove newly created file without following symlinks.
        if fs::symlink_metadata(&entry.target).is_ok() {
            fs::remove_file(&entry.target).map_err(|err| FsTxError::RollbackFailed {
                original: entry.target.clone(),
                staging: PathBuf::new(),
                message: format!("remove created file failed: {err}"),
            })?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Backup path generation
// ---------------------------------------------------------------------------

fn temp_backup_path(target: &Path) -> Result<PathBuf, FsTxError> {
    let mut backup = target.as_os_str().to_os_string();
    backup.push(".backup~");
    let mut candidate = PathBuf::from(backup);
    let mut suffix = 0u64;
    while fs::symlink_metadata(&candidate).is_ok() {
        suffix += 1;
        let mut name = target.as_os_str().to_os_string();
        name.push(format!(".backup{suffix}~"));
        candidate = PathBuf::from(name);
    }
    Ok(candidate)
}

// ---------------------------------------------------------------------------
// Staging directory helpers
// ---------------------------------------------------------------------------

/// Create a unique staging directory sibling to the given path.
///
/// Uses process-local entropy (PID + nanosecond time + attempt counter)
/// so the resulting path is unpredictable and immune to symlink
/// pre-creation TOCTOU races. The final directory is created with
/// `create_dir` (not `create_dir_all`) because the parent already exists;
/// `create_dir` atomically returns `AlreadyExists` when anything—including
/// a symlink—already occupies the name.
///
/// Returns the staging directory path. The caller is responsible for cleanup
/// on failure (use `cleanup_staging`).
pub fn create_staging_sibling(target: &Path) -> Result<PathBuf, FsTxError> {
    use std::hash::{Hash, Hasher};

    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|err| FsTxError::Io {
        path: parent.to_path_buf(),
        message: format!("create parent dir for staging: {err}"),
    })?;
    let stem = target
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("staging");

    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    for attempt in 0..100u32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        pid.hash(&mut hasher);
        nanos.hash(&mut hasher);
        attempt.hash(&mut hasher);
        let hash = hasher.finish();

        let name = format!(".{stem}.{hash:016x}.{attempt}");
        let staging = parent.join(name);

        // create_dir is atomic: returns AlreadyExists for any existing
        // entry (file, dir, symlink), closing the TOCTOU window.
        match fs::create_dir(&staging) {
            Ok(()) => return Ok(staging),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(FsTxError::Io {
                    path: staging,
                    message: format!("create staging dir: {err}"),
                });
            }
        }
    }
    Err(FsTxError::StagingFailed {
        staging_path: parent.to_path_buf(),
        message: "could not create unique staging directory".to_string(),
    })
}

/// Reserve a unique staging file sibling to the given target path.
///
/// On Unix, `O_NOFOLLOW` is added so the kernel rejects the open when the
/// path component is a symlink, closing a TOCTOU race where an attacker
/// replaces a non-existent path with a symlink between the implicit
/// existence check and file creation.
///
/// On all platforms, the filename includes process-local entropy (PID +
/// nanosecond time + attempt counter) so the path is unpredictable.
pub fn create_staging_file_sibling(target: &Path) -> Result<PathBuf, FsTxError> {
    use std::hash::{Hash, Hasher};

    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|err| FsTxError::Io {
        path: parent.to_path_buf(),
        message: format!("create parent dir for staging file: {err}"),
    })?;

    let stem = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("staging");
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    for attempt in 0..100u32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        pid.hash(&mut hasher);
        nanos.hash(&mut hasher);
        attempt.hash(&mut hasher);
        let hash = hasher.finish();

        let staging = parent.join(format!(".{stem}.{hash:016x}.{attempt}.tmp"));

        match reserve_new_file_no_follow(&staging) {
            Ok(_) => return Ok(staging),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(FsTxError::Io {
                    path: staging,
                    message: format!("reserve staging file: {err}"),
                });
            }
        }
    }
    Err(FsTxError::StagingFailed {
        staging_path: parent.to_path_buf(),
        message: "could not reserve unique staging file".to_string(),
    })
}

fn reserve_new_file_no_follow(path: &Path) -> std::io::Result<fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
    }

    #[cfg(not(unix))]
    {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
    }
}

/// Remove a staging path tree or file. Errors are logged through `eprintln` but not returned
/// as cleanup failure should not obscure the original operation error.
pub fn cleanup_staging(staging: &Path) {
    let result = match fs::symlink_metadata(staging) {
        Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(staging),
        Ok(_) => fs::remove_file(staging),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    };
    if let Err(err) = result {
        eprintln!(
            "warning: cleanup of staging path '{}' failed: {err}",
            staging.display()
        );
    }
}

/// Atomically rename `staging` to `target`. Both must be on the same filesystem
/// for atomicity guarantees.
pub fn publish_staging(staging: &Path, target: &Path) -> Result<(), FsTxError> {
    if fs::symlink_metadata(target).is_ok() {
        return Err(FsTxError::ExistingTarget(target.to_path_buf()));
    }
    fs::rename(staging, target).map_err(|err| FsTxError::PublicationFailed {
        target: target.to_path_buf(),
        staging: staging.to_path_buf(),
        message: format!("atomic rename failed: {err}"),
    })?;
    Ok(())
}

/// Atomically rename a staged file over an existing regular file.
pub fn replace_file_with_staging(staging: &Path, target: &Path) -> Result<(), FsTxError> {
    let metadata = inspect_entry_no_follow(target)?;
    if !metadata.is_file() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "replacement target is not a regular file: '{}'",
            target.display()
        )));
    }
    fs::rename(staging, target).map_err(|err| FsTxError::PublicationFailed {
        target: target.to_path_buf(),
        staging: staging.to_path_buf(),
        message: format!("atomic replacement rename failed: {err}"),
    })
}

// ---------------------------------------------------------------------------
// Atomic directory publication with no-replace semantics (Phase 08)
// ---------------------------------------------------------------------------

/// Test hook that runs after preflight checks but before the atomic
/// no-replace syscall. Only set during testing; production code is
/// never aware of it.
static PRE_PUBLISH_HOOK: std::sync::Mutex<Option<Box<dyn Fn() + Send + 'static>>> =
    std::sync::Mutex::new(None);

/// Set a hook that runs inside `publish_directory_no_replace` after all
/// preflight checks but immediately before the `renameat2` syscall.
///
/// # Panics
/// Panics if a hook is already installed (call `clear_pre_publish_hook` first).
pub fn set_pre_publish_hook<F: Fn() + Send + 'static>(hook: F) {
    let mut guard = PRE_PUBLISH_HOOK
        .lock()
        .expect("pre-publish hook lock poisoned");
    assert!(
        guard.is_none(),
        "pre-publish hook already set; call clear_pre_publish_hook first"
    );
    *guard = Some(Box::new(hook));
}

/// Remove a previously installed pre-publish hook.
pub fn clear_pre_publish_hook() {
    let mut guard = PRE_PUBLISH_HOOK
        .lock()
        .expect("pre-publish hook lock poisoned");
    *guard = None;
}

fn run_pre_publish_hook() {
    let hook = PRE_PUBLISH_HOOK
        .lock()
        .expect("pre-publish hook lock poisoned")
        .take();
    if let Some(hook) = hook {
        hook();
    }
}

/// Atomically rename `staging` directory to `target` with no-replace
/// semantics.
///
/// On Linux, uses `renameat2(AT_FDCWD, staging, AT_FDCWD, target,
/// RENAME_NOREPLACE)` so the kernel rejects the rename if anything
/// already exists at `target`. This closes the TOCTOU race between
/// existence check and rename.
///
/// On non-Linux platforms, returns `FsTxError::UnsupportedPlatform`.
/// Callers must handle this gracefully.
///
/// Before the syscall, runs any test hook installed via
/// `set_pre_publish_hook`.
pub fn publish_directory_no_replace(staging: &Path, target: &Path) -> Result<(), FsTxError> {
    // Verify staging is a real directory (not a symlink)
    let staging_meta = inspect_entry_no_follow(staging)?;
    if !staging_meta.is_dir() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "staging is not a directory: '{}'",
            staging.display()
        )));
    }

    // Verify parent of target exists and is a real directory
    let target_parent = target.parent().unwrap_or_else(|| Path::new("."));
    let parent_meta =
        inspect_entry_no_follow(target_parent).map_err(|_| FsTxError::PublicationFailed {
            target: target.to_path_buf(),
            staging: staging.to_path_buf(),
            message: format!(
                "target parent does not exist or is a symlink: '{}'",
                target_parent.display()
            ),
        })?;
    if !parent_meta.is_dir() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "target parent is not a directory: '{}'",
            target_parent.display()
        )));
    }

    // Verify staging and target are on the same filesystem (renameat2
    // requires this for atomicity). We check by comparing the parent
    // directory device IDs where possible, but the primary guarantee
    // is that both are under the same canonical parent tree.
    let staging_parent = staging.parent().unwrap_or_else(|| Path::new("."));
    let staging_parent_canon = staging_parent.canonicalize().map_err(|err| FsTxError::Io {
        path: staging_parent.to_path_buf(),
        message: format!("canonicalize staging parent: {err}"),
    })?;
    let target_parent_canon = target_parent.canonicalize().map_err(|err| FsTxError::Io {
        path: target_parent.to_path_buf(),
        message: format!("canonicalize target parent: {err}"),
    })?;
    if staging_parent_canon != target_parent_canon {
        return Err(FsTxError::PublicationFailed {
            target: target.to_path_buf(),
            staging: staging.to_path_buf(),
            message: format!(
                "staging parent '{}' and target parent '{}' are different directories; \
                 renameat2 requires same filesystem",
                staging_parent_canon.display(),
                target_parent_canon.display()
            ),
        });
    }

    // Run any test hook just before the syscall.
    run_pre_publish_hook();

    // Ownership metadata is valid only while this is a staging directory.
    // Check after the hook so a race-injection test cannot publish it.
    ensure_staging_marker_absent(staging)?;

    // Attempt the atomic no-replace rename.
    rename_directory_no_replace(staging, target)
}

/// Platform-specific no-replace directory rename.
fn rename_directory_no_replace(staging: &Path, target: &Path) -> Result<(), FsTxError> {
    #[cfg(target_os = "linux")]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let staging_cstr = CString::new(staging.as_os_str().as_bytes()).map_err(|_| {
            FsTxError::InvalidEntryPath(format!(
                "staging path contains NUL byte: '{}'",
                staging.display()
            ))
        })?;
        let target_cstr = CString::new(target.as_os_str().as_bytes()).map_err(|_| {
            FsTxError::InvalidEntryPath(format!(
                "target path contains NUL byte: '{}'",
                target.display()
            ))
        })?;

        // renameat2 with RENAME_NOREPLACE is atomic and fails with
        // EEXIST if the target already exists.
        let ret = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                staging_cstr.as_ptr(),
                libc::AT_FDCWD,
                target_cstr.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };

        if ret == 0 {
            return Ok(());
        }

        let err = std::io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EEXIST) => {
                // Determine what kind of entry already exists
                let existing = match fs::symlink_metadata(target) {
                    Ok(meta) => {
                        if meta.is_dir() {
                            "directory".to_string()
                        } else if meta.is_file() {
                            "file".to_string()
                        } else if meta.file_type().is_symlink() {
                            "symlink".to_string()
                        } else {
                            "entry".to_string()
                        }
                    }
                    Err(_) => "entry".to_string(),
                };
                Err(FsTxError::PreExistingDestination {
                    target: target.to_path_buf(),
                    message: format!("a {existing} already exists at destination"),
                })
            }
            Some(libc::ENOSYS) | Some(libc::EINVAL) => Err(FsTxError::UnsupportedPlatform {
                operation: "renameat2(RENAME_NOREPLACE)".to_string(),
                reason: format!("kernel does not support renameat2: {err}"),
            }),
            Some(libc::EXDEV) => Err(FsTxError::PublicationFailed {
                target: target.to_path_buf(),
                staging: staging.to_path_buf(),
                message: "staging and target are on different filesystems".to_string(),
            }),
            _ => Err(FsTxError::PublicationFailed {
                target: target.to_path_buf(),
                staging: staging.to_path_buf(),
                message: format!("renameat2 failed: {err}"),
            }),
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = staging;
        let _ = target;
        Err(FsTxError::UnsupportedPlatform {
            operation: "publish_directory_no_replace".to_string(),
            reason: "atomic no-replace directory publication requires Linux renameat2".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Staging artifact set validation (Phase 08)
// ---------------------------------------------------------------------------

/// Compute SHA-256 hex digests for all regular files in a directory tree
/// (sorted by relative path). Returns `Vec<(relative_path, sha256_hex)>`.
pub fn compute_dir_file_hashes(root: &Path) -> Result<Vec<(String, String)>, FsTxError> {
    let mut hashes = Vec::new();
    collect_file_hashes(root, root, &mut hashes)?;
    hashes.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(hashes)
}

fn collect_file_hashes(
    root: &Path,
    dir: &Path,
    hashes: &mut Vec<(String, String)>,
) -> Result<(), FsTxError> {
    let dir_meta = inspect_entry_no_follow(dir)?;
    if !dir_meta.is_dir() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "not a directory: '{}'",
            dir.display()
        )));
    }

    let mut entries: Vec<_> = fs::read_dir(dir)
        .map_err(|err| FsTxError::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir: {err}"),
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| FsTxError::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir entry: {err}"),
        })?;
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        let meta = inspect_entry_no_follow(&path)?;
        if meta.is_dir() {
            collect_file_hashes(root, &path, hashes)?;
        } else if meta.is_file() {
            let relative = path.strip_prefix(root).map_err(|_| FsTxError::RootEscape {
                candidate: path.clone(),
                root: root.to_path_buf(),
            })?;
            let hash = sha256_file(&path)?;
            hashes.push((slash_path(relative), hash));
        }
    }
    Ok(())
}

/// Compute SHA-256 of a regular file.
fn sha256_file(path: &Path) -> Result<String, FsTxError> {
    use std::io::Read;

    let mut file = fs::File::open(path).map_err(|err| FsTxError::Io {
        path: path.to_path_buf(),
        message: format!("open for hash: {err}"),
    })?;

    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut buf = [0u8; 64];
    let mut buf_len = 0usize;
    let mut total_len = 0u64;

    loop {
        let n = file
            .read(&mut buf[buf_len..])
            .map_err(|err| FsTxError::Io {
                path: path.to_path_buf(),
                message: format!("read for hash: {err}"),
            })?;
        if n == 0 {
            break;
        }
        total_len += n as u64;
        buf_len += n;
        if buf_len == 64 {
            sha256_process_block(&mut state, &buf);
            buf_len = 0;
        }
    }

    // Padding
    let total_bits = total_len * 8;
    buf[buf_len] = 0x80;
    buf_len += 1;
    if buf_len > 56 {
        for i in buf_len..64 {
            buf[i] = 0;
        }
        sha256_process_block(&mut state, &buf);
        buf_len = 0;
    }
    for i in buf_len..56 {
        buf[i] = 0;
    }
    buf[56..64].copy_from_slice(&total_bits.to_be_bytes());
    sha256_process_block(&mut state, &buf);

    let mut result = String::with_capacity(64);
    for &word in &state {
        result.push_str(&format!("{word:08x}"));
    }
    Ok(result)
}

fn sha256_process_block(state: &mut [u32; 8], block: &[u8; 64]) {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut w = [0u32; 64];
    for i in 0..16 {
        let base = i * 4;
        w[i] = u32::from_be_bytes([
            block[base],
            block[base + 1],
            block[base + 2],
            block[base + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

fn slash_path(path: &Path) -> String {
    let mut parts = Vec::new();
    for component in path.components() {
        if let std::path::Component::Normal(part) = component {
            if let Some(s) = part.to_str() {
                parts.push(s.to_string());
            }
        }
    }
    parts.join("/")
}

// ---------------------------------------------------------------------------
// Publication outcome (Phase 04)
// ---------------------------------------------------------------------------

/// Structured publication outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicationOutcome {
    /// Published successfully — staging renamed to destination.
    Published {
        /// The destination path.
        target: PathBuf,
        /// SHA-256 of the canonical manifest.
        manifest_sha256: String,
    },
    /// Staging and existing destination are byte-identical — no-op.
    Unchanged {
        /// The existing destination path.
        target: PathBuf,
        /// SHA-256 of the canonical manifest.
        manifest_sha256: String,
    },
    /// Destination exists with a different complete closure.
    LateCollision {
        /// The existing destination path.
        target: PathBuf,
        /// SHA-256 of the new manifest.
        new_manifest_sha256: String,
        /// SHA-256 of the existing manifest (if parseable).
        existing_manifest_sha256: Option<String>,
    },
    /// Destination exists but is incomplete or malformed.
    IncompleteDestination {
        /// The existing destination path.
        target: PathBuf,
        /// What was wrong with it.
        reason: String,
    },
}

impl std::fmt::Display for PublicationOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublicationOutcome::Published {
                target,
                manifest_sha256,
            } => {
                write!(
                    f,
                    "published: {} (manifest sha256:{})",
                    target.display(),
                    manifest_sha256
                )
            }
            PublicationOutcome::Unchanged {
                target,
                manifest_sha256,
            } => {
                write!(
                    f,
                    "unchanged: {} (manifest sha256:{})",
                    target.display(),
                    manifest_sha256
                )
            }
            PublicationOutcome::LateCollision {
                target,
                new_manifest_sha256,
                existing_manifest_sha256,
            } => {
                write!(
                    f,
                    "late-collision: {} new={} existing={}",
                    target.display(),
                    new_manifest_sha256,
                    existing_manifest_sha256
                        .as_deref()
                        .unwrap_or("<unparseable>")
                )
            }
            PublicationOutcome::IncompleteDestination { target, reason } => {
                write!(
                    f,
                    "incomplete-destination: {} ({})",
                    target.display(),
                    reason
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Owned staging markers (Phase 04)
// ---------------------------------------------------------------------------

/// Name of the ownership marker written into every staging directory.
pub const STAGING_MARKER_NAME: &str = ".engine-pack-staging";

/// Write an ownership marker into the staging directory that binds it to the
/// intended destination.
///
/// The marker deliberately excludes process IDs, timestamps, and absolute
/// paths. It exists only while the directory is staging and must be removed
/// before final closure validation and publication.
pub fn write_staging_marker(staging: &Path, destination: &Path) -> Result<(), FsTxError> {
    let destination_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            FsTxError::InvalidEntryPath(format!(
                "destination has no valid final path component: '{}'",
                destination.display()
            ))
        })?;
    let marker_path = staging.join(STAGING_MARKER_NAME);
    let content = format!("engine-pack-staging-v1\ndestination_name={destination_name}\n");
    fs::write(&marker_path, content).map_err(|err| FsTxError::Io {
        path: marker_path,
        message: format!("write staging marker: {err}"),
    })
}

/// Remove the regular ownership marker before the staging directory becomes a
/// published package.
pub fn remove_staging_marker(staging: &Path) -> Result<(), FsTxError> {
    let marker_path = staging.join(STAGING_MARKER_NAME);
    let metadata = inspect_entry_no_follow(&marker_path)?;
    if !metadata.is_file() {
        return Err(FsTxError::StagingArtifactInvariant {
            staging: staging.to_path_buf(),
            message: format!(
                "staging ownership marker is not a regular file: '{}'",
                marker_path.display()
            ),
        });
    }
    fs::remove_file(&marker_path).map_err(|err| FsTxError::Io {
        path: marker_path,
        message: format!("remove staging marker: {err}"),
    })
}

fn ensure_staging_marker_absent(staging: &Path) -> Result<(), FsTxError> {
    let marker_path = staging.join(STAGING_MARKER_NAME);
    match fs::symlink_metadata(&marker_path) {
        Ok(_) => Err(FsTxError::StagingArtifactInvariant {
            staging: staging.to_path_buf(),
            message: format!(
                "staging ownership marker must be removed before publication: '{}'",
                marker_path.display()
            ),
        }),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(FsTxError::Io {
            path: marker_path,
            message: format!("inspect staging marker before publication: {err}"),
        }),
    }
}

/// Check whether a directory contains a regular, non-symlink staging marker.
pub fn has_staging_marker(dir: &Path) -> bool {
    fs::symlink_metadata(dir.join(STAGING_MARKER_NAME))
        .map(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
        .unwrap_or(false)
}

/// Check whether an ownership marker binds a staging sibling to `destination`.
pub fn staging_marker_matches_destination(dir: &Path, destination: &Path) -> bool {
    let Some(destination_name) = destination.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let expected = format!("engine-pack-staging-v1\ndestination_name={destination_name}\n");
    fs::read_to_string(dir.join(STAGING_MARKER_NAME))
        .map(|actual| actual == expected)
        .unwrap_or(false)
}

/// Recover orphaned staging directories that are direct siblings of the
/// given destination. Only removes directories that contain the staging
/// marker AND are named with the engine-pack staging prefix.
pub fn recover_orphaned_staging(destination: &Path) {
    let parent = match destination.parent() {
        Some(p) => p,
        None => return,
    };
    let dest_stem = destination
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let staging_prefix = format!(".{dest_stem}.");

    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Only consider directories with the staging prefix
        if !name.starts_with(&staging_prefix) {
            continue;
        }
        // Skip the destination itself
        if path == *destination {
            continue;
        }
        // Only recover a real directory with a matching ownership marker
        // and a fully inspectable, symlink-free tree.
        let Ok(metadata) = inspect_entry_no_follow(&path) else {
            continue;
        };
        if !metadata.is_dir()
            || !has_staging_marker(&path)
            || !staging_marker_matches_destination(&path, destination)
            || !owned_staging_tree_is_safe(&path)
        {
            continue;
        }
        let _ = fs::remove_dir_all(&path);
    }
}

fn owned_staging_tree_is_safe(dir: &Path) -> bool {
    let Ok(metadata) = inspect_entry_no_follow(dir) else {
        return false;
    };
    if !metadata.is_dir() {
        return false;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries {
        let Ok(entry) = entry else {
            return false;
        };
        let path = entry.path();
        let Ok(metadata) = inspect_entry_no_follow(&path) else {
            return false;
        };
        if metadata.is_dir() {
            if !owned_staging_tree_is_safe(&path) {
                return false;
            }
        } else if !metadata.is_file() {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Manifest closure validation (Phase 04)
// ---------------------------------------------------------------------------

/// Validate that every declared payload in a manifest is present, regular,
/// confined, size-matching, and hash-matching in the staging directory.
/// Also rejects any undeclared regular artifact.
pub fn validate_manifest_closure(
    staging: &Path,
    manifest_bytes: &[u8],
) -> Result<Vec<(String, String)>, FsTxError> {
    let manifest_str =
        std::str::from_utf8(manifest_bytes).map_err(|e| FsTxError::ValidationBeforePublish {
            staging: staging.to_path_buf(),
            diagnostics: vec![format!("manifest is not valid UTF-8: {e}")],
        })?;
    let manifest: toml::Value =
        toml::from_str(manifest_str).map_err(|e| FsTxError::ValidationBeforePublish {
            staging: staging.to_path_buf(),
            diagnostics: vec![format!("invalid manifest TOML: {e}")],
        })?;

    let table = manifest
        .as_table()
        .ok_or_else(|| FsTxError::ValidationBeforePublish {
            staging: staging.to_path_buf(),
            diagnostics: vec!["manifest root is not a table".to_string()],
        })?;

    let mut diagnostics: Vec<String> = Vec::new();
    if table.get("manifest_schema").and_then(toml::Value::as_str) != Some("engine-pack-canonical/1")
    {
        diagnostics.push("manifest has an unsupported or missing manifest_schema".to_string());
    }
    if table.get("strict").and_then(toml::Value::as_bool) != Some(true) {
        diagnostics.push("manifest must declare strict = true".to_string());
    }

    let mut declared: Vec<(String, String, u64)> = Vec::new();
    match table
        .get("published_artifacts")
        .and_then(toml::Value::as_array)
    {
        Some(artifacts) => {
            for entry in artifacts {
                let Some(entry) = entry.as_table() else {
                    diagnostics.push("published_artifact is not a table".to_string());
                    continue;
                };
                let Some(path) = entry.get("path").and_then(toml::Value::as_str) else {
                    diagnostics.push("published_artifact is missing a string path".to_string());
                    continue;
                };
                let Some(sha256) = entry.get("sha256").and_then(toml::Value::as_str) else {
                    diagnostics.push(format!("published_artifact '{path}' is missing sha256"));
                    continue;
                };
                let Some(bytes) = entry.get("bytes").and_then(toml::Value::as_integer) else {
                    diagnostics.push(format!("published_artifact '{path}' is missing bytes"));
                    continue;
                };
                let Ok(bytes) = u64::try_from(bytes) else {
                    diagnostics.push(format!("published_artifact '{path}' has invalid bytes"));
                    continue;
                };
                let canonical_path = match normalize_logical_key(Path::new(path)) {
                    Ok(path) => path,
                    Err(_) => {
                        diagnostics.push(format!(
                            "published_artifact path escape, absolute path, or noncanonical path: '{path}'"
                        ));
                        continue;
                    }
                };
                if canonical_path != path {
                    diagnostics.push(format!(
                        "published_artifact path is not canonical: '{path}' (expected '{canonical_path}')"
                    ));
                    continue;
                }
                if canonical_path == STAGING_MARKER_NAME
                    || canonical_path.ends_with(".manifest.toml")
                {
                    diagnostics.push(format!(
                        "published_artifact '{canonical_path}' is transaction metadata, not a payload"
                    ));
                    continue;
                }
                if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    diagnostics.push(format!(
                        "published_artifact '{canonical_path}' has invalid sha256"
                    ));
                    continue;
                }
                declared.push((canonical_path, sha256.to_ascii_lowercase(), bytes));
            }
        }
        None => diagnostics.push("manifest has no published_artifacts array".to_string()),
    }

    if declared.is_empty() {
        diagnostics.push("manifest has no published_artifacts".to_string());
    }

    let mut seen_paths: HashSet<String> = HashSet::new();
    for (path, _, _) in &declared {
        if !seen_paths.insert(path.clone()) {
            diagnostics.push(format!("duplicate declared path: '{path}'"));
        }
    }

    let mut actual_files: Vec<(String, String, u64)> = Vec::new();
    collect_staging_files(staging, staging, &mut actual_files, &mut diagnostics)?;

    let manifest_paths: Vec<&str> = actual_files
        .iter()
        .map(|(path, _, _)| path.as_str())
        .filter(|path| path.ends_with(".manifest.toml"))
        .collect();
    let canonical_manifest_path = match manifest_paths.as_slice() {
        [path] => {
            let path = *path;
            match fs::read(staging.join(path)) {
                Ok(bytes) if bytes == manifest_bytes => Some(path),
                Ok(_) => {
                    diagnostics.push(format!(
                        "manifest bytes do not match staged manifest '{path}'"
                    ));
                    None
                }
                Err(err) => {
                    diagnostics.push(format!("cannot read staged manifest '{path}': {err}"));
                    None
                }
            }
        }
        [] => {
            diagnostics.push("staging has no canonical manifest file".to_string());
            None
        }
        _ => {
            diagnostics.push("staging has multiple manifest files".to_string());
            None
        }
    };

    let actual_map: HashMap<&str, (&str, u64)> = actual_files
        .iter()
        .map(|(path, hash, bytes)| (path.as_str(), (hash.as_str(), *bytes)))
        .collect();
    for (path, declared_hash, declared_bytes) in &declared {
        match actual_map.get(path.as_str()) {
            Some((actual_hash, actual_bytes))
                if *actual_hash == declared_hash && *actual_bytes == *declared_bytes => {}
            Some((actual_hash, actual_bytes)) => diagnostics.push(format!(
                "artifact mismatch for '{path}': declared sha256={declared_hash} bytes={declared_bytes}, actual sha256={actual_hash} bytes={actual_bytes}"
            )),
            None => diagnostics.push(format!(
                "declared artifact '{path}' not found in staging"
            )),
        }
    }

    let declared_paths: HashSet<&str> = declared.iter().map(|(path, _, _)| path.as_str()).collect();
    for (path, _, _) in &actual_files {
        if canonical_manifest_path == Some(path.as_str()) {
            continue;
        }
        if !declared_paths.contains(path.as_str()) {
            diagnostics.push(format!("undeclared regular file in staging: '{path}'"));
        }
    }

    if !diagnostics.is_empty() {
        return Err(FsTxError::ValidationBeforePublish {
            staging: staging.to_path_buf(),
            diagnostics,
        });
    }

    Ok(declared
        .into_iter()
        .map(|(path, hash, _)| (path, hash))
        .collect())
}

/// Recursively collect all regular files in staging, including allowed
/// subdirectories such as `textures/`.
fn collect_staging_files(
    root: &Path,
    dir: &Path,
    files: &mut Vec<(String, String, u64)>,
    diagnostics: &mut Vec<String>,
) -> Result<(), FsTxError> {
    let dir_meta = inspect_entry_no_follow(dir)?;
    if !dir_meta.is_dir() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "not a directory: '{}'",
            dir.display()
        )));
    }

    let mut entries: Vec<_> = fs::read_dir(dir)
        .map_err(|err| FsTxError::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir: {err}"),
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| FsTxError::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir entry: {err}"),
        })?;
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        let meta = inspect_entry_no_follow(&path)?;

        if meta.is_dir() {
            let relative = path.strip_prefix(root).map_err(|_| FsTxError::RootEscape {
                candidate: path.clone(),
                root: root.to_path_buf(),
            })?;
            let logical_path = slash_path(relative);
            if logical_path == ".compile-work" || logical_path.starts_with(".compile-work/") {
                diagnostics.push(format!(
                    "stale compiler work directory in staging: '{logical_path}'"
                ));
            }
            // Recurse into subdirectories (e.g. textures/)
            collect_staging_files(root, &path, files, diagnostics)?;
        } else if meta.is_file() {
            let relative = path.strip_prefix(root).map_err(|_| FsTxError::RootEscape {
                candidate: path.clone(),
                root: root.to_path_buf(),
            })?;
            let hash = sha256_file(&path)?;
            files.push((slash_path(relative), hash, meta.len()));
        } else {
            diagnostics.push(format!(
                "non-regular entry in staging: '{}'",
                path.display()
            ));
        }
    }
    Ok(())
}

/// Compute the SHA-256 of the canonical manifest bytes.
pub fn compute_manifest_sha256(manifest_bytes: &[u8]) -> String {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut buf = [0u8; 64];
    let mut buf_len = 0usize;
    let mut total_len = 0u64;

    let mut offset = 0usize;
    while offset < manifest_bytes.len() {
        let space = 64 - buf_len;
        let copy = (manifest_bytes.len() - offset).min(space);
        buf[buf_len..buf_len + copy].copy_from_slice(&manifest_bytes[offset..offset + copy]);
        buf_len += copy;
        total_len += copy as u64;
        offset += copy;
        if buf_len == 64 {
            sha256_process_block(&mut state, &buf);
            buf_len = 0;
        }
    }

    let total_bits = total_len * 8;
    buf[buf_len] = 0x80;
    buf_len += 1;
    if buf_len > 56 {
        for i in buf_len..64 {
            buf[i] = 0;
        }
        sha256_process_block(&mut state, &buf);
        buf_len = 0;
    }
    for i in buf_len..56 {
        buf[i] = 0;
    }
    buf[56..64].copy_from_slice(&total_bits.to_be_bytes());
    sha256_process_block(&mut state, &buf);

    let mut result = String::with_capacity(64);
    for &word in &state {
        result.push_str(&format!("{word:08x}"));
    }
    result
}

/// Validate a staged compiler artifact set.
///
/// Checks:
/// - No `.compile-work` directory remains
/// - No stale pointfile (`.pts`, `.prt`)
/// - No duplicate basenames
/// - No symlinks in staging
/// - For BSP2 profile: requires nonempty `.lit` alongside `.bsp`
/// - For BSP29 profile: `.lit` is optional (but diagnosed if present)
/// - Exactly one `.bsp` file
/// - Optional `.provenance.toml`
///
/// Returns the list of relative file paths (sorted) found in staging.
pub fn validate_staged_artifact_set(
    staging: &Path,
    bsp_name: &str,
    require_lit: bool,
) -> Result<Vec<String>, FsTxError> {
    let mut files: Vec<String> = Vec::new();
    let mut diagnostics: Vec<String> = Vec::new();

    collect_staged_entries(staging, staging, &mut files, &mut diagnostics)?;

    // Check for forbidden entries
    for f in &files {
        let basename = std::path::Path::new(f)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Reject .compile-work directories
        if f.contains(".compile-work") {
            diagnostics.push(format!("stale .compile-work found: '{f}'"));
        }
        // Reject stale pointfiles
        if basename.ends_with(".pts") || basename.ends_with(".prt") {
            diagnostics.push(format!("stale pointfile found: '{f}'"));
        }
    }

    // Check duplicate basenames
    let mut seen_basenames: std::collections::HashSet<String> = std::collections::HashSet::new();
    for f in &files {
        let basename = std::path::Path::new(f)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if !seen_basenames.insert(basename) {
            diagnostics.push(format!("duplicate basename in staging: '{f}'"));
        }
    }

    // Check for required .bsp
    let bsp_expected = format!("{bsp_name}.bsp");
    let has_bsp = files.iter().any(|f| f == &bsp_expected);
    if !has_bsp {
        diagnostics.push(format!(
            "missing required .bsp file '{bsp_expected}' in staging"
        ));
    }

    // Check .lit requirement
    let lit_expected = format!("{bsp_name}.lit");
    let has_lit = files.iter().any(|f| f == &lit_expected);
    if require_lit && !has_lit {
        diagnostics.push(format!(
            "missing required .lit file '{lit_expected}' for BSP2 profile"
        ));
    }

    if !diagnostics.is_empty() {
        return Err(FsTxError::ValidationBeforePublish {
            staging: staging.to_path_buf(),
            diagnostics,
        });
    }

    // Verify .lit is nonempty when required
    if require_lit && has_lit {
        let lit_path = staging.join(&lit_expected);
        let lit_meta = inspect_entry_no_follow(&lit_path)?;
        if lit_meta.len() <= 8 {
            // QLIT header is 8 bytes; payload must be nonempty
            return Err(FsTxError::StagingArtifactInvariant {
                staging: staging.to_path_buf(),
                message: format!(
                    "required .lit file '{lit_expected}' has no payload ({} bytes)",
                    lit_meta.len()
                ),
            });
        }
    }

    Ok(files)
}

fn collect_staged_entries(
    root: &Path,
    dir: &Path,
    files: &mut Vec<String>,
    diagnostics: &mut Vec<String>,
) -> Result<(), FsTxError> {
    let dir_meta = inspect_entry_no_follow(dir)?;
    if !dir_meta.is_dir() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "not a directory: '{}'",
            dir.display()
        )));
    }

    let mut entries: Vec<_> = fs::read_dir(dir)
        .map_err(|err| FsTxError::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir: {err}"),
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| FsTxError::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir entry: {err}"),
        })?;
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        let meta = inspect_entry_no_follow(&path)?;

        if meta.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some(".compile-work") {
                diagnostics.push(format!(
                    ".compile-work directory found in staging: '{}'",
                    path.display()
                ));
            }
            collect_staged_entries(root, &path, files, diagnostics)?;
        } else if meta.is_file() {
            let relative = path.strip_prefix(root).map_err(|_| FsTxError::RootEscape {
                candidate: path.clone(),
                root: root.to_path_buf(),
            })?;
            files.push(slash_path(relative));
        }
    }
    Ok(())
}

/// Compare two directories by content hash. Returns `true` if both
/// contain the same set of files (by relative path) with identical
/// SHA-256 content.
pub fn artifact_sets_identical(staging: &Path, existing: &Path) -> Result<bool, FsTxError> {
    let staging_hashes = compute_dir_file_hashes(staging)?;
    let existing_hashes = compute_dir_file_hashes(existing)?;

    if staging_hashes.len() != existing_hashes.len() {
        return Ok(false);
    }

    for (s, e) in staging_hashes.iter().zip(existing_hashes.iter()) {
        if s.0 != e.0 || s.1 != e.1 {
            return Ok(false);
        }
    }

    Ok(true)
}

// ---------------------------------------------------------------------------
// Safe traversal (Gate 6: never follow links, root containment)
// ---------------------------------------------------------------------------

/// Inspect a path entry with `symlink_metadata` and reject symlinks.
///
/// Returns the metadata if the entry is not a symlink.
pub fn inspect_entry_no_follow(path: &Path) -> Result<fs::Metadata, FsTxError> {
    let metadata = fs::symlink_metadata(path).map_err(|err| FsTxError::Io {
        path: path.to_path_buf(),
        message: format!("symlink_metadata failed: {err}"),
    })?;
    if metadata.file_type().is_symlink() {
        return Err(FsTxError::SymlinkRejected {
            path: path.to_path_buf(),
            reason: "symlinks are not followed by default".to_string(),
        });
    }
    Ok(metadata)
}

/// Verify an existing root-relative child path stays contained and has no symlink component.
pub fn contained_child_no_symlinks(root: &Path, relative: &Path) -> Result<PathBuf, FsTxError> {
    if relative.as_os_str().is_empty() || relative.is_absolute() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "child path must be non-empty and relative: '{}'",
            relative.display()
        )));
    }

    let canonical_root = root.canonicalize().map_err(|err| FsTxError::Io {
        path: root.to_path_buf(),
        message: format!("canonicalize root failed: {err}"),
    })?;

    let mut current = canonical_root.clone();
    for component in relative.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => {
                current.push(part);
                let metadata = inspect_entry_no_follow(&current)?;
                if metadata.file_type().is_symlink() {
                    return Err(FsTxError::SymlinkRejected {
                        path: current,
                        reason: "symlink component in contained path".to_string(),
                    });
                }
            }
            Component::ParentDir | Component::Prefix(_) | Component::RootDir => {
                return Err(FsTxError::InvalidEntryPath(format!(
                    "child path escapes root: '{}'",
                    relative.display()
                )));
            }
        }
    }

    let canonical = current.canonicalize().map_err(|err| FsTxError::Io {
        path: current.clone(),
        message: format!("canonicalize child failed: {err}"),
    })?;
    if !canonical.starts_with(&canonical_root) {
        return Err(FsTxError::RootEscape {
            candidate: current,
            root: canonical_root,
        });
    }
    Ok(canonical)
}

/// Canonicalize a path and verify it is contained within `root`.
///
/// For non-existent paths, does a lexical containment check instead.
pub fn canonicalize_contained(path: &Path, root: &Path) -> Result<PathBuf, FsTxError> {
    let canonical_root = root.canonicalize().map_err(|err| FsTxError::Io {
        path: root.to_path_buf(),
        message: format!("canonicalize root failed: {err}"),
    })?;

    let canonical = if path.exists() {
        path.canonicalize().map_err(|err| FsTxError::Io {
            path: path.to_path_buf(),
            message: format!("canonicalize failed: {err}"),
        })?
    } else if path.is_absolute() {
        resolve_lexical(Path::new("/"), path)
    } else {
        // Lexical containment for paths that don't exist yet
        let resolved = resolve_lexical(&canonical_root, path);
        // Check lexical containment under canonical root
        if !resolved.starts_with(&canonical_root) {
            return Err(FsTxError::CrossRootCanonical {
                candidate: path.to_path_buf(),
                resolved,
                root: canonical_root,
            });
        }
        resolved
    };

    if !canonical.starts_with(&canonical_root) {
        return Err(FsTxError::RootEscape {
            candidate: path.to_path_buf(),
            root: canonical_root,
        });
    }
    Ok(canonical)
}

/// Lexically resolve a path against a root without accessing the filesystem.
fn resolve_lexical(root: &Path, relative: &Path) -> PathBuf {
    let mut out = root.to_path_buf();
    for component in relative.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Prefix(_) | Component::RootDir => {}
        }
    }
    out
}

/// Recursively collect plan entries from a directory without following symlinks.
///
/// Each entry is checked with `symlink_metadata`. Symlinks are rejected.
/// Directory entries are recursed into after metadata check.
pub fn collect_scan_plan(
    root: &Path,
    dir: &Path,
    entries: &mut Vec<PlanEntry>,
    visited: &mut HashSet<PathBuf>,
) -> Result<(), FsTxError> {
    let canonical_root = root.canonicalize().map_err(|err| FsTxError::Io {
        path: root.to_path_buf(),
        message: format!("canonicalize root: {err}"),
    })?;

    // Check self for symlink before reading
    let dir_meta = inspect_entry_no_follow(dir)?;
    if !dir_meta.is_dir() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "not a directory: '{}'",
            dir.display()
        )));
    }

    // Track filesystem identity to reject cycles
    if !visited.insert(canonicalize_identity(dir)?) {
        // Cycle detected — skip this directory silently
        return Ok(());
    }

    let mut read_entries: Vec<_> = fs::read_dir(dir)
        .map_err(|err| FsTxError::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir failed: {err}"),
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| FsTxError::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir entry error: {err}"),
        })?;

    // Sort for deterministic output
    read_entries.sort_by_key(|e| e.path());

    for entry in read_entries {
        let path = entry.path();
        let meta = inspect_entry_no_follow(&path)?;

        // Containment check
        let canonical = path.canonicalize().map_err(|err| FsTxError::Io {
            path: path.clone(),
            message: format!("canonicalize failed: {err}"),
        })?;
        if !canonical.starts_with(&canonical_root) {
            return Err(FsTxError::RootEscape {
                candidate: path,
                root: canonical_root,
            });
        }

        if meta.is_dir() {
            collect_scan_plan(root, &path, entries, visited)?;
        } else if meta.is_file() {
            let relative = path.strip_prefix(root).map_err(|_| FsTxError::RootEscape {
                candidate: path.clone(),
                root: root.to_path_buf(),
            })?;
            entries.push(PlanEntry {
                source: path.clone(),
                destination: relative.to_path_buf(),
                entry_type: EntryType::File,
                label: format!("scan: {}", relative.display()),
            });
        }
    }

    Ok(())
}

/// Get a unique filesystem identity for a path (device + inode or canonical path).
fn canonicalize_identity(path: &Path) -> Result<PathBuf, FsTxError> {
    path.canonicalize().map_err(|err| FsTxError::Io {
        path: path.to_path_buf(),
        message: format!("canonicalize for identity: {err}"),
    })
}

// ---------------------------------------------------------------------------
// Multi-artifact commit order (Gate 5)
// ---------------------------------------------------------------------------

/// The commit order for multi-artifact publication.
///
/// Assets first, then manifests, then project files. Each stage commits
/// atomically; if a stage fails, prior stages are rolled back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CommitPhase {
    /// Asset files (models, textures, etc.)
    Assets = 0,
    /// Package manifests
    Manifests = 1,
    /// Project files, scene files, reports
    Project = 2,
}

// ---------------------------------------------------------------------------
// Plan construction and validation
// ---------------------------------------------------------------------------

/// Build a publication plan from a list of source→destination pairs.
///
/// Validates: no duplicates, no empty destinations, no path escapes.
pub fn build_publication_plan(entries: Vec<PlanEntry>) -> Result<Vec<PlanEntry>, FsTxError> {
    let mut seen = HashSet::new();
    let mut result = Vec::with_capacity(entries.len());

    for entry in entries {
        // Validate destination
        if entry.destination.as_os_str().is_empty() {
            return Err(FsTxError::InvalidEntryPath("empty destination".to_string()));
        }
        if entry.destination.is_absolute() {
            return Err(FsTxError::InvalidEntryPath(format!(
                "absolute destination: '{}'",
                entry.destination.display()
            )));
        }
        // Reject parent-dir components
        for component in entry.destination.components() {
            if component == Component::ParentDir {
                return Err(FsTxError::InvalidEntryPath(format!(
                    "parent-dir component in: '{}'",
                    entry.destination.display()
                )));
            }
            match component {
                Component::Prefix(_) | Component::RootDir => {
                    return Err(FsTxError::InvalidEntryPath(format!(
                        "absolute or prefixed component in: '{}'",
                        entry.destination.display()
                    )));
                }
                _ => {}
            }
        }

        // Validate source exists and is not a symlink.
        let metadata = match inspect_entry_no_follow(&entry.source) {
            Ok(metadata) => metadata,
            Err(FsTxError::Io { path, .. }) => return Err(FsTxError::MissingInput(path)),
            Err(err) => return Err(err),
        };
        match entry.entry_type {
            EntryType::File if !metadata.is_file() => {
                return Err(FsTxError::InvalidEntryPath(format!(
                    "source is not a file: '{}'",
                    entry.source.display()
                )));
            }
            EntryType::Directory if !metadata.is_dir() => {
                return Err(FsTxError::InvalidEntryPath(format!(
                    "source is not a directory: '{}'",
                    entry.source.display()
                )));
            }
            _ => {}
        }

        // Check for duplicate destinations through the same UTF-8 logical
        // identity used by package manifests. Invalid byte sequences fail
        // closed instead of collapsing through replacement characters.
        let dest_key = normalize_logical_key(&entry.destination)?;
        if !seen.insert(dest_key) {
            return Err(FsTxError::DuplicateDestination(entry.destination.clone()));
        }

        result.push(entry);
    }

    Ok(result)
}

/// Copy a plan entry from source to a staging root, preserving relative structure.
pub fn stage_entry(staging_root: &Path, entry: &PlanEntry) -> Result<PathBuf, FsTxError> {
    let dest = staging_root.join(&entry.destination);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|err| FsTxError::Io {
            path: parent.to_path_buf(),
            message: format!("create parent dirs: {err}"),
        })?;
    }
    let metadata = inspect_entry_no_follow(&entry.source)?;
    match entry.entry_type {
        EntryType::File => {
            if !metadata.is_file() {
                return Err(FsTxError::InvalidEntryPath(format!(
                    "source is not a file: '{}'",
                    entry.source.display()
                )));
            }
            fs::copy(&entry.source, &dest).map_err(|err| FsTxError::Io {
                path: entry.source.clone(),
                message: format!("copy to staging: {err}"),
            })?;
        }
        EntryType::Directory => {
            if !metadata.is_dir() {
                return Err(FsTxError::InvalidEntryPath(format!(
                    "source is not a directory: '{}'",
                    entry.source.display()
                )));
            }
            fs::create_dir_all(&dest).map_err(|err| FsTxError::Io {
                path: dest.clone(),
                message: format!("create dir in staging: {err}"),
            })?;
        }
    }
    Ok(dest)
}

// ---------------------------------------------------------------------------
// Normalized asset path (logical key) utilities
// ---------------------------------------------------------------------------

/// Normalize a path to a canonical project-relative logical key.
///
/// - Removes `.` components
/// - Resolves lexical `..` only within root (rejects escape attempts)
/// - Normalizes separators to `/`
/// - Rejects empty, absolute, prefix, and root-escape paths
pub fn normalize_logical_key(path: &Path) -> Result<String, FsTxError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "invalid logical key: '{}'",
            path.display()
        )));
    }

    let mut parts = VecDeque::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => parts.push_back(part),
            Component::ParentDir => {
                if parts.pop_back().is_none() {
                    return Err(FsTxError::InvalidEntryPath(format!(
                        "logical key escapes root: '{}'",
                        path.display()
                    )));
                }
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(FsTxError::InvalidEntryPath(format!(
                    "logical key has absolute component: '{}'",
                    path.display()
                )));
            }
        }
    }

    if parts.is_empty() {
        return Err(FsTxError::InvalidEntryPath(format!(
            "logical key resolves to empty: '{}'",
            path.display()
        )));
    }

    let key: Vec<String> = parts
        .into_iter()
        .map(|p| {
            p.to_str().map(String::from).ok_or_else(|| {
                FsTxError::InvalidEntryPath(format!(
                    "path component is not valid UTF-8 in '{}'",
                    path.display()
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(key.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn unique_tmp(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("fs-tx-{label}-{}-{nanos}", std::process::id()))
    }

    #[test]
    fn normalize_logical_key_rejects_escapes() {
        assert!(normalize_logical_key(Path::new("../escape")).is_err());
        assert!(normalize_logical_key(Path::new("a/../../escape")).is_err());
        assert!(normalize_logical_key(Path::new("/absolute")).is_err());
        assert!(normalize_logical_key(Path::new("")).is_err());
    }

    #[test]
    fn normalize_logical_key_resolves_lexical() {
        assert_eq!(normalize_logical_key(Path::new("a/b/../c")).unwrap(), "a/c");
        assert_eq!(
            normalize_logical_key(Path::new("a/./b/c")).unwrap(),
            "a/b/c"
        );
        assert_eq!(
            normalize_logical_key(Path::new("models/crate.glb")).unwrap(),
            "models/crate.glb"
        );
    }

    #[test]
    fn manifest_closure_rejects_undeclared_or_mismatched_payloads() {
        let dir = unique_tmp("manifest-closure");
        fs::create_dir_all(&dir).unwrap();
        let payload_path = dir.join("payload.bin");
        fs::write(&payload_path, b"payload").unwrap();
        let hash = sha256_file(&payload_path).unwrap();
        let manifest = format!(
            "format_version = 1\nmanifest_schema = \"engine-pack-canonical/1\"\nstrict = true\n\n[[published_artifacts]]\npath = \"payload.bin\"\nsha256 = \"{hash}\"\nbytes = 7\nkind = \"test\"\n"
        );
        fs::write(dir.join("generated.manifest.toml"), &manifest).unwrap();

        assert!(validate_manifest_closure(&dir, manifest.as_bytes()).is_ok());

        fs::write(dir.join("rogue.bin"), b"rogue").unwrap();
        let error = validate_manifest_closure(&dir, manifest.as_bytes()).unwrap_err();
        match error {
            FsTxError::ValidationBeforePublish { diagnostics, .. } => {
                assert!(diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.contains("undeclared regular file")));
            }
            other => panic!("expected closure validation failure, got {other:?}"),
        }
        fs::remove_file(dir.join("rogue.bin")).unwrap();

        fs::write(&payload_path, b"changed").unwrap();
        let error = validate_manifest_closure(&dir, manifest.as_bytes()).unwrap_err();
        assert!(matches!(error, FsTxError::ValidationBeforePublish { .. }));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn manifest_closure_rejects_staging_marker() {
        let dir = unique_tmp("manifest-marker");
        fs::create_dir_all(&dir).unwrap();
        let payload_path = dir.join("payload.bin");
        fs::write(&payload_path, b"payload").unwrap();
        let hash = sha256_file(&payload_path).unwrap();
        let manifest = format!(
            "format_version = 1\nmanifest_schema = \"engine-pack-canonical/1\"\nstrict = true\n\n[[published_artifacts]]\npath = \"payload.bin\"\nsha256 = \"{hash}\"\nbytes = 7\nkind = \"test\"\n"
        );
        fs::write(dir.join("generated.manifest.toml"), &manifest).unwrap();
        write_staging_marker(&dir, &dir.join("published")).unwrap();

        let error = validate_manifest_closure(&dir, manifest.as_bytes()).unwrap_err();
        match error {
            FsTxError::ValidationBeforePublish { diagnostics, .. } => {
                assert!(diagnostics.iter().any(|diagnostic| diagnostic
                    .contains("undeclared regular file in staging: '.engine-pack-staging'")))
            }
            other => panic!("expected closure validation failure, got {other:?}"),
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn staging_marker_must_be_removed_before_directory_publication() {
        let dir = unique_tmp("publish-marker");
        fs::create_dir_all(&dir).unwrap();
        let staging = dir.join("staging");
        let target = dir.join("published");
        fs::create_dir_all(&staging).unwrap();
        write_staging_marker(&staging, &target).unwrap();

        let error = publish_directory_no_replace(&staging, &target).unwrap_err();
        assert!(matches!(error, FsTxError::StagingArtifactInvariant { .. }));
        assert!(staging.exists());
        assert!(!target.exists());

        remove_staging_marker(&staging).unwrap();
        assert!(!has_staging_marker(&staging));
        cleanup_staging(&staging);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn orphan_recovery_requires_owned_symlink_free_tree() {
        let dir = unique_tmp("orphan-recovery");
        fs::create_dir_all(&dir).unwrap();
        let destination = dir.join("published");
        let owned = dir.join(".published.owned");
        fs::create_dir_all(&owned).unwrap();
        write_staging_marker(&owned, &destination).unwrap();
        fs::write(owned.join("payload"), b"safe").unwrap();

        let wrong_destination = dir.join(".published.wrong-destination");
        fs::create_dir_all(&wrong_destination).unwrap();
        write_staging_marker(&wrong_destination, &dir.join("different")).unwrap();

        recover_orphaned_staging(&destination);
        assert!(!owned.exists(), "matching safe orphan must be recovered");
        assert!(
            wrong_destination.exists(),
            "mismatched ownership marker must not be removed"
        );

        fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn orphan_recovery_never_removes_symlinked_tree() {
        let dir = unique_tmp("orphan-symlink");
        fs::create_dir_all(&dir).unwrap();
        let destination = dir.join("published");
        let unsafe_orphan = dir.join(".published.unsafe");
        fs::create_dir_all(&unsafe_orphan).unwrap();
        write_staging_marker(&unsafe_orphan, &destination).unwrap();
        let outside = dir.join("outside");
        fs::write(&outside, b"outside").unwrap();
        std::os::unix::fs::symlink(&outside, unsafe_orphan.join("link")).unwrap();

        recover_orphaned_staging(&destination);
        assert!(
            unsafe_orphan.exists(),
            "an orphan containing a symlink must be retained for manual inspection"
        );
        assert_eq!(fs::read(&outside).unwrap(), b"outside");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn staging_publish_and_cleanup() {
        let dir = unique_tmp("staging");
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("final.txt");

        let staging = create_staging_sibling(&target).unwrap();
        assert!(staging.exists());

        // Write something into staging
        let staged_file = staging.join("hello.txt");
        fs::write(&staged_file, b"hello").unwrap();

        // Publish
        publish_staging(&staging, &target).unwrap();
        assert!(!staging.exists());
        assert!(target.exists());
        assert!(target.join("hello.txt").exists());

        fs::remove_dir_all(&target).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn publish_rejects_existing_target() {
        let dir = unique_tmp("existing");
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("final.txt");
        fs::create_dir_all(&target).unwrap();

        let staging = create_staging_sibling(&target).unwrap();
        let result = publish_staging(&staging, &target);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), FsTxError::ExistingTarget(_)));

        cleanup_staging(&staging);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rollback_journal_creates_and_restores() {
        let dir = unique_tmp("rollback");
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("data.txt");

        let mut journal = RollbackJournal::new();
        // Record creation
        journal.record_creation(&target);
        // Write file (simulate operation)
        fs::write(&target, b"staged content").unwrap();
        // Rollback
        journal.rollback().unwrap();
        assert!(!target.exists());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rollback_journal_backup_and_restore() {
        let dir = unique_tmp("rollback-backup");
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("data.txt");

        // Original content
        fs::write(&target, b"original").unwrap();

        let mut journal = RollbackJournal::new();
        journal.record_backup(&target).unwrap();
        // Overwrite (simulate failed operation)
        fs::write(&target, b"staged content").unwrap();
        // Rollback
        journal.rollback().unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "original");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn build_plan_rejects_duplicates_and_escapes() {
        let tmp = unique_tmp("plan");
        fs::create_dir_all(&tmp).unwrap();
        let source = tmp.join("a.txt");
        fs::write(&source, b"content").unwrap();

        // Duplicate destination
        let entries = vec![
            PlanEntry {
                source: source.clone(),
                destination: PathBuf::from("a.txt"),
                entry_type: EntryType::File,
                label: "first".into(),
            },
            PlanEntry {
                source: source.clone(),
                destination: PathBuf::from("a.txt"),
                entry_type: EntryType::File,
                label: "second".into(),
            },
        ];
        let err = build_publication_plan(entries).unwrap_err();
        assert!(matches!(err, FsTxError::DuplicateDestination(_)));

        // Parent-dir escape
        let entries = vec![PlanEntry {
            source: source.clone(),
            destination: PathBuf::from("../escape.txt"),
            entry_type: EntryType::File,
            label: "escape".into(),
        }];
        let err = build_publication_plan(entries).unwrap_err();
        assert!(matches!(err, FsTxError::InvalidEntryPath(_)));

        fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn symlink_rejected_by_inspect() {
        let dir = unique_tmp("symlink");
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("real.txt");
        fs::write(&target, b"real").unwrap();

        #[cfg(unix)]
        {
            let link = dir.join("link.txt");
            std::os::unix::fs::symlink(&target, &link).unwrap();
            let err = inspect_entry_no_follow(&link).unwrap_err();
            assert!(matches!(err, FsTxError::SymlinkRejected { .. }));
        }

        fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn staging_file_reservation_uses_no_follow_for_symlink_candidate() {
        let dir = unique_tmp("nofollow");
        fs::create_dir_all(&dir).unwrap();
        let referent = dir.join("real.txt");
        fs::write(&referent, b"real").unwrap();
        let link = dir.join("candidate.tmp");
        std::os::unix::fs::symlink(&referent, &link).unwrap();

        let err = reserve_new_file_no_follow(&link).unwrap_err();
        assert!(
            err.kind() == std::io::ErrorKind::AlreadyExists
                || err.raw_os_error() == Some(libc::ELOOP)
        );
        assert_eq!(fs::read(&referent).unwrap(), b"real");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn logical_keys_reject_non_utf8_before_duplicate_collapse() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let dir = unique_tmp("invalid-utf8");
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.txt");
        fs::write(&source, b"content").unwrap();
        let invalid_a = PathBuf::from(OsString::from_vec(vec![b'a', 0xff, b'.', b't', b'x', b't']));
        let invalid_b = PathBuf::from(OsString::from_vec(vec![b'a', 0xfe, b'.', b't', b'x', b't']));

        assert!(normalize_logical_key(&invalid_a).is_err());
        let err = build_publication_plan(vec![
            PlanEntry {
                source: source.clone(),
                destination: invalid_a,
                entry_type: EntryType::File,
                label: "first".into(),
            },
            PlanEntry {
                source,
                destination: invalid_b,
                entry_type: EntryType::File,
                label: "second".into(),
            },
        ])
        .unwrap_err();
        assert!(matches!(err, FsTxError::InvalidEntryPath(_)));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage_entry_preserves_structure() {
        let dir = unique_tmp("stage-entry");
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("input.txt");
        fs::write(&source, b"test").unwrap();

        let staging = dir.join("staging");
        fs::create_dir_all(&staging).unwrap();

        let entry = PlanEntry {
            source: source.clone(),
            destination: PathBuf::from("sub/dir/input.txt"),
            entry_type: EntryType::File,
            label: "test".into(),
        };
        let dest = stage_entry(&staging, &entry).unwrap();
        assert!(dest.exists());
        assert_eq!(
            dest,
            staging.join("sub/dir/input.txt").canonicalize().unwrap()
        );

        fs::remove_dir_all(&dir).unwrap();
    }
}
