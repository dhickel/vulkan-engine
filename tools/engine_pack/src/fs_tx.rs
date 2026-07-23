//! Filesystem transaction primitives for fail-closed tooling.
//!
//! Provides staging, safe traversal, atomic rename, rollback journal, and
//! symlink-free containment checks for `engine_pack` commands.

use std::collections::{HashSet, VecDeque};
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
