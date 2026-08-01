//! EnhancedV3 background generation for the BSP beta explorer.
//!
//! Compilation is deliberately isolated from the event/render thread. Every
//! request owns a distinct package target because engine_pack publication is
//! atomic and no-replace; a later request must never collide with an earlier
//! package or remove one still being imported.

use bsp_generator::enhanced_v3::{ArchType, V3Config, V3Preset};
use std::collections::VecDeque;
use std::env;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};

const DEFAULT_ERICW_TOOLS_DIR: &str = ".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolDiscoveryError {
    ExplicitInvalid(PathBuf),
}

impl fmt::Display for ToolDiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExplicitInvalid(path) => write!(
                f,
                "--ericw-tools '{}' does not contain executable qbsp, vis, and light",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ToolDiscoveryError {}

/// Discover a common directory containing `qbsp`, `vis`, and `light`.
///
/// Explicit input is authoritative: an invalid explicit directory is an error.
/// Environment, HOME, and PATH candidates are searched in that order.
pub fn discover_ericw_tools(
    explicit: Option<&Path>,
) -> Result<Option<PathBuf>, ToolDiscoveryError> {
    if let Some(dir) = explicit {
        return if tools_available(dir) {
            Ok(Some(dir.to_path_buf()))
        } else {
            Err(ToolDiscoveryError::ExplicitInvalid(dir.to_path_buf()))
        };
    }

    if let Ok(value) = env::var("ERICW_TOOLS_DIR") {
        let dir = PathBuf::from(value);
        if tools_available(&dir) {
            return Ok(Some(dir));
        }
        log::warn!("ERICW_TOOLS_DIR does not contain qbsp/vis/light; continuing discovery");
    }

    if let Ok(home) = env::var("HOME") {
        let dir = PathBuf::from(home).join(DEFAULT_ERICW_TOOLS_DIR);
        if tools_available(&dir) {
            return Ok(Some(dir));
        }
    }

    if let Some(path) = env::var_os("PATH") {
        for dir in env::split_paths(&path) {
            if tools_available(&dir) {
                return Ok(Some(dir));
            }
        }
    }
    Ok(None)
}

pub fn tools_available(dir: &Path) -> bool {
    ["qbsp", "vis", "light"]
        .into_iter()
        .map(|tool| dir.join(tool))
        .all(|path| is_executable_file(&path))
}

fn is_executable_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        return std::fs::metadata(path)
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false);
    }

    #[cfg(not(unix))]
    {
        true
    }
}

/// Create a process-unique root for packages created by this explorer run.
///
/// `create_dir` is the ownership claim: unlike a timestamp-only path it never
/// reuses a pre-existing directory, even when two launches share a clock tick.
pub fn create_unique_package_root() -> std::io::Result<PathBuf> {
    static ROOT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    for _ in 0..64 {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or_default();
        let sequence = ROOT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = env::temp_dir().join(format!(
            "bsp-beta-m3-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        match std::fs::create_dir(&root) {
            Ok(()) => return Ok(root),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not reserve a unique BSP beta generation directory",
    ))
}

/// Confined package directory for one request. IDs are generated internally;
/// callers never supply a path component.
pub fn package_dir_for_request(root: &Path, id: u64) -> PathBuf {
    root.join(format!("request-{id:020}"))
}

pub fn startup_package_dir(root: &Path) -> PathBuf {
    root.join("startup")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenConfig {
    pub seed: u64,
    pub preset: V3Preset,
    pub extent: u32,
    pub chamfer: bool,
    pub arch_type: ArchType,
    pub stairs: bool,
}

impl GenConfig {
    pub fn default_config() -> Self {
        Self {
            seed: 42,
            preset: V3Preset::Sparse,
            extent: 2048,
            chamfer: true,
            arch_type: ArchType::Pointed,
            stairs: true,
        }
    }

    pub fn to_v3_config(&self) -> Result<V3Config, bsp_generator::enhanced_v3::V3Error> {
        let mut config = V3Config::new(self.seed, self.preset, self.extent)?;
        config.chamfer = self.chamfer;
        config.arch_type = self.arch_type;
        config.stairs = self.stairs;
        config.validate()?;
        Ok(config)
    }

    pub fn increment_seed(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(1);
        self.seed
    }

    pub fn cycle_preset(&mut self) {
        self.preset = match self.preset {
            V3Preset::Sparse => V3Preset::Moderate,
            V3Preset::Moderate => V3Preset::Rich,
            V3Preset::Rich => V3Preset::Sparse,
        };
        self.extent = if self.preset == V3Preset::Rich {
            3072
        } else {
            2048
        };
    }

    pub fn cycle_arch_type(&mut self) {
        self.arch_type = self.arch_type.cycle();
    }

    pub fn describe(&self) -> String {
        format!(
            "seed={} preset={} extent={} chamfer={} arch={} stairs={}",
            self.seed,
            self.preset.tag(),
            self.extent,
            self.chamfer,
            self.arch_type.tag(),
            self.stairs,
        )
    }
}

#[derive(Debug, Clone)]
pub struct GenRequest {
    pub id: u64,
    pub config: GenConfig,
    pub tools_dir: PathBuf,
    pub package_dir: PathBuf,
}

#[derive(Debug)]
pub struct GenResult {
    pub id: u64,
    pub config: GenConfig,
    pub package_dir: PathBuf,
    pub success: bool,
    pub error: Option<String>,
}

struct GenWorkerShared {
    pending: Mutex<Option<GenRequest>>,
    completed: Mutex<VecDeque<GenResult>>,
    quit: AtomicBool,
    next_id: AtomicU64,
}

pub struct GenWorker {
    shared: Arc<GenWorkerShared>,
    thread: Option<JoinHandle<()>>,
}

impl GenWorker {
    pub fn spawn() -> Self {
        let shared = Arc::new(GenWorkerShared {
            pending: Mutex::new(None),
            completed: Mutex::new(VecDeque::new()),
            quit: AtomicBool::new(false),
            next_id: AtomicU64::new(0),
        });
        let thread_shared = Arc::clone(&shared);
        let thread = thread::Builder::new()
            .name("bsp-gen-worker".into())
            .spawn(move || gen_worker_loop(thread_shared))
            .expect("spawn generation worker thread");
        Self {
            shared,
            thread: Some(thread),
        }
    }

    /// Coalesce queued work, while leaving an already-building request alone.
    /// Each assigned ID receives an independent package directory.
    pub fn enqueue(&self, config: GenConfig, tools_dir: PathBuf, package_root: &Path) -> u64 {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let request = GenRequest {
            id,
            config,
            tools_dir,
            package_dir: package_dir_for_request(package_root, id),
        };
        let mut pending = self
            .shared
            .pending
            .lock()
            .expect("generation queue poisoned");
        coalesce_pending(&mut pending, request);
        id
    }

    pub fn poll_result(&self) -> Option<GenResult> {
        self.shared
            .completed
            .lock()
            .expect("generation result poisoned")
            .pop_front()
    }

    pub fn shutdown(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        self.shared.quit.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for GenWorker {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

/// Replace queued work with the most recent request. In-progress work is not
/// represented by this slot and is therefore never interrupted unsafely.
fn coalesce_pending(slot: &mut Option<GenRequest>, request: GenRequest) {
    *slot = Some(request);
}

fn gen_worker_loop(shared: Arc<GenWorkerShared>) {
    loop {
        let request = shared
            .pending
            .lock()
            .expect("generation queue poisoned")
            .take();
        if let Some(request) = request {
            let result = process_request(&request);
            // Preserve every completion until the UI takes custody. The UI
            // compares IDs before publication and removes stale successful
            // package directories; overwriting this queue would leak one.
            shared
                .completed
                .lock()
                .expect("generation result poisoned")
                .push_back(result);
            continue;
        }
        if shared.quit.load(Ordering::Relaxed) {
            return;
        }
        thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn process_request(request: &GenRequest) -> GenResult {
    let failure = |message: String| GenResult {
        id: request.id,
        config: request.config.clone(),
        package_dir: request.package_dir.clone(),
        success: false,
        error: Some(message),
    };
    let config = match request.config.to_v3_config() {
        Ok(config) => config,
        Err(error) => return failure(format!("invalid config: {error}")),
    };
    if !tools_available(&request.tools_dir) {
        return failure(format!(
            "ericw-tools not found in {}",
            request.tools_dir.display()
        ));
    }
    if let Some(parent) = request.package_dir.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            return failure(format!("create package root: {error}"));
        }
    }
    match engine_pack::enhanced_dungeon_v3::build_v3_package_from_config(
        &config,
        &request.package_dir,
        Some(&request.tools_dir),
        "bsp_beta_gen",
        None,
    ) {
        Ok(_) => GenResult {
            id: request.id,
            config: request.config.clone(),
            package_dir: request.package_dir.clone(),
            success: true,
            error: None,
        },
        Err(error) => failure(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_directories_are_distinct_and_confined() {
        let root = PathBuf::from("/tmp/bsp-beta-packages");
        let first = package_dir_for_request(&root, 1);
        let second = package_dir_for_request(&root, 2);
        assert_ne!(first, second);
        assert_eq!(first.parent(), Some(root.as_path()));
        assert_eq!(startup_package_dir(&root).parent(), Some(root.as_path()));
    }

    #[test]
    fn default_config_and_hotkey_cycles_are_valid() {
        let mut config = GenConfig::default_config();
        assert!(config.to_v3_config().is_ok());
        assert_eq!(config.increment_seed(), 43);
        config.cycle_preset();
        assert_eq!(config.preset, V3Preset::Moderate);
        config.cycle_preset();
        assert_eq!(config.extent, 3072);
        config.cycle_arch_type();
        assert_eq!(config.arch_type, ArchType::Segmented);
    }

    #[test]
    fn invalid_explicit_tools_are_not_silently_fallback() {
        let path = Path::new("/definitely/not/ericw");
        assert_eq!(
            discover_ericw_tools(Some(path)).unwrap_err(),
            ToolDiscoveryError::ExplicitInvalid(path.to_path_buf())
        );
    }

    #[cfg(unix)]
    #[test]
    fn tool_discovery_rejects_non_executable_files() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir().unwrap();
        for tool in ["qbsp", "vis", "light"] {
            let path = root.path().join(tool);
            std::fs::write(&path, "not executable").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        }

        assert!(!tools_available(root.path()));
        assert_eq!(
            discover_ericw_tools(Some(root.path())).unwrap_err(),
            ToolDiscoveryError::ExplicitInvalid(root.path().to_path_buf())
        );
    }

    #[test]
    fn completed_results_retain_every_request_in_order() {
        let mut completed = VecDeque::new();
        completed.push_back(GenResult {
            id: 1,
            config: GenConfig::default_config(),
            package_dir: PathBuf::from("/tmp/first"),
            success: true,
            error: None,
        });
        completed.push_back(GenResult {
            id: 2,
            config: GenConfig::default_config(),
            package_dir: PathBuf::from("/tmp/second"),
            success: false,
            error: Some("failed".into()),
        });
        assert_eq!(completed.pop_front().map(|result| result.id), Some(1));
        assert_eq!(completed.pop_front().map(|result| result.id), Some(2));
    }

    #[test]
    fn queued_requests_coalesce_to_latest_without_timing() {
        let root = PathBuf::from("/tmp/bsp-beta-packages");
        let first = GenRequest {
            id: 1,
            config: GenConfig::default_config(),
            tools_dir: PathBuf::from("/missing"),
            package_dir: package_dir_for_request(&root, 1),
        };
        let second = GenRequest {
            id: 2,
            config: GenConfig::default_config(),
            tools_dir: PathBuf::from("/missing"),
            package_dir: package_dir_for_request(&root, 2),
        };
        let mut queued = Some(first);
        coalesce_pending(&mut queued, second);
        let queued = queued.unwrap();
        assert_eq!(queued.id, 2);
        assert_eq!(queued.package_dir, package_dir_for_request(&root, 2));
    }
}
