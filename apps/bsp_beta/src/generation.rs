//! EnhancedV3 background generation for the BSP beta explorer.
//!
//! Compilation is deliberately isolated from the event/render thread. Every
//! request owns a distinct package target because engine_pack publication is
//! atomic and no-replace; a later request must never collide with an earlier
//! package or remove one still being imported.

use bsp_generator::enhanced_v3::{ArchType, FeatureFlags, GrammarMode, V3Config, V3Preset};
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

/// Explorer configuration holding every public V3Config knob.
///
/// Optional fields use `None`-means-preset/default semantics matching
/// [`V3Config`]. The config is a draft that can be edited incrementally;
/// call [`to_v3_config`] to produce a validated immutable config.
#[derive(Debug, Clone)]
pub struct GenConfig {
    pub seed: u64,
    pub preset: V3Preset,
    pub extent: u32,
    pub rooms: Option<u32>,
    pub corridors: Option<u32>,
    pub loops: Option<u32>,
    pub vertical_edges: Option<u32>,
    pub chamfer: bool,
    pub arch_type: ArchType,
    pub stairs: bool,
    pub room_span_min: Option<u32>,
    pub room_span_max: Option<u32>,
    pub grammar_families: Vec<String>,
    pub grammar_mode: GrammarMode,
    pub features: FeatureFlags,
    pub feature_density: f32,
    pub minlight: u32,
    pub light_count: Option<u32>,
}

impl PartialEq for GenConfig {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed
            && self.preset == other.preset
            && self.extent == other.extent
            && self.rooms == other.rooms
            && self.corridors == other.corridors
            && self.loops == other.loops
            && self.vertical_edges == other.vertical_edges
            && self.chamfer == other.chamfer
            && self.arch_type == other.arch_type
            && self.stairs == other.stairs
            && self.room_span_min == other.room_span_min
            && self.room_span_max == other.room_span_max
            && self.grammar_families == other.grammar_families
            && self.grammar_mode == other.grammar_mode
            && self.features == other.features
            && self.feature_density.to_bits() == other.feature_density.to_bits()
            && self.minlight == other.minlight
            && self.light_count == other.light_count
    }
}

impl Eq for GenConfig {}

impl GenConfig {
    // ── Preset accessors ──────────────────────────────────────────

    /// Effective room count: explicit override or preset default.
    pub fn effective_rooms(&self) -> u32 {
        self.rooms.unwrap_or_else(|| self.preset.min_rooms())
    }

    /// Effective loop count.
    pub fn effective_loops(&self) -> u32 {
        self.loops.unwrap_or_else(|| self.preset.target_loops())
    }

    /// Effective route count (rooms - 2 + loops).
    pub fn effective_routes(&self) -> u32 {
        self.effective_rooms() - 2 + self.effective_loops()
    }

    /// Effective corridor count.
    pub fn effective_corridors(&self) -> u32 {
        self.corridors.unwrap_or_else(|| self.effective_routes())
    }

    /// Effective vertical edges.
    pub fn effective_vertical_edges(&self) -> u32 {
        if self.stairs {
            self.vertical_edges.unwrap_or(1)
        } else {
            0
        }
    }

    /// Effective room span minimum.
    pub fn effective_room_span_min(&self) -> u32 {
        self.room_span_min
            .unwrap_or(bsp_generator::enhanced_v3::config::DEFAULT_ROOM_SPAN_MIN)
    }

    /// Effective room span maximum.
    pub fn effective_room_span_max(&self) -> u32 {
        self.room_span_max
            .unwrap_or(bsp_generator::enhanced_v3::config::DEFAULT_ROOM_SPAN_MAX)
    }

    /// Effective light count.
    pub fn effective_light_count(&self) -> u32 {
        self.light_count.unwrap_or_else(|| self.effective_rooms())
    }

    // ── Construction ──────────────────────────────────────────────

    pub fn default_config() -> Self {
        Self {
            seed: 42,
            preset: V3Preset::Sparse,
            extent: 2048,
            rooms: None,
            corridors: None,
            loops: None,
            vertical_edges: None,
            chamfer: true,
            arch_type: ArchType::Pointed,
            stairs: true,
            room_span_min: None,
            room_span_max: None,
            grammar_families: Vec::new(),
            grammar_mode: GrammarMode::Mixed,
            features: FeatureFlags::ALL,
            feature_density: 0.5,
            minlight: 16,
            light_count: None,
        }
    }

    /// Convert to an immutable validated [`V3Config`].
    ///
    /// Every field is forwarded; optional fields preserve `None` semantics.
    pub fn to_v3_config(&self) -> Result<V3Config, bsp_generator::enhanced_v3::V3Error> {
        let mut config = V3Config::new(self.seed, self.preset, self.extent)?;
        config.rooms = self.rooms;
        config.corridors = self.corridors;
        config.loops = self.loops;
        config.vertical_edges = self.vertical_edges;
        config.chamfer = self.chamfer;
        config.arch_type = self.arch_type;
        config.stairs = self.stairs;
        config.room_span_min = self.room_span_min;
        config.room_span_max = self.room_span_max;
        config.grammar_families = self.grammar_families.clone();
        config.grammar_mode = self.grammar_mode;
        config.features = self.features;
        config.feature_density = self.feature_density;
        config.minlight = self.minlight;
        config.light_count = self.light_count;
        config.validate()?;
        Ok(config)
    }

    /// Return `true` when the draft passes V3 validation without errors.
    pub fn is_valid(&self) -> bool {
        self.to_v3_config().is_ok()
    }

    /// Normalize dependent fields so the draft stays buildable after an
    /// upstream change (e.g. rooms/loops changed, corridors must fit).
    pub fn normalize(&mut self) {
        let rooms = self.effective_rooms();
        let loops = self.effective_loops();
        let routes = rooms.saturating_sub(2).saturating_add(loops);

        // Clamp corridors to the valid range [routes, routes*3].
        if let Some(ref mut corridors) = self.corridors {
            let min = routes;
            let max = routes.saturating_mul(3).max(min);
            if *corridors < min {
                *corridors = min;
            } else if *corridors > max {
                *corridors = max;
            }
        }

        let lower = rooms.div_ceil(2);
        let upper = rooms / 2;
        let max_vert = lower
            .min(upper)
            .min(bsp_generator::enhanced_v3::config::VERTICAL_EDGE_MAX);
        if let Some(ref mut vert) = self.vertical_edges {
            if self.stairs {
                if *vert > max_vert {
                    *vert = max_vert;
                }
            } else {
                *vert = 0;
            }
        }

        // Clamp light_count to [0, rooms].
        if let Some(ref mut lc) = self.light_count {
            if *lc > rooms {
                *lc = rooms;
            }
        }

        // Clamp room spans to extent.
        let extent = self.extent;
        if let Some(ref mut span) = self.room_span_min {
            if *span > extent {
                *span = extent;
            }
            // Keep quantum alignment.
            let q = bsp_generator::enhanced_v3::config::CONSTRUCTION_QUANTUM as u32;
            if *span % q != 0 {
                *span = (*span / q) * q;
            }
        }
        if let Some(ref mut span) = self.room_span_max {
            if *span > extent {
                *span = extent;
            }
            let q = bsp_generator::enhanced_v3::config::CONSTRUCTION_QUANTUM as u32;
            if *span % q != 0 {
                *span = (*span / q) * q;
            }
        }
        // Ensure span_min <= span_max.
        if let (Some(ref mut min), Some(ref mut max)) =
            (self.room_span_min.as_mut(), self.room_span_max.as_mut())
        {
            if *min > *max {
                *min = *max;
            }
        }
    }

    /// Reset to the default V3 production configuration.
    pub fn reset_defaults(&mut self) {
        *self = Self::default_config();
    }

    /// Randomize every explorer category using OS entropy.
    ///
    /// Values are assembled before replacing the draft, so an entropy failure
    /// never leaves a partly-randomized configuration behind.
    pub fn randomize_all(&mut self) -> Result<(), getrandom::Error> {
        self.randomize_with(|| {
            let mut bytes = [0u8; 8];
            getrandom::getrandom(&mut bytes)?;
            Ok(u64::from_le_bytes(bytes))
        })
    }

    /// Deterministic/injectable randomization core used by tests and UI error
    /// handling. Every public V3 knob receives a concrete valid value; no
    /// category is quietly reset to `None` and presented as randomized.
    pub(crate) fn randomize_with<E>(
        &mut self,
        mut next: impl FnMut() -> Result<u64, E>,
    ) -> Result<(), E> {
        use bsp_generator::enhanced_v3::config::GRAMMAR_FAMILIES;

        let pick = |value: u64, count: u64| value % count;
        let preset = match pick(next()?, 3) {
            0 => V3Preset::Sparse,
            1 => V3Preset::Moderate,
            _ => V3Preset::Rich,
        };
        let seed = next()?;
        // These known production extents retain the frozen placement headroom.
        let extent = if preset == V3Preset::Rich { 3072 } else { 2048 };
        let rooms = preset.min_rooms();
        let loops = preset.target_loops();
        let corridors = rooms - 2 + loops;
        let stairs = pick(next()?, 2) == 0;
        let vertical_edges = if stairs { 1 } else { 0 };
        let chamfer = pick(next()?, 2) == 0;
        let arch_type = match pick(next()?, 3) {
            0 => ArchType::None,
            1 => ArchType::Pointed,
            _ => ArchType::Segmented,
        };
        let grammar_mode = if pick(next()?, 2) == 0 {
            GrammarMode::Single
        } else {
            GrammarMode::Mixed
        };

        // Choose a non-empty feature subset, then derive a canonical explicit
        // grammar allowlist from it. Terraced shrines are intentionally always
        // eligible, matching V3Config's feature mapping. This randomizes both
        // categories without ever naming a family whose required feature was
        // disabled.
        let feature_bits = (next()? as u32 & FeatureFlags::ALL.bits()).max(1);
        let mut features = FeatureFlags::empty();
        for flag in [
            FeatureFlags::PILLARS,
            FeatureFlags::BUTTRESSES,
            FeatureFlags::BLADES,
            FeatureFlags::VAULT_RIBS,
            FeatureFlags::MONOLITHS,
        ] {
            if feature_bits & flag.bits() != 0 {
                features |= flag;
            }
        }
        let grammar_families = GRAMMAR_FAMILIES
            .iter()
            .filter(|family| features.enables_family(family))
            .map(|tag| (*tag).to_owned())
            .collect();
        let feature_density = 0.50 + pick(next()?, 51) as f32 / 100.0;
        let minlight = pick(next()?, 128) as u32;
        let light_count = pick(next()?, rooms as u64 + 1) as u32;

        let randomized = Self {
            seed,
            preset,
            extent,
            rooms: Some(rooms),
            corridors: Some(corridors),
            loops: Some(loops),
            vertical_edges: Some(vertical_edges),
            chamfer,
            arch_type,
            stairs,
            room_span_min: Some(112),
            room_span_max: Some(256),
            grammar_families,
            grammar_mode,
            features,
            feature_density,
            minlight,
            light_count: Some(light_count),
        };
        // Validation is deterministic and the chosen ranges are frozen V3
        // ranges. Keep this assertion local to protect this constructor from
        // future accidental invalid combinations without widening its error
        // surface to callers.
        debug_assert!(randomized.is_valid());
        *self = randomized;
        Ok(())
    }

    // ── Hotkey helpers (retained from original API) ───────────────

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
            "seed={} preset={} extent={} rooms={} loops={} chamfer={} arch={} stairs={} density={:.2} minlight={}",
            self.seed,
            self.preset.tag(),
            self.extent,
            self.effective_rooms(),
            self.effective_loops(),
            self.chamfer,
            self.arch_type.tag(),
            self.stairs,
            self.feature_density,
            self.minlight,
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

    // ── Existing tests ────────────────────────────────────────────

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

    // ── Expanded GenConfig tests ───────────────────────────────────

    #[test]
    fn genconfig_all_fields_roundtrip_to_v3config() {
        let config = GenConfig {
            seed: 99,
            preset: V3Preset::Moderate,
            extent: 2048,
            rooms: Some(20),
            corridors: Some(22),
            loops: Some(2),
            vertical_edges: Some(1),
            chamfer: true,
            arch_type: ArchType::Segmented,
            stairs: true,
            room_span_min: Some(128),
            room_span_max: Some(240),
            grammar_families: vec!["portal-chamber".into(), "buttressed-hall".into()],
            grammar_mode: GrammarMode::Single,
            features: FeatureFlags::BLADES | FeatureFlags::BUTTRESSES,
            feature_density: 0.75,
            minlight: 32,
            light_count: Some(20),
        };
        let v3 = config.to_v3_config().expect("valid config should convert");
        assert_eq!(v3.seed, 99);
        assert_eq!(v3.preset, V3Preset::Moderate);
        assert_eq!(v3.xy_extent, 2048);
        assert_eq!(v3.rooms, Some(20));
        assert_eq!(v3.corridors, Some(22));
        assert_eq!(v3.loops, Some(2));
        assert_eq!(v3.vertical_edges, Some(1));
        assert!(v3.chamfer);
        assert_eq!(v3.arch_type, ArchType::Segmented);
        assert!(v3.stairs);
        assert_eq!(v3.room_span_min, Some(128));
        assert_eq!(v3.room_span_max, Some(240));
        assert_eq!(
            v3.grammar_families,
            vec!["portal-chamber", "buttressed-hall"]
        );
        assert_eq!(v3.grammar_mode, GrammarMode::Single);
        assert_eq!(v3.features, FeatureFlags::BLADES | FeatureFlags::BUTTRESSES);
        assert_eq!(v3.feature_density, 0.75);
        assert_eq!(v3.minlight, 32);
        assert_eq!(v3.light_count, Some(20));
    }

    #[test]
    fn genconfig_none_fields_preserve_default_semantics() {
        let config = GenConfig::default_config();
        let v3 = config
            .to_v3_config()
            .expect("default config should be valid");
        // All optional fields should be None.
        assert_eq!(v3.rooms, None);
        assert_eq!(v3.corridors, None);
        assert_eq!(v3.loops, None);
        assert_eq!(v3.vertical_edges, None);
        assert_eq!(v3.room_span_min, None);
        assert_eq!(v3.room_span_max, None);
        assert!(v3.grammar_families.is_empty());
        assert_eq!(v3.light_count, None);
    }

    #[test]
    fn genconfig_effective_accessors() {
        let config = GenConfig::default_config();
        // Sparse preset: 12 rooms, 0 loops.
        assert_eq!(config.effective_rooms(), 12);
        assert_eq!(config.effective_loops(), 0);
        assert_eq!(config.effective_routes(), 10); // 12 - 2 + 0
        assert_eq!(config.effective_corridors(), 10);
        assert_eq!(config.effective_vertical_edges(), 1);
        assert_eq!(config.effective_room_span_min(), 112);
        assert_eq!(config.effective_room_span_max(), 256);
        assert_eq!(config.effective_light_count(), 12);
    }

    #[test]
    fn genconfig_explicit_overrides_change_effective_values() {
        let mut config = GenConfig::default_config();
        config.rooms = Some(30);
        config.loops = Some(4);
        config.vertical_edges = Some(2);
        config.light_count = Some(15);
        assert_eq!(config.effective_rooms(), 30);
        assert_eq!(config.effective_loops(), 4);
        assert_eq!(config.effective_vertical_edges(), 2);
        assert_eq!(config.effective_light_count(), 15);
    }

    #[test]
    fn genconfig_stairs_disabled_forces_zero_vertical_edges() {
        let mut config = GenConfig::default_config();
        config.stairs = false;
        config.vertical_edges = Some(3);
        assert_eq!(config.effective_vertical_edges(), 0);
    }

    #[test]
    fn genconfig_normalize_clamps_corridors_to_route_range() {
        let mut config = GenConfig::default_config();
        config.rooms = Some(12);
        config.loops = Some(0);
        config.corridors = Some(100); // routes * 3 = 30, so 100 is too high.
        config.normalize();
        assert_eq!(config.corridors, Some(30));

        config.corridors = Some(3); // below routes = 10
        config.normalize();
        assert_eq!(config.corridors, Some(10));
    }

    #[test]
    fn genconfig_normalize_clamps_vertical_edges() {
        let mut config = GenConfig::default_config();
        config.rooms = Some(6); // 3 lower, 3 upper → max vertical = 3, but VERTICAL_EDGE_MAX=3
        config.vertical_edges = Some(5);
        config.stairs = true;
        config.normalize();
        assert_eq!(config.vertical_edges, Some(3));

        // stairs disabled
        config.stairs = false;
        config.normalize();
        assert_eq!(config.vertical_edges, Some(0));
    }

    #[test]
    fn genconfig_to_v3_config_rejects_invalid() {
        let config = GenConfig {
            extent: 500, // below XY_MIN=1024, not quantum-aligned
            ..GenConfig::default_config()
        };
        assert!(config.to_v3_config().is_err());
    }

    #[test]
    fn genconfig_is_valid_detects_invalid() {
        let valid = GenConfig::default_config();
        assert!(valid.is_valid());

        let invalid = GenConfig {
            extent: 500,
            ..GenConfig::default_config()
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn genconfig_reset_defaults() {
        let mut config = GenConfig {
            seed: 99,
            preset: V3Preset::Rich,
            extent: 3072,
            chamfer: false,
            ..GenConfig::default_config()
        };
        config.reset_defaults();
        assert_eq!(config, GenConfig::default_config());
    }

    #[test]
    fn deterministic_randomization_is_valid_explicit_and_atomic_on_entropy_error() {
        let mut config = GenConfig::default_config();
        let values = [2, 255, 0, 1, 2, 1, 31, 50, 32, 7];
        let mut values = values.into_iter();
        config
            .randomize_with(|| Ok::<_, &'static str>(values.next().unwrap()))
            .unwrap();
        assert!(config.is_valid());
        assert!(config.rooms.is_some() && config.corridors.is_some() && config.loops.is_some());
        assert!(config.vertical_edges.is_some());
        assert!(config.room_span_min.is_some() && config.room_span_max.is_some());
        assert_eq!(config.grammar_families.len(), 6);
        assert_eq!(config.features, FeatureFlags::ALL);
        assert!(config.light_count.is_some());

        let before = config.clone();
        assert_eq!(
            config.randomize_with(|| Err::<u64, _>("entropy unavailable")),
            Err("entropy unavailable")
        );
        assert_eq!(
            config, before,
            "failed entropy must not partially mutate the draft"
        );
    }

    #[test]
    fn genconfig_describe_includes_all_fields() {
        let config = GenConfig::default_config();
        let desc = config.describe();
        assert!(desc.contains("seed=42"));
        assert!(desc.contains("preset=sparse"));
        assert!(desc.contains("extent=2048"));
        assert!(desc.contains("chamfer=true"));
        assert!(desc.contains("arch=pointed"));
        assert!(desc.contains("stairs=true"));
    }

    #[test]
    fn genconfig_equality_uses_f32_bits() {
        let a = GenConfig {
            feature_density: 0.5,
            ..GenConfig::default_config()
        };
        let b = GenConfig {
            feature_density: (-0.0f32), // negative zero — different bit pattern
            ..GenConfig::default_config()
        };
        assert_ne!(a, b);

        let c = GenConfig {
            feature_density: 0.5,
            ..GenConfig::default_config()
        };
        assert_eq!(a, c);
    }
}
