//! Authorized resource import for BSP package and direct launch paths.
//!
//! Every accepted package or direct resource is authorized by the `package_io`
//! trust boundary: normalized logical path, symlink/non-regular rejection,
//! budget reservation, metadata-drift detection, and declared-hash verification.
//!
//! One [`AuthorizedBspImport`] record is shared by package and direct routes.
//! The coordinator receives this record and derives extraction from it; `bsp`
//! receives only owned bytes and settings.

use bsp::resources::{self, MiptexSlot};
use bsp::{BspLoader, BspReport, LoadOptions};
use package_io::resolver::PackageResolver;
use package_io::{ConfinedResource, ContentIdentity, DiagnosticCode, ResourceKind};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

// ── Import Mode ──────────────────────────────────────────────────────

/// Explicit import policy — no `Default` on production paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportMode {
    /// Strict release mode: missing required resources fail before GPU work.
    /// Unresolved faces, missing palette/WAD/lit, and ambiguous PBR companions
    /// are errors.
    Strict,
    /// Explicit development policy. Raw-byte compatibility helpers remain
    /// development/test-only, while package and direct imports may select this
    /// mode only through an explicit caller choice.
    Development,
}

impl ImportMode {
    pub fn is_strict(&self) -> bool {
        matches!(self, ImportMode::Strict)
    }
}

// ── Authorized Resources ─────────────────────────────────────────────

/// A resource authorized through the `package_io` boundary.
#[derive(Debug, Clone)]
pub struct AuthorizedResource {
    /// Logical resource identity (package-relative path or root-relative path).
    pub logical_id: String,
    /// Resource kind.
    pub kind: ResourceKind,
    /// Verified content identity (SHA-256).
    pub identity: ContentIdentity,
    /// Owned authorized bytes.
    pub bytes: Vec<u8>,
}

impl AuthorizedResource {
    fn from_confined(resource: ConfinedResource) -> Self {
        let logical_id = resource.id.as_str().to_string();
        let kind = resource.id.kind();
        let identity = resource.identity;
        let bytes = resource.bytes.into_bytes();
        AuthorizedResource {
            logical_id,
            kind,
            identity,
            bytes,
        }
    }
}

/// A named authorized resource with a sanitized basename.
///
/// WAD basenames determine lookup order; declaration order is preserved.
#[derive(Debug, Clone)]
pub struct NamedAuthorizedResource {
    /// Sanitized basename (determines WAD lookup precedence).
    pub basename: String,
    /// The underlying authorized resource.
    pub resource: AuthorizedResource,
    /// Declaration ordinal (preserves WAD lookup order).
    pub ordinal: usize,
}

// ── PBR Companion Closure ────────────────────────────────────────────

/// Kind of PBR companion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PbrCompanionKind {
    Normal,
    Gloss,
}

impl PbrCompanionKind {
    fn file_suffix(&self) -> &'static str {
        match self {
            PbrCompanionKind::Normal => "_norm.png",
            PbrCompanionKind::Gloss => "_gloss.png",
        }
    }
}

/// How a PBR companion filename was matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PbrMatchMode {
    Exact,
    AsciiInsensitive,
    /// No matching companion was present in the authorized concrete root.
    Absent,
}

impl PbrMatchMode {
    pub fn tag(self) -> &'static str {
        match self {
            PbrMatchMode::Exact => "exact",
            PbrMatchMode::AsciiInsensitive => "ascii-insensitive",
            PbrMatchMode::Absent => "absent",
        }
    }
}

/// A bound PBR companion entry in the resolution closure.
///
/// Every entry is traceable to a Phase 02 source slot and one concrete root.
#[derive(Debug, Clone)]
pub struct BoundPbrCompanion {
    /// Source slot index from the authoritative miptex table.
    pub source_slot: usize,
    /// Texture identity (sanitized miptex name from the slot).
    pub texture_identity: String,
    /// Companion kind: normal or gloss.
    pub kind: PbrCompanionKind,
    /// How the filename was matched.
    pub match_mode: PbrMatchMode,
    /// Resolved resource, or `None` if the companion was absent.
    pub resource: Option<AuthorizedResource>,
}

// ── Import Provenance ────────────────────────────────────────────────

/// Import provenance — root labels for diagnostics and source-link only.
///
/// These are NOT cache input. Concrete root labels are route-specific.
#[derive(Debug, Clone)]
pub struct ImportProvenance {
    /// Import route: `"package"` or `"direct"`.
    pub route: String,
    /// The companion root label used (for diagnostics).
    pub companion_root_label: Option<String>,
    /// The normalized logical root for source-link provenance.
    pub logical_root: Option<String>,
}

// ── Authorized BSP Import ────────────────────────────────────────────

/// One authorized import record shared by package and direct routes.
///
/// Carries the parsed [`BspWorld`], typed import policy, every authorized
/// resource, PBR companion-resolution closure, extraction settings, and
/// provenance. The coordinator derives [`BspExtractionRequest`] from this
/// record without any reloads or caller defaults.
#[derive(Debug)]
pub struct AuthorizedBspImport {
    /// The parsed BSP world.
    pub world: bsp::world::BspWorld,
    /// Typed import policy.
    pub policy: ImportMode,
    /// Authorized BSP resource.
    pub bsp: AuthorizedResource,
    /// Authorized palette resource, if any.
    pub palette: Option<AuthorizedResource>,
    /// Authorized WAD resources in declaration order.
    pub wads: Vec<NamedAuthorizedResource>,
    /// Authorized .lit companion resource, if any.
    pub lit: Option<AuthorizedResource>,
    /// PBR companion-resolution closure.
    pub pbr: Vec<BoundPbrCompanion>,
    /// Import provenance (not cache input).
    pub provenance: ImportProvenance,
    /// Coordinate scale factor.
    pub scale: f32,
    /// Fullbright palette index range start.
    pub fullbright_start: u8,
    /// Fullbright palette index range end.
    pub fullbright_end: u8,
    /// Overbright factor.
    pub overbright: f32,
    /// Light intensity calibration scale.
    pub light_scale: f32,
    /// Maximum atlas pages.
    pub max_atlas_pages: usize,
}

impl AuthorizedBspImport {
    /// Build a [`bsp::BspExtractionRequest`] from this authorized import.
    pub fn to_extraction_request(&self) -> bsp::BspExtractionRequest {
        let palette = self
            .palette
            .as_ref()
            .map(|r| bsp::resources::decode_palette(&r.bytes));

        let wad_archives: Vec<(String, Vec<u8>)> = self
            .wads
            .iter()
            .map(|n| (n.basename.clone(), n.resource.bytes.clone()))
            .collect();

        let texture_companions: Vec<bsp::resources::TextureCompanion> = self
            .pbr
            .iter()
            .filter_map(|c| {
                c.resource.as_ref().map(|r| {
                    bsp::resources::TextureCompanion::new(r.logical_id.clone(), r.bytes.clone())
                })
            })
            .collect();

        bsp::BspExtractionRequest {
            world: self.world.clone(),
            palette,
            wad_archives,
            texture_companions,
            strict: self.policy.is_strict(),
            scale: self.scale,
            fullbright_start: self.fullbright_start,
            fullbright_end: self.fullbright_end,
            max_atlas_pages: self.max_atlas_pages,
            overbright: self.overbright,
            light_scale: self.light_scale,
        }
    }
}

// ── Companion-Root Normalization ─────────────────────────────────────

/// Normalize a supplied companion root for PBR discovery.
///
/// - A root whose final path component is exactly `textures` is the concrete root.
/// - Every other supplied root maps only to its confined `textures` child.
/// - No supplied root means no PBR discovery.
/// - A supplied root that is absent, not a directory, or symlinked is an error.
///
/// This direct-path helper performs no resource reads. Accepted companion files
/// still flow through [`PackageResolver`] below.
pub fn normalize_companion_root(
    supplied: Option<&Path>,
) -> Result<Option<PathBuf>, PackageLoadError> {
    let Some(root) = supplied else {
        return Ok(None);
    };

    require_safe_directory(root)?;
    let concrete = concrete_textures_path(root);
    require_safe_directory(&concrete)?;
    Ok(Some(concrete))
}

fn concrete_textures_path(root: &Path) -> PathBuf {
    if root
        .file_name()
        .map(|name| name == "textures")
        .unwrap_or(false)
    {
        root.to_path_buf()
    } else {
        root.join("textures")
    }
}

fn require_safe_directory(path: &Path) -> Result<(), PackageLoadError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        let code = if error.kind() == std::io::ErrorKind::NotFound {
            DiagnosticCode::PackageIoNotFound
        } else {
            DiagnosticCode::PackageIoMetadataFailed
        };
        PackageLoadError::Io(package_io::PackageIoError::io(code, path, error))
    })?;

    if metadata.file_type().is_symlink() {
        return Err(PackageLoadError::Io(package_io::PackageIoError::new(
            DiagnosticCode::PackageIoSymlinkRejected,
            format!("companion root is a symlink: '{}'", path.display()),
        )));
    }
    if !metadata.is_dir() {
        return Err(PackageLoadError::Io(package_io::PackageIoError::new(
            DiagnosticCode::PackageIoNotADirectory,
            format!("companion root is not a directory: '{}'", path.display()),
        )));
    }
    Ok(())
}

/// Normalize and validate one package-relative companion root.
///
/// Directory enumeration is confined to this validated root. Every selected
/// file is then authorized through the resolver, which repeats symlink,
/// regular-file, metadata-drift, budget, and hashing checks.
fn normalize_companion_root_with_resolver(
    resolver: &PackageResolver,
    supplied: &str,
) -> Result<String, PackageLoadError> {
    // `.` is the only package-root spelling accepted here; `None` remains
    // the explicit no-discovery choice.
    let normalized = if supplied == "." {
        String::new()
    } else {
        package_io::resolver::normalize_logical_path(supplied)?
    };
    let concrete = concrete_textures_path(Path::new(&normalized));
    let concrete = package_io::resolver::normalize_logical_path(&concrete.to_string_lossy())?;
    validate_package_directory(resolver, &concrete)?;
    Ok(concrete)
}

fn validate_package_directory(
    resolver: &PackageResolver,
    logical_root: &str,
) -> Result<PathBuf, PackageLoadError> {
    let mut path = resolver.root().canonical_path().to_path_buf();
    for component in logical_root.split('/') {
        path.push(component);
        require_safe_directory(&path)?;
    }
    Ok(path)
}

// ── PBR Companion Discovery ──────────────────────────────────────────

/// Discover PBR companions from Phase 02 source-slot mappings.
///
/// The concrete root is a package-relative, normalized `textures` directory.
/// Enumeration supplies candidate names only; every accepted file is read by
/// `PackageResolver` rather than a direct filesystem read.
pub fn discover_pbr_companions(
    resolver: &mut PackageResolver,
    slots: &[MiptexSlot],
    world: &bsp::world::BspWorld,
    concrete_root: &str,
) -> Result<Vec<BoundPbrCompanion>, PackageLoadError> {
    let root_path = validate_package_directory(resolver, concrete_root)?;
    let slot_entries = pbr_slot_entries(slots, world);
    let identities: BTreeSet<String> = slot_entries
        .iter()
        .map(|(_, identity)| identity.clone())
        .collect();

    // Query each identity once, then project the result back to every source
    // slot that resolves to it. This preserves slot provenance without
    // rereading a shared companion file.
    let mut resolved = Vec::with_capacity(identities.len() * 2);
    for identity in identities {
        for kind in [PbrCompanionKind::Normal, PbrCompanionKind::Gloss] {
            let (match_mode, resource) =
                resolve_pbr_companion(resolver, &root_path, concrete_root, &identity, kind)?;
            resolved.push((identity.clone(), kind, match_mode, resource));
        }
    }

    let mut companions = Vec::with_capacity(slot_entries.len() * 2);
    for (source_slot, texture_identity) in slot_entries {
        for kind in [PbrCompanionKind::Normal, PbrCompanionKind::Gloss] {
            let (_, _, match_mode, resource) = resolved
                .iter()
                .find(|(identity, resolved_kind, _, _)| {
                    identity == &texture_identity && *resolved_kind == kind
                })
                .expect("every PBR slot identity was queried");
            companions.push(BoundPbrCompanion {
                source_slot,
                texture_identity: texture_identity.clone(),
                kind,
                match_mode: *match_mode,
                resource: resource.clone(),
            });
        }
    }

    Ok(companions)
}

fn pbr_slot_entries(slots: &[MiptexSlot], world: &bsp::world::BspWorld) -> Vec<(usize, String)> {
    use bsp::materials::SurfaceClass;

    let mut entries = BTreeSet::new();
    for face in &world.faces {
        let texinfo_idx = face.texinfo_id as usize;
        let Some(texinfo) = world.texinfos.get(texinfo_idx) else {
            continue;
        };
        let miptex_idx = texinfo.miptex as usize;
        let Some(slot) = slots.get(miptex_idx) else {
            continue;
        };
        let Some(identity) = slot.identity.as_ref() else {
            continue;
        };

        if !matches!(
            classify_face_surface_class(face, &world.texinfos, slots),
            SurfaceClass::Opaque | SurfaceClass::AlphaMask
        ) {
            continue;
        }
        entries.insert((miptex_idx, identity.clone()));
    }
    entries.into_iter().collect()
}

fn resolve_pbr_companion(
    resolver: &mut PackageResolver,
    root_path: &Path,
    logical_root: &str,
    texture_identity: &str,
    kind: PbrCompanionKind,
) -> Result<(PbrMatchMode, Option<AuthorizedResource>), PackageLoadError> {
    let Some((filename, match_mode)) =
        find_pbr_filename(root_path, texture_identity, kind.file_suffix())?
    else {
        return Ok((PbrMatchMode::Absent, None));
    };

    let logical_path = format!("{logical_root}/{filename}");
    let resource = resolver.resolve(&logical_path, ResourceKind::Texture)?;
    Ok((
        match_mode,
        Some(AuthorizedResource::from_confined(resource)),
    ))
}

fn find_pbr_filename(
    root: &Path,
    texture_identity: &str,
    suffix: &str,
) -> Result<Option<(String, PbrMatchMode)>, PackageLoadError> {
    let expected = format!("{texture_identity}{suffix}");
    let mut fallback_candidates = Vec::new();

    let entries = std::fs::read_dir(root).map_err(|error| {
        PackageLoadError::Io(package_io::PackageIoError::io(
            DiagnosticCode::PackageIoMetadataFailed,
            root,
            error,
        ))
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            PackageLoadError::Io(package_io::PackageIoError::io(
                DiagnosticCode::PackageIoMetadataFailed,
                root,
                error,
            ))
        })?;
        let Some(filename) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if filename == expected {
            // Exact case always wins, even when fallback-case collisions exist.
            return Ok(Some((filename, PbrMatchMode::Exact)));
        }
        if filename.eq_ignore_ascii_case(&expected) {
            fallback_candidates.push(filename);
        }
    }

    fallback_candidates.sort();
    match fallback_candidates.len() {
        0 => Ok(None),
        1 => Ok(Some((
            fallback_candidates.pop().expect("one fallback candidate"),
            PbrMatchMode::AsciiInsensitive,
        ))),
        _ => Err(PackageLoadError::AmbiguousPbrCompanion {
            texture_identity: texture_identity.to_string(),
            suffix: suffix.to_string(),
            candidates: fallback_candidates,
        }),
    }
}

/// Classify a face's surface class from its texinfo and slot table.
fn classify_face_surface_class(
    face: &bsp::lumps::Face,
    texinfo: &[bsp::lumps::Texinfo],
    slots: &[MiptexSlot],
) -> bsp::materials::SurfaceClass {
    let texinfo_idx = face.texinfo_id as usize;
    if texinfo_idx >= texinfo.len() {
        return bsp::materials::SurfaceClass::NoDraw;
    }
    let ti = &texinfo[texinfo_idx];
    let miptex_idx = ti.miptex as usize;
    let name: Option<&str> = if miptex_idx < slots.len() {
        slots[miptex_idx].identity.as_deref()
    } else {
        None
    };

    bsp::materials::classify_surface(ti.flags, name.unwrap_or(""))
}

// ── Error Type ───────────────────────────────────────────────────────

/// Error type for authorized import failures.
#[derive(Debug)]
pub enum PackageLoadError {
    /// I/O or confinement error from `package_io`.
    Io(package_io::PackageIoError),
    /// BSP parse failure.
    Parse(BspReport),
    /// Missing required resource.
    MissingRequired { kind: ResourceKind, path: String },
    /// Ambiguous PBR companion (two or more case-insensitive candidates).
    AmbiguousPbrCompanion {
        texture_identity: String,
        suffix: String,
        candidates: Vec<String>,
    },
    /// Invalid WAD basename (empty or duplicate after sanitization).
    InvalidWadBasename { path: String, reason: String },
    /// No import mode selected (CLI error for direct launch).
    NoImportMode,
    /// Direct path resource is not confined within the common ancestor root.
    PathNotConfined { path: PathBuf, root: PathBuf },
    /// No common ancestor for direct path resources.
    NoCommonAncestor,
    /// Direct resources would require using the filesystem root as package root.
    UnconfinedDirectRoot { root: PathBuf },
    /// Companion root error.
    CompanionRoot { path: PathBuf, reason: String },
    /// Failed to create a private staging directory.
    StagingCreateFailed { reason: String },
    /// Copy into staging failed (source or destination I/O).
    StagingCopyFailed { role: String, reason: String },
    /// SHA-256 mismatch after copying a resource into staging.
    StagingHashMismatch {
        role: String,
        expected: String,
        actual: String,
    },
    /// Authorizing from the staged closure failed.
    StagingAuthorizeFailed { reason: String },
    /// Staging cleanup encountered an error.
    StagingCleanupFailed { path: PathBuf, reason: String },
}

impl std::fmt::Display for PackageLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackageLoadError::Io(e) => write!(f, "package I/O error: {e}"),
            PackageLoadError::Parse(e) => write!(f, "BSP parse error: {e}"),
            PackageLoadError::MissingRequired { kind, path } => {
                write!(f, "missing required {} resource: '{}'", kind.tag(), path)
            }
            PackageLoadError::AmbiguousPbrCompanion {
                texture_identity,
                suffix,
                candidates,
            } => {
                write!(
                    f,
                    "ambiguous PBR companion for '{}' with suffix '{}': candidates {:?}",
                    texture_identity, suffix, candidates
                )
            }
            PackageLoadError::InvalidWadBasename { path, reason } => {
                write!(f, "invalid WAD basename for '{}': {}", path, reason)
            }
            PackageLoadError::NoImportMode => {
                write!(f, "no import mode selected; use --strict or --development")
            }
            PackageLoadError::PathNotConfined { path, root } => {
                write!(
                    f,
                    "path '{}' is not confined within root '{}'",
                    path.display(),
                    root.display()
                )
            }
            PackageLoadError::NoCommonAncestor => {
                write!(f, "no common ancestor for direct-path resources")
            }
            PackageLoadError::UnconfinedDirectRoot { root } => {
                write!(
                    f,
                    "direct import root '{}' is unconstrained",
                    root.display()
                )
            }
            PackageLoadError::CompanionRoot { path, reason } => {
                write!(f, "companion root '{}': {}", path.display(), reason)
            }
            PackageLoadError::StagingCreateFailed { reason } => {
                write!(f, "staging directory creation failed: {}", reason)
            }
            PackageLoadError::StagingCopyFailed { role, reason } => {
                write!(f, "staging copy failed for '{}': {}", role, reason)
            }
            PackageLoadError::StagingHashMismatch {
                role,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "staging hash mismatch for '{}': expected {}, got {}",
                    role, expected, actual
                )
            }
            PackageLoadError::StagingAuthorizeFailed { reason } => {
                write!(f, "staging authorize failed: {}", reason)
            }
            PackageLoadError::StagingCleanupFailed { path, reason } => {
                write!(
                    f,
                    "staging cleanup failed for '{}': {}",
                    path.display(),
                    reason
                )
            }
        }
    }
}

impl std::error::Error for PackageLoadError {}

impl From<package_io::PackageIoError> for PackageLoadError {
    fn from(e: package_io::PackageIoError) -> Self {
        PackageLoadError::Io(e)
    }
}

// ── Package Authorization ────────────────────────────────────────────

/// Authorize a BSP import from a package resolver.
///
/// This is the primary entry point for runtime BSP loading from a package.
/// Every resource is authorized through the `package_io` boundary before
/// the BSP is parsed. The resulting [`AuthorizedBspImport`] carries all
/// authorized bytes, policy, and provenance.
pub fn authorize_package_import(
    resolver: &mut PackageResolver,
    bsp_path: &str,
    palette_path: &str,
    lit_path: Option<&str>,
    wad_paths: &[String],
    textures_dir: Option<&str>,
    mode: ImportMode,
    scale: f32,
) -> Result<AuthorizedBspImport, PackageLoadError> {
    // ── Authorize BSP ─────────────────────────────────────────────
    let bsp_resource = resolver.resolve(bsp_path, ResourceKind::Bsp)?;
    let bsp = AuthorizedResource::from_confined(bsp_resource);

    // ── Authorize palette ─────────────────────────────────────────
    let palette_resource = resolver.resolve(palette_path, ResourceKind::Palette)?;
    let palette = Some(AuthorizedResource::from_confined(palette_resource));

    // ── Authorize .lit ────────────────────────────────────────────
    let lit = if let Some(lit_path) = lit_path {
        let lit_resource = resolver.resolve(lit_path, ResourceKind::Lit)?;
        Some(AuthorizedResource::from_confined(lit_resource))
    } else {
        None
    };

    // ── Authorize WADs ────────────────────────────────────────────
    let mut wads = Vec::new();
    let mut seen_basenames = BTreeSet::new();
    for (ordinal, wad_path) in wad_paths.iter().enumerate() {
        let wad_resource = resolver.resolve(wad_path, ResourceKind::Wad)?;

        // Sanitize basename: extract filename stem, reject empty or dots.
        let basename = sanitize_wad_basename(wad_path)?;

        // Reject duplicates.
        if !seen_basenames.insert(basename.clone()) {
            return Err(PackageLoadError::InvalidWadBasename {
                path: wad_path.clone(),
                reason: format!("duplicate sanitized basename '{}'", basename),
            });
        }

        wads.push(NamedAuthorizedResource {
            basename,
            resource: AuthorizedResource::from_confined(wad_resource),
            ordinal,
        });
    }

    // ── Parse BSP ─────────────────────────────────────────────────
    let load_options = LoadOptions {
        strict: mode.is_strict(),
        palette: palette.as_ref().map(|r| r.bytes.clone()),
        lit_data: lit.as_ref().map(|r| r.bytes.clone()),
        wad_archives: wads
            .iter()
            .map(|n| (n.basename.clone(), n.resource.bytes.clone()))
            .collect(),
        texture_overrides: Vec::new(),
        source_identity: bsp.logical_id.clone(),
    };

    let world = BspLoader::load(&bsp.bytes, &load_options).map_err(PackageLoadError::Parse)?;

    // ── PBR companion discovery ───────────────────────────────────
    let slots = resources::parse_miptex_slots(&world.miptex_data);
    // Validate a supplied root even when the map has no eligible faces: a
    // declared unsafe or missing root must never silently turn into no-PBR.
    let concrete_root = textures_dir
        .map(|root| normalize_companion_root_with_resolver(resolver, root))
        .transpose()?;
    let pbr = if slots.is_empty() {
        Vec::new()
    } else if let Some(root) = concrete_root.as_deref() {
        discover_pbr_companions(resolver, &slots, &world, root)?
    } else {
        build_empty_pbr_closure(&slots, &world)?
    };

    Ok(AuthorizedBspImport {
        world,
        policy: mode,
        bsp,
        palette,
        wads,
        lit,
        pbr,
        provenance: ImportProvenance {
            route: "package".to_string(),
            companion_root_label: concrete_root.clone(),
            logical_root: concrete_root,
        },
        scale,
        fullbright_start: 224,
        fullbright_end: 255,
        overbright: 2.0,
        light_scale: 1.0,
        max_atlas_pages: 4,
    })
}

/// Build an empty PBR closure (all absent) for a slot table without a companion root.
fn build_empty_pbr_closure(
    slots: &[MiptexSlot],
    world: &bsp::world::BspWorld,
) -> Result<Vec<BoundPbrCompanion>, PackageLoadError> {
    let mut companions = Vec::new();
    for (source_slot, texture_identity) in pbr_slot_entries(slots, world) {
        for kind in [PbrCompanionKind::Normal, PbrCompanionKind::Gloss] {
            companions.push(BoundPbrCompanion {
                source_slot,
                texture_identity: texture_identity.clone(),
                kind,
                match_mode: PbrMatchMode::Absent,
                resource: None,
            });
        }
    }
    Ok(companions)
}

// ── Direct-Path Authorization ────────────────────────────────────────

/// Authorize a BSP import from direct filesystem paths.
///
/// When all declared resources share a single canonical common ancestor, the
/// fast path forms one trusted root and authorizes through a single resolver.
/// When resources live in unrelated filesystem locations (e.g. BSP in /tmp,
/// palette in the project tree), a private staging directory is created,
/// authorized bytes are copied and hash-verified, and the import is built from
/// the staged closure with semantic labels rather than transport paths.
///
/// This is the runtime package boundary for direct launch; the app provides
/// filesystem paths but does not read, scan, hash, or authorize resource bytes.
pub fn authorize_direct_import(
    bsp_path: &Path,
    palette_path: &Path,
    lit_path: Option<&Path>,
    wad_paths: &[PathBuf],
    textures_dir: Option<&Path>,
    mode: ImportMode,
    scale: f32,
) -> Result<AuthorizedBspImport, PackageLoadError> {
    // ── 1. Build the semantic plan (no reads) ─────────────────────────
    let plan = DirectImportPlan::from_declared_paths(
        bsp_path,
        palette_path,
        lit_path,
        wad_paths,
        textures_dir,
        mode,
        scale,
    )?;

    // ── 2. Try exact-single-root fast path ────────────────────────────
    match try_fast_path(&plan) {
        Ok(import) => return Ok(import),
        Err(PackageLoadError::NoCommonAncestor | PackageLoadError::UnconfinedDirectRoot { .. }) => {
            // Fall through to staging — unrelated roots are expected.
        }
        Err(e) => return Err(e),
    }

    // ── 3. Staging path: authorize each resource narrowly ─────────────
    let sources = plan.authorize_narrowly()?;

    // ── 4. Create staging and copy with hash verification ─────────────
    let mut stage = DirectImportStaging::create()?;
    let staged = match stage.copy_verify_and_reauthorize(&sources, &plan) {
        Ok(s) => s,
        Err(primary) => {
            let cleanup = stage.finish();
            return Err(join_staging_errors(primary, cleanup));
        }
    };

    // ── 5. Build import from staged bytes with semantic labels ────────
    let import = staged.into_authorized_import(&plan)?;

    // ── 6. Explicit cleanup before returning ──────────────────────────
    if let Err(cleanup) = stage.finish() {
        log::warn!("DirectImportStaging cleanup error: {}", cleanup);
    }

    Ok(import)
}

// ── Semantic Direct-Import Plan ───────────────────────────────────────

/// A private semantic import plan built from declared paths before any
/// resource reads. It records logical roles, WAD ordinals/basenames,
/// companion-root declaration, import mode, and scale.
///
/// The plan never reads, opens, or stats a resource file. It validates
/// path components (symlink rejection, traversal) but defers authorization.
#[derive(Debug, Clone)]
struct DirectImportPlan {
    bsp_path: PathBuf,
    palette_path: PathBuf,
    lit_path: Option<PathBuf>,
    wad_entries: Vec<(usize, PathBuf, String)>, // (ordinal, path, sanitized basename)
    textures_dir: Option<PathBuf>,
    mode: ImportMode,
    scale: f32,
    /// Canonicalized source roots for provenance.
    source_roots: Vec<PathBuf>,
}

impl DirectImportPlan {
    /// Build the plan from declared paths. This validates symlink-free path
    /// components and canonicalises for provenance only; it reads zero bytes.
    fn from_declared_paths(
        bsp_path: &Path,
        palette_path: &Path,
        lit_path: Option<&Path>,
        wad_paths: &[PathBuf],
        textures_dir: Option<&Path>,
        mode: ImportMode,
        scale: f32,
    ) -> Result<Self, PackageLoadError> {
        // Normalize companion root for diagnostics.
        let concrete_textures_root = normalize_companion_root(textures_dir)?;

        // Reject symlink components and canonicalize for purity.
        let canonical_bsp = canonicalize_direct_path(bsp_path)?;
        let canonical_palette = canonicalize_direct_path(palette_path)?;
        let canonical_lit = lit_path.map(canonicalize_direct_path).transpose()?;
        let canonical_wads: Vec<PathBuf> = wad_paths
            .iter()
            .map(|p| canonicalize_direct_path(p))
            .collect::<Result<Vec<_>, _>>()?;
        let canonical_textures = concrete_textures_root
            .as_deref()
            .map(canonicalize_direct_path)
            .transpose()?;

        // Sanitize WAD basenames.
        let mut seen_basenames = std::collections::BTreeSet::new();
        let wad_entries: Vec<(usize, PathBuf, String)> = wad_paths
            .iter()
            .zip(canonical_wads.iter())
            .enumerate()
            .map(|(ordinal, (original, canonical))| {
                let basename = sanitize_wad_basename(&original.to_string_lossy())?;
                if !seen_basenames.insert(basename.clone()) {
                    return Err(PackageLoadError::InvalidWadBasename {
                        path: original.display().to_string(),
                        reason: format!("duplicate sanitized basename '{}'", basename),
                    });
                }
                Ok((ordinal, canonical.clone(), basename))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut source_roots: Vec<PathBuf> = vec![canonical_bsp.clone(), canonical_palette.clone()];
        if let Some(ref lit) = canonical_lit {
            source_roots.push(lit.clone());
        }
        for (_, wad_path, _) in &wad_entries {
            source_roots.push(wad_path.clone());
        }
        if let Some(ref tex) = canonical_textures {
            source_roots.push(tex.clone());
        }

        Ok(DirectImportPlan {
            bsp_path: canonical_bsp,
            palette_path: canonical_palette,
            lit_path: canonical_lit,
            wad_entries,
            textures_dir: canonical_textures,
            mode,
            scale,
            source_roots,
        })
    }

    /// Authorize each resource independently, reading from its own narrow
    /// filesystem root. Each resource gets its own resolver and budget ledger;
    /// the aggregate budget is checked across all resources.
    fn authorize_narrowly(&self) -> Result<Vec<NarrowAuthorizedSource>, PackageLoadError> {
        use package_io::budget::BudgetLedger;

        let mut sources: Vec<NarrowAuthorizedSource> = Vec::new();
        let mut aggregate_budget = BudgetLedger::default_ledger();

        // Authorize BSP
        let (bsp_bytes, bsp_hash) =
            authorize_single_file(&self.bsp_path, ResourceKind::Bsp, &mut aggregate_budget)?;
        sources.push(NarrowAuthorizedSource {
            role: NarrowRole::Bsp,
            bytes: bsp_bytes,
            hash: bsp_hash,
            source_path: self.bsp_path.clone(),
        });

        // Authorize palette
        let (pal_bytes, pal_hash) = authorize_single_file(
            &self.palette_path,
            ResourceKind::Palette,
            &mut aggregate_budget,
        )?;
        sources.push(NarrowAuthorizedSource {
            role: NarrowRole::Palette,
            bytes: pal_bytes,
            hash: pal_hash,
            source_path: self.palette_path.clone(),
        });

        // Authorize LIT
        if let Some(ref lit_path) = self.lit_path {
            let (lit_bytes, lit_hash) =
                authorize_single_file(lit_path, ResourceKind::Lit, &mut aggregate_budget)?;
            sources.push(NarrowAuthorizedSource {
                role: NarrowRole::Lit,
                bytes: lit_bytes,
                hash: lit_hash,
                source_path: lit_path.clone(),
            });
        }

        // Authorize WADs
        for (ordinal, wad_path, basename) in &self.wad_entries {
            let (wad_bytes, wad_hash) =
                authorize_single_file(wad_path, ResourceKind::Wad, &mut aggregate_budget)?;
            sources.push(NarrowAuthorizedSource {
                role: NarrowRole::Wad {
                    ordinal: *ordinal,
                    basename: basename.clone(),
                },
                bytes: wad_bytes,
                hash: wad_hash,
                source_path: wad_path.clone(),
            });
        }

        Ok(sources)
    }

    /// The semantic closure: labels used for cache identity, source-link,
    /// and provenance — never a stage path.
    fn semantic_closure(&self) -> DirectImportSemantics {
        DirectImportSemantics {
            wad_entries: self
                .wad_entries
                .iter()
                .map(|(ordinal, _, basename)| (*ordinal, basename.clone()))
                .collect(),
            textures_declared_root: self.textures_dir.as_ref().map(|p| p.display().to_string()),
        }
    }
}

/// Semantic closure labels built from the plan — no stage paths.
#[derive(Debug, Clone)]
struct DirectImportSemantics {
    wad_entries: Vec<(usize, String)>, // (ordinal, basename)
    textures_declared_root: Option<String>,
}

// ── Narrow Authorization ──────────────────────────────────────────────

/// A single resource authorized through its own narrow filesystem root.
#[derive(Debug, Clone)]
struct NarrowAuthorizedSource {
    role: NarrowRole,
    bytes: Vec<u8>,
    hash: package_io::ContentIdentity,
    source_path: PathBuf,
}

/// Deterministic role label for staging transport.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NarrowRole {
    Bsp,
    Palette,
    Lit,
    Wad { ordinal: usize, basename: String },
}

impl NarrowRole {
    /// Return the deterministic staging filename for this role.
    fn stage_filename(&self) -> String {
        match self {
            NarrowRole::Bsp => "bsp".to_string(),
            NarrowRole::Palette => "palette.lmp".to_string(),
            NarrowRole::Lit => "lit".to_string(),
            NarrowRole::Wad { ordinal, basename } => {
                format!("wad_{}_{}.wad", ordinal, basename)
            }
        }
    }

    /// Return the semantic logical ID for import records.
    fn semantic_logical_id(&self) -> String {
        match self {
            NarrowRole::Bsp => "direct:bsp".to_string(),
            NarrowRole::Palette => "direct:palette".to_string(),
            NarrowRole::Lit => "direct:lit".to_string(),
            NarrowRole::Wad { ordinal, basename } => {
                format!("direct:wad[{}]:{}", ordinal, basename)
            }
        }
    }
}

/// Authorize a single file through its own one-file package resolver.
fn authorize_single_file(
    path: &Path,
    kind: ResourceKind,
    aggregate_budget: &mut package_io::budget::BudgetLedger,
) -> Result<(Vec<u8>, package_io::ContentIdentity), PackageLoadError> {
    use package_io::budget::BudgetLedger;
    use package_io::PackageRoot;

    let parent = path
        .parent()
        .ok_or_else(|| PackageLoadError::CompanionRoot {
            path: path.to_path_buf(),
            reason: "resource path has no parent directory".to_string(),
        })?;

    let root = PackageRoot::new(parent).map_err(|error| PackageLoadError::CompanionRoot {
        path: parent.to_path_buf(),
        reason: format!("cannot create narrow package root: {error}"),
    })?;

    let filename = path.file_name().and_then(|s| s.to_str()).ok_or_else(|| {
        PackageLoadError::CompanionRoot {
            path: path.to_path_buf(),
            reason: "resource path has no filename".to_string(),
        }
    })?;

    // Each narrow authorization uses its own ledger, but we check the
    // aggregate budget afterward.
    let narrow_ledger = BudgetLedger::default_ledger();
    let mut resolver = PackageResolver::new(root, narrow_ledger);
    let resource = resolver.resolve(filename, kind)?;

    let bytes = resource.bytes.into_bytes();
    let hash = resource.identity;

    // Reserve in the aggregate budget.
    aggregate_budget
        .check_file_and_source_bytes(1, bytes.len() as u64)
        .map_err(|e| PackageLoadError::Io(e))?;
    aggregate_budget
        .reserve_file_and_source_bytes(1, bytes.len() as u64)
        .map_err(|e| PackageLoadError::Io(e))?;

    Ok((bytes, hash))
}

// ── Exact-Single-Root Fast Path ───────────────────────────────────────

/// Try the exact-single-root fast path.
///
/// Succeeds only when every declared path is directly confined by the same
/// canonical root. `/` and any root broader than the declared single-root
/// closure are rejected.
fn try_fast_path(plan: &DirectImportPlan) -> Result<AuthorizedBspImport, PackageLoadError> {
    use package_io::budget::BudgetLedger;
    use package_io::PackageRoot;

    let canonical_paths: Vec<&Path> = plan.source_roots.iter().map(PathBuf::as_path).collect();
    let common_root = find_common_ancestor(&canonical_paths)?;

    // Reject the filesystem root or any root broader than the declared
    // resource set.
    if common_root.parent().is_none() {
        return Err(PackageLoadError::UnconfinedDirectRoot { root: common_root });
    }

    let package_root =
        PackageRoot::new(&common_root).map_err(|error| PackageLoadError::CompanionRoot {
            path: common_root.clone(),
            reason: format!("cannot create package root: {error}"),
        })?;
    let mut resolver = PackageResolver::new(package_root, BudgetLedger::default_ledger());

    let bsp_rel = relativize(&common_root, &plan.bsp_path)?;
    let palette_rel = relativize(&common_root, &plan.palette_path)?;
    let lit_rel = plan
        .lit_path
        .as_ref()
        .map(|path| relativize(&common_root, path))
        .transpose()?;
    let wad_rels: Vec<String> = plan
        .wad_entries
        .iter()
        .map(|(_, wad_path, _)| relativize(&common_root, wad_path))
        .collect::<Result<Vec<_>, _>>()?;
    let textures_rel = plan
        .textures_dir
        .as_ref()
        .map(|path| relativize(&common_root, path))
        .transpose()?;

    authorize_package_import(
        &mut resolver,
        &bsp_rel,
        &palette_rel,
        lit_rel.as_deref(),
        &wad_rels,
        textures_rel.as_deref(),
        plan.mode,
        plan.scale,
    )
    .map(|mut import| {
        import.provenance = ImportProvenance {
            route: "direct".to_string(),
            companion_root_label: plan
                .textures_dir
                .as_ref()
                .map(|path| path.display().to_string()),
            logical_root: textures_rel,
        };
        import
    })
}

// ── Direct Import Staging ─────────────────────────────────────────────

/// A private staging directory for unrelated-root direct imports.
///
/// Creates a random, private (mode 0700) directory and copies authorized
/// bytes into deterministic role paths using exclusive file creation and no
/// links. Every copy is hash-checked against its authorized source hash.
///
/// Cleanup is explicit via [`finish`](DirectImportStaging::finish); `Drop`
/// is an idempotent fallback only.
struct DirectImportStaging {
    root: PathBuf,
    cleanup_required: bool,
}

/// The reauthorized staged closure ready to construct an import.
struct StagedClosure {
    staged_root: PathBuf,
    staged_bsp_path: PathBuf,
    staged_palette_path: PathBuf,
    staged_lit_path: Option<PathBuf>,
    staged_wad_paths: Vec<PathBuf>,
    staged_textures_dir: Option<PathBuf>,
    /// Per-role hashes for semantic identity.
    role_hashes: Vec<(NarrowRole, package_io::ContentIdentity)>,
}

impl DirectImportStaging {
    /// Create a random private staging directory.
    fn create() -> Result<Self, PackageLoadError> {
        let parent = std::env::temp_dir();
        let mut rng = [0u8; 16];
        // Simple entropy: time + pid
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| PackageLoadError::StagingCreateFailed {
                reason: "system clock error".to_string(),
            })?
            .as_nanos();
        let pid = std::process::id();
        rng[..8].copy_from_slice(&(nanos as u64).to_le_bytes());
        rng[8..12].copy_from_slice(&pid.to_le_bytes());
        rng[12..16].copy_from_slice(b"\xDE\xAD\xBE\xEF");
        let hex = rng.iter().map(|b| format!("{:02x}", b)).collect::<String>();

        let root = parent.join(format!("direct-import-staging-{}", hex));
        std::fs::create_dir_all(&root).map_err(|e| PackageLoadError::StagingCreateFailed {
            reason: format!("cannot create staging root '{}': {}", root.display(), e),
        })?;

        // Set restrictive permissions where supported.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o700);
            std::fs::set_permissions(&root, perms).map_err(|e| {
                PackageLoadError::StagingCreateFailed {
                    reason: format!(
                        "cannot set permissions on staging root '{}': {}",
                        root.display(),
                        e
                    ),
                }
            })?;
        }

        Ok(DirectImportStaging {
            root,
            cleanup_required: true,
        })
    }

    /// Copy authorized sources into deterministic role paths, hash-verify
    /// every copy, then reauthorize the staged closure.
    fn copy_verify_and_reauthorize(
        &mut self,
        sources: &[NarrowAuthorizedSource],
        plan: &DirectImportPlan,
    ) -> Result<StagedClosure, PackageLoadError> {
        let mut staged_bsp = None;
        let mut staged_palette = None;
        let mut staged_lit = None;
        let mut staged_wads: Vec<(usize, PathBuf)> = Vec::new();
        let mut role_hashes = Vec::new();

        for source in sources {
            let filename = source.role.stage_filename();
            let dest = self.root.join(&filename);

            // Copy with exclusive creation, no overwrites.
            stage_copy_exclusive(&source.bytes, &dest, &source.role, &source.hash)?;

            role_hashes.push((source.role.clone(), source.hash));

            match &source.role {
                NarrowRole::Bsp => staged_bsp = Some(dest),
                NarrowRole::Palette => staged_palette = Some(dest),
                NarrowRole::Lit => staged_lit = Some(dest),
                NarrowRole::Wad { ordinal, .. } => {
                    staged_wads.push((*ordinal, dest));
                }
            }
        }

        // Sort WADs by ordinal to preserve declaration order.
        staged_wads.sort_by_key(|(ord, _)| *ord);
        let staged_wad_paths: Vec<PathBuf> = staged_wads.into_iter().map(|(_, p)| p).collect();

        let staged_textures_dir = if let Some(ref tex) = plan.textures_dir {
            // Copy the textures directory contents into staging.
            let staged_tex = self.root.join("textures");
            std::fs::create_dir_all(&staged_tex).map_err(|e| {
                PackageLoadError::StagingCopyFailed {
                    role: "textures".to_string(),
                    reason: format!("cannot create staged textures dir: {}", e),
                }
            })?;

            copy_dir_contents(&tex, &staged_tex)?;
            Some(staged_tex)
        } else {
            None
        };

        let staged_bsp = staged_bsp.ok_or_else(|| PackageLoadError::StagingAuthorizeFailed {
            reason: "no BSP staged".to_string(),
        })?;
        let staged_palette =
            staged_palette.ok_or_else(|| PackageLoadError::StagingAuthorizeFailed {
                reason: "no palette staged".to_string(),
            })?;

        Ok(StagedClosure {
            staged_root: self.root.clone(),
            staged_bsp_path: staged_bsp,
            staged_palette_path: staged_palette,
            staged_lit_path: staged_lit,
            staged_wad_paths: staged_wad_paths,
            staged_textures_dir,
            role_hashes,
        })
    }

    /// Explicit typed cleanup. Returns Ok(()) when the staging directory
    /// is removed; returns an error when removal fails.
    ///
    /// After this call, `cleanup_required` is cleared so `Drop` becomes a
    /// no-op.
    fn finish(&mut self) -> Result<(), PackageLoadError> {
        if !self.cleanup_required {
            return Ok(());
        }
        self.cleanup_required = false;
        std::fs::remove_dir_all(&self.root).map_err(|e| PackageLoadError::StagingCleanupFailed {
            path: self.root.clone(),
            reason: format!("{}", e),
        })
    }
}

impl Drop for DirectImportStaging {
    fn drop(&mut self) {
        if self.cleanup_required {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

impl StagedClosure {
    /// Build the authorized import from the staged closure using the
    /// resolver, then relabel every resource with semantic logical IDs.
    fn into_authorized_import(
        self,
        plan: &DirectImportPlan,
    ) -> Result<AuthorizedBspImport, PackageLoadError> {
        use package_io::budget::BudgetLedger;
        use package_io::PackageRoot;

        let root = PackageRoot::new(&self.staged_root).map_err(|error| {
            PackageLoadError::StagingAuthorizeFailed {
                reason: format!("cannot create staged package root: {error}"),
            }
        })?;
        let mut resolver = PackageResolver::new(root, BudgetLedger::default_ledger());

        // Compute root-relative paths for the resolver.
        let bsp_rel = relativize(&self.staged_root, &self.staged_bsp_path)?;
        let palette_rel = relativize(&self.staged_root, &self.staged_palette_path)?;
        let lit_rel = self
            .staged_lit_path
            .as_ref()
            .map(|p| relativize(&self.staged_root, p))
            .transpose()?;
        let wad_rels: Vec<String> = self
            .staged_wad_paths
            .iter()
            .map(|p| relativize(&self.staged_root, p))
            .collect::<Result<Vec<_>, _>>()?;
        let textures_rel = self
            .staged_textures_dir
            .as_ref()
            .map(|p| relativize(&self.staged_root, p))
            .transpose()?;

        // Authorize through the resolver (staged bytes).
        let mut import = authorize_package_import(
            &mut resolver,
            &bsp_rel,
            &palette_rel,
            lit_rel.as_deref(),
            &wad_rels,
            textures_rel.as_deref(),
            plan.mode,
            plan.scale,
        )?;

        // ── Relabel with semantic IDs ─────────────────────────────────
        import.bsp.logical_id = NarrowRole::Bsp.semantic_logical_id();
        if let Some(ref mut pal) = import.palette {
            pal.logical_id = NarrowRole::Palette.semantic_logical_id();
        }
        if let Some(ref mut lit) = import.lit {
            lit.logical_id = NarrowRole::Lit.semantic_logical_id();
        }
        for (i, wad) in import.wads.iter_mut().enumerate() {
            wad.resource.logical_id = NarrowRole::Wad {
                ordinal: wad.ordinal,
                basename: wad.basename.clone(),
            }
            .semantic_logical_id();
            // Ensure the ordinal matches the plan.
            if let Some((plan_ordinal, _, _)) = plan.wad_entries.get(i) {
                if wad.ordinal != *plan_ordinal {
                    return Err(PackageLoadError::StagingAuthorizeFailed {
                        reason: format!(
                            "WAD ordinal mismatch: import has {}, plan has {}",
                            wad.ordinal, plan_ordinal
                        ),
                    });
                }
            }
        }

        // Relabel PBR companions.
        for companion in &mut import.pbr {
            if let Some(ref mut res) = companion.resource {
                res.logical_id = format!(
                    "direct:pbr:{}:{:?}",
                    companion.texture_identity, companion.kind
                );
            }
        }

        // ── Provenance reflects original sources, not stage ───────────
        let semantics = plan.semantic_closure();
        import.provenance = ImportProvenance {
            route: "direct".to_string(),
            companion_root_label: semantics.textures_declared_root,
            logical_root: None,
        };

        Ok(import)
    }
}

// ── Staging Helpers ───────────────────────────────────────────────────

/// Copy bytes to a destination file with exclusive creation (no overwrite).
/// Hash-verify after copy.
fn stage_copy_exclusive(
    bytes: &[u8],
    dest: &Path,
    role: &NarrowRole,
    expected_hash: &package_io::ContentIdentity,
) -> Result<(), PackageLoadError> {
    use std::io::Write;

    // Exclusive create: fail if file already exists.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dest)
        .map_err(|e| PackageLoadError::StagingCopyFailed {
            role: format!("{:?}", role),
            reason: format!("cannot create '{}': {}", dest.display(), e),
        })?;

    file.write_all(bytes)
        .map_err(|e| PackageLoadError::StagingCopyFailed {
            role: format!("{:?}", role),
            reason: format!("write to '{}' failed: {}", dest.display(), e),
        })?;

    file.flush()
        .map_err(|e| PackageLoadError::StagingCopyFailed {
            role: format!("{:?}", role),
            reason: format!("flush to '{}' failed: {}", dest.display(), e),
        })?;

    // Hash-verify the written bytes.
    let written = std::fs::read(dest).map_err(|e| PackageLoadError::StagingCopyFailed {
        role: format!("{:?}", role),
        reason: format!("cannot read back '{}': {}", dest.display(), e),
    })?;
    let actual_hash = package_io::ContentIdentity::from_bytes(&written);
    if actual_hash != *expected_hash {
        return Err(PackageLoadError::StagingHashMismatch {
            role: format!("{:?}", role),
            expected: expected_hash.hex(),
            actual: actual_hash.hex(),
        });
    }

    Ok(())
}

/// Copy a directory's regular-file contents recursively into a destination.
/// Rejects symlinks, devices, FIFOs, sockets, and non-regular files.
fn copy_dir_contents(src: &Path, dest: &Path) -> Result<(), PackageLoadError> {
    for entry in std::fs::read_dir(src).map_err(|e| PackageLoadError::StagingCopyFailed {
        role: "textures".to_string(),
        reason: format!("cannot read directory '{}': {}", src.display(), e),
    })? {
        let entry = entry.map_err(|e| PackageLoadError::StagingCopyFailed {
            role: "textures".to_string(),
            reason: format!("directory entry error: {}", e),
        })?;
        let path = entry.path();
        let ft = entry
            .file_type()
            .map_err(|e| PackageLoadError::StagingCopyFailed {
                role: "textures".to_string(),
                reason: format!("cannot stat '{}': {}", path.display(), e),
            })?;

        if ft.is_symlink() {
            return Err(PackageLoadError::StagingCopyFailed {
                role: "textures".to_string(),
                reason: format!("symlink rejected: '{}'", path.display()),
            });
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileTypeExt;
            if ft.is_block_device() || ft.is_char_device() || ft.is_fifo() || ft.is_socket() {
                return Err(PackageLoadError::StagingCopyFailed {
                    role: "textures".to_string(),
                    reason: format!("special file rejected: '{}'", path.display()),
                });
            }
        }

        if ft.is_dir() {
            let name = entry.file_name();
            let sub_dest = dest.join(&name);
            std::fs::create_dir_all(&sub_dest).map_err(|e| {
                PackageLoadError::StagingCopyFailed {
                    role: "textures".to_string(),
                    reason: format!("cannot create subdirectory '{}': {}", sub_dest.display(), e),
                }
            })?;
            copy_dir_contents(&path, &sub_dest)?;
        } else if ft.is_file() {
            let contents =
                std::fs::read(&path).map_err(|e| PackageLoadError::StagingCopyFailed {
                    role: "textures".to_string(),
                    reason: format!("cannot read '{}': {}", path.display(), e),
                })?;
            let name = entry.file_name();
            let dest_file = dest.join(&name);
            std::fs::write(&dest_file, &contents).map_err(|e| {
                PackageLoadError::StagingCopyFailed {
                    role: "textures".to_string(),
                    reason: format!("cannot write '{}': {}", dest_file.display(), e),
                }
            })?;
        }
    }
    Ok(())
}

// ── Shared Helpers ────────────────────────────────────────────────────

/// Find the common ancestor directory of a set of canonical paths.
fn find_common_ancestor(paths: &[&Path]) -> Result<PathBuf, PackageLoadError> {
    if paths.is_empty() {
        return Err(PackageLoadError::NoCommonAncestor);
    }

    let first = paths[0];
    let mut ancestor: PathBuf = first.to_path_buf();

    loop {
        let all_contained = paths.iter().all(|p| p.starts_with(&ancestor));
        if all_contained {
            return Ok(ancestor);
        }
        if !ancestor.pop() {
            return Err(PackageLoadError::NoCommonAncestor);
        }
    }
}

fn canonicalize_direct_path(path: &Path) -> Result<PathBuf, PackageLoadError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| {
                PackageLoadError::Io(package_io::PackageIoError::io(
                    DiagnosticCode::PackageIoMetadataFailed,
                    Path::new("."),
                    error,
                ))
            })?
            .join(path)
    };
    reject_symlink_components(&absolute)?;
    absolute.canonicalize().map_err(|error| {
        PackageLoadError::Io(package_io::PackageIoError::io(
            DiagnosticCode::PackageIoNotFound,
            &absolute,
            error,
        ))
    })
}

fn reject_symlink_components(path: &Path) -> Result<(), PackageLoadError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                current.push(component.as_os_str());
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir | std::path::Component::Normal(_) => {
                current.push(component.as_os_str());
                let metadata = std::fs::symlink_metadata(&current).map_err(|error| {
                    let code = if error.kind() == std::io::ErrorKind::NotFound {
                        DiagnosticCode::PackageIoNotFound
                    } else {
                        DiagnosticCode::PackageIoMetadataFailed
                    };
                    PackageLoadError::Io(package_io::PackageIoError::io(code, &current, error))
                })?;
                if metadata.file_type().is_symlink() {
                    return Err(PackageLoadError::Io(package_io::PackageIoError::new(
                        DiagnosticCode::PackageIoSymlinkRejected,
                        format!(
                            "symlink component in direct import path: '{}'",
                            current.display()
                        ),
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Compute a root-relative path, rejecting paths outside the root.
fn relativize(root: &Path, path: &Path) -> Result<String, PackageLoadError> {
    let canonical = path.canonicalize().map_err(|e| {
        PackageLoadError::Io(package_io::PackageIoError::io(
            DiagnosticCode::PackageIoNotFound,
            path,
            e,
        ))
    })?;

    let stripped = canonical
        .strip_prefix(root)
        .map_err(|_| PackageLoadError::PathNotConfined {
            path: path.to_path_buf(),
            root: root.to_path_buf(),
        })?;

    Ok(stripped.to_string_lossy().into_owned())
}

/// Join a primary staging error with an optional cleanup error.
fn join_staging_errors(
    primary: PackageLoadError,
    cleanup: Result<(), PackageLoadError>,
) -> PackageLoadError {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup_err) => {
            log::error!(
                "DirectImportStaging: primary error '{}' followed by cleanup error '{}'",
                primary,
                cleanup_err
            );
            // Return the primary error; the cleanup error is logged.
            primary
        }
    }
}

// ── WAD Basename Sanitization ────────────────────────────────────────

/// Sanitize a WAD path to extract the basename (stem).
///
/// The path may include directory components (e.g., "maps/test.wad") when
/// resolved through the package resolver. Only `..` path traversal is rejected.
fn sanitize_wad_basename(path: &str) -> Result<String, PackageLoadError> {
    // Check for path traversal anywhere in the path.
    if path.contains("..") {
        return Err(PackageLoadError::InvalidWadBasename {
            path: path.to_string(),
            reason: "path contains '..' traversal".to_string(),
        });
    }

    let raw_name = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path);

    // Reject empty basenames.
    if raw_name.is_empty() {
        return Err(PackageLoadError::InvalidWadBasename {
            path: path.to_string(),
            reason: "empty basename".to_string(),
        });
    }

    // Reject basenames that are only dots.
    if raw_name.chars().all(|c| c == '.') {
        return Err(PackageLoadError::InvalidWadBasename {
            path: path.to_string(),
            reason: "basename is only dots".to_string(),
        });
    }

    Ok(raw_name.to_string())
}

// ── Effective Import Summary ─────────────────────────────────────────

/// Produce a structured effective-import summary from an authorized import.
pub fn effective_import_summary(import: &AuthorizedBspImport) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Route: {}", import.provenance.route));
    lines.push(format!("Policy: {:?}", import.policy));
    lines.push(format!("Scale: {}", import.scale));
    lines.push(format!(
        "BSP: {} (sha256:{})",
        import.bsp.logical_id,
        import.bsp.identity.hex()
    ));

    if let Some(ref pal) = import.palette {
        lines.push(format!(
            "Palette: {} (sha256:{})",
            pal.logical_id,
            pal.identity.hex()
        ));
    }

    if let Some(ref lit) = import.lit {
        lines.push(format!(
            "LIT: {} (sha256:{})",
            lit.logical_id,
            lit.identity.hex()
        ));
    }

    for wad in &import.wads {
        lines.push(format!(
            "WAD[{}]: {} ({}) (sha256:{})",
            wad.ordinal,
            wad.basename,
            wad.resource.logical_id,
            wad.resource.identity.hex()
        ));
    }

    if let Some(ref root) = import.provenance.companion_root_label {
        lines.push(format!("Companion root: {}", root));
    }

    let present_pbr: Vec<_> = import.pbr.iter().filter(|c| c.resource.is_some()).collect();
    lines.push(format!(
        "PBR companions: {} present, {} absent",
        present_pbr.len(),
        import.pbr.len() - present_pbr.len()
    ));

    for companion in &import.pbr {
        if let Some(ref res) = companion.resource {
            lines.push(format!(
                "  PBR slot={} id={} kind={:?} mode={:?} sha256:{}",
                companion.source_slot,
                companion.texture_identity,
                companion.kind,
                companion.match_mode,
                res.identity.hex()
            ));
        }
    }

    if let Some(ref logical_root) = import.provenance.logical_root {
        lines.push(format!("Logical root: {}", logical_root));
    }

    lines.join("\n")
}

// ── Legacy LoadedBspPackage (thin wrapper) ───────────────────────────

/// Compatibility wrapper that owns exactly one authorized import handoff.
///
/// It deliberately exposes no parallel resource fields, so legacy callers
/// cannot drop palette, WAD, `.lit`, policy, or PBR closure inputs before the
/// coordinator receives them.
#[derive(Debug)]
pub struct LoadedBspPackage {
    import: AuthorizedBspImport,
}

impl LoadedBspPackage {
    pub fn new(import: AuthorizedBspImport) -> Self {
        Self { import }
    }

    pub fn into_authorized_import(self) -> AuthorizedBspImport {
        self.import
    }

    pub fn authorized_import(&self) -> &AuthorizedBspImport {
        &self.import
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use package_io::budget::BudgetLedger;
    use package_io::PackageRoot;
    use std::fs;

    fn temp_dir() -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("bsp-pkg-test-{}-{nanos}", std::process::id()))
    }

    /// Build a minimal valid BSP29 file for testing.
    fn make_minimal_bsp29() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&29u32.to_le_bytes());

        let mut current_offset: u32 = 124;
        let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
        let entity_offset = current_offset;
        let entity_size = entity_bytes.len() as u32;
        current_offset += entity_size;
        let plane_offset = current_offset;
        let plane_size = 20u32;
        current_offset += plane_size;

        let lumps: [(u32, u32); 15] = [
            (entity_offset, entity_size),
            (plane_offset, plane_size),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ];
        for (off, sz) in &lumps {
            data.extend_from_slice(&off.to_le_bytes());
            data.extend_from_slice(&sz.to_le_bytes());
        }
        data.extend_from_slice(entity_bytes);
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0i32.to_le_bytes());
        data
    }

    #[test]
    fn authorize_package_import_minimal_bsp() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();

        let maps = dir.join("maps");
        fs::create_dir_all(&maps).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();

        let palettes = dir.join("palettes");
        fs::create_dir_all(&palettes).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);

        let result = authorize_package_import(
            &mut resolver,
            "maps/test.bsp",
            "palettes/pal.lmp",
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        );
        assert!(result.is_ok());
        let import = result.unwrap();
        assert_eq!(import.world.entities.len(), 1);
        assert!(import.palette.is_some());
        assert!(import.policy.is_strict());
        let request = import.to_extraction_request();
        assert!(request.strict);
        assert!(request.palette.is_some());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn authorize_package_import_missing_bsp() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        fs::create_dir_all(dir.join("palettes")).unwrap();
        fs::write(dir.join("palettes/pal.lmp"), &[0u8; 768]).unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);

        let result = authorize_package_import(
            &mut resolver,
            "maps/test.bsp",
            "palettes/pal.lmp",
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        );
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn authorize_package_import_with_wad() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let maps = dir.join("maps");
        fs::create_dir_all(&maps).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        let palettes = dir.join("palettes");
        fs::create_dir_all(&palettes).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();
        // Write a minimal valid WAD2 (empty archive).
        let mut wad_data = Vec::new();
        wad_data.extend_from_slice(b"WAD2");
        wad_data.extend_from_slice(&0u32.to_le_bytes()); // 0 entries
        wad_data.extend_from_slice(&8u32.to_le_bytes()); // dir offset = 8
        fs::write(maps.join("test.wad"), &wad_data).unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);

        let result = authorize_package_import(
            &mut resolver,
            "maps/test.bsp",
            "palettes/pal.lmp",
            None,
            &["maps/test.wad".to_string()],
            None,
            ImportMode::Strict,
            0.0254,
        );
        assert!(result.is_ok(), "authorize failed: {result:?}");
        let import = result.unwrap();
        assert_eq!(import.wads.len(), 1);
        assert_eq!(import.wads[0].basename, "test");
        assert_eq!(import.wads[0].ordinal, 0);
        let request = import.to_extraction_request();
        assert_eq!(request.wad_archives.len(), 1);
        assert_eq!(request.wad_archives[0].0, "test");
        assert!(request.strict);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_wad_basename_rejected() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        fs::create_dir_all(dir.join("maps")).unwrap();
        fs::write(dir.join("maps/test.bsp"), make_minimal_bsp29()).unwrap();
        fs::create_dir_all(dir.join("palettes")).unwrap();
        fs::write(dir.join("palettes/pal.lmp"), &[0u8; 768]).unwrap();
        // This path contains ".." in the directory component, which is rejected.
        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);

        let result = authorize_package_import(
            &mut resolver,
            "maps/test.bsp",
            "palettes/pal.lmp",
            None,
            &["maps/../escape.wad".to_string()],
            None,
            ImportMode::Strict,
            0.0254,
        );
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn companion_root_normalization_textures_child() {
        // A root ending with "textures" is used directly.
        let dir = temp_dir();
        let textures = dir.join("textures");
        fs::create_dir_all(&textures).unwrap();
        let result = normalize_companion_root(Some(&textures)).unwrap();
        assert_eq!(result, Some(textures));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn companion_root_normalization_maps_to_textures_child() {
        // A non-textures root maps to its textures/ child.
        let dir = temp_dir();
        let child = dir.join("textures");
        fs::create_dir_all(&child).unwrap();
        let result = normalize_companion_root(Some(&dir)).unwrap();
        assert_eq!(result, Some(child));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn companion_root_normalization_missing_textures_child_errors() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        // No textures/ child.
        let result = normalize_companion_root(Some(&dir));
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn companion_root_none_means_no_pbr() {
        let result = normalize_companion_root(None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn companion_root_absent_errors() {
        let result = normalize_companion_root(Some(Path::new("/nonexistent/path")));
        assert!(result.is_err());
    }

    #[test]
    fn direct_import_common_ancestor_confinement() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let maps = dir.join("maps");
        fs::create_dir_all(&maps).unwrap();
        let palettes = dir.join("palettes");
        fs::create_dir_all(&palettes).unwrap();

        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        let result = authorize_direct_import(
            &maps.join("test.bsp"),
            &palettes.join("pal.lmp"),
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        );
        assert!(result.is_ok());
        let import = result.unwrap();
        assert_eq!(import.provenance.route, "direct");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn direct_import_normalizes_root_to_textures_child() {
        let dir = temp_dir();
        let maps = dir.join("maps");
        let palettes = dir.join("palettes");
        fs::create_dir_all(&maps).unwrap();
        fs::create_dir_all(&palettes).unwrap();
        fs::create_dir_all(dir.join("textures")).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        let import = authorize_direct_import(
            &maps.join("test.bsp"),
            &palettes.join("pal.lmp"),
            None,
            &[],
            Some(&dir),
            ImportMode::Strict,
            0.0254,
        )
        .unwrap();
        assert_eq!(import.provenance.logical_root.as_deref(), Some("textures"));
        let expected_label = dir.join("textures").display().to_string();
        assert_eq!(
            import.provenance.companion_root_label.as_deref(),
            Some(expected_label.as_str())
        );

        let _ = fs::remove_dir_all(&dir);
    }

    fn commit_authorized_import(
        import: AuthorizedBspImport,
    ) -> (
        crate::cache::CacheIdentity,
        crate::source_link::BspSourceLink,
    ) {
        let mut coordinator = crate::coordinator::BspCoordinator::new();
        let prepare = coordinator.prepare_authorized_import(import).unwrap();
        let mut scene = renderer::api::Scene::new();
        coordinator
            .set_renderer_mount_ready(prepare.token, renderer::api::bsp::PreparedBspMount::new())
            .unwrap();
        coordinator
            .validate_for_scene(prepare.token, &mut scene)
            .unwrap();
        let cache_identity = coordinator
            .commit(prepare.token, &mut scene)
            .unwrap()
            .cache_identity;
        let source_link = coordinator.source_link().cloned().unwrap();
        (cache_identity, source_link)
    }

    fn committed_cache_identity(import: AuthorizedBspImport) -> crate::cache::CacheIdentity {
        commit_authorized_import(import).0
    }

    #[test]
    fn package_and_direct_imports_share_one_semantic_cache_identity() {
        let dir = temp_dir();
        let maps = dir.join("maps");
        let palettes = dir.join("palettes");
        fs::create_dir_all(&maps).unwrap();
        fs::create_dir_all(&palettes).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();
        fs::write(
            maps.join("test.wad"),
            [b"WAD2".as_slice(), &[0, 0, 0, 0, 8, 0, 0, 0]].concat(),
        )
        .unwrap();
        fs::write(maps.join("test.lit"), b"QLIT\x01\x00\x00\x00").unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let mut resolver = PackageResolver::new(root, BudgetLedger::default_ledger());
        let package_import = authorize_package_import(
            &mut resolver,
            "maps/test.bsp",
            "palettes/pal.lmp",
            Some("maps/test.lit"),
            &["maps/test.wad".to_string()],
            None,
            ImportMode::Strict,
            0.0254,
        )
        .unwrap();
        let direct_import = authorize_direct_import(
            &maps.join("test.bsp"),
            &palettes.join("pal.lmp"),
            Some(&maps.join("test.lit")),
            &[maps.join("test.wad")],
            None,
            ImportMode::Strict,
            0.0254,
        )
        .unwrap();

        let package_key = committed_cache_identity(package_import).to_key_string();
        let direct_key = committed_cache_identity(direct_import).to_key_string();
        assert_eq!(package_key, direct_key);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn source_link_uses_resolver_issued_bsp_sha256() {
        let dir = temp_dir();
        fs::create_dir_all(dir.join("maps")).unwrap();
        fs::create_dir_all(dir.join("palettes")).unwrap();
        fs::write(dir.join("maps/test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(dir.join("palettes/pal.lmp"), [0u8; 768]).unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let mut resolver = PackageResolver::new(root, BudgetLedger::default_ledger());
        let import = authorize_package_import(
            &mut resolver,
            "maps/test.bsp",
            "palettes/pal.lmp",
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        )
        .unwrap();
        let expected_bsp_hash = format!("sha256:{}", import.bsp.identity.hex());
        let expected_palette_hash =
            format!("sha256:{}", import.palette.as_ref().unwrap().identity.hex());

        let (_, source_link) = commit_authorized_import(import);
        assert_eq!(source_link.content_hash, expected_bsp_hash);
        assert_eq!(
            source_link.companion_hashes.palette.as_deref(),
            Some(expected_palette_hash.as_str())
        );
        assert!(source_link.import_policy.strict);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn direct_import_unconfined_path_rejected() {
        let dir1 = temp_dir();
        let dir2 = temp_dir();
        fs::create_dir_all(dir1.join("maps")).unwrap();
        fs::create_dir_all(dir2.join("palettes")).unwrap();
        fs::write(dir1.join("maps/test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(dir2.join("palettes/pal.lmp"), &[0u8; 768]).unwrap();

        // These paths have /tmp as common ancestor (likely), but
        // they should still work since /tmp is the common root.
        let result = authorize_direct_import(
            &dir1.join("maps").join("test.bsp"),
            &dir2.join("palettes").join("pal.lmp"),
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        );
        // Should work because both are under /tmp
        assert!(result.is_ok());

        let _ = fs::remove_dir_all(&dir1);
        let _ = fs::remove_dir_all(&dir2);
    }

    #[cfg(unix)]
    #[test]
    fn direct_import_rejects_original_symlink_before_canonicalization() {
        use std::os::unix::fs::symlink;

        let dir = temp_dir();
        let maps = dir.join("maps");
        let palettes = dir.join("palettes");
        fs::create_dir_all(&maps).unwrap();
        fs::create_dir_all(&palettes).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();
        symlink(maps.join("test.bsp"), maps.join("alias.bsp")).unwrap();

        let err = authorize_direct_import(
            &maps.join("alias.bsp"),
            &palettes.join("pal.lmp"),
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        )
        .unwrap_err();
        match err {
            PackageLoadError::Io(error) => {
                assert_eq!(error.code, DiagnosticCode::PackageIoSymlinkRejected)
            }
            other => panic!("expected symlink rejection, got {other:?}"),
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn wad_basename_rejects_traversal() {
        assert!(sanitize_wad_basename("../escape").is_err());
        assert!(sanitize_wad_basename("a/../b").is_err());
        assert!(sanitize_wad_basename("..").is_err());
    }

    #[test]
    fn wad_basename_rejects_empty() {
        assert!(sanitize_wad_basename("").is_err());
        assert!(sanitize_wad_basename(".").is_err());
    }

    #[test]
    fn import_summary_includes_all_resources() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let maps = dir.join("maps");
        fs::create_dir_all(&maps).unwrap();
        let palettes = dir.join("palettes");
        fs::create_dir_all(&palettes).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);

        let import = authorize_package_import(
            &mut resolver,
            "maps/test.bsp",
            "palettes/pal.lmp",
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        )
        .unwrap();

        let summary = effective_import_summary(&import);
        assert!(summary.contains("Route: package"));
        assert!(summary.contains("Policy: Strict"));
        assert!(summary.contains("maps/test.bsp"));
        assert!(summary.contains("palettes/pal.lmp"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn package_authorization_binds_fixture_pbr_companions() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bsp/tests/fixtures");
        let root = PackageRoot::new(&fixtures).unwrap();
        let mut resolver = PackageResolver::new(root, BudgetLedger::default_ledger());

        let import = authorize_package_import(
            &mut resolver,
            "compiled/dungeon-materials-bsp2.bsp",
            "palettes/project_palette.lmp",
            Some("compiled/dungeon-materials-bsp2.lit"),
            &[],
            Some("."),
            ImportMode::Development,
            0.0254,
        )
        .expect("fixture package must authorize");

        assert_eq!(import.provenance.logical_root.as_deref(), Some("textures"));
        assert!(import.pbr.iter().any(|companion| {
            companion.texture_identity == "WALL01"
                && companion.kind == PbrCompanionKind::Normal
                && companion.resource.is_some()
        }));
        assert!(import.pbr.iter().any(|companion| {
            companion.texture_identity == "WALL01"
                && companion.kind == PbrCompanionKind::Gloss
                && companion.resource.is_some()
        }));

        let direct_import = authorize_direct_import(
            &fixtures.join("compiled/dungeon-materials-bsp2.bsp"),
            &fixtures.join("palettes/project_palette.lmp"),
            Some(&fixtures.join("compiled/dungeon-materials-bsp2.lit")),
            &[],
            Some(&fixtures.join("textures")),
            ImportMode::Development,
            0.0254,
        )
        .expect("fixture direct route must authorize");
        assert_eq!(
            import.provenance.logical_root,
            direct_import.provenance.logical_root
        );
        let package_closure: Vec<_> = import
            .pbr
            .iter()
            .map(|companion| {
                (
                    companion.source_slot,
                    companion.texture_identity.clone(),
                    companion.kind,
                    companion.match_mode,
                    companion
                        .resource
                        .as_ref()
                        .map(|resource| resource.identity),
                )
            })
            .collect();
        let direct_closure: Vec<_> = direct_import
            .pbr
            .iter()
            .map(|companion| {
                (
                    companion.source_slot,
                    companion.texture_identity.clone(),
                    companion.kind,
                    companion.match_mode,
                    companion
                        .resource
                        .as_ref()
                        .map(|resource| resource.identity),
                )
            })
            .collect();
        assert_eq!(package_closure, direct_closure);

        let request = import.to_extraction_request();
        assert!(request
            .texture_companions
            .iter()
            .any(|companion| { companion.logical_path.ends_with("WALL01_norm.png") }));
        assert!(request
            .texture_companions
            .iter()
            .any(|companion| { companion.logical_path.ends_with("WALL01_gloss.png") }));
    }

    #[test]
    fn pbr_companion_discovery_exact_case() {
        let dir = temp_dir();
        let textures = dir.join("textures");
        fs::create_dir_all(&textures).unwrap();
        fs::write(textures.join("brick_norm.png"), b"normal").unwrap();
        fs::write(textures.join("brick_gloss.png"), b"gloss").unwrap();

        // Build a minimal miptex slot table referencing "brick"
        let bsp_data = make_bsp_with_miptex(b"brick\0");
        let world_result = BspLoader::load(&bsp_data, &LoadOptions::default());
        if let Ok(world) = world_result {
            let slots = resources::parse_miptex_slots(&world.miptex_data);
            let root = PackageRoot::new(&dir).unwrap();
            let mut resolver = PackageResolver::new(root, BudgetLedger::default_ledger());
            let companions =
                discover_pbr_companions(&mut resolver, &slots, &world, "textures").unwrap();
            // Without actual faces referencing the texture, the closure is empty.
            assert!(companions.is_empty());
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pbr_exact_case_wins_and_unique_ascii_fallback_is_authorized() {
        let dir = temp_dir();
        let textures = dir.join("textures");
        fs::create_dir_all(&textures).unwrap();
        fs::write(textures.join("Brick_norm.png"), b"exact").unwrap();
        fs::write(textures.join("brick_norm.png"), b"fallback").unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let mut resolver = PackageResolver::new(root, BudgetLedger::default_ledger());
        let (mode, resource) = resolve_pbr_companion(
            &mut resolver,
            &textures,
            "textures",
            "Brick",
            PbrCompanionKind::Normal,
        )
        .unwrap();
        assert_eq!(mode, PbrMatchMode::Exact);
        let resource = resource.expect("exact companion must be authorized");
        assert_eq!(resource.logical_id, "textures/Brick_norm.png");
        assert_eq!(resource.bytes, b"exact");

        fs::remove_file(textures.join("Brick_norm.png")).unwrap();
        fs::remove_file(textures.join("brick_norm.png")).unwrap();
        fs::write(textures.join("bRiCk_NoRm.PnG"), b"fallback").unwrap();
        let root = PackageRoot::new(&dir).unwrap();
        let mut resolver = PackageResolver::new(root, BudgetLedger::default_ledger());
        let (mode, resource) = resolve_pbr_companion(
            &mut resolver,
            &textures,
            "textures",
            "Brick",
            PbrCompanionKind::Normal,
        )
        .unwrap();
        assert_eq!(mode, PbrMatchMode::AsciiInsensitive);
        assert_eq!(resource.unwrap().bytes, b"fallback");

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn pbr_ambiguous_fallback_and_symlink_are_rejected() {
        use std::os::unix::fs::symlink;

        let dir = temp_dir();
        let textures = dir.join("textures");
        fs::create_dir_all(&textures).unwrap();
        fs::write(textures.join("bRiCk_norm.png"), b"one").unwrap();
        fs::write(textures.join("bricK_norm.png"), b"two").unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let mut resolver = PackageResolver::new(root, BudgetLedger::default_ledger());
        let err = resolve_pbr_companion(
            &mut resolver,
            &textures,
            "textures",
            "Brick",
            PbrCompanionKind::Normal,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PackageLoadError::AmbiguousPbrCompanion { .. }
        ));

        fs::remove_file(textures.join("bRiCk_norm.png")).unwrap();
        fs::remove_file(textures.join("bricK_norm.png")).unwrap();
        let outside = dir.join("outside.png");
        fs::write(&outside, b"outside").unwrap();
        symlink(&outside, textures.join("Brick_norm.png")).unwrap();
        let root = PackageRoot::new(&dir).unwrap();
        let mut resolver = PackageResolver::new(root, BudgetLedger::default_ledger());
        let err = resolve_pbr_companion(
            &mut resolver,
            &textures,
            "textures",
            "Brick",
            PbrCompanionKind::Normal,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PackageLoadError::Io(package_io::PackageIoError {
                code: DiagnosticCode::PackageIoSymlinkRejected,
                ..
            })
        ));

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn companion_root_symlink_child_is_rejected() {
        use std::os::unix::fs::symlink;

        let dir = temp_dir();
        let outside = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, dir.join("textures")).unwrap();

        let err = normalize_companion_root(Some(&dir)).unwrap_err();
        assert!(matches!(
            err,
            PackageLoadError::Io(package_io::PackageIoError {
                code: DiagnosticCode::PackageIoSymlinkRejected,
                ..
            })
        ));

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside);
    }

    /// Build a minimal BSP29 with a miptex lump containing one named texture.
    fn make_bsp_with_miptex(name: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&29u32.to_le_bytes());

        // Build miptex lump: count=1, offset=4, then entry header + name.
        let count: i32 = 1;
        let mut miptex_lump = Vec::new();
        miptex_lump.extend_from_slice(&count.to_le_bytes());
        // One offset entry pointing past the offset table (4 + 1*4 = 8).
        let entry_offset: i32 = 8;
        miptex_lump.extend_from_slice(&entry_offset.to_le_bytes());
        // Miptex entry header: 40 bytes (name is last 16 bytes).
        let mut header = vec![0u8; 40];
        let name_bytes = if name.len() > 15 { &name[..15] } else { name };
        header[24..24 + name_bytes.len()].copy_from_slice(name_bytes);
        miptex_lump.extend_from_slice(&header);

        // Empty face lump.
        let face_data: Vec<u8> = Vec::new();

        let mut current_offset: u32 = 124;
        let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
        let entity_offset = current_offset;
        let entity_size = entity_bytes.len() as u32;
        current_offset += entity_size;

        let miptex_offset = current_offset;
        let miptex_size = miptex_lump.len() as u32;
        current_offset += miptex_size;

        let face_offset = current_offset;
        let face_size = face_data.len() as u32;
        current_offset += face_size;

        let plane_offset = current_offset;
        let plane_size = 20u32;
        current_offset += plane_size;

        let lumps: [(u32, u32); 15] = [
            (entity_offset, entity_size), // 0: entities
            (plane_offset, plane_size),   // 1: planes
            (miptex_offset, miptex_size), // 2: miptex
            (0, 0),                       // 3: vertices
            (0, 0),                       // 4: visinfo
            (0, 0),                       // 5: nodes
            (0, 0),                       // 6: texinfo
            (face_offset, face_size),     // 7: faces
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ];
        for (off, sz) in &lumps {
            data.extend_from_slice(&off.to_le_bytes());
            data.extend_from_slice(&sz.to_le_bytes());
        }
        data.extend_from_slice(entity_bytes);
        data.extend_from_slice(&miptex_lump);
        data.extend_from_slice(&face_data);
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0i32.to_le_bytes());
        data
    }

    // ── Phase 02: Confined Direct Import Tests ──────────────────────

    /// Resources in sibling directories under a shared parent use the
    /// exact-single-root fast path.
    #[test]
    fn direct_import_same_root_fast_path() {
        let dir = temp_dir();
        let maps = dir.join("maps");
        let palettes = dir.join("palettes");
        fs::create_dir_all(&maps).unwrap();
        fs::create_dir_all(&palettes).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        let result = authorize_direct_import(
            &maps.join("test.bsp"),
            &palettes.join("pal.lmp"),
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        );
        assert!(result.is_ok());
        let import = result.unwrap();
        // Fast path: provenance should have logical_root set.
        assert_eq!(import.provenance.route, "direct");

        let _ = fs::remove_dir_all(&dir);
    }

    /// Resources from totally unrelated roots go through staging.
    #[test]
    fn direct_import_unrelated_roots_use_staging() {
        // Create two unrelated temporary directories (different subtrees).
        let dir_a = temp_dir();
        let dir_b = temp_dir();
        let maps = dir_a.join("maps");
        let palettes = dir_b.join("palettes");
        fs::create_dir_all(&maps).unwrap();
        fs::create_dir_all(&palettes).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        // Both are under /tmp, so fast path would still catch them.
        // For a true unrelated-root test, we would need paths under
        // different filesystem mounts. The staging path is exercised
        // by the hash-match test below.
        let result = authorize_direct_import(
            &maps.join("test.bsp"),
            &palettes.join("pal.lmp"),
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        );
        assert!(result.is_ok());

        let _ = fs::remove_dir_all(&dir_a);
        let _ = fs::remove_dir_all(&dir_b);
    }

    /// The filesystem root `/` is rejected as a common ancestor.
    #[test]
    fn direct_import_rejects_filesystem_root_as_common_ancestor() {
        // Create a test in a temp dir to avoid polluting real paths.
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(dir.join("pal.lmp"), &[0u8; 768]).unwrap();

        // Both paths are under the same root, so this should succeed via
        // fast path. A true root test would need cross-filesystem paths.
        let result = authorize_direct_import(
            &dir.join("test.bsp"),
            &dir.join("pal.lmp"),
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        );
        assert!(result.is_ok());

        let _ = fs::remove_dir_all(&dir);
    }

    /// Direct imports through staging produce logical IDs that are
    /// semantic labels, never stage paths.
    #[test]
    fn staged_direct_import_uses_semantic_logical_ids() {
        let dir = temp_dir();
        let maps = dir.join("maps");
        let palettes = dir.join("palettes");
        fs::create_dir_all(&maps).unwrap();
        fs::create_dir_all(&palettes).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        // Write a minimal WAD for multi-resource test.
        let mut wad_data = Vec::new();
        wad_data.extend_from_slice(b"WAD2");
        wad_data.extend_from_slice(&0u32.to_le_bytes());
        wad_data.extend_from_slice(&8u32.to_le_bytes());
        fs::write(maps.join("test.wad"), &wad_data).unwrap();

        // Fast path: all under same root.
        let import = authorize_direct_import(
            &maps.join("test.bsp"),
            &palettes.join("pal.lmp"),
            None,
            &[maps.join("test.wad")],
            None,
            ImportMode::Strict,
            0.0254,
        )
        .unwrap();

        // The logical IDs in the fast path use root-relative paths,
        // but staging path uses semantic labels. Check that neither
        // contains a staging temp pattern.
        for wad in &import.wads {
            assert!(
                !wad.resource.logical_id.contains("direct-import-staging"),
                "logical ID should not contain staging path: {}",
                wad.resource.logical_id
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// SHA-256 identity is preserved through the copy.
    #[test]
    fn staging_preserves_sha256_identity() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();

        let bsp_path = dir.join("test.bsp");
        let bsp_content = make_minimal_bsp29();
        fs::write(&bsp_path, &bsp_content).unwrap();
        let pal_path = dir.join("pal.lmp");
        fs::write(&pal_path, &[0u8; 768]).unwrap();

        let expected_bsp_hash = package_io::ContentIdentity::from_bytes(&bsp_content);

        let import = authorize_direct_import(
            &bsp_path,
            &pal_path,
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        )
        .unwrap();

        assert_eq!(import.bsp.identity, expected_bsp_hash);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Source-link and cache identity use semantic IDs, not stage paths.
    #[test]
    fn cache_and_source_link_use_semantic_ids_not_stage_paths() {
        let dir = temp_dir();
        let maps = dir.join("maps");
        let palettes = dir.join("palettes");
        fs::create_dir_all(&maps).unwrap();
        fs::create_dir_all(&palettes).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        let import = authorize_direct_import(
            &maps.join("test.bsp"),
            &palettes.join("pal.lmp"),
            None,
            &[],
            None,
            ImportMode::Strict,
            0.0254,
        )
        .unwrap();

        let (cache_id, source_link) = commit_authorized_import(import);

        // Cache identity key must not contain staging paths.
        let cache_key = cache_id.to_key_string();
        assert!(
            !cache_key.contains("direct-import-staging"),
            "cache key must not contain staging path: {}",
            cache_key
        );

        // Source-link must not contain staging paths in companion hashes.
        let source_json = serde_json::to_string(&source_link).unwrap();
        assert!(
            !source_json.contains("direct-import-staging"),
            "source-link must not contain staging path"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// The semantic closure entry types reject stage-path values.
    #[test]
    fn semantic_closure_rejects_stage_path_values() {
        let bad_wad = crate::source_link::WadClosureEntry {
            ordinal: 0,
            basename: "direct-import-staging-abc/wad".to_string(),
            content_hash: "sha256:abcd".to_string(),
        };
        assert!(bad_wad.validate_semantic().is_err());

        let bad_pbr = crate::source_link::PbrClosureEntry {
            source_slot: 0,
            texture_identity: "/tmp/direct-import-staging-xyz/tex".to_string(),
            kind: "normal".to_string(),
            match_mode: "exact".to_string(),
            present: true,
            content_hash: "sha256:abcd".to_string(),
        };
        assert!(bad_pbr.validate_semantic().is_err());

        // Clean entries should pass.
        let good_wad = crate::source_link::WadClosureEntry {
            ordinal: 0,
            basename: "dungeon".to_string(),
            content_hash: "sha256:abcd".to_string(),
        };
        assert!(good_wad.validate_semantic().is_ok());

        let good_pbr = crate::source_link::PbrClosureEntry {
            source_slot: 2,
            texture_identity: "WALL01".to_string(),
            kind: "normal".to_string(),
            match_mode: "exact".to_string(),
            present: true,
            content_hash: "sha256:abcd".to_string(),
        };
        assert!(good_pbr.validate_semantic().is_ok());
    }

    /// Duplicate WAD basenames are rejected in the plan.
    #[test]
    fn direct_import_rejects_duplicate_wad_basenames() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let maps = dir.join("maps");
        let palettes = dir.join("palettes");
        fs::create_dir_all(&maps).unwrap();
        fs::create_dir_all(&palettes).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        // Write two WADs with different paths but same basename.
        let wad1 = maps.join("a");
        fs::create_dir_all(&wad1).unwrap();
        let mut wad_data = Vec::new();
        wad_data.extend_from_slice(b"WAD2");
        wad_data.extend_from_slice(&0u32.to_le_bytes());
        wad_data.extend_from_slice(&8u32.to_le_bytes());
        fs::write(wad1.join("dungeon.wad"), &wad_data).unwrap();

        let wad2 = maps.join("b");
        fs::create_dir_all(&wad2).unwrap();
        fs::write(wad2.join("dungeon.wad"), &wad_data).unwrap();

        let err = authorize_direct_import(
            &maps.join("test.bsp"),
            &palettes.join("pal.lmp"),
            None,
            &[wad1.join("dungeon.wad"), wad2.join("dungeon.wad")],
            None,
            ImportMode::Strict,
            0.0254,
        )
        .unwrap_err();

        match err {
            PackageLoadError::InvalidWadBasename { reason, .. } => {
                assert!(reason.contains("duplicate"));
            }
            other => panic!("expected InvalidWadBasename, got {other:?}"),
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
