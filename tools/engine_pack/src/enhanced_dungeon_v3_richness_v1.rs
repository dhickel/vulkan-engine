//! Enhanced V3 Richness V1 dungeon package candidate builder.
//!
//! Takes a `RichnessDocumentV1` and output directory, generates a `.map`
//! via `bsp_generator::generate_richness_v1`, compiles it through
//! ericw-tools with a pinned profile, and collects BSP+LIT+WAD+palette+
//! request+metadata+manifest into a validated closure published atomically
//! through a same-filesystem no-replace transaction.
//!
//! # Design rules
//!
//! - Publication is fail-closed: a pre-existing valid destination with
//!   identical content is a no-op; a different valid destination is a
//!   LateCollision; any incomplete, malformed, or non-directory destination
//!   is rejected without modification.
//! - Baseline package generation never calls a Richness serializer,
//!   resolver, asset selector, or RNG path — this is guarded by construction
//!   and test.
//! - All revisions, inherited/explicit values, tool hashes, asset hashes,
//!   semantic identity, and output hashes are serialized into the manifest.
//! - Strict theme closure with provenance verification across three themes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use bsp_generator::{InheritedOr, RichnessDocumentV1, RichnessPreset, RichnessTheme};

use crate::compiler;
use crate::fs_tx;
use crate::richness_assets;

/// Outcome of a Richness V1 package publication.
#[derive(Debug)]
pub enum BuildRichnessV1Result {
    /// Staging was renamed atomically into a new destination.
    Published { target: PathBuf, message: String },
    /// Destination already held an identical validated closure — no-op.
    Unchanged { target: PathBuf, message: String },
}

/// Error type for Richness V1 package building.
#[derive(Debug)]
pub enum BuildRichnessV1Error {
    Config(String),
    Generation(String),
    Io {
        path: PathBuf,
        message: String,
    },
    Profile(String),
    Input {
        path: PathBuf,
        message: String,
    },
    FsTx(fs_tx::FsTxError),
    Compilation(String),
    PbrClosure(String),
    PackageValidation(String),
    AssetValidation(String),
    LateCollision {
        target: PathBuf,
        new_manifest_sha256: String,
        existing_manifest_sha256: Option<String>,
    },
    IncompleteDestination {
        target: PathBuf,
        reason: String,
    },
    PublicationBlocked {
        target: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for BuildRichnessV1Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(msg) => write!(f, "invalid richness config: {msg}"),
            Self::Generation(msg) => write!(f, "richness generation failed: {msg}"),
            Self::Io { path, message } => write!(f, "I/O error at '{}': {message}", path.display()),
            Self::Profile(msg) => write!(f, "invalid compiler profile: {msg}"),
            Self::Input { path, message } => {
                write!(f, "invalid input '{}': {message}", path.display())
            }
            Self::FsTx(e) => write!(f, "fs_tx error: {e}"),
            Self::Compilation(msg) => write!(f, "compilation failed: {msg}"),
            Self::PbrClosure(msg) => write!(f, "PBR closure: {msg}"),
            Self::PackageValidation(msg) => write!(f, "package validation: {msg}"),
            Self::AssetValidation(msg) => write!(f, "asset validation: {msg}"),
            Self::LateCollision {
                target,
                new_manifest_sha256,
                existing_manifest_sha256,
            } => write!(
                f,
                "late-collision at '{}': new manifest sha256={} existing={}",
                target.display(),
                new_manifest_sha256,
                existing_manifest_sha256
                    .as_deref()
                    .unwrap_or("<unparseable>")
            ),
            Self::IncompleteDestination { target, reason } => {
                write!(
                    f,
                    "incomplete destination at '{}': {reason}",
                    target.display()
                )
            }
            Self::PublicationBlocked { target, message } => {
                write!(
                    f,
                    "publication blocked at '{}': {message}",
                    target.display()
                )
            }
        }
    }
}

impl std::error::Error for BuildRichnessV1Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FsTx(e) => Some(e),
            _ => None,
        }
    }
}

impl From<fs_tx::FsTxError> for BuildRichnessV1Error {
    fn from(e: fs_tx::FsTxError) -> Self {
        Self::FsTx(e)
    }
}

/// Default compiler profile bundled with engine_pack.
const DEFAULT_BSP2_PROFILE: &str =
    include_str!("../../bsp_authoring/ericw-q1-bsp2-generated-profile.toml");

/// Build and publish a Richness V1 dungeon package through an atomic
/// no-replace transaction.
///
/// On success, returns a `BuildRichnessV1Result` describing the outcome
/// (published or unchanged). On failure, the destination is never created,
/// modified, or partially populated.
pub fn build_richness_v1_package(
    doc: &RichnessDocumentV1,
    out_dir: &Path,
    tool_path: Option<&Path>,
    name: &str,
    profile_override: Option<&str>,
) -> Result<BuildRichnessV1Result, BuildRichnessV1Error> {
    // ── 1. Validate the document ────────────────────────────────────
    doc.validate_raw_fields()
        .map_err(|e| BuildRichnessV1Error::Config(e.to_string()))?;

    let seed = doc.seed();
    let preset = doc.preset();
    let theme = doc.theme();

    // ── 2. Run generator ────────────────────────────────────────────
    let output = bsp_generator::generate_richness_v1(doc)
        .map_err(|e| BuildRichnessV1Error::Generation(e.to_string()))?;

    // ── 3. Resolve compiler profile ──────────────────────────────────
    let profile_content = if let Some(profile_path) = profile_override {
        std::fs::read_to_string(profile_path).map_err(|err| BuildRichnessV1Error::Io {
            path: PathBuf::from(profile_path),
            message: format!("read profile: {err}"),
        })?
    } else {
        DEFAULT_BSP2_PROFILE.to_string()
    };
    let profile = compiler::parse_compiler_profile(&profile_content)
        .map_err(|msg| BuildRichnessV1Error::Profile(msg))?;

    // ── 4. Resolve theme assets ─────────────────────────────────────
    let theme_def = theme_definition_for(theme);
    let theme_dir = richness_theme_dir(theme)?;

    // Validate the complete theme closure at startup
    richness_assets::validate_theme_closure(&theme_dir, &theme_def)
        .map_err(|e| BuildRichnessV1Error::AssetValidation(e.to_string()))?;

    let palette_path = theme_dir.join(theme_def.palette_filename);
    let wad_path = theme_dir.join(theme_def.wad_filename);

    for input in [&palette_path, &wad_path] {
        compiler::validate_input_regular(input).map_err(|err| BuildRichnessV1Error::Input {
            path: input.clone(),
            message: format!("invalid theme asset: {err}"),
        })?;
    }

    // ── 5. Recover orphaned staging ─────────────────────────────────
    fs_tx::recover_orphaned_staging(out_dir);

    // ── 6. Create staging ───────────────────────────────────────────
    let staging = fs_tx::create_staging_sibling(out_dir)?;
    fs_tx::write_staging_marker(&staging, out_dir)?;

    let result = (|| -> Result<BuildRichnessV1Result, BuildRichnessV1Error> {
        // Write .map source into staging
        let map_filename = format!("{name}.map");
        let map_path = staging.join(&map_filename);
        std::fs::write(&map_path, &output.map_text).map_err(|err| BuildRichnessV1Error::Io {
            path: map_path.clone(),
            message: format!("write .map: {err}"),
        })?;

        // Compile
        let work_dir = staging.join(".compile-work");
        std::fs::create_dir_all(&work_dir).map_err(|err| BuildRichnessV1Error::Io {
            path: work_dir.clone(),
            message: format!("create work dir: {err}"),
        })?;

        let compile_result = compiler::compile_map(
            &map_path,
            &profile,
            &work_dir,
            &palette_path,
            tool_path,
            &[wad_path.clone()],
        )
        .map_err(|err| BuildRichnessV1Error::Compilation(format!("{err}")))?;

        let _ = std::fs::remove_dir_all(&work_dir);

        // Stage compiled .bsp
        let bsp_path = staging.join(format!("{name}.bsp"));
        std::fs::write(&bsp_path, &compile_result.bsp_data).map_err(|err| {
            BuildRichnessV1Error::Io {
                path: bsp_path.clone(),
                message: format!("write .bsp: {err}"),
            }
        })?;

        // Stage .lit companion
        if let Some(ref lit_data) = compile_result.lit_data {
            let lit_path = staging.join(format!("{name}.lit"));
            std::fs::write(&lit_path, lit_data).map_err(|err| BuildRichnessV1Error::Io {
                path: lit_path.clone(),
                message: format!("write .lit: {err}"),
            })?;
        }

        // Stage palette
        let palette_bytes =
            std::fs::read(&palette_path).map_err(|err| BuildRichnessV1Error::Io {
                path: palette_path.clone(),
                message: format!("read palette: {err}"),
            })?;
        let palette_staged = staging.join("palette.lmp");
        std::fs::write(&palette_staged, &palette_bytes).map_err(|err| {
            BuildRichnessV1Error::Io {
                path: palette_staged.clone(),
                message: format!("write palette: {err}"),
            }
        })?;

        // Stage WAD
        let wad_basename = wad_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("theme.wad");
        let wad_staged = staging.join(wad_basename);
        std::fs::copy(&wad_path, &wad_staged).map_err(|err| BuildRichnessV1Error::Io {
            path: wad_path.clone(),
            message: format!("copy WAD: {err}"),
        })?;

        // Stage PBR companion textures from the theme's textures/ directory
        let staged_pbr = stage_richness_pbr_companions(
            &staging,
            name,
            &compile_result,
            &[wad_path.clone()],
            &palette_bytes,
        )?;
        require_complete_richness_pbr_closure(&staged_pbr)?;

        // Validate staged closure through isolated strict authorization
        validate_richness_staged_import(
            &staging,
            name,
            compile_result.lit_data.is_some(),
            &staged_pbr
                .required_wad_basenames
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        )?;

        // Write request.json (canonical request export)
        let request_path = staging.join(format!("{name}.request.json"));
        std::fs::write(&request_path, &output.request_export).map_err(|err| {
            BuildRichnessV1Error::Io {
                path: request_path.clone(),
                message: format!("write request: {err}"),
            }
        })?;

        // Write request metadata
        let meta_bytes = output.generation_metadata.to_canonical_bytes();
        let meta_path = staging.join(format!("{name}.generation.txt"));
        std::fs::write(&meta_path, &meta_bytes).map_err(|err| BuildRichnessV1Error::Io {
            path: meta_path.clone(),
            message: format!("write generation metadata: {err}"),
        })?;

        // Write metadata.json
        let metadata = build_richness_metadata_json(doc, &output, &compile_result, &map_filename);
        let metadata_path = staging.join("metadata.json");
        std::fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&metadata).unwrap(),
        )
        .map_err(|err| BuildRichnessV1Error::Io {
            path: metadata_path,
            message: format!("write metadata: {err}"),
        })?;

        // ── Build canonical manifest ───────────────────────────────
        let staged_hashes = fs_tx::compute_dir_file_hashes(&staging)?;
        let manifest_toml = build_richness_canonical_manifest(
            &staging,
            name,
            doc,
            &output,
            &compile_result,
            &staged_hashes,
        )?;
        let manifest_path = staging.join(format!("{name}.manifest.toml"));
        std::fs::write(&manifest_path, &manifest_toml).map_err(|err| BuildRichnessV1Error::Io {
            path: manifest_path.clone(),
            message: format!("write manifest: {err}"),
        })?;

        // Remove staging marker before final closure validation
        fs_tx::remove_staging_marker(&staging)?;

        // Validate manifest closure
        fs_tx::validate_manifest_closure(&staging, manifest_toml.as_bytes())?;

        // Publish atomically
        let publish_result = publish_richness_or_resolve_collision(&staging, out_dir)?;

        match publish_result {
            fs_tx::PublicationOutcome::Published { target, .. } => {
                Ok(BuildRichnessV1Result::Published {
                    message: format!(
                        "published[enhanced-dungeon-v3-richness-v1]: seed={seed} preset={} theme={} -> {}/ ({} faces, {} lights)",
                        preset.tag(),
                        theme.tag(),
                        out_dir.display(),
                        output.actual.faces,
                        output.actual.lights,
                    ),
                    target,
                })
            }
            fs_tx::PublicationOutcome::Unchanged { target, .. } => {
                Ok(BuildRichnessV1Result::Unchanged {
                    message: format!(
                        "unchanged[enhanced-dungeon-v3-richness-v1]: seed={seed} preset={} theme={} -> {}/ ({} faces, {} lights)",
                        preset.tag(),
                        theme.tag(),
                        out_dir.display(),
                        output.actual.faces,
                        output.actual.lights,
                    ),
                    target,
                })
            }
            fs_tx::PublicationOutcome::LateCollision {
                target,
                new_manifest_sha256,
                existing_manifest_sha256,
            } => Err(BuildRichnessV1Error::LateCollision {
                target,
                new_manifest_sha256,
                existing_manifest_sha256,
            }),
            fs_tx::PublicationOutcome::IncompleteDestination { target, reason } => {
                Err(BuildRichnessV1Error::IncompleteDestination { target, reason })
            }
        }
    })();

    if result.is_err() {
        fs_tx::cleanup_staging(&staging);
    }
    result
}

// ── Theme resolution ──────────────────────────────────────────────────────

/// Map a RichnessTheme to its frozen ThemeDefinition.
fn theme_definition_for(theme: RichnessTheme) -> richness_assets::ThemeDefinition {
    match theme {
        RichnessTheme::Ancient => richness_assets::THEME_ANCIENT,
        RichnessTheme::Egyptian => richness_assets::THEME_EGYPTIAN,
        RichnessTheme::Brutalist => richness_assets::THEME_BRUTALIST,
    }
}

/// Resolve the workspace theme directory for a RichnessTheme.
fn richness_theme_dir(theme: RichnessTheme) -> Result<PathBuf, BuildRichnessV1Error> {
    let engine_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| BuildRichnessV1Error::Io {
            path: PathBuf::from("<manifest>"),
            message: "engine_pack manifest directory is not under the workspace tools directory"
                .into(),
        })?;
    let theme_def = theme_definition_for(theme);
    Ok(engine_root
        .join("src/bsp_generator/themes")
        .join(theme_def.dir_name))
}

// ── PBR companion staging ─────────────────────────────────────────────────

struct StagedRichnessPbrClosure {
    required_wad_basenames: BTreeSet<String>,
    eligible_identities: BTreeSet<String>,
    staged_companions: BTreeSet<String>,
}

fn require_complete_richness_pbr_closure(
    staged: &StagedRichnessPbrClosure,
) -> Result<(), BuildRichnessV1Error> {
    let expected = staged
        .eligible_identities
        .iter()
        .flat_map(|identity| {
            [
                format!("{identity}_norm.png"),
                format!("{identity}_gloss.png"),
            ]
        })
        .collect::<BTreeSet<_>>();
    if staged.staged_companions == expected {
        return Ok(());
    }

    let missing = expected
        .difference(&staged.staged_companions)
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = staged
        .staged_companions
        .difference(&expected)
        .cloned()
        .collect::<Vec<_>>();
    Err(BuildRichnessV1Error::PbrClosure(format!(
        "incomplete Richness PBR companion closure; missing: {missing:?}; unexpected: {unexpected:?}"
    )))
}

fn stage_richness_pbr_companions(
    staging: &Path,
    _bsp_name: &str,
    compile_result: &bsp::CompileResult,
    wad_paths: &[PathBuf],
    palette_bytes: &[u8],
) -> Result<StagedRichnessPbrClosure, BuildRichnessV1Error> {
    let mut wad_archives: Vec<(String, Vec<u8>)> = Vec::new();
    for wad_path in wad_paths {
        validate_no_symlink_components_richness(wad_path, "WAD input")?;
        let basename = wad_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.wad")
            .to_string();
        let bytes = std::fs::read(wad_path).map_err(|err| BuildRichnessV1Error::Io {
            path: wad_path.clone(),
            message: format!("read WAD: {err}"),
        })?;
        wad_archives.push((basename, bytes));
    }

    let load_options = bsp::LoadOptions {
        strict: true,
        palette: Some(palette_bytes.to_vec()),
        lit_data: compile_result.lit_data.clone(),
        wad_archives,
        texture_overrides: Vec::new(),
        source_identity: "richness-compile".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compile_result.bsp_data, &load_options).map_err(|report| {
            BuildRichnessV1Error::Compilation(format!("strict-load validation failed: {report}"))
        })?;

    if !world.diagnostics.is_empty() {
        return Err(BuildRichnessV1Error::Compilation(format!(
            "strict-load produced diagnostics: {:?}",
            world
                .diagnostics
                .iter()
                .map(|d| d.message.as_str())
                .collect::<Vec<_>>()
        )));
    }

    let slots = bsp::resources::parse_miptex_slots(&world.miptex_data);
    let required_wad_basenames = required_wad_basenames_richness(&world, &slots);

    let mut eligible: BTreeSet<String> = BTreeSet::new();
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
        let class = bsp::materials::classify_surface(texinfo.flags, identity);
        if matches!(
            class,
            bsp::materials::SurfaceClass::Opaque | bsp::materials::SurfaceClass::AlphaMask
        ) {
            eligible.insert(identity.clone());
        }
    }

    let base_dimensions = eligible
        .iter()
        .map(|identity| {
            resolve_base_texture_dimensions_richness(&world, &slots, identity)
                .map(|dims| (identity.clone(), dims))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;

    let textures_dir = staging.join("textures");
    std::fs::create_dir_all(&textures_dir).map_err(|err| BuildRichnessV1Error::Io {
        path: textures_dir.clone(),
        message: format!("create textures dir: {err}"),
    })?;
    let mut staged_companions = BTreeSet::new();

    for identity in &eligible {
        for suffix in &["_norm.png", "_gloss.png"] {
            let expected = format!("{identity}{suffix}");
            let mut found: Option<PathBuf> = None;
            for wad_path in wad_paths {
                let wad_parent = wad_path.parent().unwrap_or_else(|| Path::new("."));
                let search_dir = if wad_parent
                    .file_name()
                    .map(|n| n == "textures")
                    .unwrap_or(false)
                {
                    wad_parent.to_path_buf()
                } else {
                    wad_parent.join("textures")
                };
                let search_metadata = match std::fs::symlink_metadata(&search_dir) {
                    Ok(meta) => meta,
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(err) => {
                        return Err(BuildRichnessV1Error::Io {
                            path: search_dir.clone(),
                            message: format!("inspect companion root: {err}"),
                        })
                    }
                };
                if search_metadata.file_type().is_symlink() {
                    return Err(BuildRichnessV1Error::PbrClosure(format!(
                        "PBR companion root must not be a symlink: '{}'",
                        search_dir.display()
                    )));
                }
                if !search_metadata.is_dir() {
                    continue;
                }
                let all_entries =
                    std::fs::read_dir(&search_dir).map_err(|err| BuildRichnessV1Error::Io {
                        path: search_dir.clone(),
                        message: format!("read companion root: {err}"),
                    })?;
                for entry in all_entries {
                    let entry = entry.map_err(|err| BuildRichnessV1Error::Io {
                        path: search_dir.clone(),
                        message: format!("read companion entry: {err}"),
                    })?;
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str == expected {
                        found = Some(entry.path());
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }

            if let Some(src) = found {
                let expected_dimensions =
                    base_dimensions.get(identity).copied().ok_or_else(|| {
                        BuildRichnessV1Error::PbrClosure(format!(
                            "missing base dimensions for companion '{identity}'"
                        ))
                    })?;
                validate_pbr_companion_richness(&src, identity, expected_dimensions)?;
                let dest = textures_dir.join(&expected);
                std::fs::copy(&src, &dest).map_err(|err| BuildRichnessV1Error::Io {
                    path: src.clone(),
                    message: format!("copy companion: {err}"),
                })?;
                staged_companions.insert(expected);
            }
        }
    }

    Ok(StagedRichnessPbrClosure {
        required_wad_basenames,
        eligible_identities: eligible,
        staged_companions,
    })
}

fn validate_no_symlink_components_richness(
    path: &Path,
    label: &str,
) -> Result<(), BuildRichnessV1Error> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|err| BuildRichnessV1Error::Io {
                path: PathBuf::from("."),
                message: format!("current_dir: {err}"),
            })?
            .join(path)
    };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            std::path::Component::RootDir => current.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                return Err(BuildRichnessV1Error::Input {
                    path: path.to_path_buf(),
                    message: format!("{label} path contains parent traversal"),
                })
            }
            std::path::Component::Normal(part) => {
                current.push(part);
                let metadata = std::fs::symlink_metadata(&current).map_err(|err| {
                    BuildRichnessV1Error::Io {
                        path: current.clone(),
                        message: format!("inspect component: {err}"),
                    }
                })?;
                if metadata.file_type().is_symlink() {
                    return Err(BuildRichnessV1Error::Input {
                        path: current.clone(),
                        message: format!("{label} path contains symlink"),
                    });
                }
            }
        }
    }
    Ok(())
}

fn resolve_base_texture_dimensions_richness(
    world: &bsp::world::BspWorld,
    slots: &[bsp::resources::MiptexSlot],
    identity: &str,
) -> Result<(u32, u32), BuildRichnessV1Error> {
    for (_, wad) in &world.wad_archives {
        if let Some(bytes) = bsp::wad::read_wad_lump(wad, identity) {
            let info = bsp::wad::parse_miptex_header(bytes).map_err(|err| {
                BuildRichnessV1Error::PbrClosure(format!(
                    "invalid base miptex '{identity}' in WAD: {err}"
                ))
            })?;
            return Ok((info.width, info.height));
        }
    }
    for slot in slots {
        if slot.identity.as_deref() != Some(identity) {
            continue;
        }
        if let Some(bytes) =
            bsp::wad::read_embedded_miptex_entry(&world.miptex_data, slot.source_slot)
        {
            let info = bsp::wad::parse_miptex_header(bytes).map_err(|err| {
                BuildRichnessV1Error::PbrClosure(format!(
                    "invalid embedded miptex '{identity}': {err}"
                ))
            })?;
            return Ok((info.width, info.height));
        }
    }
    Err(BuildRichnessV1Error::PbrClosure(format!(
        "cannot determine dimensions for PBR base texture '{identity}'"
    )))
}

fn validate_pbr_companion_richness(
    path: &Path,
    identity: &str,
    expected_dimensions: (u32, u32),
) -> Result<(), BuildRichnessV1Error> {
    compiler::validate_input_regular(path).map_err(|err| BuildRichnessV1Error::Input {
        path: path.to_path_buf(),
        message: format!("invalid PBR companion '{}': {err}", path.display()),
    })?;
    let bytes = std::fs::read(path).map_err(|err| BuildRichnessV1Error::Io {
        path: path.to_path_buf(),
        message: format!("read companion: {err}"),
    })?;
    let dimensions = parse_png_dimensions_richness(&bytes).map_err(|reason| {
        BuildRichnessV1Error::PbrClosure(format!(
            "malformed PBR companion '{}' for '{identity}': {reason}",
            path.display()
        ))
    })?;
    if dimensions != expected_dimensions {
        return Err(BuildRichnessV1Error::PbrClosure(format!(
            "PBR companion '{}' dimensions {}x{} do not match base texture '{identity}' {}x{}",
            path.display(),
            dimensions.0,
            dimensions.1,
            expected_dimensions.0,
            expected_dimensions.1
        )));
    }
    Ok(())
}

fn required_wad_basenames_richness(
    world: &bsp::world::BspWorld,
    slots: &[bsp::resources::MiptexSlot],
) -> BTreeSet<String> {
    let mut required = BTreeSet::new();
    for face in &world.faces {
        let Some(texinfo) = world.texinfos.get(face.texinfo_id as usize) else {
            continue;
        };
        let Some(identity) = slots
            .get(texinfo.miptex as usize)
            .and_then(|slot| slot.identity.as_deref())
        else {
            continue;
        };
        if let Some((wad_name, _)) = world.wad_archives.iter().find(|(_, wad)| {
            wad.entries
                .iter()
                .any(|entry| entry.name == identity || entry.name.eq_ignore_ascii_case(identity))
        }) {
            required.insert(wad_name.clone());
        }
    }
    required
}

fn validate_richness_staged_import(
    staging: &Path,
    bsp_name: &str,
    has_lit: bool,
    wad_paths: &[String],
) -> Result<(), BuildRichnessV1Error> {
    use package_io::budget::BudgetLedger;
    use package_io::PackageRoot;

    let root = PackageRoot::new(staging).map_err(|err| {
        BuildRichnessV1Error::PackageValidation(format!("cannot create staged package root: {err}"))
    })?;
    let mut resolver =
        package_io::resolver::PackageResolver::new(root, BudgetLedger::default_ledger());
    let lit_path = has_lit.then(|| format!("{bsp_name}.lit"));

    let import = bsp_runtime::package::authorize_package_import(
        &mut resolver,
        &format!("{bsp_name}.bsp"),
        "palette.lmp",
        lit_path.as_deref(),
        wad_paths,
        Some("textures"),
        bsp_runtime::package::ImportMode::Strict,
        0.0254,
    )
    .map_err(|err| {
        BuildRichnessV1Error::PackageValidation(format!(
            "isolated strict package import failed: {err}"
        ))
    })?;

    if !import.world.diagnostics.is_empty() {
        return Err(BuildRichnessV1Error::PackageValidation(format!(
            "isolated strict package import produced diagnostics: {:?}",
            import
                .world
                .diagnostics
                .iter()
                .map(|d| d.message.as_str())
                .collect::<Vec<_>>()
        )));
    }
    Ok(())
}

// ── JSON helpers ──────────────────────────────────────────────────────────

fn build_richness_metadata_json(
    doc: &RichnessDocumentV1,
    output: &bsp_generator::RichnessPipelineOutput,
    compile_result: &bsp::CompileResult,
    map_filename: &str,
) -> serde_json::Value {
    let mut inherited: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut explicit: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    let push = |map: &mut BTreeMap<_, _>, key: &str, val: serde_json::Value| {
        map.insert(key.to_string(), val);
    };
    match doc.critical_path_landmarks() {
        InheritedOr::Inherited => push(&mut inherited, "landmarks", serde_json::Value::Null),
        InheritedOr::Explicit(v) => push(&mut explicit, "landmarks", v.into()),
    }
    match doc.zone_count() {
        InheritedOr::Inherited => push(&mut inherited, "zones", serde_json::Value::Null),
        InheritedOr::Explicit(v) => push(&mut explicit, "zones", v.into()),
    }
    match doc.cave_mode() {
        InheritedOr::Inherited => push(&mut inherited, "cave_mode", serde_json::Value::Null),
        InheritedOr::Explicit(v) => push(&mut explicit, "cave_mode", v.tag().into()),
    }
    match doc.vertical_openings() {
        InheritedOr::Inherited => {
            push(&mut inherited, "vertical_openings", serde_json::Value::Null)
        }
        InheritedOr::Explicit(v) => push(&mut explicit, "vertical_openings", v.into()),
    }
    match doc.budget_ceiling() {
        InheritedOr::Inherited => push(&mut inherited, "budget", serde_json::Value::Null),
        InheritedOr::Explicit(v) => push(&mut explicit, "budget", v.into()),
    }

    serde_json::json!({
        "format_version": 1,
        "schema_version": output.request_metadata.schema_version(),
        "generator": "bsp_generator/enhanced_v3/richness/v1",
        "seed": doc.seed(),
        "preset": doc.preset().tag(),
        "theme": doc.theme().tag(),
        "extent": doc.extent(),
        "controls": {
            "inherited": inherited,
            "explicit": explicit,
        },
        "output": {
            "faces": output.actual.faces,
            "brushes": output.actual.brushes,
            "entities": output.actual.entities,
            "lights": output.actual.lights,
            "support_contacts": output.actual.support_contacts,
            "openings": output.actual.openings,
        },
        "compiler": {
            "identity": compile_result.provenance.compiler_identity,
            "version": compile_result.provenance.compiler_version,
        },
        "revisions": {
            "request_schema": doc.request_schema_revision().tag(),
            "algorithm": doc.algorithm_revision().tag(),
            "content": doc.content_revision().tag(),
            "preset": doc.preset_revision().tag(),
            "theme": doc.theme_revision().tag(),
            "asset": doc.asset_revision().tag(),
            "convention": doc.convention_revision().tag(),
        },
        "asset_roles": output.asset_roles,
        "map_filename": map_filename,
    })
}

// ── Atomic publication ────────────────────────────────────────────────────

fn publish_richness_or_resolve_collision(
    staging: &Path,
    destination: &Path,
) -> Result<fs_tx::PublicationOutcome, BuildRichnessV1Error> {
    match fs_tx::publish_directory_no_replace(staging, destination) {
        Ok(()) => {
            let hashes = fs_tx::compute_dir_file_hashes(destination)?;
            let manifest_hash = hashes
                .iter()
                .find(|(name, _)| name.ends_with(".manifest.toml"))
                .map(|(_, hash)| hash.clone())
                .unwrap_or_else(|| "<no-manifest>".to_string());
            Ok(fs_tx::PublicationOutcome::Published {
                target: destination.to_path_buf(),
                manifest_sha256: manifest_hash,
            })
        }
        Err(fs_tx::FsTxError::PreExistingDestination { target, .. }) => {
            resolve_existing_richness_destination(staging, &target)
        }
        Err(e) => Err(BuildRichnessV1Error::FsTx(e)),
    }
}

fn resolve_existing_richness_destination(
    staging: &Path,
    destination: &Path,
) -> Result<fs_tx::PublicationOutcome, BuildRichnessV1Error> {
    let dest_meta = match fs_tx::inspect_entry_no_follow(destination) {
        Ok(meta) if meta.is_dir() => meta,
        Ok(_) => {
            return Err(BuildRichnessV1Error::PublicationBlocked {
                target: destination.to_path_buf(),
                message: "destination exists but is not a directory".to_string(),
            });
        }
        Err(_) => {
            return Err(BuildRichnessV1Error::PublicationBlocked {
                target: destination.to_path_buf(),
                message: "destination cannot be inspected".to_string(),
            });
        }
    };
    let _ = dest_meta;

    let manifest_path = find_richness_manifest_in_dir(destination)?;
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|err| BuildRichnessV1Error::Io {
        path: manifest_path.clone(),
        message: format!("read existing manifest: {err}"),
    })?;
    let existing_manifest_hash = fs_tx::compute_manifest_sha256(&manifest_bytes);

    let existing_valid = fs_tx::validate_manifest_closure(destination, &manifest_bytes).is_ok();

    if !existing_valid {
        return Ok(fs_tx::PublicationOutcome::IncompleteDestination {
            target: destination.to_path_buf(),
            reason: "existing destination fails full-closure validation".to_string(),
        });
    }

    let identical = fs_tx::artifact_sets_identical(staging, destination).unwrap_or(false);

    if identical {
        Ok(fs_tx::PublicationOutcome::Unchanged {
            target: destination.to_path_buf(),
            manifest_sha256: existing_manifest_hash,
        })
    } else {
        let new_hashes = fs_tx::compute_dir_file_hashes(staging)?;
        let new_manifest_hash = new_hashes
            .iter()
            .find(|(name, _)| name.ends_with(".manifest.toml"))
            .map(|(_, hash)| hash.clone())
            .unwrap_or_else(|| "<no-manifest>".to_string());
        Ok(fs_tx::PublicationOutcome::LateCollision {
            target: destination.to_path_buf(),
            new_manifest_sha256: new_manifest_hash,
            existing_manifest_sha256: Some(existing_manifest_hash),
        })
    }
}

fn find_richness_manifest_in_dir(dir: &Path) -> Result<PathBuf, BuildRichnessV1Error> {
    let entries = std::fs::read_dir(dir).map_err(|err| BuildRichnessV1Error::Io {
        path: dir.to_path_buf(),
        message: format!("read_dir for manifest: {err}"),
    })?;
    for entry in entries {
        let entry = entry.map_err(|err| BuildRichnessV1Error::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir entry: {err}"),
        })?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.ends_with(".manifest.toml") {
            return Ok(entry.path());
        }
    }
    Err(BuildRichnessV1Error::IncompleteDestination {
        target: dir.to_path_buf(),
        reason: "no manifest found at existing destination".to_string(),
    })
}

// ── Canonical manifest ────────────────────────────────────────────────────

fn build_richness_canonical_manifest(
    staging: &Path,
    _name: &str,
    doc: &RichnessDocumentV1,
    output: &bsp_generator::RichnessPipelineOutput,
    compile_result: &bsp::CompileResult,
    staged_hashes: &[(String, String)],
) -> Result<String, BuildRichnessV1Error> {
    use toml::Value;

    let mut root = toml::Table::new();
    root.insert("format_version".into(), Value::Integer(1));
    root.insert(
        "manifest_schema".into(),
        Value::String("engine-pack-canonical/1".into()),
    );
    root.insert("strict".into(), Value::Boolean(true));

    // ── Generator identity ────────────────────────────────────
    {
        let mut gen = toml::Table::new();
        gen.insert(
            "generator".into(),
            Value::String("bsp_generator/enhanced_v3/richness/v1".into()),
        );
        gen.insert(
            "schema_version".into(),
            Value::String(output.request_metadata.schema_version().into()),
        );
        gen.insert("seed".into(), Value::Integer(doc.seed() as i64));
        gen.insert("preset".into(), Value::String(doc.preset().tag().into()));
        gen.insert("theme".into(), Value::String(doc.theme().tag().into()));
        gen.insert("extent".into(), Value::Integer(doc.extent() as i64));

        // Inherited/explicit controls
        {
            let mut controls = toml::Table::new();
            serialize_inherited_or(
                &mut controls,
                "landmarks",
                doc.critical_path_landmarks(),
                |v| Value::Integer(v as i64),
            );
            serialize_inherited_or(&mut controls, "zones", doc.zone_count(), |v| {
                Value::Integer(v as i64)
            });
            serialize_inherited_or(&mut controls, "cave_mode", doc.cave_mode(), |v| {
                Value::String(v.tag().into())
            });
            serialize_inherited_or(
                &mut controls,
                "vertical_openings",
                doc.vertical_openings(),
                |v| Value::Integer(v as i64),
            );
            serialize_inherited_or(&mut controls, "budget_ceiling", doc.budget_ceiling(), |v| {
                Value::Integer(v as i64)
            });
            gen.insert("controls".into(), Value::Table(controls));
        }

        // Revisions
        {
            let mut revisions = toml::Table::new();
            revisions.insert(
                "request_schema".into(),
                Value::String(doc.request_schema_revision().tag().into()),
            );
            revisions.insert(
                "algorithm".into(),
                Value::String(doc.algorithm_revision().tag().into()),
            );
            revisions.insert(
                "content".into(),
                Value::String(doc.content_revision().tag().into()),
            );
            revisions.insert(
                "preset".into(),
                Value::String(doc.preset_revision().tag().into()),
            );
            revisions.insert(
                "theme".into(),
                Value::String(doc.theme_revision().tag().into()),
            );
            revisions.insert(
                "asset".into(),
                Value::String(doc.asset_revision().tag().into()),
            );
            revisions.insert(
                "convention".into(),
                Value::String(doc.convention_revision().tag().into()),
            );
            gen.insert("revisions".into(), Value::Table(revisions));
        }

        root.insert("generator".into(), Value::Table(gen));
    }

    // ── Compiler provenance ───────────────────────────────────
    {
        let mut prov = toml::Table::new();
        prov.insert(
            "compiler_identity".into(),
            Value::String(compile_result.provenance.compiler_identity.clone()),
        );
        prov.insert(
            "compiler_version".into(),
            Value::String(compile_result.provenance.compiler_version.clone()),
        );
        if let Some(ref hashes) = compile_result.provenance.compiler_hashes {
            let mut h = toml::Table::new();
            h.insert(
                "qbsp_sha256".into(),
                Value::String(hashes.qbsp_sha256.clone()),
            );
            h.insert(
                "vis_sha256".into(),
                Value::String(hashes.vis_sha256.clone()),
            );
            h.insert(
                "light_sha256".into(),
                Value::String(hashes.light_sha256.clone()),
            );
            prov.insert("compiler_hashes".into(), Value::Table(h));
        }
        root.insert("compiler_provenance".into(), Value::Table(prov));
    }

    // ── Generation metadata ─────────────────────────────────────
    {
        let meta_bytes = output.generation_metadata.to_canonical_bytes();
        let meta_str = std::str::from_utf8(&meta_bytes).unwrap_or("");
        let mut facts = toml::Table::new();
        for line in meta_str.lines() {
            if let Some((k, v)) = line.split_once(": ") {
                facts.insert(k.to_string(), Value::String(v.to_string()));
            }
        }
        root.insert("generation_facts".into(), Value::Table(facts));
    }

    // ── Semantic identity ──────────────────────────────────────
    {
        let semantic = output.generation_metadata.semantic_identity();
        root.insert(
            "semantic_identity_sha256".into(),
            Value::String(richness_sha256_hex(semantic)),
        );
        let reservation = output.generation_metadata.reservation_fingerprint();
        root.insert(
            "reservation_fingerprint_sha256".into(),
            Value::String(richness_sha256_hex(reservation)),
        );
    }

    // ── Published artifacts ───────────────────────────────────
    {
        let mut artifacts: Vec<Value> = Vec::new();
        let wad_basename = doc.theme();
        theme_definition_for(wad_basename).wad_filename;
        for (rel_path, sha256) in staged_hashes {
            if rel_path.ends_with(".manifest.toml") || rel_path == fs_tx::STAGING_MARKER_NAME {
                continue;
            }
            let artifact_path = staging.join(rel_path);
            let metadata = std::fs::symlink_metadata(&artifact_path).map_err(|err| {
                BuildRichnessV1Error::Io {
                    path: artifact_path.clone(),
                    message: format!("metadata: {err}"),
                }
            })?;
            let file_bytes = metadata.len();
            let mut entry = toml::Table::new();
            entry.insert("path".into(), Value::String(rel_path.clone()));
            entry.insert("sha256".into(), Value::String(sha256.clone()));
            entry.insert(
                "bytes".into(),
                Value::Integer(i64::try_from(file_bytes).map_err(|_| {
                    BuildRichnessV1Error::PbrClosure(format!(
                        "artifact '{rel_path}' exceeds TOML integer range"
                    ))
                })?),
            );
            let kind = if rel_path.ends_with(".bsp") {
                "bsp"
            } else if rel_path.ends_with(".lit") {
                "lit"
            } else if rel_path.ends_with(".lmp") {
                "palette"
            } else if rel_path.ends_with(".wad") {
                "wad"
            } else if rel_path.ends_with(".map") {
                "map_source"
            } else if rel_path.ends_with(".request.json") {
                "request"
            } else if rel_path.ends_with(".generation.txt") {
                "generation_metadata"
            } else if rel_path.starts_with("textures/") {
                "texture_companion"
            } else if rel_path == "metadata.json" {
                "metadata"
            } else {
                "unknown"
            };
            entry.insert("kind".into(), Value::String(kind.into()));
            artifacts.push(Value::Table(entry));
        }
        artifacts.sort_by(|a, b| {
            let a_path = a.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let b_path = b.get("path").and_then(|v| v.as_str()).unwrap_or("");
            a_path.cmp(b_path)
        });
        root.insert("published_artifacts".into(), Value::Array(artifacts));
    }

    // ── Asset roles ────────────────────────────────────────────
    {
        let roles: Vec<Value> = output
            .asset_roles
            .iter()
            .map(|r| Value::String(r.clone()))
            .collect();
        root.insert("asset_roles".into(), Value::Array(roles));
    }

    // ── Theme identity ─────────────────────────────────────────
    {
        let theme_def = theme_definition_for(doc.theme());
        let theme_dir = richness_theme_dir(doc.theme())?;
        let wad_path = theme_dir.join(theme_def.wad_filename);
        let palette_path = theme_dir.join(theme_def.palette_filename);

        let wad_hash = if wad_path.exists() {
            let wad_bytes = std::fs::read(&wad_path).map_err(|err| BuildRichnessV1Error::Io {
                path: wad_path.clone(),
                message: format!("read wad for manifest: {err}"),
            })?;
            richness_sha256_hex(&wad_bytes)
        } else {
            String::new()
        };
        let palette_hash = if palette_path.exists() {
            let pal_bytes =
                std::fs::read(&palette_path).map_err(|err| BuildRichnessV1Error::Io {
                    path: palette_path.clone(),
                    message: format!("read palette for manifest: {err}"),
                })?;
            richness_sha256_hex(&pal_bytes)
        } else {
            String::new()
        };

        let mut theme = toml::Table::new();
        theme.insert("name".into(), Value::String(theme_def.dir_name.into()));
        theme.insert("wad_sha256".into(), Value::String(wad_hash));
        theme.insert("palette_sha256".into(), Value::String(palette_hash));
        root.insert("theme_identity".into(), Value::Table(theme));
    }

    let doc = toml::Value::Table(root);
    toml::to_string_pretty(&doc)
        .map_err(|err| BuildRichnessV1Error::PbrClosure(format!("serialize manifest: {err}")))
}

fn serialize_inherited_or<T: std::fmt::Display>(
    table: &mut toml::Table,
    key: &str,
    value: InheritedOr<T>,
    to_toml: impl FnOnce(T) -> toml::Value,
) {
    match value {
        InheritedOr::Inherited => {
            let mut inner = toml::Table::new();
            inner.insert("source".into(), toml::Value::String("inherited".into()));
            table.insert(key.into(), toml::Value::Table(inner));
        }
        InheritedOr::Explicit(v) => {
            let mut inner = toml::Table::new();
            inner.insert("source".into(), toml::Value::String("explicit".into()));
            inner.insert("value".into(), to_toml(v));
            table.insert(key.into(), toml::Value::Table(inner));
        }
    }
}

fn richness_sha256_hex(data: &[u8]) -> String {
    richness_assets::sha256_hex(data)
}

// ── PNG dimension parsing (embedded, no external PNG decoder) ─────────────

fn parse_png_dimensions_richness(bytes: &[u8]) -> Result<(u32, u32), String> {
    const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < SIGNATURE.len() || &bytes[..SIGNATURE.len()] != SIGNATURE {
        return Err("missing PNG signature".to_string());
    }
    let mut offset = SIGNATURE.len();
    let mut dimensions = None;
    let mut saw_idat = false;
    let mut saw_iend = false;
    while offset < bytes.len() {
        let header_end = offset
            .checked_add(8)
            .ok_or_else(|| "chunk header overflow".to_string())?;
        if header_end > bytes.len() {
            return Err("truncated chunk header".to_string());
        }
        let length = u32::from_be_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| "invalid chunk length".to_string())?,
        ) as usize;
        let kind = &bytes[offset + 4..header_end];
        let data_start = header_end;
        let data_end = data_start
            .checked_add(length)
            .ok_or_else(|| "chunk length overflow".to_string())?;
        let chunk_end = data_end
            .checked_add(4)
            .ok_or_else(|| "CRC offset overflow".to_string())?;
        if chunk_end > bytes.len() {
            return Err("truncated chunk data".to_string());
        }
        match kind {
            b"IHDR" if dimensions.is_none() && offset == SIGNATURE.len() => {
                if length != 13 {
                    return Err("IHDR must be exactly 13 bytes".to_string());
                }
                let width =
                    u32::from_be_bytes(bytes[data_start..data_start + 4].try_into().unwrap());
                let height =
                    u32::from_be_bytes(bytes[data_start + 4..data_start + 8].try_into().unwrap());
                if width == 0 || height == 0 {
                    return Err("IHDR dimensions must be non-zero".to_string());
                }
                dimensions = Some((width, height));
            }
            b"IDAT" if dimensions.is_some() && !saw_iend => saw_idat = true,
            b"IEND" if dimensions.is_some() && !saw_iend => {
                if length != 0 {
                    return Err("IEND must be empty".to_string());
                }
                saw_iend = true;
                if chunk_end != bytes.len() {
                    return Err("trailing bytes after IEND".to_string());
                }
            }
            b"IHDR" => return Err("IHDR must be the first chunk".to_string()),
            _ if saw_iend => return Err("chunk after IEND".to_string()),
            _ => {}
        }
        offset = chunk_end;
    }
    if !saw_idat {
        return Err("no IDAT chunk".to_string());
    }
    if !saw_iend {
        return Err("no IEND chunk".to_string());
    }
    dimensions.ok_or_else(|| "no IHDR chunk".to_string())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_richness_v1_package_with_minimal_doc_produces_valid_manifest() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .expect("valid doc");
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tmp.path().join("out");

        let result = build_richness_v1_package(
            &doc,
            &out,
            default_ericw_tools_dir().as_deref(),
            "test_richness",
            None,
        );

        match result {
            Ok(BuildRichnessV1Result::Published { message, .. }) => {
                assert!(message.contains("published"));
                assert!(out.join("test_richness.bsp").exists());
                assert!(out.join("test_richness.manifest.toml").exists());
                assert!(out.join("test_richness.map").exists());
                assert!(out.join("palette.lmp").exists());
                assert!(out.join("metadata.json").exists());
            }
            Ok(BuildRichnessV1Result::Unchanged { .. }) => {
                // This can happen in CI if the same seed was already published
            }
            Err(e) => {
                // May fail if ericw-tools not available
                let msg = e.to_string();
                if !msg.contains("compilation failed") && !msg.contains("not found") {
                    panic!("unexpected error: {msg}");
                }
            }
        }
    }

    #[test]
    fn build_richness_v1_is_deterministic() {
        let doc =
            RichnessDocumentV1::new(7, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient)
                .expect("valid doc");
        let tmp = tempfile::tempdir().expect("tempdir");
        let out1 = tmp.path().join("out1");
        let out2 = tmp.path().join("out2");

        let tools = default_ericw_tools_dir();
        let r1 = build_richness_v1_package(&doc, &out1, tools.as_deref(), "test_det", None);
        let r2 = build_richness_v1_package(&doc, &out2, tools.as_deref(), "test_det", None);

        match (r1, r2) {
            (
                Ok(BuildRichnessV1Result::Published { .. }),
                Ok(BuildRichnessV1Result::Published { .. }),
            ) => {
                let bsp1 = std::fs::read(out1.join("test_det.bsp")).unwrap();
                let bsp2 = std::fs::read(out2.join("test_det.bsp")).unwrap();
                assert_eq!(bsp1, bsp2);
            }
            (Err(e1), Err(e2)) => {
                let m1 = e1.to_string();
                let m2 = e2.to_string();
                if !m1.contains("compilation") && !m1.contains("not found") {
                    panic!("unexpected error: {m1}");
                }
                if !m2.contains("compilation") && !m2.contains("not found") {
                    panic!("unexpected error: {m2}");
                }
            }
            _ => {
                // toolchain may or may not be available
            }
        }
    }
}

/// Locate default ericw-tools installation path (same as main.rs).
fn default_ericw_tools_dir() -> Option<PathBuf> {
    let candidate = PathBuf::from(std::env::var_os("HOME")?)
        .join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin");
    ["qbsp", "vis", "light"]
        .iter()
        .all(|executable| candidate.join(executable).is_file())
        .then_some(candidate)
}
