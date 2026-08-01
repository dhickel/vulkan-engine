//! Enhanced V3 dungeon package candidate builder.
//!
//! Takes a validated full `V3Config` and output directory, with a legacy
//! seed/preset/extent wrapper for byte-compatible callers. Generates a `.map`
//! via `bsp_generator::generate_enhanced_v3`, compiles it through
//! ericw-tools, and collects BSP+LIT+WAD+palette+metadata into a validated
//! closure published atomically through a same-filesystem no-replace transaction.
//!
//! Publication is fail-closed: a pre-existing valid destination with identical
//! content is a no-op; a different valid destination is a LateCollision; any
//! incomplete, malformed, or non-directory destination is rejected without
//! modification.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use bsp_generator::enhanced_v3::{V3Config, V3Preset};

use crate::compiler;
use crate::fs_tx;

/// Outcome of a V3 package publication.
#[derive(Debug)]
pub enum BuildV3Result {
    /// Staging was renamed atomically into a new destination.
    Published { target: PathBuf, message: String },
    /// Destination already held an identical validated closure — no-op.
    Unchanged { target: PathBuf, message: String },
}

/// Build and publish a V3 dungeon package candidate through an atomic
/// no-replace transaction.
///
/// On success, returns a `BuildV3Result` describing the outcome (published or
/// unchanged). On failure, the destination is never created, modified, or
/// partially populated.
#[allow(clippy::too_many_arguments)]
pub fn build_v3_package(
    seed: u64,
    preset: V3Preset,
    xy_extent: u32,
    out_dir: &Path,
    tool_path: Option<&Path>,
    name: &str,
    profile_override: Option<&str>,
) -> Result<BuildV3Result, BuildV3Error> {
    let config = V3Config::new(seed, preset, xy_extent).map_err(BuildV3Error::Config)?;
    build_v3_package_from_config(&config, out_dir, tool_path, name, profile_override)
}

/// Build and publish a package from a fully validated EnhancedV3 explorer
/// configuration. This is the runtime explorer entry point; the legacy
/// seed/preset/extent wrapper above remains byte-compatible.
pub fn build_v3_package_from_config(
    config: &V3Config,
    out_dir: &Path,
    tool_path: Option<&Path>,
    name: &str,
    profile_override: Option<&str>,
) -> Result<BuildV3Result, BuildV3Error> {
    config.validate().map_err(BuildV3Error::Config)?;
    let seed = config.seed;
    let preset = config.preset;

    // ── 1. Generate .map ─────────────────────────────────────────────
    let (map_text, meta) =
        bsp_generator::generate_enhanced_v3(config).map_err(BuildV3Error::Generation)?;

    // ── 3. Resolve compiler profile ──────────────────────────────────
    let profile_content = if let Some(profile_path) = profile_override {
        std::fs::read_to_string(profile_path).map_err(|err| BuildV3Error::Io {
            path: PathBuf::from(profile_path),
            message: format!("read profile: {err}"),
        })?
    } else {
        DEFAULT_BSP2_PROFILE.to_string()
    };
    let profile = compiler::parse_compiler_profile(&profile_content)
        .map_err(|msg| BuildV3Error::Profile(msg))?;

    // ── 4. Resolve theme assets ─────────────────────────────────────
    let theme_dir = cc0_dungeon_v2_dir()?;
    let palette_path = theme_dir.join("palette.lmp");
    let wad_path = theme_dir.join("cc0_dungeon_v2.wad");

    for input in [&palette_path, &wad_path] {
        compiler::validate_input_regular(input).map_err(|err| BuildV3Error::Input {
            path: input.clone(),
            message: format!("invalid theme asset: {err}"),
        })?;
    }

    // ── 5. Recover orphaned staging ─────────────────────────────────
    fs_tx::recover_orphaned_staging(out_dir);

    // ── 6. Create staging ───────────────────────────────────────────
    let staging = fs_tx::create_staging_sibling(out_dir)?;

    // Write ownership marker for orphan recovery
    fs_tx::write_staging_marker(&staging, out_dir)?;

    let result = (|| -> Result<BuildV3Result, BuildV3Error> {
        // Write .map source into staging
        let map_filename = format!("{name}.map");
        let map_path = staging.join(&map_filename);
        std::fs::write(&map_path, &map_text).map_err(|err| BuildV3Error::Io {
            path: map_path.clone(),
            message: format!("write .map: {err}"),
        })?;

        // Compile
        let work_dir = staging.join(".compile-work");
        std::fs::create_dir_all(&work_dir).map_err(|err| BuildV3Error::Io {
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
        .map_err(|err| BuildV3Error::Compilation(format!("{err}")))?;

        // Clean up work directory
        let _ = std::fs::remove_dir_all(&work_dir);

        // Stage compiled .bsp
        let bsp_path = staging.join(format!("{name}.bsp"));
        std::fs::write(&bsp_path, &compile_result.bsp_data).map_err(|err| BuildV3Error::Io {
            path: bsp_path.clone(),
            message: format!("write .bsp: {err}"),
        })?;

        // Stage .lit companion
        if let Some(ref lit_data) = compile_result.lit_data {
            let lit_path = staging.join(format!("{name}.lit"));
            std::fs::write(&lit_path, lit_data).map_err(|err| BuildV3Error::Io {
                path: lit_path.clone(),
                message: format!("write .lit: {err}"),
            })?;
        }

        // Stage palette
        let palette_bytes = std::fs::read(&palette_path).map_err(|err| BuildV3Error::Io {
            path: palette_path.clone(),
            message: format!("read palette: {err}"),
        })?;
        let palette_staged = staging.join("palette.lmp");
        std::fs::write(&palette_staged, &palette_bytes).map_err(|err| BuildV3Error::Io {
            path: palette_staged.clone(),
            message: format!("write palette: {err}"),
        })?;

        // Stage WAD
        let wad_basename = wad_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("cc0_dungeon_v2.wad");
        let wad_staged = staging.join(wad_basename);
        std::fs::copy(&wad_path, &wad_staged).map_err(|err| BuildV3Error::Io {
            path: wad_path.clone(),
            message: format!("copy WAD: {err}"),
        })?;

        // Stage PBR companion textures
        let staged_pbr = stage_pbr_companions_v3(
            &staging,
            name,
            &compile_result,
            &[wad_path.clone()],
            &palette_bytes,
        )?;
        require_complete_enhanced_pbr_closure_v3(&staged_pbr)?;

        // Validate staged closure through isolated strict authorization
        validate_staged_authorized_import_v3(
            &staging,
            name,
            compile_result.lit_data.is_some(),
            &staged_pbr
                .required_wad_basenames
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
        )?;

        // Write metadata.json
        let metadata = serde_json::json!({
            "format_version": 1,
            "schema_version": meta.schema_version(),
            "generator": meta.generator(),
            "seed": meta.seed(),
            "preset": meta.preset(),
            "xy_extent": meta.xy_extent(),
            "config": v3_config_json(config),
            "output": {
                "room_count": meta.room_count(),
                "lower_room_count": meta.lower_room_count(),
                "upper_room_count": meta.upper_room_count(),
                "portal_count": meta.portal_count(),
                "transition_count": meta.transition_count(),
                "route_count": meta.route_count(),
                "spawn_origin": meta.spawn_origin(),
                "light_count": meta.light_count(),
                "bounds": meta.bounds(),
                "actual_faces": meta.actual_faces(),
                "actual_brushes": meta.actual_brushes(),
                "actual_entities": meta.actual_entities(),
                "has_upper_layer": meta.has_upper_layer(),
                "identity_satisfied": meta.identity_satisfied(),
                "face_budget_satisfied": meta.face_budget_satisfied(),
                "entity_budget_satisfied": meta.entity_budget_satisfied(),
            },
            "compiler": {
                "identity": compile_result.provenance.compiler_identity,
                "version": compile_result.provenance.compiler_version,
            },
            "map_filename": map_filename,
        });
        let metadata_path = staging.join("metadata.json");
        std::fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&metadata).unwrap(),
        )
        .map_err(|err| BuildV3Error::Io {
            path: metadata_path,
            message: format!("write metadata: {err}"),
        })?;

        // ── Build canonical manifest ───────────────────────────────
        // The manifest must be generated before marker removal so
        // staged_hashes includes the marker for completeness, but the
        // marker is excluded from the manifest's artifact list.
        let staged_hashes = fs_tx::compute_dir_file_hashes(&staging)?;
        let manifest_toml = build_v3_canonical_manifest(
            &staging,
            name,
            &compile_result,
            config,
            &meta,
            &staged_hashes,
        )?;
        let manifest_path = staging.join(format!("{name}.manifest.toml"));
        std::fs::write(&manifest_path, &manifest_toml).map_err(|err| BuildV3Error::Io {
            path: manifest_path.clone(),
            message: format!("write manifest: {err}"),
        })?;

        // Remove staging marker before final closure validation and publication
        fs_tx::remove_staging_marker(&staging)?;

        // Validate manifest closure
        fs_tx::validate_manifest_closure(&staging, manifest_toml.as_bytes())?;

        // Publish atomically via same-filesystem no-replace primitive
        let publish_result = publish_or_resolve_collision_v3(&staging, out_dir)?;

        match publish_result {
            fs_tx::PublicationOutcome::Published { target, .. } => {
                Ok(BuildV3Result::Published {
                    message: format!(
                        "published[enhanced-dungeon-v3]: seed={seed} preset={} -> {}/ ({} rooms, {} portals, {} transitions)",
                        preset.tag(),
                        out_dir.display(),
                        meta.room_count(),
                        meta.portal_count(),
                        meta.transition_count(),
                    ),
                    target,
                })
            }
            fs_tx::PublicationOutcome::Unchanged { target, .. } => {
                Ok(BuildV3Result::Unchanged {
                    message: format!(
                        "unchanged[enhanced-dungeon-v3]: seed={seed} preset={} -> {}/ ({} rooms, {} portals, {} transitions)",
                        preset.tag(),
                        out_dir.display(),
                        meta.room_count(),
                        meta.portal_count(),
                        meta.transition_count(),
                    ),
                    target,
                })
            }
            fs_tx::PublicationOutcome::LateCollision {
                target,
                new_manifest_sha256,
                existing_manifest_sha256,
            } => Err(BuildV3Error::LateCollision {
                target,
                new_manifest_sha256,
                existing_manifest_sha256,
            }),
            fs_tx::PublicationOutcome::IncompleteDestination { target, reason } => {
                Err(BuildV3Error::IncompleteDestination { target, reason })
            }
        }
    })();

    if result.is_err() {
        fs_tx::cleanup_staging(&staging);
    }
    result
}

// ── Error type ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum BuildV3Error {
    Config(bsp_generator::enhanced_v3::V3Error),
    Generation(bsp_generator::enhanced_v3::V3Error),
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

impl std::fmt::Display for BuildV3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(e) => write!(f, "invalid V3 config: {e}"),
            Self::Generation(e) => write!(f, "V3 generation failed: {e}"),
            Self::Io { path, message } => write!(f, "I/O error at '{}': {message}", path.display()),
            Self::Profile(msg) => write!(f, "invalid compiler profile: {msg}"),
            Self::Input { path, message } => {
                write!(f, "invalid input '{}': {message}", path.display())
            }
            Self::FsTx(e) => write!(f, "fs_tx error: {e}"),
            Self::Compilation(msg) => write!(f, "compilation failed: {msg}"),
            Self::PbrClosure(msg) => write!(f, "PBR closure: {msg}"),
            Self::PackageValidation(msg) => write!(f, "package validation: {msg}"),
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

impl std::error::Error for BuildV3Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(e) => Some(e),
            Self::Generation(e) => Some(e),
            Self::FsTx(e) => Some(e),
            _ => None,
        }
    }
}

impl From<fs_tx::FsTxError> for BuildV3Error {
    fn from(e: fs_tx::FsTxError) -> Self {
        Self::FsTx(e)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────

/// Default compiler profile bundled with engine_pack.
const DEFAULT_BSP2_PROFILE: &str =
    include_str!("../../bsp_authoring/ericw-q1-bsp2-generated-profile.toml");

/// Resolve the CC0 Dungeon v2 theme directory.
fn cc0_dungeon_v2_dir() -> Result<PathBuf, BuildV3Error> {
    let engine_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| BuildV3Error::Io {
            path: PathBuf::from("<manifest>"),
            message: "engine_pack manifest directory is not under the workspace tools directory"
                .into(),
        })?;
    Ok(engine_root.join("src/bsp_generator/themes/cc0_dungeon_v2"))
}

fn v3_config_json(config: &V3Config) -> serde_json::Value {
    let mut value = serde_json::json!({
        "seed": config.seed,
        "preset": config.preset.tag(),
        "xy_extent": config.xy_extent,
    });
    if config.has_overrides() {
        value
            .as_object_mut()
            .expect("V3 config JSON root is an object")
            .insert(
                "overrides".into(),
                serde_json::json!({
                    "rooms": config.rooms,
                    "corridors": config.corridors,
                    "loops": config.loops,
                    "vertical_edges": config.vertical_edges,
                    "chamfer": config.chamfer,
                    "arch_type": config.arch_type.tag(),
                    "stairs": config.stairs,
                    "room_span_min": config.room_span_min,
                    "room_span_max": config.room_span_max,
                    "grammar_families": config.grammar_families,
                    "grammar_mode": config.grammar_mode.tag(),
                    "features": config.features.tags(),
                    "feature_density": config.feature_density,
                    "minlight": config.minlight,
                    "light_count": config.light_count,
                }),
            );
    }
    value
}

fn v3_config_override_table(config: &V3Config) -> toml::Table {
    use toml::Value;

    let mut table = toml::Table::new();
    table.insert(
        "rooms".into(),
        Value::Integer(config.effective_rooms() as i64),
    );
    table.insert(
        "rooms_explicit".into(),
        Value::Boolean(config.rooms.is_some()),
    );
    table.insert(
        "corridors".into(),
        Value::Integer(config.effective_corridors() as i64),
    );
    table.insert(
        "corridors_explicit".into(),
        Value::Boolean(config.corridors.is_some()),
    );
    table.insert(
        "loops".into(),
        Value::Integer(config.effective_loops() as i64),
    );
    table.insert(
        "loops_explicit".into(),
        Value::Boolean(config.loops.is_some()),
    );
    table.insert(
        "vertical_edges".into(),
        Value::Integer(config.effective_vertical_edges() as i64),
    );
    table.insert(
        "vertical_edges_explicit".into(),
        Value::Boolean(config.vertical_edges.is_some()),
    );
    table.insert("chamfer".into(), Value::Boolean(config.chamfer));
    table.insert(
        "arch_type".into(),
        Value::String(config.arch_type.tag().into()),
    );
    table.insert("stairs".into(), Value::Boolean(config.stairs));
    table.insert(
        "room_span_min".into(),
        Value::Integer(config.effective_room_span_min() as i64),
    );
    table.insert(
        "room_span_min_explicit".into(),
        Value::Boolean(config.room_span_min.is_some()),
    );
    table.insert(
        "room_span_max".into(),
        Value::Integer(config.effective_room_span_max() as i64),
    );
    table.insert(
        "room_span_max_explicit".into(),
        Value::Boolean(config.room_span_max.is_some()),
    );
    table.insert(
        "grammar_families".into(),
        Value::Array(
            config
                .grammar_families
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        ),
    );
    table.insert(
        "grammar_mode".into(),
        Value::String(config.grammar_mode.tag().into()),
    );
    table.insert(
        "features".into(),
        Value::Array(
            config
                .features
                .tags()
                .into_iter()
                .map(|tag| Value::String(tag.into()))
                .collect(),
        ),
    );
    table.insert(
        "feature_density".into(),
        Value::Float(config.feature_density as f64),
    );
    table.insert("minlight".into(), Value::Integer(config.minlight as i64));
    table.insert(
        "light_count".into(),
        Value::Integer(config.effective_light_count() as i64),
    );
    table.insert(
        "light_count_explicit".into(),
        Value::Boolean(config.light_count.is_some()),
    );
    table
}

// ── Atomic no-replace publication with collision resolution ───────────────

/// Attempt atomic no-replace publication of a validated staging directory.
///
/// On `PreExistingDestination`, validates the existing destination through
/// the Phase 06 full-closure validator and applies the Phase 01-approved
/// collision-winner policy:
/// - Identical validated closure → `Unchanged` (no-op, idempotent)
/// - Different validated closure → `LateCollision` (blocked)
/// - Incomplete / malformed / non-directory → `IncompleteDestination`
fn publish_or_resolve_collision_v3(
    staging: &Path,
    destination: &Path,
) -> Result<fs_tx::PublicationOutcome, BuildV3Error> {
    match fs_tx::publish_directory_no_replace(staging, destination) {
        Ok(()) => {
            // Compute manifest hash for the published record
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
            resolve_existing_destination_v3(staging, &target)
        }
        Err(e) => Err(BuildV3Error::FsTx(e)),
    }
}

/// Validate an existing destination and compare with the staging candidate.
fn resolve_existing_destination_v3(
    staging: &Path,
    destination: &Path,
) -> Result<fs_tx::PublicationOutcome, BuildV3Error> {
    // Destination must be a real directory (not symlink)
    let dest_meta = match fs_tx::inspect_entry_no_follow(destination) {
        Ok(meta) if meta.is_dir() => meta,
        Ok(_) => {
            return Err(BuildV3Error::PublicationBlocked {
                target: destination.to_path_buf(),
                message: "destination exists but is not a directory".to_string(),
            });
        }
        Err(_) => {
            return Err(BuildV3Error::PublicationBlocked {
                target: destination.to_path_buf(),
                message: "destination cannot be inspected".to_string(),
            });
        }
    };
    let _ = dest_meta; // used for directory check above

    // Find a manifest in the existing destination
    let manifest_path = find_manifest_in_dir(destination)?;

    // Read and validate the existing closure
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|err| BuildV3Error::Io {
        path: manifest_path.clone(),
        message: format!("read existing manifest: {err}"),
    })?;

    let existing_manifest_hash = fs_tx::compute_manifest_sha256(&manifest_bytes);

    // Validate existing destination through the full-closure validator
    let existing_valid = fs_tx::validate_manifest_closure(destination, &manifest_bytes).is_ok();

    if !existing_valid {
        return Ok(fs_tx::PublicationOutcome::IncompleteDestination {
            target: destination.to_path_buf(),
            reason: "existing destination fails full-closure validation".to_string(),
        });
    }

    // Compare staging and destination content
    let identical = fs_tx::artifact_sets_identical(staging, destination).unwrap_or(false);

    if identical {
        Ok(fs_tx::PublicationOutcome::Unchanged {
            target: destination.to_path_buf(),
            manifest_sha256: existing_manifest_hash,
        })
    } else {
        // Compute new manifest hash from staging
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

/// Build a canonical manifest TOML for the v3 closure.
fn build_v3_canonical_manifest(
    staging: &Path,
    _name: &str,
    compile_result: &bsp::CompileResult,
    config: &V3Config,
    meta: &bsp_generator::enhanced_v3::EnhancedV3Metadata,
    staged_hashes: &[(String, String)],
) -> Result<String, BuildV3Error> {
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
            Value::String("bsp_generator/enhanced_v3".into()),
        );
        gen.insert(
            "schema_version".into(),
            Value::String(meta.schema_version().into()),
        );
        gen.insert("seed".into(), Value::Integer(config.seed as i64));
        gen.insert("preset".into(), Value::String(config.preset.tag().into()));
        gen.insert("xy_extent".into(), Value::Integer(config.xy_extent as i64));
        if config.has_overrides() {
            gen.insert(
                "overrides".into(),
                Value::Table(v3_config_override_table(config)),
            );
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

    // ── Published artifacts ───────────────────────────────────
    {
        let mut artifacts: Vec<Value> = Vec::new();
        for (rel_path, sha256) in staged_hashes {
            if rel_path.ends_with(".manifest.toml") || rel_path == fs_tx::STAGING_MARKER_NAME {
                continue;
            }
            let artifact_path = staging.join(rel_path);
            let metadata =
                std::fs::symlink_metadata(&artifact_path).map_err(|err| BuildV3Error::Io {
                    path: artifact_path.clone(),
                    message: format!("metadata: {err}"),
                })?;
            let file_bytes = metadata.len();
            let mut entry = toml::Table::new();
            entry.insert("path".into(), Value::String(rel_path.clone()));
            entry.insert("sha256".into(), Value::String(sha256.clone()));
            entry.insert(
                "bytes".into(),
                Value::Integer(i64::try_from(file_bytes).map_err(|_| {
                    BuildV3Error::PbrClosure(format!(
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

    let doc = toml::Value::Table(root);
    toml::to_string_pretty(&doc)
        .map_err(|err| BuildV3Error::PbrClosure(format!("serialize manifest: {err}")))
}

/// Find a `.manifest.toml` file in a directory.
fn find_manifest_in_dir(dir: &Path) -> Result<PathBuf, BuildV3Error> {
    let entries = std::fs::read_dir(dir).map_err(|err| BuildV3Error::Io {
        path: dir.to_path_buf(),
        message: format!("read_dir for manifest: {err}"),
    })?;
    let mut manifest_path = None;
    for entry in entries {
        let entry = entry.map_err(|err| BuildV3Error::Io {
            path: dir.to_path_buf(),
            message: format!("read_dir entry: {err}"),
        })?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.ends_with(".manifest.toml") {
            manifest_path = Some(entry.path());
            break;
        }
    }
    manifest_path.ok_or_else(|| BuildV3Error::IncompleteDestination {
        target: dir.to_path_buf(),
        reason: "no manifest found at existing destination".to_string(),
    })
}

// ── PBR companion staging (mirrors main.rs patterns) ──────────────────────

struct StagedPbrClosureV3 {
    required_wad_basenames: BTreeSet<String>,
    eligible_identities: BTreeSet<String>,
    staged_companions: BTreeSet<String>,
}

fn require_complete_enhanced_pbr_closure_v3(
    staged: &StagedPbrClosureV3,
) -> Result<(), BuildV3Error> {
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
    Err(BuildV3Error::PbrClosure(format!(
        "incomplete Enhanced PBR companion closure; missing: {missing:?}; unexpected: {unexpected:?}"
    )))
}

/// Stage PBR companion textures from the theme's textures/ directory.
fn stage_pbr_companions_v3(
    staging: &Path,
    _bsp_name: &str,
    compile_result: &bsp::CompileResult,
    wad_paths: &[PathBuf],
    palette_bytes: &[u8],
) -> Result<StagedPbrClosureV3, BuildV3Error> {
    // Strict-load the BSP to get face/texinfo/miptex data
    let mut wad_archives: Vec<(String, Vec<u8>)> = Vec::new();
    for wad_path in wad_paths {
        validate_no_symlink_components(wad_path, "WAD input")?;
        let basename = wad_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.wad")
            .to_string();
        let bytes = std::fs::read(wad_path).map_err(|err| BuildV3Error::Io {
            path: wad_path.clone(),
            message: format!("read WAD: {err}"),
        })?;
        wad_archives.push((basename, bytes));
    }

    let lit_data = compile_result.lit_data.clone();
    let load_options = bsp::LoadOptions {
        strict: true,
        palette: Some(palette_bytes.to_vec()),
        lit_data,
        wad_archives,
        texture_overrides: Vec::new(),
        source_identity: "compile-bsp".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compile_result.bsp_data, &load_options).map_err(|report| {
            BuildV3Error::Compilation(format!("strict-load validation failed: {report}"))
        })?;

    if !world.diagnostics.is_empty() {
        return Err(BuildV3Error::Compilation(format!(
            "strict-load produced diagnostics: {:?}",
            world
                .diagnostics
                .iter()
                .map(|d| d.message.as_str())
                .collect::<Vec<_>>()
        )));
    }

    let slots = bsp::resources::parse_miptex_slots(&world.miptex_data);
    let required_wad_basenames = required_wad_basenames(&world, &slots);

    // Collect eligible identities — only opaque and alpha-mask surfaces
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
            resolve_base_texture_dimensions(&world, &slots, identity)
                .map(|dimensions| (identity.clone(), dimensions))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;

    // Resolve companions from WAD-adjacent textures/ directories
    let textures_dir = staging.join("textures");
    std::fs::create_dir_all(&textures_dir).map_err(|err| BuildV3Error::Io {
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
                        return Err(BuildV3Error::Io {
                            path: search_dir.clone(),
                            message: format!("inspect companion root: {err}"),
                        })
                    }
                };
                if search_metadata.file_type().is_symlink() {
                    return Err(BuildV3Error::PbrClosure(format!(
                        "PBR companion root must not be a symlink: '{}'",
                        search_dir.display()
                    )));
                }
                if !search_metadata.is_dir() {
                    continue;
                }
                validate_no_symlink_components(&search_dir, "PBR companion root")?;
                let all_entries =
                    std::fs::read_dir(&search_dir).map_err(|err| BuildV3Error::Io {
                        path: search_dir.clone(),
                        message: format!("read companion root: {err}"),
                    })?;
                for entry in all_entries {
                    let entry = entry.map_err(|err| BuildV3Error::Io {
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
                // ASCII-insensitive fallback
                let mut fallback: Option<PathBuf> = None;
                let mut ambiguous = false;
                let all_entries =
                    std::fs::read_dir(&search_dir).map_err(|err| BuildV3Error::Io {
                        path: search_dir.clone(),
                        message: format!("read companion root: {err}"),
                    })?;
                for entry in all_entries {
                    let entry = entry.map_err(|err| BuildV3Error::Io {
                        path: search_dir.clone(),
                        message: format!("read companion entry: {err}"),
                    })?;
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.eq_ignore_ascii_case(&expected) && name_str != expected {
                        if fallback.is_some() {
                            ambiguous = true;
                        } else {
                            fallback = Some(entry.path());
                        }
                    }
                }
                if ambiguous {
                    return Err(BuildV3Error::PbrClosure(format!(
                        "ambiguous PBR companion for '{identity}': multiple case-insensitive matches for {suffix}"
                    )));
                }
                if let Some(p) = fallback {
                    found = Some(p);
                    break;
                }
            }

            if let Some(src) = found {
                let expected_dimensions =
                    base_dimensions.get(identity).copied().ok_or_else(|| {
                        BuildV3Error::PbrClosure(format!(
                            "missing base dimensions for companion '{identity}'"
                        ))
                    })?;
                validate_selected_pbr_companion(&src, identity, expected_dimensions)?;

                let dest = textures_dir.join(&expected);
                std::fs::copy(&src, &dest).map_err(|err| BuildV3Error::Io {
                    path: src.clone(),
                    message: format!("copy companion: {err}"),
                })?;
                staged_companions.insert(expected);
            }
        }
    }

    Ok(StagedPbrClosureV3 {
        required_wad_basenames,
        eligible_identities: eligible,
        staged_companions,
    })
}

fn validate_no_symlink_components(path: &Path, label: &str) -> Result<(), BuildV3Error> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|err| BuildV3Error::Io {
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
                return Err(BuildV3Error::PbrClosure(format!(
                    "{label} path contains parent traversal: '{}'",
                    path.display()
                )));
            }
            std::path::Component::Normal(part) => {
                current.push(part);
                let metadata =
                    std::fs::symlink_metadata(&current).map_err(|err| BuildV3Error::Io {
                        path: current.clone(),
                        message: format!("inspect component: {err}"),
                    })?;
                if metadata.file_type().is_symlink() {
                    return Err(BuildV3Error::PbrClosure(format!(
                        "{label} path contains symlink: '{}'",
                        current.display()
                    )));
                }
            }
        }
    }
    Ok(())
}

fn resolve_base_texture_dimensions(
    world: &bsp::world::BspWorld,
    slots: &[bsp::resources::MiptexSlot],
    identity: &str,
) -> Result<(u32, u32), BuildV3Error> {
    for (_, wad) in &world.wad_archives {
        if let Some(bytes) = bsp::wad::read_wad_lump(wad, identity) {
            let info = bsp::wad::parse_miptex_header(bytes).map_err(|err| {
                BuildV3Error::PbrClosure(format!(
                    "invalid base miptex '{identity}' in staged WAD: {err}"
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
                BuildV3Error::PbrClosure(format!("invalid embedded miptex '{identity}': {err}"))
            })?;
            return Ok((info.width, info.height));
        }
    }

    Err(BuildV3Error::PbrClosure(format!(
        "cannot determine dimensions for PBR base texture '{identity}'"
    )))
}

fn validate_selected_pbr_companion(
    path: &Path,
    identity: &str,
    expected_dimensions: (u32, u32),
) -> Result<(), BuildV3Error> {
    compiler::validate_input_regular(path).map_err(|err| BuildV3Error::Input {
        path: path.to_path_buf(),
        message: format!("invalid PBR companion '{}': {err}", path.display()),
    })?;
    let bytes = std::fs::read(path).map_err(|err| BuildV3Error::Io {
        path: path.to_path_buf(),
        message: format!("read companion: {err}"),
    })?;
    let dimensions = parse_png_dimensions_v3(&bytes).map_err(|reason| {
        BuildV3Error::PbrClosure(format!(
            "malformed PBR companion '{}' for '{identity}': {reason}",
            path.display()
        ))
    })?;
    if dimensions != expected_dimensions {
        return Err(BuildV3Error::PbrClosure(format!(
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

fn parse_png_dimensions_v3(bytes: &[u8]) -> Result<(u32, u32), String> {
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
            .ok_or_else(|| "PNG chunk header overflow".to_string())?;
        if header_end > bytes.len() {
            return Err("truncated PNG chunk header".to_string());
        }
        let length = u32::from_be_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| "invalid PNG chunk length".to_string())?,
        ) as usize;
        let kind = &bytes[offset + 4..header_end];
        let data_start = header_end;
        let data_end = data_start
            .checked_add(length)
            .ok_or_else(|| "PNG chunk length overflow".to_string())?;
        let chunk_end = data_end
            .checked_add(4)
            .ok_or_else(|| "PNG CRC offset overflow".to_string())?;
        if chunk_end > bytes.len() {
            return Err("truncated PNG chunk data".to_string());
        }
        let expected_crc = u32::from_be_bytes(
            bytes[data_end..chunk_end]
                .try_into()
                .map_err(|_| "invalid PNG CRC".to_string())?,
        );
        let actual_crc = png_crc32_v3(&bytes[offset + 4..data_end]);
        if actual_crc != expected_crc {
            return Err("PNG chunk CRC mismatch".to_string());
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
                let bit_depth = bytes[data_start + 8];
                let color_type = bytes[data_start + 9];
                if !matches!(bit_depth, 1 | 2 | 4 | 8 | 16)
                    || !matches!(color_type, 0 | 2 | 3 | 4 | 6)
                    || bytes[data_start + 10] != 0
                    || bytes[data_start + 11] != 0
                    || bytes[data_start + 12] > 1
                {
                    return Err("IHDR contains unsupported PNG parameters".to_string());
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
            b"IHDR" => return Err("IHDR must be the first PNG chunk".to_string()),
            _ if saw_iend => return Err("chunk after IEND".to_string()),
            _ => {}
        }
        offset = chunk_end;
    }
    if !saw_idat {
        return Err("PNG has no IDAT chunk".to_string());
    }
    if !saw_iend {
        return Err("PNG has no IEND chunk".to_string());
    }
    dimensions.ok_or_else(|| "PNG has no IHDR chunk".to_string())
}

fn png_crc32_v3(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn required_wad_basenames(
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

fn validate_staged_authorized_import_v3(
    staging: &Path,
    bsp_name: &str,
    has_lit: bool,
    wad_paths: &[String],
) -> Result<(), BuildV3Error> {
    use package_io::budget::BudgetLedger;
    use package_io::PackageRoot;

    let root = PackageRoot::new(staging).map_err(|err| {
        BuildV3Error::PackageValidation(format!("cannot create staged package root: {err}"))
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
        BuildV3Error::PackageValidation(format!("isolated strict package import failed: {err}"))
    })?;

    if !import.world.diagnostics.is_empty() {
        return Err(BuildV3Error::PackageValidation(format!(
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

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_config_construction() {
        let config = V3Config::new(42, V3Preset::Moderate, 2048).expect("valid config");
        assert_eq!(config.seed, 42);
        assert_eq!(config.preset, V3Preset::Moderate);
        assert_eq!(config.xy_extent, 2048);
    }

    #[test]
    fn v3_generation_produces_map() {
        let config = V3Config::new(0, V3Preset::Sparse, 2048).expect("valid config");
        let (map, meta) = bsp_generator::generate_enhanced_v3(&config).expect("generation");
        assert!(!map.is_empty());
        assert!(map.contains("worldspawn"));
        assert_eq!(meta.schema_version(), "v3");
        assert_eq!(meta.preset(), "sparse");
    }

    #[test]
    fn png_parser_rejects_non_png() {
        let not_png = b"not a png file";
        let result = parse_png_dimensions_v3(not_png);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing PNG signature"));
    }

    #[test]
    fn png_parser_reads_valid_png() {
        // Read a real companion PNG from the theme directory
        let theme_dir = cc0_dungeon_v2_dir().expect("theme dir");
        let textures_dir = theme_dir.join("textures");
        if textures_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&textures_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("png") {
                        let bytes = std::fs::read(&path).expect("read PNG");
                        let dims = parse_png_dimensions_v3(&bytes).expect("valid PNG");
                        assert!(dims.0 > 0 && dims.1 > 0);
                        eprintln!("PNG {path}: {dims:?}", path = path.display());
                        return;
                    }
                }
            }
        }
        // Fallback: test signature check only
        let png_sig_ok = b"\x89PNG\r\n\x1a\nXXXX";
        let result = parse_png_dimensions_v3(png_sig_ok);
        assert!(result.is_err()); // needs IHDR, but signature passes
    }
}
