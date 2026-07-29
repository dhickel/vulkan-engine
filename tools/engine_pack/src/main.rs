use std::collections::HashSet;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use renderer::prelude::{
    validate_package_manifest_file, validate_package_manifest_str, validate_project_file,
    validate_scene_file_with_options, PackageManifest, PackageValidationOptions, Project,
    ProjectValidationOptions, SceneValidationOptions, ValidationArea, ValidationDiagnostic,
    ValidationError,
};
use renderer::AssetKind;
use serde::Serialize;

use engine_pack::cli;
use engine_pack::compiler;
use engine_pack::fs_tx;
use engine_pack::fs_tx::{
    build_publication_plan, cleanup_staging, contained_child_no_symlinks,
    create_staging_file_sibling, create_staging_sibling, publish_directory_no_replace,
    publish_staging, replace_file_with_staging, stage_entry, EntryType, PlanEntry, RollbackJournal,
};

type CliResult<T> = Result<T, CliError>;

fn main() {
    let code = match run(env::args().skip(1).collect()) {
        Ok(message) => {
            println!("{message}");
            0
        }
        Err(CliError::Usage(message)) => {
            eprintln!("error[cli.usage]: {message}");
            2
        }
        Err(CliError::FsTx(err)) => {
            eprintln!("error[cli.fs_tx]: {err}");
            1
        }
        Err(CliError::Validation(err)) => {
            print_validation_error(&err);
            1
        }
    };
    std::process::exit(code);
}

fn run(args: Vec<String>) -> CliResult<String> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(CliError::Usage(cli::global_help()));
    };
    let rest = &args[1..];
    if rest.len() == 1 && matches!(rest[0].as_str(), "-h" | "--help") {
        if let Some(help) = cli::command_help(command) {
            return Ok(help);
        }
    }
    match command {
        "validate-package" => validate_package_cmd(rest),
        "validate-project" => validate_project_cmd(rest),
        "validate-scene" => validate_scene_cmd(rest),
        "validate-bsp" => validate_bsp_cmd(rest),
        "compile-bsp" => compile_bsp_cmd(rest),
        "new-app" => new_app_cmd(rest),
        "new-project" => new_project_cmd(rest),
        "new-package" => new_package_cmd(rest),
        "scan-assets" => scan_assets_cmd(rest),
        "add-asset" => add_asset_cmd(rest),
        "pack" => pack_cmd(rest),
        "enhanced-dungeon" => enhanced_dungeon_cmd(rest),
        "-h" | "--help" | "help" => Ok(cli::global_help()),
        other => Err(CliError::Usage(format!(
            "unknown command '{other}'\n\n{}",
            cli::global_help()
        ))),
    }
}

// ---------------------------------------------------------------------------
// new-app — transactional scaffold via staging
// ---------------------------------------------------------------------------

fn new_app_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("new-app", cli::new_app_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let app_id = require_option("--id", &parsed)?;
    let name = require_option("--name", &parsed)?;
    let dir = require_positional(&parsed, "new-app <dir> --id <app_id> --name <display_name>")?;

    if path_exists_no_follow(&dir) {
        return Err(CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "app.path_exists",
                ValidationArea::Project,
                format!("app target already exists: {}", dir.display()),
            )
            .with_path(&dir)
            .with_durable_id(app_id),
        )));
    }

    let engine_root = engine_root_path()?;

    // Stage generated files into a sibling directory and publish only via rename.
    let staging = create_staging_sibling(&dir).map_err(|e| CliError::FsTx(e))?;
    let result = (|| -> CliResult<()> {
        fs::create_dir_all(staging.join("src")).map_err(|e| fs_tx_err("app.io", &staging, e))?;

        let cargo_path = staging.join("Cargo.toml");
        fs::write(&cargo_path, app_cargo_toml(&app_id, &engine_root)?)
            .map_err(|e| fs_tx_err("app.io", &cargo_path, e))?;

        let main_path = staging.join("src/main.rs");
        fs::write(&main_path, app_main_rs(&app_id, &name))
            .map_err(|e| fs_tx_err("app.io", &main_path, e))?;

        let readme_path = staging.join("README.md");
        fs::write(&readme_path, app_readme_md(&app_id, &name))
            .map_err(|e| fs_tx_err("app.io", &readme_path, e))?;

        Ok(())
    })();

    match result {
        Ok(()) => {
            publish_staging(&staging, &dir).map_err(|e| CliError::FsTx(e))?;
            Ok(format!(
                "created[app]: {}",
                dir.join("Cargo.toml").display()
            ))
        }
        Err(err) => {
            cleanup_staging(&staging);
            Err(err)
        }
    }
}

// ---------------------------------------------------------------------------
// Validation commands (unchanged — read-only)
// ---------------------------------------------------------------------------

fn validate_package_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("validate-package", cli::validate_package_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let expected_package_id = parsed.singleton_value("--expected-package-id");
    let project_root = parsed.singleton_value("--project-root");
    let path = require_positional(&parsed, "validate-package <package.toml>")?;

    let base_options = PackageValidationOptions::default().check_source_files(true);
    let options = if let Some(expected_package_id) = expected_package_id {
        base_options.with_expected_package_id(expected_package_id)
    } else {
        base_options
    };

    let records = if let Some(project_root) = project_root {
        let content = std::fs::read_to_string(&path).map_err(|err| {
            CliError::Validation(ValidationError::single(
                ValidationDiagnostic::new(
                    "package.io",
                    ValidationArea::Package,
                    format!("failed to read package manifest: {err}"),
                )
                .with_path(&path),
            ))
        })?;
        validate_package_manifest_str(&content, PathBuf::from(project_root), &options)?
    } else {
        validate_package_manifest_file(&path, &options)?
    };

    Ok(format!(
        "valid[package]: {} ({} assets)",
        path.display(),
        records.len()
    ))
}

fn validate_project_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("validate-project", cli::validate_project_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;
    let path = require_positional(&parsed, "validate-project <engine.project.toml>")?;

    let project = validate_project_and_startup_scene(&path)?;
    Ok(format!(
        "valid[project]: {} ({})",
        path.display(),
        project.project_id
    ))
}

fn validate_scene_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("validate-scene", cli::validate_scene_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;
    let project = require_option("--project", &parsed)?;
    let path = require_positional(
        &parsed,
        "validate-scene <scene.engine.scene.json> --project <engine.project.toml>",
    )?;

    let asset_ids = collect_project_asset_ids(&PathBuf::from(project))?;
    validate_scene_file_with_options(
        &path,
        &SceneValidationOptions::default().with_known_asset_ids(asset_ids),
    )?;
    Ok(format!("valid[scene]: {}", path.display()))
}

// ---------------------------------------------------------------------------
// validate-bsp — re-parse .bsp through the bsp parser
// ---------------------------------------------------------------------------

fn validate_bsp_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("validate-bsp", cli::validate_bsp_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let strict = parsed.flag_present("--strict");
    let palette_path = parsed.singleton_value("--palette");
    let path = require_positional(
        &parsed,
        "validate-bsp <file.bsp> [--palette <file.lmp>] [--strict]",
    )?;

    fs_tx::inspect_entry_no_follow(&path).map_err(CliError::FsTx)?;
    let bsp_data = std::fs::read(&path).map_err(|err| io_error("bsp.io", &path, err))?;

    let palette_data = palette_path
        .map(|pal_path| {
            let pal = PathBuf::from(pal_path);
            std::fs::read(&pal).map_err(|err| io_error("bsp.io", &pal, err))
        })
        .transpose()?;

    let options = bsp::LoadOptions {
        strict,
        palette: palette_data,
        lit_data: None,
        wad_archives: Vec::new(),
        texture_overrides: Vec::new(),
        source_identity: path.display().to_string(),
    };

    let world = bsp::BspLoader::load(&bsp_data, &options).map_err(|report| {
        CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "bsp.validation",
                ValidationArea::Asset,
                format!("BSP validation failed: {report}"),
            )
            .with_path(&path),
        ))
    })?;

    let diag_count = world.diagnostics.len();
    let profile_name = match world.profile {
        bsp::profile::BspProfile::Bsp29 => "BSP29",
        bsp::profile::BspProfile::Bsp2 => "BSP2",
    };

    Ok(format!(
        "valid[bsp]: {} ({}) {} entities, {} faces, {} diagnostics",
        path.display(),
        profile_name,
        world.entities.len(),
        world.faces.len(),
        diag_count,
    ))
}

// ---------------------------------------------------------------------------
// compile-bsp — shell-free external compiler invocation
// ---------------------------------------------------------------------------

fn compile_bsp_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("compile-bsp", cli::compile_bsp_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let profile_path = require_option("--profile", &parsed)?;
    let out_dir = require_option("--out", &parsed)?;
    let palette_path = parsed.singleton_value("--palette");
    let tool_path = parsed.singleton_value("--tool-path");
    let wad_paths: Vec<PathBuf> = parsed
        .repeated_values("--wad")
        .iter()
        .map(PathBuf::from)
        .collect();
    let source_map = require_positional(
        &parsed,
        "compile-bsp <source.map> --profile <profile.toml> --out <dir>",
    )?;

    // Read and parse compiler profile
    let profile_content = std::fs::read_to_string(&profile_path)
        .map_err(|err| io_error("compile-bsp.profile", Path::new(&profile_path), err))?;
    let profile = compiler::parse_compiler_profile(&profile_content).map_err(|msg| {
        CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "bsp.profile",
                ValidationArea::Project,
                format!("invalid compiler profile: {msg}"),
            )
            .with_path(&profile_path),
        ))
    })?;

    // Determine profile family and exact identity
    let profile_family = "q1-portable-ericw";
    let exact_profile_name = profile.name.clone();
    let profile_sha256 = compiler::sha256_file(Path::new(&profile_path)).map_err(|err| {
        CliError::FsTx(fs_tx::FsTxError::StagingArtifactInvariant {
            staging: PathBuf::from(&profile_path),
            message: format!("cannot hash profile: {err}"),
        })
    })?;

    // Resolve palette path
    let palette = if let Some(pal_path) = &palette_path {
        PathBuf::from(pal_path)
    } else {
        // Default: look for palette.lmp in same directory as profile
        let profile_dir = Path::new(&profile_path)
            .parent()
            .unwrap_or_else(|| Path::new("."));
        profile_dir.join("palette.lmp")
    };

    // Validate source inputs are regular non-symlink files
    compiler::validate_input_regular(&source_map).map_err(|err| {
        CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "bsp.compile",
                ValidationArea::Asset,
                format!("invalid source map: {err}"),
            )
            .with_path(&source_map),
        ))
    })?;
    compiler::validate_input_regular(&palette).map_err(|err| {
        CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "bsp.compile",
                ValidationArea::Asset,
                format!("invalid palette: {err}"),
            )
            .with_path(&palette),
        ))
    })?;
    validate_no_symlink_path_components(&source_map, "source map")?;
    validate_no_symlink_path_components(&palette, "palette")?;
    for wad in &wad_paths {
        compiler::validate_input_regular(wad).map_err(|err| {
            CliError::Validation(ValidationError::single(
                ValidationDiagnostic::new(
                    "bsp.compile",
                    ValidationArea::Asset,
                    format!("invalid WAD: {err}"),
                )
                .with_path(wad),
            ))
        })?;
        validate_no_symlink_path_components(wad, "WAD input")?;
    }

    let out_dir = PathBuf::from(&out_dir);
    let destination_exists = path_exists_no_follow(&out_dir);

    // Recover any orphaned staging directories before starting
    if !destination_exists {
        fs_tx::recover_orphaned_staging(&out_dir);
    }

    let bsp_name = source_map
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    // Use fs_tx staging for output
    let staging = create_staging_sibling(&out_dir).map_err(CliError::FsTx)?;
    // Write ownership marker
    fs_tx::write_staging_marker(&staging, &out_dir).map_err(CliError::FsTx)?;

    let result = (|| -> CliResult<String> {
        let work_dir = staging.join(".compile-work");
        std::fs::create_dir_all(&work_dir)
            .map_err(|err| io_error("compile-bsp.workdir", &work_dir, err))?;

        let tool_path_opt = tool_path.map(PathBuf::from);
        let compile_result = compiler::compile_map(
            &source_map,
            &profile,
            &work_dir,
            &palette,
            tool_path_opt.as_deref(),
            &wad_paths,
        )
        .map_err(|err| {
            CliError::Validation(ValidationError::single(
                ValidationDiagnostic::new(
                    "bsp.compile",
                    ValidationArea::Asset,
                    format!("compilation failed: {err}"),
                )
                .with_path(&source_map),
            ))
        })?;

        // Remove compiler work directory; clean up after successful compile
        let _ = std::fs::remove_dir_all(&work_dir);

        // ── Stage BSP and optional .lit ───────────────────────────
        let bsp_path = staging.join(format!("{bsp_name}.bsp"));
        std::fs::write(&bsp_path, &compile_result.bsp_data)
            .map_err(|err| io_error("compile-bsp.write", &bsp_path, err))?;

        if let Some(ref lit_data) = compile_result.lit_data {
            let lit_path = staging.join(format!("{bsp_name}.lit"));
            std::fs::write(&lit_path, lit_data)
                .map_err(|err| io_error("compile-bsp.write", &lit_path, err))?;
        }

        // ── Stage palette ─────────────────────────────────────────
        let palette_bytes = std::fs::read(&palette)
            .map_err(|err| io_error("compile-bsp.palette", &palette, err))?;
        let palette_staged = staging.join("palette.lmp");
        std::fs::write(&palette_staged, &palette_bytes)
            .map_err(|err| io_error("compile-bsp.write", &palette_staged, err))?;

        // ── Stage WADs ────────────────────────────────────────────
        // Compiler input order remains intact here. After strict resolution
        // determines the actual closure, unused archives are removed before
        // the manifest is written.
        let mut seen_wad_basenames = HashSet::new();
        let staged_wad_basenames = wad_paths
            .iter()
            .map(|wad_path| {
                let basename = wad_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| {
                        CliError::Validation(internal_error(
                            "bsp.compile",
                            format!("WAD has no valid filename: '{}'", wad_path.display()),
                        ))
                    })?;
                if !seen_wad_basenames.insert(basename.to_string()) {
                    return Err(CliError::Validation(internal_error(
                        "bsp.compile",
                        format!("duplicate WAD basename in package closure: '{basename}'"),
                    )));
                }
                let dest = staging.join(basename);
                std::fs::copy(wad_path, &dest)
                    .map_err(|err| io_error("compile-bsp.copy_wad", wad_path, err))?;
                Ok(basename.to_string())
            })
            .collect::<CliResult<Vec<_>>>()?;

        // ── Stage PBR companion textures and trim WAD closure ─────
        let staged_pbr = stage_pbr_companions(
            &staging,
            bsp_name,
            &compile_result,
            &wad_paths,
            &palette_bytes,
        )?;
        let selected_wad_basenames = staged_wad_basenames
            .iter()
            .filter(|basename| staged_pbr.required_wad_basenames.contains(*basename))
            .cloned()
            .collect::<Vec<_>>();
        for basename in staged_wad_basenames
            .iter()
            .filter(|basename| !staged_pbr.required_wad_basenames.contains(*basename))
        {
            let path = staging.join(basename);
            std::fs::remove_file(&path)
                .map_err(|err| io_error("compile-bsp.remove_unused_wad", &path, err))?;
        }

        // ── Build canonical manifest ──────────────────────────────
        // The ownership marker is transaction metadata while staging and is
        // excluded from this pre-removal hash list. It is removed before
        // closure validation and publication. The manifest itself is excluded
        // from its payload list to avoid a recursive self-hash.
        let staged_hashes = fs_tx::compute_dir_file_hashes(&staging).map_err(CliError::FsTx)?;
        let source_map_identity = source_map
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                CliError::Usage("compile-bsp source map has no valid filename".into())
            })?;
        let controlled_env = compiler::controlled_environment_identity();
        let manifest_toml = build_canonical_manifest(
            &compile_result,
            source_map_identity,
            &profile_sha256,
            profile_family,
            &exact_profile_name,
            &controlled_env,
            &staging,
            &staged_hashes,
        )?;
        let manifest_sha256 = fs_tx::compute_manifest_sha256(manifest_toml.as_bytes());

        let manifest_path = staging.join(format!("{bsp_name}.manifest.toml"));
        std::fs::write(&manifest_path, &manifest_toml)
            .map_err(|err| io_error("compile-bsp.write", &manifest_path, err))?;

        // The marker proves ownership only while this is staging. Published
        // packages contain only declared payloads plus their canonical manifest.
        fs_tx::remove_staging_marker(&staging).map_err(CliError::FsTx)?;

        // ── Validate manifest closure ─────────────────────────────
        let _declared = fs_tx::validate_manifest_closure(&staging, manifest_toml.as_bytes())
            .map_err(CliError::FsTx)?;

        // ── Isolated Phase-03 strict import ───────────────────────
        // Validate only the staged package root through the shared package
        // boundary. This proves the manifest closure has no source-tree
        // fallback before it becomes visible at the destination.
        validate_staged_authorized_import(
            &staging,
            bsp_name,
            compile_result.lit_data.is_some(),
            &selected_wad_basenames,
        )?;

        // ── Publication ───────────────────────────────────────────
        if destination_exists {
            // Validate an existing destination before comparing it. An
            // incomplete directory is never repaired, merged, or overwritten.
            match validate_existing_destination(&out_dir) {
                Ok(Some(existing_manifest_sha256))
                    if existing_manifest_sha256 == manifest_sha256 =>
                {
                    cleanup_staging(&staging);
                    return Ok(format!(
                        "compiled[bsp]: {} -> {}/ (unchanged, manifest sha256:{})",
                        source_map.display(),
                        out_dir.display(),
                        manifest_sha256
                    ));
                }
                Ok(Some(existing_manifest_sha256)) => {
                    return Err(CliError::FsTx(fs_tx::FsTxError::PreExistingDestination {
                        target: out_dir.clone(),
                        message: format!(
                            "late collision: existing manifest sha256:{} != new sha256:{}",
                            existing_manifest_sha256, manifest_sha256
                        ),
                    }));
                }
                Ok(None) => {
                    return Err(CliError::FsTx(fs_tx::FsTxError::PreExistingDestination {
                        target: out_dir.clone(),
                        message: "incomplete destination: no valid manifest found".to_string(),
                    }))
                }
                Err(reason) => {
                    return Err(CliError::FsTx(fs_tx::FsTxError::PreExistingDestination {
                        target: out_dir.clone(),
                        message: format!("incomplete destination: {reason}"),
                    }))
                }
            }
        }

        match publish_directory_no_replace(&staging, &out_dir) {
            Ok(()) => {}
            Err(fs_tx::FsTxError::PreExistingDestination { .. }) if matches!(validate_existing_destination(&out_dir), Ok(Some(ref hash)) if hash == &manifest_sha256) =>
            {
                cleanup_staging(&staging);
                return Ok(format!(
                    "compiled[bsp]: {} -> {}/ (unchanged, manifest sha256:{})",
                    source_map.display(),
                    out_dir.display(),
                    manifest_sha256
                ));
            }
            Err(err) => return Err(CliError::FsTx(err)),
        }

        Ok(format!(
            "published[bsp]: {} -> {}/ profile={} ({}) family={} strict=true manifest_sha256:{}",
            source_map.display(),
            out_dir.display(),
            exact_profile_name,
            profile.compiler_identity,
            profile_family,
            manifest_sha256
        ))
    })();

    if result.is_err() {
        cleanup_staging(&staging);
    }
    result
}

// ---------------------------------------------------------------------------
// enhanced-dungeon — generate, compile, and publish an Enhanced v2 dungeon
// ---------------------------------------------------------------------------

/// Default compiler profile bundled with engine_pack.
const DEFAULT_BSP2_PROFILE: &str = include_str!("../../bsp_authoring/ericw-q1-bsp2-generated-profile.toml");

/// Resolve the Enhanced v2 CC0 Dungeon theme directory without parent traversal.
fn cc0_dungeon_v2_dir() -> CliResult<PathBuf> {
    let engine_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| {
            CliError::Validation(internal_error(
                "enhanced-dungeon.theme",
                "engine_pack manifest directory is not under the workspace tools directory".into(),
            ))
        })?;
    Ok(engine_root.join("src/bsp_generator/themes/cc0_dungeon_v2"))
}

/// Locate the project-documented user installation when the caller did not
/// supply `--tool-path`. Explicit paths remain authoritative.
fn default_ericw_tools_dir() -> Option<PathBuf> {
    let candidate = PathBuf::from(env::var_os("HOME")?)
        .join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin");
    ["qbsp", "vis", "light"]
        .iter()
        .all(|executable| candidate.join(executable).is_file())
        .then_some(candidate)
}

fn enhanced_dungeon_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("enhanced-dungeon", cli::enhanced_dungeon_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let seed_str = require_option("--seed", &parsed)?;
    let seed: u64 = seed_str.parse().map_err(|_| {
        CliError::Usage(format!("invalid --seed value: '{seed_str}'"))
    })?;

    let out_dir = PathBuf::from(require_option("--out", &parsed)?);
    let tool_path = parsed
        .singleton_value("--tool-path")
        .map(PathBuf::from)
        .or_else(default_ericw_tools_dir);
    let name = parsed
        .singleton_value("--name")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "enhanced_dungeon".to_string());

    // Build EnhancedConfig from optional overrides, falling back to nominal
    let rooms: u32 = parsed
        .singleton_value("--rooms")
        .map(|s| {
            s.parse()
                .map_err(|_| CliError::Usage(format!("invalid --rooms: '{s}'")))
        })
        .unwrap_or(Ok(28))?;
    let loops: u32 = parsed
        .singleton_value("--loops")
        .map(|s| {
            s.parse()
                .map_err(|_| CliError::Usage(format!("invalid --loops: '{s}'")))
        })
        .unwrap_or(Ok(3))?;
    let vertical_edges: u32 = parsed
        .singleton_value("--vertical-edges")
        .map(|s| {
            s.parse()
                .map_err(|_| CliError::Usage(format!("invalid --vertical-edges: '{s}'")))
        })
        .unwrap_or(Ok(1))?;

    let config = bsp_generator::enhanced::config::EnhancedConfig::new(
        rooms,
        loops,
        vertical_edges,
        bsp_generator::enhanced::config::ENHANCED_TREAD_DEFAULT,
        2048,
    )
    .map_err(|err| {
        CliError::Usage(format!("invalid enhanced config: {err}"))
    })?;

    // ── Read or use default profile ───────────────────────────────
    let profile_content = if let Some(profile_path) = parsed.singleton_value("--profile") {
        std::fs::read_to_string(&profile_path).map_err(|err| {
            io_error("enhanced-dungeon.profile", Path::new(&profile_path), err)
        })?
    } else {
        DEFAULT_BSP2_PROFILE.to_string()
    };
    let profile = compiler::parse_compiler_profile(&profile_content).map_err(|msg| {
        CliError::Validation(internal_error(
            "bsp.profile",
            format!("invalid compiler profile: {msg}"),
        ))
    })?;

    // ── Resolve theme assets ──────────────────────────────────────
    let theme_dir = cc0_dungeon_v2_dir()?;
    let palette_path = theme_dir.join("palette.lmp");
    let wad_path = theme_dir.join("cc0_dungeon_v2.wad");

    for input in [&palette_path, &wad_path] {
        let path = input.as_path();
        compiler::validate_input_regular(path).map_err(|err| {
            CliError::Validation(ValidationError::single(
                ValidationDiagnostic::new(
                    "enhanced-dungeon.input",
                    ValidationArea::Asset,
                    format!("invalid theme asset: {err}"),
                )
                .with_path(path),
            ))
        })?;
    }

    // ── Check destination ─────────────────────────────────────────
    if path_exists_no_follow(&out_dir) {
        return Err(CliError::FsTx(fs_tx::FsTxError::ExistingTarget(
            out_dir.clone(),
        )));
    }

    // ── Recover orphaned staging ──────────────────────────────────
    fs_tx::recover_orphaned_staging(&out_dir);

    // ── Create staging ────────────────────────────────────────────
    let staging = create_staging_sibling(&out_dir).map_err(CliError::FsTx)?;

    let result = (|| -> CliResult<String> {
        // 1. Generate enhanced map
        let (map_text, meta) = bsp_generator::generate_enhanced(seed, config.clone())
            .map_err(|err| {
                CliError::Validation(internal_error(
                    "enhanced-dungeon.generate",
                    format!("generation failed: {err}"),
                ))
            })?;

        // 2. Write .map source into staging
        let map_filename = format!("{name}.map");
        let map_path = staging.join(&map_filename);
        std::fs::write(&map_path, &map_text)
            .map_err(|err| io_error("enhanced-dungeon.write_map", &map_path, err))?;

        // 3. Compile
        let work_dir = staging.join(".compile-work");
        std::fs::create_dir_all(&work_dir)
            .map_err(|err| io_error("enhanced-dungeon.workdir", &work_dir, err))?;

        let compile_result = compiler::compile_map(
            &map_path,
            &profile,
            &work_dir,
            &palette_path,
            tool_path.as_deref(),
            &[wad_path.clone()],
        )
        .map_err(|err| {
            CliError::Validation(ValidationError::single(
                ValidationDiagnostic::new(
                    "enhanced-dungeon.compile",
                    ValidationArea::Asset,
                    format!("compilation failed: {err}"),
                )
                .with_path(&map_path),
            ))
        })?;

        // Clean up work directory
        let _ = std::fs::remove_dir_all(&work_dir);

        // 4. Stage compiled .bsp
        let bsp_path = staging.join(format!("{name}.bsp"));
        std::fs::write(&bsp_path, &compile_result.bsp_data)
            .map_err(|err| io_error("enhanced-dungeon.write_bsp", &bsp_path, err))?;

        // 5. Stage .lit companion
        if let Some(ref lit_data) = compile_result.lit_data {
            let lit_path = staging.join(format!("{name}.lit"));
            std::fs::write(&lit_path, lit_data)
                .map_err(|err| io_error("enhanced-dungeon.write_lit", &lit_path, err))?;
        }

        // 6. Stage palette
        let palette_bytes = std::fs::read(&palette_path)
            .map_err(|err| io_error("enhanced-dungeon.read_palette", &palette_path, err))?;
        let palette_staged = staging.join("palette.lmp");
        std::fs::write(&palette_staged, &palette_bytes)
            .map_err(|err| io_error("enhanced-dungeon.write_palette", &palette_staged, err))?;

        // 7. Stage WAD
        let wad_basename = wad_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("cc0_dungeon_v2.wad");
        let wad_staged = staging.join(wad_basename);
        std::fs::copy(&wad_path, &wad_staged)
            .map_err(|err| io_error("enhanced-dungeon.copy_wad", &wad_path, err))?;

        // 8. Stage PBR companion textures from the theme's textures/ directory
        let staged_pbr = stage_pbr_companions(
            &staging,
            &name,
            &compile_result,
            &[wad_path.clone()],
            &palette_bytes,
        )?;
        require_complete_enhanced_pbr_closure(&staged_pbr)?;
        let selected_wad_basenames: Vec<String> = staged_pbr
            .required_wad_basenames
            .iter()
            .cloned()
            .collect();

        // 9. Validate the complete staged closure through isolated strict authorization
        validate_staged_authorized_import(
            &staging,
            &name,
            compile_result.lit_data.is_some(),
            &selected_wad_basenames,
        )?;

        // 10. Write metadata.json
        let metadata = serde_json::json!({
            "format_version": 1,
            "generator": "bsp_generator::enhanced",
            "seed": seed,
            "config": {
                "rooms": rooms,
                "loops": loops,
                "vertical_edges": vertical_edges,
                "tread_depth": bsp_generator::enhanced::config::ENHANCED_TREAD_DEFAULT,
                "xy_extent": 2048_u32,
            },
            "output": {
                "room_count": meta.room_count,
                "route_count": meta.route_count,
                "transition_count": meta.transition_count,
                "lower_floor_z": meta.lower_floor_z,
                "upper_floor_z": meta.upper_floor_z,
                "spawn_origin": meta.spawn_origin,
                "light_count": meta.light_count,
                "pillar_count": meta.pillar_count,
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
        .map_err(|err| io_error("enhanced-dungeon.write_metadata", &metadata_path, err))?;

        // 11. Publish atomically
        publish_staging(&staging, &out_dir).map_err(CliError::FsTx)?;

        Ok(format!(
            "published[enhanced-dungeon]: seed={seed} -> {}/ ({} rooms, {} routes, {} transitions)",
            out_dir.display(),
            meta.room_count,
            meta.route_count,
            meta.transition_count,
        ))
    })();

    if result.is_err() {
        cleanup_staging(&staging);
    }
    result
}

/// Validate an existing destination directory as a complete canonical closure.
/// Returns `Ok(Some(manifest_sha256))` if it's a valid complete closure,
/// `Ok(None)` if incomplete, or `Err(reason)` on I/O error.
fn validate_existing_destination(out_dir: &Path) -> Result<Option<String>, String> {
    if !out_dir.is_dir() {
        return Err("destination is not a directory".to_string());
    }
    // Find a .manifest.toml file
    let mut manifest_path = None;
    if let Ok(entries) = std::fs::read_dir(out_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".manifest.toml") {
                manifest_path = Some(entry.path());
                break;
            }
        }
    }
    let manifest_path = match manifest_path {
        Some(p) => p,
        None => return Ok(None),
    };
    let manifest_bytes =
        std::fs::read(&manifest_path).map_err(|e| format!("read manifest: {e}"))?;
    let sha256 = fs_tx::compute_manifest_sha256(&manifest_bytes);

    // Validate the closure
    fs_tx::validate_manifest_closure(out_dir, &manifest_bytes)
        .map_err(|e| format!("validation failed: {e}"))?;

    Ok(Some(sha256))
}

/// Strict-load a staged closure through the Phase-03 authorization boundary.
fn validate_staged_authorized_import(
    staging: &Path,
    bsp_name: &str,
    has_lit: bool,
    wad_paths: &[String],
) -> CliResult<()> {
    use package_io::budget::BudgetLedger;
    use package_io::PackageRoot;

    let root = PackageRoot::new(staging).map_err(|err| {
        CliError::Validation(internal_error(
            "bsp.package",
            format!("cannot create staged package root: {err}"),
        ))
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
        CliError::Validation(internal_error(
            "bsp.package",
            format!("isolated strict package import failed: {err}"),
        ))
    })?;

    if !import.world.diagnostics.is_empty() {
        return Err(CliError::Validation(internal_error(
            "bsp.package",
            format!(
                "isolated strict package import produced diagnostics: {:?}",
                import
                    .world
                    .diagnostics
                    .iter()
                    .map(|diagnostic| diagnostic.message.as_str())
                    .collect::<Vec<_>>()
            ),
        )));
    }
    Ok(())
}

/// Return the mip-0 dimensions for a texture identity selected from an
/// external WAD or, when present, the embedded BSP miptex slot.
fn resolve_base_texture_dimensions(
    world: &bsp::world::BspWorld,
    slots: &[bsp::resources::MiptexSlot],
    identity: &str,
) -> CliResult<(u32, u32)> {
    for (_, wad) in &world.wad_archives {
        if let Some(bytes) = bsp::wad::read_wad_lump(wad, identity) {
            let info = bsp::wad::parse_miptex_header(bytes).map_err(|err| {
                CliError::Validation(internal_error(
                    "bsp.compile",
                    format!("invalid base miptex '{identity}' in staged WAD: {err}"),
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
                CliError::Validation(internal_error(
                    "bsp.compile",
                    format!("invalid embedded miptex '{identity}': {err}"),
                ))
            })?;
            return Ok((info.width, info.height));
        }
    }

    Err(CliError::Validation(internal_error(
        "bsp.compile",
        format!("cannot determine dimensions for selected PBR base texture '{identity}'"),
    )))
}

/// Reject a source path with any symlink component before it is used for
/// companion discovery or copying. This keeps PBR lookup confined to the
/// explicit WAD-adjacent source root instead of silently traversing aliases.
fn validate_no_symlink_path_components(path: &Path, label: &str) -> CliResult<()> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|err| io_error("compile-bsp.current_dir", Path::new("."), err))?
            .join(path)
    };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            std::path::Component::RootDir => current.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                return Err(CliError::Validation(internal_error(
                    "bsp.compile",
                    format!(
                        "{label} path contains a parent traversal: '{}'",
                        path.display()
                    ),
                )));
            }
            std::path::Component::Normal(part) => {
                current.push(part);
                let metadata = std::fs::symlink_metadata(&current)
                    .map_err(|err| io_error("compile-bsp.inspect_source", &current, err))?;
                if metadata.file_type().is_symlink() {
                    return Err(CliError::Validation(internal_error(
                        "bsp.compile",
                        format!(
                            "{label} path contains a symlink component: '{}'",
                            current.display()
                        ),
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Validate selected PNG companion content before it enters the publication
/// closure. This checks the complete PNG chunk envelope, CRCs, and IHDR
/// dimensions without adding a runtime image-decoder dependency to the tool.
fn validate_selected_pbr_companion(
    path: &Path,
    identity: &str,
    expected_dimensions: (u32, u32),
) -> CliResult<()> {
    compiler::validate_input_regular(path).map_err(|err| {
        CliError::Validation(internal_error(
            "bsp.compile",
            format!("invalid PBR companion '{}': {err}", path.display()),
        ))
    })?;
    let bytes =
        std::fs::read(path).map_err(|err| io_error("compile-bsp.read_companion", path, err))?;
    let dimensions = parse_png_dimensions(&bytes).map_err(|reason| {
        CliError::Validation(internal_error(
            "bsp.compile",
            format!(
                "malformed PBR companion '{}' for '{identity}': {reason}",
                path.display()
            ),
        ))
    })?;
    if dimensions != expected_dimensions {
        return Err(CliError::Validation(internal_error(
            "bsp.compile",
            format!(
                "PBR companion '{}' dimensions {}x{} do not match base texture '{identity}' dimensions {}x{}",
                path.display(),
                dimensions.0,
                dimensions.1,
                expected_dimensions.0,
                expected_dimensions.1
            ),
        )));
    }
    Ok(())
}

fn parse_png_dimensions(bytes: &[u8]) -> Result<(u32, u32), String> {
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
        let actual_crc = png_crc32(&bytes[offset + 4..data_end]);
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

fn png_crc32(bytes: &[u8]) -> u32 {
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

struct StagedPbrClosure {
    required_wad_basenames: std::collections::BTreeSet<String>,
    eligible_identities: std::collections::BTreeSet<String>,
    staged_companions: std::collections::BTreeSet<String>,
}

/// Enhanced packages are a complete PBR closure: unlike generic `compile-bsp`,
/// every eligible BSP identity must stage both canonical companion maps.
fn require_complete_enhanced_pbr_closure(staged: &StagedPbrClosure) -> CliResult<()> {
    let expected = staged
        .eligible_identities
        .iter()
        .flat_map(|identity| [
            format!("{identity}_norm.png"),
            format!("{identity}_gloss.png"),
        ])
        .collect::<std::collections::BTreeSet<_>>();
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
    Err(CliError::Validation(internal_error(
        "enhanced-dungeon.pbr-closure",
        format!(
            "incomplete Enhanced PBR companion closure; missing: {missing:?}; unexpected: {unexpected:?}"
        ),
    )))
}

/// Determine the WAD archives actually needed by face texture resolution.
/// The first archive containing a referenced identity wins, matching the
/// ordered WAD lookup policy used during strict loading.
fn required_wad_basenames(
    world: &bsp::world::BspWorld,
    slots: &[bsp::resources::MiptexSlot],
) -> std::collections::BTreeSet<String> {
    let mut required = std::collections::BTreeSet::new();
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

/// Stage PBR companion textures by strict-loading the BSP to derive
/// companion eligibility from miptex slot identities.
fn stage_pbr_companions(
    staging: &Path,
    _bsp_name: &str,
    compile_result: &bsp::CompileResult,
    wad_paths: &[PathBuf],
    palette_bytes: &[u8],
) -> CliResult<StagedPbrClosure> {
    // Strict-load the BSP to get face/texinfo/miptex data
    let mut wad_archives: Vec<(String, Vec<u8>)> = Vec::new();
    for wad_path in wad_paths {
        validate_no_symlink_path_components(wad_path, "WAD input")?;
        let basename = wad_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.wad")
            .to_string();
        let bytes = std::fs::read(wad_path)
            .map_err(|err| io_error("compile-bsp.read_wad", wad_path, err))?;
        wad_archives.push((basename, bytes));
    }

    let lit_data = compile_result.lit_data.clone();
    let load_options = bsp::LoadOptions {
        strict: true,
        palette: Some(palette_bytes.to_vec()),
        lit_data: lit_data.clone(),
        wad_archives,
        texture_overrides: Vec::new(),
        source_identity: "compile-bsp".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compile_result.bsp_data, &load_options).map_err(|report| {
            CliError::Validation(ValidationError::single(ValidationDiagnostic::new(
                "bsp.compile",
                ValidationArea::Asset,
                format!("strict-load validation failed: {report}"),
            )))
        })?;

    if !world.diagnostics.is_empty() {
        return Err(CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "bsp.compile",
                ValidationArea::Asset,
                format!(
                    "strict-load produced diagnostics: {:?}",
                    world
                        .diagnostics
                        .iter()
                        .map(|d| d.message.as_str())
                        .collect::<Vec<_>>()
                ),
            ),
        )));
    }

    // Derive companion eligibility from face.texinfo.miptex → source slot
    let slots = bsp::resources::parse_miptex_slots(&world.miptex_data);
    let required_wad_basenames = required_wad_basenames(&world, &slots);

    // Collect eligible identities — only opaque and alpha-mask surfaces
    use std::collections::BTreeSet;
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
        .collect::<CliResult<std::collections::BTreeMap<_, _>>>()?;

    // Resolve companions from ordered WAD texture roots.
    // Normalize each WAD parent to its confined textures/ child.
    let textures_dir = staging.join("textures");
    std::fs::create_dir_all(&textures_dir)
        .map_err(|err| io_error("compile-bsp.textures", &textures_dir, err))?;
    let mut staged_companions = BTreeSet::new();

    for identity in &eligible {
        for suffix in &["_norm.png", "_gloss.png"] {
            let expected = format!("{identity}{suffix}");

            // Search in WAD sibling directories: each WAD's parent's textures/ dir
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
                    Ok(metadata) => metadata,
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(err) => {
                        return Err(io_error(
                            "compile-bsp.inspect_companion_root",
                            &search_dir,
                            err,
                        ))
                    }
                };
                if search_metadata.file_type().is_symlink() {
                    return Err(CliError::Validation(internal_error(
                        "bsp.compile",
                        format!(
                            "PBR companion root must not be a symlink: '{}'",
                            search_dir.display()
                        ),
                    )));
                }
                if !search_metadata.is_dir() {
                    continue;
                }
                validate_no_symlink_path_components(&search_dir, "PBR companion root")?;
                let all_entries = std::fs::read_dir(&search_dir)
                    .map_err(|err| io_error("compile-bsp.read_companion_root", &search_dir, err))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|err| io_error("compile-bsp.read_companion_root", &search_dir, err))?;
                for entry in &all_entries {
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
                // ASCII-insensitive fallback: find exactly one unique match
                let mut fallback: Option<PathBuf> = None;
                let mut ambiguous = false;
                for entry in &all_entries {
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
                    return Err(CliError::Validation(ValidationError::single(
                        ValidationDiagnostic::new(
                            "bsp.compile",
                            ValidationArea::Asset,
                            format!(
                                "ambiguous PBR companion for '{identity}': multiple case-insensitive matches for {suffix}"
                            ),
                        ),
                    )));
                }
                if let Some(p) = fallback {
                    found = Some(p);
                    break;
                }
            }

            if let Some(src) = found {
                let expected_dimensions = base_dimensions.get(identity).copied().ok_or_else(|| {
                    CliError::Validation(internal_error(
                        "bsp.compile",
                        format!("missing base texture dimensions for companion identity '{identity}'"),
                    ))
                })?;
                validate_selected_pbr_companion(&src, identity, expected_dimensions)?;

                // Publish the canonical identity-derived name even when generic
                // discovery selected its permitted case-insensitive source fallback.
                let dest = textures_dir.join(&expected);
                std::fs::copy(&src, &dest)
                    .map_err(|err| io_error("compile-bsp.copy_companion", &src, err))?;
                staged_companions.insert(expected);
            }
            // Absence is not an error — legacy fallback
        }
    }

    Ok(StagedPbrClosure {
        required_wad_basenames,
        eligible_identities: eligible,
        staged_companions,
    })
}

#[cfg(test)]
mod pbr_closure_tests {
    use super::*;

    #[test]
    fn enhanced_pbr_closure_rejects_a_missing_required_companion() {
        let staged = StagedPbrClosure {
            required_wad_basenames: std::collections::BTreeSet::new(),
            eligible_identities: std::collections::BTreeSet::from(["bs_floor".to_string()]),
            staged_companions: std::collections::BTreeSet::from(["bs_floor_norm.png".to_string()]),
        };

        assert!(require_complete_enhanced_pbr_closure(&staged).is_err());
    }
}

/// Build a canonical package manifest capturing the full closure.
fn build_canonical_manifest(
    compile_result: &bsp::CompileResult,
    source_map_identity: &str,
    profile_sha256: &str,
    profile_family: &str,
    exact_profile_name: &str,
    controlled_env: &str,
    staging: &Path,
    staged_hashes: &[(String, String)],
) -> CliResult<String> {
    use toml::Value;

    let mut root = toml::Table::new();

    root.insert("format_version".into(), Value::Integer(1));
    root.insert(
        "manifest_schema".into(),
        Value::String("engine-pack-canonical/1".into()),
    );
    root.insert("strict".into(), Value::Boolean(true));

    // ── Profile identity ──────────────────────────────────────
    root.insert(
        "profile_family".into(),
        Value::String(profile_family.into()),
    );
    root.insert(
        "exact_profile".into(),
        Value::String(exact_profile_name.into()),
    );
    root.insert(
        "profile_sha256".into(),
        Value::String(profile_sha256.into()),
    );

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
        if !compile_result.provenance.qbsp_args.is_empty() {
            prov.insert(
                "qbsp_args".into(),
                Value::Array(
                    compile_result
                        .provenance
                        .qbsp_args
                        .iter()
                        .map(|a| Value::String(a.clone()))
                        .collect(),
                ),
            );
        }
        if !compile_result.provenance.vis_args.is_empty() {
            prov.insert(
                "vis_args".into(),
                Value::Array(
                    compile_result
                        .provenance
                        .vis_args
                        .iter()
                        .map(|a| Value::String(a.clone()))
                        .collect(),
                ),
            );
        }
        if !compile_result.provenance.light_args.is_empty() {
            prov.insert(
                "light_args".into(),
                Value::Array(
                    compile_result
                        .provenance
                        .light_args
                        .iter()
                        .map(|a| Value::String(a.clone()))
                        .collect(),
                ),
            );
        }
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
        if !compile_result.provenance.source_hashes.is_empty() {
            let hashes = compile_result
                .provenance
                .source_hashes
                .iter()
                .map(|hash| {
                    let mut entry = toml::Table::new();
                    entry.insert(
                        "path".into(),
                        Value::String(hash.path.to_string_lossy().into_owned()),
                    );
                    entry.insert("sha256".into(), Value::String(hash.sha256.clone()));
                    Value::Table(entry)
                })
                .collect();
            prov.insert("source_hashes".into(), Value::Array(hashes));
        }
        if !compile_result.provenance.output_hashes.is_empty() {
            let hashes = compile_result
                .provenance
                .output_hashes
                .iter()
                .map(|hash| {
                    let mut entry = toml::Table::new();
                    entry.insert(
                        "path".into(),
                        Value::String(hash.path.to_string_lossy().into_owned()),
                    );
                    entry.insert("sha256".into(), Value::String(hash.sha256.clone()));
                    Value::Table(entry)
                })
                .collect();
            prov.insert("output_hashes".into(), Value::Array(hashes));
        }
        prov.insert(
            "controlled_environment_identity".into(),
            Value::String(controlled_env.into()),
        );
        root.insert("compiler_provenance".into(), Value::Table(prov));
    }

    // ── Source identity ───────────────────────────────────────
    // Keep provenance relocatable: source hashes below identify the copied
    // inputs; no host or staging path is serialized into the closure.
    {
        let mut src = toml::Table::new();
        src.insert(
            "source_map".into(),
            Value::String(source_map_identity.into()),
        );
        root.insert("source_identity".into(), Value::Table(src));
    }

    // ── Published artifacts ───────────────────────────────────
    {
        let mut artifacts: Vec<Value> = Vec::new();
        // The manifest itself is not included (non-recursive). The ownership
        // marker still exists while this pre-removal hash list is collected,
        // but it is staging-only metadata and is removed before publication.
        for (rel_path, sha256) in staged_hashes {
            if rel_path.ends_with(".manifest.toml") || rel_path == fs_tx::STAGING_MARKER_NAME {
                continue;
            }
            let bytes = std::fs::metadata(staging.join(rel_path))
                .map_err(|err| io_error("compile-bsp.metadata", &staging.join(rel_path), err))?
                .len();
            let mut entry = toml::Table::new();
            entry.insert("path".into(), Value::String(rel_path.clone()));
            entry.insert("sha256".into(), Value::String(sha256.clone()));
            entry.insert(
                "bytes".into(),
                Value::Integer(i64::try_from(bytes).map_err(|_| {
                    CliError::Validation(internal_error(
                        "bsp.serialize",
                        format!("artifact '{rel_path}' exceeds TOML integer range"),
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
            } else if rel_path.starts_with("textures/") {
                "texture_companion"
            } else {
                "unknown"
            };
            entry.insert("kind".into(), Value::String(kind.into()));
            artifacts.push(Value::Table(entry));
        }
        // Sort for determinism
        artifacts.sort_by(|a, b| {
            let a_path = a.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let b_path = b.get("path").and_then(|v| v.as_str()).unwrap_or("");
            a_path.cmp(b_path)
        });
        root.insert("published_artifacts".into(), Value::Array(artifacts));
    }

    // Record the selected PBR identities separately from generic artifacts so
    // consumers can diagnose exactly which normal/gloss companions qualified
    // for this closure without reconstructing names from paths.
    {
        let mut companions = Vec::new();
        for (path, sha256) in staged_hashes {
            let Some(file_name) = path.strip_prefix("textures/") else {
                continue;
            };
            let (identity, map_kind) = if let Some(identity) = file_name.strip_suffix("_norm.png") {
                (identity, "normal")
            } else if let Some(identity) = file_name.strip_suffix("_gloss.png") {
                (identity, "gloss")
            } else {
                continue;
            };
            let mut companion = toml::Table::new();
            companion.insert("identity".into(), Value::String(identity.to_string()));
            companion.insert("map_kind".into(), Value::String(map_kind.to_string()));
            companion.insert("path".into(), Value::String(path.clone()));
            companion.insert("sha256".into(), Value::String(sha256.clone()));
            companions.push(Value::Table(companion));
        }
        companions.sort_by(|a, b| {
            let a_path = a.get("path").and_then(|value| value.as_str()).unwrap_or("");
            let b_path = b.get("path").and_then(|value| value.as_str()).unwrap_or("");
            a_path.cmp(b_path)
        });
        root.insert("selected_pbr_companions".into(), Value::Array(companions));
    }

    toml::to_string_pretty(&Value::Table(root))
        .map_err(|e| CliError::Validation(internal_error("bsp.serialize", e.to_string())))
}

// ---------------------------------------------------------------------------
// new-project — transactional staging
// ---------------------------------------------------------------------------

fn new_project_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("new-project", cli::new_project_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let project_id = require_option("--id", &parsed)?;
    let name = require_option("--name", &parsed)?;
    let dir = require_positional(&parsed, "new-project <dir> --id <project_id> --name <name>")?;

    if path_exists_no_follow(&dir) {
        return Err(CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "project.path_exists",
                ValidationArea::Project,
                format!("project target already exists: {}", dir.display()),
            )
            .with_path(&dir)
            .with_durable_id(project_id),
        )));
    }

    let staging = create_staging_sibling(&dir).map_err(|e| CliError::FsTx(e))?;
    let result = (|| -> CliResult<()> {
        fs::create_dir_all(staging.join("assets"))
            .map_err(|e| fs_tx_err("project.io", &staging, e))?;
        fs::create_dir_all(staging.join("scenes"))
            .map_err(|e| fs_tx_err("project.io", &staging, e))?;

        let project_path = staging.join("engine.project.toml");
        let content = project_toml_content(&project_id, &name)?;
        fs::write(&project_path, content).map_err(|e| fs_tx_err("project.io", &project_path, e))?;

        let scene_path = staging.join("scenes/start.engine.scene.json");
        fs::write(&scene_path, starter_scene_json(&project_id)?)
            .map_err(|e| fs_tx_err("project.io", &scene_path, e))?;

        // Validate from staging before publishing
        validate_project_and_startup_scene(&project_path)?;

        Ok(())
    })();

    match result {
        Ok(()) => {
            publish_staging(&staging, &dir).map_err(|e| CliError::FsTx(e))?;
            Ok(format!(
                "created[project]: {}",
                dir.join("engine.project.toml").display()
            ))
        }
        Err(err) => {
            cleanup_staging(&staging);
            Err(err)
        }
    }
}

// ---------------------------------------------------------------------------
// new-package — transactional staging
// ---------------------------------------------------------------------------

fn new_package_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("new-package", cli::new_package_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let package_id = require_option("--id", &parsed)?;
    let name = require_option("--name", &parsed)?;
    let path = require_positional(
        &parsed,
        "new-package <path> --id <package_id> --name <display_name>",
    )?;

    if path_exists_no_follow(&path) {
        return Err(CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "package.path_exists",
                ValidationArea::Package,
                format!("package manifest already exists: {}", path.display()),
            )
            .with_path(&path)
            .with_durable_id(package_id),
        )));
    }

    let staging = create_staging_file_sibling(&path).map_err(CliError::FsTx)?;
    let result = (|| -> CliResult<()> {
        let content = toml::to_string(&PackageScaffoldToml {
            format_version: 1,
            package_id: package_id.clone(),
            display_name: name.clone(),
            package_version: "0.1.0".to_string(),
        })
        .map_err(|e| CliError::Validation(internal_error("package.serialize", e.to_string())))?;
        fs::write(&staging, content).map_err(|e| fs_tx_err("package.io", &staging, e))?;

        validate_package_manifest_file(
            &staging,
            &PackageValidationOptions::default().with_expected_package_id(&package_id),
        )?;

        Ok(())
    })();

    match result {
        Ok(()) => {
            publish_staging(&staging, &path).map_err(CliError::FsTx)?;
            Ok(format!("created[package]: {}", path.display()))
        }
        Err(err) => {
            cleanup_staging(&staging);
            Err(err)
        }
    }
}

// ---------------------------------------------------------------------------
// scan-assets — safe traversal (no-follow, root-contained)
// ---------------------------------------------------------------------------

fn scan_assets_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("scan-assets", cli::scan_assets_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let package_id = parsed
        .singleton_value("--package-id")
        .unwrap_or("scanned")
        .to_string();
    let root = require_positional(&parsed, "scan-assets <asset-root> [--package-id <id>]")?;

    let mut records = Vec::new();
    let mut plan = Vec::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    collect_scan_records(
        &root,
        &root,
        &package_id,
        &mut records,
        &mut plan,
        &mut visited,
    )?;

    records.sort_by(|left, right| left.path.cmp(&right.path));
    if records.is_empty() {
        return Ok("scan[assets]: no supported assets found".to_string());
    }

    // Serialize with TOML serializer for safe output
    let mut output = String::new();
    for record in records {
        // Serialize each record through serde to ensure valid TOML
        let value = toml::Value::try_from(ScanRecordToml {
            id: record.id.clone(),
            kind: record.kind.as_str().to_string(),
            path: record.path.clone(),
            display_name: record.display_name.clone(),
        })
        .map_err(|e| CliError::Validation(internal_error("scan.serialize", e.to_string())))?;

        output.push_str(
            &toml::to_string(&value).map_err(|e| {
                CliError::Validation(internal_error("scan.serialize", e.to_string()))
            })?,
        );
        output.push('\n');
    }
    Ok(output.trim_end().to_string())
}

#[derive(Serialize)]
struct ScanRecordToml {
    id: String,
    kind: String,
    path: String,
    display_name: String,
}

#[derive(Serialize)]
struct PackageScaffoldToml {
    format_version: u32,
    package_id: String,
    display_name: String,
    package_version: String,
}

// ---------------------------------------------------------------------------
// add-asset — transactional manifest mutation with rollback journal
// ---------------------------------------------------------------------------

fn add_asset_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("add-asset", cli::add_asset_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let asset_id = require_option("--id", &parsed)?;
    let kind = require_option("--kind", &parsed)?;
    let asset_path = require_option("--path", &parsed)?;
    let tags = parsed.repeated_values("--tag");
    let manifest_path = require_positional(
        &parsed,
        "add-asset <package.toml> --id <asset_id> --kind <kind> --path <path>",
    )?;

    let kind = parse_asset_kind(&kind)?;
    let asset_path_buf = PathBuf::from(&asset_path);
    if normalize_relative_path(&asset_path_buf).is_none() {
        return Err(CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "asset.invalid_path",
                ValidationArea::Asset,
                format!("invalid asset path '{}'", asset_path),
            )
            .with_path(&manifest_path)
            .with_durable_id(asset_id),
        )));
    }

    fs_tx::inspect_entry_no_follow(&manifest_path).map_err(CliError::FsTx)?;

    // Build new content via serializer (not ad-hoc escaping)
    let original_content = fs::read_to_string(&manifest_path)
        .map_err(|err| io_error("package.io", &manifest_path, err))?;

    let mut manifest: toml::Value = toml::from_str(&original_content).map_err(|e| {
        CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "package.parse",
                ValidationArea::Package,
                format!("invalid package TOML: {e}"),
            )
            .with_path(&manifest_path),
        ))
    })?;

    let assets = {
        let table = manifest.as_table_mut().ok_or_else(|| {
            CliError::Validation(ValidationError::single(
                ValidationDiagnostic::new(
                    "package.parse",
                    ValidationArea::Package,
                    "manifest is not a table".to_string(),
                )
                .with_path(&manifest_path),
            ))
        })?;
        table
            .entry("assets".to_string())
            .or_insert_with(|| toml::Value::Array(vec![]))
            .as_array_mut()
            .ok_or_else(|| {
                CliError::Validation(ValidationError::single(
                    ValidationDiagnostic::new(
                        "package.parse",
                        ValidationArea::Package,
                        "assets is not an array".to_string(),
                    )
                    .with_path(&manifest_path),
                ))
            })?
    };

    let mut asset_table = toml::value::Table::new();
    asset_table.insert("id".to_string(), toml::Value::String(asset_id.clone()));
    asset_table.insert(
        "kind".to_string(),
        toml::Value::String(kind.as_str().to_string()),
    );
    asset_table.insert("path".to_string(), toml::Value::String(asset_path.clone()));

    if !tags.is_empty() {
        let tag_values: Vec<toml::Value> = tags
            .iter()
            .map(|t| toml::Value::String(t.to_string()))
            .collect();
        asset_table.insert("tags".to_string(), toml::Value::Array(tag_values));
    }

    assets.push(toml::Value::Table(asset_table));

    let new_content = toml::to_string_pretty(&manifest)
        .map_err(|e| CliError::Validation(internal_error("add-asset.serialize", e.to_string())))?;

    let staging = create_staging_file_sibling(&manifest_path).map_err(CliError::FsTx)?;
    let stage_result = (|| -> CliResult<()> {
        fs::write(&staging, new_content).map_err(|err| io_error("package.io", &staging, err))?;
        validate_package_manifest_file(
            &staging,
            &PackageValidationOptions::default().check_source_files(true),
        )?;
        Ok(())
    })();
    if let Err(err) = stage_result {
        cleanup_staging(&staging);
        return Err(err);
    }

    let mut journal = RollbackJournal::new();
    if let Err(err) = journal.record_backup(&manifest_path) {
        cleanup_staging(&staging);
        return Err(CliError::FsTx(err));
    }
    if let Err(err) = replace_file_with_staging(&staging, &manifest_path) {
        cleanup_staging(&staging);
        if let Err(rollback_err) = journal.rollback() {
            return Err(CliError::FsTx(rollback_err));
        }
        return Err(CliError::FsTx(err));
    }
    journal.commit().map_err(CliError::FsTx)?;

    Ok(format!(
        "added[asset]: {} -> {}",
        asset_id,
        manifest_path.display()
    ))
}

// ---------------------------------------------------------------------------
// pack — transactional multi-artifact publication
// ---------------------------------------------------------------------------

fn pack_cmd(args: &[String]) -> CliResult<String> {
    let parsed = cli::parse_command("pack", cli::pack_schema(), args);
    let parsed = parsed.into_result().map_err(CliError::Usage)?;

    let out = require_option("--out", &parsed)?;
    let out = PathBuf::from(out);
    let project_path = require_positional(&parsed, "pack <engine.project.toml> --out <dir>")?;

    if path_exists_no_follow(&out) {
        return Err(CliError::FsTx(fs_tx::FsTxError::ExistingTarget(
            out.clone(),
        )));
    }

    // Validate everything first (preflight) before staging.
    fs_tx::inspect_entry_no_follow(&project_path).map_err(CliError::FsTx)?;
    let project = validate_project_and_startup_scene(&project_path)?;
    let project_root = project_path.parent().unwrap_or_else(|| Path::new(""));
    let canonical_project_root = project_root
        .canonicalize()
        .map_err(|err| fs_tx_err("pack.project_root", project_root, err))?;
    let project_source = fs_tx::canonicalize_contained(&project_path, &canonical_project_root)
        .map_err(CliError::FsTx)?;

    // Build plan from preflight
    let mut plan_entries = Vec::new();

    plan_entries.push(PlanEntry {
        source: project_source,
        destination: PathBuf::from("engine.project.toml"),
        entry_type: EntryType::File,
        label: "engine.project.toml".into(),
    });

    if let Some(scene) = &project.startup_scene {
        let scene_rel = normalize_relative_path(scene).ok_or_else(|| {
            CliError::Validation(ValidationError::single(
                ValidationDiagnostic::new(
                    "pack.invalid_scene_path",
                    ValidationArea::Scene,
                    format!("invalid startup_scene path '{}'", scene.display()),
                )
                .with_path(&project_path),
            ))
        })?;
        let scene_source = contained_child_no_symlinks(&canonical_project_root, &scene_rel)
            .map_err(CliError::FsTx)?;
        plan_entries.push(PlanEntry {
            source: scene_source,
            destination: scene_rel.clone(),
            entry_type: EntryType::File,
            label: format!("scene: {}", scene_rel.display()),
        });
    }
    for package in &project.packages {
        if !package.enabled {
            continue;
        }
        let manifest_rel = normalize_relative_path(&package.manifest).ok_or_else(|| {
            CliError::Validation(ValidationError::single(
                ValidationDiagnostic::new(
                    "pack.invalid_project_path",
                    ValidationArea::Package,
                    format!(
                        "invalid package manifest path '{}'",
                        package.manifest.display()
                    ),
                )
                .with_path(&project_path)
                .with_durable_id(package.package_id.clone()),
            ))
        })?;

        let manifest_path = contained_child_no_symlinks(&canonical_project_root, &manifest_rel)
            .map_err(CliError::FsTx)?;
        plan_entries.push(PlanEntry {
            source: manifest_path.clone(),
            destination: manifest_rel.clone(),
            entry_type: EntryType::File,
            label: format!("manifest: {}", manifest_rel.display()),
        });

        validate_package_manifest_file(
            &manifest_path,
            &PackageValidationOptions::default()
                .check_source_files(true)
                .with_expected_package_id(package.package_id.clone()),
        )?;
        let manifest = read_package_manifest(&manifest_path)?;
        let manifest_dir = manifest_rel.parent().unwrap_or_else(|| Path::new(""));

        for asset in &manifest.assets {
            let asset_package_rel = normalize_relative_path(&asset.path).ok_or_else(|| {
                CliError::Validation(ValidationError::single(
                    ValidationDiagnostic::new(
                        "pack.invalid_asset_path",
                        ValidationArea::Asset,
                        format!("invalid asset path '{}'", asset.path.display()),
                    )
                    .with_path(&manifest_path)
                    .with_durable_id(asset.id.clone()),
                ))
            })?;
            let asset_rel = normalize_relative_path(&manifest_dir.join(asset_package_rel))
                .ok_or_else(|| {
                    CliError::Validation(ValidationError::single(
                        ValidationDiagnostic::new(
                            "pack.invalid_asset_path",
                            ValidationArea::Asset,
                            format!("invalid asset path '{}'", asset.path.display()),
                        )
                        .with_path(&manifest_path)
                        .with_durable_id(asset.id.clone()),
                    ))
                })?;
            let source_path = contained_child_no_symlinks(&canonical_project_root, &asset_rel)
                .map_err(CliError::FsTx)?;
            plan_entries.push(PlanEntry {
                source: source_path,
                destination: asset_rel.clone(),
                entry_type: EntryType::File,
                label: format!("asset: {}", asset.id),
            });
        }
    }

    // Validate plan before staging
    let _plan = build_publication_plan(plan_entries.clone()).map_err(|e| CliError::FsTx(e))?;

    // Stage
    let staging = create_staging_sibling(&out).map_err(|e| CliError::FsTx(e))?;
    let result = (|| -> CliResult<()> {
        for entry in &plan_entries {
            stage_entry(&staging, entry).map_err(|e| CliError::FsTx(e))?;
        }

        // Write PACK_REPORT.json via serde_json (not ad-hoc)
        let mut report = PackReport {
            source_project: project_path.display().to_string(),
            copied_files: plan_entries
                .iter()
                .filter(|e| e.entry_type == EntryType::File)
                .map(|e| slash_path(&e.destination))
                .collect::<CliResult<Vec<_>>>()?,
            skipped_disabled_packages: project
                .packages
                .iter()
                .filter(|p| !p.enabled)
                .map(|p| p.package_id.clone())
                .collect(),
            warnings: Vec::new(),
            validation_status: "passed".to_string(),
        };

        // Sort for determinism
        report.copied_files.sort();

        let report_json = serde_json::to_string_pretty(&report).map_err(|err| {
            CliError::Validation(ValidationError::single(ValidationDiagnostic::new(
                "pack.report",
                ValidationArea::Project,
                format!("failed to serialize pack report: {err}"),
            )))
        })?;
        let report_path = staging.join("PACK_REPORT.json");
        fs::write(&report_path, report_json)
            .map_err(|err| io_error("pack.io", &report_path, err))?;

        Ok(())
    })();

    match result {
        Ok(()) => {
            publish_staging(&staging, &out).map_err(|e| CliError::FsTx(e))?;
            Ok(format!(
                "packed[project]: {} -> {} ({} files)",
                project_path.display(),
                out.display(),
                plan_entries
                    .iter()
                    .filter(|e| e.entry_type == EntryType::File)
                    .count()
            ))
        }
        Err(err) => {
            cleanup_staging(&staging);
            Err(err)
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn validate_project_and_startup_scene(path: &Path) -> CliResult<Project> {
    let project =
        validate_project_file(path, &ProjectValidationOptions::default().check_files(true))?;
    if let Some(startup_scene) = &project.startup_scene {
        let project_root = path.parent().unwrap_or_else(|| Path::new(""));
        let scene_path = project_root.join(startup_scene);
        let asset_ids = collect_project_asset_ids(path)?;
        validate_scene_file_with_options(
            &scene_path,
            &SceneValidationOptions::default().with_known_asset_ids(asset_ids),
        )?;
    }
    Ok(project)
}

fn collect_project_asset_ids(project_path: &Path) -> CliResult<HashSet<String>> {
    let project = validate_project_file(
        project_path,
        &ProjectValidationOptions::default().check_files(true),
    )?;
    let project_root = project_path.parent().unwrap_or_else(|| Path::new(""));
    let mut asset_ids = HashSet::new();
    for package in project.packages.iter().filter(|package| package.enabled) {
        let manifest_path = project_root.join(&package.manifest);
        let records = validate_package_manifest_file(
            &manifest_path,
            &PackageValidationOptions::default()
                .check_source_files(true)
                .with_expected_package_id(package.package_id.clone()),
        )?;
        asset_ids.extend(records.into_iter().map(|record| record.asset_id));
    }
    Ok(asset_ids)
}

fn collect_scan_records(
    root: &Path,
    dir: &Path,
    package_id: &str,
    records: &mut Vec<ScanRecord>,
    _plan: &mut Vec<PlanEntry>,
    visited: &mut HashSet<PathBuf>,
) -> CliResult<()> {
    // Safe traversal: use symlink_metadata, reject symlinks, check root containment
    let dir_meta = fs_tx::inspect_entry_no_follow(dir).map_err(CliError::FsTx)?;
    if !dir_meta.is_dir() {
        return Err(CliError::FsTx(fs_tx::FsTxError::InvalidEntryPath(format!(
            "not a directory: '{}'",
            dir.display()
        ))));
    }
    let canonical_root = root
        .canonicalize()
        .map_err(|err| io_error("scan.io", root, err))?;

    // Track filesystem identity to reject cycles
    let canonical_dir = dir
        .canonicalize()
        .map_err(|err| io_error("scan.io", dir, err))?;
    if !canonical_dir.starts_with(&canonical_root) {
        return Err(CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "scan.root_escape",
                ValidationArea::Asset,
                format!(
                    "scan entry '{}' escapes root '{}'",
                    dir.display(),
                    root.display()
                ),
            ),
        )));
    }
    if !visited.insert(canonical_dir) {
        return Ok(()); // cycle detected, skip
    }

    let mut entries = fs::read_dir(dir)
        .map_err(|err| io_error("scan.io", dir, err))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| io_error("scan.io", dir, err))?;
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let path = entry.path();
        let meta = fs_tx::inspect_entry_no_follow(&path).map_err(CliError::FsTx)?;

        if meta.is_dir() {
            collect_scan_records(root, &path, package_id, records, _plan, visited)?;
            continue;
        }

        let Some(kind) = classify_asset_kind(&path)? else {
            continue;
        };

        // Containment check
        let canonical = path
            .canonicalize()
            .map_err(|err| io_error("scan.io", &path, err))?;
        if !canonical.starts_with(&canonical_root) {
            return Err(CliError::Validation(ValidationError::single(
                ValidationDiagnostic::new(
                    "scan.root_escape",
                    ValidationArea::Asset,
                    format!(
                        "file '{}' escapes root '{}'",
                        path.display(),
                        root.display()
                    ),
                ),
            )));
        }

        let relative = path.strip_prefix(root).map_err(|err| {
            CliError::Validation(ValidationError::single(ValidationDiagnostic::new(
                "scan.path",
                ValidationArea::Asset,
                format!("failed to relativize '{}': {err}", path.display()),
            )))
        })?;
        let relative_string = slash_path(relative)?;
        let stem = relative.with_extension("");
        let name = sanitize_id_component(&slash_path(&stem)?);
        records.push(ScanRecord {
            id: format!("{package_id}.{}.{}", kind.as_str(), name),
            kind,
            path: relative_string,
            display_name: display_name(path.file_stem()),
        });
    }
    Ok(())
}

fn read_package_manifest(path: &Path) -> CliResult<PackageManifest> {
    let content = fs::read_to_string(path).map_err(|err| io_error("package.io", path, err))?;
    toml::from_str(&content).map_err(|err| {
        CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "package.parse",
                ValidationArea::Package,
                format!("failed to parse package manifest: {err}"),
            )
            .with_path(path),
        ))
    })
}

fn classify_asset_kind(path: &Path) -> CliResult<Option<AssetKind>> {
    let Some(extension) = path.extension() else {
        return Ok(None);
    };
    let extension = extension.to_str().ok_or_else(|| {
        CliError::FsTx(fs_tx::FsTxError::InvalidEntryPath(format!(
            "asset extension is not valid UTF-8: '{}'",
            path.display()
        )))
    })?;
    Ok(match extension.to_ascii_lowercase().as_str() {
        "gltf" | "glb" | "obj" => Some(AssetKind::Model),
        "png" | "jpg" | "jpeg" | "ktx" | "ktx2" => Some(AssetKind::Texture),
        "hdr" | "exr" => Some(AssetKind::Environment),
        "wav" | "ogg" | "flac" | "mp3" => Some(AssetKind::Audio),
        "bsp" => Some(AssetKind::Bsp),
        _ => None,
    })
}

fn parse_asset_kind(value: &str) -> CliResult<AssetKind> {
    match value {
        "model" => Ok(AssetKind::Model),
        "texture" => Ok(AssetKind::Texture),
        "material" => Ok(AssetKind::Material),
        "environment" => Ok(AssetKind::Environment),
        "prefab" => Ok(AssetKind::Prefab),
        "wall_chunk" => Ok(AssetKind::WallChunk),
        "scene_fragment" => Ok(AssetKind::SceneFragment),
        "audio" => Ok(AssetKind::Audio),
        "bsp" => Ok(AssetKind::Bsp),
        other => Err(CliError::Validation(ValidationError::single(
            ValidationDiagnostic::new(
                "asset.unsupported_kind",
                ValidationArea::Asset,
                format!("unsupported asset kind '{other}'"),
            ),
        ))),
    }
}

fn normalize_relative_path(path: &Path) -> Option<PathBuf> {
    let mut normalized = PathBuf::new();
    if path.as_os_str().is_empty() || path.is_absolute() {
        return None;
    }
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::Normal(part) => normalized.push(part),
            std::path::Component::ParentDir => return None,
            std::path::Component::Prefix(_) | std::path::Component::RootDir => return None,
        }
    }
    (!normalized.as_os_str().is_empty()).then_some(normalized)
}

fn slash_path(path: &Path) -> CliResult<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        if let std::path::Component::Normal(part) = component {
            let part = part.to_str().ok_or_else(|| {
                CliError::FsTx(fs_tx::FsTxError::InvalidEntryPath(format!(
                    "path component is not valid UTF-8: '{}'",
                    path.display()
                )))
            })?;
            parts.push(part.to_string());
        }
    }
    Ok(parts.join("/"))
}

fn sanitize_id_component(value: &str) -> String {
    let mut output = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch.to_ascii_lowercase());
        } else if !output.ends_with('.') {
            output.push('.');
        }
    }
    output.trim_matches('.').replace("..", ".")
}

fn display_name(stem: Option<&OsStr>) -> String {
    stem.and_then(OsStr::to_str)
        .unwrap_or("asset")
        .replace(['_', '-'], " ")
}

fn sanitize_crate_name(value: &str) -> String {
    let mut output = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch.to_ascii_lowercase());
        } else if ch == '-' || ch == '_' {
            output.push(ch);
        } else if !output.ends_with('_') {
            output.push('_');
        }
    }
    let output = output.trim_matches(['-', '_']).replace("__", "_");
    if output
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_')
    {
        output
    } else if output.is_empty() {
        "engine_app".to_string()
    } else {
        format!("app_{output}")
    }
}

fn rust_string_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn engine_root_path() -> CliResult<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| {
            CliError::Validation(ValidationError::single(ValidationDiagnostic::new(
                "app.engine_root",
                ValidationArea::Project,
                "failed to resolve engine workspace root for app template",
            )))
        })
        .and_then(|path| {
            path.canonicalize()
                .map_err(|err| io_error("app.engine_root", path, err))
        })
}

fn app_cargo_toml(app_id: &str, engine_root: &Path) -> CliResult<String> {
    let mut root = toml::value::Table::new();
    root.insert(
        "package".to_string(),
        toml::Value::Table(toml_table([
            ("name", toml::Value::String(sanitize_crate_name(app_id))),
            ("version", toml::Value::String("0.1.0".to_string())),
            ("edition", toml::Value::String("2021".to_string())),
        ])),
    );
    root.insert(
        "dependencies".to_string(),
        toml::Value::Table(toml_table([
            (
                "engine_events",
                toml::Value::Table(dep_path_table(&engine_root.join("src/events"))),
            ),
            (
                "input",
                toml::Value::Table(dep_path_table(&engine_root.join("src/input"))),
            ),
            (
                "physics",
                toml::Value::Table(dep_path_table(&engine_root.join("src/physics"))),
            ),
        ])),
    );
    toml::to_string(&toml::Value::Table(root))
        .map_err(|e| CliError::Validation(internal_error("app.serialize", e.to_string())))
}

fn toml_table<const N: usize>(entries: [(&str, toml::Value); N]) -> toml::value::Table {
    entries
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

fn dep_path_table(path: &Path) -> toml::value::Table {
    toml_table([("path", toml::Value::String(cargo_path_literal(path)))])
}

fn cargo_path_literal(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn app_main_rs(app_id: &str, name: &str) -> String {
    format!(
        "use engine_events::{{EngineEvent, EventBus, EventStage, LifecycleEvent}};\nuse input::{{ActionId, InputSnapshot}};\nuse physics::PhysicsWorld;\n\nconst APP_ID: &str = \"{}\";\nconst APP_NAME: &str = \"{}\";\n\nfn main() {{\n    let mut events = EventBus::default();\n    events.emit(\n        EventStage::Startup,\n        None,\n        EngineEvent::Lifecycle(LifecycleEvent::AppStarted {{\n            app_name: APP_NAME.to_string(),\n        }}),\n    );\n\n    let input = InputSnapshot::default();\n    let confirm_action = ActionId::new(format!(\"{{APP_ID}}.confirm\"));\n\n    let mut physics = PhysicsWorld::new();\n    physics.set_gravity(0.0, -9.81, 0.0);\n\n    println!(\n        \"{{APP_NAME}} initialized: {{}} pending event(s), confirm={{}}\",\n        events.pending_len(),\n        input.action_value(&confirm_action)\n    );\n}}\n",
        rust_string_escape(app_id),
        rust_string_escape(name)
    )
}

fn app_readme_md(app_id: &str, name: &str) -> String {
    format!(
        "# {}\n\nGenerated by `engine_pack new-app`.\n\nRun from this directory with:\n\n```sh\ncargo run\n```\n\nThis scaffold is a standalone Rust app crate that depends on public engine support crates only: `engine_events`, `input`, and `physics`. It does not mutate the engine root workspace and does not implement dynamic Rust reload, plugin ABI loading, or runtime hot reload.\n\nApp ID: `{}`\n",
        name,
        app_id
    )
}

fn starter_scene_json(project_id: &str) -> CliResult<String> {
    let scene = serde_json::json!({
        "format_version": 1,
        "scene_id": format!("{project_id}.start"),
        "root_nodes": ["node.root"],
        "nodes": [{
            "id": "node.root",
            "parent": null,
            "name": "Root",
            "transform": {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0]
            },
            "asset": null
        }],
        "lights": [],
        "environment": null,
        "editor": {}
    });
    serde_json::to_string_pretty(&scene)
        .map(|json| format!("{json}\n"))
        .map_err(|e| CliError::Validation(internal_error("scene.serialize", e.to_string())))
}

fn project_toml_content(project_id: &str, name: &str) -> CliResult<String> {
    let mut root = toml::value::Table::new();
    root.insert("format_version".to_string(), toml::Value::Integer(1));
    root.insert(
        "project_id".to_string(),
        toml::Value::String(project_id.to_string()),
    );
    root.insert("name".to_string(), toml::Value::String(name.to_string()));
    root.insert(
        "project_version".to_string(),
        toml::Value::String("0.1.0".to_string()),
    );
    root.insert(
        "asset_root".to_string(),
        toml::Value::String("assets".to_string()),
    );
    root.insert(
        "startup_scene".to_string(),
        toml::Value::String("scenes/start.engine.scene.json".to_string()),
    );
    root.insert("packages".to_string(), toml::Value::Array(Vec::new()));
    root.insert(
        "settings".to_string(),
        toml::Value::Table(toml_table([
            ("window_width", toml::Value::Integer(1280)),
            ("window_height", toml::Value::Integer(720)),
            ("fullscreen", toml::Value::Boolean(false)),
            ("vsync", toml::Value::Boolean(true)),
        ])),
    );
    toml::to_string(&toml::Value::Table(root))
        .map_err(|e| CliError::Validation(internal_error("project.serialize", e.to_string())))
}

fn require_option(name: &str, parsed: &launch_shared::CliParseResult) -> CliResult<String> {
    parsed
        .singleton_value(name)
        .map(|s| s.to_string())
        .ok_or_else(|| CliError::Usage(format!("missing required {name}")))
}

fn require_positional(parsed: &launch_shared::CliParseResult, usage: &str) -> CliResult<PathBuf> {
    match parsed.positionals.as_slice() {
        [path] => Ok(PathBuf::from(path)),
        [] => Err(CliError::Usage(format!("missing path: {usage}"))),
        extra => Err(CliError::Usage(format!(
            "expected exactly one positional path for {usage}, got {} ({})",
            extra.len(),
            extra.join(", ")
        ))),
    }
}

fn path_exists_no_follow(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
}

fn internal_error(code: &str, message: String) -> ValidationError {
    ValidationError::single(ValidationDiagnostic::new(
        code,
        ValidationArea::Project,
        message,
    ))
}

fn io_error(code: &'static str, path: &Path, err: std::io::Error) -> CliError {
    CliError::Validation(ValidationError::single(
        ValidationDiagnostic::new(
            code,
            ValidationArea::Project,
            format!("{}: {err}", path.display()),
        )
        .with_path(path),
    ))
}

fn fs_tx_err(code: &'static str, path: &Path, err: std::io::Error) -> CliError {
    CliError::FsTx(engine_pack::fs_tx::FsTxError::Io {
        path: path.to_path_buf(),
        message: format!("{code}: {err}"),
    })
}

fn print_validation_error(err: &ValidationError) {
    for diagnostic in err.diagnostics() {
        eprintln!("{diagnostic}");
    }
}

#[derive(Debug)]
struct ScanRecord {
    id: String,
    kind: AssetKind,
    path: String,
    display_name: String,
}

#[derive(Debug, Serialize)]
struct PackReport {
    source_project: String,
    copied_files: Vec<String>,
    skipped_disabled_packages: Vec<String>,
    warnings: Vec<String>,
    validation_status: String,
}

#[derive(Debug)]
enum CliError {
    Usage(String),
    FsTx(fs_tx::FsTxError),
    Validation(ValidationError),
}

impl From<ValidationError> for CliError {
    fn from(value: ValidationError) -> Self {
        Self::Validation(value)
    }
}
