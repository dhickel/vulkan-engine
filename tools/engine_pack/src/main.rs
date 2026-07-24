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
    create_staging_file_sibling, create_staging_sibling,
    publish_directory_no_replace, publish_staging,
    replace_file_with_staging, stage_entry, validate_staged_artifact_set,
    EntryType, PlanEntry, RollbackJournal,
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

    let out_dir = PathBuf::from(&out_dir);
    let destination_exists = path_exists_no_follow(&out_dir);

    // Use fs_tx staging for output
    let staging = create_staging_sibling(&out_dir).map_err(CliError::FsTx)?;

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

        std::fs::remove_dir_all(&work_dir)
            .map_err(|err| io_error("compile-bsp.workdir", &work_dir, err))?;

        // Write compiled outputs
        let bsp_name = source_map
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let bsp_path = staging.join(format!("{bsp_name}.bsp"));
        std::fs::write(&bsp_path, &compile_result.bsp_data)
            .map_err(|err| io_error("compile-bsp.write", &bsp_path, err))?;

        if let Some(ref lit_data) = compile_result.lit_data {
            let lit_path = staging.join(format!("{bsp_name}.lit"));
            std::fs::write(&lit_path, lit_data)
                .map_err(|err| io_error("compile-bsp.write", &lit_path, err))?;
        }

        // Write provenance
        let provenance_path = staging.join(format!("{bsp_name}.provenance.toml"));
        let provenance_toml = {
            let mut root = toml::value::Table::new();
            root.insert(
                "compiler_identity".into(),
                toml::Value::String(compile_result.provenance.compiler_identity.clone()),
            );
            root.insert(
                "compiler_version".into(),
                toml::Value::String(compile_result.provenance.compiler_version.clone()),
            );
            if !compile_result.provenance.qbsp_args.is_empty() {
                root.insert(
                    "qbsp_args".into(),
                    toml::Value::Array(
                        compile_result
                            .provenance
                            .qbsp_args
                            .iter()
                            .map(|a| toml::Value::String(a.clone()))
                            .collect(),
                    ),
                );
            }
            if !compile_result.provenance.vis_args.is_empty() {
                root.insert(
                    "vis_args".into(),
                    toml::Value::Array(
                        compile_result
                            .provenance
                            .vis_args
                            .iter()
                            .map(|a| toml::Value::String(a.clone()))
                            .collect(),
                    ),
                );
            }
            if !compile_result.provenance.light_args.is_empty() {
                root.insert(
                    "light_args".into(),
                    toml::Value::Array(
                        compile_result
                            .provenance
                            .light_args
                            .iter()
                            .map(|a| toml::Value::String(a.clone()))
                            .collect(),
                    ),
                );
            }
            if !compile_result.provenance.source_hashes.is_empty() {
                root.insert(
                    "source_hashes".into(),
                    toml::Value::Array(
                        compile_result
                            .provenance
                            .source_hashes
                            .iter()
                            .map(|hash| {
                                let mut table = toml::value::Table::new();
                                table.insert(
                                    "path".into(),
                                    toml::Value::String(
                                        hash.path.to_string_lossy().replace('\\', "/"),
                                    ),
                                );
                                table.insert(
                                    "sha256".into(),
                                    toml::Value::String(hash.sha256.clone()),
                                );
                                toml::Value::Table(table)
                            })
                            .collect(),
                    ),
                );
            }
            if !compile_result.provenance.output_hashes.is_empty() {
                root.insert(
                    "output_hashes".into(),
                    toml::Value::Array(
                        compile_result
                            .provenance
                            .output_hashes
                            .iter()
                            .map(|hash| {
                                let mut table = toml::value::Table::new();
                                table.insert(
                                    "path".into(),
                                    toml::Value::String(
                                        hash.path.to_string_lossy().replace('\\', "/"),
                                    ),
                                );
                                table.insert(
                                    "sha256".into(),
                                    toml::Value::String(hash.sha256.clone()),
                                );
                                toml::Value::Table(table)
                            })
                            .collect(),
                    ),
                );
            }
            if let Some(hashes) = &compile_result.provenance.compiler_hashes {
                let mut hash_table = toml::value::Table::new();
                hash_table.insert(
                    "qbsp_sha256".into(),
                    toml::Value::String(hashes.qbsp_sha256.clone()),
                );
                hash_table.insert(
                    "vis_sha256".into(),
                    toml::Value::String(hashes.vis_sha256.clone()),
                );
                hash_table.insert(
                    "light_sha256".into(),
                    toml::Value::String(hashes.light_sha256.clone()),
                );
                root.insert("compiler_hashes".into(), toml::Value::Table(hash_table));
            }
            root.insert(
                "stdout".into(),
                toml::Value::String(compile_result.stdout.clone()),
            );
            root.insert(
                "stderr".into(),
                toml::Value::String(compile_result.stderr.clone()),
            );
            toml::Value::Table(root)
        };
        let provenance_str = toml::to_string_pretty(&provenance_toml)
            .map_err(|e| CliError::Validation(internal_error("bsp.serialize", e.to_string())))?;
        std::fs::write(&provenance_path, provenance_str)
            .map_err(|err| io_error("compile-bsp.write", &provenance_path, err))?;

        // Validate staged artifact set before publication (Phase 08)
        let uses_bsp2 = profile.default_qbsp_args.iter().any(|a| a == "-bsp2")
            || profile.default_light_args.iter().any(|a| a == "-bsp2");
        let _staged_files = validate_staged_artifact_set(&staging, bsp_name, uses_bsp2)
            .map_err(CliError::FsTx)?;

        // Compute file hashes for provenance verification
        let staged_hashes = engine_pack::fs_tx::compute_dir_file_hashes(&staging)
            .map_err(CliError::FsTx)?;

        // Verify provenance output hashes match staged bytes
        if !compile_result.provenance.output_hashes.is_empty() {
            let hash_map: std::collections::HashMap<&str, &str> = staged_hashes
                .iter()
                .map(|(rel, h)| (rel.as_str(), h.as_str()))
                .collect();
            for output_hash in &compile_result.provenance.output_hashes {
                let key = output_hash.path.to_string_lossy().replace('\\', "/");
                match hash_map.get(key.as_str()) {
                    Some(actual_hash) if actual_hash == &output_hash.sha256 => {}
                    Some(actual_hash) => {
                        return Err(CliError::FsTx(
                            engine_pack::fs_tx::FsTxError::StagingArtifactInvariant {
                                staging: staging.clone(),
                                message: format!(
                                    "provenance hash mismatch for '{}': expected {}, got {}",
                                    key, output_hash.sha256, actual_hash
                                ),
                            },
                        ));
                    }
                    None => {
                        return Err(CliError::FsTx(
                            engine_pack::fs_tx::FsTxError::StagingArtifactInvariant {
                                staging: staging.clone(),
                                message: format!(
                                    "provenance references missing file '{}'",
                                    key
                                ),
                            },
                        ));
                    }
                }
            }
        }

        // Atomic no-replace publication (Phase 08)
        // If destination already exists, compare content hashes for idempotent skip
        if destination_exists {
            if engine_pack::fs_tx::artifact_sets_identical(&staging, &out_dir)
                .map_err(CliError::FsTx)?
            {
                // Idempotent: staging matches existing destination — skip
                return Ok(format!(
                    "compiled[bsp]: {} -> {}/{}.bsp (unchanged, skipped)",
                    source_map.display(),
                    out_dir.display(),
                    bsp_name
                ));
            } else {
                return Err(CliError::FsTx(
                    engine_pack::fs_tx::FsTxError::PreExistingDestination {
                        target: out_dir.clone(),
                        message: "destination exists with different content; \
                                  publication blocked to prevent clobber"
                            .to_string(),
                    },
                ));
            }
        }

        publish_directory_no_replace(&staging, &out_dir).map_err(CliError::FsTx)?;

        Ok(format!(
            "compiled[bsp]: {} -> {}/{}.bsp",
            source_map.display(),
            out_dir.display(),
            bsp_name
        ))
    })();

    if result.is_err() {
        cleanup_staging(&staging);
    }
    result
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
