use std::collections::HashSet;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path, PathBuf};

use renderer::{
    validate_package_manifest_file, validate_project_file, validate_scene_file_with_options,
    AssetKind, PackageManifest, PackageValidationOptions, Project, ProjectValidationOptions,
    SceneValidationOptions, ValidationArea, ValidationDiagnostic, ValidationError,
};
use serde::Serialize;

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
        Err(CliError::Validation(err)) => {
            print_validation_error(&err);
            1
        }
    };
    std::process::exit(code);
}

fn run(args: Vec<String>) -> CliResult<String> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(CliError::Usage(usage()));
    };
    let rest = &args[1..];
    match command {
        "validate-package" => validate_package_cmd(rest),
        "validate-project" => validate_project_cmd(rest),
        "validate-scene" => validate_scene_cmd(rest),
        "new-project" => new_project_cmd(rest),
        "new-package" => new_package_cmd(rest),
        "scan-assets" => scan_assets_cmd(rest),
        "add-asset" => add_asset_cmd(rest),
        "pack" => pack_cmd(rest),
        "-h" | "--help" | "help" => Ok(usage()),
        other => Err(CliError::Usage(format!(
            "unknown command '{other}'\n\n{}",
            usage()
        ))),
    }
}

fn validate_package_cmd(args: &[String]) -> CliResult<String> {
    let mut parser = ArgParser::new(args);
    let expected_package_id = parser.optional_value("--expected-package-id")?;
    let project_root = parser.optional_path("--project-root")?;
    let path = parser.required_path("validate-package <package.toml>")?;
    parser.finish()?;

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
                    renderer::ValidationArea::Package,
                    format!("failed to read package manifest: {err}"),
                )
                .with_path(&path),
            ))
        })?;
        renderer::validate_package_manifest_str(&content, project_root, &options)?
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
    let mut parser = ArgParser::new(args);
    let path = parser.required_path("validate-project <engine.project.toml>")?;
    parser.finish()?;

    let project = validate_project_and_startup_scene(&path)?;
    Ok(format!(
        "valid[project]: {} ({})",
        path.display(),
        project.project_id
    ))
}

fn validate_scene_cmd(args: &[String]) -> CliResult<String> {
    let mut parser = ArgParser::new(args);
    let project = parser.required_option_path("--project")?;
    let path = parser.required_path(
        "validate-scene <scene.engine.scene.json> --project <engine.project.toml>",
    )?;
    parser.finish()?;

    let asset_ids = collect_project_asset_ids(&project)?;
    validate_scene_file_with_options(
        &path,
        &SceneValidationOptions::default().with_known_asset_ids(asset_ids),
    )?;
    Ok(format!("valid[scene]: {}", path.display()))
}

fn new_project_cmd(args: &[String]) -> CliResult<String> {
    let mut parser = ArgParser::new(args);
    let project_id = parser.required_option_value("--id")?;
    let name = parser.required_option_value("--name")?;
    let dir = parser.required_path("new-project <dir> --id <project_id> --name <name>")?;
    parser.finish()?;

    fs::create_dir_all(dir.join("assets")).map_err(|err| io_error("project.io", &dir, err))?;
    fs::create_dir_all(dir.join("scenes")).map_err(|err| io_error("project.io", &dir, err))?;
    let project_path = dir.join("engine.project.toml");
    let content = format!(
        "format_version = 1\nproject_id = \"{}\"\nname = \"{}\"\nproject_version = \"0.1.0\"\nasset_root = \"assets\"\nstartup_scene = \"scenes/start.engine.scene.json\"\npackages = []\n\n[settings]\nwindow_width = 1280\nwindow_height = 720\nfullscreen = false\nvsync = true\n",
        toml_escape(&project_id),
        toml_escape(&name)
    );
    fs::write(&project_path, content).map_err(|err| io_error("project.io", &project_path, err))?;
    let scene_path = dir.join("scenes/start.engine.scene.json");
    fs::write(&scene_path, starter_scene_json(&project_id))
        .map_err(|err| io_error("project.io", &scene_path, err))?;
    validate_project_and_startup_scene(&project_path)?;

    Ok(format!("created[project]: {}", project_path.display()))
}

fn new_package_cmd(args: &[String]) -> CliResult<String> {
    let mut parser = ArgParser::new(args);
    let package_id = parser.required_option_value("--id")?;
    let name = parser.required_option_value("--name")?;
    let path =
        parser.required_path("new-package <path> --id <package_id> --name <display_name>")?;
    parser.finish()?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| io_error("package.io", parent, err))?;
    }
    let content = format!(
        "format_version = 1\npackage_id = \"{}\"\ndisplay_name = \"{}\"\npackage_version = \"0.1.0\"\n",
        toml_escape(&package_id),
        toml_escape(&name)
    );
    fs::write(&path, content).map_err(|err| io_error("package.io", &path, err))?;
    validate_package_manifest_file(
        &path,
        &PackageValidationOptions::default().with_expected_package_id(package_id),
    )?;

    Ok(format!("created[package]: {}", path.display()))
}

fn scan_assets_cmd(args: &[String]) -> CliResult<String> {
    let mut parser = ArgParser::new(args);
    let package_id = parser
        .optional_value("--package-id")?
        .unwrap_or_else(|| "scanned".to_string());
    let root = parser.required_path("scan-assets <asset-root> [--package-id <id>]")?;
    parser.finish()?;

    let mut records = Vec::new();
    collect_scan_records(&root, &root, &package_id, &mut records)?;
    records.sort_by(|left, right| left.path.cmp(&right.path));
    if records.is_empty() {
        return Ok("scan[assets]: no supported assets found".to_string());
    }

    let mut output = String::new();
    for record in records {
        output.push_str("[[assets]]\n");
        output.push_str(&format!("id = \"{}\"\n", toml_escape(&record.id)));
        output.push_str(&format!("kind = \"{}\"\n", record.kind.as_str()));
        output.push_str(&format!("path = \"{}\"\n", toml_escape(&record.path)));
        output.push_str(&format!(
            "display_name = \"{}\"\n\n",
            toml_escape(&record.display_name)
        ));
    }
    Ok(output.trim_end().to_string())
}

fn add_asset_cmd(args: &[String]) -> CliResult<String> {
    let mut parser = ArgParser::new(args);
    let asset_id = parser.required_option_value("--id")?;
    let kind = parser.required_option_value("--kind")?;
    let asset_path = parser.required_option_value("--path")?;
    let tags = parser.optional_values("--tag")?;
    let manifest_path = parser
        .required_path("add-asset <package.toml> --id <asset_id> --kind <kind> --path <path>")?;
    parser.finish()?;

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

    let mut content = fs::read_to_string(&manifest_path)
        .map_err(|err| io_error("package.io", &manifest_path, err))?;
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content.push('\n');
    content.push_str("[[assets]]\n");
    content.push_str(&format!("id = \"{}\"\n", toml_escape(&asset_id)));
    content.push_str(&format!("kind = \"{}\"\n", kind.as_str()));
    content.push_str(&format!("path = \"{}\"\n", toml_escape(&asset_path)));
    if !tags.is_empty() {
        let tags = tags
            .iter()
            .map(|tag| format!("\"{}\"", toml_escape(tag)))
            .collect::<Vec<_>>()
            .join(", ");
        content.push_str(&format!("tags = [{tags}]\n"));
    }
    fs::write(&manifest_path, content)
        .map_err(|err| io_error("package.io", &manifest_path, err))?;
    validate_package_manifest_file(
        &manifest_path,
        &PackageValidationOptions::default().check_source_files(true),
    )?;

    Ok(format!(
        "added[asset]: {} -> {}",
        asset_id,
        manifest_path.display()
    ))
}

fn pack_cmd(args: &[String]) -> CliResult<String> {
    let mut parser = ArgParser::new(args);
    let out = parser.required_option_path("--out")?;
    let project_path = parser.required_path("pack <engine.project.toml> --out <dir>")?;
    parser.finish()?;

    fs::create_dir_all(&out).map_err(|err| io_error("pack.io", &out, err))?;
    remove_stale_pack_report(&out)?;
    let project = validate_project_and_startup_scene(&project_path)?;
    let project_root = project_path.parent().unwrap_or_else(|| Path::new(""));

    let mut report = PackReport {
        source_project: project_path.display().to_string(),
        copied_files: Vec::new(),
        skipped_disabled_packages: Vec::new(),
        warnings: Vec::new(),
        validation_status: "passed".to_string(),
    };

    copy_file_to_pack(
        project_root,
        Path::new("engine.project.toml"),
        &out,
        &mut report,
    )?;
    if let Some(scene) = &project.startup_scene {
        copy_file_to_pack(project_root, scene, &out, &mut report)?;
    }

    for package in &project.packages {
        if !package.enabled {
            report
                .skipped_disabled_packages
                .push(package.package_id.clone());
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
        copy_file_to_pack(project_root, &manifest_rel, &out, &mut report)?;
        let manifest_path = project_root.join(&manifest_rel);
        let records = validate_package_manifest_file(
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
            copy_file_to_pack(project_root, &asset_rel, &out, &mut report)?;
        }
        debug_assert_eq!(records.len(), manifest.assets.len());
    }

    let report_path = out.join("PACK_REPORT.json");
    let report_json = serde_json::to_string_pretty(&report).map_err(|err| {
        CliError::Validation(ValidationError::single(ValidationDiagnostic::new(
            "pack.report",
            ValidationArea::Project,
            format!("failed to serialize pack report: {err}"),
        )))
    })?;
    fs::write(&report_path, report_json).map_err(|err| io_error("pack.io", &report_path, err))?;

    Ok(format!(
        "packed[project]: {} -> {} ({} files)",
        project_path.display(),
        out.display(),
        report.copied_files.len()
    ))
}

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
) -> CliResult<()> {
    let mut entries = fs::read_dir(dir)
        .map_err(|err| io_error("scan.io", dir, err))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| io_error("scan.io", dir, err))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_scan_records(root, &path, package_id, records)?;
            continue;
        }
        let Some(kind) = classify_asset_kind(&path) else {
            continue;
        };
        let relative = path.strip_prefix(root).map_err(|err| {
            CliError::Validation(ValidationError::single(ValidationDiagnostic::new(
                "scan.path",
                ValidationArea::Asset,
                format!("failed to relativize '{}': {err}", path.display()),
            )))
        })?;
        let relative_string = slash_path(relative);
        let stem = relative.with_extension("");
        let name = sanitize_id_component(&slash_path(&stem));
        records.push(ScanRecord {
            id: format!("{package_id}.{}.{}", kind.as_str(), name),
            kind,
            path: relative_string,
            display_name: display_name(path.file_stem()),
        });
    }
    Ok(())
}

fn copy_file_to_pack(
    project_root: &Path,
    relative: &Path,
    out: &Path,
    report: &mut PackReport,
) -> CliResult<()> {
    let relative = normalize_relative_path(relative).ok_or_else(|| {
        CliError::Validation(ValidationError::single(ValidationDiagnostic::new(
            "pack.invalid_copy_path",
            ValidationArea::Project,
            format!("invalid pack copy path '{}'", relative.display()),
        )))
    })?;
    let source = project_root.join(&relative);
    let destination = out.join(&relative);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|err| io_error("pack.io", parent, err))?;
    }
    fs::copy(&source, &destination).map_err(|err| io_error("pack.io", &source, err))?;
    report.copied_files.push(slash_path(&relative));
    Ok(())
}

fn remove_stale_pack_report(out: &Path) -> CliResult<()> {
    let report_path = out.join("PACK_REPORT.json");
    match fs::remove_file(&report_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(io_error("pack.io", &report_path, err)),
    }
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

fn classify_asset_kind(path: &Path) -> Option<AssetKind> {
    let extension = path.extension()?.to_string_lossy().to_ascii_lowercase();
    match extension.as_str() {
        "gltf" | "glb" | "obj" => Some(AssetKind::Model),
        "png" | "jpg" | "jpeg" | "ktx" | "ktx2" => Some(AssetKind::Texture),
        "hdr" | "exr" => Some(AssetKind::Environment),
        "wav" | "ogg" | "flac" | "mp3" => Some(AssetKind::Audio),
        _ => None,
    }
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
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => return None,
            Component::Prefix(_) | Component::RootDir => return None,
        }
    }
    (!normalized.as_os_str().is_empty()).then_some(normalized)
}

fn slash_path(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
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

fn toml_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn starter_scene_json(project_id: &str) -> String {
    format!(
        "{{\n  \"format_version\": 1,\n  \"scene_id\": \"{}.start\",\n  \"root_nodes\": [\"node.root\"],\n  \"nodes\": [\n    {{\n      \"id\": \"node.root\",\n      \"parent\": null,\n      \"name\": \"Root\",\n      \"transform\": {{\n        \"translation\": [0.0, 0.0, 0.0],\n        \"rotation\": [0.0, 0.0, 0.0, 1.0],\n        \"scale\": [1.0, 1.0, 1.0]\n      }},\n      \"asset\": null\n    }}\n  ],\n  \"lights\": [],\n  \"environment\": null,\n  \"editor\": {{}}\n}}\n",
        toml_escape(project_id)
    )
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

fn print_validation_error(err: &ValidationError) {
    for diagnostic in err.diagnostics() {
        eprintln!("{diagnostic}");
    }
}

fn usage() -> String {
    [
        "engine_pack commands:",
        "  validate-package <package.toml> [--expected-package-id <id>] [--project-root <path>]",
        "  validate-project <engine.project.toml>",
        "  validate-scene <scene.engine.scene.json> --project <engine.project.toml>",
        "  new-project <dir> --id <project_id> --name <name>",
        "  new-package <path> --id <package_id> --name <display_name>",
        "  scan-assets <asset-root> [--package-id <id>]",
        "  add-asset <package.toml> --id <asset_id> --kind <kind> --path <path> [--tag <tag>]",
        "  pack <engine.project.toml> --out <dir>",
    ]
    .join("\n")
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
    Validation(ValidationError),
}

impl From<ValidationError> for CliError {
    fn from(value: ValidationError) -> Self {
        Self::Validation(value)
    }
}

struct ArgParser<'a> {
    args: &'a [String],
    used: Vec<bool>,
}

impl<'a> ArgParser<'a> {
    fn new(args: &'a [String]) -> Self {
        Self {
            args,
            used: vec![false; args.len()],
        }
    }

    fn required_path(&mut self, usage: &str) -> CliResult<PathBuf> {
        for (index, arg) in self.args.iter().enumerate() {
            if !self.used[index] && !arg.starts_with("--") {
                self.used[index] = true;
                return Ok(PathBuf::from(arg));
            }
        }
        Err(CliError::Usage(format!("missing path: {usage}")))
    }

    fn optional_value(&mut self, flag: &str) -> CliResult<Option<String>> {
        let Some(index) = self.flag_index(flag) else {
            return Ok(None);
        };
        let value_index = index + 1;
        if value_index >= self.args.len() || self.args[value_index].starts_with("--") {
            return Err(CliError::Usage(format!("{flag} requires a value")));
        }
        self.used[index] = true;
        self.used[value_index] = true;
        Ok(Some(self.args[value_index].clone()))
    }

    fn optional_path(&mut self, flag: &str) -> CliResult<Option<PathBuf>> {
        Ok(self.optional_value(flag)?.map(PathBuf::from))
    }

    fn required_option_path(&mut self, flag: &str) -> CliResult<PathBuf> {
        self.optional_path(flag)?
            .ok_or_else(|| CliError::Usage(format!("missing required {flag}")))
    }

    fn required_option_value(&mut self, flag: &str) -> CliResult<String> {
        self.optional_value(flag)?
            .ok_or_else(|| CliError::Usage(format!("missing required {flag}")))
    }

    fn optional_values(&mut self, flag: &str) -> CliResult<Vec<String>> {
        let mut values = Vec::new();
        while let Some(value) = self.optional_value(flag)? {
            values.push(value);
        }
        Ok(values)
    }

    fn finish(&self) -> CliResult<()> {
        let unused: Vec<_> = self
            .args
            .iter()
            .enumerate()
            .filter_map(|(index, arg)| (!self.used[index]).then_some(arg.as_str()))
            .collect();
        if unused.is_empty() {
            Ok(())
        } else {
            Err(CliError::Usage(format!(
                "unexpected arguments: {}",
                unused.join(" ")
            )))
        }
    }

    fn flag_index(&self, flag: &str) -> Option<usize> {
        self.args
            .iter()
            .enumerate()
            .find_map(|(index, arg)| (!self.used[index] && arg == flag).then_some(index))
    }
}
