use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};

use renderer::{
    validate_package_manifest_file, validate_project_file, validate_scene_file_with_options,
    PackageValidationOptions, Project, ProjectValidationOptions, SceneValidationOptions,
    ValidationDiagnostic, ValidationError,
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
    ]
    .join("\n")
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
