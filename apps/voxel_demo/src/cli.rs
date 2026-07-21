//! CLI parsing with explicit presence tracking and typed errors.

use std::collections::HashSet;
use std::ffi::OsString;
use std::path::PathBuf;

/// CLI arguments with presence tracking for document and runtime overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct CliArgs {
    pub preset: Option<String>,
    pub config: Option<PathBuf>,

    pub seed: Option<u64>,
    pub resolution: Option<u32>,
    pub shell_thickness: Option<u32>,
    pub cavern_count: Option<u32>,
    pub tunnel_count: Option<u32>,
    pub tunnel_radius_min: Option<f32>,
    pub tunnel_radius_max: Option<f32>,
    pub cavern_radius_min: Option<f32>,
    pub cavern_radius_max: Option<f32>,
    pub spline_tension: Option<f32>,
    pub roughness: Option<f32>,
    pub maze_density: Option<f32>,
    pub maze_twistiness: Option<f32>,
    pub maze_radius: Option<f32>,
    pub maze_retries: Option<u32>,
    pub maze_search_budget: Option<u32>,
    pub floor_threshold: Option<f32>,
    pub wall_uv_scale: Option<f32>,
    pub floor_uv_scale: Option<f32>,

    pub light_budget: Option<u32>,
    pub headless: bool,
    pub capture_dir: Option<PathBuf>,
    pub env_path: Option<PathBuf>,

    /// True when a v2-only selector or override was present. Shared legacy
    /// flags alone continue to select the unchanged v1 command route.
    pub is_v2: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    NonUtf8Argument,
    UnknownArgument(String),
    MissingValue {
        flag: String,
    },
    MalformedValue {
        flag: String,
        value: String,
        expected: &'static str,
    },
    DuplicateOption {
        flag: String,
    },
    BaseConflict,
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonUtf8Argument => write!(f, "arguments must be valid UTF-8"),
            Self::UnknownArgument(arg) => write!(f, "unknown argument: {arg}"),
            Self::MissingValue { flag } => write!(f, "{flag} requires a value"),
            Self::MalformedValue {
                flag,
                value,
                expected,
            } => {
                write!(f, "{flag} expects {expected}, got '{value}'")
            }
            Self::DuplicateOption { flag } => write!(f, "{flag} specified more than once"),
            Self::BaseConflict => write!(f, "--preset and --config are mutually exclusive"),
        }
    }
}

impl CliArgs {
    pub fn parse() -> Self {
        match Self::parse_from(std::env::args_os().skip(1)) {
            Ok(ParseOutcome::Run(args)) => args,
            Ok(ParseOutcome::Help) => {
                print_help();
                std::process::exit(0);
            }
            Err(error) => {
                eprintln!("error: {error}");
                eprintln!("use --help for usage");
                std::process::exit(1);
            }
        }
    }

    pub fn parse_from<I, S>(args: I) -> Result<ParseOutcome, CliError>
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        let args = args
            .into_iter()
            .map(|arg| {
                arg.into()
                    .into_string()
                    .map_err(|_| CliError::NonUtf8Argument)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut parsed = Self::default();
        let mut seen = HashSet::new();
        let mut i = 0;

        while i < args.len() {
            let flag = args[i].as_str();
            let logical_flag = match flag {
                "--capture_dir" => "--capture-dir",
                other => other,
            };
            if flag == "--help" {
                return Ok(ParseOutcome::Help);
            }
            if !seen.insert(logical_flag.to_string()) {
                return Err(CliError::DuplicateOption {
                    flag: logical_flag.into(),
                });
            }

            macro_rules! value {
                ($expected:literal, $ty:ty) => {{
                    let raw = require_value(&args, &mut i, flag)?;
                    raw.parse::<$ty>().map_err(|_| CliError::MalformedValue {
                        flag: flag.into(),
                        value: raw,
                        expected: $expected,
                    })?
                }};
            }

            match flag {
                "--preset" => {
                    parsed.preset = Some(require_value(&args, &mut i, flag)?);
                    parsed.is_v2 = true;
                }
                "--config" => {
                    parsed.config = Some(PathBuf::from(require_value(&args, &mut i, flag)?));
                    parsed.is_v2 = true;
                }
                "--seed" => parsed.seed = Some(value!("a non-negative integer", u64)),
                "--resolution" => parsed.resolution = Some(value!("a non-negative integer", u32)),
                "--shell-thickness" => {
                    parsed.shell_thickness = Some(value!("a non-negative integer", u32));
                }
                "--cavern-count" => {
                    parsed.cavern_count = Some(value!("a non-negative integer", u32));
                    parsed.is_v2 = true;
                }
                "--tunnel-count" => {
                    parsed.tunnel_count = Some(value!("a non-negative integer", u32));
                    parsed.is_v2 = true;
                }
                "--tunnel-radius-min" => {
                    parsed.tunnel_radius_min = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--tunnel-radius-max" => {
                    parsed.tunnel_radius_max = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--cavern-radius-min" => {
                    parsed.cavern_radius_min = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--cavern-radius-max" => {
                    parsed.cavern_radius_max = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--spline-tension" => {
                    parsed.spline_tension = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--roughness" => {
                    parsed.roughness = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--maze-density" => {
                    parsed.maze_density = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--maze-twistiness" => {
                    parsed.maze_twistiness = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--maze-radius" => {
                    parsed.maze_radius = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--maze-retries" => {
                    parsed.maze_retries = Some(value!("a non-negative integer", u32));
                    parsed.is_v2 = true;
                }
                "--maze-search-budget" => {
                    parsed.maze_search_budget = Some(value!("a non-negative integer", u32));
                    parsed.is_v2 = true;
                }
                "--floor-threshold" => {
                    parsed.floor_threshold = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--wall-uv-scale" => {
                    parsed.wall_uv_scale = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--floor-uv-scale" => {
                    parsed.floor_uv_scale = Some(value!("a floating-point number", f32));
                    parsed.is_v2 = true;
                }
                "--light-budget" => {
                    parsed.light_budget = Some(value!("a non-negative integer", u32));
                }
                "--headless" => {
                    parsed.headless = true;
                    i += 1;
                }
                "--capture_dir" | "--capture-dir" => {
                    parsed.capture_dir = Some(PathBuf::from(require_value(&args, &mut i, flag)?));
                }
                "--env" => {
                    parsed.env_path = Some(PathBuf::from(require_value(&args, &mut i, flag)?));
                }
                other => return Err(CliError::UnknownArgument(other.into())),
            }
        }

        if parsed.preset.is_some() && parsed.config.is_some() {
            return Err(CliError::BaseConflict);
        }
        Ok(ParseOutcome::Run(parsed))
    }

    pub fn to_v1_normalized(&self) -> crate::config::NormalizedConfig {
        crate::config::NormalizedConfig {
            seed: self.seed.unwrap_or(0),
            resolution: self.resolution.unwrap_or(96),
            shell_thickness: self.shell_thickness.unwrap_or(2),
            light_budget: self.light_budget.unwrap_or(9),
        }
    }

    pub fn to_v1_presentation(&self) -> crate::config::PresentationConfig {
        crate::config::PresentationConfig {
            headless: self.headless,
            capture_dir: self.capture_dir.clone(),
            env_path: self.env_path.clone(),
        }
    }
}

fn require_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, CliError> {
    *index += 1;
    let value = args
        .get(*index)
        .ok_or_else(|| CliError::MissingValue { flag: flag.into() })?;
    if value.starts_with("--") {
        return Err(CliError::MissingValue { flag: flag.into() });
    }
    *index += 1;
    Ok(value.clone())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseOutcome {
    Run(CliArgs),
    Help,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            preset: None,
            config: None,
            seed: None,
            resolution: None,
            shell_thickness: None,
            cavern_count: None,
            tunnel_count: None,
            tunnel_radius_min: None,
            tunnel_radius_max: None,
            cavern_radius_min: None,
            cavern_radius_max: None,
            spline_tension: None,
            roughness: None,
            maze_density: None,
            maze_twistiness: None,
            maze_radius: None,
            maze_retries: None,
            maze_search_budget: None,
            floor_threshold: None,
            wall_uv_scale: None,
            floor_uv_scale: None,
            light_budget: None,
            headless: false,
            capture_dir: None,
            env_path: None,
            is_v2: false,
        }
    }
}

fn print_help() {
    println!("voxel_demo — Voxel cave generation and rendering demo");
    println!("USAGE: cargo run -p voxel_demo -- [OPTIONS]");
    println!("BASE: --preset <default|cavernous|mazy|tight> | --config <PATH>");
    println!("GENERATOR: --seed --resolution --shell-thickness --cavern-count --tunnel-count");
    println!("           --tunnel-radius-min --tunnel-radius-max --cavern-radius-min");
    println!("           --cavern-radius-max --spline-tension --roughness --maze-density");
    println!("           --maze-twistiness --maze-radius --maze-retries --maze-search-budget");
    println!("           --floor-threshold --wall-uv-scale --floor-uv-scale");
    println!("RUNTIME: --light-budget <9-16> --headless --capture-dir <PATH> --env <PATH>");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<CliArgs, CliError> {
        match CliArgs::parse_from(args.iter().copied())? {
            ParseOutcome::Run(args) => Ok(args),
            ParseOutcome::Help => panic!("unexpected help"),
        }
    }

    #[test]
    fn default_is_legacy_with_presence_absent() {
        let args = parse(&[]).unwrap();
        assert!(!args.is_v2);
        assert_eq!(args.seed, None);
        assert_eq!(args.light_budget, None);
    }

    #[test]
    fn parses_explicit_zero_and_all_overrides() {
        let args = parse(&[
            "--preset",
            "default",
            "--seed",
            "0",
            "--resolution",
            "64",
            "--shell-thickness",
            "0",
            "--cavern-count",
            "5",
            "--tunnel-count",
            "4",
            "--tunnel-radius-min",
            "0",
            "--tunnel-radius-max",
            "2",
            "--cavern-radius-min",
            "2",
            "--cavern-radius-max",
            "4",
            "--spline-tension",
            "0",
            "--roughness",
            "0",
            "--maze-density",
            "0",
            "--maze-twistiness",
            "0",
            "--maze-radius",
            "1",
            "--maze-retries",
            "1",
            "--maze-search-budget",
            "1",
            "--floor-threshold",
            "0",
            "--wall-uv-scale",
            "1",
            "--floor-uv-scale",
            "1",
            "--light-budget",
            "9",
        ])
        .unwrap();
        assert!(args.is_v2);
        assert_eq!(args.seed, Some(0));
        assert_eq!(args.shell_thickness, Some(0));
        assert_eq!(args.maze_density, Some(0.0));
        assert_eq!(args.light_budget, Some(9));
    }

    #[test]
    fn rejects_base_conflict() {
        assert_eq!(
            parse(&["--preset", "default", "--config", "x.toml"]),
            Err(CliError::BaseConflict)
        );
    }

    #[test]
    fn rejects_duplicate_selector_and_override() {
        assert!(matches!(
            parse(&["--preset", "default", "--preset", "mazy"]),
            Err(CliError::DuplicateOption { .. })
        ));
        assert!(matches!(
            parse(&["--seed", "1", "--seed", "2"]),
            Err(CliError::DuplicateOption { .. })
        ));
    }

    #[test]
    fn rejects_missing_malformed_and_unknown_values() {
        assert!(matches!(
            parse(&["--seed"]),
            Err(CliError::MissingValue { .. })
        ));
        assert!(matches!(
            parse(&["--seed", "nope"]),
            Err(CliError::MalformedValue { .. })
        ));
        assert!(matches!(
            parse(&["--no-such-option"]),
            Err(CliError::UnknownArgument(_))
        ));
    }

    #[test]
    fn capture_aliases_are_equivalent_and_duplicate_as_one_option() {
        assert_eq!(
            parse(&["--capture_dir", "a"]).unwrap().capture_dir,
            Some(PathBuf::from("a"))
        );
        assert_eq!(
            parse(&["--capture-dir", "b"]).unwrap().capture_dir,
            Some(PathBuf::from("b"))
        );
        assert!(matches!(
            parse(&["--capture_dir", "a", "--capture-dir", "b"]),
            Err(CliError::DuplicateOption { .. })
        ));
    }

    #[test]
    fn shared_flags_preserve_legacy_route() {
        let args = parse(&[
            "--seed",
            "42",
            "--resolution",
            "64",
            "--shell-thickness",
            "3",
            "--light-budget",
            "12",
            "--headless",
        ])
        .unwrap();
        assert!(!args.is_v2);
        let config = args.to_v1_normalized();
        assert_eq!(config.seed, 42);
        assert_eq!(config.light_budget, 12);
        assert!(args.to_v1_presentation().headless);
    }

    #[test]
    fn v2_specific_override_selects_v2() {
        assert!(parse(&["--maze-density", "0.2"]).unwrap().is_v2);
    }
}
