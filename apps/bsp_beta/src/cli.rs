//! CLI argument parsing for the BSP beta application.
//!
//! Supported flags are intentionally space-separated only:
//!   --bsp <path>          Path to a compiled .bsp file (required by runtime).
//!   --scale <float>       Quake→engine scale factor (default: 0.0254).
//!   --headless            Run in headless mode (no window).
//!   --capture-frames <n>  Number of frames to capture in headless mode.
//!   --lights              Log all imported light descriptors.

use std::fmt;
use std::path::PathBuf;

/// Parsed CLI arguments for the BSP beta application.
#[derive(Debug, Clone, PartialEq)]
pub struct CliArgs {
    /// Path to the compiled BSP file.
    pub bsp_path: Option<PathBuf>,
    /// Quake-to-engine scale factor.
    pub scale: f32,
    /// Whether to run in headless mode.
    pub headless: bool,
    /// Number of frames to capture (headless only, 0 = no capture).
    pub capture_frames: u32,
    /// Whether to log imported light descriptors at startup.
    pub show_lights: bool,
}

/// CLI parse failure with usage-facing wording.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    MissingValue(&'static str),
    InvalidScale(String),
    NonFiniteScale(String),
    InvalidCaptureFrames(String),
    UnknownArgument(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::MissingValue(flag) => write!(f, "{flag} requires a value"),
            CliError::InvalidScale(value) => {
                write!(
                    f,
                    "--scale expects a finite floating-point number, got '{value}'"
                )
            }
            CliError::NonFiniteScale(value) => {
                write!(
                    f,
                    "--scale expects a finite floating-point number, got '{value}'"
                )
            }
            CliError::InvalidCaptureFrames(value) => {
                write!(
                    f,
                    "--capture-frames expects a non-negative integer, got '{value}'"
                )
            }
            CliError::UnknownArgument(arg) => write!(f, "unknown argument: {arg}"),
        }
    }
}

impl std::error::Error for CliError {}

impl CliArgs {
    /// Parse CLI arguments from the process args (expects `std::env::args()`).
    pub fn parse() -> Self {
        match parse_from(std::env::args().skip(1)) {
            Ok(args) => args,
            Err(err) => {
                eprintln!("{err}");
                print_usage();
                std::process::exit(1);
            }
        }
    }
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            bsp_path: None,
            scale: 0.0254,
            headless: false,
            capture_frames: 0,
            show_lights: false,
        }
    }
}

/// Parse BSP beta CLI arguments. Only space-separated flag values are supported.
pub fn parse_from(args: impl IntoIterator<Item = impl Into<String>>) -> Result<CliArgs, CliError> {
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut opts = CliArgs::default();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--bsp" => {
                let value = next_value(&args, i, "--bsp")?;
                opts.bsp_path = Some(PathBuf::from(value));
                i += 2;
            }
            "--scale" => {
                let value = next_value(&args, i, "--scale")?;
                let scale = value
                    .parse::<f32>()
                    .map_err(|_| CliError::InvalidScale(value.to_string()))?;
                if !scale.is_finite() {
                    return Err(CliError::NonFiniteScale(value.to_string()));
                }
                opts.scale = scale;
                i += 2;
            }
            "--headless" => {
                opts.headless = true;
                i += 1;
            }
            "--capture-frames" => {
                let value = next_value(&args, i, "--capture-frames")?;
                opts.capture_frames = value
                    .parse::<u32>()
                    .map_err(|_| CliError::InvalidCaptureFrames(value.to_string()))?;
                i += 2;
            }
            "--lights" => {
                opts.show_lights = true;
                i += 1;
            }
            other => return Err(CliError::UnknownArgument(other.to_string())),
        }
    }

    Ok(opts)
}

fn next_value<'a>(
    args: &'a [String],
    flag_index: usize,
    flag: &'static str,
) -> Result<&'a str, CliError> {
    let Some(value) = args.get(flag_index + 1) else {
        return Err(CliError::MissingValue(flag));
    };
    if value.starts_with("--") {
        return Err(CliError::MissingValue(flag));
    }
    Ok(value)
}

fn print_usage() {
    eprintln!();
    eprintln!("BSP Beta — Maintained Load-Query-Physics-Behavior-Reload Proof");
    eprintln!();
    eprintln!("Usage: bsp_beta [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --bsp <path>           Path to compiled .bsp file");
    eprintln!("  --scale <float>        Quake→engine scale factor (default: 0.0254)");
    eprintln!("  --headless             Run headless (no window, renders N frames)");
    eprintln!("  --capture-frames <n>   Frame count for headless capture (default: 0)");
    eprintln!("  --lights               Log all imported light descriptors at startup");
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_args() {
        let args = parse_from(Vec::<&str>::new()).unwrap();
        assert!(args.bsp_path.is_none());
        assert!(!args.headless);
        assert_eq!(args.capture_frames, 0);
        assert!(!args.show_lights);
        assert!((args.scale - 0.0254).abs() < 1e-6);
    }

    #[test]
    fn parse_bsp_path() {
        let args = parse_from(["--bsp", "maps/test.bsp"]).unwrap();
        assert_eq!(args.bsp_path, Some(PathBuf::from("maps/test.bsp")));
    }

    #[test]
    fn parse_headless_with_capture() {
        let args = parse_from(["--headless", "--capture-frames", "10"]).unwrap();
        assert!(args.headless);
        assert_eq!(args.capture_frames, 10);
    }

    #[test]
    fn parse_scale() {
        let args = parse_from(["--scale", "0.03125"]).unwrap();
        assert!((args.scale - 0.03125).abs() < 1e-6);
    }

    #[test]
    fn parse_lights_flag() {
        let args = parse_from(["--lights"]).unwrap();
        assert!(args.show_lights);
    }

    #[test]
    fn parse_combined() {
        let args = parse_from([
            "--bsp",
            "maps/e1m1.bsp",
            "--scale",
            "0.05",
            "--headless",
            "--capture-frames",
            "5",
            "--lights",
        ])
        .unwrap();
        assert_eq!(args.bsp_path, Some(PathBuf::from("maps/e1m1.bsp")));
        assert!((args.scale - 0.05).abs() < 1e-6);
        assert!(args.headless);
        assert_eq!(args.capture_frames, 5);
        assert!(args.show_lights);
    }

    #[test]
    fn reject_equals_form_and_unknown_flags() {
        assert_eq!(
            parse_from(["--bsp=maps/e1m1.bsp"]).unwrap_err(),
            CliError::UnknownArgument("--bsp=maps/e1m1.bsp".to_string())
        );
        assert_eq!(
            parse_from(["--unknown"]).unwrap_err(),
            CliError::UnknownArgument("--unknown".to_string())
        );
    }

    #[test]
    fn reject_missing_and_malformed_values() {
        assert_eq!(
            parse_from(["--bsp"]).unwrap_err(),
            CliError::MissingValue("--bsp")
        );
        assert_eq!(
            parse_from(["--bsp", "--headless"]).unwrap_err(),
            CliError::MissingValue("--bsp")
        );
        assert!(matches!(
            parse_from(["--scale", "nan"]).unwrap_err(),
            CliError::NonFiniteScale(_)
        ));
        assert!(matches!(
            parse_from(["--capture-frames", "-1"]).unwrap_err(),
            CliError::InvalidCaptureFrames(_)
        ));
    }
}
