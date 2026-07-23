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
    /// Path to a 768-byte palette .lmp file.
    pub palette_path: Option<PathBuf>,
    /// Path to a .lit colored-light companion file.
    pub lit_path: Option<PathBuf>,
    /// Directory to auto-discover .lit and palette companions next to the .bsp.
    pub companion_dir: Option<PathBuf>,
}

/// CLI parse failure with usage-facing wording.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    MissingValue(&'static str),
    InvalidScale(String),
    NonFiniteScale(String),
    InvalidCaptureFrames(String),
    UnknownArgument(String),
    PaletteNotFound(String),
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
            CliError::PaletteNotFound(path) => write!(f, "palette file not found: {path}"),
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

    /// Resolve the effective palette path from explicit arg, companion dir, or default search.
    pub fn resolve_palette_path(&self) -> Result<PathBuf, CliError> {
        if let Some(ref p) = self.palette_path {
            if p.is_file() {
                return Ok(p.clone());
            }
            return Err(CliError::PaletteNotFound(p.display().to_string()));
        }
        if let Some(ref dir) = self.companion_dir {
            let candidate = dir.join("palette.lmp");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        // Auto-discover next to the .bsp
        if let Some(ref bsp) = self.bsp_path {
            for name in &["palette.lmp", "project_palette.lmp"] {
                let candidate = bsp.with_file_name(name);
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
        // Fallback to the test fixture
        let fixture = PathBuf::from("src/bsp/tests/fixtures/palettes/project_palette.lmp");
        if fixture.is_file() {
            return Ok(fixture);
        }
        Err(CliError::PaletteNotFound("no palette found".into()))
    }

    /// Resolve the effective .lit path from explicit arg, companion dir, or auto-discovery.
    pub fn resolve_lit_path(&self) -> Option<PathBuf> {
        if let Some(ref p) = self.lit_path {
            if p.is_file() {
                return Some(p.clone());
            }
        }
        if let Some(ref dir) = self.companion_dir {
            let candidate = dir.join(
                self.bsp_path
                    .as_ref()
                    .and_then(|b| b.file_stem())
                    .map(|s| format!("{}.lit", s.to_string_lossy()))
                    .unwrap_or_default(),
            );
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        // Auto-discover next to the .bsp (same stem, .lit extension)
        if let Some(ref bsp) = self.bsp_path {
            let candidate = bsp.with_extension("lit");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        None
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
            palette_path: None,
            lit_path: None,
            companion_dir: None,
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
            "--palette" => {
                let value = next_value(&args, i, "--palette")?;
                opts.palette_path = Some(PathBuf::from(value));
                i += 2;
            }
            "--lit" => {
                let value = next_value(&args, i, "--lit")?;
                opts.lit_path = Some(PathBuf::from(value));
                i += 2;
            }
            "--companion-dir" => {
                let value = next_value(&args, i, "--companion-dir")?;
                opts.companion_dir = Some(PathBuf::from(value));
                i += 2;
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
    eprintln!("  --palette <path>       Path to 768-byte palette .lmp file");
    eprintln!("  --lit <path>           Path to .lit colored-light companion file");
    eprintln!("  --companion-dir <path> Directory to auto-discover .lit and palette");
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
        assert!(args.palette_path.is_none());
        assert!(args.lit_path.is_none());
        assert!(args.companion_dir.is_none());
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
            "--palette",
            "gfx/palette.lmp",
            "--lit",
            "maps/e1m1.lit",
            "--companion-dir",
            "maps/",
        ])
        .unwrap();
        assert_eq!(args.bsp_path, Some(PathBuf::from("maps/e1m1.bsp")));
        assert!((args.scale - 0.05).abs() < 1e-6);
        assert!(args.headless);
        assert_eq!(args.capture_frames, 5);
        assert!(args.show_lights);
        assert_eq!(args.palette_path, Some(PathBuf::from("gfx/palette.lmp")));
        assert_eq!(args.lit_path, Some(PathBuf::from("maps/e1m1.lit")));
        assert_eq!(args.companion_dir, Some(PathBuf::from("maps/")));
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
    fn parse_palette_and_lit_flags() {
        let args = parse_from(["--palette", "gfx/pal.lmp", "--lit", "maps/test.lit"]).unwrap();
        assert_eq!(args.palette_path, Some(PathBuf::from("gfx/pal.lmp")));
        assert_eq!(args.lit_path, Some(PathBuf::from("maps/test.lit")));
    }

    #[test]
    fn parse_companion_dir() {
        let args = parse_from(["--companion-dir", "assets/companions"]).unwrap();
        assert_eq!(
            args.companion_dir,
            Some(PathBuf::from("assets/companions"))
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
