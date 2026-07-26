//! CLI argument parsing for the BSP beta application.
//!
//! Supported flags are intentionally space-separated only:
//!   --bsp <path>          Path to a compiled .bsp file (required by runtime).
//!   --scale <float>       Quake→engine scale factor (default: 0.0254).
//!   --headless            Run in headless mode (no window).
//!   --mcp                 Run a headless MCP JSON-RPC server over stdio.
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
    /// Whether to serve MCP JSON-RPC requests over stdio.
    pub mcp: bool,
    /// Number of frames to capture (headless only, 0 = no capture).
    pub capture_frames: u32,
    /// Whether to log imported light descriptors at startup.
    pub show_lights: bool,
    /// Path to a 768-byte palette .lmp file.
    pub palette_path: Option<PathBuf>,
    /// Path to a .lit colored-light companion file.
    pub lit_path: Option<PathBuf>,
    /// Explicit path to a WAD file for texture resolution.
    pub wad_path: Option<PathBuf>,
    /// Import mode: strict or development.
    pub import_mode: Option<ImportMode>,
    /// Companion textures directory for PBR discovery.
    pub textures_dir: Option<PathBuf>,
}

/// Import mode for CLI — mutually exclusive --strict and --development.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportMode {
    Strict,
    Development,
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
    WadNotFound(String),
    /// Both --strict and --development specified.
    ConflictingImportMode,
    /// No import mode selected; --strict or --development required.
    NoImportMode,
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
            CliError::WadNotFound(path) => write!(f, "WAD file not found: {path}"),
            CliError::ConflictingImportMode => {
                write!(f, "--strict and --development are mutually exclusive")
            }
            CliError::NoImportMode => {
                write!(f, "no import mode selected; use --strict or --development")
            }
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

    /// Resolve the effective palette path: explicit --palette only.
    pub fn resolve_palette_path(&self) -> Result<PathBuf, CliError> {
        if let Some(ref path) = self.palette_path {
            if path.is_file() {
                return Ok(path.clone());
            }
            return Err(CliError::PaletteNotFound(path.display().to_string()));
        }
        Err(CliError::PaletteNotFound(
            "provide --palette <path>".to_string(),
        ))
    }

    /// Resolve the effective .lit path: explicit --lit only.
    pub fn resolve_lit_path(&self) -> Option<PathBuf> {
        if let Some(ref p) = self.lit_path {
            if p.is_file() {
                return Some(p.clone());
            }
        }
        None
    }

    /// Resolve the effective WAD path: explicit --wad only.
    pub fn resolve_wad_path(&self) -> Result<Option<PathBuf>, CliError> {
        if let Some(ref p) = self.wad_path {
            if p.is_file() {
                return Ok(Some(p.clone()));
            }
            return Err(CliError::WadNotFound(p.display().to_string()));
        }
        Ok(None)
    }

    /// Resolve the import mode; error if not specified.
    pub fn require_import_mode(&self) -> Result<ImportMode, CliError> {
        self.import_mode.ok_or(CliError::NoImportMode)
    }
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            bsp_path: None,
            scale: 0.0254,
            headless: false,
            mcp: false,
            capture_frames: 0,
            show_lights: false,
            palette_path: None,
            lit_path: None,
            wad_path: None,
            import_mode: None,
            textures_dir: None,
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
            "--mcp" => {
                opts.mcp = true;
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
            "--wad" => {
                let value = next_value(&args, i, "--wad")?;
                opts.wad_path = Some(PathBuf::from(value));
                i += 2;
            }
            "--strict" => {
                if opts.import_mode == Some(ImportMode::Development) {
                    return Err(CliError::ConflictingImportMode);
                }
                opts.import_mode = Some(ImportMode::Strict);
                i += 1;
            }
            "--development" => {
                if opts.import_mode == Some(ImportMode::Strict) {
                    return Err(CliError::ConflictingImportMode);
                }
                opts.import_mode = Some(ImportMode::Development);
                i += 1;
            }
            "--textures" => {
                let value = next_value(&args, i, "--textures")?;
                opts.textures_dir = Some(PathBuf::from(value));
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
    eprintln!("Usage: bsp_beta --strict|--development [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --strict               Strict import mode");
    eprintln!("  --development          Development import mode");
    eprintln!("  --bsp <path>           Path to compiled .bsp file");
    eprintln!("  --scale <float>        Quake→engine scale factor (default: 0.0254)");
    eprintln!("  --headless             Run headless (no window, renders N frames)");
    eprintln!("  --mcp                  Run headless MCP JSON-RPC server over stdio");
    eprintln!("  --capture-frames <n>   Frame count for headless capture (default: 0)");
    eprintln!("  --lights               Log all imported light descriptors at startup");
    eprintln!("  --palette <path>       Path to 768-byte palette .lmp file");
    eprintln!("  --lit <path>           Path to .lit colored-light companion file");
    eprintln!("  --wad <path>           Path to WAD file for texture resolution");
    eprintln!("  --textures <dir>       Textures directory for PBR companion discovery");
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
        assert!(!args.mcp);
        assert_eq!(args.capture_frames, 0);
        assert!(!args.show_lights);
        assert!(args.palette_path.is_none());
        assert!(args.lit_path.is_none());
        assert!(args.textures_dir.is_none());
        assert!(args.import_mode.is_none());
        assert!((args.scale - 0.0254).abs() < 1e-6);
    }

    #[test]
    fn parse_bsp_path() {
        let args = parse_from(["--strict", "--bsp", "maps/test.bsp"]).unwrap();
        assert_eq!(args.bsp_path, Some(PathBuf::from("maps/test.bsp")));
    }

    #[test]
    fn parse_headless_with_capture() {
        let args =
            parse_from(["--development", "--headless", "--capture-frames", "10"]).unwrap();
        assert!(args.headless);
        assert!(!args.mcp);
        assert_eq!(args.capture_frames, 10);
    }

    #[test]
    fn parse_mcp_implies_headless() {
        let args = parse_from(["--strict", "--mcp"]).unwrap();
        assert!(args.mcp);
        assert!(args.headless);
    }

    #[test]
    fn parse_scale() {
        let args = parse_from(["--strict", "--scale", "0.03125"]).unwrap();
        assert!((args.scale - 0.03125).abs() < 1e-6);
    }

    #[test]
    fn parse_lights_flag() {
        let args = parse_from(["--development", "--lights"]).unwrap();
        assert!(args.show_lights);
    }

    #[test]
    fn parse_strict_mode() {
        let args = parse_from(["--strict"]).unwrap();
        assert_eq!(args.import_mode, Some(ImportMode::Strict));
    }

    #[test]
    fn parse_development_mode() {
        let args = parse_from(["--development"]).unwrap();
        assert_eq!(args.import_mode, Some(ImportMode::Development));
    }

    #[test]
    fn parse_textures_dir() {
        let args = parse_from(["--strict", "--textures", "gfx/textures"]).unwrap();
        assert_eq!(args.textures_dir, Some(PathBuf::from("gfx/textures")));
    }

    #[test]
    fn conflicting_import_modes_rejected() {
        let err = parse_from(["--strict", "--development"]).unwrap_err();
        assert_eq!(err, CliError::ConflictingImportMode);
    }

    #[test]
    fn no_import_mode_is_allowed_by_parser() {
        // Parser allows no mode; app validates later.
        let args = parse_from(["--bsp", "maps/test.bsp"]).unwrap();
        assert!(args.import_mode.is_none());
        assert!(args.require_import_mode().is_err());
    }

    #[test]
    fn parse_combined() {
        let args = parse_from([
            "--strict",
            "--bsp",
            "maps/e1m1.bsp",
            "--scale",
            "0.05",
            "--headless",
            "--mcp",
            "--capture-frames",
            "5",
            "--lights",
            "--palette",
            "gfx/palette.lmp",
            "--lit",
            "maps/e1m1.lit",
            "--wad",
            "maps/dungeon.wad",
            "--textures",
            "gfx/textures",
        ])
        .unwrap();
        assert_eq!(args.bsp_path, Some(PathBuf::from("maps/e1m1.bsp")));
        assert!((args.scale - 0.05).abs() < 1e-6);
        assert!(args.headless);
        assert!(args.mcp);
        assert_eq!(args.capture_frames, 5);
        assert!(args.show_lights);
        assert_eq!(args.palette_path, Some(PathBuf::from("gfx/palette.lmp")));
        assert_eq!(args.lit_path, Some(PathBuf::from("maps/e1m1.lit")));
        assert_eq!(args.wad_path, Some(PathBuf::from("maps/dungeon.wad")));
        assert_eq!(args.textures_dir, Some(PathBuf::from("gfx/textures")));
        assert_eq!(args.import_mode, Some(ImportMode::Strict));
    }

    #[test]
    fn reject_equals_form_and_unknown_flags() {
        assert_eq!(
            parse_from(["--strict", "--bsp=maps/e1m1.bsp"]).unwrap_err(),
            CliError::UnknownArgument("--bsp=maps/e1m1.bsp".to_string())
        );
        assert_eq!(
            parse_from(["--strict", "--unknown"]).unwrap_err(),
            CliError::UnknownArgument("--unknown".to_string())
        );
    }

    #[test]
    fn parse_palette_and_lit_flags() {
        let args =
            parse_from(["--strict", "--palette", "gfx/pal.lmp", "--lit", "maps/test.lit"])
                .unwrap();
        assert_eq!(args.palette_path, Some(PathBuf::from("gfx/pal.lmp")));
        assert_eq!(args.lit_path, Some(PathBuf::from("maps/test.lit")));
    }

    #[test]
    fn parse_wad_and_textures() {
        let args = parse_from([
            "--strict",
            "--wad",
            "assets/dungeon.wad",
            "--textures",
            "assets/textures",
        ])
        .unwrap();
        assert_eq!(args.wad_path, Some(PathBuf::from("assets/dungeon.wad")));
        assert_eq!(args.textures_dir, Some(PathBuf::from("assets/textures")));
    }

    #[test]
    fn palette_resolution_requires_explicit_path() {
        let args = CliArgs {
            palette_path: Some(PathBuf::from("/tmp/pal.lmp")),
            ..CliArgs::default()
        };
        // File doesn't exist, so it fails.
        assert!(matches!(
            args.resolve_palette_path(),
            Err(CliError::PaletteNotFound(_))
        ));
    }

    #[test]
    fn no_mode_reports_error() {
        let args = CliArgs::default();
        assert_eq!(args.require_import_mode().unwrap_err(), CliError::NoImportMode);
    }

    #[test]
    fn reject_missing_and_malformed_values() {
        assert_eq!(
            parse_from(["--strict", "--bsp"]).unwrap_err(),
            CliError::MissingValue("--bsp")
        );
        assert_eq!(
            parse_from(["--strict", "--bsp", "--headless"]).unwrap_err(),
            CliError::MissingValue("--bsp")
        );
        assert!(matches!(
            parse_from(["--development", "--scale", "nan"]).unwrap_err(),
            CliError::NonFiniteScale(_)
        ));
        assert!(matches!(
            parse_from(["--development", "--capture-frames", "-1"]).unwrap_err(),
            CliError::InvalidCaptureFrames(_)
        ));
    }
}
