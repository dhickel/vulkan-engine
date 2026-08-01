//! CLI argument parsing for the BSP beta application.
//!
//! Supported flags are intentionally space-separated only:
//!   --bsp <path>          Path to a compiled .bsp file (required by runtime).
//!   --m3-generate         Generate an EnhancedV3 dungeon from scratch.
//!   --seed <u64>          Generation seed (default: current system time).
//!   --preset <name>       sparse, moderate, or rich (default: moderate).
//!   --rooms <n>           Exact room-count override.
//!   --corridors <n>       Exact corridor-count override.
//!   --loops <n>           Exact loop-count override.
//!   --chamfer             Enable chamfered rooms (default).
//!   --no-chamfer          Disable chamfered rooms.
//!   --arch-type <name>    none, pointed, or segmented (default: pointed).
//!   --grammar-families    Comma-separated grammar allowlist (default: all six).
//!   --ericw-tools <dir>   Path to ericw-tools bin directory for compilation.
//!   --scale <float>       Quake→engine scale factor (default: 0.0254).
//!   --headless            Run in headless mode (no window).
//!   --mcp                 Run a headless MCP JSON-RPC server over stdio.
//!   --capture-frames <n>  Number of frames to capture in headless mode.
//!   --lights              Log all imported light descriptors.

use std::fmt;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use bsp_generator::enhanced_v3::{ArchType, V3Preset, GRAMMAR_FAMILIES};

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
    /// Phase 07: Request a stats evidence report after mount (headless mode).
    pub stats: bool,
    /// Phase 07: Use all-visible visibility for evidence (valid only with --stats).
    pub all_visible: bool,
    /// Phase 07: Corpus identity for evidence report.
    pub corpus_identity: Option<String>,
    /// Phase 09: Acceptance camera label (spawn, corridor, junction).
    /// When set, applies frozen capture settings (1280×720, exposure 1.0,
    /// overbright 2.0, style 0, animation 0.0). Only available in headless
    /// acceptance mode; rejected in ordinary windowed launches.
    pub acceptance_camera: Option<String>,
    /// Phase B: Generate an EnhancedV3 dungeon map from scratch.
    pub m3_generate: bool,
    /// Phase B: Path to ericw-tools bin directory.
    pub ericw_tools_dir: Option<PathBuf>,
    /// EnhancedV3 generation seed. Defaults to the current system time.
    pub m3_seed: u64,
    /// EnhancedV3 density preset.
    pub m3_preset: V3Preset,
    /// Optional exact room-count override.
    pub m3_rooms: Option<u32>,
    /// Optional exact corridor-count override.
    pub m3_corridors: Option<u32>,
    /// Optional exact loop-count override.
    pub m3_loops: Option<u32>,
    /// Whether seeded room footprints may be chamfered.
    pub m3_chamfer: bool,
    /// Portal surround type.
    pub m3_arch_type: ArchType,
    /// Empty means all six grammar families are eligible.
    pub m3_grammar_families: Vec<String>,
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
    /// Both --strict and --development specified.
    ConflictingImportMode,
    /// No import mode selected; --strict or --development required.
    NoImportMode,
    /// --m3-generate conflicts with --bsp.
    M3GenerateBspConflict,
    /// --m3-generate conflicts with explicit --palette.
    M3GeneratePaletteConflict,
    /// --m3-generate conflicts with explicit --lit.
    M3GenerateLitConflict,
    /// --m3-generate conflicts with explicit --wad.
    M3GenerateWadConflict,
    /// --m3-generate owns the generated texture closure.
    M3GenerateTexturesConflict,
    /// Generation-only option used without --m3-generate.
    M3OptionRequiresGenerate(&'static str),
    /// Invalid EnhancedV3 generation option.
    InvalidM3Value {
        flag: &'static str,
        value: String,
        expected: &'static str,
    },
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
            CliError::PaletteNotFound(path) => write!(f, "palette path is required: {path}"),
            CliError::ConflictingImportMode => {
                write!(f, "--strict and --development are mutually exclusive")
            }
            CliError::NoImportMode => {
                write!(f, "no import mode selected; use --strict or --development")
            }
            CliError::M3GenerateBspConflict => {
                write!(f, "--m3-generate and --bsp are mutually exclusive")
            }
            CliError::M3GeneratePaletteConflict => {
                write!(f, "--m3-generate and --palette are mutually exclusive")
            }
            CliError::M3GenerateLitConflict => {
                write!(f, "--m3-generate and --lit are mutually exclusive")
            }
            CliError::M3GenerateWadConflict => {
                write!(f, "--m3-generate and --wad are mutually exclusive")
            }
            CliError::M3GenerateTexturesConflict => {
                write!(f, "--m3-generate and --textures are mutually exclusive")
            }
            CliError::M3OptionRequiresGenerate(flag) => {
                write!(f, "{flag} is only valid with --m3-generate")
            }
            CliError::InvalidM3Value {
                flag,
                value,
                expected,
            } => write!(f, "{flag} expects {expected}, got '{value}'"),
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

    /// Return the declared palette path. Resource existence and authorization
    /// belong exclusively to the runtime package boundary.
    pub fn resolve_palette_path(&self) -> Result<PathBuf, CliError> {
        self.palette_path
            .clone()
            .ok_or_else(|| CliError::PaletteNotFound("provide --palette <path>".to_string()))
    }

    /// Return the declared optional `.lit` path without probing it.
    pub fn resolve_lit_path(&self) -> Option<PathBuf> {
        self.lit_path.clone()
    }

    /// Return the declared optional WAD path without probing it.
    pub fn resolve_wad_path(&self) -> Result<Option<PathBuf>, CliError> {
        Ok(self.wad_path.clone())
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
            stats: false,
            all_visible: false,
            corpus_identity: None,
            acceptance_camera: None,
            m3_generate: false,
            ericw_tools_dir: None,
            m3_seed: system_time_seed(),
            m3_preset: V3Preset::Moderate,
            m3_rooms: None,
            m3_corridors: None,
            m3_loops: None,
            m3_chamfer: true,
            m3_arch_type: ArchType::Pointed,
            m3_grammar_families: Vec::new(),
        }
    }
}

/// Parse BSP beta CLI arguments. Only space-separated flag values are supported.
pub fn parse_from(args: impl IntoIterator<Item = impl Into<String>>) -> Result<CliArgs, CliError> {
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut opts = CliArgs::default();
    let mut m3_option_used = None;
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
            "--stats" => {
                opts.stats = true;
                i += 1;
            }
            "--all-visible" => {
                opts.all_visible = true;
                i += 1;
            }
            "--corpus" => {
                let value = next_value(&args, i, "--corpus")?;
                opts.corpus_identity = Some(value.to_string());
                i += 2;
            }
            "--acceptance-camera" => {
                let value = next_value(&args, i, "--acceptance-camera")?;
                let label = value.to_string();
                if !matches!(label.as_str(), "spawn" | "corridor" | "junction") {
                    return Err(CliError::UnknownArgument(format!(
                        "--acceptance-camera must be spawn, corridor, or junction, got '{label}'"
                    )));
                }
                opts.acceptance_camera = Some(label);
                i += 2;
            }
            "--m3-generate" => {
                opts.m3_generate = true;
                i += 1;
            }
            "--seed" => {
                opts.m3_seed = parse_u64_value(&args, i, "--seed")?;
                m3_option_used = Some("--seed");
                i += 2;
            }
            "--preset" => {
                let value = next_value(&args, i, "--preset")?;
                opts.m3_preset = V3Preset::from_tag(value).ok_or_else(|| {
                    invalid_m3_value("--preset", value, "sparse, moderate, or rich")
                })?;
                m3_option_used = Some("--preset");
                i += 2;
            }
            "--rooms" => {
                opts.m3_rooms = Some(parse_u32_value(&args, i, "--rooms")?);
                m3_option_used = Some("--rooms");
                i += 2;
            }
            "--corridors" => {
                opts.m3_corridors = Some(parse_u32_value(&args, i, "--corridors")?);
                m3_option_used = Some("--corridors");
                i += 2;
            }
            "--loops" => {
                opts.m3_loops = Some(parse_u32_value(&args, i, "--loops")?);
                m3_option_used = Some("--loops");
                i += 2;
            }
            "--chamfer" => {
                opts.m3_chamfer = true;
                m3_option_used = Some("--chamfer");
                i += 1;
            }
            "--no-chamfer" => {
                opts.m3_chamfer = false;
                m3_option_used = Some("--no-chamfer");
                i += 1;
            }
            "--arch-type" => {
                let value = next_value(&args, i, "--arch-type")?;
                opts.m3_arch_type = ArchType::from_tag(value).ok_or_else(|| {
                    invalid_m3_value("--arch-type", value, "none, pointed, or segmented")
                })?;
                m3_option_used = Some("--arch-type");
                i += 2;
            }
            "--grammar-families" => {
                let value = next_value(&args, i, "--grammar-families")?;
                opts.m3_grammar_families = parse_grammar_families(value)?;
                m3_option_used = Some("--grammar-families");
                i += 2;
            }
            "--ericw-tools" => {
                let value = next_value(&args, i, "--ericw-tools")?;
                opts.ericw_tools_dir = Some(PathBuf::from(value));
                i += 2;
            }
            other => return Err(CliError::UnknownArgument(other.to_string())),
        }
    }

    // ── m3-generate conflict checks ──────────────────────────────
    if !opts.m3_generate {
        if let Some(flag) = m3_option_used {
            return Err(CliError::M3OptionRequiresGenerate(flag));
        }
    } else {
        if opts.bsp_path.is_some() {
            return Err(CliError::M3GenerateBspConflict);
        }
        if opts.palette_path.is_some() {
            return Err(CliError::M3GeneratePaletteConflict);
        }
        if opts.lit_path.is_some() {
            return Err(CliError::M3GenerateLitConflict);
        }
        if opts.wad_path.is_some() {
            return Err(CliError::M3GenerateWadConflict);
        }
        if opts.textures_dir.is_some() {
            return Err(CliError::M3GenerateTexturesConflict);
        }
        // Generated mode owns a complete closure and is always authorized
        // strictly at runtime. Preserve an explicit mode for script-level CLI
        // compatibility, otherwise expose strict as the effective default.
        if opts.import_mode.is_none() {
            opts.import_mode = Some(ImportMode::Strict);
        }
    }

    if opts.bsp_path.is_some() && opts.import_mode.is_none() {
        return Err(CliError::NoImportMode);
    }

    Ok(opts)
}

fn system_time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

fn invalid_m3_value(flag: &'static str, value: &str, expected: &'static str) -> CliError {
    CliError::InvalidM3Value {
        flag,
        value: value.to_string(),
        expected,
    }
}

fn parse_u64_value(args: &[String], i: usize, flag: &'static str) -> Result<u64, CliError> {
    let value = next_value(args, i, flag)?;
    value
        .parse()
        .map_err(|_| invalid_m3_value(flag, value, "an unsigned 64-bit integer"))
}

fn parse_u32_value(args: &[String], i: usize, flag: &'static str) -> Result<u32, CliError> {
    let value = next_value(args, i, flag)?;
    value
        .parse()
        .map_err(|_| invalid_m3_value(flag, value, "an unsigned 32-bit integer"))
}

fn parse_grammar_families(value: &str) -> Result<Vec<String>, CliError> {
    if value == "all" {
        return Ok(Vec::new());
    }
    let mut families = Vec::new();
    for family in value.split(',') {
        if family.is_empty() || !GRAMMAR_FAMILIES.contains(&family) {
            return Err(invalid_m3_value(
                "--grammar-families",
                value,
                "all or a comma-separated list of the six grammar family tags",
            ));
        }
        if families.iter().any(|existing| existing == family) {
            return Err(invalid_m3_value(
                "--grammar-families",
                value,
                "a duplicate-free comma-separated grammar family list",
            ));
        }
        families.push(family.to_string());
    }
    Ok(families)
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
    eprintln!("Usage:");
    eprintln!("  Direct BSP launch:   bsp_beta --strict|--development --bsp <path> [OPTIONS]");
    eprintln!("  Generate & launch:    bsp_beta --m3-generate [--strict|--development] [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --strict               Strict import mode");
    eprintln!("  --development          Development import mode");
    eprintln!("  --bsp <path>           Path to compiled .bsp file");
    eprintln!("  --m3-generate          Generate EnhancedV3 dungeon from scratch");
    eprintln!("  --seed <u64>           Generation seed (default: current system time)");
    eprintln!("  --preset <name>        sparse|moderate|rich (default: moderate)");
    eprintln!("  --rooms <n>            Exact room-count override");
    eprintln!("  --corridors <n>        Exact corridor-count override");
    eprintln!("  --loops <n>            Exact same-layer loop-count override");
    eprintln!("  --chamfer              Enable chamfered rooms (default)");
    eprintln!("  --no-chamfer           Disable chamfered rooms");
    eprintln!("  --arch-type <name>     none|pointed|segmented (default: pointed)");
    eprintln!("  --grammar-families <csv|all>  Grammar allowlist (default: all six)");
    eprintln!("  --ericw-tools <dir>    Path to ericw-tools bin directory");
    eprintln!("  --scale <float>        Quake→engine scale factor (default: 0.0254)");
    eprintln!("  --headless             Run headless (no window, renders N frames)");
    eprintln!("  --mcp                  Run headless MCP JSON-RPC server over stdio");
    eprintln!("  --capture-frames <n>   Frame count for headless capture (default: 0)");
    eprintln!("  --lights               Log all imported light descriptors at startup");
    eprintln!("  --palette <path>       Path to 768-byte palette .lmp file");
    eprintln!("  --lit <path>           Path to .lit colored-light companion file");
    eprintln!("  --wad <path>           Path to WAD file for texture resolution");
    eprintln!("  --textures <dir>       Textures directory for PBR companion discovery");
    eprintln!("  --stats                Print draw evidence report after mount (headless)");
    eprintln!("  --all-visible          Use all-visible evidence mode (with --stats)");
    eprintln!("  --corpus <name>        Corpus identity for evidence report");
    eprintln!("  --acceptance-camera <label>  Phase 09 frozen camera: spawn|corridor|junction");
    eprintln!();
    eprintln!("Hotkeys (live windowed m3-generate mode):");
    eprintln!("  F5                     Increment seed & regenerate");
    eprintln!("  F6                     Cycle Sparse→Moderate→Rich→Sparse");
    eprintln!("  F7                     Toggle chamfer & regenerate");
    eprintln!("  F8                     Cycle Pointed→Segmented→None→Pointed");
    eprintln!("  F9                     Toggle stairs & regenerate");
    eprintln!("  Ctrl+R                 Regenerate with unchanged config");
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
        let args = parse_from(["--development", "--headless", "--capture-frames", "10"]).unwrap();
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
    fn bsp_launch_without_import_mode_is_a_cli_error() {
        assert_eq!(
            parse_from(["--bsp", "maps/test.bsp"]).unwrap_err(),
            CliError::NoImportMode
        );
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
        let args = parse_from([
            "--strict",
            "--palette",
            "gfx/pal.lmp",
            "--lit",
            "maps/test.lit",
        ])
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
    fn palette_resolution_preserves_declared_path_without_probing() {
        let path = PathBuf::from("/tmp/not-yet-authorized-palette.lmp");
        let args = CliArgs {
            palette_path: Some(path.clone()),
            ..CliArgs::default()
        };
        assert_eq!(args.resolve_palette_path().unwrap(), path);
    }

    #[test]
    fn no_mode_reports_error() {
        let args = CliArgs::default();
        assert_eq!(
            args.require_import_mode().unwrap_err(),
            CliError::NoImportMode
        );
    }

    // ── m3-generate tests ─────────────────────────────────────────────

    #[test]
    fn m3_generate_alone_is_valid() {
        let args = parse_from(["--m3-generate"]).unwrap();
        assert!(args.m3_generate);
        assert!(args.bsp_path.is_none());
        assert_eq!(args.import_mode, Some(ImportMode::Strict));
        assert_eq!(args.m3_preset, V3Preset::Moderate);
        assert!(args.m3_chamfer);
        assert_eq!(args.m3_arch_type, ArchType::Pointed);
        assert!(args.m3_grammar_families.is_empty());
    }

    #[test]
    fn m3_generate_accepts_development_mode_for_launcher_compatibility() {
        let args = parse_from(["--development", "--m3-generate"]).unwrap();
        assert_eq!(args.import_mode, Some(ImportMode::Development));
    }

    #[test]
    fn m3_generation_options_parse() {
        let args = parse_from([
            "--m3-generate",
            "--seed",
            "42",
            "--preset",
            "rich",
            "--rooms",
            "28",
            "--corridors",
            "30",
            "--loops",
            "4",
            "--no-chamfer",
            "--arch-type",
            "segmented",
            "--grammar-families",
            "portal-chamber,column-grove,terraced-shrine",
        ])
        .unwrap();
        assert_eq!(args.m3_seed, 42);
        assert_eq!(args.m3_preset, V3Preset::Rich);
        assert_eq!(args.m3_rooms, Some(28));
        assert_eq!(args.m3_corridors, Some(30));
        assert_eq!(args.m3_loops, Some(4));
        assert!(!args.m3_chamfer);
        assert_eq!(args.m3_arch_type, ArchType::Segmented);
        assert_eq!(
            args.m3_grammar_families,
            ["portal-chamber", "column-grove", "terraced-shrine"]
        );
    }

    #[test]
    fn m3_generation_options_require_generate_mode() {
        assert_eq!(
            parse_from(["--seed", "42"]).unwrap_err(),
            CliError::M3OptionRequiresGenerate("--seed")
        );
    }

    #[test]
    fn malformed_m3_generation_options_are_rejected() {
        assert!(matches!(
            parse_from(["--m3-generate", "--seed", "-1"]).unwrap_err(),
            CliError::InvalidM3Value { flag: "--seed", .. }
        ));
        assert!(matches!(
            parse_from(["--m3-generate", "--preset", "dense"]).unwrap_err(),
            CliError::InvalidM3Value {
                flag: "--preset",
                ..
            }
        ));
        assert!(matches!(
            parse_from([
                "--m3-generate",
                "--grammar-families",
                "portal-chamber,unknown"
            ])
            .unwrap_err(),
            CliError::InvalidM3Value {
                flag: "--grammar-families",
                ..
            }
        ));
    }

    #[test]
    fn m3_generate_conflicts_with_bsp() {
        let err = parse_from(["--m3-generate", "--bsp", "maps/test.bsp"]).unwrap_err();
        assert_eq!(err, CliError::M3GenerateBspConflict);
    }

    #[test]
    fn m3_generate_conflicts_with_palette() {
        let err = parse_from(["--m3-generate", "--palette", "gfx/pal.lmp"]).unwrap_err();
        assert_eq!(err, CliError::M3GeneratePaletteConflict);
    }

    #[test]
    fn m3_generate_conflicts_with_lit() {
        let err = parse_from(["--m3-generate", "--lit", "maps/test.lit"]).unwrap_err();
        assert_eq!(err, CliError::M3GenerateLitConflict);
    }

    #[test]
    fn m3_generate_conflicts_with_wad() {
        let err = parse_from(["--m3-generate", "--wad", "maps/dungeon.wad"]).unwrap_err();
        assert_eq!(err, CliError::M3GenerateWadConflict);
    }

    #[test]
    fn m3_generate_conflicts_with_textures() {
        let err = parse_from(["--m3-generate", "--textures", "textures"]).unwrap_err();
        assert_eq!(err, CliError::M3GenerateTexturesConflict);
    }

    #[test]
    fn m3_generate_accepts_ericw_tools_dir() {
        let args = parse_from(["--m3-generate", "--ericw-tools", "/opt/ericw/bin"]).unwrap();
        assert!(args.m3_generate);
        assert_eq!(args.ericw_tools_dir, Some(PathBuf::from("/opt/ericw/bin")));
    }

    #[test]
    fn m3_generate_accepts_strict_flag() {
        let args = parse_from(["--m3-generate", "--strict"]).unwrap();
        assert!(args.m3_generate);
        assert_eq!(args.import_mode, Some(ImportMode::Strict));
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
