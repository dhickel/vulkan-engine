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
    /// Phase 17: explicit acceptance camera origin (x y z) for headless
    /// capture at arbitrary poses (grand-volume / prop evidence).
    pub acceptance_camera_origin: Option<(f32, f32, f32)>,
    /// Phase 17: explicit acceptance camera look-at target (x y z).
    pub acceptance_camera_look_at: Option<(f32, f32, f32)>,
    /// Phase 17: windowed WSI lifecycle test (resize -> minimize -> restore)
    /// exercised from inside the app when scriptable WM control is absent.
    pub wsi_lifecycle_test: bool,
    /// Phase B: Generate an EnhancedV3 dungeon map from scratch.
    pub m3_generate: bool,
    /// Phase C: Launch the Richness V1 explorer GUI.
    pub m3_richness: bool,
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
    /// --m3-generate and --m3-richness-v1 are mutually exclusive.
    M3GenerateRichnessConflict,
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
    /// Richness-only option used without --m3-richness-v1.
    RichnessOptionRequiresRichness(&'static str),
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
            CliError::M3GenerateRichnessConflict => {
                write!(
                    f,
                    "--m3-generate and --m3-richness-v1 are mutually exclusive"
                )
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
            CliError::RichnessOptionRequiresRichness(flag) => {
                write!(f, "{flag} is only valid with --m3-richness-v1")
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
            acceptance_camera_origin: None,
            acceptance_camera_look_at: None,
            wsi_lifecycle_test: false,
            m3_generate: false,
            m3_richness: false,
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
    let mut richness_option_used = None;
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
            "--acceptance-camera-origin" => {
                let triple = parse_f32_triple(&args, i, "--acceptance-camera-origin")?;
                opts.acceptance_camera_origin = Some(triple);
                i += 4;
            }
            "--acceptance-camera-look-at" => {
                let triple = parse_f32_triple(&args, i, "--acceptance-camera-look-at")?;
                opts.acceptance_camera_look_at = Some(triple);
                i += 4;
            }
            "--wsi-lifecycle-test" => {
                opts.wsi_lifecycle_test = true;
                i += 1;
            }
            "--m3-generate" => {
                opts.m3_generate = true;
                i += 1;
            }
            "--m3-richness-v1" => {
                opts.m3_richness = true;
                i += 1;
            }
            // Richness overrides are parsed into the launch token after the
            // primary mode gate. Validate their arity here so malformed
            // options cannot be silently skipped.
            flag if richness_value_flag(flag).is_some() => {
                let flag = richness_value_flag(flag).expect("matched Richness value flag");
                let _ = next_value(&args, i, flag)?;
                richness_option_used = Some(flag);
                i += 2;
            }
            flag if richness_inherited_flag(flag).is_some() => {
                richness_option_used = richness_inherited_flag(flag);
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

    // ── m3-generate / m3-richness-v1 conflict checks ────────────
    if opts.m3_generate && opts.m3_richness {
        return Err(CliError::M3GenerateRichnessConflict);
    }
    if opts.m3_richness {
        if let Some(flag) = m3_option_used {
            return Err(CliError::M3OptionRequiresGenerate(flag));
        }
        // A Richness launch without --bsp owns and builds its startup closure.
        // With --bsp, the complete prebuilt closure is authorized by the
        // Richness runtime path; this is how the cache-backed script avoids a
        // redundant startup build while still launching the Richness GUI.
        if opts.bsp_path.is_none() {
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
        }
        if opts.import_mode.is_none() {
            opts.import_mode = Some(ImportMode::Strict);
        }
    } else if !opts.m3_generate {
        if let Some(flag) = m3_option_used {
            return Err(CliError::M3OptionRequiresGenerate(flag));
        }
        if let Some(flag) = richness_option_used {
            return Err(CliError::RichnessOptionRequiresRichness(flag));
        }
    } else {
        if let Some(flag) = richness_option_used {
            return Err(CliError::RichnessOptionRequiresRichness(flag));
        }
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

fn richness_value_flag(flag: &str) -> Option<&'static str> {
    match flag {
        "--richness-preset" => Some("--richness-preset"),
        "--richness-theme" => Some("--richness-theme"),
        "--richness-extent" => Some("--richness-extent"),
        "--richness-seed" => Some("--richness-seed"),
        "--richness-pacing" => Some("--richness-pacing"),
        "--richness-landmarks" => Some("--richness-landmarks"),
        "--richness-zones" => Some("--richness-zones"),
        "--richness-cave-mode" => Some("--richness-cave-mode"),
        "--richness-vertical-openings" => Some("--richness-vertical-openings"),
        "--richness-budget-ceiling" => Some("--richness-budget-ceiling"),
        "--richness-prop-density" => Some("--richness-prop-density"),
        "--richness-light-density" => Some("--richness-light-density"),
        "--richness-variation" => Some("--richness-variation"),
        _ => None,
    }
}

fn richness_inherited_flag(flag: &str) -> Option<&'static str> {
    match flag {
        "--richness-landmarks-inherited" => Some("--richness-landmarks-inherited"),
        "--richness-zones-inherited" => Some("--richness-zones-inherited"),
        "--richness-cave-mode-inherited" => Some("--richness-cave-mode-inherited"),
        "--richness-vertical-openings-inherited" => Some("--richness-vertical-openings-inherited"),
        "--richness-budget-ceiling-inherited" => Some("--richness-budget-ceiling-inherited"),
        "--richness-pacing-inherited" => Some("--richness-pacing-inherited"),
        "--richness-variation-inherited" => Some("--richness-variation-inherited"),
        "--richness-prop-density-inherited" => Some("--richness-prop-density-inherited"),
        "--richness-light-density-inherited" => Some("--richness-light-density-inherited"),
        _ => None,
    }
}

fn parse_f32_value(args: &[String], i: usize, flag: &'static str) -> Result<f32, CliError> {
    let value = next_value(args, i, flag)?;
    value
        .parse()
        .map_err(|_| invalid_m3_value(flag, value, "a 32-bit float"))
}

fn parse_f32_triple(
    args: &[String],
    i: usize,
    flag: &'static str,
) -> Result<(f32, f32, f32), CliError> {
    let mut out = [0.0_f32; 3];
    for slot in 0..3 {
        out[slot] = parse_f32_value(args, i + slot, flag)?;
    }
    if !out.iter().all(|v| v.is_finite()) {
        return Err(CliError::InvalidScale(format!(
            "{flag} requires three finite numbers, got {:?}",
            out
        )));
    }
    Ok((out[0], out[1], out[2]))
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

/// Full usage text (testable; the printer routes it to stderr).
pub(crate) fn usage_text() -> String {
    let mut out = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "BSP Beta — Maintained Load-Query-Physics-Behavior-Reload Proof"
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "Usage:");
    let _ = writeln!(
        out,
        "  Direct BSP launch:   bsp_beta --strict|--development --bsp <path> [OPTIONS]"
    );
    let _ = writeln!(
        out,
        "  Generate & launch:    bsp_beta --m3-generate [--strict|--development] [OPTIONS]
  Richness explorer:    bsp_beta --m3-richness-v1 [--strict|--development] [OPTIONS]"
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "Options:");
    let _ = writeln!(out, "  --strict               Strict import mode");
    let _ = writeln!(out, "  --development          Development import mode");
    let _ = writeln!(out, "  --bsp <path>           Path to compiled .bsp file");
    let _ = writeln!(
        out,
        "  --m3-generate          Generate EnhancedV3 dungeon from scratch"
    );
    let _ = writeln!(out, "  --m3-richness-v1       Launch Richness V1 explorer");
    let _ = writeln!(out, "  --richness-preset <name>  sparse|moderate|rich");
    let _ = writeln!(
        out,
        "  --richness-theme <name>   ancient|egyptian|brutalist"
    );
    let _ = writeln!(out, "  --richness-seed <u64>     Richness generation seed");
    let _ = writeln!(out, "  --richness-extent <u32>   Richness XY extent");
    let _ = writeln!(out, "  --richness-pacing <name>  relaxed|normal|intense");
    let _ = writeln!(out, "  --richness-variation <name>  subtle|moderate|wild");
    let _ = writeln!(out, "  --richness-landmarks <u32>  Critical-path landmarks");
    let _ = writeln!(out, "  --richness-zones <u32>      Semantic zone count");
    let _ = writeln!(
        out,
        "  --richness-vertical-openings <u32>  Vertical opening count"
    );
    let _ = writeln!(
        out,
        "  --richness-cave-mode <name>  disabled|allowed|required"
    );
    let _ = writeln!(out, "  --richness-prop-density <u32>   UI prop density");
    let _ = writeln!(out, "  --richness-light-density <u32>  UI light density");
    let _ = writeln!(out, "  --richness-budget-ceiling <u32> Complexity ceiling");
    let _ = writeln!(
        out,
        "  --richness-landmarks-inherited|--richness-zones-inherited"
    );
    let _ = writeln!(
        out,
        "  --richness-cave-mode-inherited|--richness-vertical-openings-inherited"
    );
    let _ = writeln!(
        out,
        "  --richness-pacing-inherited|--richness-variation-inherited"
    );
    let _ = writeln!(
        out,
        "  --richness-prop-density-inherited|--richness-light-density-inherited"
    );
    let _ = writeln!(out, "  --richness-budget-ceiling-inherited");
    let _ = writeln!(
        out,
        "  --seed <u64>           Generation seed (default: current system time)"
    );
    let _ = writeln!(
        out,
        "  --preset <name>        sparse|moderate|rich (default: moderate)"
    );
    let _ = writeln!(out, "  --rooms <n>            Exact room-count override");
    let _ = writeln!(
        out,
        "  --corridors <n>        Exact corridor-count override"
    );
    let _ = writeln!(
        out,
        "  --loops <n>            Exact same-layer loop-count override"
    );
    let _ = writeln!(
        out,
        "  --chamfer              Enable chamfered rooms (default)"
    );
    let _ = writeln!(out, "  --no-chamfer           Disable chamfered rooms");
    let _ = writeln!(
        out,
        "  --arch-type <name>     none|pointed|segmented (default: pointed)"
    );
    let _ = writeln!(
        out,
        "  --grammar-families <csv|all>  Grammar allowlist (default: all six)"
    );
    let _ = writeln!(
        out,
        "  --ericw-tools <dir>    Path to ericw-tools bin directory"
    );
    let _ = writeln!(
        out,
        "  --scale <float>        Quake→engine scale factor (default: 0.0254)"
    );
    let _ = writeln!(
        out,
        "  --headless             Run headless (no window, renders N frames)"
    );
    let _ = writeln!(
        out,
        "  --mcp                  Run headless MCP JSON-RPC server over stdio"
    );
    let _ = writeln!(
        out,
        "  --capture-frames <n>   Frame count for headless capture (default: 0)"
    );
    let _ = writeln!(
        out,
        "  --lights               Log all imported light descriptors at startup"
    );
    let _ = writeln!(
        out,
        "  --palette <path>       Path to 768-byte palette .lmp file"
    );
    let _ = writeln!(
        out,
        "  --lit <path>           Path to .lit colored-light companion file"
    );
    let _ = writeln!(
        out,
        "  --wad <path>           Path to WAD file for texture resolution"
    );
    let _ = writeln!(
        out,
        "  --textures <dir>       Textures directory for PBR companion discovery"
    );
    let _ = writeln!(
        out,
        "  --stats                Print draw evidence report after mount (headless)"
    );
    let _ = writeln!(
        out,
        "  --all-visible          Use all-visible evidence mode (with --stats)"
    );
    let _ = writeln!(
        out,
        "  --corpus <name>        Corpus identity for evidence report"
    );
    let _ = writeln!(
        out,
        "  --acceptance-camera <label>  Phase 09 frozen camera: spawn|corridor|junction"
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "Hotkeys (live windowed m3-generate mode):");
    let _ = writeln!(out, "  F1                     Toggle keyboard GUI");
    let _ = writeln!(out, "  F2                     Toggle mouse GUI");
    let _ = writeln!(out);
    let _ = writeln!(out, "Hotkeys (live windowed m3-richness-v1 mode):");
    let _ = writeln!(out, "  F3                     Toggle keyboard GUI");
    let _ = writeln!(out, "  F4                     Toggle mouse GUI");
    let _ = writeln!(out, "  F5                     Increment seed & regenerate");
    let _ = writeln!(
        out,
        "  F6                     Cycle Sparse→Moderate→Rich→Sparse"
    );
    let _ = writeln!(out, "  F7                     Toggle chamfer & regenerate");
    let _ = writeln!(
        out,
        "  F8                     Cycle Pointed→Segmented→None→Pointed"
    );
    let _ = writeln!(out, "  F9                     Toggle stairs & regenerate");
    let _ = writeln!(
        out,
        "  Ctrl+R                 Regenerate with unchanged config"
    );
    let _ = writeln!(out);
    out
}

fn print_usage() {
    eprint!("{}", usage_text());
}
// ── Richness V1 CLI override parser ───────────────────────────────────────
//
// The primary parser enforces the mode gate and option arity. This parser
// converts the Richness-prefixed controls into the explorer launch token.

/// Parsed Richness V1 launch token with all draft controls.
///
/// Every InheritedOr field uses `Option<T>` with `None` meaning inherited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RichnessLaunchToken {
    pub richness_preset: Option<String>,
    pub richness_theme: Option<String>,
    pub richness_extent: Option<u32>,
    pub richness_seed: Option<u64>,
    pub richness_pacing: Option<String>,
    pub richness_landmarks: Option<u32>,
    pub richness_zones: Option<u32>,
    pub richness_cave_mode: Option<String>,
    pub richness_vertical_openings: Option<u32>,
    pub richness_variation: Option<String>,
    pub richness_prop_density: Option<u32>,
    pub richness_light_density: Option<u32>,
    pub richness_budget_ceiling: Option<u32>,
    /// Inherited-marker fields: `--richness-<name>-inherited` sets the
    /// corresponding control to inherited (clearing any explicit value).
    pub richness_landmarks_inherited: bool,
    pub richness_zones_inherited: bool,
    pub richness_cave_mode_inherited: bool,
    pub richness_vertical_openings_inherited: bool,
    pub richness_budget_ceiling_inherited: bool,
    pub richness_pacing_inherited: bool,
    pub richness_variation_inherited: bool,
    pub richness_prop_density_inherited: bool,
    pub richness_light_density_inherited: bool,
}

impl Default for RichnessLaunchToken {
    fn default() -> Self {
        Self::new()
    }
}

impl RichnessLaunchToken {
    /// Create an empty launch token (all fields `None`, all inherited flags `false`).
    pub fn new() -> Self {
        Self {
            richness_preset: None,
            richness_theme: None,
            richness_extent: None,
            richness_seed: None,
            richness_pacing: None,
            richness_landmarks: None,
            richness_zones: None,
            richness_cave_mode: None,
            richness_vertical_openings: None,
            richness_variation: None,
            richness_prop_density: None,
            richness_light_density: None,
            richness_budget_ceiling: None,
            richness_landmarks_inherited: false,
            richness_zones_inherited: false,
            richness_cave_mode_inherited: false,
            richness_vertical_openings_inherited: false,
            richness_budget_ceiling_inherited: false,
            richness_pacing_inherited: false,
            richness_variation_inherited: false,
            richness_prop_density_inherited: false,
            richness_light_density_inherited: false,
        }
    }

    /// Returns `true` if every field is `None` and every inherited flag is `false`.
    pub fn is_empty(&self) -> bool {
        self == &Self::new()
    }
}

/// Parse a Richness V1 launch token from CLI arguments.
///
/// Non-Richness arguments are ignored because production passes the original
/// process argument vector. Unknown `--richness-*` flags still fail closed.
///
/// All Richness options use the `--richness-` prefix to avoid collision
/// with existing M3 and BSP flags.
pub(crate) fn parse_richness_launch_token(
    args: impl IntoIterator<Item = impl Into<String>>,
) -> Result<RichnessLaunchToken, CliError> {
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut token = RichnessLaunchToken::new();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--richness-preset" => {
                let value = next_value(&args, i, "--richness-preset")?;
                token.richness_preset = Some(value.to_string());
                i += 2;
            }
            "--richness-theme" => {
                let value = next_value(&args, i, "--richness-theme")?;
                token.richness_theme = Some(value.to_string());
                i += 2;
            }
            "--richness-extent" => {
                let value = next_value(&args, i, "--richness-extent")?;
                token.richness_extent = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| invalid_m3_value("--richness-extent", value, "a u32"))?,
                );
                i += 2;
            }
            "--richness-seed" => {
                let value = next_value(&args, i, "--richness-seed")?;
                token.richness_seed = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_| invalid_m3_value("--richness-seed", value, "a u64"))?,
                );
                i += 2;
            }
            "--richness-pacing" => {
                let value = next_value(&args, i, "--richness-pacing")?;
                token.richness_pacing = Some(value.to_string());
                i += 2;
            }
            "--richness-landmarks" => {
                let value = next_value(&args, i, "--richness-landmarks")?;
                token.richness_landmarks = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| invalid_m3_value("--richness-landmarks", value, "a u32"))?,
                );
                i += 2;
            }
            "--richness-zones" => {
                let value = next_value(&args, i, "--richness-zones")?;
                token.richness_zones = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| invalid_m3_value("--richness-zones", value, "a u32"))?,
                );
                i += 2;
            }
            "--richness-cave-mode" => {
                let value = next_value(&args, i, "--richness-cave-mode")?;
                token.richness_cave_mode = Some(value.to_string());
                i += 2;
            }
            "--richness-vertical-openings" => {
                let value = next_value(&args, i, "--richness-vertical-openings")?;
                token.richness_vertical_openings = Some(value.parse::<u32>().map_err(|_| {
                    invalid_m3_value("--richness-vertical-openings", value, "a u32")
                })?);
                i += 2;
            }
            "--richness-variation" => {
                let value = next_value(&args, i, "--richness-variation")?;
                token.richness_variation = Some(value.to_string());
                i += 2;
            }
            "--richness-prop-density" => {
                let value = next_value(&args, i, "--richness-prop-density")?;
                token.richness_prop_density =
                    Some(value.parse::<u32>().map_err(|_| {
                        invalid_m3_value("--richness-prop-density", value, "a u32")
                    })?);
                i += 2;
            }
            "--richness-light-density" => {
                let value = next_value(&args, i, "--richness-light-density")?;
                token.richness_light_density =
                    Some(value.parse::<u32>().map_err(|_| {
                        invalid_m3_value("--richness-light-density", value, "a u32")
                    })?);
                i += 2;
            }
            "--richness-budget-ceiling" => {
                let value = next_value(&args, i, "--richness-budget-ceiling")?;
                token.richness_budget_ceiling =
                    Some(value.parse::<u32>().map_err(|_| {
                        invalid_m3_value("--richness-budget-ceiling", value, "a u32")
                    })?);
                i += 2;
            }
            // Inherited flags (no value — just the flag)
            "--richness-landmarks-inherited" => {
                token.richness_landmarks_inherited = true;
                i += 1;
            }
            "--richness-zones-inherited" => {
                token.richness_zones_inherited = true;
                i += 1;
            }
            "--richness-cave-mode-inherited" => {
                token.richness_cave_mode_inherited = true;
                i += 1;
            }
            "--richness-vertical-openings-inherited" => {
                token.richness_vertical_openings_inherited = true;
                i += 1;
            }
            "--richness-budget-ceiling-inherited" => {
                token.richness_budget_ceiling_inherited = true;
                i += 1;
            }
            "--richness-pacing-inherited" => {
                token.richness_pacing_inherited = true;
                i += 1;
            }
            "--richness-variation-inherited" => {
                token.richness_variation_inherited = true;
                i += 1;
            }
            "--richness-prop-density-inherited" => {
                token.richness_prop_density_inherited = true;
                i += 1;
            }
            "--richness-light-density-inherited" => {
                token.richness_light_density_inherited = true;
                i += 1;
            }
            other if other.starts_with("--richness-") => {
                return Err(CliError::UnknownArgument(other.to_string()));
            }
            _ => i += 1,
        }
    }

    Ok(token)
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
