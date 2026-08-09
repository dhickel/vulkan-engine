use std::path::PathBuf;

use bsp_generator::{
    ArchType, FeatureFlags, GrammarMode, InheritedOr, RichnessCaveMode, RichnessDocumentV1,
    RichnessPreset, RichnessTheme, V3Config, V3Preset,
};

#[derive(Default)]
/// Richness V1 CLI options (for --class m3-richness-v1).
struct RichnessOptions {
    preset: Option<String>,
    theme: Option<String>,
    extent: Option<u32>,
    landmarks: Option<u32>,
    zones: Option<u32>,
    cave_mode: Option<String>,
    vertical_openings: Option<u32>,
    budget: Option<u32>,
}

impl RichnessOptions {
    fn first_used_flag(&self) -> Option<&'static str> {
        [
            self.preset.as_ref().map(|_| "--preset"),
            self.theme.as_ref().map(|_| "--theme"),
            self.extent.as_ref().map(|_| "--extent"),
            self.landmarks.as_ref().map(|_| "--landmarks"),
            self.zones.as_ref().map(|_| "--zones"),
            self.cave_mode.as_ref().map(|_| "--cave-mode"),
            self.vertical_openings
                .as_ref()
                .map(|_| "--vertical-openings"),
            self.budget.as_ref().map(|_| "--budget"),
        ]
        .into_iter()
        .flatten()
        .next()
    }

    fn into_document(self, seed: u64) -> Result<RichnessDocumentV1, String> {
        let preset_tag = self.preset.as_deref().unwrap_or("sparse");
        let preset = RichnessPreset::from_tag(preset_tag).ok_or_else(|| {
            format!("unknown --preset '{preset_tag}'. Use sparse, moderate, or rich")
        })?;
        let theme_tag = self.theme.as_deref().unwrap_or("ancient");
        let theme = RichnessTheme::from_tag(theme_tag).ok_or_else(|| {
            format!("unknown --theme '{theme_tag}'. Use ancient, egyptian, or brutalist")
        })?;
        let extent = self.extent.unwrap_or(2048);

        let mut doc = RichnessDocumentV1::new(seed, extent, preset, theme)
            .map_err(|err| format!("richness config invalid: {err}"))?;

        if let Some(v) = self.landmarks {
            doc = RichnessDocumentV1::with_all_explicit(
                doc.seed(),
                doc.extent(),
                doc.preset(),
                doc.theme(),
                doc.request_schema_revision(),
                doc.algorithm_revision(),
                doc.content_revision(),
                doc.preset_revision(),
                doc.theme_revision(),
                doc.asset_revision(),
                doc.convention_revision(),
                InheritedOr::Explicit(v),
                doc.zone_count(),
                doc.cave_mode(),
                doc.vertical_openings(),
                doc.budget_ceiling(),
            )
            .map_err(|err| format!("richness config invalid: {err}"))?;
        }

        if let Some(v) = self.zones {
            let prev = doc.clone();
            doc = RichnessDocumentV1::with_all_explicit(
                prev.seed(),
                prev.extent(),
                prev.preset(),
                prev.theme(),
                prev.request_schema_revision(),
                prev.algorithm_revision(),
                prev.content_revision(),
                prev.preset_revision(),
                prev.theme_revision(),
                prev.asset_revision(),
                prev.convention_revision(),
                prev.critical_path_landmarks(),
                InheritedOr::Explicit(v),
                prev.cave_mode(),
                prev.vertical_openings(),
                prev.budget_ceiling(),
            )
            .map_err(|err| format!("richness config invalid: {err}"))?;
        }

        if let Some(ref mode_tag) = self.cave_mode {
            let mode = RichnessCaveMode::from_tag(mode_tag).ok_or_else(|| {
                format!("unknown --cave-mode '{mode_tag}'. Use required, preferred, or omitted")
            })?;
            let prev = doc.clone();
            doc = RichnessDocumentV1::with_all_explicit(
                prev.seed(),
                prev.extent(),
                prev.preset(),
                prev.theme(),
                prev.request_schema_revision(),
                prev.algorithm_revision(),
                prev.content_revision(),
                prev.preset_revision(),
                prev.theme_revision(),
                prev.asset_revision(),
                prev.convention_revision(),
                prev.critical_path_landmarks(),
                prev.zone_count(),
                InheritedOr::Explicit(mode),
                prev.vertical_openings(),
                prev.budget_ceiling(),
            )
            .map_err(|err| format!("richness config invalid: {err}"))?;
        }

        if let Some(v) = self.vertical_openings {
            let prev = doc.clone();
            doc = RichnessDocumentV1::with_all_explicit(
                prev.seed(),
                prev.extent(),
                prev.preset(),
                prev.theme(),
                prev.request_schema_revision(),
                prev.algorithm_revision(),
                prev.content_revision(),
                prev.preset_revision(),
                prev.theme_revision(),
                prev.asset_revision(),
                prev.convention_revision(),
                prev.critical_path_landmarks(),
                prev.zone_count(),
                prev.cave_mode(),
                InheritedOr::Explicit(v),
                prev.budget_ceiling(),
            )
            .map_err(|err| format!("richness config invalid: {err}"))?;
        }

        if let Some(v) = self.budget {
            let prev = doc.clone();
            doc = RichnessDocumentV1::with_all_explicit(
                prev.seed(),
                prev.extent(),
                prev.preset(),
                prev.theme(),
                prev.request_schema_revision(),
                prev.algorithm_revision(),
                prev.content_revision(),
                prev.preset_revision(),
                prev.theme_revision(),
                prev.asset_revision(),
                prev.convention_revision(),
                prev.critical_path_landmarks(),
                prev.zone_count(),
                prev.cave_mode(),
                prev.vertical_openings(),
                InheritedOr::Explicit(v),
            )
            .map_err(|err| format!("richness config invalid: {err}"))?;
        }

        doc.validate_raw_fields()
            .map_err(|err| format!("richness config invalid: {err}"))?;
        Ok(doc)
    }
}

#[derive(Default)]
struct M3Options {
    preset: Option<String>,
    extent: Option<u32>,
    rooms: Option<u32>,
    corridors: Option<u32>,
    loops: Option<u32>,
    vertical_edges: Option<u32>,
    chamfer: Option<bool>,
    arch_type: Option<String>,
    stairs: Option<bool>,
    room_span_min: Option<u32>,
    room_span_max: Option<u32>,
    grammar_families: Option<String>,
    grammar_mode: Option<String>,
    features: Option<String>,
    feature_density: Option<f32>,
    minlight: Option<u32>,
    light_count: Option<u32>,
}

impl M3Options {
    fn first_used_flag(&self) -> Option<&'static str> {
        [
            self.preset.as_ref().map(|_| "--preset"),
            self.extent.as_ref().map(|_| "--extent"),
            self.rooms.as_ref().map(|_| "--rooms"),
            self.corridors.as_ref().map(|_| "--corridors"),
            self.loops.as_ref().map(|_| "--loops"),
            self.vertical_edges.as_ref().map(|_| "--vertical-edges"),
            self.chamfer.as_ref().map(|_| "--chamfer/--no-chamfer"),
            self.arch_type.as_ref().map(|_| "--arch-type"),
            self.stairs.as_ref().map(|_| "--stairs/--no-stairs"),
            self.room_span_min.as_ref().map(|_| "--room-span-min"),
            self.room_span_max.as_ref().map(|_| "--room-span-max"),
            self.grammar_families.as_ref().map(|_| "--grammar-families"),
            self.grammar_mode.as_ref().map(|_| "--grammar-mode"),
            self.features.as_ref().map(|_| "--features"),
            self.feature_density.as_ref().map(|_| "--feature-density"),
            self.minlight.as_ref().map(|_| "--minlight"),
            self.light_count.as_ref().map(|_| "--light-count"),
        ]
        .into_iter()
        .flatten()
        .next()
    }

    fn into_config(self, seed: u64) -> Result<V3Config, String> {
        let preset_tag = self.preset.as_deref().unwrap_or("sparse");
        let preset = V3Preset::from_tag(preset_tag).ok_or_else(|| {
            format!("unknown --preset '{preset_tag}'. Use sparse, moderate, or rich")
        })?;
        let default_extent = match preset {
            V3Preset::Sparse | V3Preset::Moderate => 2048,
            V3Preset::Rich => 3072,
        };
        let mut config = V3Config::new(seed, preset, self.extent.unwrap_or(default_extent))
            .map_err(|err| format!("v3 config invalid: {err}"))?;

        config.rooms = self.rooms;
        config.corridors = self.corridors;
        config.loops = self.loops;
        config.vertical_edges = self.vertical_edges;
        if let Some(chamfer) = self.chamfer {
            config.chamfer = chamfer;
        }
        if let Some(tag) = self.arch_type {
            config.arch_type = ArchType::from_tag(&tag).ok_or_else(|| {
                format!("unknown --arch-type '{tag}'. Use none, pointed, or segmented")
            })?;
        }
        if let Some(stairs) = self.stairs {
            config.stairs = stairs;
        }
        config.room_span_min = self.room_span_min;
        config.room_span_max = self.room_span_max;
        if let Some(families) = self.grammar_families {
            config.grammar_families = parse_grammar_families(&families)?;
        }
        if let Some(mode) = self.grammar_mode {
            config.grammar_mode = GrammarMode::from_tag(&mode)
                .ok_or_else(|| format!("unknown --grammar-mode '{mode}'. Use single or mixed"))?;
        }
        if let Some(features) = self.features {
            config.features = parse_feature_flags(&features)?;
        }
        if let Some(density) = self.feature_density {
            config.feature_density = density;
        }
        if let Some(minlight) = self.minlight {
            config.minlight = minlight;
        }
        config.light_count = self.light_count;
        config
            .validate()
            .map_err(|err| format!("v3 config invalid: {err}"))?;
        Ok(config)
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("dungeon_gen: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut seed: Option<u64> = None;
    let mut class: Option<String> = None;
    let mut out: Option<PathBuf> = None;
    let mut m3 = M3Options::default();
    let mut richness = RichnessOptions::default();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => seed = Some(parse_u64(&mut args, "--seed")?),
            "--class" => class = Some(next_value(&mut args, "--class")?),
            "--out" => out = Some(PathBuf::from(next_value(&mut args, "--out")?)),
            "--preset" => {
                let val = next_value(&mut args, "--preset")?;
                m3.preset = Some(val.clone());
                richness.preset = Some(val);
            }
            "--extent" => {
                let val = parse_u32(&mut args, "--extent")?;
                m3.extent = Some(val);
                richness.extent = Some(val);
            }
            "--theme" => richness.theme = Some(next_value(&mut args, "--theme")?),
            "--landmarks" => richness.landmarks = Some(parse_u32(&mut args, "--landmarks")?),
            "--zones" => richness.zones = Some(parse_u32(&mut args, "--zones")?),
            "--cave-mode" => richness.cave_mode = Some(next_value(&mut args, "--cave-mode")?),
            "--vertical-openings" => {
                richness.vertical_openings = Some(parse_u32(&mut args, "--vertical-openings")?)
            }
            "--budget" => richness.budget = Some(parse_u32(&mut args, "--budget")?),
            "--rooms" => m3.rooms = Some(parse_u32(&mut args, "--rooms")?),
            "--corridors" => m3.corridors = Some(parse_u32(&mut args, "--corridors")?),
            "--loops" => m3.loops = Some(parse_u32(&mut args, "--loops")?),
            "--vertical-edges" => {
                m3.vertical_edges = Some(parse_u32(&mut args, "--vertical-edges")?)
            }
            "--chamfer" => m3.chamfer = Some(true),
            "--no-chamfer" => m3.chamfer = Some(false),
            "--arch-type" => m3.arch_type = Some(next_value(&mut args, "--arch-type")?),
            "--stairs" => m3.stairs = Some(true),
            "--no-stairs" => m3.stairs = Some(false),
            "--room-span-min" => m3.room_span_min = Some(parse_u32(&mut args, "--room-span-min")?),
            "--room-span-max" => m3.room_span_max = Some(parse_u32(&mut args, "--room-span-max")?),
            "--grammar-families" => {
                m3.grammar_families = Some(next_value(&mut args, "--grammar-families")?)
            }
            "--grammar-mode" => m3.grammar_mode = Some(next_value(&mut args, "--grammar-mode")?),
            "--features" => m3.features = Some(next_value(&mut args, "--features")?),
            "--feature-density" => {
                m3.feature_density = Some(parse_f32(&mut args, "--feature-density")?)
            }
            "--minlight" => m3.minlight = Some(parse_u32(&mut args, "--minlight")?),
            "--light-count" => m3.light_count = Some(parse_u32(&mut args, "--light-count")?),
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let seed = seed.unwrap_or(0);
    match class.as_deref().unwrap_or("m1") {
        "m1" => {
            reject_m3_options(&m3, "m1")?;
            reject_richness_options_for_baseline(&richness, "m1")?;
            let config = bsp_generator::DungeonConfig::nominal_m1();
            let (map_text, meta) = bsp_generator::generate(seed, config)
                .map_err(|err| format!("generation failed: {err:?}"))?;
            write_output(
                map_text,
                meta.room_count,
                meta.corridor_count,
                meta.face_count_estimate,
                seed,
                out,
            )?;
        }
        "m2" => {
            reject_m3_options(&m3, "m2")?;
            reject_richness_options_for_baseline(&richness, "m2")?;
            let config = bsp_generator::enhanced::config::EnhancedConfig::nominal();
            let (map_text, meta) = bsp_generator::generate_enhanced(seed, config)
                .map_err(|err| format!("enhanced generation failed: {err}"))?;
            write_output(
                map_text,
                meta.room_count,
                meta.route_count,
                (meta.room_count + meta.route_count + meta.transition_count) * 6,
                seed,
                out,
            )?;
        }
        "m3" => {
            reject_richness_options_for_baseline(&richness, "m3")?;
            let config = m3.into_config(seed)?;
            let (map_text, meta) = bsp_generator::generate_enhanced_v3(&config)
                .map_err(|err| format!("v3 generation failed: {err}"))?;
            write_output(
                map_text,
                meta.room_count(),
                config.effective_corridors(),
                meta.actual_faces(),
                seed,
                out,
            )?;
        }
        "m3-richness-v1" => {
            let doc = richness.into_document(seed)?;
            let output = bsp_generator::generate_richness_v1(&doc)
                .map_err(|err| format!("richness generation failed: {err}"))?;
            let meta_bytes = output.generation_metadata.to_canonical_bytes();
            let meta_str =
                std::str::from_utf8(&meta_bytes).map_err(|e| format!("metadata not utf-8: {e}"))?;
            let rooms = extract_fact_u32(meta_str, "rooms").unwrap_or(0);
            let routes = extract_fact_u32(meta_str, "routes").unwrap_or(0);
            write_output(
                output.map_text,
                rooms,
                routes,
                output.actual.faces as u32,
                seed,
                out,
            )?;
        }
        other => {
            return Err(format!(
                "--class must be m1, m2, m3, or m3-richness-v1, got {other}"
            ))
        }
    }

    Ok(())
}

fn reject_m3_options(options: &M3Options, class: &str) -> Result<(), String> {
    if let Some(flag) = options.first_used_flag() {
        return Err(format!("{flag} is not valid for class {class}"));
    }
    Ok(())
}

fn reject_richness_options_for_baseline(
    options: &RichnessOptions,
    class: &str,
) -> Result<(), String> {
    if let Some(flag) = options.first_used_flag() {
        return Err(format!(
            "{flag} is a richness-only option and is not valid for class {class}"
        ));
    }
    Ok(())
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_u64(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, String> {
    let value = next_value(args, flag)?;
    value
        .parse::<u64>()
        .map_err(|_| format!("invalid {flag} value: {value}"))
}

fn parse_u32(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<u32, String> {
    let value = next_value(args, flag)?;
    value
        .parse::<u32>()
        .map_err(|_| format!("invalid {flag} value: {value}"))
}

fn parse_f32(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<f32, String> {
    let value = next_value(args, flag)?;
    value
        .parse::<f32>()
        .map_err(|_| format!("invalid {flag} value: {value}"))
}

fn parse_grammar_families(value: &str) -> Result<Vec<String>, String> {
    if value == "all" {
        return Ok(Vec::new());
    }
    let mut families = Vec::new();
    for family in value.split(',') {
        if family.is_empty() {
            return Err("--grammar-families contains an empty family".to_string());
        }
        if !bsp_generator::enhanced_v3::GRAMMAR_FAMILIES.contains(&family) {
            return Err(format!("unknown grammar family '{family}'"));
        }
        if families.iter().any(|existing| existing == family) {
            return Err(format!("duplicate grammar family '{family}'"));
        }
        families.push(family.to_string());
    }
    Ok(families)
}

fn parse_feature_flags(value: &str) -> Result<FeatureFlags, String> {
    if value == "all" {
        return Ok(FeatureFlags::ALL);
    }
    if value == "none" {
        return Ok(FeatureFlags::empty());
    }
    let mut flags = FeatureFlags::empty();
    for tag in value.split(',') {
        let flag = FeatureFlags::from_tag(tag).ok_or_else(|| format!("unknown feature '{tag}'"))?;
        if flags.contains(flag) {
            return Err(format!("duplicate feature '{tag}'"));
        }
        flags |= flag;
    }
    Ok(flags)
}

fn extract_fact_u32(meta_str: &str, key: &str) -> Option<u32> {
    let prefix = format!("{key}: ");
    for line in meta_str.lines() {
        if let Some(rest) = line.strip_prefix(&prefix) {
            return rest.parse::<u32>().ok();
        }
    }
    None
}

fn write_output(
    map_text: String,
    rooms: u32,
    corridors: u32,
    estimated_faces: u32,
    seed: u64,
    out: Option<PathBuf>,
) -> Result<(), String> {
    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("create output directory {}: {err}", parent.display()))?;
        }
        std::fs::write(&path, map_text)
            .map_err(|err| format!("write {}: {err}", path.display()))?;
        eprintln!(
            "generated {}: seed={seed} rooms={rooms} corridors={corridors} estimated_faces={estimated_faces}",
            path.display(),
        );
    } else {
        print!("{map_text}");
    }
    Ok(())
}

fn print_usage() {
    eprintln!("Usage: dungeon_gen [--seed <u64>] [--class m1|m2|m3|m3-richness-v1] [--out <path>] [OPTIONS]");
    eprintln!("  m1: Legacy v1 single-layer dungeon");
    eprintln!("  m2: Enhanced v2 two-layer dungeon with stairs");
    eprintln!("  m3: Enhanced v3 two-layer generation explorer");
    eprintln!("  m3-richness-v1: Enhanced V3 Richness V1 with archetypes, props, and themes");
    eprintln!();
    eprintln!("M3 options:");
    eprintln!("  --preset sparse|moderate|rich       Density preset (default: sparse)");
    eprintln!("  --extent <1024..3072>              XY extent (multiple of 16)");
    eprintln!("  --rooms <3..40>                    Exact room count");
    eprintln!("  --corridors <n>                    Exact physical corridor segments");
    eprintln!("  --loops <0..6>                     Exact same-layer graph loops");
    eprintln!("  --vertical-edges <0..3>            Exact stair connections");
    eprintln!("  --chamfer | --no-chamfer           Enable/disable seeded chamfers");
    eprintln!("  --arch-type none|pointed|segmented Portal surround");
    eprintln!("  --stairs | --no-stairs             Enable/disable stairs");
    eprintln!("  --room-span-min <n>                Quantum-aligned minimum room span");
    eprintln!("  --room-span-max <n>                Quantum-aligned maximum room span");
    eprintln!("  --grammar-families <all|csv>       Filter the six grammar families");
    eprintln!("  --grammar-mode single|mixed        Per-map or per-room grammar assignment");
    eprintln!(
        "  --features <all|none|csv>          pillars,buttresses,blades,vault-ribs,monoliths"
    );
    eprintln!("  --feature-density <0.0..1.0>       Feature-bearing room density");
    eprintln!("  --minlight <0..255>                Worldspawn _minlight");
    eprintln!("  --light-count <0..rooms>           Exact baked light entity count");
    eprintln!();
    eprintln!("M3 Richness V1 options (--class m3-richness-v1 only):");
    eprintln!("  --preset sparse|moderate|rich       Richness density preset");
    eprintln!("  --theme ancient|egyptian|brutalist  Visual theme (default: ancient)");
    eprintln!("  --extent <1024|2048|3072>          XY extent (multiple of 16)");
    eprintln!("  --landmarks <1..5>                 Critical-path landmark count");
    eprintln!("  --zones <1..6>                     Semantic zone count");
    eprintln!("  --cave-mode required|preferred|omitted  Cave eligibility");
    eprintln!("  --vertical-openings <0..12>        Vertical feature count");
    eprintln!("  --budget <1000..8000>              Budget ceiling in source faces");
}
