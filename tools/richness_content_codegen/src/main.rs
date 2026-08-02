//! Deterministic code generation tool for EnhancedV3 Richness V1 authored content.
//!
//! Reads RON catalog files, validates them against the closed schema,
//! and emits checked-in Rust constants. RON is NEVER parsed at runtime;
//! the generated Rust file is the sole runtime data source.
//!
//! # Usage
//!
//! ```text
//! richness_content_codegen <catalog_dir> <output_file>
//! ```
//!
//! Where `<catalog_dir>` contains:
//! - `archetypes.ron`
//! - `props.ron`
//! - `lighting.ron`
//! - `themes.ron`
//! - `spiral_steps.ron`
//!
//! Validation errors produce a non-zero exit code with structured
//! messages to stderr.

mod schema;

use schema::{
    Archetype, ArchetypesFile, LightRecipe, LightingFile, Prop, PropsFile, RichnessCatalog,
    SpiralFile, Theme, ThemesFile, SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <catalog_dir> <output_file>", args[0]);
        eprintln!();
        eprintln!("  catalog_dir  — directory containing archetypes.ron, props.ron,");
        eprintln!("                 lighting.ron, themes.ron, spiral_steps.ron");
        eprintln!("  output_file  — path for generated Rust constants file");
        process::exit(1);
    }

    let catalog_dir = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);

    match run(catalog_dir, output_path) {
        Ok(()) => process::exit(0),
        Err(e) => {
            eprintln!("ERROR: {e}");
            process::exit(1);
        }
    }
}

fn run(catalog_dir: &Path, output_path: &Path) -> Result<(), String> {
    // 1. Read all RON files
    let catalog = load_catalog(catalog_dir)?;

    // 2. Validate the complete catalog
    catalog.validate().map_err(|errs| {
        format!(
            "Catalog validation failed with {} error(s):\n{errs}",
            errs.errors.len()
        )
    })?;

    // 3. Compute source hash over the canonical RON bytes
    let source_hash = compute_source_hash(catalog_dir)?;

    // 4. Generate Rust code
    let rust_code = generate_rust(&catalog, &source_hash);

    // 5. Write to temporary file, fsync, compare, atomically replace
    write_atomic(output_path, &rust_code)?;

    let archetype_count = catalog.archetypes.len();
    let prop_count = catalog.props.len();
    let light_count = catalog.lighting.len();
    let theme_count = catalog.themes.len();

    eprintln!(
        "OK: generated {} ({} archetypes, {} props, {} lights, {} themes, source_hash={})",
        output_path.display(),
        archetype_count,
        prop_count,
        light_count,
        theme_count,
        &source_hash[..16]
    );

    Ok(())
}

// ── Catalog loading ────────────────────────────────────────────────────────

fn load_catalog(dir: &Path) -> Result<RichnessCatalog, String> {
    if !dir.is_dir() {
        return Err(format!(
            "catalog directory '{}' does not exist or is not a directory",
            dir.display()
        ));
    }

    let archetypes_file: ArchetypesFile =
        load_ron(&dir.join("archetypes.ron")).map_err(|e| format!("archetypes.ron: {e}"))?;
    let props_file: PropsFile =
        load_ron(&dir.join("props.ron")).map_err(|e| format!("props.ron: {e}"))?;
    let lighting_file: LightingFile =
        load_ron(&dir.join("lighting.ron")).map_err(|e| format!("lighting.ron: {e}"))?;
    let themes_file: ThemesFile =
        load_ron(&dir.join("themes.ron")).map_err(|e| format!("themes.ron: {e}"))?;
    let spiral_file: SpiralFile =
        load_ron(&dir.join("spiral_steps.ron")).map_err(|e| format!("spiral_steps.ron: {e}"))?;

    for (name, version) in [
        ("archetypes.ron", archetypes_file.schema_version.as_str()),
        ("props.ron", props_file.schema_version.as_str()),
        ("lighting.ron", lighting_file.schema_version.as_str()),
        ("themes.ron", themes_file.schema_version.as_str()),
        ("spiral_steps.ron", spiral_file.schema_version.as_str()),
    ] {
        if version != SCHEMA_VERSION {
            return Err(format!(
                "{name}: schema_version '{version}' does not match expected '{SCHEMA_VERSION}'"
            ));
        }
    }

    Ok(RichnessCatalog {
        schema_version: archetypes_file.schema_version,
        archetypes: archetypes_file.archetypes,
        props: props_file.props,
        lighting: lighting_file.lighting,
        themes: themes_file.themes,
        spiral_template: spiral_file.spiral_template,
    })
}

fn load_ron<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    ron::de::from_str::<T>(&content)
        .map_err(|e| format!("RON parse error in '{}': {e}", path.display()))
}

// ── Source hash ────────────────────────────────────────────────────────────

fn compute_source_hash(dir: &Path) -> Result<String, String> {
    let mut hasher = Sha256::new();
    // Hash files in canonical order for determinism
    let files = [
        "archetypes.ron",
        "props.ron",
        "lighting.ron",
        "themes.ron",
        "spiral_steps.ron",
    ];
    for fname in &files {
        let path = dir.join(fname);
        let content = fs::read(&path)
            .map_err(|e| format!("cannot read '{}' for hashing: {e}", path.display()))?;
        // Normalize line endings to LF before hashing (RON may have CRLF)
        let normalized = normalize_line_endings(&content);
        hasher.update(&normalized);
        // Use a separator byte between files
        hasher.update(b"\n");
    }
    let digest = hasher.finalize();
    Ok(hex::encode_upper(digest))
}

fn normalize_line_endings(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if data[i] == b'\r' && i + 1 < data.len() && data[i + 1] == b'\n' {
            result.push(b'\n');
            i += 2;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }
    result
}

// ── Rust code generation ───────────────────────────────────────────────────

fn generate_rust(catalog: &RichnessCatalog, source_hash: &str) -> String {
    let mut out = String::new();

    // Header
    out.push_str("//! Generated by richness_content_codegen. DO NOT EDIT.\n");
    out.push_str("//!\n");
    out.push_str(&format!("//! Schema version: {}\n", SCHEMA_VERSION));
    out.push_str(&format!("//! Source hash: {}\n", source_hash));
    out.push_str("\n");

    // Use statements for supporting types defined in the parent richness module
    out.push_str("use super::content_types::{\n");
    out.push_str("    light_index, prop_index, CollisionBehavior, FalloffStyle, LayerOccupancy, PlacementClass,\n");
    out.push_str("    RarityTier, ShapeRule, VerticalRecipe,\n");
    out.push_str("};\n");
    out.push_str("\n");

    // Schema version constant
    out.push_str(&format!(
        "pub const SCHEMA_VERSION: &str = {:?};\n",
        SCHEMA_VERSION
    ));
    out.push_str(&format!(
        "pub const SOURCE_HASH: &str = {:?};\n",
        source_hash
    ));
    out.push('\n');

    // Count constants
    out.push_str(&format!(
        "pub const ARCHETYPE_COUNT: usize = {};\n",
        catalog.archetypes.len()
    ));
    out.push_str(&format!(
        "pub const PROP_COUNT: usize = {};\n",
        catalog.props.len()
    ));
    out.push_str(&format!(
        "pub const LIGHT_RECIPE_COUNT: usize = {};\n",
        catalog.lighting.len()
    ));
    out.push_str(&format!(
        "pub const THEME_COUNT: usize = {};\n",
        catalog.themes.len()
    ));
    out.push('\n');

    // ── Archetype constants ────────────────────────────────────────────
    emit_archetypes(&mut out, &catalog.archetypes);

    // ── Prop constants ─────────────────────────────────────────────────
    emit_props(&mut out, &catalog.props);

    // ── Lighting recipe constants ──────────────────────────────────────
    emit_lighting(&mut out, &catalog.lighting);

    // ── Theme constants ────────────────────────────────────────────────
    emit_themes(&mut out, &catalog.themes);

    // ── Spiral template constants ──────────────────────────────────────
    emit_spiral(&mut out, &catalog.spiral_template);

    // The generator owns integer layout, so its raw output is directly
    // byte-comparable with the checked-in Rust and cargo fmt leaves it intact.
    out = out.replace("pub const ", "#[rustfmt::skip]\npub const ");
    while out.ends_with("\n\n") {
        out.pop();
    }
    out
}

// ── Archetype emission ─────────────────────────────────────────────────────

fn emit_archetypes(out: &mut String, archetypes: &[Archetype]) {
    // Emit ID array
    out.push_str("pub const ARCHETYPE_IDS: &[&str] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    {:?},\n", a.id));
    }
    out.push_str("];\n\n");

    // Emit span min array
    out.push_str("pub const ARCHETYPE_SPAN_MIN: &[[u32; 2]] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    [{}, {}],\n", a.span_min[0], a.span_min[1]));
    }
    out.push_str("];\n\n");

    // Emit span max array
    out.push_str("pub const ARCHETYPE_SPAN_MAX: &[[u32; 2]] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    [{}, {}],\n", a.span_max[0], a.span_max[1]));
    }
    out.push_str("];\n\n");

    // Emit shape rule
    out.push_str("pub const ARCHETYPE_SHAPE: &[ShapeRule] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    ShapeRule::{},\n", a.shape.as_str()));
    }
    out.push_str("];\n\n");

    // Emit exit degree bounds
    out.push_str("pub const ARCHETYPE_EXIT_DEGREE_MIN: &[u32] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    {},\n", a.exit_degree_min));
    }
    out.push_str("];\n\n");

    out.push_str("pub const ARCHETYPE_EXIT_DEGREE_MAX: &[u32] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    {},\n", a.exit_degree_max));
    }
    out.push_str("];\n\n");

    // Emit layer occupancy
    out.push_str("pub const ARCHETYPE_LAYER_OCCUPANCY: &[LayerOccupancy] = &[\n");
    for a in archetypes {
        out.push_str(&format!(
            "    LayerOccupancy::{},\n",
            a.layer_occupancy.as_str()
        ));
    }
    out.push_str("];\n\n");

    // Emit route witness envelope
    out.push_str("pub const ARCHETYPE_ROUTE_WITNESS: &[[u32; 2]] = &[\n");
    for a in archetypes {
        out.push_str(&format!(
            "    [{}, {}],\n",
            a.route_witness_envelope[0], a.route_witness_envelope[1]
        ));
    }
    out.push_str("];\n\n");

    // Emit vertical recipe
    out.push_str("pub const ARCHETYPE_VERTICAL_RECIPE: &[VerticalRecipe] = &[\n");
    for a in archetypes {
        out.push_str(&format!(
            "    VerticalRecipe::{},\n",
            a.vertical_recipe.as_str()
        ));
    }
    out.push_str("];\n\n");

    // Emit rarity
    out.push_str("pub const ARCHETYPE_RARITY: &[RarityTier] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    RarityTier::{},\n", a.rarity.as_str()));
    }
    out.push_str("];\n\n");

    // Emit negative space budget
    out.push_str("pub const ARCHETYPE_NEGATIVE_SPACE_BUDGET: &[u32] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    {},\n", a.negative_space_budget));
    }
    out.push_str("];\n\n");

    // Emit support rules
    out.push_str("pub const ARCHETYPE_SUPPORT_RULES: &[&str] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    {:?},\n", a.support_rules));
    }
    out.push_str("];\n\n");

    // Emit costs
    emit_archetype_costs(out, archetypes);

    // Emit zone compatibility (references into ZONE_NAMES)
    let zone_names = collect_sorted_strings(
        archetypes
            .iter()
            .flat_map(|a| a.zone_compatibility.iter().cloned()),
    );
    out.push_str("pub const ZONE_NAMES: &[&str] = &[\n");
    for z in &zone_names {
        out.push_str(&format!("    {:?},\n", z));
    }
    out.push_str("];\n\n");

    out.push_str("pub const ARCHETYPE_ZONE_COMPAT: &[&[u32]] = &[\n");
    for a in archetypes {
        let indices: Vec<String> = a
            .zone_compatibility
            .iter()
            .map(|z| zone_names.iter().position(|n| n == z).unwrap().to_string())
            .collect();
        out.push_str(&format!("    &[{}],\n", indices.join(", ")));
    }
    out.push_str("];\n\n");

    // Emit grammar compatibility
    let grammar_names = collect_sorted_strings(
        archetypes
            .iter()
            .flat_map(|a| a.grammar_compatibility.iter().cloned()),
    );
    out.push_str("pub const GRAMMAR_NAMES: &[&str] = &[\n");
    for g in &grammar_names {
        out.push_str(&format!("    {:?},\n", g));
    }
    out.push_str("];\n\n");

    out.push_str("pub const ARCHETYPE_GRAMMAR_COMPAT: &[&[u32]] = &[\n");
    for a in archetypes {
        let indices: Vec<String> = a
            .grammar_compatibility
            .iter()
            .map(|g| {
                grammar_names
                    .iter()
                    .position(|n| n == g)
                    .unwrap()
                    .to_string()
            })
            .collect();
        out.push_str(&format!("    &[{}],\n", indices.join(", ")));
    }
    out.push_str("];\n\n");

    // Emit prop references (indexed into PROP_IDS)
    out.push_str("pub const ARCHETYPE_PROP_REFS: &[&[u32]] = &[\n");
    for a in archetypes {
        let indices: Vec<String> = a
            .prop_references
            .iter()
            .map(|pr| format!("prop_index({pr:?})"))
            .collect();
        out.push_str(&format!("    &[{}],\n", indices.join(", ")));
    }
    out.push_str("];\n\n");

    // Emit light references (indexed into LIGHT_IDS)
    out.push_str("pub const ARCHETYPE_LIGHT_REFS: &[&[u32]] = &[\n");
    for a in archetypes {
        let indices: Vec<String> = a
            .light_references
            .iter()
            .map(|lr| format!("light_index({lr:?})"))
            .collect();
        out.push_str(&format!("    &[{}],\n", indices.join(", ")));
    }
    out.push_str("];\n\n");

    // Emit material roles
    emit_archetype_material_roles(out, archetypes);

    // Emit theme variant data
    emit_archetype_theme_variants(out, archetypes);
}

fn emit_archetype_costs(out: &mut String, archetypes: &[Archetype]) {
    out.push_str("pub const ARCHETYPE_COST_SOURCE_FACES: &[u32] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    {},\n", a.costs.source_faces));
    }
    out.push_str("];\n\n");

    out.push_str("pub const ARCHETYPE_COST_BRUSHES: &[u32] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    {},\n", a.costs.brushes));
    }
    out.push_str("];\n\n");

    out.push_str("pub const ARCHETYPE_COST_ENTITIES: &[u32] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    {},\n", a.costs.entities));
    }
    out.push_str("];\n\n");

    out.push_str("pub const ARCHETYPE_COST_LIGHTS: &[u32] = &[\n");
    for a in archetypes {
        out.push_str(&format!("    {},\n", a.costs.lights));
    }
    out.push_str("];\n\n");
}

fn emit_archetype_material_roles(out: &mut String, archetypes: &[Archetype]) {
    // Collect all unique role names
    let role_names: BTreeSet<&str> = archetypes
        .iter()
        .flat_map(|a| a.material_roles.iter().map(|m| m.role.as_str()))
        .collect();
    let role_names: Vec<&str> = role_names.into_iter().collect();

    out.push_str("pub const MATERIAL_ROLE_NAMES: &[&str] = &[\n");
    for r in &role_names {
        out.push_str(&format!("    {:?},\n", r));
    }
    out.push_str("];\n\n");

    // Per-archetype: (role_index, texture) pairs
    out.push_str("pub const ARCHETYPE_MATERIAL_ROLES: &[&[(u32, &str)]] = &[\n");
    for a in archetypes {
        let pairs: Vec<String> = a
            .material_roles
            .iter()
            .map(|m| {
                let idx = role_names
                    .iter()
                    .position(|&r| r == m.role.as_str())
                    .unwrap();
                format!("({}, {:?})", idx, m.texture)
            })
            .collect();
        out.push_str(&format!("    &[{}],\n", pairs.join(", ")));
    }
    out.push_str("];\n\n");
}

fn emit_archetype_theme_variants(out: &mut String, archetypes: &[Archetype]) {
    // Theme order is always [ancient, egyptian, brutalist] = [0, 1, 2]
    out.push_str("// Theme variants: [ancient=0, egyptian=1, brutalist=2]\n");

    out.push_str("pub const ARCHETYPE_THEME_MASSING: &[[&str; 3]] = &[\n");
    for a in archetypes {
        let massings = get_theme_field(a, |v| v.massing.clone());
        out.push_str(&format!(
            "    [{:?}, {:?}, {:?}],\n",
            massings[0], massings[1], massings[2]
        ));
    }
    out.push_str("];\n\n");

    out.push_str("pub const ARCHETYPE_THEME_SUPPORT: &[[&str; 3]] = &[\n");
    for a in archetypes {
        let supports = get_theme_field(a, |v| v.support_data.clone());
        out.push_str(&format!(
            "    [{:?}, {:?}, {:?}],\n",
            supports[0], supports[1], supports[2]
        ));
    }
    out.push_str("];\n\n");

    // Emit theme variant prop refs
    out.push_str("pub const ARCHETYPE_THEME_PROP_REFS: &[[&[u32]; 3]] = &[\n");
    for a in archetypes {
        let prop_refs: Vec<String> = ["ancient", "egyptian", "brutalist"]
            .iter()
            .map(|theme_name| {
                let variant = a
                    .theme_variants
                    .iter()
                    .find(|v| v.theme == *theme_name)
                    .unwrap();
                let indices: Vec<String> = variant
                    .props
                    .iter()
                    .map(|pr| format!("prop_index({pr:?})"))
                    .collect();
                format!("&[{}]", indices.join(", "))
            })
            .collect();
        out.push_str(&format!(
            "    [{}, {}, {}],\n",
            prop_refs[0], prop_refs[1], prop_refs[2]
        ));
    }
    out.push_str("];\n\n");

    // Emit theme variant light refs
    out.push_str("pub const ARCHETYPE_THEME_LIGHT_REFS: &[[&[u32]; 3]] = &[\n");
    for a in archetypes {
        let light_refs: Vec<String> = ["ancient", "egyptian", "brutalist"]
            .iter()
            .map(|theme_name| {
                let variant = a
                    .theme_variants
                    .iter()
                    .find(|v| v.theme == *theme_name)
                    .unwrap();
                let indices: Vec<String> = variant
                    .lights
                    .iter()
                    .map(|lr| format!("light_index({lr:?})"))
                    .collect();
                format!("&[{}]", indices.join(", "))
            })
            .collect();
        out.push_str(&format!(
            "    [{}, {}, {}],\n",
            light_refs[0], light_refs[1], light_refs[2]
        ));
    }
    out.push_str("];\n\n");

    // Emit theme variant materials
    out.push_str("pub const ARCHETYPE_THEME_MATERIALS: &[[&[&str]; 3]] = &[\n");
    for a in archetypes {
        let mats: Vec<String> = ["ancient", "egyptian", "brutalist"]
            .iter()
            .map(|theme_name| {
                let variant = a
                    .theme_variants
                    .iter()
                    .find(|v| v.theme == *theme_name)
                    .unwrap();
                let quoted: Vec<String> =
                    variant.materials.iter().map(|m| format!("{m:?}")).collect();
                format!("&[{}]", quoted.join(", "))
            })
            .collect();
        out.push_str(&format!("    [{}, {}, {}],\n", mats[0], mats[1], mats[2]));
    }
    out.push_str("];\n\n");
}

fn get_theme_field<F: Fn(&schema::ThemeGeometryVariant) -> String>(
    a: &Archetype,
    f: F,
) -> [String; 3] {
    let ancient = f(a
        .theme_variants
        .iter()
        .find(|v| v.theme == "ancient")
        .unwrap());
    let egyptian = f(a
        .theme_variants
        .iter()
        .find(|v| v.theme == "egyptian")
        .unwrap());
    let brutalist = f(a
        .theme_variants
        .iter()
        .find(|v| v.theme == "brutalist")
        .unwrap());
    [ancient, egyptian, brutalist]
}

// ── Prop emission ──────────────────────────────────────────────────────────

fn emit_props(out: &mut String, props: &[Prop]) {
    out.push_str("pub const PROP_IDS: &[&str] = &[\n");
    for p in props {
        out.push_str(&format!("    {:?},\n", p.id));
    }
    out.push_str("];\n\n");

    out.push_str("pub const PROP_CONVEX_PIECES: &[u32] = &[\n");
    for p in props {
        out.push_str(&format!("    {},\n", p.convex_pieces));
    }
    out.push_str("];\n\n");

    out.push_str("pub const PROP_DIMENSIONS: &[[u32; 3]] = &[\n");
    for p in props {
        out.push_str(&format!(
            "    [{}, {}, {}],\n",
            p.dimensions[0], p.dimensions[1], p.dimensions[2]
        ));
    }
    out.push_str("];\n\n");

    out.push_str("pub const PROP_COLLISION: &[CollisionBehavior] = &[\n");
    for p in props {
        out.push_str(&format!(
            "    CollisionBehavior::{},\n",
            p.collision_behavior.as_str()
        ));
    }
    out.push_str("];\n\n");

    out.push_str("pub const PROP_SWEPT_OCCUPANCY: &[[u32; 3]] = &[\n");
    for p in props {
        out.push_str(&format!(
            "    [{}, {}, {}],\n",
            p.swept_occupancy[0], p.swept_occupancy[1], p.swept_occupancy[2]
        ));
    }
    out.push_str("];\n\n");

    out.push_str("pub const PROP_SUPPORT_CONTACTS: &[u32] = &[\n");
    for p in props {
        out.push_str(&format!("    {},\n", p.support_contacts));
    }
    out.push_str("];\n\n");

    // Prop light coupling
    out.push_str("pub const PROP_LIGHT_COUPLING: &[&[u32]] = &[\n");
    for p in props {
        let indices: Vec<String> = p
            .light_coupling
            .iter()
            .map(|lr| format!("light_index({lr:?})"))
            .collect();
        out.push_str(&format!("    &[{}],\n", indices.join(", ")));
    }
    out.push_str("];\n\n");

    // Prop costs
    out.push_str("pub const PROP_COST_SOURCE_FACES: &[u32] = &[\n");
    for p in props {
        out.push_str(&format!("    {},\n", p.costs.source_faces));
    }
    out.push_str("];\n\n");
    out.push_str("pub const PROP_COST_BRUSHES: &[u32] = &[\n");
    for p in props {
        out.push_str(&format!("    {},\n", p.costs.brushes));
    }
    out.push_str("];\n\n");
    out.push_str("pub const PROP_COST_ENTITIES: &[u32] = &[\n");
    for p in props {
        out.push_str(&format!("    {},\n", p.costs.entities));
    }
    out.push_str("];\n\n");
    out.push_str("pub const PROP_COST_LIGHTS: &[u32] = &[\n");
    for p in props {
        out.push_str(&format!("    {},\n", p.costs.lights));
    }
    out.push_str("];\n\n");

    // Prop theme variants
    out.push_str("pub const PROP_THEME_MODEL_OVERRIDE: &[[&str; 3]] = &[\n");
    for p in props {
        let models = prop_theme_field(p, |v| v.model_override.clone());
        out.push_str(&format!(
            "    [{:?}, {:?}, {:?}],\n",
            models[0], models[1], models[2]
        ));
    }
    out.push_str("];\n\n");

    out.push_str("pub const PROP_THEME_DIMENSIONS: &[[[u32; 3]; 3]] = &[\n");
    for p in props {
        let dimensions: Vec<[u32; 3]> = ["ancient", "egyptian", "brutalist"]
            .iter()
            .map(|theme| {
                p.theme_variants
                    .iter()
                    .find(|variant| variant.theme == *theme)
                    .and_then(|variant| variant.dimensions_override)
                    .expect("validated prop theme dimensions")
            })
            .collect();
        out.push_str(&format!(
            "    [[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]],\n",
            dimensions[0][0],
            dimensions[0][1],
            dimensions[0][2],
            dimensions[1][0],
            dimensions[1][1],
            dimensions[1][2],
            dimensions[2][0],
            dimensions[2][1],
            dimensions[2][2],
        ));
    }
    out.push_str("];\n\n");

    out.push_str(
        "pub const PROP_THEME_COLLISION_OVERRIDE: &[[Option<CollisionBehavior>; 3]] = &[\n",
    );
    for p in props {
        let values: Vec<String> = ["ancient", "egyptian", "brutalist"]
            .iter()
            .map(|theme| {
                match p
                    .theme_variants
                    .iter()
                    .find(|variant| variant.theme == *theme)
                    .and_then(|variant| variant.collision_behavior)
                {
                    Some(value) => format!("Some(CollisionBehavior::{})", value.as_str()),
                    None => "None".to_string(),
                }
            })
            .collect();
        out.push_str(&format!(
            "    [{}, {}, {}],\n",
            values[0], values[1], values[2]
        ));
    }
    out.push_str("];\n\n");
}

fn prop_theme_field<F: Fn(&schema::PropThemeVariant) -> String>(p: &Prop, f: F) -> [String; 3] {
    let ancient = f(p
        .theme_variants
        .iter()
        .find(|v| v.theme == "ancient")
        .unwrap());
    let egyptian = f(p
        .theme_variants
        .iter()
        .find(|v| v.theme == "egyptian")
        .unwrap());
    let brutalist = f(p
        .theme_variants
        .iter()
        .find(|v| v.theme == "brutalist")
        .unwrap());
    [ancient, egyptian, brutalist]
}

// ── Lighting emission ──────────────────────────────────────────────────────

fn emit_lighting(out: &mut String, lighting: &[LightRecipe]) {
    out.push_str("pub const LIGHT_RECIPE_IDS: &[&str] = &[\n");
    for l in lighting {
        out.push_str(&format!("    {:?},\n", l.id));
    }
    out.push_str("];\n\n");

    out.push_str("pub const LIGHT_COLOR: &[[u8; 3]] = &[\n");
    for l in lighting {
        out.push_str(&format!(
            "    [{}, {}, {}],\n",
            l.color[0], l.color[1], l.color[2]
        ));
    }
    out.push_str("];\n\n");

    out.push_str("pub const LIGHT_INTENSITY: &[u32] = &[\n");
    for l in lighting {
        out.push_str(&format!("    {},\n", l.intensity));
    }
    out.push_str("];\n\n");

    out.push_str("pub const LIGHT_PLACEMENT_CLASS: &[PlacementClass] = &[\n");
    for l in lighting {
        out.push_str(&format!(
            "    PlacementClass::{},\n",
            l.placement_class.as_str()
        ));
    }
    out.push_str("];\n\n");

    out.push_str("pub const LIGHT_FALLOFF: &[FalloffStyle] = &[\n");
    for l in lighting {
        out.push_str(&format!("    FalloffStyle::{},\n", l.falloff.as_str()));
    }
    out.push_str("];\n\n");

    out.push_str("pub const LIGHT_READABILITY_FLOOR: &[u32] = &[\n");
    for l in lighting {
        out.push_str(&format!("    {},\n", l.readability_floor));
    }
    out.push_str("];\n\n");

    out.push_str("pub const LIGHT_COUNT: &[u32] = &[\n");
    for l in lighting {
        out.push_str(&format!("    {},\n", l.count));
    }
    out.push_str("];\n\n");

    // Light costs
    out.push_str("pub const LIGHT_COST_SOURCE_FACES: &[u32] = &[\n");
    for l in lighting {
        out.push_str(&format!("    {},\n", l.costs.source_faces));
    }
    out.push_str("];\n\n");
    out.push_str("pub const LIGHT_COST_BRUSHES: &[u32] = &[\n");
    for l in lighting {
        out.push_str(&format!("    {},\n", l.costs.brushes));
    }
    out.push_str("];\n\n");
    out.push_str("pub const LIGHT_COST_ENTITIES: &[u32] = &[\n");
    for l in lighting {
        out.push_str(&format!("    {},\n", l.costs.entities));
    }
    out.push_str("];\n\n");
    out.push_str("pub const LIGHT_COST_LIGHTS: &[u32] = &[\n");
    for l in lighting {
        out.push_str(&format!("    {},\n", l.costs.lights));
    }
    out.push_str("];\n\n");

    // Entity keys are per-light; emit as function or const data
    // We emit entity key count per light and a concatenated key/value array
    let mut all_keys: Vec<&str> = Vec::new();
    let mut all_values: Vec<&str> = Vec::new();
    let mut key_ranges: Vec<(usize, usize)> = Vec::new();

    for l in lighting {
        let start = all_keys.len();
        for ek in &l.entity_keys {
            all_keys.push(ek.key.as_str());
            all_values.push(ek.value.as_str());
        }
        let end = all_keys.len();
        key_ranges.push((start, end));
    }

    out.push_str("pub const LIGHT_ENTITY_KEYS: &[&str] = &[\n");
    for k in &all_keys {
        out.push_str(&format!("    {:?},\n", k));
    }
    out.push_str("];\n\n");

    out.push_str("pub const LIGHT_ENTITY_VALUES: &[&str] = &[\n");
    for v in &all_values {
        out.push_str(&format!("    {:?},\n", v));
    }
    out.push_str("];\n\n");

    out.push_str("pub const LIGHT_ENTITY_KEY_SPANS: &[(usize, usize)] = &[\n");
    for (s, e) in &key_ranges {
        out.push_str(&format!("    ({}, {}),\n", s, e));
    }
    out.push_str("];\n\n");
}

// ── Theme emission ─────────────────────────────────────────────────────────

fn emit_themes(out: &mut String, themes: &[Theme]) {
    out.push_str("pub const THEME_IDS: &[&str] = &[\n");
    for t in themes {
        out.push_str(&format!("    {:?},\n", t.id));
    }
    out.push_str("];\n\n");

    // Collect unique semantic roles
    let roles: Vec<String> = collect_sorted_strings(
        themes
            .iter()
            .flat_map(|t| t.semantic_roles.iter().map(|s| s.as_str())),
    );
    out.push_str("pub const THEME_SEMANTIC_ROLE_NAMES: &[&str] = &[\n");
    for r in &roles {
        out.push_str(&format!("    {:?},\n", r));
    }
    out.push_str("];\n\n");

    out.push_str("pub const THEME_SEMANTIC_ROLES: &[&[u32]] = &[\n");
    for t in themes {
        let indices: Vec<String> = t
            .semantic_roles
            .iter()
            .map(|s| {
                roles
                    .iter()
                    .position(|r| r.as_str() == s.as_str())
                    .unwrap()
                    .to_string()
            })
            .collect();
        out.push_str(&format!("    &[{}],\n", indices.join(", ")));
    }
    out.push_str("];\n\n");

    // Transitions
    out.push_str("pub const THEME_TRANSITIONS: &[&[&str]] = &[\n");
    for t in themes {
        let quoted: Vec<String> = t.transitions.iter().map(|tr| format!("{tr:?}")).collect();
        out.push_str(&format!("    &[{}],\n", quoted.join(", ")));
    }
    out.push_str("];\n\n");

    // Geometry vocabulary
    out.push_str("pub const THEME_GEOMETRY_VOCABULARY: &[&[&str]] = &[\n");
    for t in themes {
        let quoted: Vec<String> = t
            .geometry_vocabulary
            .iter()
            .map(|gv| format!("{gv:?}"))
            .collect();
        out.push_str(&format!("    &[{}],\n", quoted.join(", ")));
    }
    out.push_str("];\n\n");

    // Theme material roles
    let theme_mat_roles: Vec<String> = collect_sorted_strings(
        themes
            .iter()
            .flat_map(|t| t.material_roles.iter().map(|m| m.role.as_str())),
    );
    out.push_str("pub const THEME_MATERIAL_ROLE_NAMES: &[&str] = &[\n");
    for r in &theme_mat_roles {
        out.push_str(&format!("    {:?},\n", r));
    }
    out.push_str("];\n\n");

    out.push_str("pub const THEME_MATERIAL_ROLES: &[&[(u32, &str)]] = &[\n");
    for t in themes {
        let pairs: Vec<String> = t
            .material_roles
            .iter()
            .map(|m| {
                let idx = theme_mat_roles
                    .iter()
                    .position(|r| r.as_str() == m.role.as_str())
                    .unwrap();
                format!("({}, {:?})", idx, m.texture)
            })
            .collect();
        out.push_str(&format!("    &[{}],\n", pairs.join(", ")));
    }
    out.push_str("];\n\n");

    // Prop compatibility
    out.push_str("pub const THEME_PROP_COMPAT: &[&[u32]] = &[\n");
    for t in themes {
        let indices: Vec<String> = t
            .prop_compatibility
            .iter()
            .map(|pr| format!("prop_index({pr:?})"))
            .collect();
        out.push_str(&format!("    &[{}],\n", indices.join(", ")));
    }
    out.push_str("];\n\n");

    // Light compatibility
    out.push_str("pub const THEME_LIGHT_COMPAT: &[&[u32]] = &[\n");
    for t in themes {
        let indices: Vec<String> = t
            .light_compatibility
            .iter()
            .map(|lr| format!("light_index({lr:?})"))
            .collect();
        out.push_str(&format!("    &[{}],\n", indices.join(", ")));
    }
    out.push_str("];\n\n");

    // Theme budget
    out.push_str("pub const THEME_BUDGET_SOURCE_FACES: &[u32] = &[\n");
    for t in themes {
        out.push_str(&format!("    {},\n", t.budget.source_faces));
    }
    out.push_str("];\n\n");
    out.push_str("pub const THEME_BUDGET_BRUSHES: &[u32] = &[\n");
    for t in themes {
        out.push_str(&format!("    {},\n", t.budget.brushes));
    }
    out.push_str("];\n\n");
    out.push_str("pub const THEME_BUDGET_ENTITIES: &[u32] = &[\n");
    for t in themes {
        out.push_str(&format!("    {},\n", t.budget.entities));
    }
    out.push_str("];\n\n");
    out.push_str("pub const THEME_BUDGET_LIGHTS: &[u32] = &[\n");
    for t in themes {
        out.push_str(&format!("    {},\n", t.budget.lights));
    }
    out.push_str("];\n\n");
}

// ── Spiral template emission ───────────────────────────────────────────────

fn emit_spiral(out: &mut String, st: &schema::SpiralTemplate) {
    out.push_str("pub const SPIRAL_LAYER_OFFSET: u32 = ");
    out.push_str(&st.layer_offset.to_string());
    out.push_str(";\n\n");

    out.push_str("pub const SPIRAL_ENVELOPE_MIN: [u32; 2] = [");
    out.push_str(&format!(
        "{}, {}];\n\n",
        st.envelope_min[0], st.envelope_min[1]
    ));

    out.push_str("pub const SPIRAL_STEP_COUNT: usize = 12;\n\n");

    out.push_str("pub const SPIRAL_STEP_INDEX: &[u32] = &[\n");
    for s in &st.steps {
        out.push_str(&format!("    {},\n", s.step_index));
    }
    out.push_str("];\n\n");

    out.push_str("pub const SPIRAL_STEP_RISE: &[u32] = &[\n");
    for s in &st.steps {
        out.push_str(&format!("    {},\n", s.rise));
    }
    out.push_str("];\n\n");

    out.push_str("pub const SPIRAL_STEP_ENVELOPE: &[[u32; 2]] = &[\n");
    for s in &st.steps {
        out.push_str(&format!("    [{}, {}],\n", s.envelope[0], s.envelope[1]));
    }
    out.push_str("];\n\n");

    out.push_str("pub const SPIRAL_STEP_CENTER_COLUMN: &[[u32; 2]] = &[\n");
    for s in &st.steps {
        out.push_str(&format!(
            "    [{}, {}],\n",
            s.center_column[0], s.center_column[1]
        ));
    }
    out.push_str("];\n\n");

    out.push_str("pub const SPIRAL_STEP_TREAD_DEPTH: &[u32] = &[\n");
    for s in &st.steps {
        out.push_str(&format!("    {},\n", s.tread_depth));
    }
    out.push_str("];\n\n");

    out.push_str("pub const SPIRAL_STEP_IS_CONVEX: &[bool] = &[\n");
    for s in &st.steps {
        out.push_str(&format!("    {},\n", s.is_convex_recipe));
    }
    out.push_str("];\n\n");
}

// ── Atomic write ───────────────────────────────────────────────────────────

fn write_atomic(path: &Path, content: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("output path '{}' has no parent directory", path.display()))?;

    // Create parent directory if needed
    fs::create_dir_all(parent)
        .map_err(|e| format!("cannot create parent directory '{}': {e}", parent.display()))?;

    let tmp_path = path.with_extension("tmp");

    // Check if existing file already has the same content
    if path.exists() {
        match fs::read_to_string(path) {
            Ok(existing) if existing == content => {
                eprintln!("output unchanged, skipping write: {}", path.display());
                return Ok(());
            }
            _ => {}
        }
    }

    // Write to temporary file
    let mut tmp = fs::File::create(&tmp_path)
        .map_err(|e| format!("cannot create temp file '{}': {e}", tmp_path.display()))?;

    tmp.write_all(content.as_bytes())
        .map_err(|e| format!("cannot write to temp file '{}': {e}", tmp_path.display()))?;

    // fsync the temp file
    tmp.sync_all()
        .map_err(|e| format!("cannot fsync temp file '{}': {e}", tmp_path.display()))?;

    // Compare byte-for-byte with existing file (re-read to be safe)
    if path.exists() {
        if let Ok(existing) = fs::read(path) {
            if existing == content.as_bytes() {
                // Content is byte-identical; clean up temp
                let _ = fs::remove_file(&tmp_path);
                eprintln!(
                    "output unchanged (byte-compare), skipping replace: {}",
                    path.display()
                );
                return Ok(());
            }
        }
    }

    // Atomic rename
    fs::rename(&tmp_path, path).map_err(|e| {
        format!(
            "cannot rename '{}' to '{}': {e}",
            tmp_path.display(),
            path.display()
        )
    })?;

    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn collect_sorted_strings(iter: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    let strings: BTreeSet<String> = iter.into_iter().map(|s| s.as_ref().to_string()).collect();
    strings.into_iter().collect()
}

/// Simple hex encoding (no external crate needed).
mod hex {
    pub fn encode_upper(bytes: impl AsRef<[u8]>) -> String {
        let bytes = bytes.as_ref();
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push(nibble_to_hex((b >> 4) & 0xF));
            s.push(nibble_to_hex(b & 0xF));
        }
        s
    }

    fn nibble_to_hex(n: u8) -> char {
        match n {
            0 => '0',
            1 => '1',
            2 => '2',
            3 => '3',
            4 => '4',
            5 => '5',
            6 => '6',
            7 => '7',
            8 => '8',
            9 => '9',
            10 => 'A',
            11 => 'B',
            12 => 'C',
            13 => 'D',
            14 => 'E',
            15 => 'F',
            _ => unreachable!(),
        }
    }
}

// ── Lookup helper functions emitted in the generated code ──────────────────

/// Emit the `prop_index` and `light_index` lookup functions in a separate
/// section. These are const-compatible functions generated alongside the data.
/// However, for simplicity, we inline the index lookups directly and emit
/// the function definitions at the end of the generated file.
pub fn emit_lookup_fns(out: &mut String) {
    out.push_str("\n// ── Lookup helpers ───────────────────────────────────────────────────\n\n");
    out.push_str(
        r#"/// Look up a prop index by its stable ID. Panics (in const context: compile error) on unknown IDs.
pub const fn prop_index(id: &str) -> u32 {
    // Binary search (IDs are in lexical order)
    let ids = PROP_IDS;
    let mut lo = 0;
    let mut hi = ids.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        // const-friendly string comparison
        let cmp = const_str_cmp(id, ids[mid]);
        if cmp < 0 {
            hi = mid;
        } else if cmp > 0 {
            lo = mid + 1;
        } else {
            return mid as u32;
        }
    }
    panic!("unknown prop ID");
}

/// Look up a light recipe index by its stable ID.
pub const fn light_index(id: &str) -> u32 {
    let ids = LIGHT_RECIPE_IDS;
    let mut lo = 0;
    let mut hi = ids.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let cmp = const_str_cmp(id, ids[mid]);
        if cmp < 0 {
            hi = mid;
        } else if cmp > 0 {
            lo = mid + 1;
        } else {
            return mid as u32;
        }
    }
    panic!("unknown light recipe ID");
}

/// Const-compatible string comparison. Returns negative/zero/positive.
const fn const_str_cmp(a: &str, b: &str) -> i32 {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let mut i = 0;
    while i < a.len() && i < b.len() {
        if a[i] < b[i] {
            return -1;
        } else if a[i] > b[i] {
            return 1;
        }
        i += 1;
    }
    if a.len() < b.len() {
        -1
    } else if a.len() > b.len() {
        1
    } else {
        0
    }
}
"#,
    );
}
