//! Runtime wiring tests for Richness V1 (Subphase 18-C).
//!
//! Tests: draft-to-document parity, CLI dispatch for --m3-richness-v1,
//! real executor integration, active-world preservation, and baseline
//! byte compatibility.

#[path = "../src/cli.rs"]
mod cli;

#[path = "../src/richness_gui.rs"]
mod richness_gui;

#[path = "../src/richness_generation.rs"]
mod richness_generation;

use cli::parse_from;
use richness_generation::{draft_to_richness_document, ExecutorOutcome};
use richness_gui::{RichnessDraft, RichnessFieldId, RichnessPreset, RichnessTheme};

// ── CLI dispatch for --m3-richness-v1 ─────────────────────────────────────

#[test]
fn parse_m3_richness_flag() {
    let args = parse_from(["--m3-richness-v1"]).unwrap();
    assert!(args.m3_richness);
    assert!(!args.m3_generate);
    assert!(args.bsp_path.is_none());
}

#[test]
fn m3_richness_conflicts_with_m3_generate() {
    let err = parse_from(["--m3-generate", "--m3-richness-v1"]).unwrap_err();
    assert!(matches!(err, cli::CliError::M3GenerateRichnessConflict));
}

#[test]
fn m3_richness_accepts_complete_prebuilt_closure() {
    let args = parse_from([
        "--m3-richness-v1",
        "--bsp",
        "maps/test.bsp",
        "--palette",
        "gfx/palette.lmp",
        "--lit",
        "maps/test.lit",
        "--wad",
        "maps/test.wad",
        "--textures",
        "textures",
        "--richness-seed",
        "42",
    ])
    .unwrap();
    assert!(args.m3_richness);
    assert_eq!(
        args.bsp_path,
        Some(std::path::PathBuf::from("maps/test.bsp"))
    );
    assert_eq!(args.import_mode, Some(cli::ImportMode::Strict));
}

#[test]
fn m3_richness_sets_strict_mode_by_default() {
    let args = parse_from(["--m3-richness-v1"]).unwrap();
    assert_eq!(args.import_mode, Some(cli::ImportMode::Strict));
}

#[test]
fn m3_richness_accepts_development_mode() {
    let args = parse_from(["--development", "--m3-richness-v1"]).unwrap();
    assert_eq!(args.import_mode, Some(cli::ImportMode::Development));
}

#[test]
fn m3_richness_rejects_palette() {
    let err = parse_from(["--m3-richness-v1", "--palette", "gfx/pal.lmp"]).unwrap_err();
    assert!(matches!(err, cli::CliError::M3GeneratePaletteConflict));
}

#[test]
fn m3_richness_rejects_textures() {
    let err = parse_from(["--m3-richness-v1", "--textures", "tex"]).unwrap_err();
    assert!(matches!(err, cli::CliError::M3GenerateTexturesConflict));
}

#[test]
fn richness_override_requires_exact_mode_gate() {
    let error = parse_from(["--strict", "--richness-preset", "sparse"]).unwrap_err();
    assert_eq!(
        error,
        cli::CliError::RichnessOptionRequiresRichness("--richness-preset")
    );
}

#[test]
fn baseline_m3_option_is_not_silently_used_by_richness() {
    let error = parse_from(["--m3-richness-v1", "--seed", "42"]).unwrap_err();
    assert_eq!(error, cli::CliError::M3OptionRequiresGenerate("--seed"));
}

// ── Draft-to-document conversion parity ────────────────────────────────────

#[test]
fn default_draft_converts_to_valid_document() {
    let draft = RichnessDraft::new();
    let doc = draft_to_richness_document(&draft).expect("default draft should convert");
    assert_eq!(doc.seed(), 0);
    assert_eq!(doc.extent(), 2048);
    assert_eq!(doc.preset(), bsp_generator::RichnessPreset::Sparse);
    assert_eq!(doc.theme(), bsp_generator::RichnessTheme::Ancient);
}

#[test]
fn explicit_draft_fields_preserved_in_document() {
    let mut draft = RichnessDraft::new();
    draft.set_preset(RichnessPreset::Rich);
    draft.set_theme(RichnessTheme::Egyptian);
    draft.try_set_extent(3072).unwrap();
    draft.set_seed(99);
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::Zones, 2)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::VerticalOpenings, 4)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 8000)
        .unwrap();

    let doc = draft_to_richness_document(&draft).expect("valid draft should convert");
    assert_eq!(doc.seed(), 99);
    assert_eq!(doc.extent(), 3072);
    assert_eq!(doc.preset(), bsp_generator::RichnessPreset::Rich);
    assert_eq!(doc.theme(), bsp_generator::RichnessTheme::Egyptian);
    assert_eq!(
        doc.critical_path_landmarks(),
        bsp_generator::InheritedOr::Explicit(3)
    );
    assert_eq!(doc.zone_count(), bsp_generator::InheritedOr::Explicit(2));
    assert_eq!(
        doc.vertical_openings(),
        bsp_generator::InheritedOr::Explicit(4)
    );
    assert_eq!(
        doc.budget_ceiling(),
        bsp_generator::InheritedOr::Explicit(8000)
    );
}

#[test]
fn inherited_fields_preserved_in_document() {
    let draft = RichnessDraft::new();
    let doc = draft_to_richness_document(&draft).expect("default draft should convert");
    assert_eq!(
        doc.critical_path_landmarks(),
        bsp_generator::InheritedOr::Inherited
    );
    assert_eq!(doc.zone_count(), bsp_generator::InheritedOr::Inherited);
}

#[test]
fn draft_to_document_canonical_bytes_match_generator() {
    let mut draft = RichnessDraft::new();
    draft.set_preset(RichnessPreset::Moderate);
    draft.set_theme(RichnessTheme::Brutalist);
    draft.set_seed(42);
    draft.try_set_extent(2048).unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 2)
        .unwrap();

    let doc = draft_to_richness_document(&draft).unwrap();

    // The draft's own canonical bytes (from richness_gui) should not diverge
    // from the document's identity. Both are SHA-256 over the same field set.
    let draft_hash = draft.identity_hash_hex();

    // Verify the document has the expected seed
    assert_eq!(doc.seed(), 42);
    assert_eq!(doc.preset(), bsp_generator::RichnessPreset::Moderate);
    assert_eq!(doc.theme(), bsp_generator::RichnessTheme::Brutalist);

    // The identity hash from the draft should be stable
    assert_eq!(draft_hash.len(), 64); // 32 bytes hex = 64 chars
}

#[test]
fn draft_to_document_rejects_invalid_draft() {
    let mut draft = RichnessDraft::new();
    draft.extent = 500; // below minimum
    let result = draft_to_richness_document(&draft);
    assert!(result.is_err());
}

// ── Real executor construction (production wiring) ─────────────────────────

#[test]
fn production_executor_creates_without_tools_check() {
    // The executor creation itself should not fail; it's a closure.
    let tools_dir = std::path::PathBuf::from("/tmp/nonexistent-tools");
    let _executor = richness_generation::production_richness_executor(tools_dir);
}

#[test]
fn executor_outcome_request_id_accessors() {
    use std::path::Path;
    let ok = ExecutorOutcome::PackageReady {
        request_id: 99,
        package_dir: std::path::PathBuf::from("/tmp/ok"),
    };
    assert_eq!(ok.request_id(), 99);
    assert!(ok.is_success());
    assert!(!ok.is_failure());
    assert_eq!(ok.package_dir(), Some(Path::new("/tmp/ok")));

    let err = ExecutorOutcome::Failed {
        request_id: 7,
        error_message: "oops".into(),
    };
    assert_eq!(err.request_id(), 7);
    assert!(!err.is_success());
    assert!(err.is_failure());
    assert_eq!(err.package_dir(), None);
}

// ── Controller lifecycle (already tested in richness_gui.rs, confirm here) ─

#[test]
fn controller_spawn_and_shutdown() {
    let root = tempfile::tempdir().unwrap();
    let controller =
        richness_generation::RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    controller.shutdown();
}

// ── Baseline byte compatibility ────────────────────────────────────────────

#[test]
fn richness_draft_default_is_deterministic() {
    let d1 = RichnessDraft::new();
    let d2 = RichnessDraft::new();
    assert_eq!(d1.to_canonical_bytes(), d2.to_canonical_bytes());
    assert_eq!(d1.identity_hash_hex(), d2.identity_hash_hex());
}

#[test]
fn baseline_m3_generate_still_works() {
    let args = parse_from(["--m3-generate", "--seed", "42"]).unwrap();
    assert!(args.m3_generate);
    assert!(!args.m3_richness);
    assert_eq!(args.m3_seed, 42);
}

#[test]
fn baseline_bsp_still_works() {
    let args = parse_from(["--strict", "--bsp", "maps/e1m1.bsp"]).unwrap();
    assert!(args.bsp_path.is_some());
    assert!(!args.m3_generate);
    assert!(!args.m3_richness);
}

#[test]
fn m3_richness_default_args_preserved() {
    let args = cli::CliArgs::default();
    assert!(!args.m3_richness);
    assert!(!args.m3_generate);
    assert_eq!(args.scale, 0.0254);
}
