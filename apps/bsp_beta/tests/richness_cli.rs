//! CLI parser tests for the registered Richness V1 launch token parser.
//!
//! These tests verify that `parse_richness_launch_token` correctly parses
//! every RichnessDraft control, that the primary parser enforces the exact
//! mode gate, and that baseline CLI behavior remains compatible.

// Include cli.rs directly to access the unregistered parser + all types.
#[path = "../src/cli.rs"]
mod cli;

// Include richness_gui.rs for draft application.
#[path = "../src/richness_gui.rs"]
mod richness_gui;

use cli::{parse_from, parse_richness_launch_token, CliError, RichnessLaunchToken};
use richness_gui::{
    InheritedOr, RichnessCaveMode, RichnessDraft, RichnessFieldId, RichnessPacing, RichnessPreset,
    RichnessTheme, RichnessVariation,
};
use std::path::PathBuf;

// ── Helper: apply parsed token to a RichnessDraft ─────────────────────────

fn apply_token_to_draft(
    token: &RichnessLaunchToken,
    draft: &mut RichnessDraft,
) -> Result<(), String> {
    if let Some(ref tag) = token.richness_preset {
        let p = RichnessPreset::from_tag(tag)
            .ok_or_else(|| format!("unknown richness preset '{tag}'"))?;
        draft.set_preset(p);
    }
    if let Some(ref tag) = token.richness_theme {
        let t = RichnessTheme::from_tag(tag)
            .ok_or_else(|| format!("unknown richness theme '{tag}'"))?;
        draft.set_theme(t);
    }
    if let Some(v) = token.richness_extent {
        draft.try_set_extent(v)?;
    }
    if let Some(v) = token.richness_seed {
        draft.set_seed(v);
    }

    // InheritedOr<u32> fields
    if token.richness_landmarks_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::Landmarks, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_landmarks {
        draft.try_set_explicit_u32(RichnessFieldId::Landmarks, v)?;
    }

    if token.richness_zones_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::Zones, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_zones {
        draft.try_set_explicit_u32(RichnessFieldId::Zones, v)?;
    }

    if token.richness_vertical_openings_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::VerticalOpenings, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_vertical_openings {
        draft.try_set_explicit_u32(RichnessFieldId::VerticalOpenings, v)?;
    }

    if token.richness_budget_ceiling_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::BudgetCeiling, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_budget_ceiling {
        draft.try_set_explicit_u32(RichnessFieldId::BudgetCeiling, v)?;
    }

    if token.richness_prop_density_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::PropDensity, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_prop_density {
        draft.try_set_explicit_u32(RichnessFieldId::PropDensity, v)?;
    }

    if token.richness_light_density_inherited {
        draft.try_set_inherited_u32(RichnessFieldId::LightDensity, InheritedOr::Inherited)?;
    } else if let Some(v) = token.richness_light_density {
        draft.try_set_explicit_u32(RichnessFieldId::LightDensity, v)?;
    }

    // Cave mode
    if token.richness_cave_mode_inherited {
        draft.try_set_cave_mode(InheritedOr::Inherited)?;
    } else if let Some(ref tag) = token.richness_cave_mode {
        let m = RichnessCaveMode::from_tag(tag)
            .ok_or_else(|| format!("unknown richness cave mode '{tag}'"))?;
        draft.try_set_cave_mode(InheritedOr::Explicit(m))?;
    }

    // Pacing
    if token.richness_pacing_inherited {
        draft.try_set_pacing(InheritedOr::Inherited)?;
    } else if let Some(ref tag) = token.richness_pacing {
        let p = RichnessPacing::from_tag(tag)
            .ok_or_else(|| format!("unknown richness pacing '{tag}'"))?;
        draft.try_set_pacing(InheritedOr::Explicit(p))?;
    }

    // Variation
    if token.richness_variation_inherited {
        draft.try_set_variation(InheritedOr::Inherited)?;
    } else if let Some(ref tag) = token.richness_variation {
        let v = RichnessVariation::from_tag(tag)
            .ok_or_else(|| format!("unknown richness variation '{tag}'"))?;
        draft.try_set_variation(InheritedOr::Explicit(v))?;
    }

    Ok(())
}

// ── Baseline CLI tests (unchanged) ────────────────────────────────────────

#[test]
fn baseline_default_args_unchanged() {
    let args = parse_from(Vec::<&str>::new()).unwrap();
    assert!(args.bsp_path.is_none());
    assert!(!args.headless);
    assert_eq!(args.capture_frames, 0);
    assert!((args.scale - 0.0254).abs() < 1e-6);
}

#[test]
fn baseline_bsp_parse_unchanged() {
    let args = parse_from(["--strict", "--bsp", "maps/test.bsp"]).unwrap();
    assert_eq!(args.bsp_path, Some(PathBuf::from("maps/test.bsp")));
}

#[test]
fn baseline_m3_generate_parse_unchanged() {
    let args = parse_from(["--m3-generate", "--seed", "42", "--preset", "rich"]).unwrap();
    assert!(args.m3_generate);
    assert_eq!(args.m3_seed, 42);
}

#[test]
fn registered_richness_gate_and_help_text() {
    let args = parse_from([
        "--m3-richness-v1",
        "--richness-preset",
        "sparse",
        "--richness-theme",
        "ancient",
    ])
    .unwrap();
    assert!(args.m3_richness);
    assert_eq!(args.import_mode, Some(cli::ImportMode::Strict));

    let usage = cli::usage_text();
    for flag in [
        "--m3-richness-v1",
        "--richness-preset",
        "--richness-theme",
        "--richness-seed",
        "--richness-budget-ceiling",
    ] {
        assert!(usage.contains(flag), "registered help missing {flag}");
    }
}

#[test]
fn baseline_m3_conflicts_unchanged() {
    let err = parse_from(["--m3-generate", "--bsp", "maps/test.bsp"]).unwrap_err();
    assert_eq!(err, CliError::M3GenerateBspConflict);
}

// ── Richness launch token: empty ───────────────────────────────────────────

#[test]
fn empty_token_is_default() {
    let token = parse_richness_launch_token(Vec::<&str>::new()).unwrap();
    assert!(token.is_empty());
    assert!(token.richness_preset.is_none());
    assert!(token.richness_theme.is_none());
    assert!(token.richness_extent.is_none());
    assert!(token.richness_seed.is_none());
    assert!(token.richness_pacing.is_none());
    assert!(token.richness_landmarks.is_none());
    assert!(token.richness_zones.is_none());
    assert!(token.richness_cave_mode.is_none());
    assert!(token.richness_vertical_openings.is_none());
    assert!(token.richness_variation.is_none());
    assert!(token.richness_prop_density.is_none());
    assert!(token.richness_light_density.is_none());
    assert!(token.richness_budget_ceiling.is_none());
    assert!(!token.richness_landmarks_inherited);
    assert!(!token.richness_zones_inherited);
}

// ── Parsing individual fields ──────────────────────────────────────────────

#[test]
fn parse_preset() {
    let token = parse_richness_launch_token(["--richness-preset", "moderate"]).unwrap();
    assert_eq!(token.richness_preset.as_deref(), Some("moderate"));
}

#[test]
fn parse_theme() {
    let token = parse_richness_launch_token(["--richness-theme", "egyptian"]).unwrap();
    assert_eq!(token.richness_theme.as_deref(), Some("egyptian"));
}

#[test]
fn parse_extent() {
    let token = parse_richness_launch_token(["--richness-extent", "2048"]).unwrap();
    assert_eq!(token.richness_extent, Some(2048));
}

#[test]
fn parse_seed() {
    let token = parse_richness_launch_token(["--richness-seed", "123456789"]).unwrap();
    assert_eq!(token.richness_seed, Some(123456789));
}

#[test]
fn parse_pacing() {
    let token = parse_richness_launch_token(["--richness-pacing", "intense"]).unwrap();
    assert_eq!(token.richness_pacing.as_deref(), Some("intense"));
}

#[test]
fn parse_landmarks() {
    let token = parse_richness_launch_token(["--richness-landmarks", "3"]).unwrap();
    assert_eq!(token.richness_landmarks, Some(3));
}

#[test]
fn parse_zones() {
    let token = parse_richness_launch_token(["--richness-zones", "2"]).unwrap();
    assert_eq!(token.richness_zones, Some(2));
}

#[test]
fn parse_cave_mode() {
    let token = parse_richness_launch_token(["--richness-cave-mode", "required"]).unwrap();
    assert_eq!(token.richness_cave_mode.as_deref(), Some("required"));
}

#[test]
fn parse_vertical_openings() {
    let token = parse_richness_launch_token(["--richness-vertical-openings", "4"]).unwrap();
    assert_eq!(token.richness_vertical_openings, Some(4));
}

#[test]
fn parse_variation() {
    let token = parse_richness_launch_token(["--richness-variation", "wild"]).unwrap();
    assert_eq!(token.richness_variation.as_deref(), Some("wild"));
}

#[test]
fn parse_prop_density() {
    let token = parse_richness_launch_token(["--richness-prop-density", "75"]).unwrap();
    assert_eq!(token.richness_prop_density, Some(75));
}

#[test]
fn parse_light_density() {
    let token = parse_richness_launch_token(["--richness-light-density", "80"]).unwrap();
    assert_eq!(token.richness_light_density, Some(80));
}

#[test]
fn parse_budget_ceiling() {
    let token = parse_richness_launch_token(["--richness-budget-ceiling", "5000"]).unwrap();
    assert_eq!(token.richness_budget_ceiling, Some(5000));
}

// ── Parsing inherited flags ────────────────────────────────────────────────

#[test]
fn parse_landmarks_inherited() {
    let token = parse_richness_launch_token(["--richness-landmarks-inherited"]).unwrap();
    assert!(token.richness_landmarks_inherited);
    assert!(token.richness_landmarks.is_none());
}

#[test]
fn parse_zones_inherited() {
    let token = parse_richness_launch_token(["--richness-zones-inherited"]).unwrap();
    assert!(token.richness_zones_inherited);
}

#[test]
fn parse_cave_mode_inherited() {
    let token = parse_richness_launch_token(["--richness-cave-mode-inherited"]).unwrap();
    assert!(token.richness_cave_mode_inherited);
}

#[test]
fn parse_vertical_openings_inherited() {
    let token = parse_richness_launch_token(["--richness-vertical-openings-inherited"]).unwrap();
    assert!(token.richness_vertical_openings_inherited);
}

#[test]
fn parse_budget_ceiling_inherited() {
    let token = parse_richness_launch_token(["--richness-budget-ceiling-inherited"]).unwrap();
    assert!(token.richness_budget_ceiling_inherited);
}

#[test]
fn parse_pacing_inherited() {
    let token = parse_richness_launch_token(["--richness-pacing-inherited"]).unwrap();
    assert!(token.richness_pacing_inherited);
}

#[test]
fn parse_variation_inherited() {
    let token = parse_richness_launch_token(["--richness-variation-inherited"]).unwrap();
    assert!(token.richness_variation_inherited);
}

#[test]
fn parse_prop_density_inherited() {
    let token = parse_richness_launch_token(["--richness-prop-density-inherited"]).unwrap();
    assert!(token.richness_prop_density_inherited);
}

#[test]
fn parse_light_density_inherited() {
    let token = parse_richness_launch_token(["--richness-light-density-inherited"]).unwrap();
    assert!(token.richness_light_density_inherited);
}

// ── Combined parsing ───────────────────────────────────────────────────────

#[test]
fn parse_all_fields_combined() {
    let token = parse_richness_launch_token([
        "--richness-preset",
        "rich",
        "--richness-theme",
        "brutalist",
        "--richness-extent",
        "3072",
        "--richness-seed",
        "99",
        "--richness-pacing",
        "intense",
        "--richness-landmarks",
        "3",
        "--richness-zones",
        "2",
        "--richness-cave-mode",
        "preferred",
        "--richness-vertical-openings",
        "4",
        "--richness-variation",
        "wild",
        "--richness-prop-density",
        "80",
        "--richness-light-density",
        "60",
        "--richness-budget-ceiling",
        "8000",
    ])
    .unwrap();
    assert_eq!(token.richness_preset.as_deref(), Some("rich"));
    assert_eq!(token.richness_theme.as_deref(), Some("brutalist"));
    assert_eq!(token.richness_extent, Some(3072));
    assert_eq!(token.richness_seed, Some(99));
    assert_eq!(token.richness_pacing.as_deref(), Some("intense"));
    assert_eq!(token.richness_landmarks, Some(3));
    assert_eq!(token.richness_zones, Some(2));
    assert_eq!(token.richness_cave_mode.as_deref(), Some("preferred"));
    assert_eq!(token.richness_vertical_openings, Some(4));
    assert_eq!(token.richness_variation.as_deref(), Some("wild"));
    assert_eq!(token.richness_prop_density, Some(80));
    assert_eq!(token.richness_light_density, Some(60));
    assert_eq!(token.richness_budget_ceiling, Some(8000));
}

#[test]
fn parse_all_inherited_flags() {
    let token = parse_richness_launch_token([
        "--richness-landmarks-inherited",
        "--richness-zones-inherited",
        "--richness-cave-mode-inherited",
        "--richness-vertical-openings-inherited",
        "--richness-budget-ceiling-inherited",
        "--richness-pacing-inherited",
        "--richness-variation-inherited",
        "--richness-prop-density-inherited",
        "--richness-light-density-inherited",
    ])
    .unwrap();
    assert!(token.richness_landmarks_inherited);
    assert!(token.richness_zones_inherited);
    assert!(token.richness_cave_mode_inherited);
    assert!(token.richness_vertical_openings_inherited);
    assert!(token.richness_budget_ceiling_inherited);
    assert!(token.richness_pacing_inherited);
    assert!(token.richness_variation_inherited);
    assert!(token.richness_prop_density_inherited);
    assert!(token.richness_light_density_inherited);
}

// ── Invalid values ─────────────────────────────────────────────────────────

#[test]
fn reject_invalid_extent() {
    let err = parse_richness_launch_token(["--richness-extent", "not-a-number"]).unwrap_err();
    assert!(matches!(
        err,
        CliError::InvalidM3Value {
            flag: "--richness-extent",
            ..
        }
    ));
}

#[test]
fn reject_invalid_seed() {
    let err = parse_richness_launch_token(["--richness-seed", "-1"]).unwrap_err();
    assert!(matches!(
        err,
        CliError::InvalidM3Value {
            flag: "--richness-seed",
            ..
        }
    ));
}

#[test]
fn reject_unknown_flag() {
    let err = parse_richness_launch_token(["--richness-unknown"]).unwrap_err();
    assert!(matches!(err, CliError::UnknownArgument(_)));
}

#[test]
fn reject_missing_value() {
    let err = parse_richness_launch_token(["--richness-preset"]).unwrap_err();
    assert_eq!(err, CliError::MissingValue("--richness-preset"));
}

#[test]
fn reject_value_starting_with_dashes() {
    let err = parse_richness_launch_token(["--richness-preset", "--other-flag"]).unwrap_err();
    assert_eq!(err, CliError::MissingValue("--richness-preset"));
}

// ── Apply to draft: explicit values ────────────────────────────────────────

#[test]
fn apply_explicit_values_to_draft() {
    let token = parse_richness_launch_token([
        "--richness-preset",
        "rich",
        "--richness-theme",
        "brutalist",
        "--richness-extent",
        "3072",
        "--richness-seed",
        "42",
        "--richness-landmarks",
        "3",
        "--richness-zones",
        "2",
        "--richness-cave-mode",
        "required",
        "--richness-vertical-openings",
        "4",
        "--richness-budget-ceiling",
        "8000",
        "--richness-pacing",
        "intense",
        "--richness-variation",
        "wild",
        "--richness-prop-density",
        "75",
        "--richness-light-density",
        "60",
    ])
    .unwrap();

    let mut draft = RichnessDraft::new();
    apply_token_to_draft(&token, &mut draft).unwrap();

    assert_eq!(draft.preset, RichnessPreset::Rich);
    assert_eq!(draft.theme, RichnessTheme::Brutalist);
    assert_eq!(draft.extent, 3072);
    assert_eq!(draft.seed, 42);
    assert_eq!(draft.landmarks, InheritedOr::Explicit(3));
    assert_eq!(draft.zones, InheritedOr::Explicit(2));
    assert_eq!(
        draft.cave_mode,
        InheritedOr::Explicit(RichnessCaveMode::Required)
    );
    assert_eq!(draft.vertical_openings, InheritedOr::Explicit(4));
    assert_eq!(draft.budget_ceiling, InheritedOr::Explicit(8000));
    assert_eq!(draft.pacing, InheritedOr::Explicit(RichnessPacing::Intense));
    assert_eq!(
        draft.variation,
        InheritedOr::Explicit(RichnessVariation::Wild)
    );
    assert_eq!(draft.prop_density, InheritedOr::Explicit(75));
    assert_eq!(draft.light_density, InheritedOr::Explicit(60));
}

// ── Apply to draft: inherited flags ────────────────────────────────────────

#[test]
fn apply_inherited_flags_to_draft() {
    let mut draft = RichnessDraft::new();
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::Zones, 2)
        .unwrap();
    draft
        .try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Required))
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 5000)
        .unwrap();
    assert!(draft.landmarks.is_explicit());
    assert!(draft.zones.is_explicit());

    let token = parse_richness_launch_token([
        "--richness-landmarks-inherited",
        "--richness-zones-inherited",
        "--richness-cave-mode-inherited",
        "--richness-budget-ceiling-inherited",
    ])
    .unwrap();

    apply_token_to_draft(&token, &mut draft).unwrap();
    assert!(draft.landmarks.is_inherited());
    assert!(draft.zones.is_inherited());
    assert!(draft.cave_mode.is_inherited());
    assert!(draft.budget_ceiling.is_inherited());
}

// ── Apply to draft: round-trip preservation ────────────────────────────────

#[test]
fn apply_preserves_preset_and_theme() {
    let token = parse_richness_launch_token([
        "--richness-preset",
        "moderate",
        "--richness-theme",
        "egyptian",
    ])
    .unwrap();
    let mut draft = RichnessDraft::new();
    apply_token_to_draft(&token, &mut draft).unwrap();
    assert_eq!(draft.preset, RichnessPreset::Moderate);
    assert_eq!(draft.theme, RichnessTheme::Egyptian);
}

#[test]
fn apply_is_idempotent_for_same_token() {
    let token =
        parse_richness_launch_token(["--richness-preset", "rich", "--richness-landmarks", "3"])
            .unwrap();
    let mut draft = RichnessDraft::new();
    apply_token_to_draft(&token, &mut draft).unwrap();
    let after_first = draft.clone();
    apply_token_to_draft(&token, &mut draft).unwrap();
    assert_eq!(draft, after_first);
}

#[test]
fn apply_rejects_invalid_tags() {
    let token = parse_richness_launch_token(["--richness-preset", "unknown"]).unwrap();
    let mut draft = RichnessDraft::new();
    let err = apply_token_to_draft(&token, &mut draft).unwrap_err();
    assert!(err.contains("unknown richness preset"));
}

#[test]
fn apply_rejects_out_of_range_values() {
    let token = parse_richness_launch_token(["--richness-extent", "500"]).unwrap();
    let mut draft = RichnessDraft::new();
    let err = apply_token_to_draft(&token, &mut draft).unwrap_err();
    assert!(err.contains("Extent"));
}

// ── Inherited vs explicit preservation ─────────────────────────────────────

#[test]
fn inherited_flag_overrides_explicit_value() {
    let token = parse_richness_launch_token([
        "--richness-landmarks",
        "3",
        "--richness-landmarks-inherited",
    ])
    .unwrap();

    let mut draft = RichnessDraft::new();
    apply_token_to_draft(&token, &mut draft).unwrap();
    // The order in apply_token_to_draft processes inherited after explicit
    assert!(draft.landmarks.is_inherited());
}

// ── Baseline byte compatibility with CLI ───────────────────────────────────

#[test]
fn richness_flags_require_exact_mode_gate() {
    for &flag in &[
        "--richness-preset",
        "--richness-theme",
        "--richness-extent",
        "--richness-seed",
        "--richness-pacing",
        "--richness-landmarks",
        "--richness-zones",
        "--richness-cave-mode",
        "--richness-vertical-openings",
        "--richness-variation",
        "--richness-prop-density",
        "--richness-light-density",
        "--richness-budget-ceiling",
    ] {
        let error = parse_from(["--strict", flag, "1"]).unwrap_err();
        assert_eq!(
            error,
            CliError::RichnessOptionRequiresRichness(flag),
            "wrong gate error for {flag}"
        );
    }
}

#[test]
fn parse_from_baseline_passes_without_richness() {
    let args = parse_from([
        "--strict",
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
fn richness_launch_token_default_is_empty() {
    let token = parse_richness_launch_token(Vec::<&str>::new()).unwrap();
    assert!(token.is_empty());
    assert!(!token.richness_landmarks_inherited);
    assert_eq!(token.richness_preset, None);
}

#[test]
fn parse_richness_token_does_not_affect_cli_args() {
    let token =
        parse_richness_launch_token(["--richness-preset", "sparse", "--richness-seed", "99"])
            .unwrap();
    assert_eq!(token.richness_preset.as_deref(), Some("sparse"));
    assert_eq!(token.richness_seed, Some(99));

    let args = parse_from(["--m3-generate", "--seed", "42"]).unwrap();
    assert_eq!(args.m3_seed, 42);
    assert!(args.m3_generate);
}

// ── Field inventory completeness ───────────────────────────────────────────

#[test]
fn all_richness_fields_have_parser_support() {
    let token = parse_richness_launch_token([
        "--richness-preset",
        "sparse",
        "--richness-theme",
        "ancient",
        "--richness-extent",
        "2048",
        "--richness-seed",
        "0",
        "--richness-pacing",
        "normal",
        "--richness-landmarks",
        "1",
        "--richness-zones",
        "1",
        "--richness-cave-mode",
        "preferred",
        "--richness-vertical-openings",
        "0",
        "--richness-variation",
        "moderate",
        "--richness-prop-density",
        "50",
        "--richness-light-density",
        "50",
        "--richness-budget-ceiling",
        "3000",
    ])
    .unwrap();

    assert!(token.richness_preset.is_some());
    assert!(token.richness_theme.is_some());
    assert!(token.richness_extent.is_some());
    assert!(token.richness_seed.is_some());
    assert!(token.richness_pacing.is_some());
    assert!(token.richness_landmarks.is_some());
    assert!(token.richness_zones.is_some());
    assert!(token.richness_cave_mode.is_some());
    assert!(token.richness_vertical_openings.is_some());
    assert!(token.richness_variation.is_some());
    assert!(token.richness_prop_density.is_some());
    assert!(token.richness_light_density.is_some());
    assert!(token.richness_budget_ceiling.is_some());
}
