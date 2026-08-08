//! GUI and controller integration tests for Richness V1.
//!
//! Verifies the full contract: field inventory, inherited/explicit preservation,
//! reset semantics, keyboard/mouse parity, scaling, scrolling, input capture,
//! stale/latest results from the generation controller, worker shutdown,
//! active-world preservation on failure, and baseline CLI/GUI byte compatibility.

// Include source modules directly so we can access private internals.
#[path = "../src/richness_gui.rs"]
mod richness_gui;

#[path = "../src/richness_generation.rs"]
mod richness_generation;

use richness_generation::{
    package_dir_for_richness_request, ExecutorOutcome, GenerationExecutor,
    RichnessGenerationController, RichnessGenerationRequest,
};
use richness_gui::{
    DrawItem, InheritedOr, RichnessCaveMode, RichnessDraft, RichnessFieldId, RichnessGui,
    RichnessGuiAction, RichnessGuiMode, RichnessInputAction, RichnessPacing, RichnessPreset,
    RichnessTheme, RichnessVariation, BUDGET_CEILING_MAX, BUDGET_CEILING_MIN, LANDMARKS_MAX,
    LANDMARKS_MIN, LIGHT_DENSITY_MAX, LIGHT_DENSITY_MIN, PROP_DENSITY_MAX, PROP_DENSITY_MIN,
    RICHNESS_EXTENT_MAX, RICHNESS_EXTENT_MIN, RICHNESS_QUANTUM, UI_PREFERENCES_HEADER,
    VERTICAL_FEATURES_MAX, VERTICAL_FEATURES_MIN, ZONES_MAX, ZONES_MIN,
};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

use std::thread;
use std::time::Duration;

// ── Helpers ────────────────────────────────────────────────────────────────

fn press() -> RichnessInputAction {
    RichnessInputAction::Press
}

fn item_index_for(fid: RichnessFieldId) -> usize {
    RichnessFieldId::ALL.iter().position(|f| *f == fid).unwrap()
}

fn failing_executor(msg: &'static str) -> GenerationExecutor {
    let msg = msg.to_string();
    Box::new(
        move |req: &RichnessGenerationRequest| ExecutorOutcome::Failed {
            request_id: req.id,
            error_message: msg.clone(),
        },
    )
}

// ── Draft: field inventory ─────────────────────────────────────────────────

#[test]
fn all_fields_indexed() {
    assert_eq!(RichnessFieldId::COUNT, 13);
    assert_eq!(RichnessFieldId::ALL.len(), 13);
}

#[test]
fn every_field_has_label_and_tooltip() {
    for id in RichnessFieldId::ALL {
        assert!(!id.label().is_empty(), "{id:?} has no label");
        assert!(!id.tooltip().is_empty(), "{id:?} has no tooltip");
    }
}

#[test]
fn canonical_fields_are_identifiable() {
    let canonical: Vec<_> = RichnessFieldId::ALL
        .iter()
        .copied()
        .filter(|f| f.is_canonical())
        .collect();
    assert_eq!(canonical.len(), 9);
}

#[test]
fn every_field_has_kind() {
    for id in RichnessFieldId::ALL {
        let _kind = id.kind(); // must not panic
    }
}

// ── Draft: inherited / explicit preservation ───────────────────────────────

#[test]
fn new_draft_all_inherited() {
    let draft = RichnessDraft::new();
    assert!(draft.landmarks.is_inherited());
    assert!(draft.zones.is_inherited());
    assert!(draft.cave_mode.is_inherited());
    assert!(draft.vertical_openings.is_inherited());
    assert!(draft.budget_ceiling.is_inherited());
    assert!(draft.pacing.is_inherited());
    assert!(draft.variation.is_inherited());
    assert!(draft.prop_density.is_inherited());
    assert!(draft.light_density.is_inherited());
}

#[test]
fn explicit_values_preserved() {
    let mut draft = RichnessDraft::new();
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
        .unwrap();
    assert_eq!(draft.landmarks, InheritedOr::Explicit(3));
    assert!(draft.landmarks.is_explicit());
}

#[test]
fn inherited_state_preserved_on_invalid_set() {
    let mut draft = RichnessDraft::new();
    let before = draft.landmarks;
    assert!(draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, u32::MAX)
        .is_err());
    assert_eq!(draft.landmarks, before);
}

#[test]
fn reset_field_to_inherited_works() {
    let mut draft = RichnessDraft::new();
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
        .unwrap();
    assert!(draft.landmarks.is_explicit());
    draft.reset_field_to_inherited(RichnessFieldId::Landmarks);
    assert!(draft.landmarks.is_inherited());
}

#[test]
fn reset_all_to_inherited_works() {
    let mut draft = RichnessDraft::new();
    // Use valid values within each field's range
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 2)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::Zones, 2)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 2000)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::PropDensity, 50)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::LightDensity, 50)
        .unwrap();
    assert!(draft
        .get_inherited_u32(RichnessFieldId::Landmarks)
        .is_explicit());
    assert!(draft
        .get_inherited_u32(RichnessFieldId::BudgetCeiling)
        .is_explicit());
    draft.reset_all_to_inherited();
    assert!(draft
        .get_inherited_u32(RichnessFieldId::Landmarks)
        .is_inherited());
    assert!(draft
        .get_inherited_u32(RichnessFieldId::Zones)
        .is_inherited());
    assert!(draft
        .get_inherited_u32(RichnessFieldId::BudgetCeiling)
        .is_inherited());
    assert!(draft
        .get_inherited_u32(RichnessFieldId::PropDensity)
        .is_inherited());
    assert!(draft
        .get_inherited_u32(RichnessFieldId::LightDensity)
        .is_inherited());
}

#[test]
fn reset_to_defaults_restores_factory_state() {
    let mut draft = RichnessDraft::new();
    draft.set_preset(RichnessPreset::Rich);
    draft.set_theme(RichnessTheme::Brutalist);
    draft.set_seed(42);
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 5)
        .unwrap();
    draft.reset_to_defaults();
    let default = RichnessDraft::new();
    assert_eq!(draft.preset, default.preset);
    assert_eq!(draft.theme, default.theme);
    assert_eq!(draft.seed, default.seed);
    assert_eq!(draft.landmarks, default.landmarks);
}

// ── Draft: invalid prevention ──────────────────────────────────────────────

#[test]
fn reject_extent_out_of_range() {
    let mut draft = RichnessDraft::new();
    assert!(draft.try_set_extent(500).is_err());
    assert!(draft.try_set_extent(5000).is_err());
}

#[test]
fn reject_extent_not_quantum_aligned() {
    let mut draft = RichnessDraft::new();
    assert!(draft.try_set_extent(1025).is_err());
}

#[test]
fn reject_landmarks_out_of_range() {
    let mut draft = RichnessDraft::new();
    assert!(draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, LANDMARKS_MAX + 1)
        .is_err());
}

#[test]
fn reject_zones_out_of_range() {
    let mut draft = RichnessDraft::new();
    assert!(draft
        .try_set_explicit_u32(RichnessFieldId::Zones, ZONES_MAX + 1)
        .is_err());
}

#[test]
fn reject_budget_out_of_range() {
    let mut draft = RichnessDraft::new();
    assert!(draft
        .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, BUDGET_CEILING_MIN - 1)
        .is_err());
    assert!(draft
        .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, BUDGET_CEILING_MAX + 1)
        .is_err());
}

#[test]
fn reject_cave_required_with_insufficient_landmarks() {
    let mut draft = RichnessDraft::new();
    assert!(draft
        .try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Required))
        .is_err());
}

#[test]
fn reject_cave_required_with_small_extent() {
    let mut draft = RichnessDraft::new();
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
        .unwrap();
    draft.try_set_extent(1024).unwrap();
    assert!(draft
        .try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Required))
        .is_err());
}

#[test]
fn reject_prop_density_out_of_range() {
    let mut draft = RichnessDraft::new();
    assert!(draft
        .try_set_explicit_u32(RichnessFieldId::PropDensity, PROP_DENSITY_MAX + 1)
        .is_err());
}

#[test]
fn reject_light_density_out_of_range() {
    let mut draft = RichnessDraft::new();
    assert!(draft
        .try_set_explicit_u32(RichnessFieldId::LightDensity, LIGHT_DENSITY_MAX + 1)
        .is_err());
}

#[test]
fn reject_extent_below_2048_when_cave_required_active() {
    let mut draft = RichnessDraft::new();
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
        .unwrap();
    draft
        .try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Required))
        .unwrap();
    assert!(draft.try_set_extent(1024).is_err());
}

// ── Draft: canonical conversion round trip ─────────────────────────────────

#[test]
fn canonical_round_trip_preserves_all_fields() {
    let mut draft = RichnessDraft::new();
    draft.set_preset(RichnessPreset::Moderate);
    draft.set_theme(RichnessTheme::Egyptian);
    draft.try_set_extent(2048).unwrap();
    draft.set_seed(12345);
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 2)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::Zones, 1)
        .unwrap();
    draft
        .try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Preferred))
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::VerticalOpenings, 3)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 4000)
        .unwrap();

    let bytes = draft.to_canonical_bytes();
    let restored = RichnessDraft::from_canonical_bytes(&bytes).unwrap();

    assert_eq!(restored.preset, draft.preset);
    assert_eq!(restored.theme, draft.theme);
    assert_eq!(restored.extent, draft.extent);
    assert_eq!(restored.seed, draft.seed);
    assert_eq!(restored.landmarks, draft.landmarks);
    assert_eq!(restored.zones, draft.zones);
    assert_eq!(restored.cave_mode, draft.cave_mode);
    assert_eq!(restored.vertical_openings, draft.vertical_openings);
    assert_eq!(restored.budget_ceiling, draft.budget_ceiling);
    assert_eq!(restored.to_canonical_bytes(), bytes);
}

#[test]
fn canonical_round_trip_with_inherited_fields() {
    let draft = RichnessDraft::new();
    let bytes = draft.to_canonical_bytes();
    let restored = RichnessDraft::from_canonical_bytes(&bytes).unwrap();
    assert!(restored.landmarks.is_inherited());
    assert!(restored.zones.is_inherited());
}

#[test]
fn canonical_rejects_unknown_gate() {
    let bytes = RichnessDraft::new().to_canonical_bytes();
    let text = String::from_utf8(bytes).unwrap();
    let corrupted = text.replace("richness-v1", "unsupported-gate");
    let result = RichnessDraft::from_canonical_bytes(corrupted.as_bytes());
    assert!(result.is_err());
}

#[test]
fn canonical_roundtrip_is_byte_identical_for_every_inherited_explicit_combo() {
    // The four direct canonical fields are set explicitly below. The five
    // provenance-bearing canonical controls exercise all 2^5 combinations.
    for mask in 0_u8..32 {
        let mut draft = RichnessDraft::new();
        draft.set_preset(RichnessPreset::Rich);
        draft.set_theme(RichnessTheme::Egyptian);
        draft.try_set_extent(3072).unwrap();
        draft.set_seed(42);
        if mask & 1 != 0 {
            draft
                .try_set_explicit_u32(RichnessFieldId::Landmarks, 3)
                .unwrap();
        }
        if mask & 2 != 0 {
            draft
                .try_set_explicit_u32(RichnessFieldId::Zones, 2)
                .unwrap();
        }
        if mask & 4 != 0 {
            draft
                .try_set_cave_mode(InheritedOr::Explicit(RichnessCaveMode::Required))
                .unwrap();
        }
        if mask & 8 != 0 {
            draft
                .try_set_explicit_u32(RichnessFieldId::VerticalOpenings, 4)
                .unwrap();
        }
        if mask & 16 != 0 {
            draft
                .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 8000)
                .unwrap();
        }

        let canonical = draft.to_canonical_bytes();
        let restored = RichnessDraft::from_canonical_bytes(&canonical).unwrap();
        assert_eq!(restored.to_canonical_bytes(), canonical, "mask {mask:05b}");
    }
}

#[test]
fn ui_preferences_are_labeled_roundtrip_and_excluded_from_canonical_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let canonical_path = dir.path().join("request.richness");
    let mut draft = RichnessDraft::new();
    draft
        .try_set_pacing(InheritedOr::Explicit(RichnessPacing::Intense))
        .unwrap();
    draft
        .try_set_variation(InheritedOr::Explicit(RichnessVariation::Wild))
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::PropDensity, 80)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::LightDensity, 20)
        .unwrap();

    let canonical = draft.to_canonical_bytes();
    let companion = draft
        .save_canonical_and_ui_preferences(&canonical_path)
        .unwrap();
    assert_eq!(std::fs::read(&canonical_path).unwrap(), canonical);
    assert_eq!(
        companion,
        RichnessDraft::ui_preferences_path(&canonical_path)
    );
    let preferences = std::fs::read(&companion).unwrap();
    assert!(preferences.starts_with(UI_PREFERENCES_HEADER.as_bytes()));
    assert_eq!(preferences, draft.ui_preferences().to_toml_bytes());
    let canonical_text = std::str::from_utf8(&canonical).unwrap();
    for field in ["pacing", "variation", "prop_density", "light_density"] {
        assert!(
            !canonical_text.contains(field),
            "UI preference '{field}' leaked into canonical request"
        );
    }

    let canonical_only = RichnessDraft::from_canonical_bytes(&canonical).unwrap();
    assert!(canonical_only.pacing.is_inherited());
    assert!(canonical_only.variation.is_inherited());
    assert!(canonical_only.prop_density.is_inherited());
    assert!(canonical_only.light_density.is_inherited());

    let loaded = RichnessDraft::load_canonical_and_ui_preferences(&canonical_path).unwrap();
    assert_eq!(loaded.pacing, draft.pacing);
    assert_eq!(loaded.variation, draft.variation);
    assert_eq!(loaded.prop_density, draft.prop_density);
    assert_eq!(loaded.light_density, draft.light_density);

    let mut gui = RichnessGui::new();
    gui.draft = canonical_only;
    let display = gui.text_render();
    for field in [
        RichnessFieldId::Pacing,
        RichnessFieldId::Variation,
        RichnessFieldId::PropDensity,
        RichnessFieldId::LightDensity,
    ] {
        assert_eq!(field.provenance_badge(), Some("UI preference"));
        assert!(display.contains(field.label()));
    }
    assert!(display.contains("[UI preference]"));
}

// ── Draft: identity hash ───────────────────────────────────────────────────

#[test]
fn identity_hash_is_deterministic() {
    let draft = RichnessDraft::new();
    let h1 = draft.identity_hash_hex();
    let h2 = draft.identity_hash_hex();
    assert_eq!(h1, h2);
}

#[test]
fn identity_hash_changes_with_seed() {
    let mut draft = RichnessDraft::new();
    let h1 = draft.identity_hash_hex();
    draft.set_seed(99);
    let h2 = draft.identity_hash_hex();
    assert_ne!(h1, h2);
}

#[test]
fn identity_hash_changes_with_extent() {
    let mut draft = RichnessDraft::new();
    let h1 = draft.identity_hash_hex();
    draft.try_set_extent(3072).unwrap();
    let h2 = draft.identity_hash_hex();
    assert_ne!(h1, h2);
}

// ── Draft: validation ──────────────────────────────────────────────────────

#[test]
fn valid_draft_passes_validation() {
    let draft = RichnessDraft::new();
    assert!(draft.is_valid());
    let report = draft.validate();
    assert!(report.is_valid());
}

#[test]
fn invalid_draft_reports_errors() {
    let mut draft = RichnessDraft::new();
    draft.extent = 1025;
    let report = draft.validate();
    assert!(!report.is_valid());
    assert!(report
        .errors
        .iter()
        .any(|e| e.field_id == RichnessFieldId::Extent));
}

#[test]
fn validation_reports_budget_ceiling_errors() {
    let mut draft = RichnessDraft::new();
    // Set landmarks high so the budget is too low for the demand
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 5)
        .unwrap();
    draft
        .try_set_explicit_u32(RichnessFieldId::BudgetCeiling, 1500)
        .unwrap();
    let report = draft.validate();
    assert!(!report.is_valid());
    assert!(report
        .errors
        .iter()
        .any(|e| e.field_id == RichnessFieldId::BudgetCeiling));
}

// ── GUI: keyboard/mouse parity ─────────────────────────────────────────────

#[test]
fn keyboard_mode_discards_mouse() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Keyboard;
    let before = gui.selected_item;
    gui.handle_mouse_input(100, 100, MouseButton::Left, press());
    assert_eq!(gui.selected_item, before);
}

#[test]
fn keyboard_mode_discards_wheel() {
    let mut gui = RichnessGui::new();
    gui.set_viewport(640, 200);
    gui.mode = RichnessGuiMode::Keyboard;
    gui.scroll_by(-1000);
    assert_eq!(gui.scroll_offset, 0);
}

#[test]
fn mouse_mode_discards_keyboard_except_escape() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Mouse;
    let before = gui.selected_item;
    gui.handle_keyboard_input(KeyCode::ArrowDown, press());
    assert_eq!(gui.selected_item, before);
    gui.handle_keyboard_input(KeyCode::Enter, press());
    assert_eq!(gui.selected_item, before);
    assert_eq!(
        gui.handle_keyboard_input(KeyCode::Escape, press()),
        RichnessGuiAction::Close
    );
}

#[test]
fn escape_closes_in_any_mode() {
    let mut gui_kb = RichnessGui::new();
    gui_kb.mode = RichnessGuiMode::Keyboard;
    assert_eq!(
        gui_kb.handle_keyboard_input(KeyCode::Escape, press()),
        RichnessGuiAction::Close
    );

    let mut gui_mouse = RichnessGui::new();
    gui_mouse.mode = RichnessGuiMode::Mouse;
    assert_eq!(
        gui_mouse.handle_keyboard_input(KeyCode::Escape, press()),
        RichnessGuiAction::Close
    );

    let mut gui_none = RichnessGui::new();
    gui_none.mode = RichnessGuiMode::None;
    assert_eq!(
        gui_none.handle_keyboard_input(KeyCode::Escape, press()),
        RichnessGuiAction::Close
    );
}

// ── GUI: scaling ───────────────────────────────────────────────────────────

#[test]
fn scale_clamped_to_range() {
    let mut gui = RichnessGui::new();
    gui.set_scale(0);
    assert_eq!(gui.scale_pct(), 50);
    gui.set_scale(1000);
    assert_eq!(gui.scale_pct(), 400);
}

#[test]
fn hit_test_at_scaled_coords() {
    let mut gui = RichnessGui::new();
    gui.set_viewport(1280, 720);
    gui.set_scale(200);
    let base = gui.layout_base();
    let bx = base.panel.x + base.panel.w as i32 / 2;
    let by = base.panel.y + 50;
    let px = bx * 2;
    let py = by * 2;
    let hit = gui.hit_test(px, py);
    assert!(hit.is_some(), "expected hit at physical ({px}, {py})");
}

// ── GUI: scrolling ─────────────────────────────────────────────────────────

#[test]
fn scroll_clamps_to_bounds() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Mouse;
    gui.set_viewport(640, 200);
    let max = gui.max_scroll();
    gui.scroll_by(-(max + 1000));
    assert_eq!(gui.scroll_offset, max);
    gui.scroll_by(max as i32 + 1000);
    assert_eq!(gui.scroll_offset, 0);
}

#[test]
fn scroll_offset_affects_layout() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Mouse;
    gui.set_viewport(1280, 720);
    let before = gui.layout_base();
    let first_y_before = before.sections[0].rows[0].rect.y;
    gui.scroll_offset = 50;
    let after = gui.layout_base();
    let first_y_after = after.sections[0].rows[0].rect.y;
    assert_eq!(first_y_before - first_y_after, 50);
}

// ── GUI: input capture ─────────────────────────────────────────────────────

#[test]
fn is_inside_panel_detects_boundaries() {
    let mut gui = RichnessGui::new();
    gui.set_viewport(1280, 720);
    let panel = gui.layout_base().panel;
    let cx = panel.x + panel.w as i32 / 2;
    let cy = panel.y + 50;
    assert!(gui.is_inside_panel(cx, cy));
    assert!(!gui.is_inside_panel(0, 0));
}

#[test]
fn hit_test_outside_returns_none() {
    let gui = RichnessGui::new();
    assert!(gui.hit_test(-10, -10).is_none());
    assert!(gui.hit_test(2000, 2000).is_none());
}

// ── GUI: draw list determinism ─────────────────────────────────────────────

#[test]
fn draw_list_is_deterministic() {
    let gui = RichnessGui::new();
    let dl1 = gui.draw_list();
    let dl2 = gui.draw_list();
    assert_eq!(dl1.items.len(), dl2.items.len());
    for (a, b) in dl1.items.iter().zip(dl2.items.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn draw_list_has_background_and_content() {
    let gui = RichnessGui::new();
    let dl = gui.draw_list();
    let rects: Vec<_> = dl
        .items
        .iter()
        .filter(|item| matches!(item, DrawItem::Rect { .. }))
        .collect();
    assert!(!rects.is_empty());
    let texts: Vec<_> = dl
        .items
        .iter()
        .filter(|item| matches!(item, DrawItem::Text { .. }))
        .collect();
    assert!(!texts.is_empty());
}

// ── GUI: actions ───────────────────────────────────────────────────────────

#[test]
fn generate_action_produces_draft_on_valid() {
    let mut gui = RichnessGui::new();
    gui.selected_item = 13; // Generate action
    let result = gui.handle_keyboard_input(KeyCode::Enter, press());
    assert!(matches!(result, RichnessGuiAction::Generate(_)));
}

#[test]
fn generate_rejected_on_invalid_draft() {
    let mut gui = RichnessGui::new();
    gui.draft.extent = 1025;
    gui.selected_item = 13; // Generate action
    let result = gui.handle_keyboard_input(KeyCode::Enter, press());
    assert_eq!(result, RichnessGuiAction::None);
}

// ── Controller: stale/latest results ───────────────────────────────────────

#[test]
fn controller_latest_request_wins() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());

    let id1 = controller.enqueue(RichnessDraft::new(), failing_executor("old"));
    let id2 = controller.enqueue(RichnessDraft::new(), failing_executor("new"));

    assert!(id2 > id1);
    assert_eq!(controller.latest_submitted_id(), id2);

    let mut results = Vec::new();
    for _ in 0..300 {
        while let Some(r) = controller.poll_result() {
            results.push(r);
        }
        if !results.is_empty() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    let found_id1 = results.iter().any(|r| r.request_id() == id1);
    let found_id2 = results.iter().any(|r| r.request_id() == id2);
    assert!(!found_id1, "stale result {id1} should not appear");
    assert!(
        found_id2,
        "fresh result {id2} should appear (got {results:?})"
    );

    controller.shutdown();
}

#[test]
fn controller_stale_success_directory_cleaned() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());

    // Submit A with a slow executor that uses the request's actual dir
    let id_a = controller.enqueue(
        RichnessDraft::new(),
        Box::new(|req: &RichnessGenerationRequest| {
            thread::sleep(Duration::from_millis(200));
            let _ = std::fs::create_dir_all(&req.package_dir);
            ExecutorOutcome::PackageReady {
                request_id: req.id,
                package_dir: req.package_dir.clone(),
            }
        }),
    );
    let dir_a = package_dir_for_richness_request(controller.package_root(), id_a);

    // Immediately submit B
    let _id_b = controller.enqueue(RichnessDraft::new(), failing_executor("replacement"));

    // Wait for B to complete
    for _ in 0..100 {
        if controller.poll_result().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    while controller.poll_result().is_some() {}

    assert!(
        !dir_a.exists(),
        "stale success directory {dir_a:?} should be cleaned"
    );
    controller.shutdown();
}

#[test]
fn controller_submit_a_then_b_a_discarded() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());

    let id_a = controller.enqueue(RichnessDraft::new(), failing_executor("stale"));
    let id_b = controller.enqueue(RichnessDraft::new(), failing_executor("latest"));

    let mut results = Vec::new();
    for _ in 0..300 {
        while let Some(r) = controller.poll_result() {
            results.push(r);
        }
        if !results.is_empty() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert!(
        !results.iter().any(|r| r.request_id() == id_a),
        "stale A ({id_a}) should not appear"
    );
    assert!(
        results.iter().any(|r| r.request_id() == id_b),
        "fresh B ({id_b}) should appear"
    );

    controller.shutdown();
}

// ── Controller: active-world preservation on failure ───────────────────────

#[test]
fn controller_failure_does_not_produce_package_ready() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    let _id = controller.enqueue(RichnessDraft::new(), failing_executor("generation failed"));

    let mut outcome = None;
    for _ in 0..200 {
        outcome = controller.poll_result();
        if outcome.is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    let outcome = outcome.expect("should have an outcome");
    assert!(
        outcome.is_failure(),
        "failure must be Failed, not PackageReady"
    );
    controller.shutdown();
}

#[test]
fn controller_failure_preserves_close_intent() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    controller.set_close_intent(1);
    let _id = controller.enqueue(RichnessDraft::new(), failing_executor("fail"));

    for _ in 0..200 {
        if controller.poll_result().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert_eq!(controller.close_intent(), Some(1));
    controller.shutdown();
}

#[test]
fn controller_unwritable_destination_dir_preserves_state() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    controller.set_close_intent(5);
    let id = controller.enqueue(
        RichnessDraft::new(),
        Box::new(|req: &RichnessGenerationRequest| ExecutorOutcome::Failed {
            request_id: req.id,
            error_message: "cannot write to destination: permission denied".into(),
        }),
    );

    let mut outcome = None;
    for _ in 0..200 {
        outcome = controller.poll_result();
        if outcome.is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    let outcome = outcome.expect("should get outcome");
    assert!(outcome.is_failure());
    assert_eq!(outcome.request_id(), id);
    assert_eq!(controller.close_intent(), Some(5));
    controller.shutdown();
}

// ── Controller: worker shutdown ────────────────────────────────────────────

#[test]
fn controller_shutdown_idle_worker() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    controller.shutdown();
}

#[test]
fn controller_shutdown_with_pending_work() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    let _id = controller.enqueue(RichnessDraft::new(), failing_executor("work"));
    controller.shutdown();
}

#[test]
fn controller_drop_joins_worker() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    drop(controller);
}

#[test]
fn controller_shutdown_while_worker_inflight() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    let outcome = ExecutorOutcome::Failed {
        request_id: 1,
        error_message: "slow".into(),
    };
    let _id = controller.enqueue(
        RichnessDraft::new(),
        Box::new(move |_req| {
            thread::sleep(Duration::from_millis(50));
            outcome.clone()
        }),
    );
    thread::sleep(Duration::from_millis(10));
    controller.shutdown();
}

// ── Controller: close intent ───────────────────────────────────────────────

#[test]
fn controller_close_intent_set_and_clear() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    assert!(controller.close_intent().is_none());
    controller.set_close_intent(42);
    assert_eq!(controller.close_intent(), Some(42));
    controller.clear_close_intent();
    assert!(controller.close_intent().is_none());
    controller.shutdown();
}

// ── Controller: request IDs are monotonic ──────────────────────────────────

#[test]
fn controller_request_ids_monotonic() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    let id1 = controller.enqueue(RichnessDraft::new(), failing_executor("a"));
    let id2 = controller.enqueue(RichnessDraft::new(), failing_executor("b"));
    let id3 = controller.enqueue(RichnessDraft::new(), failing_executor("c"));
    assert_eq!(id1, 1);
    assert_eq!(id2, 2);
    assert_eq!(id3, 3);
    controller.shutdown();
}

// ── Controller: package directories ────────────────────────────────────────

#[test]
fn controller_package_dirs_are_distinct() {
    let root = tempfile::tempdir().unwrap();
    let controller = RichnessGenerationController::spawn_at_root(root.path().to_path_buf());
    let dir1 = package_dir_for_richness_request(controller.package_root(), 1);
    let dir2 = package_dir_for_richness_request(controller.package_root(), 2);
    assert_ne!(dir1, dir2);
    assert!(dir1.starts_with(controller.package_root()));
    assert!(dir2.starts_with(controller.package_root()));
    controller.shutdown();
}

// ── Baseline byte compatibility ────────────────────────────────────────────

#[test]
fn richness_draft_default_is_byte_stable() {
    let draft1 = RichnessDraft::new();
    let draft2 = RichnessDraft::new();
    assert_eq!(draft1.to_canonical_bytes(), draft2.to_canonical_bytes());
}

#[test]
fn richness_gui_default_is_byte_stable() {
    let gui1 = RichnessGui::new();
    let gui2 = RichnessGui::new();
    assert_eq!(gui1.draw_list().items, gui2.draw_list().items);
}

// ── Input capture: releases and repeats ignored ────────────────────────────

#[test]
fn releases_and_repeats_cannot_activate() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Keyboard;
    let before = gui.draft.preset;
    gui.handle_keyboard_input(KeyCode::ArrowRight, RichnessInputAction::Release);
    gui.handle_keyboard_input(KeyCode::ArrowRight, RichnessInputAction::Repeat);
    assert_eq!(gui.draft.preset, before);
}

// ── Draft: effective values ────────────────────────────────────────────────

#[test]
fn effective_values_respect_preset() {
    let draft = RichnessDraft::new();
    assert_eq!(draft.effective_u32(RichnessFieldId::Landmarks), 1);
    assert_eq!(draft.effective_u32(RichnessFieldId::Zones), 1);
    assert_eq!(draft.effective_cave_mode(), RichnessCaveMode::Preferred);
    assert_eq!(draft.effective_u32(RichnessFieldId::VerticalOpenings), 0);
    assert_eq!(draft.effective_u32(RichnessFieldId::BudgetCeiling), 3000);
}

#[test]
fn effective_values_change_with_preset() {
    let mut draft = RichnessDraft::new();
    draft.set_preset(RichnessPreset::Rich);
    assert_eq!(draft.effective_u32(RichnessFieldId::Landmarks), 3);
    assert_eq!(draft.effective_u32(RichnessFieldId::BudgetCeiling), 8000);
}

#[test]
fn explicit_overrides_effective() {
    let mut draft = RichnessDraft::new();
    assert_eq!(draft.effective_u32(RichnessFieldId::Landmarks), 1);
    draft
        .try_set_explicit_u32(RichnessFieldId::Landmarks, 4)
        .unwrap();
    assert_eq!(draft.effective_u32(RichnessFieldId::Landmarks), 4);
}

// ── GUI: keyboard navigation ───────────────────────────────────────────────

#[test]
fn keyboard_arrow_keys_navigate() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Keyboard;
    assert_eq!(gui.selected_item, 0);
    gui.handle_keyboard_input(KeyCode::ArrowDown, press());
    assert_eq!(gui.selected_item, 1);
    gui.handle_keyboard_input(KeyCode::ArrowUp, press());
    assert_eq!(gui.selected_item, 0);
}

#[test]
fn keyboard_arrow_right_cycles_enum() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Keyboard;
    assert_eq!(gui.draft.preset, RichnessPreset::Sparse);
    gui.handle_keyboard_input(KeyCode::ArrowRight, press());
    assert_eq!(gui.draft.preset, RichnessPreset::Moderate);
}

#[test]
fn keyboard_numeric_editing() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Keyboard;
    for _ in 0..3 {
        gui.handle_keyboard_input(KeyCode::ArrowDown, press());
    }
    gui.handle_keyboard_input(KeyCode::Digit4, press());
    gui.handle_keyboard_input(KeyCode::Digit2, press());
    gui.handle_keyboard_input(KeyCode::Enter, press());
    assert_eq!(gui.draft.seed, 42);
}

// ── GUI: mouse hit testing ─────────────────────────────────────────────────

#[test]
fn mouse_click_on_field_selects_it() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Mouse;
    gui.set_viewport(1280, 720);

    let item_idx = item_index_for(RichnessFieldId::Theme);
    let layout = gui.layout_base();
    for section in &layout.sections {
        for row in &section.rows {
            if row.item_index == item_idx {
                gui.handle_mouse_input(row.rect.x + 10, row.rect.y + 4, MouseButton::Left, press());
                assert_eq!(gui.selected_item, item_idx);
                return;
            }
        }
    }
    panic!("Theme field not found in layout");
}

#[test]
fn mouse_click_dropdown_opens_list() {
    let mut gui = RichnessGui::new();
    gui.mode = RichnessGuiMode::Mouse;
    gui.set_viewport(1280, 720);

    let idx = item_index_for(RichnessFieldId::Preset);
    let row = gui
        .layout_base()
        .sections
        .iter()
        .flat_map(|s| s.rows.iter())
        .find(|r| r.item_index == idx)
        .copied()
        .unwrap();
    let right = row.rect.x + row.rect.w as i32;
    gui.selected_item = idx;
    gui.handle_mouse_input(right - 5, row.rect.y + 4, MouseButton::Left, press());
    let layout = gui.layout_base();
    assert!(!layout.dropdowns.is_empty());
}
