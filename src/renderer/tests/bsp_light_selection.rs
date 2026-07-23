//! Tests for production BSP light selection: PVS filtering, scoring, hysteresis, and fallback.

#![cfg(feature = "bsp")]

use bsp::extract::LightDescriptor;
use bsp::visibility::PvsSet;
use glam::Vec3;
use renderer::api::bsp::BspMountState;

fn light(entity_index: u32, x: f32, intensity: f32) -> LightDescriptor {
    LightDescriptor {
        entity_index,
        origin: Vec3::new(x, 0.0, 0.0),
        intensity,
        color: [1.0, 1.0, 1.0],
        radius: 100.0,
        style: None,
    }
}

fn mount_with_lights(lights: Vec<LightDescriptor>, light_leafs: Vec<Option<u32>>) -> BspMountState {
    let mut mount = BspMountState::new();
    mount.activate();
    mount.set_render_assets(Vec::new(), Vec::new(), Vec::new(), lights);
    mount.light_leafs = light_leafs;
    mount
}

#[test]
fn selection_scores_by_contribution_when_pvs_disabled() {
    let mut mount = mount_with_lights(
        vec![
            light(0, 100.0, 100.0),
            light(1, 1.0, 100.0),
            light(2, 10.0, 500.0),
        ],
        vec![None, None, None],
    );

    let selected = mount.select_light_indices_for_camera(Vec3::ZERO, 3);

    assert_eq!(selected, vec![1, 2, 0]);
}

#[test]
fn ties_break_by_entity_index_then_source_order() {
    let mut mount = mount_with_lights(
        vec![
            light(10, 10.0, 100.0),
            light(3, 10.0, 100.0),
            light(3, 10.0, 100.0),
        ],
        vec![None, None, None],
    );

    let selected = mount.select_light_indices_for_camera(Vec3::ZERO, 3);

    assert_eq!(selected, vec![1, 2, 0]);
}

#[test]
fn valid_pvs_selects_visible_lights_first_then_fallback_fills() {
    let mut mount = mount_with_lights(
        vec![
            light(0, 8.0, 100.0),
            light(1, 1.0, 100.0),
            light(2, 2.0, 100.0),
            light(3, 3.0, 100.0),
        ],
        vec![Some(0), Some(1), Some(2), Some(3)],
    );
    mount.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits: vec![0b0000_0001],
        valid: true,
    });

    let selected = mount.select_light_indices_for_camera(Vec3::ZERO, 3);

    assert_eq!(selected, vec![0, 1, 2]);
}

#[test]
fn invalid_pvs_is_conservative_all_lights_are_primary_candidates() {
    let mut mount = mount_with_lights(
        vec![
            light(0, 100.0, 100.0),
            light(1, 1.0, 100.0),
            light(2, 2.0, 100.0),
        ],
        vec![Some(0), Some(1), Some(2)],
    );
    mount.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits: vec![0xFF],
        valid: false,
    });

    let selected = mount.select_light_indices_for_camera(Vec3::ZERO, 2);

    assert_eq!(selected, vec![1, 2]);
}

#[test]
fn missing_light_leafs_are_non_pvs_fallback_candidates() {
    let mut mount = mount_with_lights(
        vec![light(0, 10.0, 100.0), light(1, 1.0, 100.0)],
        vec![Some(0), None],
    );
    mount.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits: vec![0b0000_0001],
        valid: true,
    });

    let selected = mount.select_light_indices_for_camera(Vec3::ZERO, 2);

    assert_eq!(selected, vec![0, 1]);
}

#[test]
fn hysteresis_retains_previous_selection_before_swapping() {
    let mut mount = mount_with_lights(
        vec![
            light(0, 1.0, 100.0),
            light(1, 2.0, 100.0),
            light(2, 3.0, 100.0),
            light(3, 4.0, 100.0),
        ],
        vec![Some(0), Some(0), Some(1), Some(1)],
    );
    mount.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits: vec![0b0000_0001],
        valid: true,
    });
    assert_eq!(
        mount.select_light_indices_for_camera(Vec3::ZERO, 2),
        vec![0, 1]
    );

    mount.current_pvs = Some(PvsSet {
        leaf_index: 1,
        bits: vec![0b0000_0010],
        valid: true,
    });
    assert_eq!(
        mount.select_light_indices_for_camera(Vec3::ZERO, 2),
        vec![0, 1]
    );
    assert_eq!(
        mount.select_light_indices_for_camera(Vec3::ZERO, 2),
        vec![0, 1]
    );
    assert_eq!(
        mount.select_light_indices_for_camera(Vec3::ZERO, 2),
        vec![2, 3]
    );
}

#[test]
fn camera_discontinuity_resets_hysteresis() {
    let mut mount = mount_with_lights(
        vec![light(0, 1.0, 100.0), light(1, 2.0, 100.0)],
        vec![Some(0), Some(1)],
    );
    mount.current_pvs = Some(PvsSet {
        leaf_index: 0,
        bits: vec![0b0000_0001],
        valid: true,
    });
    assert_eq!(
        mount.select_light_indices_for_camera(Vec3::ZERO, 1),
        vec![0]
    );

    mount.current_pvs = Some(PvsSet {
        leaf_index: 1,
        bits: vec![0b0000_0010],
        valid: true,
    });
    assert_eq!(
        mount.select_light_indices_for_camera(Vec3::new(1000.0, 0.0, 0.0), 1),
        vec![1]
    );
}

#[test]
fn frame_light_conversion_clamps_invalid_negative_values() {
    let mut mount = mount_with_lights(
        vec![LightDescriptor {
            entity_index: 0,
            origin: Vec3::ZERO,
            intensity: -1.0,
            color: [-1.0, 0.5, 2.0],
            radius: -5.0,
            style: None,
        }],
        vec![None],
    );

    let selected = mount.select_frame_lights_for_camera(Vec3::ZERO, 1);

    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].intensity, 0.0);
    assert_eq!(selected[0].color, Vec3::new(0.0, 0.5, 2.0));
    assert_eq!(selected[0].range, 0.001);
}
