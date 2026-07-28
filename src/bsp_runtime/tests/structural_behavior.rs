//! Integration tests for structural behavior adapters.
//!
//! Tests deterministic update ordering, activation cascades, and
//! trigger→target chains through the `StructuralBehaviorAdapter`.

use bsp_runtime::behavior::{
    BehaviorEntityInfo, DoorPhase, StructuralBehaviorAdapter, TriggerEvent,
};
use bsp_runtime::Activation;
use std::collections::HashSet;

/// Build a standard set of inter-connected structural entities.
fn build_test_scene() -> StructuralBehaviorAdapter {
    let mut adapter = StructuralBehaviorAdapter::new();
    adapter.register_entities(vec![
        BehaviorEntityInfo {
            entity_index: 1,
            classname: "func_door".into(),
            targetname: Some("door_main".into()),
            target: None,
            killtarget: None,
            origin: [0.0, 0.0, 0.0],
            movedir: Some([1.0, 0.0, 0.0]),
            speed: Some(200.0),
            wait: Some(1.0),
            lip: Some(0.0),
            height: None,
            light_style: None,
        },
        BehaviorEntityInfo {
            entity_index: 2,
            classname: "func_button".into(),
            targetname: Some("btn_open".into()),
            target: Some("door_main".into()),
            killtarget: None,
            origin: [10.0, 0.0, 5.0],
            movedir: Some([0.0, 0.0, 1.0]),
            speed: Some(40.0),
            wait: Some(1.0),
            lip: Some(4.0),
            height: None,
            light_style: None,
        },
        BehaviorEntityInfo {
            entity_index: 3,
            classname: "trigger_once".into(),
            targetname: None,
            target: Some("door_main".into()),
            killtarget: None,
            origin: [-5.0, 0.0, 0.0],
            movedir: None,
            speed: None,
            wait: None,
            lip: None,
            height: None,
            light_style: None,
        },
        BehaviorEntityInfo {
            entity_index: 4,
            classname: "trigger_multiple".into(),
            targetname: None,
            target: Some("btn_open".into()),
            killtarget: None,
            origin: [5.0, 0.0, 0.0],
            movedir: None,
            speed: None,
            wait: None,
            lip: None,
            height: None,
            light_style: None,
        },
        BehaviorEntityInfo {
            entity_index: 5,
            classname: "func_plat".into(),
            targetname: Some("lift".into()),
            target: None,
            killtarget: None,
            origin: [20.0, 0.0, 0.0],
            movedir: None,
            speed: Some(150.0),
            wait: None,
            lip: Some(8.0),
            height: Some(64.0),
            light_style: None,
        },
    ]);
    adapter
}

#[test]
fn door_cycle_full_open_close() {
    let mut adapter = build_test_scene();

    // Door starts closed
    assert!(!adapter.is_moving(1));

    // Activate door directly
    adapter.activate_by_index(1, Activation::Toggle);
    assert!(adapter.is_moving(1));
    assert_eq!(adapter.doors.get(&1).unwrap().phase, DoorPhase::Opening);

    // Fast-forward through opening
    let _updates = adapter.update(0.1); // speed 200, dist 1 → opens in 0.005s
    let door = adapter.doors.get(&1).unwrap();
    // Door should be open or closing (wait is 1.0, but we stepped 0.1)
    assert!(door.phase == DoorPhase::Open || door.phase == DoorPhase::Closing);

    // Reset and verify
    adapter.reset();
    assert!(!adapter.is_moving(1));
}

#[test]
fn button_press_triggers_door() {
    let mut adapter = build_test_scene();

    // Activate button → should cascade to door
    adapter.activate_by_index(2, Activation::On);
    let door = adapter.doors.get(&1).unwrap();
    assert_eq!(door.phase, DoorPhase::Opening);
}

#[test]
fn trigger_once_fires_target() {
    let mut adapter = build_test_scene();

    // Update trigger occupants → should fire target (door)
    let event = adapter
        .update_trigger_occupants(3, HashSet::from([100]))
        .unwrap();
    assert!(matches!(event, TriggerEvent::Fired { .. }));

    let door = adapter.doors.get(&1).unwrap();
    assert_eq!(door.phase, DoorPhase::Opening);

    // Second activation should not fire (trigger_once)
    let event2 = adapter
        .update_trigger_occupants(3, HashSet::from([100]))
        .unwrap();
    assert!(matches!(event2, TriggerEvent::Occupied));
}

#[test]
fn trigger_multiple_chains_to_button_to_door() {
    let mut adapter = build_test_scene();

    // trigger_multiple (4) → btn_open (2) → door_main (1)
    let event = adapter
        .update_trigger_occupants(4, HashSet::from([200]))
        .unwrap();
    assert!(matches!(event, TriggerEvent::Fired { .. }));

    // Both button and door should be activated
    let button = adapter.buttons.get(&2).unwrap();
    let door = adapter.doors.get(&1).unwrap();
    assert_eq!(button.phase, bsp_runtime::ButtonPhase::Pressing);
    assert_eq!(door.phase, DoorPhase::Opening);
}

#[test]
fn platform_cycle() {
    let mut adapter = build_test_scene();

    adapter.activate_by_index(5, Activation::Toggle);
    let plat = adapter.platforms.get(&5).unwrap();
    assert_eq!(plat.phase, bsp_runtime::PlatformPhase::Raising);

    // Fast-forward
    let _updates = adapter.update(1.0);
    let plat = adapter.platforms.get(&5).unwrap();
    assert_eq!(plat.phase, bsp_runtime::PlatformPhase::High);
}

#[test]
fn light_style_intensity() {
    let mut adapter = StructuralBehaviorAdapter::new();
    adapter.register_entities(vec![BehaviorEntityInfo {
        entity_index: 10,
        classname: "light".into(),
        targetname: None,
        target: None,
        killtarget: None,
        origin: [0.0, 5.0, 0.0],
        movedir: None,
        speed: None,
        wait: None,
        lip: None,
        height: None,
        light_style: Some("pulse".into()),
    }]);

    assert!(adapter.light_style_active("pulse"));
    adapter.set_light_style_intensity("pulse", 0.0);
    assert!(!adapter.light_style_active("pulse"));
    adapter.set_light_style_intensity("pulse", 0.8);
    assert!(adapter.light_style_active("pulse"));
}

#[test]
fn door_terminal_target_fires_once_when_terminal_state_is_entered() {
    let mut adapter = StructuralBehaviorAdapter::new();
    adapter.register_entities(vec![
        BehaviorEntityInfo {
            entity_index: 1,
            classname: "func_door".into(),
            targetname: Some("source".into()),
            target: Some("receiver".into()),
            killtarget: None,
            origin: [0.0, 0.0, 0.0],
            movedir: Some([1.0, 0.0, 0.0]),
            speed: Some(100.0),
            wait: Some(-1.0),
            lip: Some(0.0),
            height: None,
            light_style: None,
        },
        BehaviorEntityInfo {
            entity_index: 2,
            classname: "func_door".into(),
            targetname: Some("receiver".into()),
            target: None,
            killtarget: None,
            origin: [5.0, 0.0, 0.0],
            movedir: Some([1.0, 0.0, 0.0]),
            speed: Some(100.0),
            wait: Some(-1.0),
            lip: Some(0.0),
            height: None,
            light_style: None,
        },
    ]);

    adapter.activate_by_target("source", Activation::On);
    adapter.update(1.0);
    assert_eq!(adapter.doors.get(&2).unwrap().phase, DoorPhase::Closing);

    adapter.doors.get_mut(&2).unwrap().phase = DoorPhase::Open;
    adapter.update(0.016);
    assert_eq!(adapter.doors.get(&2).unwrap().phase, DoorPhase::Open);
}

#[test]
fn deterministic_entity_position_tracks_movement() {
    let mut adapter = StructuralBehaviorAdapter::new();
    adapter.register_entities(vec![BehaviorEntityInfo {
        entity_index: 1,
        classname: "func_door".into(),
        targetname: None,
        target: None,
        killtarget: None,
        origin: [0.0, 0.0, 0.0],
        movedir: Some([1.0, 0.0, 0.0]),
        speed: Some(100.0),
        wait: Some(1.0),
        lip: Some(0.0),
        height: None,
        light_style: None,
    }]);

    adapter.activate_by_index(1, Activation::On);
    let pos_before = adapter.entity_position(1).unwrap();
    let _updates = adapter.update(0.005); // Half-open at speed 100
    let pos_after = adapter.entity_position(1).unwrap();
    assert!(pos_after[0] > pos_before[0]);
}

// ── Door-Specific Tests (Phase 07) ─────────────────────────────────

#[test]
fn door_trigger_chain_opens_door() {
    let mut adapter = StructuralBehaviorAdapter::new();
    adapter.register_entities(vec![
        BehaviorEntityInfo {
            entity_index: 1,
            classname: "func_door".into(),
            targetname: Some("door_main".into()),
            target: None,
            killtarget: None,
            origin: [0.0, 0.0, 0.0],
            movedir: Some([0.0, 0.0, 1.0]),
            speed: Some(100.0),
            wait: Some(3.0),
            lip: Some(0.0),
            height: None,
            light_style: None,
        },
        BehaviorEntityInfo {
            entity_index: 2,
            classname: "trigger_multiple".into(),
            targetname: None,
            target: Some("door_main".into()),
            killtarget: None,
            origin: [-64.0, 0.0, 0.0],
            movedir: None,
            speed: None,
            wait: None,
            lip: None,
            height: None,
            light_style: None,
        },
    ]);

    // Trigger fires via occupant entry
    let event = adapter
        .update_trigger_occupants(2, HashSet::from([100]))
        .unwrap();
    assert!(matches!(event, TriggerEvent::Fired { .. }));

    let door = adapter.doors.get(&1).unwrap();
    assert_eq!(door.phase, DoorPhase::Opening);
}

#[test]
fn door_position_tracks_vertical_movement() {
    let mut adapter = StructuralBehaviorAdapter::new();
    adapter.register_entities(vec![BehaviorEntityInfo {
        entity_index: 1,
        classname: "func_door".into(),
        targetname: None,
        target: None,
        killtarget: None,
        origin: [0.0, 0.0, 0.0],
        movedir: Some([0.0, 0.0, 1.0]),
        speed: Some(100.0),
        wait: Some(3.0),
        lip: Some(0.0),
        height: None,
        light_style: None,
    }]);

    adapter.activate_by_index(1, Activation::On);
    // At speed=100 and travel_distance=1, full open takes 0.01s
    let _updates = adapter.update(0.005); // Halfway
    let pos = adapter.entity_position(1).unwrap();
    assert!(pos[2] > 0.0 && pos[2] < 1.0);
    assert!(adapter.is_moving(1));
}

#[test]
fn door_closed_state_has_correct_position() {
    let adapter = StructuralBehaviorAdapter::new();
    let mut adapter = adapter;
    adapter.register_entities(vec![BehaviorEntityInfo {
        entity_index: 1,
        classname: "func_door".into(),
        targetname: None,
        target: None,
        killtarget: None,
        origin: [10.0, 5.0, 0.0],
        movedir: Some([1.0, 0.0, 0.0]),
        speed: Some(100.0),
        wait: Some(1.0),
        lip: Some(0.0),
        height: None,
        light_style: None,
    }]);

    let pos = adapter.entity_position(1).unwrap();
    assert_eq!(pos, [10.0, 5.0, 0.0]);
    assert!(!adapter.is_moving(1));
}

#[test]
fn door_export_import_state_round_trip() {
    let mut adapter = StructuralBehaviorAdapter::new();
    adapter.register_entities(vec![BehaviorEntityInfo {
        entity_index: 1,
        classname: "func_door".into(),
        targetname: Some("d1".into()),
        target: None,
        killtarget: None,
        origin: [0.0, 0.0, 0.0],
        movedir: Some([0.0, 0.0, 1.0]),
        speed: Some(100.0),
        wait: Some(3.0),
        lip: Some(0.0),
        height: None,
        light_style: None,
    }]);

    // Open the door
    adapter.activate_by_index(1, Activation::On);
    adapter.update(1.0);

    let exported = adapter.export_state();
    assert_eq!(exported.doors.len(), 1);
    assert_eq!(exported.doors[0].entity_index, 1);
    assert_eq!(exported.doors[0].phase, 2); // Open

    // Fresh adapter, import
    let mut fresh = StructuralBehaviorAdapter::new();
    fresh.register_entities(vec![BehaviorEntityInfo {
        entity_index: 1,
        classname: "func_door".into(),
        targetname: Some("d1".into()),
        target: None,
        killtarget: None,
        origin: [0.0, 0.0, 0.0],
        movedir: Some([0.0, 0.0, 1.0]),
        speed: Some(100.0),
        wait: Some(3.0),
        lip: Some(0.0),
        height: None,
        light_style: None,
    }]);

    assert_eq!(fresh.doors.get(&1).unwrap().phase, DoorPhase::Closed);
    fresh.import_state(&exported);
    assert_eq!(fresh.doors.get(&1).unwrap().phase, DoorPhase::Open);
}

#[test]
fn door_activation_toggle_behavior() {
    let mut adapter = StructuralBehaviorAdapter::new();
    adapter.register_entities(vec![BehaviorEntityInfo {
        entity_index: 1,
        classname: "func_door".into(),
        targetname: None,
        target: None,
        killtarget: None,
        origin: [0.0, 0.0, 0.0],
        movedir: Some([1.0, 0.0, 0.0]),
        speed: Some(100.0),
        wait: Some(3.0),
        lip: Some(0.0),
        height: None,
        light_style: None,
    }]);

    // Activate once → opens
    adapter.activate_by_index(1, Activation::Toggle);
    assert_eq!(adapter.doors.get(&1).unwrap().phase, DoorPhase::Opening);

    // Fast-forward through open
    adapter.update(1.0);
    assert_eq!(adapter.doors.get(&1).unwrap().phase, DoorPhase::Open);

    // Activate again → closes
    adapter.activate_by_index(1, Activation::Toggle);
    assert_eq!(adapter.doors.get(&1).unwrap().phase, DoorPhase::Closing);
}
