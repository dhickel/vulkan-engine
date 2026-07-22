//! Exact historical caller-syntax compatibility fixtures.
//!
//! These tests encode the precise pre-refactor syntax that external crates
//! depend on.  They are compiled under `#![deny(unused_must_use)]` so that an
//! accidentally introduced `Result` on a legacy method is a compile failure.
//! The file must remain its own crate-level test so the inner attribute works.

#![deny(unused_must_use)]

use glam::{Mat4, Vec3};
use renderer::animation::{
    AnimationChannel, AnimationClip, AnimationPlayer, AnimationSampler, AnimationTarget,
    Interpolation, KeyframeValue,
};
use renderer::Scene;
use std::collections::HashMap;

// ── H-A2: Scene::pick is an immutable method ──────────────────────────

/// Prove that `Scene::pick` can be assigned to a function pointer with an
/// immutable receiver.
#[test]
fn pick_is_immutable_fn() {
    // If `pick` accidentally requires `&mut self`, this assignment fails.
    let _: fn(&Scene, f32, f32, u32, u32, Mat4, Mat4, Vec3) -> Option<renderer::SceneNodeId> =
        Scene::pick;
}

/// Prove that `Scene::pick` can be called on an immutable scene reference.
#[test]
fn pick_on_immutable_scene() {
    let scene = Scene::new();
    // Must compile without requiring `let mut scene`.
    let result = scene.pick(
        100.0,
        200.0,
        800,
        600,
        Mat4::IDENTITY,
        Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0),
        Vec3::ZERO,
    );
    assert!(result.is_none());
}

/// Prove that `pick_last_camera` is also immutable.
#[test]
fn pick_last_camera_is_immutable_fn() {
    let _: fn(&Scene, f32, f32, u32, u32) -> Option<renderer::SceneNodeId> =
        Scene::pick_last_camera;
}

/// Prove that `pick_last_camera` can be called on an immutable scene.
#[test]
fn pick_last_camera_on_immutable_scene() {
    let scene = Scene::new();
    let result = scene.pick_last_camera(100.0, 200.0, 800, 600);
    assert!(result.is_none());
}

// ── H-A2 / M-A7: AnimationChannel struct literal ──────────────────────

/// Prove that `AnimationChannel` can be constructed with the historical
/// `node_index` field (not `target`).
#[test]
fn animation_channel_struct_literal() {
    let channel = AnimationChannel {
        node_index: 7,
        target_path: AnimationTarget::Translation,
        sampler_index: 0,
    };
    assert_eq!(channel.node_index, 7);
    assert_eq!(channel.target_path, AnimationTarget::Translation);
    assert_eq!(channel.sampler_index, 0);
}

// ── H-A2: AnimationPlayer historical signatures ───────────────────────

/// Prove that `set_clip` does not return a `Result` that must be used.
#[test]
fn set_clip_no_result() {
    let mut player = AnimationPlayer::new();
    let clip = AnimationClip::new("test", 1.0);
    // Must compile without `let _ = ...` or `.unwrap()`.
    player.set_clip(clip);
}

/// Prove that `update` returns `HashMap<usize, Mat4>` directly, not a
/// `Result`.
#[test]
fn update_no_result() {
    let mut player = AnimationPlayer::new();
    let mut clip = AnimationClip::new("test", 2.0);
    clip.samplers.push(AnimationSampler {
        input: vec![0.0, 1.0],
        output: vec![
            KeyframeValue::Translation(Vec3::ZERO),
            KeyframeValue::Translation(Vec3::new(1.0, 0.0, 0.0)),
        ],
        interpolation: Interpolation::Linear,
    });
    clip.channels.push(AnimationChannel {
        node_index: 0,
        target_path: AnimationTarget::Translation,
        sampler_index: 0,
    });
    player.set_clip(clip);
    player.play();

    // `update` must return `HashMap<usize, Mat4>` directly.
    let transforms: HashMap<usize, Mat4> = player.update(0.016);
    // Empty because no target map was set — but must compile.
    assert!(transforms.is_empty());
}

/// Prove that `set_speed` does not return a `Result`.
#[test]
fn set_speed_no_result() {
    let mut player = AnimationPlayer::new();
    player.set_speed(2.0);
    assert!((player.speed() - 2.0).abs() < 0.01);
}

/// Prove that NaN speed is silently rejected (non-mutating).
#[test]
fn nan_speed_rejected() {
    let mut player = AnimationPlayer::new();
    player.set_speed(2.0);
    player.set_speed(f32::NAN);
    // Speed must remain 2.0.
    assert!((player.speed() - 2.0).abs() < 0.01);
}

/// Prove that infinite speed is silently rejected (non-mutating).
#[test]
fn infinite_speed_rejected() {
    let mut player = AnimationPlayer::new();
    player.set_speed(2.0);
    player.set_speed(f32::INFINITY);
    assert!((player.speed() - 2.0).abs() < 0.01);

    player.set_speed(f32::NEG_INFINITY);
    assert!((player.speed() - 2.0).abs() < 0.01);
}

// ── Current validated syntax: try_* fallible methods ─────────────────

/// Prove that `try_set_clip` returns a `Result` and compiles.
#[test]
fn try_set_clip_returns_result() {
    let mut player = AnimationPlayer::new();
    let clip = AnimationClip::new("test", 1.0);
    let result = player.try_set_clip(clip);
    assert!(result.is_ok());
}

/// Prove that `try_set_speed` returns a `Result`.
#[test]
fn try_set_speed_returns_result() {
    let mut player = AnimationPlayer::new();
    assert!(player.try_set_speed(1.5).is_ok());
    assert!(player.try_set_speed(f32::NAN).is_err());
    assert!(player.try_set_speed(f32::INFINITY).is_err());
}

/// Prove that `try_update` returns a `Result<HashMap<SceneNodeId, Mat4>>`.
#[test]
fn try_update_returns_result() {
    use renderer::SceneNodeId;
    let mut player = AnimationPlayer::new();
    let mut clip = AnimationClip::new("test", 2.0);
    clip.samplers.push(AnimationSampler {
        input: vec![0.0, 1.0],
        output: vec![
            KeyframeValue::Translation(Vec3::ZERO),
            KeyframeValue::Translation(Vec3::new(1.0, 0.0, 0.0)),
        ],
        interpolation: Interpolation::Linear,
    });
    clip.channels.push(AnimationChannel {
        node_index: 0,
        target_path: AnimationTarget::Translation,
        sampler_index: 0,
    });

    // Set target map so resolution succeeds
    let mut target_map = HashMap::new();
    target_map.insert(0usize, SceneNodeId::new(1, 0));
    player.set_target_map(target_map);

    player.set_clip(clip);
    player.play();
    let result = player.try_update(0.016);
    assert!(result.is_ok());
    let transforms = result.unwrap();
    assert!(transforms.contains_key(&SceneNodeId::new(1, 0)));
}
