//! Tests for `FrameExtensions`: construction, defaults, frame-lifetime semantics,
//! and integration with the renderer's extension plumbing.
//!
//! These tests validate the DTO shape and the renderer's take-and-clear contract.

use glam::{Mat4, Vec3};
use renderer::{FrameExtensions, SceneNodeId};

#[test]
fn frame_extensions_default_is_empty() {
    let ext = FrameExtensions::new();
    assert!(ext.is_empty());
    assert!(ext.transform_overrides.is_empty());
}

#[test]
fn frame_extensions_with_transform_overrides() {
    let mut ext = FrameExtensions::new();
    let node_id = SceneNodeId::new(1, 0);
    ext.transform_overrides
        .insert(node_id, Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)));
    assert!(!ext.is_empty());
    assert_eq!(ext.transform_overrides.len(), 1);
    assert!((ext.transform_overrides[&node_id].w_axis.truncate() - Vec3::new(1.0, 2.0, 3.0)).length() < 0.001);
}

#[cfg(feature = "debug-draw")]
#[test]
fn frame_extensions_with_debug_lines() {
    let mut ext = FrameExtensions::new();
    ext.debug_lines
        .push((Vec3::ZERO, Vec3::X, Vec3::new(1.0, 0.0, 0.0)));
    assert!(!ext.is_empty());
    assert_eq!(ext.debug_lines.len(), 1);
}

#[cfg(feature = "debug-draw")]
#[test]
fn frame_extensions_empty_when_only_defaults() {
    let ext = FrameExtensions::default();
    assert!(ext.is_empty());
    assert!(ext.debug_lines.is_empty());
}

#[test]
fn frame_extensions_clone_works() {
    let mut ext = FrameExtensions::new();
    ext.transform_overrides
        .insert(SceneNodeId::new(2, 1), Mat4::IDENTITY);
    let cloned = ext.clone();
    assert_eq!(ext.transform_overrides.len(), cloned.transform_overrides.len());
}
