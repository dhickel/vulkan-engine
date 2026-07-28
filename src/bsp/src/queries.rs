//! Spatial queries: point_contents, stored-hull trace, and line trace.
//!
//! Contract: `bsp-spatial-physics.md` §5.
//!
//! # Hull Semantics
//!
//! The compiler (qbsp) pre-expands clipnode trees for hulls 0–2.
//! Hull 0 is the point hull (no expansion).
//! Hull 1 is the player hull with nominal extents ±(16,16,24) Quake units.
//! Hull 2 is the large-monster hull with nominal extents ±(24,24,32).
//!
//! All trace functions trace a **point** through the pre-expanded clipnode
//! tree. The caller must pass the correct headnode from the world model.
//! The `StoredHull` parameter is informational (extent queries); the actual
//! hull selection is determined by `hull_headnode`.
//!
//! The player-movement contract (bsp-spatial-physics.md §5.2.1) documents
//! two competing hull records. Phase 06 threshold tests resolve this dispute
//! by tracing hull 0 and hull 1 through known gap widths in a compiled fixture.

use glam::Vec3;

use crate::coords::QuakeToEngine;
use crate::lumps;

/// Contents codes from BSP leaves.
pub mod contents {
    pub const EMPTY: i32 = -1;
    pub const SOLID: i32 = -2;
    pub const WATER: i32 = -3;
    pub const SLIME: i32 = -4;
    pub const LAVA: i32 = -5;
    pub const SKY: i32 = -6;
}

/// Result of a point_contents query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointContents {
    /// Outside the BSP tree (empty).
    Empty,
    /// Solid leaf.
    Solid,
    /// Water volume.
    Water,
    /// Slime volume.
    Slime,
    /// Lava volume.
    Lava,
    /// Sky leaf.
    Sky,
}

impl PointContents {
    /// Convert from a leaf contents integer.
    pub fn from_leaf_contents(contents: i32) -> Self {
        match contents {
            contents::EMPTY => PointContents::Empty,
            contents::SOLID => PointContents::Solid,
            contents::WATER => PointContents::Water,
            contents::SLIME => PointContents::Slime,
            contents::LAVA => PointContents::Lava,
            contents::SKY => PointContents::Sky,
            _ => PointContents::Empty,
        }
    }

    /// Whether this contents is solid (start-solid condition).
    pub fn is_solid(self) -> bool {
        matches!(self, PointContents::Solid)
    }

    /// Whether this contents is a liquid.
    pub fn is_liquid(self) -> bool {
        matches!(self, PointContents::Water | PointContents::Slime | PointContents::Lava)
    }

    /// Whether this contents is empty (outside or void).
    pub fn is_empty(self) -> bool {
        matches!(self, PointContents::Empty | PointContents::Sky)
    }
}

/// Query the contents classification at a point in engine space.
///
/// Walks the BSP tree from root to leaf using `camera_leaf_index` and returns
/// the leaf's contents classification.
pub fn point_contents(
    point: Vec3,
    nodes: &[lumps::Node],
    leaves: &[lumps::Leaf],
    planes: &[lumps::Plane],
) -> PointContents {
    point_contents_with_transform(point, nodes, leaves, planes, &QuakeToEngine::default())
}

/// Query contents with an explicit coordinate transform/scale override.
pub fn point_contents_with_transform(
    point: Vec3,
    nodes: &[lumps::Node],
    leaves: &[lumps::Leaf],
    planes: &[lumps::Plane],
    qte: &QuakeToEngine,
) -> PointContents {
    if nodes.is_empty() || leaves.is_empty() {
        return PointContents::Empty;
    }

    let point_quake = to_quake_space(point, qte);
    let Some(leaf_index) = leaf_index_for_quake_point(point_quake, nodes, leaves, planes) else {
        return PointContents::Empty;
    };

    let leaf = &leaves[leaf_index as usize];
    PointContents::from_leaf_contents(leaf.contents)
}

fn leaf_index_for_quake_point(
    point: Vec3,
    nodes: &[lumps::Node],
    leaves: &[lumps::Leaf],
    planes: &[lumps::Plane],
) -> Option<u32> {
    let mut node = 0i32;
    loop {
        if node < 0 {
            let leaf = -1 - node;
            return (leaf >= 0 && (leaf as usize) < leaves.len()).then_some(leaf as u32);
        }
        let node_ref = nodes.get(node as usize)?;
        let plane = planes.get(node_ref.plane_id as usize)?;
        let dist = point.dot(plane.normal) - plane.dist;
        node = if dist >= 0.0 {
            node_ref.children[0]
        } else {
            node_ref.children[1]
        };
    }
}

/// Standard stored hull types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredHull {
    /// Hull 0: point-sized.
    Point = 0,
    /// Hull 1: player-sized (±16, ±16, ±24 Quake units).
    Player = 1,
    /// Hull 2: large monster (±24, ±24, ±32 Quake units).
    LargeMonster = 2,
}

impl StoredHull {
    /// Get hull extents in Quake units.
    pub fn extents_quake(self) -> Vec3 {
        match self {
            StoredHull::Point => Vec3::new(0.0, 0.0, 0.0),
            StoredHull::Player => Vec3::new(16.0, 16.0, 24.0),
            StoredHull::LargeMonster => Vec3::new(24.0, 24.0, 32.0),
        }
    }

    /// Get hull extents in engine units.
    pub fn extents_engine(self, qte: &QuakeToEngine) -> Vec3 {
        let q = self.extents_quake();
        let (mins, maxs) = qte.aabb(-q, q);
        (maxs - mins) * 0.5
    }
}

/// Trace a line from `start` to `end` using the world model's stored hull.
///
/// This is the primary query for player movement. It looks up the headnode
/// from `models[0].headnode[hull]` and delegates to [`trace_stored_hull`].
pub fn trace_line(
    start: Vec3,
    end: Vec3,
    hull: StoredHull,
    clipnodes: &[lumps::Clipnode],
    planes: &[lumps::Plane],
    models: &[lumps::Model],
    qte: &QuakeToEngine,
) -> TraceResult {
    if models.is_empty() {
        return TraceResult::no_hit();
    }
    let headnode = models[0].headnode[hull as usize];
    if headnode == 0 && hull as usize != 0 {
        // Hull not compiled; fall back to point hull.
        return trace_stored_hull(
            start, end, StoredHull::Point,
            clipnodes, planes, models[0].headnode[0], qte,
        );
    }
    trace_stored_hull(start, end, hull, clipnodes, planes, headnode, qte)
}

/// Result of a stored-hull trace.
#[derive(Debug, Clone)]
pub struct TraceResult {
    /// Hit fraction: 0.0 = start-solid, 1.0 = no hit.
    pub hit_fraction: f32,
    /// Hit plane normal (in engine space). Undefined if fraction is 0.0 or 1.0.
    pub plane_normal: Vec3,
    /// Hit plane distance (in engine units). Undefined if fraction is 0.0 or 1.0.
    pub plane_dist: f32,
    /// Contents at the hit surface.
    pub contents: PointContents,
    /// Whether the trace started in solid.
    pub starts_solid: bool,
    /// Whether the entire trace is in solid (all-solid).
    pub all_solid: bool,
    /// Whether the trace completed with no hit.
    pub no_hit: bool,
}

impl TraceResult {
    /// Create a no-hit result.
    pub fn no_hit() -> Self {
        TraceResult {
            hit_fraction: 1.0,
            plane_normal: Vec3::ZERO,
            plane_dist: 0.0,
            contents: PointContents::Empty,
            starts_solid: false,
            all_solid: false,
            no_hit: true,
        }
    }

    /// Create a start-solid result.
    pub fn start_solid(contents: PointContents) -> Self {
        TraceResult {
            hit_fraction: 0.0,
            plane_normal: Vec3::ZERO,
            plane_dist: 0.0,
            contents,
            starts_solid: true,
            all_solid: false,
            no_hit: false,
        }
    }

    /// Create an all-solid result.
    pub fn all_solid(contents: PointContents) -> Self {
        TraceResult {
            hit_fraction: 0.0,
            plane_normal: Vec3::ZERO,
            plane_dist: 0.0,
            contents,
            starts_solid: true,
            all_solid: true,
            no_hit: false,
        }
    }

    /// Create a hit result at a given fraction.
    pub fn hit(fraction: f32, normal: Vec3, dist: f32, contents: PointContents) -> Self {
        TraceResult {
            hit_fraction: fraction,
            plane_normal: normal,
            plane_dist: dist,
            contents,
            starts_solid: false,
            all_solid: false,
            no_hit: false,
        }
    }
}

/// Clipnode content codes (negative children in clipnode tree).
#[allow(dead_code)]
mod clip_contents {
    pub const EMPTY: i32 = -1;
    pub const SOLID: i32 = -2;
}

/// Trace a stored hull from `start` to `end` in engine space.
///
/// This implements the Quake `SV_RecursiveHullCheck` algorithm on the clipnode
/// tree. The hull is expanded by the hull extents.
///
/// Returns a `TraceResult` with the earliest hit (lowest fraction).
pub fn trace_stored_hull(
    start: Vec3,
    end: Vec3,
    _hull: StoredHull,
    clipnodes: &[lumps::Clipnode],
    planes: &[lumps::Plane],
    hull_headnode: i32,
    qte: &QuakeToEngine,
) -> TraceResult {
    let epsilon = qte.surface_epsilon();

    let start_q = to_quake_space(start, qte);
    let end_q = to_quake_space(end, qte);

    // Quick check: start solid?
    let start_contents = clipnode_point_contents_quake(start_q, clipnodes, planes, hull_headnode);
    if start_contents == PointContents::Solid {
        return TraceResult::start_solid(PointContents::Solid);
    }

    let mut trace = TraceResult::no_hit();

    recursive_hull_check(
        &start_q,
        &end_q,
        0.0,
        1.0,
        hull_headnode,
        clipnodes,
        planes,
        qte,
        epsilon,
        Vec3::ZERO,
        0.0,
        &mut trace,
    );

    trace
}

/// Convert engine-space point to Quake space (inverse of QuakeToEngine).
fn to_quake_space(v: Vec3, qte: &QuakeToEngine) -> Vec3 {
    if qte.scale.abs() < 1e-10 {
        return Vec3::ZERO;
    }
    let inv = 1.0 / qte.scale;
    // engine (ex, ey, ez) ← quake (qx, qy, qz) via: (s*qx, s*qz, -s*qy)
    // Inverse: qx = ex/s, qy = -ez/s, qz = ey/s
    Vec3::new(v.x * inv, -v.z * inv, v.y * inv)
}

/// Check whether a point in Quake space is solid for trace purposes.
fn clipnode_point_contents_quake(
    point: Vec3,
    clipnodes: &[lumps::Clipnode],
    planes: &[lumps::Plane],
    hull_headnode: i32,
) -> PointContents {
    if hull_headnode < 0 {
        return match hull_headnode {
            clip_contents::SOLID => PointContents::Solid,
            _ => PointContents::Empty,
        };
    }
    if clipnodes.is_empty() {
        return PointContents::Empty;
    }

    let mut node = hull_headnode;

    loop {
        if node < 0 {
            return match node {
                clip_contents::SOLID => PointContents::Solid,
                _ => PointContents::Empty,
            };
        }

        let Some(cn) = clipnodes.get(node as usize) else {
            return PointContents::Empty;
        };
        let Some(plane) = planes.get(cn.plane as usize) else {
            return PointContents::Empty;
        };

        let dist = point.dot(plane.normal) - plane.dist;
        node = if dist >= 0.0 {
            cn.children[0]
        } else {
            cn.children[1]
        };
    }
}

/// Recursive hull check through the clipnode tree.
fn recursive_hull_check(
    p1: &Vec3,
    p2: &Vec3,
    p1f: f32,
    p2f: f32,
    node_idx: i32,
    clipnodes: &[lumps::Clipnode],
    planes: &[lumps::Plane],
    qte: &QuakeToEngine,
    epsilon: f32,
    hit_normal: Vec3,
    hit_dist: f32,
    trace: &mut TraceResult,
) {
    if !trace.no_hit {
        return;
    }

    if node_idx < 0 {
        if node_idx == clip_contents::SOLID {
            *trace = TraceResult::hit(p1f.clamp(0.0, 1.0), hit_normal, hit_dist, PointContents::Solid);
        }
        return;
    }

    let Some(cn) = clipnodes.get(node_idx as usize) else {
        return;
    };
    let Some(plane) = planes.get(cn.plane as usize) else {
        return;
    };

    let d1 = p1.dot(plane.normal) - plane.dist;
    let d2 = p2.dot(plane.normal) - plane.dist;

    if d1 >= 0.0 && d2 >= 0.0 {
        recursive_hull_check(p1, p2, p1f, p2f, cn.children[0], clipnodes, planes, qte, epsilon, hit_normal, hit_dist, trace);
    } else if d1 < 0.0 && d2 < 0.0 {
        recursive_hull_check(p1, p2, p1f, p2f, cn.children[1], clipnodes, planes, qte, epsilon, hit_normal, hit_dist, trace);
    } else {
        let frac = (d1 / (d1 - d2)).clamp(0.0, 1.0);
        let mid = *p1 + frac * (*p2 - *p1);
        let midf = p1f + frac * (p2f - p1f);

        let start_front = d1 >= 0.0;
        let first_child = if start_front { cn.children[0] } else { cn.children[1] };
        let second_child = if start_front { cn.children[1] } else { cn.children[0] };

        let (engine_normal, engine_dist) = qte.plane(plane.normal, plane.dist);
        let crossing_normal = if start_front { engine_normal } else { -engine_normal };
        let crossing_dist = if start_front { engine_dist } else { -engine_dist };

        recursive_hull_check(p1, &mid, p1f, midf, first_child, clipnodes, planes, qte, epsilon, hit_normal, hit_dist, trace);
        if !trace.no_hit {
            return;
        }

        let push = if start_front { -plane.normal } else { plane.normal } * (epsilon / qte.scale.max(1e-10));
        let just_across = mid + push;
        if clipnode_point_contents_quake(just_across, clipnodes, planes, second_child) == PointContents::Solid {
            *trace = TraceResult::hit(midf.clamp(0.0, 1.0), crossing_normal, crossing_dist, PointContents::Solid);
            return;
        }

        recursive_hull_check(&mid, p2, midf, p2f, second_child, clipnodes, planes, qte, epsilon, crossing_normal, crossing_dist, trace);
    }
}

/// Unsupported trace shape error.
///
/// Stored-hull traces support hulls 0–2 only. Arbitrary-box traces are not
/// supported without implementing the expansion algorithm and golden-testing it.
#[derive(Debug, Clone)]
pub enum TraceError {
    UnsupportedTraceShape,
    MissingClipnodes,
    InvalidHullHeadnode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_contents_empty_world() {
        let result = point_contents(Vec3::ZERO, &[], &[], &[]);
        assert_eq!(result, PointContents::Empty);
    }

    #[test]
    fn point_contents_solid_leaf() {
        let planes = vec![lumps::Plane {
            normal: Vec3::X,
            dist: 0.0,
            plane_type: 0,
        }];
        let nodes = vec![lumps::Node {
            plane_id: 0,
            children: [-1, -2],
            mins: [0; 3],
            maxs: [0; 3],
            face_id: 0,
            face_num: 0,
        }];
        let leaves = vec![
            lumps::Leaf {
                contents: -2, // SOLID
                visofs: 0,
                mins: [0; 3],
                maxs: [0; 3],
                mark_id: 0,
                mark_num: 0,
                ambient: [0; 4],
            },
            lumps::Leaf {
                contents: -1, // EMPTY
                visofs: 0,
                mins: [0; 3],
                maxs: [0; 3],
                mark_id: 0,
                mark_num: 0,
                ambient: [0; 4],
            },
        ];

        // x >= 0 → leaf 0 (solid)
        let result = point_contents(Vec3::new(10.0, 0.0, 0.0), &nodes, &leaves, &planes);
        assert_eq!(result, PointContents::Solid);

        // x < 0 → leaf 1 (empty)
        let result = point_contents(Vec3::new(-10.0, 0.0, 0.0), &nodes, &leaves, &planes);
        assert_eq!(result, PointContents::Empty);
    }

    #[test]
    fn point_contents_liquid() {
        let planes = vec![lumps::Plane {
            normal: Vec3::X,
            dist: 0.0,
            plane_type: 0,
        }];
        let nodes = vec![lumps::Node {
            plane_id: 0,
            children: [-1, -2],
            mins: [0; 3],
            maxs: [0; 3],
            face_id: 0,
            face_num: 0,
        }];
        let leaves = vec![
            lumps::Leaf {
                contents: -3, // WATER
                visofs: 0,
                mins: [0; 3],
                maxs: [0; 3],
                mark_id: 0,
                mark_num: 0,
                ambient: [0; 4],
            },
            lumps::Leaf {
                contents: -5, // LAVA
                visofs: 0,
                mins: [0; 3],
                maxs: [0; 3],
                mark_id: 0,
                mark_num: 0,
                ambient: [0; 4],
            },
        ];

        let result = point_contents(Vec3::new(1.0, 0.0, 0.0), &nodes, &leaves, &planes);
        assert_eq!(result, PointContents::Water);

        let result = point_contents(Vec3::new(-1.0, 0.0, 0.0), &nodes, &leaves, &planes);
        assert_eq!(result, PointContents::Lava);
    }

    #[test]
    fn trace_no_hit_when_empty() {
        let result = trace_stored_hull(
            Vec3::ZERO,
            Vec3::X,
            StoredHull::Point,
            &[],
            &[],
            -1,
            &QuakeToEngine::default(),
        );
        assert!(result.no_hit);
    }

    #[test]
    fn stored_hull_extents() {
        let qte = QuakeToEngine::default();
        let player_ext = StoredHull::Player.extents_engine(&qte);
        assert!((player_ext.x - 16.0 * 0.0254).abs() < 1e-6);
        assert!((player_ext.y - 24.0 * 0.0254).abs() < 1e-6);
        assert!((player_ext.z - 16.0 * 0.0254).abs() < 1e-6);
    }

    #[test]
    fn trace_result_factories() {
        let no_hit = TraceResult::no_hit();
        assert_eq!(no_hit.hit_fraction, 1.0);
        assert!(no_hit.no_hit);

        let ss = TraceResult::start_solid(PointContents::Solid);
        assert_eq!(ss.hit_fraction, 0.0);
        assert!(ss.starts_solid);
        assert!(!ss.all_solid);

        let all_s = TraceResult::all_solid(PointContents::Solid);
        assert!(all_s.all_solid);

        let hit = TraceResult::hit(0.5, Vec3::Y, 10.0, PointContents::Water);
        assert!((hit.hit_fraction - 0.5).abs() < 1e-6);
        assert_eq!(hit.contents, PointContents::Water);
    }

    #[test]
    fn to_quake_space_roundtrip() {
        let qte = QuakeToEngine::default();
        let q = Vec3::new(128.0, 256.0, 64.0);
        let e = qte.position_vec3(q);
        let q2 = to_quake_space(e, &qte);
        assert!((q2.x - q.x).abs() < 1e-4);
        assert!((q2.y - q.y).abs() < 1e-4);
        assert!((q2.z - q.z).abs() < 1e-4);
    }
}
