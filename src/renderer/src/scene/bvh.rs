//! Deterministic full-rebuild scene BVH with conservative-visible side list.
//!
//! ## Purpose
//! Provides a scene-owned binary BVH built from rigid known-bounds mesh nodes.
//! Conservative-visible (skinned/deformed/unknown) entries are kept in an
//! always-visible side list. The BVH may only remove items that the Phase 06
//! linear culler would also remove — proven by property tests.
//!
//! ## Compilation
//! Gated behind `--features scene-bvh`.

use crate::data::camera::{Aabb, Frustum};
use crate::data::handles::MeshHandle;
use crate::data::retirement::FrameSerial;
use crate::scene::scene_world::SceneNodeId;
use glam::Vec3;

// ---------------------------------------------------------------------------
// BVH leaf identity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct BvhLeaf {
    pub node_id: SceneNodeId,
    pub mesh_handle: MeshHandle,
    pub world_aabb: Aabb,
    pub last_reference_serial: FrameSerial,
}

// ---------------------------------------------------------------------------
// BVH node
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum BvhNode {
    Internal {
        left: u32,
        right: u32,
        aabb: Aabb,
    },
    Leaf {
        leaf_start: u32,
        leaf_count: u32,
        aabb: Aabb,
    },
}

impl BvhNode {
    pub fn aabb(&self) -> &Aabb {
        match self {
            Self::Internal { aabb, .. } | Self::Leaf { aabb, .. } => aabb,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene BVH
// ---------------------------------------------------------------------------

pub struct SceneBvh {
    nodes: Vec<BvhNode>,
    leaves: Vec<BvhLeaf>,
    pub conservative_visible: Vec<(SceneNodeId, MeshHandle)>,
    pub build_time_ms: f64,
}

#[derive(Debug, Clone)]
pub struct BvhBuildItem {
    pub node_id: SceneNodeId,
    pub mesh_handle: MeshHandle,
    pub world_aabb: Aabb,
    pub last_reference_serial: FrameSerial,
}

impl SceneBvh {
    /// Build a BVH from rigid known-bounds items + conservative-visible list.
    pub fn build(
        rigid_items: &[BvhBuildItem],
        conservative_visible: Vec<(SceneNodeId, MeshHandle)>,
    ) -> Self {
        let start = std::time::Instant::now();

        if rigid_items.is_empty() {
            return Self {
                nodes: Vec::new(),
                leaves: Vec::new(),
                conservative_visible,
                build_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
        }

        let mut conservative_visible = conservative_visible;
        let mut leaves: Vec<BvhLeaf> = rigid_items
            .iter()
            .filter_map(|item| {
                if !item.world_aabb.is_finite() || !item.world_aabb.is_ordered() {
                    conservative_visible.push((item.node_id, item.mesh_handle));
                    return None;
                }
                Some(BvhLeaf {
                    node_id: item.node_id,
                    mesh_handle: item.mesh_handle,
                    world_aabb: item.world_aabb,
                    last_reference_serial: item.last_reference_serial,
                })
            })
            .collect();

        if leaves.is_empty() {
            return Self {
                nodes: Vec::new(),
                leaves,
                conservative_visible,
                build_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
        }

        let mut nodes = Vec::new();
        let len = leaves.len();
        Self::build_recursive(&mut leaves, 0, len, &mut nodes);

        Self {
            nodes,
            leaves,
            conservative_visible,
            build_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    fn build_recursive(
        leaves: &mut [BvhLeaf],
        start: usize,
        end: usize,
        nodes: &mut Vec<BvhNode>,
    ) -> u32 {
        let count = end - start;
        debug_assert!(count > 0);

        let mut union_aabb = leaves[start].world_aabb;
        for i in (start + 1)..end {
            union_aabb.extend_to_enclose(&leaves[i].world_aabb);
        }

        if count == 1 {
            let idx = nodes.len() as u32;
            nodes.push(BvhNode::Leaf {
                leaf_start: start as u32,
                leaf_count: 1,
                aabb: union_aabb,
            });
            return idx;
        }

        // Compute centroid extent and pick split axis.
        let mut cmin = Vec3::splat(f32::MAX);
        let mut cmax = Vec3::splat(f32::MIN);
        for leaf in &leaves[start..end] {
            let c = leaf.world_aabb.center();
            cmin = cmin.min(c);
            cmax = cmax.max(c);
        }
        let extent = cmax - cmin;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        // Median split: sort the sub-range by centroid on axis.
        leaves[start..end].sort_by_key(|leaf| {
            let c = leaf.world_aabb.center();
            (
                ordered_float_bits(match axis {
                    0 => c.x,
                    1 => c.y,
                    _ => c.z,
                }),
                leaf.node_id.slot,
                leaf.node_id.generation,
                leaf.mesh_handle.slot,
                leaf.mesh_handle.generation,
            )
        });

        let mid = start + count / 2;

        // Reserve node slot before recursion (left child gets next index).
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode::Internal {
            left: 0,
            right: 0,
            aabb: union_aabb,
        });

        let left = Self::build_recursive(leaves, start, mid, nodes);
        let right = Self::build_recursive(leaves, mid, end, nodes);

        nodes[node_idx as usize] = BvhNode::Internal {
            left,
            right,
            aabb: union_aabb,
        };

        node_idx
    }

    /// Collect visible items: traverse BVH against frustum, append
    /// conservative-visible side list, dedup, and sort.
    pub fn collect_visible(&self, frustum: &Frustum) -> Vec<(SceneNodeId, MeshHandle)> {
        let mut result: Vec<(SceneNodeId, MeshHandle)> = Vec::new();

        if !self.nodes.is_empty() {
            self.traverse_node(0, frustum, &mut result);
        }

        result.extend(self.conservative_visible.iter().copied());
        result.sort_by_key(|(nid, mh)| (nid.slot, nid.generation, mh.slot, mh.generation));
        result.dedup();
        result
    }

    fn traverse_node(
        &self,
        node_idx: u32,
        frustum: &Frustum,
        out: &mut Vec<(SceneNodeId, MeshHandle)>,
    ) {
        let node = &self.nodes[node_idx as usize];
        if !frustum.intersects_aabb(node.aabb()) {
            return;
        }
        match node {
            BvhNode::Internal { left, right, .. } => {
                self.traverse_node(*left, frustum, out);
                self.traverse_node(*right, frustum, out);
            }
            BvhNode::Leaf {
                leaf_start,
                leaf_count,
                ..
            } => {
                let start = *leaf_start as usize;
                let end = start + *leaf_count as usize;
                for leaf in &self.leaves[start..end] {
                    if frustum.intersects_aabb(&leaf.world_aabb) {
                        out.push((leaf.node_id, leaf.mesh_handle));
                    }
                }
            }
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    pub fn conservative_count(&self) -> usize {
        self.conservative_visible.len()
    }
}

// ---------------------------------------------------------------------------
// Deterministic float ordering
// ---------------------------------------------------------------------------

fn ordered_float_bits(v: f32) -> u32 {
    if v.is_nan() {
        return 0;
    }
    let bits = v.to_bits();
    if (bits >> 31) != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

// ---------------------------------------------------------------------------
// Linear culler reference (for property tests)
// ---------------------------------------------------------------------------

pub fn linear_cull(
    items: &[(SceneNodeId, MeshHandle, Aabb)],
    frustum: &Frustum,
) -> Vec<(SceneNodeId, MeshHandle)> {
    items
        .iter()
        .filter_map(|&(nid, mh, aabb)| {
            if frustum.intersects_aabb(&aabb) {
                Some((nid, mh))
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::camera::Frustum;
    use glam::{Mat4, Vec3};

    fn mk_aabb(min: Vec3, max: Vec3) -> Aabb {
        Aabb::from_min_max(min, max)
    }

    fn mk_frustum() -> Frustum {
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0);
        Frustum::from_view_projection(&(proj * view))
    }

    fn mk_item(node: u32, mesh: u32, min_x: f32, max_x: f32) -> BvhBuildItem {
        BvhBuildItem {
            node_id: SceneNodeId::new(node, 0),
            mesh_handle: MeshHandle::new(mesh, 0),
            world_aabb: mk_aabb(Vec3::new(min_x, -0.5, -1.0), Vec3::new(max_x, 0.5, -0.5)),
            last_reference_serial: FrameSerial::new(1),
        }
    }

    #[test]
    fn empty_bvh() {
        let bvh = SceneBvh::build(&[], Vec::new());
        assert_eq!(bvh.leaf_count(), 0);
        assert_eq!(bvh.node_count(), 0);
    }

    #[test]
    fn single_leaf() {
        let bvh = SceneBvh::build(&[mk_item(0, 1, -0.5, 0.5)], Vec::new());
        assert_eq!(bvh.leaf_count(), 1);
        assert_eq!(bvh.node_count(), 1);
    }

    #[test]
    fn three_leaves() {
        let items = vec![
            mk_item(0, 1, -2.0, -1.0),
            mk_item(1, 2, 0.0, 1.0),
            mk_item(2, 3, 2.0, 3.0),
        ];
        let bvh = SceneBvh::build(&items, Vec::new());
        assert_eq!(bvh.leaf_count(), 3);
        assert!(bvh.node_count() >= 3);
    }

    #[test]
    fn deterministic_rebuild() {
        let items: Vec<BvhBuildItem> = (0..20)
            .map(|i| mk_item(i, i, i as f32 * 0.5, i as f32 * 0.5 + 0.5))
            .collect();
        let mut reversed = items.clone();
        reversed.reverse();
        let b1 = SceneBvh::build(&items, Vec::new());
        let b2 = SceneBvh::build(&reversed, Vec::new());
        assert_eq!(b1.node_count(), b2.node_count());
        assert_eq!(b1.nodes, b2.nodes);
        assert_eq!(b1.leaves, b2.leaves);
    }

    #[test]
    fn conservative_always_emitted() {
        let items = vec![mk_item(0, 1, 100.0, 101.0)];
        let cv = vec![(SceneNodeId::new(99, 0), MeshHandle::new(99, 0))];
        let bvh = SceneBvh::build(&items, cv);
        let f = mk_frustum();
        let vis = bvh.collect_visible(&f);
        assert!(vis.contains(&(SceneNodeId::new(99, 0), MeshHandle::new(99, 0))));
    }

    #[test]
    fn bvh_vs_linear_parity_property() {
        // Deterministic generated scenes cover varied centers/extents without
        // introducing a randomized test dependency.
        let f = mk_frustum();
        let mut state = 0x5eed_u64;
        for scene in 0..64_u32 {
            let mut bvh_items = Vec::new();
            let mut linear_items = Vec::new();
            for item in 0..96_u32 {
                let mut next = || {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 32) as u32) as f32 / u32::MAX as f32
                };
                let center = Vec3::new(
                    next() * 80.0 - 40.0,
                    next() * 50.0 - 25.0,
                    -(next() * 120.0 + 0.01),
                );
                let extent = Vec3::new(next(), next(), next()) * 3.0 + Vec3::splat(0.001);
                let aabb = mk_aabb(center - extent, center + extent);
                let identity = scene * 96 + item;
                let nid = SceneNodeId::new(identity, scene % 3);
                let mh = MeshHandle::new(identity, scene % 5);
                bvh_items.push(BvhBuildItem {
                    node_id: nid,
                    mesh_handle: mh,
                    world_aabb: aabb,
                    last_reference_serial: FrameSerial::new(scene as u64 + 1),
                });
                linear_items.push((nid, mh, aabb));
            }

            let bvh = SceneBvh::build(&bvh_items, Vec::new());
            let bs: std::collections::BTreeSet<_> = bvh.collect_visible(&f).into_iter().collect();
            let ls: std::collections::BTreeSet<_> =
                linear_cull(&linear_items, &f).into_iter().collect();
            assert_eq!(bs, ls, "BVH parity failed for generated scene {scene}");
        }
    }

    #[test]
    fn invalid_bounds_are_conservative_visible() {
        let mut item = mk_item(7, 9, -1.0, 1.0);
        item.world_aabb.max.x = f32::NAN;
        let bvh = SceneBvh::build(&[item], Vec::new());
        assert_eq!(bvh.leaf_count(), 0);
        assert_eq!(
            bvh.collect_visible(&mk_frustum()),
            vec![(SceneNodeId::new(7, 0), MeshHandle::new(9, 0))]
        );
    }

    #[test]
    fn bvh_edge_case_near_plane() {
        // Near items intersect the near plane; the last item is far left.
        let items = vec![
            BvhBuildItem {
                node_id: SceneNodeId::new(0, 0),
                mesh_handle: MeshHandle::new(0, 0),
                world_aabb: mk_aabb(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, -0.05)),
                last_reference_serial: FrameSerial::new(1),
            },
            BvhBuildItem {
                node_id: SceneNodeId::new(1, 0),
                mesh_handle: MeshHandle::new(1, 0),
                world_aabb: mk_aabb(Vec3::new(-0.5, -0.5, -0.15), Vec3::new(0.5, 0.5, -0.12)),
                last_reference_serial: FrameSerial::new(1),
            },
            BvhBuildItem {
                node_id: SceneNodeId::new(2, 0),
                mesh_handle: MeshHandle::new(2, 0),
                world_aabb: mk_aabb(Vec3::new(-100.0, -0.5, -1.0), Vec3::new(-99.0, 0.5, -0.5)),
                last_reference_serial: FrameSerial::new(1),
            },
        ];
        let lin: Vec<_> = items
            .iter()
            .map(|i| (i.node_id, i.mesh_handle, i.world_aabb))
            .collect();
        let f = mk_frustum();
        let bvh = SceneBvh::build(&items, Vec::new());
        let bv = bvh.collect_visible(&f);
        let lv = linear_cull(&lin, &f);
        let bs: std::collections::BTreeSet<_> = bv.iter().copied().collect();
        let ls: std::collections::BTreeSet<_> = lv.iter().copied().collect();
        assert_eq!(bs, ls);
    }

    #[test]
    fn ordered_float_bits_spec() {
        // +0.0 and -0.0 are distinct in IEEE 754 but both map to valid
        // ordered values. The ordering is total but signed zeros differ.
        assert!(ordered_float_bits(1.0) > ordered_float_bits(0.0));
        assert!(ordered_float_bits(-1.0) < ordered_float_bits(0.0));
        assert_eq!(ordered_float_bits(f32::NAN), 0);
        // Verify total ordering property: distinct floats have distinct bit patterns.
        assert_ne!(ordered_float_bits(0.5), ordered_float_bits(1.0));
        assert_ne!(ordered_float_bits(-0.5), ordered_float_bits(-1.0));
    }

    #[test]
    fn build_time_recorded() {
        let bvh = SceneBvh::build(&[mk_item(0, 1, -0.5, 0.5)], Vec::new());
        assert!(bvh.build_time_ms >= 0.0);
    }
}
