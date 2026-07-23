//! PVS decompression, camera leaf lookup, and corrupt-VIS fallback.
//!
//! Contract: `bsp-spatial-physics.md` §4.

use glam::Vec3;

use crate::lumps;

/// Decompressed PVS bit array for a single leaf.
///
/// Each bit represents a leaf index: bit `i` set = leaf `i` is potentially
/// visible from the source leaf.
#[derive(Debug, Clone)]
pub struct PvsSet {
    /// Which leaf this PVS set is for.
    pub leaf_index: u32,
    /// Raw bit array: `bits[i / 8]` with bit `i % 8` for leaf `i`.
    pub bits: Vec<u8>,
    /// Whether decompression was successful.
    pub valid: bool,
}

impl PvsSet {
    /// Check whether a leaf is in this PVS set.
    pub fn is_visible(&self, leaf_index: u32) -> bool {
        let byte_idx = (leaf_index / 8) as usize;
        let bit_idx = (leaf_index % 8) as usize;
        if byte_idx >= self.bits.len() {
            return false;
        }
        (self.bits[byte_idx] >> bit_idx) & 1 != 0
    }
}

/// PVS decompression state for the entire map.
#[derive(Debug, Clone)]
pub struct PvsState {
    /// Number of leaves in the map.
    pub num_leaves: u32,
    /// Number of bytes per decompressed PVS set.
    pub pvs_bytes: u32,
    /// Whether PVS data is globally corrupted.
    pub corrupt: bool,
}

impl PvsState {
    /// Create a new PVS state.
    pub fn new(num_leaves: u32, vis_data: &[u8]) -> Self {
        let pvs_bytes = (num_leaves + 7) / 8;
        let corrupt = vis_data.is_empty();
        PvsState {
            num_leaves,
            pvs_bytes,
            corrupt,
        }
    }

    /// Decompress the PVS for a specific leaf.
    ///
    /// Uses Quake's RLE scheme:
    /// - `0x00` byte: zero-fill `next_byte * 8` bits
    /// - Non-zero byte: raw bits for next 8 leaves
    ///
    /// Returns a PvsSet with `valid = true` on success.
    /// Returns a PvsSet with `valid = false` and the all-visible conservative fallback.
    pub fn decompress_for_leaf(
        &self,
        leaf_index: u32,
        leaf: &lumps::Leaf,
        vis_data: &[u8],
    ) -> PvsSet {
        if self.corrupt || vis_data.is_empty() || leaf.visofs < 0 {
            return self.conservative_fallback(leaf_index);
        }

        let offset = leaf.visofs as usize;
        if offset >= vis_data.len() {
            return self.conservative_fallback(leaf_index);
        }

        // Compressed PVS for this leaf starts at visofs
        let mut bits = vec![0u8; self.pvs_bytes as usize];
        let mut byte_idx = 0usize;
        let mut pos = offset;

        while byte_idx < self.pvs_bytes as usize && pos < vis_data.len() {
            let cmd = vis_data[pos];
            pos += 1;

            if cmd == 0 {
                // Zero-fill: next byte is the count
                if pos >= vis_data.len() {
                    // Corrupt stream: missing count byte after zero. Discard
                    // any partial decode and return the all-visible fallback.
                    return self.conservative_fallback(leaf_index);
                }
                let count = vis_data[pos] as usize;
                pos += 1;
                if count == 0 || byte_idx + count > self.pvs_bytes as usize {
                    return self.conservative_fallback(leaf_index);
                }
                byte_idx += count;
                // Bits are already zero
            } else {
                // Non-zero byte: raw bits for next 8 leaves (write 1 byte)
                if byte_idx < bits.len() {
                    bits[byte_idx] = cmd;
                }
                byte_idx += 1;
            }
        }

        if byte_idx != self.pvs_bytes as usize {
            // Ended mid-stream. Never expose a partially decoded PVS because
            // that would create false culling; fall back to all-visible.
            return self.conservative_fallback(leaf_index);
        }

        PvsSet {
            leaf_index,
            bits,
            valid: true,
        }
    }

    /// Return a conservative all-visible fallback set.
    pub fn conservative_fallback(&self, leaf_index: u32) -> PvsSet {
        let mut bits = vec![0xFFu8; self.pvs_bytes as usize];
        // Clear any excess bits beyond the leaf count
        let excess = (self.pvs_bytes * 8) as usize - self.num_leaves as usize;
        if excess > 0 && !bits.is_empty() {
            let last = bits.last_mut().unwrap();
            let mask = 0xFFu8 >> excess.min(8);
            *last &= mask;
        }
        PvsSet {
            leaf_index,
            bits,
            valid: false,
        }
    }

}

/// Result of a camera leaf lookup.
#[derive(Debug, Clone)]
pub struct CameraLeafResult {
    /// The leaf index containing the camera.
    pub leaf_index: u32,
    /// Whether the camera is in a solid leaf.
    pub in_solid: bool,
    /// Whether the camera is outside the BSP tree (leaf -1).
    pub outside: bool,
    /// Whether the leaf is valid for PVS queries.
    pub has_valid_pvs: bool,
}

/// Find the leaf containing a point by walking the BSP tree.
///
/// Uses the classic BSP traversal: `dot(point, plane_normal) - plane_dist >= 0`
/// → front child, else back child. Camera exactly on plane → default to front.
pub fn camera_leaf_index(
    point: &Vec3,
    nodes: &[lumps::Node],
    leaves: &[lumps::Leaf],
    planes: &[lumps::Plane],
) -> CameraLeafResult {
    if nodes.is_empty() {
        return CameraLeafResult {
            leaf_index: 0,
            in_solid: true,
            outside: true,
            has_valid_pvs: false,
        };
    }

    let mut node_idx = 0i32;
    let leaf_idx: i32;

    loop {
        if node_idx < 0 {
            // Negative node index → leaf
            leaf_idx = -1 - node_idx;
            break;
        }

        let node = &nodes[node_idx as usize];
        let plane = &planes[node.plane_id as usize];

        // dot(point, normal) - dist >= 0 → front child
        let dist = point.dot(plane.normal) - plane.dist;
        let child = if dist >= 0.0 {
            node.children[0]
        } else {
            node.children[1]
        };

        node_idx = child;
    }

    let leaf = &leaves[leaf_idx as usize];

    CameraLeafResult {
        leaf_index: leaf_idx as u32,
        in_solid: leaf.contents == -2, // CONTENTS_SOLID
        outside: leaf_idx == -1,
        has_valid_pvs: leaf.visofs >= 0,
    }
}

/// Conduct a full VIS decompression for the camera leaf and return the PVS set.
///
/// Returns None if:
/// - VIS data is empty or corrupted
/// - The camera is in solid or outside
/// - Decompression fails
pub fn camera_pvs(
    cam_point: &Vec3,
    vis_data: &[u8],
    nodes: &[lumps::Node],
    leaves: &[lumps::Leaf],
    planes: &[lumps::Plane],
) -> Option<PvsSet> {
    camera_pvs_with_scale(cam_point, vis_data, nodes, leaves, planes, 0.0254)
}

/// Conduct a full VIS decompression for an engine-space camera point with an
/// explicit BSP scale.
pub fn camera_pvs_with_scale(
    cam_point: &Vec3,
    vis_data: &[u8],
    nodes: &[lumps::Node],
    leaves: &[lumps::Leaf],
    planes: &[lumps::Plane],
    scale: f32,
) -> Option<PvsSet> {
    if vis_data.is_empty() {
        return None;
    }

    let cam_quake = engine_to_quake_space(*cam_point, scale);
    let cam = camera_leaf_index(&cam_quake, nodes, leaves, planes);
    if cam.in_solid || cam.outside {
        // Camera in solid: PVS empty, conservative fallback
        return None;
    }

    let leaf = &leaves[cam.leaf_index as usize];
    if leaf.visofs < 0 {
        return None;
    }

    let state = PvsState::new(leaves.len() as u32, vis_data);
    let pvs = state.decompress_for_leaf(cam.leaf_index as u32, leaf, vis_data);

    if !pvs.valid {
        return None;
    }

    Some(pvs)
}

fn engine_to_quake_space(v: Vec3, scale: f32) -> Vec3 {
    if scale.abs() < 1e-10 {
        return Vec3::ZERO;
    }
    let inv = 1.0 / scale;
    Vec3::new(v.x * inv, -v.z * inv, v.y * inv)
}

/// Build leaf membership maps: for each leaf, which faces reference it.
///
/// This uses the markfaces lump: each leaf has a range of markfaces, and each
/// markface references a face index.
pub fn build_leaf_membership(
    leaves: &[lumps::Leaf],
    markfaces: &[u32],
) -> Vec<Vec<u32>> {
    let num_faces = markfaces.iter().max().map(|m| *m + 1).unwrap_or(0) as usize;
    // Initialize empty per-face membership
    let mut face_members: Vec<Vec<u32>> = vec![Vec::new(); num_faces as usize];

    for (leaf_idx, leaf) in leaves.iter().enumerate() {
        // Only non-solid leaves contribute to face visibility
        if leaf.contents == -2 {
            continue;
        }
        let start = leaf.mark_id as usize;
        let end = (leaf.mark_id + leaf.mark_num) as usize;
        for &mf in &markfaces[start..end.min(markfaces.len())] {
            if (mf as usize) < face_members.len() {
                face_members[mf as usize].push(leaf_idx as u32);
            }
        }
    }

    // Sort and deduplicate per-face leaf membership
    for members in &mut face_members {
        members.sort_unstable();
        members.dedup();
    }

    face_members
}

/// Determine whether a leaf is "sealed" (no sky surfaces between it and the root).
#[allow(dead_code)]
pub fn is_leaf_sealed(
    leaf_index: u32,
    nodes: &[lumps::Node],
    planes: &[lumps::Plane],
    face_texinfo_flags: &[u32],
    faces: &[lumps::Face],
) -> bool {
    if nodes.is_empty() {
        return true;
    }

    // Walk from the leaf up to the root via parent pointers
    // Simplified: walk the tree from root to leaf, tracking sky exposure
    let mut current = 0i32; // root node
    let mut sky_exposed = false;

    if leaf_index >= leaves_count_from_nodes(nodes) as u32 {
        return true;
    }

    // Convert leaf index back to node child encoding: -(leaf_idx + 1)
    let target_leaf = -(leaf_index as i32) - 1;
    let mut path: Vec<(i32, usize)> = Vec::new(); // (node, child idx taken)

    // Navigate from root to leaf
    loop {
        if current < 0 {
            // Reached a leaf
            if current == target_leaf {
                break;
            } else {
                // Wrong path — shouldn't happen in a valid BSP
                sky_exposed = true;
                break;
            }
        }
        let node = &nodes[current as usize];
        let _plane = &planes[node.plane_id as usize];

        // Check node faces for sky flags
        for fi in node.face_id..node.face_id + node.face_num {
            if (fi as usize) < faces.len() {
                let face = &faces[fi as usize];
                if (face.texinfo_id as usize) < face_texinfo_flags.len() {
                    if face_texinfo_flags[face.texinfo_id as usize] & crate::materials::tex_flags::SURF_SKY != 0 {
                        sky_exposed = true;
                    }
                }
            }
        }

        // Navigate to children — we need to find the path to the target leaf
        // For simplicity: check both children to see which one leads to the target
        // This is approximate — a full implementation would use parent pointers
        if node.children[0] < 0 {
            if node.children[0] == target_leaf {
                current = node.children[0];
                path.push((current, 0));
                continue;
            }
        }
        if node.children[1] < 0 {
            if node.children[1] == target_leaf {
                current = node.children[1];
                path.push((current, 1));
                continue;
            }
        }

        // Both children are nodes — traverse to find which branch contains the leaf
        // For robust checking, we'd need the BSP bounds. For now, follow front first.
        if node.children[0] > 0 {
            current = node.children[0];
        } else {
            current = node.children[1];
        }
    }

    !sky_exposed
}

fn leaves_count_from_nodes(nodes: &[lumps::Node]) -> usize {
    // In a BSP tree, number of leaves = nodes + 1
    nodes.len() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_leaf(contents: i32, visofs: i32, mark_id: u32, mark_num: u32) -> lumps::Leaf {
        lumps::Leaf {
            contents,
            visofs,
            mins: [0; 3],
            maxs: [0; 3],
            mark_id,
            mark_num,
            ambient: [0; 4],
        }
    }

    #[test]
    fn pvs_rle_decompression_simple() {
        let leaves = vec![make_test_leaf(0, 0, 0, 0); 16];
        let num_leaves = leaves.len() as u32;
        let pvs_bytes = (num_leaves + 7) / 8; // 2 bytes for 16 leaves

        // Build compressed PVS: all leaves visible (each RLE cmd = 0xFF)
        // For 16 leaves: 2 bytes → two 0xFF commands
        let vis_data = vec![0xFFu8, 0xFFu8];

        let state = PvsState::new(num_leaves, &vis_data);

        let leaf = make_test_leaf(0, 0, 0, 0);
        let pvs = state.decompress_for_leaf(0, &leaf, &vis_data);
        assert!(pvs.valid);
        assert_eq!(pvs.bits.len(), pvs_bytes as usize);

        // All 16 leaves should be visible
        for i in 0..16 {
            assert!(pvs.is_visible(i), "leaf {} should be visible", i);
        }
    }

    #[test]
    fn pvs_rle_zero_fill() {
        let leaves = vec![make_test_leaf(0, 0, 0, 0); 32];
        let num_leaves = leaves.len() as u32;

        // RLE: 0x00 (zero command) + 0x04 (4×8=32 bits) → all zeros
        let vis_data = vec![0x00u8, 0x04u8];

        let state = PvsState::new(num_leaves, &vis_data);
        let leaf = make_test_leaf(0, 0, 0, 0);
        let pvs = state.decompress_for_leaf(0, &leaf, &vis_data);
        assert!(pvs.valid);
        // All leaves should be NOT visible
        for i in 0..32 {
            assert!(!pvs.is_visible(i), "leaf {} should NOT be visible", i);
        }
    }

    #[test]
    fn pvs_empty_vis_is_corrupt() {
        let state = PvsState::new(16, &[]);
        assert!(state.corrupt);
    }

    #[test]
    fn pvs_negative_visofs_conservative() {
        let state = PvsState::new(16, &[0xFF]);
        let leaf = make_test_leaf(0, -1, 0, 0);
        let pvs = state.decompress_for_leaf(0, &leaf, &[0xFF]);
        assert!(!pvs.valid);
    }

    #[test]
    fn camera_leaf_basic_traversal() {
        // Build a simple BSP: root splits at x=0, both children are leaves
        let planes = vec![lumps::Plane {
            normal: Vec3::X,
            dist: 0.0,
            plane_type: 0,
        }];
        let nodes = vec![lumps::Node {
            plane_id: 0,
            children: [-1, -2], // leaf 0 (front), leaf 1 (back)
            mins: [0; 3],
            maxs: [0; 3],
            face_id: 0,
            face_num: 0,
        }];
        let leaves = vec![
            make_test_leaf(0, 0, 0, 0), // leaf 0 (x >= 0)
            make_test_leaf(0, 0, 0, 0), // leaf 1 (x < 0)
        ];

        let cam = camera_leaf_index(&Vec3::new(10.0, 0.0, 0.0), &nodes, &leaves, &planes);
        assert_eq!(cam.leaf_index, 0);
        assert!(!cam.in_solid);

        let cam = camera_leaf_index(&Vec3::new(-10.0, 0.0, 0.0), &nodes, &leaves, &planes);
        assert_eq!(cam.leaf_index, 1);
        assert!(!cam.in_solid);
    }

    #[test]
    fn camera_on_plane_defaults_front() {
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
            make_test_leaf(0, 0, 0, 0),
            make_test_leaf(0, 0, 0, 0),
        ];

        // Exactly on the plane → front child (leaf 0)
        let cam = camera_leaf_index(&Vec3::new(0.0, 0.0, 0.0), &nodes, &leaves, &planes);
        assert_eq!(cam.leaf_index, 0);
    }

    #[test]
    fn build_leaf_membership_maps_faces() {
        let leaves = vec![
            make_test_leaf(0, 0, 0, 2), // leaf 0: faces [0, 1]
            make_test_leaf(0, 0, 2, 1), // leaf 1: face [2]
        ];
        let markfaces = vec![0u32, 1, 2];

        let members = build_leaf_membership(&leaves, &markfaces);
        assert_eq!(members.len(), 3); // 3 faces (0, 1, 2)
        assert_eq!(members[0], vec![0]); // face 0 referenced by leaf 0
        assert_eq!(members[1], vec![0]); // face 1 referenced by leaf 0
        assert_eq!(members[2], vec![1]); // face 2 referenced by leaf 1
    }

    #[test]
    fn pvs_conservative_all_visible() {
        let state = PvsState::new(8, &[]); // corrupt/empty VIS
        let fallback = state.conservative_fallback(0);
        assert!(!fallback.valid);
        // All bits should be set for the 8 leaves (1 byte)
        assert_eq!(fallback.bits.len(), 1);
        assert_eq!(fallback.bits[0], 0xFF);
    }
}
