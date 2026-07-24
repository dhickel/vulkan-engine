//! BSP PVS-aware culling for scene submission.
//!
//! Locates the camera leaf via BSP tree traversal, decompresses the PVS
//! set, and filters render batches before frustum/BVH culling.
//!
//! Contract: `bsp-spatial-physics.md` §4.

#[cfg(feature = "bsp")]
use crate::data::handles::{BspMaterialHandle, MeshHandle};
#[cfg(feature = "bsp")]
use crate::scene::render_submission::{FramePointLight, MAX_POINT_LIGHTS_GPU};
#[cfg(feature = "bsp")]
use bsp::extract::ExtractedBsp;
#[cfg(feature = "bsp")]
use bsp::visibility::{camera_leaf_index, CameraLeafResult, PvsSet};
#[cfg(feature = "bsp")]
use bsp::world::BspWorld;
#[cfg(feature = "bsp")]
use glam::Vec3;

#[cfg(feature = "bsp")]
const LIGHT_SCORE_EPSILON: f32 = 0.001;
#[cfg(feature = "bsp")]
const DEFAULT_LIGHT_HYSTERESIS_FRAMES: u8 = 2;
#[cfg(feature = "bsp")]
const TELEPORT_RESET_DISTANCE_SQUARED: f32 = 64.0 * 64.0;

#[cfg(feature = "bsp")]
fn default_style_intensities() -> [f32; 64] {
    let mut intensities = [0.0; 64];
    intensities[0] = 1.0;
    intensities
}

// ── BSP mount state ─────────────────────────────────────────────────────

#[cfg(feature = "bsp")]
/// Mounted BSP visibility state used by the renderer each frame.
#[derive(Debug, Clone)]
pub struct BspMountState {
    /// Number of PVS bits per row from world model 0's `visleafs` field.
    pub num_leaves: u32,
    /// Raw VIS data for PVS decompression.
    pub vis_data: Vec<u8>,
    /// Whether PVS data is globally usable.
    pub has_pvs: bool,
    /// Decompressed PVS set for the current frame (cached).
    pub current_pvs: Option<PvsSet>,
    /// The camera leaf for the current frame.
    pub camera_leaf: Option<CameraLeafResult>,
    /// BSP tree nodes (engine space, for camera leaf lookup).
    /// These are stored in Quake space for PVS leaf lookup;
    /// the camera position is converted to Quake space before traversal.
    pub nodes: Vec<bsp::lumps::Node>,
    /// BSP tree leaves.
    pub leaves: Vec<bsp::lumps::Leaf>,
    /// BSP tree planes (Quake space).
    pub planes: Vec<bsp::lumps::Plane>,
    /// BSP scale for engine-space ↔ Quake-space conversion.
    pub scale: f32,
    /// Leaf membership per face in PVS-bit space (`raw_leaf_index - 1`).
    pub leaf_membership: Vec<Vec<u32>>,
    /// Whether a BSP mount is active.
    pub active: bool,
    /// Mesh handles per source face index. Non-rendered faces use `MeshHandle::new(0, 0)`.
    pub face_meshes: Vec<MeshHandle>,
    /// BSP material handles per source face index.
    pub face_materials: Vec<Option<BspMaterialHandle>>,
    /// Render batches emitted by the bounded renderer upload plan.
    pub render_batches: Vec<bsp::geometry::RenderBatch>,
    /// GPU mesh handle aligned one-to-one with `render_batches`.
    pub batch_meshes: Vec<MeshHandle>,
    /// GPU material handle aligned one-to-one with `render_batches`.
    pub batch_materials: Vec<BspMaterialHandle>,
    /// BSP imported lights in deterministic source order.
    pub light_descriptors: Vec<bsp::extract::LightDescriptor>,
    /// Cached origin leaf for each imported light; `None` is treated as non-PVS fallback.
    pub light_leafs: Vec<Option<u32>>,
    /// Hysteresis state for BSP imported-light selection.
    light_selection: BspLightSelectionState,
    /// Per-model transforms for inline model draws (from simulation snapshot).
    /// Keyed by model_index (1..n). Identity for static world (model 0).
    pub inline_model_transforms: std::collections::HashMap<u32, glam::Mat4>,
    /// Per-model world-space bounds for inline model culling.
    /// Keyed by model_index (1..n).
    pub inline_model_bounds: std::collections::HashMap<u32, (glam::Vec3, glam::Vec3)>,
    /// Per-frame light-style intensities (64 elements).
    pub frame_style_intensities: [f32; 64],
    /// Per-frame liquid animation time.
    pub frame_liquid_time: f32,
}

#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
struct RetainedBspLight {
    source_order: usize,
    frames_remaining: u8,
}

#[cfg(feature = "bsp")]
#[derive(Debug, Clone)]
struct BspLightSelectionState {
    retain_frames: u8,
    retained: Vec<RetainedBspLight>,
    last_camera_pos: Option<Vec3>,
}

#[cfg(feature = "bsp")]
impl Default for BspLightSelectionState {
    fn default() -> Self {
        Self {
            retain_frames: DEFAULT_LIGHT_HYSTERESIS_FRAMES,
            retained: Vec::new(),
            last_camera_pos: None,
        }
    }
}

#[cfg(feature = "bsp")]
impl BspMountState {
    /// Create an empty, inactive mount state.
    pub fn new() -> Self {
        Self {
            num_leaves: 0,
            vis_data: Vec::new(),
            has_pvs: false,
            current_pvs: None,
            camera_leaf: None,
            nodes: Vec::new(),
            leaves: Vec::new(),
            planes: Vec::new(),
            scale: 0.0254,
            leaf_membership: Vec::new(),
            active: false,
            face_meshes: Vec::new(),
            face_materials: Vec::new(),
            render_batches: Vec::new(),
            batch_meshes: Vec::new(),
            batch_materials: Vec::new(),
            light_descriptors: Vec::new(),
            light_leafs: Vec::new(),
            light_selection: BspLightSelectionState::default(),
            inline_model_transforms: std::collections::HashMap::new(),
            inline_model_bounds: std::collections::HashMap::new(),
            frame_style_intensities: default_style_intensities(),
            frame_liquid_time: 0.0,
        }
    }

    /// Initialize mount state from a BspWorld.
    pub fn from_world(world: &BspWorld, scale: f32) -> Self {
        let vis_data = world.vis_data.clone();
        let max_visleaf_count = world.leaves.len().saturating_sub(1) as u32;
        let num_leaves = world
            .models
            .first()
            .and_then(|model| u32::try_from(model.visleafs).ok())
            .filter(|&count| count > 0 && count <= max_visleaf_count)
            .unwrap_or(0);
        let has_pvs = !vis_data.is_empty() && num_leaves > 0;

        Self {
            num_leaves,
            vis_data,
            has_pvs,
            current_pvs: None,
            camera_leaf: None,
            nodes: world.nodes.clone(),
            leaves: world.leaves.clone(),
            planes: world.planes.clone(),
            scale,
            leaf_membership: Vec::new(),
            active: true,
            face_meshes: Vec::new(),
            face_materials: Vec::new(),
            render_batches: Vec::new(),
            batch_meshes: Vec::new(),
            batch_materials: Vec::new(),
            light_descriptors: Vec::new(),
            light_leafs: Vec::new(),
            light_selection: BspLightSelectionState::default(),
            inline_model_transforms: std::collections::HashMap::new(),
            inline_model_bounds: std::collections::HashMap::new(),
            frame_style_intensities: default_style_intensities(),
            frame_liquid_time: 0.0,
        }
    }

    /// Initialize mount state from the neutral extraction visibility payload.
    pub fn from_extracted(extracted: &ExtractedBsp) -> Self {
        let visibility = &extracted.visibility;
        let has_pvs = extracted.has_pvs
            && visibility.visleaf_count > 0
            && !visibility.vis_data.is_empty()
            && !visibility.nodes.is_empty()
            && !visibility.leaves.is_empty()
            && !visibility.planes.is_empty();
        Self {
            num_leaves: visibility.visleaf_count,
            vis_data: visibility.vis_data.clone(),
            has_pvs,
            current_pvs: None,
            camera_leaf: None,
            nodes: visibility.nodes.clone(),
            leaves: visibility.leaves.clone(),
            planes: visibility.planes.clone(),
            scale: extracted.transform.scale,
            leaf_membership: extracted.leaf_membership.clone(),
            active: true,
            face_meshes: Vec::new(),
            face_materials: Vec::new(),
            render_batches: Vec::new(),
            batch_meshes: Vec::new(),
            batch_materials: Vec::new(),
            light_descriptors: Vec::new(),
            light_leafs: Vec::new(),
            light_selection: BspLightSelectionState::default(),
            inline_model_transforms: std::collections::HashMap::new(),
            inline_model_bounds: std::collections::HashMap::new(),
            frame_style_intensities: default_style_intensities(),
            frame_liquid_time: 0.0,
        }
    }

    /// Set the leaf membership data from the extracted BSP.
    pub fn set_leaf_membership(&mut self, members: Vec<Vec<u32>>) {
        self.leaf_membership = members;
    }

    /// Set renderer-facing BSP assets stored alongside the visibility mount.
    pub fn set_render_assets(
        &mut self,
        face_meshes: Vec<MeshHandle>,
        face_materials: Vec<Option<BspMaterialHandle>>,
        render_batches: Vec<bsp::geometry::RenderBatch>,
        batch_meshes: Vec<MeshHandle>,
        batch_materials: Vec<BspMaterialHandle>,
        light_descriptors: Vec<bsp::extract::LightDescriptor>,
    ) {
        debug_assert_eq!(render_batches.len(), batch_meshes.len());
        debug_assert_eq!(render_batches.len(), batch_materials.len());
        self.face_meshes = face_meshes;
        self.face_materials = face_materials;
        self.render_batches = render_batches;
        self.batch_meshes = batch_meshes;
        self.batch_materials = batch_materials;
        self.light_descriptors = light_descriptors;
        self.refresh_light_leafs();
        self.reset_light_selection();
    }

    /// Clear transient light-selection state, used on load/unload/reload/teleport.
    pub fn reset_light_selection(&mut self) {
        self.light_selection.retained.clear();
        self.light_selection.last_camera_pos = None;
    }

    /// Update the PVS and camera leaf for the current camera position.
    ///
    /// Returns `Some(&PvsSet)` if PVS data is valid, `None` otherwise
    /// (corrupt VIS, solid/outside camera, or PVS disabled).
    pub fn update_pvs(&mut self, camera_pos: Vec3) -> Option<&PvsSet> {
        if !self.has_pvs || self.vis_data.is_empty() || self.nodes.is_empty() {
            self.current_pvs = None;
            self.camera_leaf = None;
            return None;
        }

        // Convert camera position to Quake space for BSP tree traversal.
        let cam_quake = engine_to_quake_space(camera_pos, self.scale);

        let cam_leaf = camera_leaf_index(&cam_quake, &self.nodes, &self.leaves, &self.planes);

        self.camera_leaf = Some(cam_leaf.clone());

        if cam_leaf.in_solid
            || cam_leaf.outside
            || !cam_leaf.has_valid_pvs
            || cam_leaf.leaf_index == 0
            || cam_leaf.leaf_index > self.num_leaves
        {
            // Camera in solid, outside, or beyond world model 0's PVS range:
            // disable PVS conservatively for this frame.
            self.current_pvs = None;
            return None;
        }

        let Some(leaf) = self.leaves.get(cam_leaf.leaf_index as usize) else {
            self.current_pvs = None;
            self.camera_leaf = None;
            return None;
        };
        let state = bsp::visibility::PvsState::new(self.num_leaves, &self.vis_data);
        let pvs = state.decompress_for_leaf(cam_leaf.leaf_index, leaf, &self.vis_data);

        if !pvs.valid {
            self.current_pvs = None;
            return None;
        }

        self.current_pvs = Some(pvs);
        self.current_pvs.as_ref()
    }

    /// Check whether a set of PVS-bit indices intersects the current PVS.
    ///
    /// Returns `true` when:
    /// - PVS is not available (conservative: all visible)
    /// - Any leaf in `leaf_indices` is in the PVS set
    ///
    /// Returns `false` only when PVS is available and no leaf is in the set.
    pub fn batch_intersects_pvs(&self, leaf_indices: &[u32]) -> bool {
        let Some(pvs) = &self.current_pvs else {
            // No PVS: conservative all-visible.
            return true;
        };

        if !pvs.valid {
            return true;
        }

        leaf_indices.iter().any(|&leaf| pvs.is_visible(leaf))
    }

    /// Check whether a single PVS-bit index is in the current PVS.
    pub fn leaf_in_pvs(&self, leaf_index: u32) -> bool {
        let Some(pvs) = &self.current_pvs else {
            return true; // conservative
        };
        if !pvs.valid {
            return true;
        }
        pvs.is_visible(leaf_index)
    }

    /// Return the current camera leaf, if known.
    pub fn camera_leaf(&self) -> Option<&CameraLeafResult> {
        self.camera_leaf.as_ref()
    }

    /// Select BSP-imported lights for the current camera using PVS, stable scoring,
    /// hysteresis, and non-PVS fallback fill.
    pub fn select_light_indices_for_camera(
        &mut self,
        camera_pos: Vec3,
        max_lights: usize,
    ) -> Vec<usize> {
        let max_lights = max_lights.min(MAX_POINT_LIGHTS_GPU);
        if !self.active || max_lights == 0 || self.light_descriptors.is_empty() {
            self.reset_light_selection();
            return Vec::new();
        }

        self.reset_light_selection_on_discontinuity(camera_pos);
        if self.light_leafs.len() != self.light_descriptors.len() {
            self.refresh_light_leafs();
        }

        let primary = self.ranked_light_indices(camera_pos, true);
        let mut target = Vec::with_capacity(max_lights);
        for idx in primary {
            if target.len() >= max_lights {
                break;
            }
            target.push(idx);
        }

        if target.len() < max_lights {
            for idx in self.ranked_light_indices(camera_pos, false) {
                if target.len() >= max_lights {
                    break;
                }
                if !target.contains(&idx) {
                    target.push(idx);
                }
            }
        }

        self.apply_light_hysteresis(target, max_lights)
    }

    /// Select BSP-imported point lights and convert them into frame light records.
    pub fn select_frame_lights_for_camera(
        &mut self,
        camera_pos: Vec3,
        max_lights: usize,
    ) -> Vec<FramePointLight> {
        self.select_light_indices_for_camera(camera_pos, max_lights)
            .into_iter()
            .filter_map(|idx| self.light_descriptors.get(idx))
            .map(light_descriptor_to_frame_light)
            .collect()
    }

    /// Deactivate the mount, clearing all visibility state.
    pub fn deactivate(&mut self) {
        self.active = false;
        self.current_pvs = None;
        self.camera_leaf = None;
        self.reset_light_selection();
    }

    /// Activate the mount.
    pub fn activate(&mut self) {
        self.active = true;
    }

    fn reset_light_selection_on_discontinuity(&mut self, camera_pos: Vec3) {
        let discontinuity = self
            .light_selection
            .last_camera_pos
            .map(|last| last.distance_squared(camera_pos) > TELEPORT_RESET_DISTANCE_SQUARED)
            .unwrap_or(false);
        if discontinuity {
            self.light_selection.retained.clear();
        }
        self.light_selection.last_camera_pos = Some(camera_pos);
    }

    fn refresh_light_leafs(&mut self) {
        self.light_leafs = self
            .light_descriptors
            .iter()
            .map(|light| self.leaf_for_engine_position(light.origin))
            .collect();
    }

    fn leaf_for_engine_position(&self, position: Vec3) -> Option<u32> {
        if self.nodes.is_empty() || self.leaves.is_empty() || self.planes.is_empty() {
            return None;
        }
        let quake_pos = engine_to_quake_space(position, self.scale);
        let leaf = camera_leaf_index(&quake_pos, &self.nodes, &self.leaves, &self.planes);
        (!leaf.in_solid
            && !leaf.outside
            && leaf.has_valid_pvs
            && leaf.leaf_index > 0
            && leaf.leaf_index <= self.num_leaves)
            .then_some(leaf.leaf_index - 1)
    }

    fn ranked_light_indices(&self, camera_pos: Vec3, pvs_primary: bool) -> Vec<usize> {
        let mut scored: Vec<LightCandidateScore> = self
            .light_descriptors
            .iter()
            .enumerate()
            .filter(|(idx, _)| self.light_is_primary_pvs_candidate(*idx) == pvs_primary)
            .map(|(source_order, light)| LightCandidateScore {
                source_order,
                entity_index: light.entity_index,
                score: light_contribution_score(light, camera_pos),
            })
            .collect();

        scored.sort_by(compare_light_candidates);
        scored.into_iter().map(|entry| entry.source_order).collect()
    }

    fn light_is_primary_pvs_candidate(&self, source_order: usize) -> bool {
        let Some(pvs) = &self.current_pvs else {
            return true;
        };
        if !pvs.valid {
            return true;
        }
        self.light_leafs
            .get(source_order)
            .and_then(|leaf| *leaf)
            .map(|leaf| pvs.is_visible(leaf))
            .unwrap_or(false)
    }

    fn apply_light_hysteresis(&mut self, target: Vec<usize>, max_lights: usize) -> Vec<usize> {
        let mut final_indices = Vec::with_capacity(max_lights);

        for retained in &self.light_selection.retained {
            if retained.frames_remaining > 0
                && retained.source_order < self.light_descriptors.len()
                && !final_indices.contains(&retained.source_order)
                && final_indices.len() < max_lights
            {
                final_indices.push(retained.source_order);
            }
        }

        for idx in target.iter().copied() {
            if final_indices.len() >= max_lights {
                break;
            }
            if !final_indices.contains(&idx) {
                final_indices.push(idx);
            }
        }

        let mut next_retained = Vec::with_capacity(final_indices.len());
        for idx in final_indices.iter().copied() {
            let frames_remaining = if target.contains(&idx) {
                self.light_selection.retain_frames
            } else {
                self.light_selection
                    .retained
                    .iter()
                    .find(|retained| retained.source_order == idx)
                    .map(|retained| retained.frames_remaining.saturating_sub(1))
                    .unwrap_or(0)
            };
            if frames_remaining > 0 {
                next_retained.push(RetainedBspLight {
                    source_order: idx,
                    frames_remaining,
                });
            }
        }
        self.light_selection.retained = next_retained;

        final_indices
    }
}

#[cfg(feature = "bsp")]
impl Default for BspMountState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Camera leaf helpers ─────────────────────────────────────────────────

#[cfg(feature = "bsp")]
/// Convert an engine-space position to Quake space for BSP tree traversal.
fn engine_to_quake_space(v: Vec3, scale: f32) -> Vec3 {
    if scale.abs() < 1e-10 {
        return Vec3::ZERO;
    }
    let inv = 1.0 / scale;
    // Invert the QuakeToEngine transform: (x,y,z)_quake = (x/s, -z/s, y/s)
    Vec3::new(v.x * inv, -v.z * inv, v.y * inv)
}

#[cfg(feature = "bsp")]
#[derive(Debug, Copy, Clone)]
struct LightCandidateScore {
    source_order: usize,
    entity_index: u32,
    score: f32,
}

#[cfg(feature = "bsp")]
fn light_contribution_score(light: &bsp::extract::LightDescriptor, camera_pos: Vec3) -> f32 {
    let distance_squared = if light.origin.is_finite() && camera_pos.is_finite() {
        light.origin.distance_squared(camera_pos).max(0.0)
    } else {
        f32::INFINITY
    };
    light.intensity.max(0.0) / (distance_squared + LIGHT_SCORE_EPSILON)
}

#[cfg(feature = "bsp")]
fn compare_light_candidates(
    a: &LightCandidateScore,
    b: &LightCandidateScore,
) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.entity_index.cmp(&b.entity_index))
        .then_with(|| a.source_order.cmp(&b.source_order))
}

#[cfg(feature = "bsp")]
fn light_descriptor_to_frame_light(light: &bsp::extract::LightDescriptor) -> FramePointLight {
    FramePointLight {
        position: light.origin,
        color: Vec3::from_array(light.color).max(Vec3::ZERO),
        intensity: light.intensity.max(0.0),
        range: light.radius.max(0.001),
    }
}

// ── PVS batch filtering ─────────────────────────────────────────────────

#[cfg(feature = "bsp")]
/// Filter a list of render batches by PVS visibility.
///
/// Returns the subset of batches whose leaf membership intersects the
/// camera PVS. When PVS is not available, returns all batches unchanged
/// (conservative).
///
/// # Inline Model PVS Bypass
///
/// Inline models (`is_inline_model = true`) are **never** rejected by
/// static-world PVS. They are always passed through for conservative
/// frustum culling. Moving inline models must not be culled by the
/// static PVS of their original leaf membership.
pub fn visible_batch_indices(
    batches: &[bsp::geometry::RenderBatch],
    mount_state: &BspMountState,
) -> Vec<usize> {
    let Some(pvs) = mount_state.current_pvs.as_ref() else {
        return (0..batches.len()).collect();
    };
    if !mount_state.active || !pvs.valid {
        return (0..batches.len()).collect();
    }

    batches
        .iter()
        .enumerate()
        .filter_map(|(index, batch)| {
            let visible = !batch.pvs_eligible
                || batch.is_inline_model
                || batch.key.leaf_signature.is_empty()
                || batch
                    .key
                    .leaf_signature
                    .iter()
                    .any(|&leaf| pvs.is_visible(leaf));
            visible.then_some(index)
        })
        .collect()
}

pub fn filter_batches_by_pvs(
    batches: &[bsp::geometry::RenderBatch],
    _leaf_membership: &[Vec<u32>],
    mount_state: &BspMountState,
) -> Vec<bsp::geometry::RenderBatch> {
    visible_batch_indices(batches, mount_state)
        .into_iter()
        .filter_map(|index| batches.get(index).cloned())
        .collect()
}

/// Test whether a world-space AABB intersects the camera frustum.
///
/// Used for conservative culling of inline model batches that bypass
/// static PVS. Returns `true` when the frustum is not available
/// (conservative: all visible).
pub fn aabb_intersects_frustum(
    world_min: glam::Vec3,
    world_max: glam::Vec3,
    frustum: Option<&crate::data::camera::Frustum>,
) -> bool {
    let Some(frustum) = frustum else {
        return true;
    };
    let aabb = crate::data::camera::Aabb::from_min_max(world_min, world_max);
    frustum.intersects_aabb(&aabb)
}

// ── Corrupt VIS fallback helpers ────────────────────────────────────────

#[cfg(feature = "bsp")]
/// Determine whether PVS data should be globally disabled.
///
/// Returns true when: VIS lump is empty, or VIS data is malformed.
pub fn pvs_should_disable(vis_data: &[u8]) -> bool {
    vis_data.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "bsp")]
    #[test]
    fn engine_to_quake_space_roundtrip() {
        let scale = 0.0254;
        let engine_pos = Vec3::new(10.0, 5.0, -2.0);
        let quake_pos = engine_to_quake_space(engine_pos, scale);

        // Expected: (x/s, -z/s, y/s) = (10/0.0254, 2/0.0254, 5/0.0254)
        let expected = Vec3::new(10.0 / 0.0254, -(-2.0) / 0.0254, 5.0 / 0.0254);
        assert!((quake_pos - expected).length() < 0.01);
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn bsp_mount_state_default_inactive() {
        let state = BspMountState::new();
        assert!(!state.active);
        assert!(!state.has_pvs);
        assert!(state.current_pvs.is_none());
        assert_eq!(state.frame_style_intensities[0], 1.0);
        assert!(state.frame_style_intensities[1..].iter().all(|&v| v == 0.0));
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn update_pvs_empty_vis_returns_none() {
        let mut state = BspMountState::new();
        state.has_pvs = false;
        state.nodes = vec![];
        let result = state.update_pvs(Vec3::ZERO);
        assert!(result.is_none());
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn filter_batches_by_pvs_no_pvs_returns_all() {
        let batch = bsp::geometry::RenderBatch {
            key: bsp::geometry::BatchKey {
                leaf_signature: vec![0, 1, 2],
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
            },
            face_indices: vec![0, 1],
            pvs_eligible: true,
            is_inline_model: false,
            model_index: 0,
        };
        let batches = vec![batch.clone()];
        let mount = BspMountState::new();
        let members = vec![vec![0u32, 1, 2]];

        let result = filter_batches_by_pvs(&batches, &members, &mount);
        assert_eq!(result.len(), 1);
    }

    #[cfg(feature = "bsp")]
    #[test]
    fn filter_batches_by_pvs_inline_model_always_passes() {
        let batch = bsp::geometry::RenderBatch {
            key: bsp::geometry::BatchKey {
                leaf_signature: vec![],
                render_class: 0,
                material_identity: 0,
                lightmap_page: 0,
            },
            face_indices: vec![0],
            pvs_eligible: false,
            is_inline_model: true,
            model_index: 1,
        };
        let batches = vec![batch.clone()];
        let mount = BspMountState::new();
        let members = vec![vec![]];

        let result = filter_batches_by_pvs(&batches, &members, &mount);
        assert_eq!(result.len(), 1);
    }
}
