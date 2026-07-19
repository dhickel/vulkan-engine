//! Authored screen-space LOD selection with asymmetric hysteresis.
//!
//! ## Purpose
//! Provides deterministic LOD level selection based on normalized projected
//! screen radius computed from camera projection and world bounding sphere.
//! Uses asymmetric hysteresis to prevent oscillation at threshold boundaries.
//!
//! ## Compilation
//! Gated behind `--features scene-bvh`.

use crate::data::handles::MeshHandle;
use glam::{Mat4, Vec3, Vec4};

// ---------------------------------------------------------------------------
// LOD level and group
// ---------------------------------------------------------------------------

/// One level in an authored LOD chain.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshLodLevel {
    /// The mesh to use at this LOD level.
    pub mesh: MeshHandle,
    /// Normalized projected screen radius at which this level is entered.
    /// Higher values = closer to camera = more detail.
    /// Must be finite and nonnegative.
    pub enter_threshold: f32,
}

/// An authored LOD group: a chain of levels with hysteresis.
///
/// Levels must be sorted strictly descending by `enter_threshold`
/// (highest detail first, lowest detail last). The group must have
/// at least one level.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshLodGroup {
    pub levels: Vec<MeshLodLevel>,
    /// Hysteresis fraction (0.0 .. 1.0) applied asymmetrically:
    /// - Switching to a *coarser* level uses the full hysteresis band.
    /// - Switching to a *finer* level uses half the hysteresis band.
    pub hysteresis: f32,
}

/// Validate a LOD group's level chain.
#[derive(Debug, Clone, PartialEq)]
pub enum LodGroupError {
    Empty,
    NonFiniteThreshold { index: usize, value: f32 },
    NegativeThreshold { index: usize, value: f32 },
    NonDescendingThreshold { index: usize, this: f32, prev: f32 },
    InvalidHysteresis { value: f32 },
}

impl MeshLodGroup {
    /// Create a new LOD group, validating the level chain.
    pub fn new(levels: Vec<MeshLodLevel>, hysteresis: f32) -> Result<Self, LodGroupError> {
        if levels.is_empty() {
            return Err(LodGroupError::Empty);
        }
        if !hysteresis.is_finite() || !(0.0..=1.0).contains(&hysteresis) {
            return Err(LodGroupError::InvalidHysteresis { value: hysteresis });
        }

        for (i, level) in levels.iter().enumerate() {
            if !level.enter_threshold.is_finite() {
                return Err(LodGroupError::NonFiniteThreshold {
                    index: i,
                    value: level.enter_threshold,
                });
            }
            if level.enter_threshold < 0.0 {
                return Err(LodGroupError::NegativeThreshold {
                    index: i,
                    value: level.enter_threshold,
                });
            }
            if i > 0 {
                let prev = levels[i - 1].enter_threshold;
                if level.enter_threshold >= prev {
                    return Err(LodGroupError::NonDescendingThreshold {
                        index: i,
                        this: level.enter_threshold,
                        prev,
                    });
                }
            }
        }

        Ok(Self { levels, hysteresis })
    }

    /// Select the appropriate LOD level for a given projected screen radius.
    ///
    /// `current_level` is the index currently selected (if any).
    /// Returns the index of the selected level.
    ///
    /// ## Hysteresis logic
    /// - To switch to a *coarser* level (index increases): the projected radius
    ///   must fall below `enter_threshold[current] * (1.0 - hysteresis)`.
    /// - To switch to a *finer* level (index decreases): the projected radius
    ///   must exceed `enter_threshold[target] * (1.0 + hysteresis * 0.5)`.
    ///
    /// Without hysteresis, the level with the largest `enter_threshold` that
    /// is ≤ the projected radius is selected.
    pub fn select_level(
        &self,
        projected_radius: f32,
        current_level: Option<usize>,
    ) -> usize {
        debug_assert!(!self.levels.is_empty());

        if let Some(cur) = current_level {
            if cur < self.levels.len() {
                let cur_threshold = self.levels[cur].enter_threshold;

                // Check if we should switch to a coarser level (moving away).
                let down_threshold = cur_threshold * (1.0 - self.hysteresis);
                if projected_radius < down_threshold && cur + 1 < self.levels.len() {
                    // Find the coarsest level whose up-threshold is not exceeded.
                    return self.select_level_no_hysteresis(projected_radius);
                }

                // Check if we should switch to a finer level (moving closer).
                if cur > 0 {
                    let finer_idx = cur - 1;
                    let finer_threshold = self.levels[finer_idx].enter_threshold;
                    let up_threshold = finer_threshold * (1.0 + self.hysteresis * 0.5);
                    if projected_radius > up_threshold {
                        // Find the finest level.
                        return self.select_level_no_hysteresis(projected_radius);
                    }
                }

                return cur;
            }
        }

        self.select_level_no_hysteresis(projected_radius)
    }

    /// Select without hysteresis: find the level with largest enter_threshold
    /// that is ≤ projected_radius, defaulting to the coarsest level.
    fn select_level_no_hysteresis(&self, projected_radius: f32) -> usize {
        for (i, level) in self.levels.iter().enumerate() {
            if projected_radius >= level.enter_threshold {
                return i;
            }
        }
        // Projected radius is below all thresholds: use coarsest.
        self.levels.len() - 1
    }

    /// Returns the mesh for a selected level index.
    /// Returns `None` if the index is out of bounds.
    pub fn mesh_for_level(&self, level_index: usize) -> Option<MeshHandle> {
        self.levels.get(level_index).map(|l| l.mesh)
    }
}

// ---------------------------------------------------------------------------
// Projected screen radius
// ---------------------------------------------------------------------------

/// Compute the normalized projected screen radius of a world-space bounding sphere.
///
/// Uses the camera's projection matrix to compute the projected extent of the
/// sphere in normalized device coordinates (NDC), independent of viewport pixels.
///
/// Returns infinity (highest-detail selection) when projection is undefined or
/// the sphere is behind the camera; visibility remains the culler's responsibility.
pub fn projected_screen_radius(
    sphere_center: Vec3,
    sphere_radius: f32,
    view: &Mat4,
    projection: &Mat4,
) -> f32 {
    let view_proj = *projection * *view;

    // Transform sphere center to clip space.
    let clip_pos = view_proj * Vec4::from((sphere_center, 1.0));
    let w = clip_pos.w;

    if !w.is_finite() || w <= 0.0 || !sphere_radius.is_finite() {
        // Conservatively retain highest detail when projection is undefined or
        // the sphere intersects the camera plane. Culling owns visibility.
        return f32::INFINITY;
    }

    // Compute the screen-space extent of the sphere.
    // Approximate: project sphere center, then compute the extent of the
    // sphere radius in screen space using the projection scale.
    //
    // A sphere of radius r at distance d from the camera subtends
    // approximately r/d in the view direction. The projection matrix's
    // [0][0] element gives the horizontal scale.
    //
    // More precisely: use the projection matrix's x-scale to convert
    // world radius to NDC radius at distance w.
    let ndc_radius = projection.x_axis.x * sphere_radius / w;

    // Normalize by the NDC extent (2.0 for full [-1, 1] range).
    ndc_radius.abs() / 2.0
}

// ---------------------------------------------------------------------------
// Stale-handle fallback
// ---------------------------------------------------------------------------

/// Result of LOD level resolution for a node.
#[derive(Debug, Clone)]
pub struct LodSelection {
    /// Selected level index within the group.
    pub level_index: usize,
    /// The resolved mesh handle (may be a fallback if the authored mesh is stale).
    pub mesh: MeshHandle,
    /// True if the authored mesh handle was stale and a fallback was used.
    pub fallback_used: bool,
}

/// Resolve a LOD group's mesh for the current frame, checking mesh validity.
///
/// `is_valid` is a caller-provided function that checks whether a MeshHandle
/// is still valid (generation matches, loaded).
///
/// If the selected level's mesh is stale/invalid, falls back to the next
/// coarser valid level. If no level is valid, returns `None`.
pub fn resolve_lod_mesh(
    group: &MeshLodGroup,
    projected_radius: f32,
    current_level: Option<usize>,
    is_valid: &dyn Fn(MeshHandle) -> bool,
) -> Option<LodSelection> {
    let level_idx = group.select_level(projected_radius, current_level);

    // Try the selected level first, then fall back to coarser levels.
    for offset in 0..group.levels.len() {
        let idx = (level_idx + offset).min(group.levels.len() - 1);
        let mesh = group.levels[idx].mesh;
        if is_valid(mesh) {
            return Some(LodSelection {
                level_index: idx,
                mesh,
                fallback_used: idx != level_idx,
            });
        }
        if idx == group.levels.len() - 1 {
            break;
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_level(mesh_slot: u32, threshold: f32) -> MeshLodLevel {
        MeshLodLevel {
            mesh: MeshHandle::new(mesh_slot, 0),
            enter_threshold: threshold,
        }
    }

    fn mk_group(thresholds: &[f32], hysteresis: f32) -> MeshLodGroup {
        let levels: Vec<_> = thresholds
            .iter()
            .enumerate()
            .map(|(i, &t)| mk_level(i as u32, t))
            .collect();
        MeshLodGroup::new(levels, hysteresis).unwrap()
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    #[test]
    fn empty_group_rejected() {
        assert!(matches!(
            MeshLodGroup::new(vec![], 0.1),
            Err(LodGroupError::Empty)
        ));
    }

    #[test]
    fn non_descending_rejected() {
        let levels = vec![mk_level(0, 0.5), mk_level(1, 0.5)];
        assert!(matches!(
            MeshLodGroup::new(levels, 0.1),
            Err(LodGroupError::NonDescendingThreshold { .. })
        ));
    }

    #[test]
    fn ascending_rejected() {
        let levels = vec![mk_level(0, 0.3), mk_level(1, 0.5)];
        assert!(matches!(
            MeshLodGroup::new(levels, 0.1),
            Err(LodGroupError::NonDescendingThreshold { .. })
        ));
    }

    #[test]
    fn negative_threshold_rejected() {
        let levels = vec![mk_level(0, -0.1)];
        assert!(matches!(
            MeshLodGroup::new(levels, 0.1),
            Err(LodGroupError::NegativeThreshold { .. })
        ));
    }

    #[test]
    fn nan_threshold_rejected() {
        let levels = vec![MeshLodLevel {
            mesh: MeshHandle::new(0, 0),
            enter_threshold: f32::NAN,
        }];
        assert!(matches!(
            MeshLodGroup::new(levels, 0.1),
            Err(LodGroupError::NonFiniteThreshold { .. })
        ));
    }

    #[test]
    fn valid_group_accepted() {
        let group = mk_group(&[0.5, 0.2, 0.05], 0.1);
        assert_eq!(group.levels.len(), 3);
    }

    #[test]
    fn invalid_hysteresis_rejected() {
        for value in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
            assert!(matches!(
                MeshLodGroup::new(vec![mk_level(0, 0.5)], value),
                Err(LodGroupError::InvalidHysteresis { .. })
            ));
        }
    }

    // -----------------------------------------------------------------------
    // Selection without hysteresis
    // -----------------------------------------------------------------------

    #[test]
    fn select_finest_level() {
        let group = mk_group(&[0.5, 0.2, 0.05], 0.0);
        assert_eq!(group.select_level(0.6, None), 0);
        assert_eq!(group.select_level(0.5, None), 0);
        assert_eq!(group.select_level(0.3, None), 1);
        assert_eq!(group.select_level(0.2, None), 1);
        assert_eq!(group.select_level(0.1, None), 2);
        assert_eq!(group.select_level(0.05, None), 2);
        assert_eq!(group.select_level(0.01, None), 2); // coarsest
    }

    // -----------------------------------------------------------------------
    // Hysteresis — downward (to coarser)
    // -----------------------------------------------------------------------

    #[test]
    fn hysteresis_resists_downward_switch() {
        // Three levels: 0.5 (fine), 0.2 (medium), 0.05 (coarse).
        // Currently at level 0 (threshold 0.5). With 10% hysteresis,
        // won't switch down until radius < 0.5 * 0.9 = 0.45.
        let group = mk_group(&[0.5, 0.2, 0.05], 0.1);
        // At radius 0.46: above 0.45, should stay at level 0.
        assert_eq!(group.select_level(0.46, Some(0)), 0);
        // At radius 0.44: below 0.45, should switch to level 1.
        assert_eq!(group.select_level(0.44, Some(0)), 1);
    }

    #[test]
    fn hysteresis_resists_upward_switch() {
        // Currently at level 1 (threshold 0.2). To switch up to level 0
        // (threshold 0.5), need radius > 0.5 * 1.05 = 0.525.
        let group = mk_group(&[0.5, 0.2, 0.05], 0.1);
        // At radius 0.52: below 0.525, stay at level 1.
        assert_eq!(group.select_level(0.52, Some(1)), 1);
        // At radius 0.53: above 0.525, switch to level 0.
        assert_eq!(group.select_level(0.53, Some(1)), 0);
    }

    // -----------------------------------------------------------------------
    // Oscillation resistance
    // -----------------------------------------------------------------------

    #[test]
    fn no_oscillation_at_boundary() {
        let group = mk_group(&[0.5, 0.2], 0.1);
        // Jitter around the downward threshold (0.45).
        let mut current = Some(0usize);
        for radius in [0.46, 0.44, 0.46, 0.44, 0.46, 0.44] {
            let next = group.select_level(radius, current);
            current = Some(next);
        }
        // After the sequence, should settle at one level (not oscillate every frame).
        // With proper hysteresis, once we cross down to level 1, we stay there
        // unless we get above the up-threshold (0.525).
        assert_eq!(current.unwrap(), 1);
    }

    #[test]
    fn jitter_does_not_cause_flicker() {
        let group = mk_group(&[0.5, 0.2, 0.05], 0.1);
        // Start at level 0, jitter around 0.44-0.46 (near downward threshold 0.45).
        let mut current = Some(0usize);
        // Sequence that would oscillate without hysteresis.
        for _ in 0..20 {
            for r in [0.46, 0.44] {
                current = Some(group.select_level(r, current));
            }
        }
        // After the first down-cross, stays at level 1 because we never
        // cross the up-threshold of 0.525.
        assert_eq!(current.unwrap(), 1);
    }

    // -----------------------------------------------------------------------
    // Stale-handle fallback
    // -----------------------------------------------------------------------

    #[test]
    fn stale_falls_back_to_coarser() {
        let group = mk_group(&[0.5, 0.2, 0.05], 0.0);
        // Mesh 0 (level 0) is stale; should fall back to mesh 1 (level 1).
        let is_valid = |mh: MeshHandle| mh.slot != 0;
        let sel = resolve_lod_mesh(&group, 1.0, None, &is_valid).unwrap();
        assert_eq!(sel.level_index, 1);
        assert_eq!(sel.mesh.slot, 1);
        assert!(sel.fallback_used);
    }

    #[test]
    fn all_stale_returns_none() {
        let group = mk_group(&[0.5, 0.2], 0.0);
        let is_valid = |_mh: MeshHandle| false;
        assert!(resolve_lod_mesh(&group, 1.0, None, &is_valid).is_none());
    }

    #[test]
    fn current_level_preserved_within_band() {
        let group = mk_group(&[0.5, 0.2], 0.1);
        // At level 1 with radius 0.25 (within band), stays level 1.
        assert_eq!(group.select_level(0.25, Some(1)), 1);
    }

    // -----------------------------------------------------------------------
    // projected_screen_radius
    // -----------------------------------------------------------------------

    #[test]
    fn projected_radius_at_origin() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let r = projected_screen_radius(Vec3::ZERO, 1.0, &view, &proj);
        assert!(r > 0.0);
    }

    #[test]
    fn projected_radius_behind_camera_is_conservative() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let r = projected_screen_radius(Vec3::new(0.0, 0.0, 10.0), 1.0, &view, &proj);
        assert_eq!(r, f32::INFINITY);
    }

    #[test]
    fn projected_radius_falls_off_with_distance() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 0.0), Vec3::NEG_Z, Vec3::Y);
        let proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let r_near = projected_screen_radius(Vec3::new(0.0, 0.0, -2.0), 1.0, &view, &proj);
        let r_far = projected_screen_radius(Vec3::new(0.0, 0.0, -10.0), 1.0, &view, &proj);
        assert!(r_near > r_far);
    }

    #[test]
    fn deterministic_selection_across_repeated_submissions() {
        let group = mk_group(&[0.5, 0.2, 0.05], 0.1);
        // Same input across multiple "frames" should give same result.
        let level = group.select_level(0.3, None);
        for _ in 0..100 {
            assert_eq!(group.select_level(0.3, Some(level)), level);
        }
    }

    #[test]
    fn authored_fixture_contains_three_distinct_valid_levels() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/lod");
        let mut counts = Vec::new();
        for name in ["high", "medium", "low"] {
            let gltf = gltf::Gltf::open(root.join(format!("{name}.gltf")))
                .unwrap_or_else(|err| panic!("invalid {name}.gltf: {err}"));
            let primitive = gltf
                .meshes()
                .next()
                .and_then(|mesh| mesh.primitives().next())
                .unwrap_or_else(|| panic!("{name}.gltf has no mesh primitive"));
            let positions = primitive
                .get(&gltf::Semantic::Positions)
                .unwrap_or_else(|| panic!("{name}.gltf has no POSITION accessor"))
                .count();
            let indices = primitive
                .indices()
                .unwrap_or_else(|| panic!("{name}.gltf has no index accessor"))
                .count();
            assert!(positions >= 3 && indices >= 3);
            counts.push((positions, indices));
        }
        counts.sort_unstable();
        counts.dedup();
        assert_eq!(counts.len(), 3, "LOD levels must have distinct geometry");
    }
}
