//! Generator trait and shared carving utilities.
//!
//! Ported from the voxel-cave spike. The `Generator` trait defines the
//! interface for cave generators that write into a `VoxelWorld`.
//!
//! Also provides a minimal `AttemptContext` for telemetry-style counting
//! (always Off mode — no clock reads, no allocation) to keep the generator
//! signature compatible with the spike.
//!
//! # v2 additions
//! `InteriorRegion`, `GenError`, multi-layer shell enforcement/verification,
//! and deterministic carving primitives bounded to the shared interior.

pub mod topology_first;

use crate::cave_gen::lattice::{Density, MaterialTag, VoxelWorld, DEFAULT_MATERIAL};
use crate::cave_gen::metrics::{RouteEdge, Site};
use crate::cave_gen::rng::PhaseTaggedRng;
use serde::{Deserialize, Serialize};

// ─── AttemptContext (minimal, always Off) ──────────────────────────────────

/// Minimal per-attempt context compatible with the spike generator signature.
/// All operations are no-ops since telemetry does not affect canonical output.
#[derive(Debug, Clone)]
pub struct AttemptContext;

impl AttemptContext {
    /// Create a new context. Always Off mode.
    pub fn new() -> Self {
        Self
    }

    pub fn cell_carved(&mut self) {}
    pub fn cavern_placed(&mut self) {}
    pub fn branch_generated(&mut self) {}
    pub fn connection_forged(&mut self) {}
    pub fn finish_attempt(&mut self) {}
}

impl Default for AttemptContext {
    fn default() -> Self {
        Self::new()
    }
}

// ─── GeneratorResult ───────────────────────────────────────────────────────

/// Result returned by a generator after producing a cave.
#[derive(Debug, Clone)]
pub struct GeneratorResult {
    /// Semantic sites placed by the generator.
    pub sites: Vec<Site>,
    /// Edges in the site adjacency graph.
    pub edges: Vec<RouteEdge>,
    /// Index into `sites` of the spawn point.
    pub spawn_index: usize,
}

/// Stable semantic kind for the five v2 core sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreSiteKind {
    Spawn,
    Junction,
    GrandCavern,
    Shaft,
    Destination,
}

/// Stable role attached to a serialized v2 site.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SerializableSiteRole {
    Core { kind: CoreSiteKind },
    Auxiliary { aux_id: u8 },
}

/// Canonical, owned v2 site record. IDs are contiguous and coordinates are
/// lattice coordinates, so this type is stable across process boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableSite {
    pub id: u8,
    pub role: SerializableSiteRole,
    pub label: String,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// How a v2 route was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteKind {
    SplineBackbone,
    SplineExtra,
    Maze,
}

/// Canonical v2 route record. Endpoint IDs are always ordered low-to-high.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SerializableEdge {
    pub kind: RouteKind,
    pub from_site_id: u8,
    pub to_site_id: u8,
    /// Minimum distance from a carved route center cell to a solid cell.
    pub clearance: f32,
}

/// Stable generator-owned anchor position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnchorKind {
    CoreViewpoint { site_id: u8 },
    CoreLight { site_id: u8 },
    BackboneLight { from_site_id: u8, to_site_id: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableAnchor {
    pub id: u8,
    pub kind: AnchorKind,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// V2 result with the legacy metric view plus stable persisted records.
/// `GeneratorResult` remains unchanged so the v1 path and goldens are isolated.
#[derive(Debug, Clone)]
pub struct V2GeneratorResult {
    pub sites: Vec<Site>,
    pub edges: Vec<RouteEdge>,
    pub spawn_index: usize,
    pub serializable_sites: Vec<SerializableSite>,
    pub serializable_edges: Vec<SerializableEdge>,
    pub viewpoints: Vec<SerializableAnchor>,
    pub light_anchors: Vec<SerializableAnchor>,
}

/// Trait for cave generators that write into a VoxelWorld.
pub trait Generator {
    /// Generate a cave into the given world, which is already initialized
    /// (filled solid). The generator carves out air cells.
    fn generate(
        &self,
        world: &mut VoxelWorld,
        rng: &mut PhaseTaggedRng,
        ctx: &mut AttemptContext,
    ) -> GeneratorResult;

    /// Human-readable generator name.
    fn name(&self) -> &'static str;
}

// ─── InteriorRegion ────────────────────────────────────────────────────────

/// The writable interior: every cavern, spline, maze, and roughness write
/// must stay within these inclusive bounds. Derived from resolution and
/// configured shell thickness before allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteriorRegion {
    pub x_min: u32,
    pub x_max: u32,
    pub y_min: u32,
    pub y_max: u32,
    pub z_min: u32,
    pub z_max: u32,
    /// Maximum configured cavern/tunnel/maze operation radius, rounded up.
    pub max_operation_radius: u32,
    /// Maximum roughness displacement reach, rounded up.
    pub roughness_displacement_radius: u32,
    /// Clearance retained between every operation footprint and extraction edge.
    pub extraction_margin: u32,
    /// Total inset required for an operation center.
    pub operation_reach: u32,
}

impl InteriorRegion {
    /// Derive the writable interior from resolution and shell thickness.
    /// Returns `None` if shell*2 >= resolution (no interior).
    pub fn from_resolution(resolution: u32, shell_thickness: u32) -> Option<Self> {
        let s = shell_thickness;
        let s2 = s.checked_mul(2)?;
        if s2 >= resolution {
            return None;
        }
        let x_min = s;
        let x_max = resolution - 1 - s;
        let y_min = s;
        let y_max = resolution - 1 - s;
        let z_min = s;
        let z_max = resolution - 1 - s;
        if x_min > x_max || y_min > y_max || z_min > z_max {
            return None;
        }
        Some(Self {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            max_operation_radius: 0,
            roughness_displacement_radius: 0,
            extraction_margin: 0,
            operation_reach: 0,
        })
    }

    /// Derive and validate a writable interior together with the complete
    /// center-to-boundary reach of every carving/extraction operation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_operation_requirements(
        resolution: u32,
        shell_thickness: u32,
        max_cavern_radius: f32,
        max_tunnel_radius: f32,
        maze_radius: f32,
        roughness_displacement_radius: f32,
        extraction_margin: u32,
    ) -> Result<Self, GenError> {
        let interior =
            Self::from_resolution(resolution, shell_thickness).ok_or(GenError::InteriorEmpty {
                interior: 0,
                max_radius: max_cavern_radius.max(max_tunnel_radius).max(maze_radius),
            })?;
        let values = [
            max_cavern_radius,
            max_tunnel_radius,
            maze_radius,
            roughness_displacement_radius,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(GenError::InvalidConfig(
                "interior operation reaches must be finite and non-negative".into(),
            ));
        }

        let max_operation_radius = max_cavern_radius
            .max(max_tunnel_radius)
            .max(maze_radius)
            .ceil() as u32;
        let roughness_displacement_radius = roughness_displacement_radius.ceil() as u32;
        let operation_reach = max_operation_radius
            .checked_add(roughness_displacement_radius)
            .and_then(|reach| reach.checked_add(extraction_margin))
            .ok_or_else(|| GenError::InvalidConfig("interior operation reach overflow".into()))?;
        let required_span = operation_reach
            .checked_mul(2)
            .and_then(|diameter| diameter.checked_add(1))
            .ok_or_else(|| GenError::InvalidConfig("interior required span overflow".into()))?;
        let (sx, sy, sz) = interior.span();
        if sx < required_span || sy < required_span || sz < required_span {
            return Err(GenError::InteriorEmpty {
                interior: sx.min(sy).min(sz),
                max_radius: operation_reach as f32,
            });
        }

        Ok(Self {
            max_operation_radius,
            roughness_displacement_radius,
            extraction_margin,
            operation_reach,
            ..interior
        })
    }

    /// Check that a coordinate is inside the shell-derived writable interior.
    #[inline]
    pub fn contains(&self, x: u32, y: u32, z: u32) -> bool {
        x >= self.x_min
            && x <= self.x_max
            && y >= self.y_min
            && y <= self.y_max
            && z >= self.z_min
            && z <= self.z_max
    }

    /// Check that a coordinate can be used as an operation center without
    /// any configured operation reaching the extraction boundary.
    pub fn contains_operation_center(&self, x: u32, y: u32, z: u32) -> bool {
        let r = self.operation_reach;
        x >= self.x_min + r
            && x <= self.x_max - r
            && y >= self.y_min + r
            && y <= self.y_max - r
            && z >= self.z_min + r
            && z <= self.z_max - r
    }

    /// Inclusive bounds for operation centers.
    pub fn operation_center_bounds(&self) -> (u32, u32, u32, u32, u32, u32) {
        let r = self.operation_reach;
        (
            self.x_min + r,
            self.x_max - r,
            self.y_min + r,
            self.y_max - r,
            self.z_min + r,
            self.z_max - r,
        )
    }

    /// Span of the interior along each axis.
    pub fn span(&self) -> (u32, u32, u32) {
        (
            self.x_max - self.x_min + 1,
            self.y_max - self.y_min + 1,
            self.z_max - self.z_min + 1,
        )
    }
}

// ─── GenError ──────────────────────────────────────────────────────────────

/// Typed generation errors returned by v2 `generate_v2`.
#[derive(Debug, Clone, PartialEq)]
pub enum GenError {
    /// Interior region is empty or impossible for the required feature radii.
    InteriorEmpty { interior: u32, max_radius: f32 },
    /// Site placement exhausted attempts.
    SitePlacement {
        site_id: u8,
        attempts: u32,
        reason: String,
    },
    /// Maze planning exhausted all candidates or global budget.
    MazeExhausted {
        requested: u32,
        planned: u32,
        retries: u32,
        search: u64,
    },
    /// Post-generation shell verification found a breach.
    ShellBreach,
    /// A site is unreachable from spawn.
    UnreachableSite { site_id: u8 },
    /// A generated viewpoint or light anchor violates an air/interior invariant.
    InvalidAnchor {
        anchor_kind: String,
        anchor_id: u8,
        reason: String,
    },
    /// A route has no finite positive clearance from its center cells to walls.
    InvalidRouteClearance {
        from_site_id: u8,
        to_site_id: u8,
        clearance: f32,
    },
    /// Configuration is invalid for generation (passed-through from validation).
    InvalidConfig(String),
}

impl std::fmt::Display for GenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InteriorEmpty {
                interior,
                max_radius,
            } => {
                write!(
                    f,
                    "interior {interior} voxels cannot contain features requiring radius {max_radius:.1}"
                )
            }
            Self::SitePlacement {
                site_id,
                attempts,
                reason,
            } => {
                write!(
                    f,
                    "site {site_id} placement failed after {attempts} attempts: {reason}"
                )
            }
            Self::MazeExhausted {
                requested,
                planned,
                retries,
                search,
            } => {
                write!(
                    f,
                    "maze exhausted: requested {requested} links, planned {planned}, \
                     retries={retries}, search_nodes={search}"
                )
            }
            Self::ShellBreach => {
                write!(f, "shell breach detected after generation")
            }
            Self::UnreachableSite { site_id } => {
                write!(f, "site {site_id} is unreachable from spawn")
            }
            Self::InvalidAnchor {
                anchor_kind,
                anchor_id,
                reason,
            } => {
                write!(f, "{anchor_kind} anchor {anchor_id} is invalid: {reason}")
            }
            Self::InvalidRouteClearance {
                from_site_id,
                to_site_id,
                clearance,
            } => {
                write!(
                    f,
                    "route {from_site_id}-{to_site_id} has invalid clearance {clearance}"
                )
            }
            Self::InvalidConfig(msg) => {
                write!(f, "invalid config: {msg}")
            }
        }
    }
}

impl std::error::Error for GenError {}

// ─── Shared carving utilities ──────────────────────────────────────────────

/// Carve an air sphere at (cx, cy, cz) with the given radius.
/// Sets density to the given value (typically 127 for air) and material.
/// Respects lattice bounds but does NOT enforce a shell margin; callers
/// must ensure the sphere stays within the intended carving region.
pub fn carve_sphere(
    world: &mut VoxelWorld,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
    density: Density,
    material: MaterialTag,
    ctx: &mut AttemptContext,
) {
    let (w, h, d) = world.dims();
    let r2 = radius * radius;
    let min_x = ((cx - radius).floor() as i32).max(0) as u32;
    let max_x = ((cx + radius).ceil() as i32).min(w as i32 - 1) as u32;
    let min_y = ((cy - radius).floor() as i32).max(0) as u32;
    let max_y = ((cy + radius).ceil() as i32).min(h as i32 - 1) as u32;
    let min_z = ((cz - radius).floor() as i32).max(0) as u32;
    let max_z = ((cz + radius).ceil() as i32).min(d as i32 - 1) as u32;

    for z in min_z..=max_z {
        let dz = z as f32 - cz;
        for y in min_y..=max_y {
            let dy = y as f32 - cy;
            for x in min_x..=max_x {
                let dx = x as f32 - cx;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    world.set_voxel(x, y, z, density, material);
                    ctx.cell_carved();
                }
            }
        }
    }
}

/// Carve an air sphere bounded to an interior region.
/// Identical to `carve_sphere` but additionally clips each coordinate to
/// the interior bounds so that no write leaves the writable region.
pub fn carve_sphere_interior(
    world: &mut VoxelWorld,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
    density: Density,
    material: MaterialTag,
    interior: &InteriorRegion,
    ctx: &mut AttemptContext,
) {
    let r2 = radius * radius;
    let min_x = ((cx - radius).floor() as i32).max(interior.x_min as i32) as u32;
    let max_x = ((cx + radius).ceil() as i32).min(interior.x_max as i32) as u32;
    let min_y = ((cy - radius).floor() as i32).max(interior.y_min as i32) as u32;
    let max_y = ((cy + radius).ceil() as i32).min(interior.y_max as i32) as u32;
    let min_z = ((cz - radius).floor() as i32).max(interior.z_min as i32) as u32;
    let max_z = ((cz + radius).ceil() as i32).min(interior.z_max as i32) as u32;

    for z in min_z..=max_z {
        let dz = z as f32 - cz;
        for y in min_y..=max_y {
            let dy = y as f32 - cy;
            for x in min_x..=max_x {
                let dx = x as f32 - cx;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    world.set_voxel(x, y, z, density, material);
                    ctx.cell_carved();
                }
            }
        }
    }
}

/// Carve a noise-warped ellipsoid centered at (cx, cy, cz) with radii (rx, ry, rz).
/// The ellipsoid surface is perturbed by sampling Perlin noise along rays
/// from the center. `noise_fn` should return a value in [-1, 1]; the result
/// modulates the effective radius at each point.
pub fn carve_ellipsoid<F>(
    world: &mut VoxelWorld,
    cx: f32,
    cy: f32,
    cz: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    noise_amplitude: f32,
    noise_fn: &F,
    density: Density,
    material: MaterialTag,
    ctx: &mut AttemptContext,
) where
    F: Fn(f64, f64, f64) -> f64,
{
    let (w, h, d) = world.dims();
    let max_r = rx.max(ry).max(rz) + noise_amplitude.abs();
    let min_x = ((cx - max_r).floor() as i32).max(0) as u32;
    let max_x = ((cx + max_r).ceil() as i32).min(w as i32 - 1) as u32;
    let min_y = ((cy - max_r).floor() as i32).max(0) as u32;
    let max_y = ((cy + max_r).ceil() as i32).min(h as i32 - 1) as u32;
    let min_z = ((cz - max_r).floor() as i32).max(0) as u32;
    let max_z = ((cz + max_r).ceil() as i32).min(d as i32 - 1) as u32;

    for z in min_z..=max_z {
        let dz = z as f32 - cz;
        for y in min_y..=max_y {
            let dy = y as f32 - cy;
            for x in min_x..=max_x {
                let dx = x as f32 - cx;

                // Ellipsoid implicit distance (scaled to unit sphere)
                if rx <= 0.0 || ry <= 0.0 || rz <= 0.0 {
                    continue;
                }
                let nx = dx / rx;
                let ny = dy / ry;
                let nz = dz / rz;
                let dist = (nx * nx + ny * ny + nz * nz).sqrt();

                // Noise warp: modulate the effective surface position
                let noise = noise_fn(x as f64 * 0.3, y as f64 * 0.3, z as f64 * 0.3) as f32;
                let effective_dist = dist - noise * noise_amplitude / rx;

                if effective_dist <= 1.0 {
                    world.set_voxel(x, y, z, density, material);
                    ctx.cell_carved();
                }
            }
        }
    }
}

/// Carve an ellipsoid bounded to an interior region.
pub fn carve_ellipsoid_interior<F>(
    world: &mut VoxelWorld,
    cx: f32,
    cy: f32,
    cz: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    noise_amplitude: f32,
    noise_fn: &F,
    density: Density,
    material: MaterialTag,
    interior: &InteriorRegion,
    ctx: &mut AttemptContext,
) where
    F: Fn(f64, f64, f64) -> f64,
{
    let max_r = rx.max(ry).max(rz) + noise_amplitude.abs();
    let min_x = ((cx - max_r).floor() as i32).max(interior.x_min as i32) as u32;
    let max_x = ((cx + max_r).ceil() as i32).min(interior.x_max as i32) as u32;
    let min_y = ((cy - max_r).floor() as i32).max(interior.y_min as i32) as u32;
    let max_y = ((cy + max_r).ceil() as i32).min(interior.y_max as i32) as u32;
    let min_z = ((cz - max_r).floor() as i32).max(interior.z_min as i32) as u32;
    let max_z = ((cz + max_r).ceil() as i32).min(interior.z_max as i32) as u32;

    for z in min_z..=max_z {
        let dz = z as f32 - cz;
        for y in min_y..=max_y {
            let dy = y as f32 - cy;
            for x in min_x..=max_x {
                let dx = x as f32 - cx;

                if rx <= 0.0 || ry <= 0.0 || rz <= 0.0 {
                    continue;
                }
                let nx = dx / rx;
                let ny = dy / ry;
                let nz = dz / rz;
                let dist = (nx * nx + ny * ny + nz * nz).sqrt();

                let noise = noise_fn(x as f64 * 0.3, y as f64 * 0.3, z as f64 * 0.3) as f32;
                let effective_dist = dist - noise * noise_amplitude / rx;

                if effective_dist <= 1.0 {
                    world.set_voxel(x, y, z, density, material);
                    ctx.cell_carved();
                }
            }
        }
    }
}

// ─── Shell enforcement ─────────────────────────────────────────────────────

/// Enforce a 1-cell solid shell on all 6 faces of the lattice.
/// Sets border cells to fully solid (-128) with the default material.
/// This is the v1 shell — preserved unchanged.
pub fn enforce_shell(world: &mut VoxelWorld) {
    let (w, h, d) = world.dims();
    for z in 0..d {
        for y in 0..h {
            world.set_voxel(0, y, z, -128, DEFAULT_MATERIAL);
            world.set_voxel(w - 1, y, z, -128, DEFAULT_MATERIAL);
        }
    }
    for z in 0..d {
        for x in 0..w {
            world.set_voxel(x, 0, z, -128, DEFAULT_MATERIAL);
            world.set_voxel(x, h - 1, z, -128, DEFAULT_MATERIAL);
        }
    }
    for y in 0..h {
        for x in 0..w {
            world.set_voxel(x, y, 0, -128, DEFAULT_MATERIAL);
            world.set_voxel(x, y, d - 1, -128, DEFAULT_MATERIAL);
        }
    }
}

/// Enforce a multi-layer solid shell on all 6 faces (v2).
/// For every layer `k` in `0..thickness`, the cell at offset `k` from each
/// face is set to solid. This is a write, not just verification.
pub fn enforce_shell_multi(world: &mut VoxelWorld, thickness: u32) {
    let (w, h, d) = world.dims();
    for k in 0..thickness {
        if k >= w || k >= h || k >= d {
            break;
        }
        for z in 0..d {
            for y in 0..h {
                world.set_voxel(k, y, z, -128, DEFAULT_MATERIAL);
                world.set_voxel(w - 1 - k, y, z, -128, DEFAULT_MATERIAL);
            }
        }
        for z in 0..d {
            for x in 0..w {
                world.set_voxel(x, k, z, -128, DEFAULT_MATERIAL);
                world.set_voxel(x, h - 1 - k, z, -128, DEFAULT_MATERIAL);
            }
        }
        for y in 0..h {
            for x in 0..w {
                world.set_voxel(x, y, k, -128, DEFAULT_MATERIAL);
                world.set_voxel(x, y, d - 1 - k, -128, DEFAULT_MATERIAL);
            }
        }
    }
}

/// Verify that the 1-cell border shell is intact (all border cells are solid).
/// Returns true if the shell is valid, false if any border cell is non-solid.
pub fn verify_shell(world: &VoxelWorld) -> bool {
    let (w, h, d) = world.dims();
    for z in 0..d {
        for y in 0..h {
            if *world.density().read(0, y, z) >= 0 {
                return false;
            }
            if *world.density().read(w - 1, y, z) >= 0 {
                return false;
            }
        }
    }
    for z in 0..d {
        for x in 0..w {
            if *world.density().read(x, 0, z) >= 0 {
                return false;
            }
            if *world.density().read(x, h - 1, z) >= 0 {
                return false;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            if *world.density().read(x, y, 0) >= 0 {
                return false;
            }
            if *world.density().read(x, y, d - 1) >= 0 {
                return false;
            }
        }
    }
    true
}

/// Verify that every cell in the configured shell layers is solid (v2).
/// For each face and each layer k in `0..thickness`, every cell must be solid.
/// Returns true if all shell cells are solid, false on any breach.
pub fn verify_shell_multi(world: &VoxelWorld, thickness: u32) -> bool {
    let (w, h, d) = world.dims();
    for k in 0..thickness {
        if k >= w || k >= h || k >= d {
            break;
        }
        // X faces
        for z in 0..d {
            for y in 0..h {
                if *world.density().read(k, y, z) >= 0 {
                    return false;
                }
                if *world.density().read(w - 1 - k, y, z) >= 0 {
                    return false;
                }
            }
        }
        // Y faces
        for z in 0..d {
            for x in 0..w {
                if *world.density().read(x, k, z) >= 0 {
                    return false;
                }
                if *world.density().read(x, h - 1 - k, z) >= 0 {
                    return false;
                }
            }
        }
        // Z faces
        for y in 0..h {
            for x in 0..w {
                if *world.density().read(x, y, k) >= 0 {
                    return false;
                }
                if *world.density().read(x, y, d - 1 - k) >= 0 {
                    return false;
                }
            }
        }
    }
    true
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cave_gen::lattice::VoxelWorld;

    #[test]
    fn interior_region_from_resolution_valid() {
        let ir = InteriorRegion::from_resolution(64, 2).unwrap();
        assert_eq!(ir.x_min, 2);
        assert_eq!(ir.x_max, 61);
        assert_eq!(ir.y_min, 2);
        assert_eq!(ir.y_max, 61);
        assert_eq!(ir.z_min, 2);
        assert_eq!(ir.z_max, 61);
    }

    #[test]
    fn interior_region_no_interior() {
        assert!(InteriorRegion::from_resolution(64, 32).is_none());
        assert!(InteriorRegion::from_resolution(64, 40).is_none());
    }

    #[test]
    fn interior_contains() {
        let ir = InteriorRegion::from_resolution(64, 2).unwrap();
        assert!(ir.contains(2, 2, 2));
        assert!(ir.contains(61, 61, 61));
        assert!(!ir.contains(1, 2, 2));
        assert!(!ir.contains(62, 2, 2));
    }

    #[test]
    fn operation_interior_accounts_for_all_reaches() {
        let ir = InteriorRegion::from_operation_requirements(64, 2, 8.0, 3.0, 1.5, 2.0, 1).unwrap();
        assert_eq!(ir.max_operation_radius, 8);
        assert_eq!(ir.roughness_displacement_radius, 2);
        assert_eq!(ir.extraction_margin, 1);
        assert_eq!(ir.operation_reach, 11);
        assert!(ir.contains_operation_center(13, 13, 13));
        assert!(!ir.contains_operation_center(12, 13, 13));
    }

    #[test]
    fn operation_interior_rejects_insufficient_margin() {
        assert!(matches!(
            InteriorRegion::from_operation_requirements(32, 2, 14.0, 2.0, 1.0, 1.0, 1,),
            Err(GenError::InteriorEmpty { .. })
        ));
    }

    #[test]
    fn enforce_shell_all_borders_solid() {
        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_air();
        enforce_shell(&mut world);

        let (w, h, d) = world.dims();
        for z in 0..d {
            for y in 0..h {
                assert!(
                    *world.density().read(0, y, z) < 0,
                    "x=0 breach at (0,{y},{z})"
                );
                assert!(*world.density().read(w - 1, y, z) < 0, "x=w-1 breach");
            }
        }
        for z in 0..d {
            for x in 0..w {
                assert!(*world.density().read(x, 0, z) < 0, "y=0 breach");
                assert!(*world.density().read(x, h - 1, z) < 0, "y=h-1 breach");
            }
        }
        for y in 0..h {
            for x in 0..w {
                assert!(*world.density().read(x, y, 0) < 0, "z=0 breach");
                assert!(*world.density().read(x, y, d - 1) < 0, "z=d-1 breach");
            }
        }
        assert!(verify_shell(&world));
    }

    #[test]
    fn enforce_shell_multi_all_layers_solid() {
        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_air();
        enforce_shell_multi(&mut world, 3);

        assert!(verify_shell_multi(&world, 3));
        // Layer 3 should still be air
        assert_eq!(*world.density().read(3, 3, 3), 127);
    }

    #[test]
    fn verify_shell_detects_breach() {
        let mut world = VoxelWorld::new(8, 8, 8);
        world.fill_solid();
        world.set_voxel(0, 4, 4, 127, 0);
        assert!(!verify_shell(&world));
    }

    #[test]
    fn verify_shell_multi_detects_deep_breach() {
        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_solid();
        // Shell thickness 2 — layer 1 should be solid
        world.set_voxel(1, 4, 4, 127, 0); // breach at offset 1
        assert!(!verify_shell_multi(&world, 2));
    }

    #[test]
    fn carve_sphere_inside_bounds() {
        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_solid();
        let mut ctx = AttemptContext::new();
        carve_sphere(&mut world, 7.0, 7.0, 7.0, 3.0, 127, 1, &mut ctx);

        assert_eq!(*world.density().read(7, 7, 7), 127);
        assert_eq!(*world.density().read(0, 0, 0), -128);
    }

    #[test]
    fn carve_sphere_clips_to_bounds() {
        let mut world = VoxelWorld::new(8, 8, 8);
        world.fill_solid();
        let mut ctx = AttemptContext::new();
        carve_sphere(&mut world, 1.0, 1.0, 1.0, 10.0, 127, 1, &mut ctx);
        assert_eq!(*world.density().read(0, 0, 0), 127);
    }

    #[test]
    fn carve_sphere_interior_respects_bounds() {
        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_solid();
        let interior = InteriorRegion::from_resolution(16, 2).unwrap();
        let mut ctx = AttemptContext::new();
        carve_sphere_interior(&mut world, 7.0, 7.0, 7.0, 10.0, 127, 0, &interior, &mut ctx);

        // Shell cells (x=0,1,14,15 etc) must remain solid
        assert!(*world.density().read(0, 7, 7) < 0, "shell breach at x=0");
        assert!(*world.density().read(1, 7, 7) < 0, "shell breach at x=1");
        assert!(*world.density().read(14, 7, 7) < 0, "shell breach at x=14");
        assert!(*world.density().read(15, 7, 7) < 0, "shell breach at x=15");
        // Interior cells should be air
        assert_eq!(*world.density().read(7, 7, 7), 127);
    }
}
