//! Generator trait and shared carving utilities.
//!
//! Ported from the voxel-cave spike. The `Generator` trait defines the
//! interface for cave generators that write into a `VoxelWorld`.
//!
//! Also provides a minimal `AttemptContext` for telemetry-style counting
//! (always Off mode — no clock reads, no allocation) to keep the generator
//! signature compatible with the spike.

pub mod topology_first;

use crate::cave_gen::lattice::{Density, MaterialTag, VoxelWorld, DEFAULT_MATERIAL};
use crate::cave_gen::metrics::{RouteEdge, Site};
use crate::cave_gen::rng::PhaseTaggedRng;

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
                let noise = noise_fn(
                    x as f64 * 0.3,
                    y as f64 * 0.3,
                    z as f64 * 0.3,
                ) as f32;
                let effective_dist = dist - noise * noise_amplitude / rx;

                if effective_dist <= 1.0 {
                    world.set_voxel(x, y, z, density, material);
                    ctx.cell_carved();
                }
            }
        }
    }
}

/// Enforce a 1-cell solid shell on all 6 faces of the lattice.
/// Sets border cells to fully solid (-128) with the default material.
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

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cave_gen::lattice::VoxelWorld;

    #[test]
    fn enforce_shell_all_borders_solid() {
        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_air();
        enforce_shell(&mut world);

        let (w, h, d) = world.dims();
        for z in 0..d {
            for y in 0..h {
                assert!(*world.density().read(0, y, z) < 0, "x=0 breach at (0,{y},{z})");
                assert!(
                    *world.density().read(w - 1, y, z) < 0,
                    "x=w-1 breach"
                );
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
    fn verify_shell_detects_breach() {
        let mut world = VoxelWorld::new(8, 8, 8);
        world.fill_solid();
        world.set_voxel(0, 4, 4, 127, 0);
        assert!(!verify_shell(&world));
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
}
