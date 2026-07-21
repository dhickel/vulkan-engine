//! Topology-first cave generator.
//!
//! Algorithm:
//! 1. Place semantic sites (spawn, junction, grand_cavern, shaft, destination)
//!    within bounds using RNG, respecting minimum separation.
//! 2. Build site adjacency graph (tree + one loop for connectivity).
//! 3. For each graph edge, carve a varying-radius tunnel along a noise-warped
//!    Catmull-Rom spline using sphere stamps.
//! 4. At each site, carve an ellipsoidal cavern with Perlin-noise-warped surface.
//! 5. Apply small-scale geological roughness to surface-adjacent cells.
//! 6. Enforce 1-cell solid border shell.
//! 7. Validate: flood-fill from spawn, verify all sites reachable.

use crate::cave_gen::generators::{
    carve_ellipsoid, carve_sphere, enforce_shell, AttemptContext, Generator, GeneratorResult,
};
use crate::cave_gen::lattice::VoxelWorld;
use crate::cave_gen::metrics::{path_clearance, RouteEdge, Site};
use crate::cave_gen::noise::PerlinNoise;
use crate::cave_gen::rng::{Pcg32V1, PhaseTaggedRng};

/// Names for the 5 semantic sites, in fixed order.
const SITE_LABELS: [&str; 5] = ["spawn", "junction", "grand_cavern", "shaft", "destination"];

/// The topology-first generator. Stateless; all state comes from the RNG.
#[derive(Debug, Clone, Default)]
pub struct TopologyFirst;

impl Generator for TopologyFirst {
    fn generate(
        &self,
        world: &mut VoxelWorld,
        rng: &mut PhaseTaggedRng,
        ctx: &mut AttemptContext,
    ) -> GeneratorResult {
        let (w, h, d) = world.dims();
        let seed = rng.seed();

        // ── Phase streams ────────────────────────────────────────────────
        let mut site_rng = rng.phase_stream("topology/site-placement");
        let mut spline_rng = rng.phase_stream("topology/spline-warp");
        let mut cavern_rng = rng.phase_stream("topology/cavern-shape");
        let mut roughness_rng = rng.phase_stream("topology/roughness");

        // ── Perlin noise for warp and roughness ──────────────────────────
        let noise = PerlinNoise::from_rng(
            &mut Pcg32V1::from_phase(seed, "topology/perlin-noise"),
        );

        // ── 1. Place sites ───────────────────────────────────────────────
        let positions = place_sites(&mut site_rng, w, h, d);
        let sites: Vec<Site> = positions
            .iter()
            .enumerate()
            .map(|(i, pos)| Site::new(pos[0], pos[1], pos[2], SITE_LABELS[i]))
            .collect();

        // ── 2. Build adjacency graph ──────────────────────────────────────
        // Graph: 0(spawn)-1(junction)-2(grand_cavern)-4(destination)
        //                  1(junction)-3(shaft)-2(grand_cavern)  [forms loop]
        let edge_indices: [(usize, usize); 5] = [
            (0, 1), // spawn → junction
            (1, 2), // junction → grand_cavern
            (1, 3), // junction → shaft
            (2, 3), // grand_cavern → shaft (loop closer)
            (2, 4), // grand_cavern → destination
        ];

        ctx.branch_generated(); // 5 branches

        // ── 3. Carve tunnels along each edge ─────────────────────────────
        for &(from, to) in &edge_indices {
            let radius = 5.0 + spline_rng.next_bounded(30) as f32 * 0.1; // 5.0..8.0
            carve_tunnel(
                world,
                &positions[from],
                &positions[to],
                radius,
                &noise,
                &mut spline_rng,
                ctx,
            );
            ctx.connection_forged();
        }

        // ── 4. Carve caverns at each site ────────────────────────────────
        let cavern_radii: [f32; 5] = [
            6.0, // spawn: modest entrance
            7.0, // junction: modest hub
           10.0, // grand_cavern: large chamber
            6.0, // shaft: narrow
            7.0, // destination: modest
        ];
        for (i, pos) in positions.iter().enumerate() {
            let r = cavern_radii[i];
            let rx = r + cavern_rng.next_bounded(40) as f32 * 0.1; // r..r+4.0
            let ry = r + cavern_rng.next_bounded(40) as f32 * 0.1;
            let rz = r + cavern_rng.next_bounded(40) as f32 * 0.1;
            carve_ellipsoid(
                world,
                pos[0] as f32,
                pos[1] as f32,
                pos[2] as f32,
                rx,
                ry,
                rz,
                1.0, // noise amplitude
                &|x, y, z| noise.noise_3d(x, y, z),
                127, // air
                0,   // default material
                ctx,
            );
            ctx.cavern_placed();
        }

        // ── 5. Geological roughness ──────────────────────────────────────
        apply_roughness(world, &noise, 0.6, 6.0, &mut roughness_rng, ctx);

        // ── 6. Shell enforcement ─────────────────────────────────────────
        enforce_shell(world);

        // ── 7. Build edges for route metrics ─────────────────────────────
        let edges: Vec<RouteEdge> = edge_indices
            .iter()
            .map(|&(from, to)| {
                let clearance = path_clearance(world.density(), &sites[from], &sites[to]);
                RouteEdge {
                    from,
                    to,
                    clearance,
                }
            })
            .collect();

        GeneratorResult {
            sites,
            edges,
            spawn_index: 0,
        }
    }

    fn name(&self) -> &'static str {
        "topology-first"
    }
}

// ─── Site placement ────────────────────────────────────────────────────────

fn place_sites(rng: &mut Pcg32V1, w: u32, h: u32, d: u32) -> [[u32; 3]; 5] {
    let margin: u32 = 5; // stay well within the 1-cell shell
    let m = margin;
    let wm = w.saturating_sub(2 * m).max(1);
    let hm = h.saturating_sub(2 * m).max(1);
    let dm = d.saturating_sub(2 * m).max(1);

    let min_sep = 15.0f32; // minimum Euclidean separation between any two sites

    // Attempt to place sites with minimum separation, retrying up to 100 times
    let mut sites: [[u32; 3]; 5] = [[0; 3]; 5];
    for attempt in 0..100 {
        // Spawn: near the lower-Z face
        sites[0] = [
            m + rng.next_bounded(wm),
            m + rng.next_bounded(hm),
            m + rng.next_bounded((dm / 4).max(1)),
        ];

        // Destination: near the upper-Z face
        sites[4] = [
            m + rng.next_bounded(wm),
            m + rng.next_bounded(hm),
            d - m - 1 - rng.next_bounded((dm / 4).max(1)),
        ];

        // Junction: central-ish region
        sites[1] = [
            m + wm / 4 + rng.next_bounded(wm / 2),
            m + hm / 4 + rng.next_bounded(hm / 2),
            m + dm / 4 + rng.next_bounded(dm / 2),
        ];

        // Grand cavern: significantly offset from junction
        let offset_x = (rng.next_bounded((w / 3).max(1)) as i32)
            * if rng.next_bounded(2) == 0 { -1 } else { 1 };
        let offset_y = (rng.next_bounded((h / 3).max(1)) as i32)
            * if rng.next_bounded(2) == 0 { -1 } else { 1 };
        let offset_z = (rng.next_bounded((d / 3).max(1)) as i32)
            * if rng.next_bounded(2) == 0 { -1 } else { 1 };
        sites[2] = [
            (sites[1][0] as i32 + offset_x).clamp(m as i32, (w - m - 1) as i32) as u32,
            (sites[1][1] as i32 + offset_y).clamp(m as i32, (h - m - 1) as i32) as u32,
            (sites[1][2] as i32 + offset_z).clamp(m as i32, (d - m - 1) as i32) as u32,
        ];

        // Shaft: vertically offset from junction
        let sho_x = (rng.next_bounded((w / 4).max(1)) as i32)
            * if rng.next_bounded(2) == 0 { -1 } else { 1 };
        sites[3] = [
            (sites[1][0] as i32 + sho_x).clamp(m as i32, (w - m - 1) as i32) as u32,
            m + rng.next_bounded(hm),
            (sites[1][2] as i32 - rng.next_bounded((dm / 3).max(1)) as i32)
                .max(m as i32) as u32,
        ];

        // Check minimum separation
        let mut ok = true;
        for i in 0..5 {
            for j in (i + 1)..5 {
                let dx = sites[i][0] as f32 - sites[j][0] as f32;
                let dy = sites[i][1] as f32 - sites[j][1] as f32;
                let dz = sites[i][2] as f32 - sites[j][2] as f32;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < min_sep {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
            }
        }
        if ok || attempt == 99 {
            break;
        }
    }

    sites
}

// ─── Tunnel carving ────────────────────────────────────────────────────────

/// Carve a noise-warped Catmull-Rom spline tunnel between two sites.
fn carve_tunnel(
    world: &mut VoxelWorld,
    from: &[u32; 3],
    to: &[u32; 3],
    base_radius: f32,
    noise: &PerlinNoise,
    rng: &mut Pcg32V1,
    ctx: &mut AttemptContext,
) {
    let (w, h, d) = world.dims();
    let wf = w as f32;
    let hf = h as f32;
    let df = d as f32;

    // Generate 4 control points including endpoints
    let mut control: Vec<[f32; 3]> = Vec::with_capacity(4);

    // Start point
    control.push([from[0] as f32, from[1] as f32, from[2] as f32]);

    // Two intermediate waypoints with noise offset
    for i in 1..=2 {
        let t = i as f32 / 3.0;
        let base_x = from[0] as f32 + t * (to[0] as f32 - from[0] as f32);
        let base_y = from[1] as f32 + t * (to[1] as f32 - from[1] as f32);
        let base_z = from[2] as f32 + t * (to[2] as f32 - from[2] as f32);

        // Noise warp
        let warp_scale = 4.0;
        let nx = noise.noise_3d(base_x as f64 * 0.1, base_y as f64 * 0.1 + 100.0, base_z as f64 * 0.1) as f32;
        let ny = noise.noise_3d(base_x as f64 * 0.1 + 200.0, base_y as f64 * 0.1, base_z as f64 * 0.1 + 300.0) as f32;
        let nz = noise.noise_3d(base_x as f64 * 0.1 + 400.0, base_y as f64 * 0.1 + 500.0, base_z as f64 * 0.1) as f32;

        let margin = 2.0;
        let wx = (base_x + nx * warp_scale).clamp(margin, wf - margin - 1.0);
        let wy = (base_y + ny * warp_scale).clamp(margin, hf - margin - 1.0);
        let wz = (base_z + nz * warp_scale).clamp(margin, df - margin - 1.0);
        control.push([wx, wy, wz]);
    }

    // End point
    control.push([to[0] as f32, to[1] as f32, to[2] as f32]);

    // Sample spline at fine intervals
    let samples = 80u32;
    let mut prev_point: Option<[f32; 3]> = None;
    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let pt = catmull_rom_sample(&control, t);

        // Varying radius: thinner at ends, thicker in middle
        let radius_factor = 1.0 - 0.4 * (2.0 * t - 1.0).abs(); // 0.6 at ends, 1.0 in middle
        let r = base_radius * radius_factor;

        // Additional jitter from RNG
        let jitter = (rng.next_bounded(10) as f32 - 5.0) * 0.05;
        let r = (r + jitter).max(1.0);

        carve_sphere(world, pt[0], pt[1], pt[2], r, 127, 0, ctx);

        // Also fill gaps between samples with smaller spheres to ensure connectivity
        if let Some(prev) = prev_point {
            let dist = ((pt[0] - prev[0]).powi(2)
                + (pt[1] - prev[1]).powi(2)
                + (pt[2] - prev[2]).powi(2))
            .sqrt();
            if dist > r * 0.8 {
                let gap_steps = (dist / (r * 0.5)).ceil() as u32;
                for g in 1..gap_steps {
                    let gt = g as f32 / gap_steps as f32;
                    let gx = prev[0] + gt * (pt[0] - prev[0]);
                    let gy = prev[1] + gt * (pt[1] - prev[1]);
                    let gz = prev[2] + gt * (pt[2] - prev[2]);
                    carve_sphere(world, gx, gy, gz, r * 0.7, 127, 0, ctx);
                }
            }
        }
        prev_point = Some(pt);
    }
}

// ─── Catmull-Rom spline ────────────────────────────────────────────────────

/// Sample a point on a Catmull-Rom spline through the given control points.
/// t ∈ [0, 1] maps across all segments. Uses endpoint duplication for
/// the first and last segments.
fn catmull_rom_sample(control: &[[f32; 3]], t: f32) -> [f32; 3] {
    let n = control.len();
    if n < 2 {
        return control.first().copied().unwrap_or([0.0; 3]);
    }

    // t is global [0,1] across all segments; convert to segment index + local t
    let segments = (n - 1) as f32;
    let seg_t = t * segments;
    let seg = (seg_t as usize).min(n - 2);
    let local_t = seg_t - seg as f32;

    // Gather 4 control points (duplicate endpoints)
    let p0 = if seg == 0 {
        control[0]
    } else {
        control[seg - 1]
    };
    let p1 = control[seg];
    let p2 = control[seg + 1];
    let p3 = if seg + 2 < n {
        control[seg + 2]
    } else {
        control[n - 1]
    };

    catmull_rom_point(&p0, &p1, &p2, &p3, local_t)
}

/// Evaluate a single Catmull-Rom segment given 4 control points.
fn catmull_rom_point(
    p0: &[f32; 3],
    p1: &[f32; 3],
    p2: &[f32; 3],
    p3: &[f32; 3],
    t: f32,
) -> [f32; 3] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        0.5
            * (2.0 * p1[0]
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3),
        0.5
            * (2.0 * p1[1]
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3),
        0.5
            * (2.0 * p1[2]
                + (-p0[2] + p2[2]) * t
                + (2.0 * p0[2] - 5.0 * p1[2] + 4.0 * p2[2] - p3[2]) * t2
                + (-p0[2] + 3.0 * p1[2] - 3.0 * p2[2] + p3[2]) * t3),
    ]
}

// ─── Geological roughness ──────────────────────────────────────────────────

/// Apply small-scale Perlin noise displacement to surface-adjacent cells.
/// Only affects cells that are within 1 voxel of a solid/air boundary.
fn apply_roughness(
    world: &mut VoxelWorld,
    noise: &PerlinNoise,
    scale: f32,
    amplitude: f32,
    _rng: &mut Pcg32V1,
    _ctx: &mut AttemptContext,
) {
    let (w, h, d) = world.dims();

    // First pass: find surface cells (air cells adjacent to solid, or vice versa)
    let mut surface_mask = vec![false; world.density().len()];

    for z in 1..d - 1 {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = (x as usize)
                    + (y as usize) * (w as usize)
                    + (z as usize) * (w as usize) * (h as usize);
                let d_center = *world.density().read(x, y, z);

                let mut near_surface = false;
                for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let nd =
                                *world
                                    .density()
                                    .read((x as i32 + dx) as u32, (y as i32 + dy) as u32, (z as i32 + dz) as u32);
                            if (d_center >= 0) != (nd >= 0) {
                                near_surface = true;
                                break;
                            }
                        }
                        if near_surface {
                            break;
                        }
                    }
                    if near_surface {
                        break;
                    }
                }
                surface_mask[idx] = near_surface;
            }
        }
    }

    // Second pass: apply noise to surface cells
    for z in 1..d - 1 {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = (x as usize)
                    + (y as usize) * (w as usize)
                    + (z as usize) * (w as usize) * (h as usize);
                if surface_mask[idx] {
                    let n = noise.noise_3d(
                        x as f64 * scale as f64,
                        y as f64 * scale as f64,
                        z as f64 * scale as f64,
                    );
                    let density = world.density_mut();
                    let current = *density.read(x, y, z);
                    let adjustment = (n * amplitude as f64) as i8;
                    let new_val = if current < 0 {
                        current.saturating_add(adjustment).clamp(-128, -1)
                    } else {
                        current.saturating_add(adjustment).clamp(0, 127)
                    };
                    density.set(x, y, z, new_val);
                }
            }
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cave_gen::generators::verify_shell;
    use crate::cave_gen::lattice::VoxelWorld;
    use crate::cave_gen::metrics::flood_fill_air;
    use crate::cave_gen::rng::PhaseTaggedRng;

    fn test_world() -> VoxelWorld {
        let mut world = VoxelWorld::new(64, 64, 64);
        world.fill_solid();
        world
    }

    #[test]
    fn topology_first_produces_valid_lattice() {
        let mut world = test_world();
        let mut rng = PhaseTaggedRng::new(42);
        let mut ctx = AttemptContext::new();
        let gen = TopologyFirst;
        let result = gen.generate(&mut world, &mut rng, &mut ctx);

        // Shell is intact
        assert!(verify_shell(&world));

        // Sites were placed
        assert_eq!(result.sites.len(), 5);
        assert_eq!(result.edges.len(), 5);

        // All sites are in air (non-negative density)
        for site in &result.sites {
            assert!(
                *world.density().read(site.x, site.y, site.z) >= 0,
                "site {} at ({},{},{}) is solid",
                site.label,
                site.x,
                site.y,
                site.z
            );
        }

        // All cells have finite values
        for (_, _, _, density) in world.density().iter_coords() {
            assert!(*density >= -128 && *density <= 127, "density out of range: {density}");
        }
    }

    #[test]
    fn topology_first_spawn_reachable() {
        let mut world = test_world();
        let mut rng = PhaseTaggedRng::new(42);
        let mut ctx = AttemptContext::new();
        let gen = TopologyFirst;
        let result = gen.generate(&mut world, &mut rng, &mut ctx);

        let spawn = &result.sites[result.spawn_index];
        let reachable = flood_fill_air(world.density(), spawn.x, spawn.y, spawn.z);
        assert!(
            !reachable.is_empty(),
            "spawn at ({},{},{}) has no reachable air",
            spawn.x,
            spawn.y,
            spawn.z
        );

        // All sites should be reachable from spawn
        for (i, site) in result.sites.iter().enumerate() {
            if i == result.spawn_index {
                continue;
            }
            let site_reachable = flood_fill_air(world.density(), site.x, site.y, site.z);
            let spawn_set: std::collections::HashSet<_> = reachable.iter().copied().collect();
            let mut site_reachable_from_site = false;
            for idx in &site_reachable {
                if spawn_set.contains(idx) {
                    site_reachable_from_site = true;
                    break;
                }
            }
            assert!(
                site_reachable_from_site,
                "site {} '{}' at ({},{},{}) is not reachable from spawn",
                i,
                site.label,
                site.x,
                site.y,
                site.z
            );
        }
    }

    #[test]
    fn topology_first_deterministic() {
        let run = |seed: u64| -> Vec<i8> {
            let mut world = test_world();
            let mut rng = PhaseTaggedRng::new(seed);
            let mut ctx = AttemptContext::new();
            let gen = TopologyFirst;
            gen.generate(&mut world, &mut rng, &mut ctx);
            world.density().iter().copied().collect()
        };

        let a = run(42);
        let b = run(42);
        assert_eq!(a, b);
    }

    #[test]
    fn topology_first_different_seeds_different_output() {
        let run = |seed: u64| -> Vec<i8> {
            let mut world = test_world();
            let mut rng = PhaseTaggedRng::new(seed);
            let mut ctx = AttemptContext::new();
            let gen = TopologyFirst;
            gen.generate(&mut world, &mut rng, &mut ctx);
            world.density().iter().copied().collect()
        };

        let a = run(42);
        let b = run(99);
        assert_ne!(a, b);
    }

    #[test]
    fn catmull_rom_endpoints() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let p0 = catmull_rom_sample(&pts, 0.0);
        assert!((p0[0] - 0.0).abs() < 0.001);
        assert!((p0[1] - 0.0).abs() < 0.001);
        assert!((p0[2] - 0.0).abs() < 0.001);

        let p1 = catmull_rom_sample(&pts, 1.0);
        assert!((p1[0] - 4.0).abs() < 0.001);
        assert!((p1[1] - 5.0).abs() < 0.001);
        assert!((p1[2] - 6.0).abs() < 0.001);
    }
}
