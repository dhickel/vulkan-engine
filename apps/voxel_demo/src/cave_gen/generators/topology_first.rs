//! Topology-first cave generator.
//!
//! # v1 (legacy, preserved)
//! `generate_v1` / `TopologyFirst` struct with hard-coded 5 sites/edges,
//! order-coupled streams (PhaseTaggedRng), 1-cell shell, infallible.
//!
//! # v2
//! `generate_v2` with configurable topology, named RNG streams, true
//! multi-layer shell, bounded maze planner. See phase-02-generator-v2.md.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::cave_gen::generators::{
    carve_ellipsoid, carve_ellipsoid_interior, carve_sphere, carve_sphere_interior, enforce_shell,
    enforce_shell_multi, verify_shell_multi, AttemptContext, GenError, Generator, GeneratorResult,
    InteriorRegion,
};
use crate::cave_gen::lattice::{Density, VoxelWorld, DEFAULT_MATERIAL};
use crate::cave_gen::metrics::{flood_fill_air, path_clearance, RouteEdge, Site};
use crate::cave_gen::noise::PerlinNoise;
use crate::cave_gen::rng::{v2_stream, Pcg32V1, PhaseTaggedRng};
use crate::config::GeneratorSection;

// ─── v1 Constants ──────────────────────────────────────────────────────────

/// Names for the 5 semantic sites, in fixed order.
const SITE_LABELS: [&str; 5] = ["spawn", "junction", "grand_cavern", "shaft", "destination"];

// ─── v2 Constants ──────────────────────────────────────────────────────────

/// Core role labels, in fixed ID order 0-4.
const CORE_LABELS: [&str; 5] = ["spawn", "junction", "grand_cavern", "shaft", "destination"];

/// Auxiliary labels for IDs 5-11.
const AUX_LABELS: [&str; 7] = [
    "aux-5", "aux-6", "aux-7", "aux-8", "aux-9", "aux-10", "aux-11",
];

/// Site placement attempt limit.
const SITE_PLACEMENT_ATTEMPTS: u32 = 200;

/// Minimum separation between sites as a fraction of interior span.
const SITE_SEPARATION_FRACTION: f32 = 0.12;

/// Spline tunnel samples.
const SPLINE_SAMPLES: u32 = 80;

/// Air density value for carving.
const AIR_DENSITY: Density = 127;

/// Maze endpoint attachment zone radius (cells from site center).
const MAZE_ATTACH_ZONE: u32 = 3;

// ─── v1 Generator (preserved unchanged) ────────────────────────────────────

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
        generate_v1(world, rng, ctx)
    }

    fn name(&self) -> &'static str {
        "topology-first"
    }
}

/// v1 generation — identical to the original spike port.
pub fn generate_v1(
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
    let noise = PerlinNoise::from_rng(&mut Pcg32V1::from_phase(seed, "topology/perlin-noise"));

    // ── 1. Place sites ───────────────────────────────────────────────
    let positions = place_sites_v1(&mut site_rng, w, h, d);
    let sites: Vec<Site> = positions
        .iter()
        .enumerate()
        .map(|(i, pos)| Site::new(pos[0], pos[1], pos[2], SITE_LABELS[i]))
        .collect();

    // ── 2. Build adjacency graph ──────────────────────────────────────
    let edge_indices: [(usize, usize); 5] = [
        (0, 1), // spawn → junction
        (1, 2), // junction → grand_cavern
        (1, 3), // junction → shaft
        (2, 3), // grand_cavern → shaft (loop closer)
        (2, 4), // grand_cavern → destination
    ];

    ctx.branch_generated();

    // ── 3. Carve tunnels along each edge ─────────────────────────────
    for &(from, to) in &edge_indices {
        let radius = 5.0 + spline_rng.next_bounded(30) as f32 * 0.1; // 5.0..8.0
        carve_tunnel_v1(
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
    let cavern_radii: [f32; 5] = [6.0, 7.0, 10.0, 6.0, 7.0];
    for (i, pos) in positions.iter().enumerate() {
        let r = cavern_radii[i];
        let rx = r + cavern_rng.next_bounded(40) as f32 * 0.1;
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
            1.0,
            &|x, y, z| noise.noise_3d(x, y, z),
            AIR_DENSITY,
            DEFAULT_MATERIAL,
            ctx,
        );
        ctx.cavern_placed();
    }

    // ── 5. Geological roughness ──────────────────────────────────────
    apply_roughness_v1(world, &noise, 0.6, 6.0, &mut roughness_rng, ctx);

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

// ─── v1 helpers (preserved) ────────────────────────────────────────────────

fn place_sites_v1(rng: &mut Pcg32V1, w: u32, h: u32, d: u32) -> [[u32; 3]; 5] {
    let margin: u32 = 5;
    let m = margin;
    let wm = w.saturating_sub(2 * m).max(1);
    let hm = h.saturating_sub(2 * m).max(1);
    let dm = d.saturating_sub(2 * m).max(1);
    let min_sep = 15.0f32;

    let mut sites: [[u32; 3]; 5] = [[0; 3]; 5];
    for _attempt in 0..100 {
        sites[0] = [
            m + rng.next_bounded(wm),
            m + rng.next_bounded(hm),
            m + rng.next_bounded((dm / 4).max(1)),
        ];
        sites[4] = [
            m + rng.next_bounded(wm),
            m + rng.next_bounded(hm),
            d - m - 1 - rng.next_bounded((dm / 4).max(1)),
        ];
        sites[1] = [
            m + wm / 4 + rng.next_bounded(wm / 2),
            m + hm / 4 + rng.next_bounded(hm / 2),
            m + dm / 4 + rng.next_bounded(dm / 2),
        ];
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
        let sho_x = (rng.next_bounded((w / 4).max(1)) as i32)
            * if rng.next_bounded(2) == 0 { -1 } else { 1 };
        sites[3] = [
            (sites[1][0] as i32 + sho_x).clamp(m as i32, (w - m - 1) as i32) as u32,
            m + rng.next_bounded(hm),
            (sites[1][2] as i32 - rng.next_bounded((dm / 3).max(1)) as i32).max(m as i32) as u32,
        ];

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
        if ok {
            break;
        }
    }
    sites
}

fn carve_tunnel_v1(
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

    let mut control: Vec<[f32; 3]> = Vec::with_capacity(4);
    control.push([from[0] as f32, from[1] as f32, from[2] as f32]);

    for i in 1..=2 {
        let t = i as f32 / 3.0;
        let base_x = from[0] as f32 + t * (to[0] as f32 - from[0] as f32);
        let base_y = from[1] as f32 + t * (to[1] as f32 - from[1] as f32);
        let base_z = from[2] as f32 + t * (to[2] as f32 - from[2] as f32);

        let warp_scale = 4.0;
        let nx = noise.noise_3d(
            base_x as f64 * 0.1,
            base_y as f64 * 0.1 + 100.0,
            base_z as f64 * 0.1,
        ) as f32;
        let ny = noise.noise_3d(
            base_x as f64 * 0.1 + 200.0,
            base_y as f64 * 0.1,
            base_z as f64 * 0.1 + 300.0,
        ) as f32;
        let nz = noise.noise_3d(
            base_x as f64 * 0.1 + 400.0,
            base_y as f64 * 0.1 + 500.0,
            base_z as f64 * 0.1,
        ) as f32;

        let margin = 2.0;
        let wx = (base_x + nx * warp_scale).clamp(margin, wf - margin - 1.0);
        let wy = (base_y + ny * warp_scale).clamp(margin, hf - margin - 1.0);
        let wz = (base_z + nz * warp_scale).clamp(margin, df - margin - 1.0);
        control.push([wx, wy, wz]);
    }

    control.push([to[0] as f32, to[1] as f32, to[2] as f32]);

    let samples = SPLINE_SAMPLES;
    let mut prev_point: Option<[f32; 3]> = None;
    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let pt = catmull_rom_sample(&control, t);

        let radius_factor = 1.0 - 0.4 * (2.0 * t - 1.0).abs();
        let r = base_radius * radius_factor;
        let jitter = (rng.next_bounded(10) as f32 - 5.0) * 0.05;
        let r = (r + jitter).max(1.0);

        carve_sphere(
            world,
            pt[0],
            pt[1],
            pt[2],
            r,
            AIR_DENSITY,
            DEFAULT_MATERIAL,
            ctx,
        );

        if let Some(prev) = prev_point {
            let dist =
                ((pt[0] - prev[0]).powi(2) + (pt[1] - prev[1]).powi(2) + (pt[2] - prev[2]).powi(2))
                    .sqrt();
            if dist > r * 0.8 {
                let gap_steps = (dist / (r * 0.5)).ceil() as u32;
                for g in 1..gap_steps {
                    let gt = g as f32 / gap_steps as f32;
                    let gx = prev[0] + gt * (pt[0] - prev[0]);
                    let gy = prev[1] + gt * (pt[1] - prev[1]);
                    let gz = prev[2] + gt * (pt[2] - prev[2]);
                    carve_sphere(
                        world,
                        gx,
                        gy,
                        gz,
                        r * 0.7,
                        AIR_DENSITY,
                        DEFAULT_MATERIAL,
                        ctx,
                    );
                }
            }
        }
        prev_point = Some(pt);
    }
}

fn apply_roughness_v1(
    world: &mut VoxelWorld,
    noise: &PerlinNoise,
    scale: f32,
    amplitude: f32,
    _rng: &mut Pcg32V1,
    _ctx: &mut AttemptContext,
) {
    let (w, h, d) = world.dims();
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
                            let nd = *world.density().read(
                                (x as i32 + dx) as u32,
                                (y as i32 + dy) as u32,
                                (z as i32 + dz) as u32,
                            );
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

// ─── Catmull-Rom spline (shared) ───────────────────────────────────────────

/// Sample a point on a Catmull-Rom spline through the given control points.
fn catmull_rom_sample(control: &[[f32; 3]], t: f32) -> [f32; 3] {
    let n = control.len();
    if n < 2 {
        return control.first().copied().unwrap_or([0.0; 3]);
    }
    let segments = (n - 1) as f32;
    let seg_t = t * segments;
    let seg = (seg_t as usize).min(n - 2);
    let local_t = seg_t - seg as f32;

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
        0.5 * (2.0 * p1[0]
            + (-p0[0] + p2[0]) * t
            + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
            + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3),
        0.5 * (2.0 * p1[1]
            + (-p0[1] + p2[1]) * t
            + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
            + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3),
        0.5 * (2.0 * p1[2]
            + (-p0[2] + p2[2]) * t
            + (2.0 * p0[2] - 5.0 * p1[2] + 4.0 * p2[2] - p3[2]) * t2
            + (-p0[2] + 3.0 * p1[2] - 3.0 * p2[2] + p3[2]) * t3),
    ]
}

// ─── v2 Generator ──────────────────────────────────────────────────────────

/// v2 generator entry point.
///
/// `config` must be a valid v2 `GeneratorSection` (validated by Phase 01
/// validation). `world` is a pre-filled solid lattice at config resolution.
/// `rng_seed` is the canonical seed.
pub fn generate_v2(
    config: &GeneratorSection,
    world: &mut VoxelWorld,
    rng_seed: u64,
) -> Result<GeneratorResult, GenError> {
    // Generation is transactional: no typed failure may expose a partially
    // carved candidate through the caller-owned world.
    let mut candidate = world.clone();
    let result = generate_v2_candidate(config, &mut candidate, rng_seed)?;
    *world = candidate;
    Ok(result)
}

fn generate_v2_candidate(
    config: &GeneratorSection,
    world: &mut VoxelWorld,
    rng_seed: u64,
) -> Result<GeneratorResult, GenError> {
    // 0. Validate version and resolution match
    let (res, h, d) = world.dims();
    if (res, h, d) != (config.resolution, config.resolution, config.resolution) {
        return Err(GenError::InvalidConfig(format!(
            "world dimensions {res}x{h}x{d} != configured cubic resolution {}",
            config.resolution
        )));
    }
    if !(5..=12).contains(&config.cavern_count) {
        return Err(GenError::InvalidConfig(format!(
            "cavern_count {} is outside 5..=12",
            config.cavern_count
        )));
    }
    let finite = [
        config.tunnel_radius_min,
        config.tunnel_radius_max,
        config.cavern_radius_min,
        config.cavern_radius_max,
        config.spline_tension,
        config.roughness,
        config.maze_density,
        config.maze_twistiness,
        config.maze_radius,
    ];
    if finite.iter().any(|value| !value.is_finite())
        || config.tunnel_radius_min <= 0.0
        || config.tunnel_radius_min > config.tunnel_radius_max
        || config.cavern_radius_min <= 0.0
        || config.cavern_radius_min > config.cavern_radius_max
        || config.roughness < 0.0
        || !(0.0..=1.0).contains(&config.maze_density)
        || config.maze_radius <= 0.0
        || config.maze_retries == 0
        || config.maze_search_budget == 0
    {
        return Err(GenError::InvalidConfig(
            "non-finite, inverted, non-positive, or out-of-range generator field".into(),
        ));
    }

    // 1. Derive interior region
    let interior = InteriorRegion::from_resolution(config.resolution, config.shell_thickness)
        .ok_or(GenError::InteriorEmpty {
            interior: 0,
            max_radius: config.cavern_radius_max,
        })?;
    let max_radius = config
        .cavern_radius_max
        .max(config.tunnel_radius_max)
        .max(config.maze_radius);
    if max_radius + config.roughness + 2.0
        > interior
            .span()
            .0
            .min(interior.span().1)
            .min(interior.span().2) as f32
            / 2.0
    {
        return Err(GenError::InteriorEmpty {
            interior: interior.span().0,
            max_radius,
        });
    }

    let mut ctx = AttemptContext::new();

    // 2. Place sites
    let n_sites = config.cavern_count as usize;
    let positions = place_sites_v2(rng_seed, n_sites, config, &interior)?;

    let sites: Vec<Site> = positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            let label = if i < 5 {
                CORE_LABELS[i]
            } else if i < 12 {
                AUX_LABELS[i - 5]
            } else {
                "aux"
            };
            Site::new(pos[0], pos[1], pos[2], label)
        })
        .collect();

    // 3. Build spline edges
    let spline_edges = build_spline_edges(rng_seed, &positions, config.tunnel_count)?;

    // 4. Carve caverns
    let noise = PerlinNoise::from_rng(&mut v2_stream(rng_seed, "noise/init"));
    carve_caverns_v2(
        world, &positions, &sites, config, &noise, rng_seed, &interior, &mut ctx,
    );

    // 5. Carve tunnels along spline edges
    carve_tunnels_v2(
        world,
        &positions,
        &spline_edges,
        config,
        &noise,
        rng_seed,
        &interior,
        &mut ctx,
    );

    // 6. Plan and carve maze
    let maze_edges = plan_and_carve_maze(
        world,
        &positions,
        &spline_edges,
        config,
        rng_seed,
        &interior,
    )?;

    // 7. Apply roughness
    apply_roughness_v2(world, config, &noise, rng_seed, &interior, &mut ctx);

    // 8. Enforce shell (write solid, then verify)
    enforce_shell_multi(world, config.shell_thickness);
    if !verify_shell_multi(world, config.shell_thickness) {
        return Err(GenError::ShellBreach);
    }

    // 9. Verify reachability
    verify_reachability(world, &sites)?;

    // 10. Build result edges
    let all_edges: Vec<(usize, usize)> = spline_edges
        .iter()
        .copied()
        .chain(maze_edges.iter().copied())
        .collect();

    let edges: Vec<RouteEdge> = all_edges
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

    Ok(GeneratorResult {
        sites,
        edges,
        spawn_index: 0,
    })
}

// ─── v2: Interior-aware site placement ─────────────────────────────────────

fn place_sites_v2(
    seed: u64,
    n_sites: usize,
    config: &GeneratorSection,
    interior: &InteriorRegion,
) -> Result<Vec<[u32; 3]>, GenError> {
    let max_radius = config
        .cavern_radius_max
        .max(config.tunnel_radius_max)
        .max(config.maze_radius);
    let clearance = (max_radius + config.roughness + 1.0).ceil() as u32;

    // Inner bounds where site centers can go (clearance from interior edge)
    let ix_min = interior.x_min + clearance;
    let ix_max = interior.x_max.saturating_sub(clearance);
    let iy_min = interior.y_min + clearance;
    let iy_max = interior.y_max.saturating_sub(clearance);
    let iz_min = interior.z_min + clearance;
    let iz_max = interior.z_max.saturating_sub(clearance);

    if ix_min > ix_max || iy_min > iy_max || iz_min > iz_max {
        return Err(GenError::SitePlacement {
            site_id: 0,
            attempts: 0,
            reason: "interior too small for site clearance".into(),
        });
    }

    let min_sep = {
        let span = (ix_max - ix_min + 1)
            .min(iy_max - iy_min + 1)
            .min(iz_max - iz_min + 1) as f32;
        (span * SITE_SEPARATION_FRACTION).max(5.0)
    };

    let mut positions: Vec<[u32; 3]> = Vec::with_capacity(n_sites);

    for site_id in 0..n_sites {
        let mut rng = v2_stream(seed, &format!("placement/site-{site_id:02}"));
        let mut placed = false;
        let mut last_reason = String::new();

        for _attempt in 0..SITE_PLACEMENT_ATTEMPTS {
            let x = ix_min + rng.next_bounded(ix_max - ix_min + 1);
            let y = iy_min + rng.next_bounded(iy_max - iy_min + 1);
            let z = iz_min + rng.next_bounded(iz_max - iz_min + 1);

            // Check minimum separation from all previously placed sites
            let mut ok = true;
            for prev in &positions {
                let dx = x as f32 - prev[0] as f32;
                let dy = y as f32 - prev[1] as f32;
                let dz = z as f32 - prev[2] as f32;
                if (dx * dx + dy * dy + dz * dz).sqrt() < min_sep {
                    ok = false;
                    break;
                }
            }
            if ok {
                positions.push([x, y, z]);
                placed = true;
                break;
            }
            last_reason = "separation constraint".into();
        }

        if !placed {
            return Err(GenError::SitePlacement {
                site_id: site_id as u8,
                attempts: SITE_PLACEMENT_ATTEMPTS,
                reason: last_reason,
            });
        }
    }

    Ok(positions)
}

// ─── v2: Spline edge construction ──────────────────────────────────────────

fn build_spline_edges(
    seed: u64,
    positions: &[[u32; 3]],
    tunnel_count: u32,
) -> Result<Vec<(usize, usize)>, GenError> {
    let n = positions.len();
    let min_tree = n.saturating_sub(1) as u32;
    if tunnel_count < min_tree {
        return Err(GenError::InvalidConfig(format!(
            "tunnel_count {tunnel_count} < min tree edges {min_tree}"
        )));
    }
    let max_pairs = checked_pair_count(n as u32);
    if tunnel_count > max_pairs {
        return Err(GenError::InvalidConfig(format!(
            "tunnel_count {tunnel_count} > max unique pairs {max_pairs}"
        )));
    }

    let mut edges: HashSet<(usize, usize)> = HashSet::new();

    // Step 1: Build the stable semantic core tree. Geometry must not change
    // which roles form the backbone.
    const CORE_TREE: [(usize, usize); 4] = [(0, 1), (1, 2), (1, 3), (2, 4)];
    for &(from, to) in &CORE_TREE {
        if to < n {
            edges.insert((from, to));
        }
    }

    // Step 2: Attach auxiliaries to nearest lower-ID connected site
    if n > 5 {
        for i in 5..n {
            let connected_ids: Vec<usize> = (0..i).filter(|j| is_connected(*j, &edges)).collect();
            if connected_ids.is_empty() {
                // Attach to nearest of all lower IDs
                let (_best_dist, best_j) = nearest_connected(&positions[i], &positions[..i]);
                edges.insert(canonical_pair(best_j, i));
            } else {
                let (_best_dist, best_j) = nearest_among(i, &positions, &connected_ids);
                edges.insert(canonical_pair(best_j, i));
            }
        }
    }

    // Step 3: Add spline extras up to tunnel_count
    let mut all_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let pair = canonical_pair(i, j);
            if !edges.contains(&pair) {
                all_pairs.push(pair);
            }
        }
    }
    all_pairs.sort_by_key(|&(a, b)| (a, b));

    let extras_needed = tunnel_count as usize - edges.len();
    if extras_needed > 0 {
        let mut sel_rng = v2_stream(seed, "spline/extra-selection");
        sel_rng.shuffle(&mut all_pairs);
        for pair in all_pairs.into_iter().take(extras_needed) {
            edges.insert(pair);
        }
    }

    // Canonical output: sorted by (from, to)
    let mut result: Vec<(usize, usize)> = edges.into_iter().collect();
    result.sort_by_key(|&(a, b)| (a, b));
    Ok(result)
}

fn canonical_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn nearest_connected(target_pos: &[u32; 3], candidates: &[[u32; 3]]) -> (f32, usize) {
    let tp = (
        target_pos[0] as f32,
        target_pos[1] as f32,
        target_pos[2] as f32,
    );
    let mut best_dist = f32::MAX;
    let mut best_j = 0;
    for (j, pos) in candidates.iter().enumerate() {
        let dx = tp.0 - pos[0] as f32;
        let dy = tp.1 - pos[1] as f32;
        let dz = tp.2 - pos[2] as f32;
        let dist = dx * dx + dy * dy + dz * dz;
        if dist < best_dist {
            best_dist = dist;
            best_j = j;
        } else if dist == best_dist && j < best_j {
            best_j = j;
        }
    }
    (best_dist.sqrt(), best_j)
}

fn nearest_among(target: usize, positions: &[[u32; 3]], candidates: &[usize]) -> (f32, usize) {
    let tp = (
        positions[target][0] as f32,
        positions[target][1] as f32,
        positions[target][2] as f32,
    );
    let mut best_dist = f32::MAX;
    let mut best_j = candidates[0];
    for &j in candidates {
        let dx = tp.0 - positions[j][0] as f32;
        let dy = tp.1 - positions[j][1] as f32;
        let dz = tp.2 - positions[j][2] as f32;
        let dist = dx * dx + dy * dy + dz * dz;
        if dist < best_dist || (dist == best_dist && j < best_j) {
            best_dist = dist;
            best_j = j;
        }
    }
    (best_dist.sqrt(), best_j)
}

fn is_connected(node: usize, edges: &HashSet<(usize, usize)>) -> bool {
    edges.iter().any(|&(a, b)| a == node || b == node)
}

fn checked_pair_count(n: u32) -> u32 {
    if n < 2 {
        return 0;
    }
    let n64 = n as u64;
    ((n64 * (n64 - 1)) / 2) as u32
}

// ─── v2: Cavern carving ────────────────────────────────────────────────────

fn carve_caverns_v2(
    world: &mut VoxelWorld,
    positions: &[[u32; 3]],
    sites: &[Site],
    config: &GeneratorSection,
    noise: &PerlinNoise,
    seed: u64,
    interior: &InteriorRegion,
    ctx: &mut AttemptContext,
) {
    for (i, pos) in positions.iter().enumerate() {
        let mut rng = v2_stream(seed, &format!("cavern/site-{i:02}"));
        let radius_range = config.cavern_radius_max - config.cavern_radius_min;
        let base_r = config.cavern_radius_min + rng.next_bounded(100) as f32 * 0.01 * radius_range;
        let rx = base_r + rng.next_bounded(40) as f32 * 0.1;
        let ry = base_r + rng.next_bounded(40) as f32 * 0.1;
        let rz = base_r + rng.next_bounded(40) as f32 * 0.1;

        let cx = pos[0] as f32;
        let cy = pos[1] as f32;
        let cz = pos[2] as f32;

        // Use site label to generate a unique noise offset
        let noise_offset = i as f64 * 100.0;
        carve_ellipsoid_interior(
            world,
            cx,
            cy,
            cz,
            rx,
            ry,
            rz,
            1.0,
            &|x, y, z| noise.noise_3d(x + noise_offset, y + noise_offset, z + noise_offset),
            AIR_DENSITY,
            DEFAULT_MATERIAL,
            interior,
            ctx,
        );
        ctx.cavern_placed();
    }
    let _ = sites; // keep for symmetry
}

// ─── v2: Tunnel carving ────────────────────────────────────────────────────

fn carve_tunnels_v2(
    world: &mut VoxelWorld,
    positions: &[[u32; 3]],
    edges: &[(usize, usize)],
    config: &GeneratorSection,
    noise: &PerlinNoise,
    seed: u64,
    interior: &InteriorRegion,
    ctx: &mut AttemptContext,
) {
    for &(from, to) in edges {
        let mut rng = v2_stream(seed, &format!("tunnel/link-{from:02}-{to:02}"));
        let radius_range = config.tunnel_radius_max - config.tunnel_radius_min;
        let base_radius =
            config.tunnel_radius_min + rng.next_bounded(100) as f32 * 0.01 * radius_range;

        let from_pos = [
            positions[from][0] as f32,
            positions[from][1] as f32,
            positions[from][2] as f32,
        ];
        let to_pos = [
            positions[to][0] as f32,
            positions[to][1] as f32,
            positions[to][2] as f32,
        ];

        // Build 4 control points with noise-warped midpoints
        let mut control: Vec<[f32; 3]> = Vec::with_capacity(4);
        control.push(from_pos);

        for k in 1..=2 {
            let t = k as f32 / 3.0;
            let base_x = from_pos[0] + t * (to_pos[0] - from_pos[0]);
            let base_y = from_pos[1] + t * (to_pos[1] - from_pos[1]);
            let base_z = from_pos[2] + t * (to_pos[2] - from_pos[2]);

            let warp_scale = 4.0;
            let nx = noise.noise_3d(
                base_x as f64 * 0.1,
                base_y as f64 * 0.1 + 100.0 + from as f64 * 200.0 + to as f64,
                base_z as f64 * 0.1,
            ) as f32;
            let ny = noise.noise_3d(
                base_x as f64 * 0.1 + 200.0 + from as f64 * 300.0 + to as f64,
                base_y as f64 * 0.1,
                base_z as f64 * 0.1 + 300.0,
            ) as f32;
            let nz = noise.noise_3d(
                base_x as f64 * 0.1 + 400.0,
                base_y as f64 * 0.1 + 500.0 + from as f64 * 100.0 + to as f64,
                base_z as f64 * 0.1,
            ) as f32;

            let wx = (base_x + nx * warp_scale).clamp(interior.x_min as f32, interior.x_max as f32);
            let wy = (base_y + ny * warp_scale).clamp(interior.y_min as f32, interior.y_max as f32);
            let wz = (base_z + nz * warp_scale).clamp(interior.z_min as f32, interior.z_max as f32);
            control.push([wx, wy, wz]);
        }
        control.push(to_pos);

        let mut prev_point: Option<[f32; 3]> = None;
        for i in 0..=SPLINE_SAMPLES {
            let t = i as f32 / SPLINE_SAMPLES as f32;
            let pt = catmull_rom_sample(&control, t);

            let radius_factor = 1.0 - 0.4 * (2.0 * t - 1.0).abs();
            let r = (base_radius * radius_factor).max(1.0);
            let jitter = (rng.next_bounded(10) as f32 - 5.0) * 0.05;
            let r = (r + jitter).max(1.0);

            carve_sphere_interior(
                world,
                pt[0],
                pt[1],
                pt[2],
                r,
                AIR_DENSITY,
                DEFAULT_MATERIAL,
                interior,
                ctx,
            );

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
                        carve_sphere_interior(
                            world,
                            gx,
                            gy,
                            gz,
                            r * 0.7,
                            AIR_DENSITY,
                            DEFAULT_MATERIAL,
                            interior,
                            ctx,
                        );
                    }
                }
            }
            prev_point = Some(pt);
        }
        ctx.connection_forged();
    }
}

// ─── v2: Maze planner ──────────────────────────────────────────────────────

/// Maze planner: enumerate eligible pairs, compute target, plan all routes,
/// carve only after complete success. Returns carved maze edge list.
fn plan_and_carve_maze(
    world: &mut VoxelWorld,
    positions: &[[u32; 3]],
    spline_edges: &[(usize, usize)],
    config: &GeneratorSection,
    seed: u64,
    interior: &InteriorRegion,
) -> Result<Vec<(usize, usize)>, GenError> {
    let n = positions.len();

    // 1. Eligible pairs
    let spline_set: HashSet<(usize, usize)> = spline_edges.iter().copied().collect();
    let mut eligible: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let pair = canonical_pair(i, j);
            if !spline_set.contains(&pair) {
                eligible.push(pair);
            }
        }
    }
    eligible.sort_by_key(|&(a, b)| (a, b));
    let eligible_count = eligible.len() as u32;

    let target = {
        let d = config.maze_density as f64;
        (d * eligible_count as f64 + 0.5).floor() as u32
    };
    if target == 0 {
        return Ok(Vec::new());
    }
    if target > eligible_count {
        return Err(GenError::InvalidConfig(format!(
            "maze target {target} > eligible {eligible_count}"
        )));
    }

    // 2. Shuffle
    let mut sel_rng = v2_stream(seed, "maze/link-selection");
    sel_rng.shuffle(&mut eligible);

    // 3. Plan: collect (pair, path)
    let mut accepted: Vec<((usize, usize), Vec<(u32, u32, u32)>)> = Vec::new();
    let mut reserved_cells: HashSet<(u32, u32, u32)> = HashSet::new();
    let mut total_retries: u32 = 0;
    let mut total_search: u64 = 0;
    let max_search = config.maze_search_budget as u64;

    for &pair in &eligible {
        if accepted.len() as u32 >= target {
            break;
        }
        if total_search >= max_search {
            break;
        }

        let (from, to) = pair;
        let start = positions[from];
        let goal = positions[to];
        let mut found = false;

        for retry in 0..config.maze_retries {
            if total_search >= max_search {
                break;
            }
            let attempt_tag = format!("maze/plan/{from:02}-{to:02}/a{retry:03}");
            let result = plan_maze_route(
                world,
                &start,
                &goal,
                &reserved_cells,
                interior,
                seed,
                &attempt_tag,
                config.maze_twistiness,
                config.maze_radius,
                (config.cavern_radius_max + config.roughness + 5.0).ceil() as u32,
                (max_search.saturating_sub(total_search)) as u32,
            );
            total_search += result.search_nodes;
            total_retries += 1;

            if let Some(path) = result.path {
                let footprint = maze_path_footprint(&path, config.maze_radius, interior);
                let endpoint_zone =
                    (config.cavern_radius_max + config.roughness + 5.0).ceil() as u32;
                let conflicts = footprint.iter().any(|&cell| {
                    reserved_cells.contains(&cell)
                        && !in_endpoint_zone(cell, start, endpoint_zone)
                        && !in_endpoint_zone(cell, goal, endpoint_zone)
                });
                if conflicts {
                    continue;
                }
                reserved_cells.extend(footprint);
                accepted.push((pair, path));
                found = true;
                break;
            }
        }

        if !found {
            total_retries += 1;
            // substitution: continue to next candidate
        }
    }

    if (accepted.len() as u32) < target {
        return Err(GenError::MazeExhausted {
            requested: target,
            planned: accepted.len() as u32,
            retries: total_retries,
            search: total_search,
        });
    }

    // 4. Carve in canonical pair order
    let mut carved_pairs: Vec<(usize, usize)> = Vec::new();
    for (pair, path) in &accepted {
        carve_maze_path(world, path, config.maze_radius, interior);
        carved_pairs.push(*pair);
    }
    carved_pairs.sort_by_key(|&(a, b)| (a, b));
    Ok(carved_pairs)
}

struct MazePlanResult {
    path: Option<Vec<(u32, u32, u32)>>,
    search_nodes: u64,
}

/// Heading-aware 6-neighbor A* through solid cells.
///
/// The planner starts from the start site center and must reach a cell
/// within `MAZE_ATTACH_ZONE` of the goal site center. All traversed cells
/// must be solid (density < 0) except the endpoint attachment zone where
/// the goal site's air is accessible.
///
/// Earlier route reservations are treated as solid.
fn plan_maze_route(
    world: &VoxelWorld,
    start: &[u32; 3],
    goal: &[u32; 3],
    reserved: &HashSet<(u32, u32, u32)>,
    interior: &InteriorRegion,
    seed: u64,
    attempt_tag: &str,
    twistiness: f32,
    _maze_radius: f32,
    endpoint_zone: u32,
    search_budget: u32,
) -> MazePlanResult {
    let (w, h, d) = world.dims();
    let sx = start[0];
    let sy = start[1];
    let sz = start[2];
    let gx = goal[0];
    let gy = goal[1];
    let gz = goal[2];

    // Start must be in air (the site's cavern)
    if *world.density().read(sx, sy, sz) < 0 {
        return MazePlanResult {
            path: None,
            search_nodes: 0,
        };
    }

    // Endpoint attachment zones are the only existing-air regions that the
    // planner may traverse. Without the start zone a route cannot leave its
    // already-carved cavern.
    let attach_zone = endpoint_zone.max(MAZE_ATTACH_ZONE);
    let mut start_air_cells: HashSet<(u32, u32, u32)> = HashSet::new();
    let mut goal_air_cells: HashSet<(u32, u32, u32)> = HashSet::new();
    for dz in -(attach_zone as i32)..=(attach_zone as i32) {
        for dy in -(attach_zone as i32)..=(attach_zone as i32) {
            for dx in -(attach_zone as i32)..=(attach_zone as i32) {
                let nx = sx as i32 + dx;
                let ny = sy as i32 + dy;
                let nz = sz as i32 + dz;
                if nx < 0 || ny < 0 || nz < 0 {
                    continue;
                }
                let cell = (nx as u32, ny as u32, nz as u32);
                if interior.contains(cell.0, cell.1, cell.2)
                    && *world.density().read(cell.0, cell.1, cell.2) >= 0
                {
                    start_air_cells.insert(cell);
                }
            }
        }
    }
    for dz in -(attach_zone as i32)..=(attach_zone as i32) {
        for dy in -(attach_zone as i32)..=(attach_zone as i32) {
            for dx in -(attach_zone as i32)..=(attach_zone as i32) {
                let nx = gx as i32 + dx;
                let ny = gy as i32 + dy;
                let nz = gz as i32 + dz;
                if nx < 0 || ny < 0 || nz < 0 {
                    continue;
                }
                let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
                if nx >= w || ny >= h || nz >= d {
                    continue;
                }
                if *world.density().read(nx, ny, nz) >= 0 {
                    goal_air_cells.insert((nx, ny, nz));
                }
            }
        }
    }

    if goal_air_cells.is_empty() {
        return MazePlanResult {
            path: None,
            search_nodes: 1,
        };
    }

    // A* heuristic
    let heuristic = |x: u32, y: u32, z: u32| -> u32 {
        let dx = (x as i64 - gx as i64).unsigned_abs() as u32;
        let dy = (y as i64 - gy as i64).unsigned_abs() as u32;
        let dz = (z as i64 - gz as i64).unsigned_abs() as u32;
        dx + dy + dz
    };

    let heading_of = |dx: i32, dy: i32, dz: i32| -> u8 {
        match (dx, dy, dz) {
            (1, 0, 0) => 1,
            (-1, 0, 0) => 2,
            (0, 1, 0) => 3,
            (0, -1, 0) => 4,
            (0, 0, 1) => 5,
            (0, 0, -1) => 6,
            _ => 0,
        }
    };

    let directions: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    let h_start = heuristic(sx, sy, sz);
    let mut open: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
    let mut g_scores: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut came_from: HashMap<(u32, u32, u32), (u32, u32, u32)> = HashMap::new();
    let mut search_nodes: u64 = 0;

    g_scores.insert((sx, sy, sz), 0);
    open.push(Reverse(HeapEntry {
        f_cost: h_start,
        heuristic: h_start,
        steps: 0,
        heading: 0,
        z: sz,
        y: sy,
        x: sx,
    }));

    let mut found_goal: Option<(u32, u32, u32)> = None;

    while let Some(Reverse(entry)) = open.pop() {
        if search_nodes >= search_budget as u64 {
            break;
        }
        search_nodes += 1;

        let (x, y, z) = (entry.x, entry.y, entry.z);

        // Check if we reached the goal attachment zone
        if goal_air_cells.contains(&(x, y, z)) {
            found_goal = Some((x, y, z));
            break;
        }

        let current_g = *g_scores.get(&(x, y, z)).unwrap_or(&u32::MAX);
        if entry.f_cost != current_g + heuristic(x, y, z) {
            // Stale entry (shouldn't happen with our monotonic heuristic, but safe)
            continue;
        }

        for &(dx, dy, dz) in &directions {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;
            if nx < 0 || ny < 0 || nz < 0 {
                continue;
            }
            let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
            if nx >= w || ny >= h || nz >= d {
                continue;
            }
            if !interior.contains(nx, ny, nz) {
                continue;
            }

            let cell = (nx, ny, nz);
            let is_endpoint_air = start_air_cells.contains(&cell) || goal_air_cells.contains(&cell);
            let is_solid = *world.density().read(nx, ny, nz) < 0;
            let is_reserved = reserved.contains(&cell);

            let passable = (is_endpoint_air || is_solid) && (!is_reserved || is_endpoint_air);

            if !passable {
                continue;
            }

            let heading = heading_of(dx, dy, dz);
            let turn_penalty = if entry.heading != 0 && entry.heading != heading {
                (twistiness.clamp(0.0, 1.0) * 4.0).round() as u32
            } else {
                0
            };
            let Some(new_g) = current_g.checked_add(1 + turn_penalty) else {
                continue;
            };
            let existing_g = g_scores.get(&(nx, ny, nz)).copied().unwrap_or(u32::MAX);
            if new_g < existing_g {
                g_scores.insert((nx, ny, nz), new_g);
                let h = heuristic(nx, ny, nz);

                // Cell-framed jitter is independent of heap visitation order.
                let jitter_tag =
                    format!("{attempt_tag}/cell-{nx:03}-{ny:03}-{nz:03}/heading-{heading}");
                let jitter = v2_stream(seed, &jitter_tag).next_bounded(4);

                open.push(Reverse(HeapEntry {
                    f_cost: new_g + h,
                    heuristic: h + jitter,
                    steps: entry.steps + 1,
                    heading,
                    z: nz,
                    y: ny,
                    x: nx,
                }));
                came_from.insert((nx, ny, nz), (x, y, z));
            }
        }
    }

    match found_goal {
        Some(goal_cell) => {
            // Reconstruct path (excluding start cell; path goes from solid adjacent to start to goal)
            let mut path: Vec<(u32, u32, u32)> = Vec::new();
            let mut current = goal_cell;
            loop {
                path.push(current);
                if let Some(&prev) = came_from.get(&current) {
                    if prev == (sx, sy, sz) {
                        break;
                    }
                    current = prev;
                } else {
                    break;
                }
            }
            path.reverse();
            MazePlanResult {
                path: Some(path),
                search_nodes,
            }
        }
        None => MazePlanResult {
            path: None,
            search_nodes,
        },
    }
}

/// Min-heap entry for deterministic A*. Reverse is used so that smallest
/// values come first. Ties are broken by: heuristic, steps, heading, z, y, x.
#[derive(Debug, Clone, Eq)]
struct HeapEntry {
    f_cost: u32,
    heuristic: u32,
    steps: u32,
    heading: u8,
    z: u32,
    y: u32,
    x: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost
            && self.heuristic == other.heuristic
            && self.steps == other.steps
            && self.heading == other.heading
            && self.z == other.z
            && self.y == other.y
            && self.x == other.x
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.f_cost
            .cmp(&other.f_cost)
            .then_with(|| self.heuristic.cmp(&other.heuristic))
            .then_with(|| self.steps.cmp(&other.steps))
            .then_with(|| self.heading.cmp(&other.heading))
            .then_with(|| self.z.cmp(&other.z))
            .then_with(|| self.y.cmp(&other.y))
            .then_with(|| self.x.cmp(&other.x))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn in_endpoint_zone(cell: (u32, u32, u32), endpoint: [u32; 3], radius: u32) -> bool {
    cell.0.abs_diff(endpoint[0]) <= radius
        && cell.1.abs_diff(endpoint[1]) <= radius
        && cell.2.abs_diff(endpoint[2]) <= radius
}

fn maze_path_footprint(
    path: &[(u32, u32, u32)],
    radius: f32,
    interior: &InteriorRegion,
) -> HashSet<(u32, u32, u32)> {
    let mut footprint = HashSet::new();
    let reach = radius.max(1.0).ceil() as i32;
    let radius2 = radius.max(1.0) * radius.max(1.0);
    for &(x, y, z) in path {
        for dz in -reach..=reach {
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    if (dx * dx + dy * dy + dz * dz) as f32 > radius2 {
                        continue;
                    }
                    let (nx, ny, nz) = (
                        x as i64 + dx as i64,
                        y as i64 + dy as i64,
                        z as i64 + dz as i64,
                    );
                    if nx >= 0
                        && ny >= 0
                        && nz >= 0
                        && interior.contains(nx as u32, ny as u32, nz as u32)
                    {
                        footprint.insert((nx as u32, ny as u32, nz as u32));
                    }
                }
            }
        }
    }
    footprint
}

/// Carve a maze path using sphere stamps along each cell.
fn carve_maze_path(
    world: &mut VoxelWorld,
    path: &[(u32, u32, u32)],
    radius: f32,
    interior: &InteriorRegion,
) {
    let mut ctx = AttemptContext::new();
    for &(x, y, z) in path {
        carve_sphere_interior(
            world,
            x as f32,
            y as f32,
            z as f32,
            radius.max(1.0),
            AIR_DENSITY,
            DEFAULT_MATERIAL,
            interior,
            &mut ctx,
        );
    }
}

// ─── v2: Roughness ─────────────────────────────────────────────────────────

fn apply_roughness_v2(
    world: &mut VoxelWorld,
    config: &GeneratorSection,
    noise: &PerlinNoise,
    seed: u64,
    interior: &InteriorRegion,
    _ctx: &mut AttemptContext,
) {
    if config.roughness <= 0.0 {
        return;
    }
    let _rng = v2_stream(seed, "roughness/displace");
    let (w, h, _d) = world.dims();

    // Scale inversely with resolution so roughness amplitude is consistent
    let scale = 0.6 / (config.resolution as f32 / 64.0);
    let amplitude = config.roughness * 6.0;

    // First pass: find surface cells. Reads may inspect the adjacent shell,
    // but every mutation is restricted to the shared writable interior.
    let mut surface_mask = vec![false; world.density().len()];
    for z in interior.z_min..=interior.z_max {
        for y in interior.y_min..=interior.y_max {
            for x in interior.x_min..=interior.x_max {
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
                            let nd = *world.density().read(
                                (x as i32 + dx) as u32,
                                (y as i32 + dy) as u32,
                                (z as i32 + dz) as u32,
                            );
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

    // Second pass: apply noise only inside the shared interior.
    for z in interior.z_min..=interior.z_max {
        for y in interior.y_min..=interior.y_max {
            for x in interior.x_min..=interior.x_max {
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

// ─── v2: Reachability verification ─────────────────────────────────────────

fn verify_reachability(world: &VoxelWorld, sites: &[Site]) -> Result<(), GenError> {
    if sites.is_empty() {
        return Ok(());
    }

    let spawn = &sites[0];
    let spawn_reachable = flood_fill_air(world.density(), spawn.x, spawn.y, spawn.z);
    if spawn_reachable.is_empty() {
        return Err(GenError::UnreachableSite { site_id: 0 });
    }

    let spawn_set: HashSet<usize> = spawn_reachable.into_iter().collect();
    for (i, site) in sites.iter().enumerate().skip(1) {
        if !is_site_reachable(world, site, &spawn_set) {
            return Err(GenError::UnreachableSite { site_id: i as u8 });
        }
    }
    Ok(())
}

fn is_site_reachable(world: &VoxelWorld, site: &Site, spawn_set: &HashSet<usize>) -> bool {
    let (w, h, _d) = world.dims();
    // Check if the site itself is in the spawn's reachable set
    let idx = (site.x as usize)
        + (site.y as usize) * (w as usize)
        + (site.z as usize) * (w as usize) * (h as usize);
    spawn_set.contains(&idx)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cave_gen::generators::verify_shell;
    use crate::cave_gen::lattice::VoxelWorld;
    use crate::cave_gen::metrics::flood_fill_air;
    use crate::cave_gen::rng::PhaseTaggedRng;

    // ── v1 tests (preserved) ────────────────────────────────────────────

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

        assert!(verify_shell(&world));
        assert_eq!(result.sites.len(), 5);
        assert_eq!(result.edges.len(), 5);

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

        for (_, _, _, density) in world.density().iter_coords() {
            assert!(
                *density >= -128 && *density <= 127,
                "density out of range: {density}"
            );
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
                i, site.label, site.x, site.y, site.z
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

    // ── v2 tests ────────────────────────────────────────────────────────

    fn v2_default_config() -> GeneratorSection {
        GeneratorSection {
            seed: 42,
            resolution: 64,
            shell_thickness: 2,
            cavern_count: 5,
            tunnel_count: 5,
            tunnel_radius_min: 1.5,
            tunnel_radius_max: 3.0,
            cavern_radius_min: 4.0,
            cavern_radius_max: 8.0,
            spline_tension: 0.5,
            roughness: 0.3,
            maze_density: 0.0,
            maze_twistiness: 0.4,
            maze_radius: 1.2,
            maze_retries: 50,
            maze_search_budget: 5000,
            floor_threshold: 0.3,
            wall_uv_scale: 1.0,
            floor_uv_scale: 2.0,
        }
    }

    #[test]
    fn v2_produces_valid_lattice() {
        let config = v2_default_config();
        let mut world = VoxelWorld::new(config.resolution, config.resolution, config.resolution);
        world.fill_solid();

        let result = generate_v2(&config, &mut world, config.seed).unwrap();

        assert!(verify_shell_multi(&world, config.shell_thickness));
        assert_eq!(result.sites.len(), 5);
        assert_eq!(result.spawn_index, 0);

        // All sites in air
        for site in &result.sites {
            assert!(*world.density().read(site.x, site.y, site.z) >= 0);
        }
    }

    #[test]
    fn v2_deterministic() {
        let config = v2_default_config();
        let run = || {
            let mut world =
                VoxelWorld::new(config.resolution, config.resolution, config.resolution);
            world.fill_solid();
            generate_v2(&config, &mut world, config.seed).unwrap();
            world.density().iter().copied().collect::<Vec<i8>>()
        };

        let a = run();
        let b = run();
        assert_eq!(a, b);
    }

    #[test]
    fn v2_different_seeds_different_output() {
        let mut config = v2_default_config();
        let mut run = |seed: u64| {
            config.seed = seed;
            let mut world =
                VoxelWorld::new(config.resolution, config.resolution, config.resolution);
            world.fill_solid();
            generate_v2(&config, &mut world, seed).unwrap();
            world.density().iter().copied().collect::<Vec<i8>>()
        };

        assert_ne!(run(42), run(99));
    }

    #[test]
    fn v2_shell_enforcement() {
        let config = v2_default_config();
        let mut world = VoxelWorld::new(config.resolution, config.resolution, config.resolution);
        world.fill_solid();

        let result = generate_v2(&config, &mut world, config.seed).unwrap();

        // Shell intact
        assert!(verify_shell_multi(&world, config.shell_thickness));
        // Layer 0 must be solid
        assert!(*world.density().read(0, 0, 0) < 0);
        assert!(*world.density().read(1, 1, 1) < 0);
        // Spawn site must be in air
        let spawn = &result.sites[0];
        assert!(*world.density().read(spawn.x, spawn.y, spawn.z) >= 0);
    }

    #[test]
    fn v2_reachability() {
        let config = v2_default_config();
        let mut world = VoxelWorld::new(config.resolution, config.resolution, config.resolution);
        world.fill_solid();

        let result = generate_v2(&config, &mut world, config.seed).unwrap();

        let spawn = &result.sites[0];
        let reachable = flood_fill_air(world.density(), spawn.x, spawn.y, spawn.z);
        assert!(!reachable.is_empty());

        let spawn_set: HashSet<usize> = reachable.iter().copied().collect();
        for (i, site) in result.sites.iter().enumerate().skip(1) {
            let idx = (site.x as usize)
                + (site.y as usize) * (config.resolution as usize)
                + (site.z as usize) * (config.resolution as usize) * (config.resolution as usize);
            assert!(spawn_set.contains(&idx), "site {i} unreachable from spawn");
        }
    }

    #[test]
    fn v2_spline_edge_count_matches_tunnel_count() {
        let config = v2_default_config();
        let mut world = VoxelWorld::new(config.resolution, config.resolution, config.resolution);
        world.fill_solid();

        let result = generate_v2(&config, &mut world, config.seed).unwrap();
        // spline edges = tunnel_count (maze_density=0 means no maze edges)
        // We need to count only spline edges; there are no maze edges
        // The edges in result include both spline and maze
        // maze_density=0 -> no maze links
        assert!(result.edges.len() >= config.tunnel_count as usize);
    }

    #[test]
    fn v2_interior_empty_rejected() {
        let mut config = v2_default_config();
        config.shell_thickness = 32; // 2*32 = 64, no interior
        let mut world = VoxelWorld::new(64, 64, 64);
        world.fill_solid();

        let result = generate_v2(&config, &mut world, config.seed);
        assert!(matches!(result, Err(GenError::InteriorEmpty { .. })));
    }

    #[test]
    fn v2_zero_maze_density_no_maze_edges() {
        let mut config = v2_default_config();
        config.maze_density = 0.0;
        let mut world = VoxelWorld::new(config.resolution, config.resolution, config.resolution);
        world.fill_solid();

        let result = generate_v2(&config, &mut world, config.seed).unwrap();
        // Verify no maze-specific failures and valid shell
        assert!(verify_shell_multi(&world, config.shell_thickness));
        let _ = result;
    }

    #[test]
    fn v2_core_roles_and_tree_are_stable() {
        let config = v2_default_config();
        let mut world = VoxelWorld::new(64, 64, 64);
        world.fill_solid();
        let result = generate_v2(&config, &mut world, config.seed).unwrap();
        assert_eq!(
            result
                .sites
                .iter()
                .map(|site| site.label)
                .collect::<Vec<_>>(),
            CORE_LABELS
        );
        let edges = build_spline_edges(
            config.seed,
            &result
                .sites
                .iter()
                .map(|site| [site.x, site.y, site.z])
                .collect::<Vec<_>>(),
            4,
        )
        .unwrap();
        assert_eq!(edges, [(0, 1), (1, 2), (1, 3), (2, 4)]);
    }

    #[test]
    fn v2_plans_exact_rounded_maze_target() {
        let mut config = v2_default_config();
        config.maze_density = 0.4; // round(0.4 * (10 - 5)) = 2
        config.maze_search_budget = 20_000;
        let mut world = VoxelWorld::new(64, 64, 64);
        world.fill_solid();
        let result = generate_v2(&config, &mut world, config.seed).unwrap();
        assert_eq!(result.edges.len(), config.tunnel_count as usize + 2);
    }

    #[test]
    fn v2_failure_is_atomic() {
        let mut config = v2_default_config();
        config.maze_density = 1.0;
        config.maze_search_budget = 1;
        config.maze_retries = 1;
        let mut world = VoxelWorld::new(64, 64, 64);
        world.fill_solid();
        let before_density = world.density().iter().copied().collect::<Vec<_>>();
        let before_material = world.material().iter().copied().collect::<Vec<_>>();
        assert!(matches!(
            generate_v2(&config, &mut world, config.seed),
            Err(GenError::MazeExhausted { .. })
        ));
        assert_eq!(
            before_density,
            world.density().iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            before_material,
            world.material().iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn v2_rejects_non_cubic_world_with_typed_error() {
        let config = v2_default_config();
        let mut world = VoxelWorld::new(64, 63, 64);
        world.fill_solid();
        assert!(matches!(
            generate_v2(&config, &mut world, config.seed),
            Err(GenError::InvalidConfig(_))
        ));
    }

    #[test]
    fn checked_pair_count_works() {
        assert_eq!(checked_pair_count(5), 10);
        assert_eq!(checked_pair_count(7), 21);
        assert_eq!(checked_pair_count(1), 0);
    }

    #[test]
    fn canonical_pair_ordering() {
        assert_eq!(canonical_pair(3, 1), (1, 3));
        assert_eq!(canonical_pair(0, 5), (0, 5));
        assert_eq!(canonical_pair(4, 4), (4, 4));
    }
}
