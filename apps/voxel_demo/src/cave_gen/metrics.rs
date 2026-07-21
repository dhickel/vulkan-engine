//! Route quality metrics over a signed density lattice.
//!
//! All metrics are pure functions over the lattice + a designated site list.
//! They produce deterministic output for known lattice configurations.
//!
//! Also provides camera-pose derivation from site positions.

use crate::cave_gen::lattice::{DenseLattice, Density};

/// Radius (in cells) for the clearance neighborhood search.
pub const CLEARANCE_RADIUS: i32 = 3;
/// Upper bound for clearance reporting. When no solid is found within
/// CLEARANCE_RADIUS cells, the reported value is at least this large.
pub const CLEARANCE_MAX_REPORT: f32 = CLEARANCE_RADIUS as f32 + 1.0;

/// A site (point of interest) in voxel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Site {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub label: &'static str,
}

impl Site {
    pub const fn new(x: u32, y: u32, z: u32, label: &'static str) -> Self {
        Self { x, y, z, label }
    }

    /// Manhattan distance between two sites.
    pub fn manhattan_to(&self, other: &Site) -> u32 {
        let dx = (self.x as i64 - other.x as i64).unsigned_abs() as u32;
        let dy = (self.y as i64 - other.y as i64).unsigned_abs() as u32;
        let dz = (self.z as i64 - other.z as i64).unsigned_abs() as u32;
        dx + dy + dz
    }
}

/// A site adjacency graph edge.
#[derive(Debug, Clone, Copy)]
pub struct RouteEdge {
    pub from: usize,
    pub to: usize,
    /// Minimum clearance along this edge: the smallest sphere radius that
    /// can pass through the tunnel segment connecting these two sites.
    pub clearance: f32,
}

impl PartialEq for RouteEdge {
    fn eq(&self, other: &Self) -> bool {
        self.from == other.from
            && self.to == other.to
            && self.clearance.to_bits() == other.clearance.to_bits()
    }
}

/// Derived camera pose from generator output, used for capture/rendering.
#[derive(Debug, Clone, Copy)]
pub struct CameraPose {
    pub eye: (f32, f32, f32),
    pub look_at: (f32, f32, f32),
    pub up: (f32, f32, f32),
}

/// Derive a camera pose facing from spawn toward the farthest site.
/// Uses the spawn point and looks toward the opposite end of the cave.
pub fn camera_pose(sites: &[Site], spawn_index: usize) -> CameraPose {
    let spawn = if spawn_index < sites.len() {
        sites[spawn_index]
    } else {
        sites
            .first()
            .copied()
            .unwrap_or(Site::new(32, 32, 32, "default"))
    };

    // Look toward the site farthest from spawn
    let target = sites
        .iter()
        .max_by_key(|s| s.manhattan_to(&spawn))
        .copied()
        .unwrap_or(Site::new(32, 32, 48, "default"));

    let eye = (
        spawn.x as f32 + 8.0,
        spawn.y as f32 + 6.0,
        spawn.z as f32 - 4.0,
    );
    let look_at = (
        (spawn.x as f32 + target.x as f32) * 0.5,
        (spawn.y as f32 + target.y as f32) * 0.5,
        (spawn.z as f32 + target.z as f32) * 0.5,
    );
    let up = (0.0, 0.0, 1.0);

    CameraPose { eye, look_at, up }
}

// ─── Connectivity: flood-fill from spawn ───────────────────────────────────

/// Flood-fill from a seed coordinate, counting reachable air cells.
/// "Air" is defined as density >= 0.
/// Returns the set of reachable cell indices.
pub fn flood_fill_air(
    lattice: &DenseLattice<Density>,
    start_x: u32,
    start_y: u32,
    start_z: u32,
) -> Vec<usize> {
    let (w, h, d) = lattice.dims();
    let total = lattice.len();
    let mut visited = vec![false; total];
    let mut stack: Vec<(u32, u32, u32)> = Vec::new();
    let mut reachable: Vec<usize> = Vec::new();

    let start_idx = xyz_to_idx(start_x, start_y, start_z, w, h, d);
    if start_idx.is_none() {
        return reachable;
    }
    let start_idx = start_idx.unwrap();
    if *lattice.get(start_x, start_y, start_z).unwrap_or(&-1) < 0 {
        return reachable; // start is solid
    }

    visited[start_idx] = true;
    stack.push((start_x, start_y, start_z));
    reachable.push(start_idx);

    while let Some((x, y, z)) = stack.pop() {
        let neighbors: [(i32, i32, i32); 6] = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];
        for (dx, dy, dz) in neighbors {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;
            if nx < 0 || ny < 0 || nz < 0 {
                continue;
            }
            let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
            let ni = match xyz_to_idx(nx, ny, nz, w, h, d) {
                Some(i) => i,
                None => continue,
            };
            if !visited[ni] && lattice.get(nx, ny, nz).map_or(false, |d| *d >= 0) {
                visited[ni] = true;
                stack.push((nx, ny, nz));
                reachable.push(ni);
            }
        }
    }

    reachable
}

// ─── Connected volume ──────────────────────────────────────────────────────

/// Count of reachable cells from spawn via flood fill.
pub fn connected_volume(lattice: &DenseLattice<Density>, spawn: Site) -> usize {
    flood_fill_air(lattice, spawn.x, spawn.y, spawn.z).len()
}

// ─── Air ratio ─────────────────────────────────────────────────────────────

/// Fraction of total cells that are reachable from spawn.
pub fn air_ratio(lattice: &DenseLattice<Density>, spawn: Site) -> f64 {
    let total = lattice.len();
    if total == 0 {
        return 0.0;
    }
    connected_volume(lattice, spawn) as f64 / total as f64
}

// ─── Landmark separation ───────────────────────────────────────────────────

/// Manhattan distance between two site centers.
pub fn landmark_separation(a: &Site, b: &Site) -> u32 {
    a.manhattan_to(b)
}

/// Mean pairwise Manhattan distance across all sites.
pub fn mean_landmark_separation(sites: &[Site]) -> f64 {
    if sites.len() < 2 {
        return 0.0;
    }
    let mut total = 0u64;
    let mut pairs = 0u64;
    for i in 0..sites.len() {
        for j in (i + 1)..sites.len() {
            total += landmark_separation(&sites[i], &sites[j]) as u64;
            pairs += 1;
        }
    }
    total as f64 / pairs as f64
}

// ─── Vertical displacement ─────────────────────────────────────────────────

/// Maximum layer difference between any two sites.
pub fn vertical_displacement(sites: &[Site]) -> u32 {
    if sites.is_empty() {
        return 0;
    }
    let min_z = sites.iter().map(|s| s.z).min().unwrap_or(0);
    let max_z = sites.iter().map(|s| s.z).max().unwrap_or(0);
    max_z - min_z
}

// ─── Loop detection ────────────────────────────────────────────────────────

/// Count connected components in the site adjacency graph.
/// An edge exists between two sites if their flood-fills are connected
/// (i.e., they share reachable cells or are reachable from each other).
/// A fully connected cave has 1 component; disconnected regions have more.
pub fn connected_components(lattice: &DenseLattice<Density>, sites: &[Site]) -> u32 {
    if sites.is_empty() {
        return 0;
    }
    let n = sites.len();
    let reachable_sets: Vec<Vec<usize>> = sites
        .iter()
        .map(|s| {
            let mut reachable = flood_fill_air(lattice, s.x, s.y, s.z);
            reachable.sort_unstable();
            reachable
        })
        .collect();

    let mut adj = vec![vec![]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let connected = reachable_sets[i]
                .iter()
                .any(|cell| reachable_sets[j].binary_search(cell).is_ok());
            if connected {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    let mut visited = vec![false; n];
    let mut components = 0u32;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(v) = stack.pop() {
            for &neighbor in &adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }
    components
}

// ─── Loop count ────────────────────────────────────────────────────────────

/// Count independent cycles in an undirected site graph.
///
/// This is the graph's cyclomatic number: `edges - vertices + components`.
/// Isolated sites count as components, parallel edges form additional cycles,
/// and invalid endpoint indices panic because they indicate a malformed metric
/// input rather than a low-quality route.
pub fn loop_count(site_count: usize, edges: &[RouteEdge]) -> u32 {
    if site_count == 0 {
        assert!(edges.is_empty(), "edges require at least one site");
        return 0;
    }

    let mut parent: Vec<usize> = (0..site_count).collect();
    let mut components = site_count;

    fn root(parent: &mut [usize], node: usize) -> usize {
        let mut current = node;
        while parent[current] != current {
            current = parent[current];
        }
        let root = current;
        let mut current = node;
        while parent[current] != current {
            let next = parent[current];
            parent[current] = root;
            current = next;
        }
        root
    }

    for edge in edges {
        assert!(
            edge.from < site_count && edge.to < site_count,
            "route edge endpoint out of bounds"
        );
        let from_root = root(&mut parent, edge.from);
        let to_root = root(&mut parent, edge.to);
        if from_root != to_root {
            parent[to_root] = from_root;
            components -= 1;
        }
    }

    let cycles = edges.len() + components - site_count;
    u32::try_from(cycles).unwrap_or(u32::MAX)
}

// ─── Clearance: minimum sphere radius along a tunnel segment ───────────────

/// Estimate clearance along a straight path from `from` to `to` using
/// a 3D Bresenham-like walk. Returns the minimum Euclidean distance from
/// the line segment to any solid cell (density < 0) encountered along the path.
///
/// The returned value is the radius of the largest sphere that can pass
/// through without intersecting solid voxels.
pub fn path_clearance(lattice: &DenseLattice<Density>, from: &Site, to: &Site) -> f32 {
    let (w, h, d) = lattice.dims();
    let w = w as i32;
    let h = h as i32;
    let d = d as i32;

    let (x0, y0, z0) = (from.x as f32, from.y as f32, from.z as f32);
    let (x1, y1, z1) = (to.x as f32, to.y as f32, to.z as f32);

    let dx = x1 - x0;
    let dy = y1 - y0;
    let dz = z1 - z0;
    let steps = (dx.abs().max(dy.abs()).max(dz.abs())).ceil() as u32;
    if steps == 0 {
        return f32::MAX;
    }

    let mut min_clearance = f32::MAX;

    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let px = (x0 + t * dx).round() as i32;
        let py = (y0 + t * dy).round() as i32;
        let pz = (z0 + t * dz).round() as i32;

        // Check a 7×7×7 neighborhood around each path point for solid cells.
        // When no solid is found, clearance is at least CLEARANCE_MAX_REPORT.
        for dz in -CLEARANCE_RADIUS..=CLEARANCE_RADIUS {
            for dy in -CLEARANCE_RADIUS..=CLEARANCE_RADIUS {
                for dx in -CLEARANCE_RADIUS..=CLEARANCE_RADIUS {
                    let nx = px + dx;
                    let ny = py + dy;
                    let nz = pz + dz;
                    if nx < 0 || ny < 0 || nz < 0 || nx >= w || ny >= h || nz >= d {
                        continue;
                    }
                    let density = lattice.get(nx as u32, ny as u32, nz as u32);
                    if let Some(&den) = density {
                        if den < 0 {
                            let d2 = point_to_segment_sq(
                                (nx as f32, ny as f32, nz as f32),
                                (x0, y0, z0),
                                (x1, y1, z1),
                            );
                            let dist = d2.sqrt();
                            if dist < min_clearance {
                                min_clearance = dist;
                            }
                        }
                    }
                }
            }
        }
    }

    min_clearance
}

// ─── Helpers ───────────────────────────────────────────────────────────────

#[inline]
fn xyz_to_idx(x: u32, y: u32, z: u32, w: u32, h: u32, d: u32) -> Option<usize> {
    if x < w && y < h && z < d {
        Some(
            (x as usize) + (y as usize) * (w as usize) + (z as usize) * (w as usize) * (h as usize),
        )
    } else {
        None
    }
}

/// Squared distance from point `p` to line segment `a` → `b`.
fn point_to_segment_sq(p: (f32, f32, f32), a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
    let ap = (p.0 - a.0, p.1 - a.1, p.2 - a.2);
    let ab2 = ab.0 * ab.0 + ab.1 * ab.1 + ab.2 * ab.2;
    if ab2 < f32::EPSILON {
        return ap.0 * ap.0 + ap.1 * ap.1 + ap.2 * ap.2;
    }
    let t = (ap.0 * ab.0 + ap.1 * ab.1 + ap.2 * ab.2) / ab2;
    let t = t.clamp(0.0, 1.0);
    let closest = (a.0 + t * ab.0, a.1 + t * ab.1, a.2 + t * ab.2);
    let d = (p.0 - closest.0, p.1 - closest.1, p.2 - closest.2);
    d.0 * d.0 + d.1 * d.1 + d.2 * d.2
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cave_gen::lattice::VoxelWorld;

    fn hollow_box_world() -> VoxelWorld {
        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_solid();
        for z in 3..13 {
            for y in 3..13 {
                for x in 3..13 {
                    world.set_voxel(x, y, z, 127i8, 0);
                }
            }
        }
        world
    }

    #[test]
    fn flood_fill_counts_reachable() {
        let world = hollow_box_world();
        let reachable = flood_fill_air(world.density(), 7, 7, 7);
        assert_eq!(reachable.len(), 10 * 10 * 10);
    }

    #[test]
    fn flood_fill_reaching_lattice_boundaries_is_safe() {
        let mut world = VoxelWorld::new(2, 2, 2);
        world.fill_air();
        let reachable = flood_fill_air(world.density(), 0, 0, 0);
        assert_eq!(reachable.len(), 8);
    }

    #[test]
    fn flood_fill_solid_start_returns_empty() {
        let world = hollow_box_world();
        let reachable = flood_fill_air(world.density(), 0, 0, 0);
        assert!(reachable.is_empty());
    }

    #[test]
    fn connected_volume_matches_expected() {
        let world = hollow_box_world();
        let vol = connected_volume(world.density(), Site::new(7, 7, 7, "spawn"));
        assert_eq!(vol, 1000);
    }

    #[test]
    fn air_ratio_fraction() {
        let world = hollow_box_world();
        let ratio = air_ratio(world.density(), Site::new(7, 7, 7, "spawn"));
        let expected = 1000.0 / 4096.0;
        assert!((ratio - expected).abs() < 0.001);
    }

    #[test]
    fn landmark_separation_manhattan() {
        let a = Site::new(3, 4, 1, "a");
        let b = Site::new(7, 1, 2, "b");
        assert_eq!(landmark_separation(&a, &b), 8);
    }

    #[test]
    fn mean_landmark_separation() {
        let sites: [Site; 3] = [
            Site::new(0, 0, 0, "a"),
            Site::new(0, 0, 10, "b"),
            Site::new(0, 10, 0, "c"),
        ];
        let mean = super::mean_landmark_separation(&sites);
        assert!((mean - 13.333).abs() < 0.01);
    }

    #[test]
    fn vertical_displacement_max_diff() {
        let sites = [
            Site::new(1, 1, 5, "a"),
            Site::new(2, 2, 3, "b"),
            Site::new(3, 3, 9, "c"),
            Site::new(4, 4, 1, "d"),
        ];
        assert_eq!(vertical_displacement(&sites), 8);
    }

    #[test]
    fn vertical_displacement_empty() {
        assert_eq!(vertical_displacement(&[]), 0);
    }

    #[test]
    fn connected_components_same_region() {
        let world = hollow_box_world();
        let sites = [
            Site::new(5, 5, 5, "a"),
            Site::new(10, 10, 10, "b"),
            Site::new(7, 7, 12, "c"),
        ];
        assert_eq!(connected_components(world.density(), &sites), 1);
    }

    #[test]
    fn connected_components_disconnected() {
        let mut world = VoxelWorld::new(32, 16, 16);
        world.fill_solid();
        for z in 1..15 {
            for y in 1..15 {
                for x in 1..14 {
                    world.set_voxel(x, y, z, 127i8, 0);
                }
            }
        }
        for z in 1..15 {
            for y in 1..15 {
                for x in 18..31 {
                    world.set_voxel(x, y, z, 127i8, 0);
                }
            }
        }
        let sites = [Site::new(7, 7, 7, "left"), Site::new(24, 7, 7, "right")];
        assert_eq!(connected_components(world.density(), &sites), 2);
    }

    #[test]
    fn loop_count_distinguishes_trees_and_cycles() {
        let edge = |from, to| RouteEdge {
            from,
            to,
            clearance: 1.0,
        };
        assert_eq!(loop_count(4, &[edge(0, 1), edge(1, 2), edge(1, 3)]), 0);
        assert_eq!(loop_count(3, &[edge(0, 1), edge(1, 2), edge(2, 0)]), 1);
        assert_eq!(
            loop_count(
                6,
                &[
                    edge(0, 1),
                    edge(1, 2),
                    edge(2, 0),
                    edge(3, 4),
                    edge(4, 5),
                    edge(5, 3),
                ],
            ),
            2
        );
    }

    #[test]
    #[should_panic(expected = "route edge endpoint out of bounds")]
    fn loop_count_rejects_invalid_edges() {
        loop_count(
            1,
            &[RouteEdge {
                from: 0,
                to: 1,
                clearance: 1.0,
            }],
        );
    }

    #[test]
    fn path_clearance_open_space() {
        let world = hollow_box_world();
        let a = Site::new(5, 5, 5, "a");
        let b = Site::new(10, 10, 10, "b");
        let clearance = path_clearance(world.density(), &a, &b);
        assert!(clearance > 1.0);
    }

    #[test]
    fn path_clearance_narrow_passage() {
        let mut world = VoxelWorld::new(16, 16, 16);
        world.fill_solid();
        for x in 0..16 {
            world.set_voxel(x, 7, 7, 127i8, 0);
        }
        for y in 6..9 {
            for z in 6..9 {
                world.set_voxel(1, y, z, 127i8, 0);
                world.set_voxel(14, y, z, 127i8, 0);
            }
        }
        let a = Site::new(1, 7, 7, "a");
        let b = Site::new(14, 7, 7, "b");
        let clearance = path_clearance(world.density(), &a, &b);
        assert!(clearance < 1.5);
    }

    #[test]
    fn metrics_are_deterministic() {
        let world = hollow_box_world();
        let sites = [Site::new(5, 5, 5, "a"), Site::new(10, 10, 10, "b")];
        let cv1 = connected_volume(world.density(), Site::new(7, 7, 7, "spawn"));
        let cv2 = connected_volume(world.density(), Site::new(7, 7, 7, "spawn"));
        assert_eq!(cv1, cv2);

        let cc1 = connected_components(world.density(), &sites);
        let cc2 = connected_components(world.density(), &sites);
        assert_eq!(cc1, cc2);

        let cl1 = path_clearance(world.density(), &sites[0], &sites[1]);
        let cl2 = path_clearance(world.density(), &sites[0], &sites[1]);
        assert_eq!(cl1, cl2);

        let edges = [
            RouteEdge {
                from: 0,
                to: 1,
                clearance: cl1,
            },
            RouteEdge {
                from: 1,
                to: 0,
                clearance: cl1,
            },
        ];
        assert_eq!(
            loop_count(sites.len(), &edges),
            loop_count(sites.len(), &edges)
        );
    }

    #[test]
    fn camera_pose_derivation() {
        let sites = [
            Site::new(10, 10, 10, "spawn"),
            Site::new(50, 30, 20, "destination"),
        ];
        let pose = camera_pose(&sites, 0);
        // Eye is offset from spawn
        assert!((pose.eye.0 - 18.0).abs() < 0.1, "eye x");
        assert!((pose.eye.1 - 16.0).abs() < 0.1, "eye y");
        assert!((pose.eye.2 - 6.0).abs() < 0.1, "eye z");
        // Look-at is midpoint-ish
        assert!(pose.look_at.0 > 20.0 && pose.look_at.0 < 40.0);
        assert!(pose.look_at.1 > 15.0 && pose.look_at.1 < 25.0);
    }
}
