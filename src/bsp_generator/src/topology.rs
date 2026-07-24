//! Connectivity graph construction from placed rooms.
//!
//! Builds a minimum spanning tree over room centers (Euclidean distance),
//! then adds exactly the requested number of extra edges (loops) to form
//! cycles. The output [`LayoutIntent`] guarantees a single connected
//! component and the exact requested cycle count.
//!
//! The entry point is [`build_topology`].

use crate::config::ValidatedConfig;
use crate::error::GeneratorError;
use crate::geometry;
use crate::intent::{LayoutIntent, RoomIntent};
use crate::StageRng;

/// Build a connectivity graph over `rooms` with exactly `config.loop_count`
/// redundant edges beyond the minimum spanning tree.
///
/// # Algorithm
///
/// 1. Compute room centers as `position + dimensions / 2`.
/// 2. Build a complete graph with Euclidean distances as edge weights.
/// 3. Run Kruskal's algorithm to extract the minimum spanning tree (MST).
/// 4. From the remaining non-MST edges, randomly select exactly
///    `config.loop_count` edges to add as spatial loops.
/// 5. Return a [`LayoutIntent`] with rooms, edges, and `loop_count`.
///
/// # Guarantees
///
/// - All rooms are reachable from any room (single connected component).
/// - The returned edge count is exactly `(room_count - 1) + loop_count` for
///   `room_count > 0`.
/// - Zero-room configs return an empty edge list.
///
/// # Errors
///
/// Returns [`GeneratorError::InvariantViolation`] if there are not enough
/// non-MST edges to satisfy the loop count (should not happen for typical
/// configs where room_count ≫ loop_count).
pub fn build_topology(
    rooms: Vec<RoomIntent>,
    config: &ValidatedConfig,
    rng: &mut StageRng,
) -> Result<LayoutIntent, GeneratorError> {
    let n = rooms.len();
    let loop_count = config.loop_count;

    if n == 0 {
        return Ok(LayoutIntent {
            rooms,
            edges: Vec::new(),
            loop_count,
        });
    }

    // Compute room centers
    let centers: Vec<(f64, f64, f64)> = rooms
        .iter()
        .map(|r| {
            (
                r.position.0 as f64 + r.dimensions.0 as f64 / 2.0,
                r.position.1 as f64 + r.dimensions.1 as f64 / 2.0,
                r.position.2 as f64 + r.dimensions.2 as f64 / 2.0,
            )
        })
        .collect();

    // Build all possible edges with Euclidean distances
    #[derive(Debug, Clone, Copy)]
    struct Edge {
        a: usize,
        b: usize,
        dist_sq: f64,
    }

    let mut all_edges: Vec<Edge> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = centers[i].0 - centers[j].0;
            let dy = centers[i].1 - centers[j].1;
            let dz = centers[i].2 - centers[j].2;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            all_edges.push(Edge {
                a: i,
                b: j,
                dist_sq,
            });
        }
    }

    // Sort by distance for Kruskal
    all_edges.sort_by(|e1, e2| {
        e1.dist_sq
            .partial_cmp(&e2.dist_sq)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Kruskal's MST
    struct UnionFind {
        parent: Vec<usize>,
        rank: Vec<usize>,
    }
    impl UnionFind {
        fn new(size: usize) -> Self {
            UnionFind {
                parent: (0..size).collect(),
                rank: vec![0; size],
            }
        }
        fn find(&mut self, x: usize) -> usize {
            if self.parent[x] != x {
                self.parent[x] = self.find(self.parent[x]);
            }
            self.parent[x]
        }
        fn union(&mut self, a: usize, b: usize) -> bool {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra == rb {
                return false;
            }
            match self.rank[ra].cmp(&self.rank[rb]) {
                std::cmp::Ordering::Less => self.parent[ra] = rb,
                std::cmp::Ordering::Greater => self.parent[rb] = ra,
                std::cmp::Ordering::Equal => {
                    self.parent[rb] = ra;
                    self.rank[ra] += 1;
                }
            }
            true
        }
    }

    let mut uf = UnionFind::new(n);
    let mut mst_edges: Vec<(usize, usize)> = Vec::with_capacity(n - 1);
    let mut remaining: Vec<(usize, usize)> = Vec::new();

    for edge in &all_edges {
        if mst_edges.len() == n - 1 {
            // MST complete — all further edges are non-MST candidates
            remaining.push((edge.a, edge.b));
        } else if uf.union(edge.a, edge.b) {
            mst_edges.push((edge.a, edge.b));
        } else {
            remaining.push((edge.a, edge.b));
        }
    }

    // Select exactly loop_count extra edges from remaining
    let needed = loop_count as usize;
    if needed > remaining.len() {
        return Err(GeneratorError::InvariantViolation(format!(
            "cannot add {} loops: only {} non-MST edges available for {} rooms",
            needed,
            remaining.len(),
            n,
        )));
    }

    // Partially shuffle the full non-MST candidate set so the selected loop
    // edge set is derived from the corridor-routing RNG, not just the nearest
    // `needed` candidates.
    for i in 0..needed {
        let tail_len = remaining.len() - i;
        let j = i + rng.range_u32(tail_len as u32) as usize;
        remaining.swap(i, j);
    }

    let mut edges = mst_edges;
    edges.extend_from_slice(&remaining[..needed]);

    // Validation
    if !geometry::validate_connectedness(&edges, n) {
        return Err(GeneratorError::InvariantViolation(
            "topology graph is not fully connected".to_string(),
        ));
    }
    geometry::validate_cycle_count(&edges, n, loop_count)?;

    Ok(LayoutIntent {
        rooms,
        edges,
        loop_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DungeonConfig;
    use crate::placement::place_rooms;
    use crate::Seed;

    fn make_rng(seed_val: u64, tag: &str) -> StageRng {
        Seed::new(seed_val).stage_seed(tag).rng()
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn make_rooms_m1(count: u32, seed: u64) -> Vec<RoomIntent> {
        let cfg = DungeonConfig {
            class: crate::config::MapClass::M1,
            room_count: count,
            loop_count: 0,
            xy_bounds: (1536, 1536),
            z_span: 256,
            placement_candidates: 16,
            max_placement_attempts: 64,
            max_astar_expansions: 131_072,
        }
        .validate()
        .unwrap();
        place_rooms(&cfg, &mut make_rng(seed, "room-placement")).unwrap()
    }

    fn make_rooms_m2(count: u32, seed: u64) -> Vec<RoomIntent> {
        let cfg = DungeonConfig {
            class: crate::config::MapClass::M2,
            room_count: count,
            loop_count: 1, // M2 requires at least 1 loop for config validation
            xy_bounds: (3072, 3072),
            z_span: 384,
            placement_candidates: 32,
            max_placement_attempts: 96,
            max_astar_expansions: 524_288,
        }
        .validate()
        .unwrap();
        place_rooms(&cfg, &mut make_rng(seed, "room-placement")).unwrap()
    }

    fn valid_m1_config(rooms: u32, loops: u32) -> ValidatedConfig {
        DungeonConfig {
            class: crate::config::MapClass::M1,
            room_count: rooms,
            loop_count: loops,
            xy_bounds: (1536, 1536),
            z_span: 256,
            placement_candidates: 16,
            max_placement_attempts: 64,
            max_astar_expansions: 131_072,
        }
        .validate()
        .unwrap()
    }

    fn valid_m2_config(rooms: u32, loops: u32) -> ValidatedConfig {
        DungeonConfig {
            class: crate::config::MapClass::M2,
            room_count: rooms,
            loop_count: loops,
            xy_bounds: (3072, 3072),
            z_span: 384,
            placement_candidates: 32,
            max_placement_attempts: 96,
            max_astar_expansions: 524_288,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn connectedness_guarantee() {
        let rooms = make_rooms_m1(12, 42);
        let cfg = valid_m1_config(12, 1);
        let mut rng = make_rng(42, "corridor-routing");
        let layout = build_topology(rooms, &cfg, &mut rng).unwrap();
        assert!(geometry::validate_connectedness(
            &layout.edges,
            layout.rooms.len()
        ));
    }

    #[test]
    fn exact_loop_count_m1_loops_2() {
        let rooms = make_rooms_m1(16, 7);
        let cfg = valid_m1_config(16, 2);
        let mut rng = make_rng(7, "corridor-routing");
        let layout = build_topology(rooms, &cfg, &mut rng).unwrap();
        let n = layout.rooms.len();
        let expected_edges = (n - 1) + 2; // MST + 2 loops
        assert_eq!(layout.edges.len(), expected_edges);
        assert_eq!(layout.loop_count, 2);
        assert!(geometry::validate_connectedness(&layout.edges, n));
        geometry::validate_cycle_count(&layout.edges, n, 2).unwrap();
    }

    #[test]
    fn minimal_graph_two_rooms_zero_loops() {
        // Construct two rooms manually — M1 doesn't allow 2 rooms via config,
        // but topology can operate on any room set.
        let rooms = vec![
            RoomIntent {
                position: (0, 0, 0),
                dimensions: (64, 64, 128),
            },
            RoomIntent {
                position: (80, 0, 0),
                dimensions: (64, 64, 128),
            },
        ];
        let cfg = valid_m1_config(8, 0); // loop_count=0 from valid config
        let mut rng = make_rng(99, "corridor-routing");
        let layout = build_topology(rooms, &cfg, &mut rng).unwrap();
        assert_eq!(layout.edges.len(), 1); // MST with exactly 1 edge
        assert!(geometry::validate_connectedness(
            &layout.edges,
            layout.rooms.len()
        ));
    }

    #[test]
    fn m2_with_6_loops() {
        let rooms = make_rooms_m2(30, 255);
        let cfg = valid_m2_config(30, 6);
        let mut rng = make_rng(255, "corridor-routing");
        let layout = build_topology(rooms, &cfg, &mut rng).unwrap();
        let n = layout.rooms.len();
        assert_eq!(layout.edges.len(), (n - 1) + 6);
        assert!(geometry::validate_connectedness(&layout.edges, n));
    }

    #[test]
    fn deterministic_topology() {
        let rooms_a = make_rooms_m1(10, 42);
        let rooms_b = rooms_a.clone();
        let cfg = valid_m1_config(10, 2);
        let layout_a =
            build_topology(rooms_a, &cfg, &mut make_rng(42, "corridor-routing")).unwrap();
        let layout_b =
            build_topology(rooms_b, &cfg, &mut make_rng(42, "corridor-routing")).unwrap();
        assert_eq!(layout_a.edges, layout_b.edges);
        assert_eq!(layout_a.loop_count, layout_b.loop_count);
    }

    #[test]
    fn loop_edge_selection_uses_full_rng_sample() {
        let rooms = make_rooms_m1(10, 42);
        let cfg = valid_m1_config(10, 2);
        let layout_a =
            build_topology(rooms.clone(), &cfg, &mut make_rng(42, "corridor-routing")).unwrap();
        let layout_b = build_topology(rooms, &cfg, &mut make_rng(99, "corridor-routing")).unwrap();
        assert_ne!(&layout_a.edges[9..], &layout_b.edges[9..]);
    }

    #[test]
    fn all_rooms_retained_in_layout() {
        let rooms = make_rooms_m1(8, 1);
        let n = rooms.len();
        let cfg = valid_m1_config(8, 1);
        let mut rng = make_rng(1, "corridor-routing");
        let layout = build_topology(rooms, &cfg, &mut rng).unwrap();
        assert_eq!(layout.rooms.len(), n);
    }

    #[test]
    fn zero_rooms_empty_edges() {
        // Test topology directly with empty rooms — no config validation needed
        let cfg = valid_m1_config(8, 0);
        let mut rng = make_rng(0, "corridor-routing");
        let layout = build_topology(Vec::new(), &cfg, &mut rng).unwrap();
        assert!(layout.edges.is_empty());
        assert_eq!(layout.loop_count, 0);
    }

    #[test]
    fn loop_count_0_for_m1() {
        let rooms = make_rooms_m1(12, 10);
        let cfg = valid_m1_config(12, 0);
        let mut rng = make_rng(10, "corridor-routing");
        let layout = build_topology(rooms, &cfg, &mut rng).unwrap();
        let n = layout.rooms.len();
        assert_eq!(layout.edges.len(), n - 1);
        assert_eq!(layout.loop_count, 0);
    }

    #[test]
    fn edge_indices_are_valid() {
        let rooms = make_rooms_m1(15, 3);
        let n = rooms.len();
        let cfg = valid_m1_config(15, 2);
        let mut rng = make_rng(3, "corridor-routing");
        let layout = build_topology(rooms, &cfg, &mut rng).unwrap();
        for &(a, b) in &layout.edges {
            assert!(a < n, "edge index {} out of range", a);
            assert!(b < n, "edge index {} out of range", b);
            assert_ne!(a, b, "self-loop at index {}", a);
        }
    }
}
