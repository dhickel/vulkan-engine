//! Phase 06 — Resource budget validation and collider preflight.
//!
//! ## Resource counting
//! - Count tiles (exactly 65,536 max for 128×128×4).
//! - Count non-empty chunks (floor + structure passes, ≤512).
//! - Count static collider bodies (≤512).
//! - Total bodies ≤513 with proof body.
//! - Vertex/index budgets (1.6M vertices, 2.4M indices).
//!
//! ## Collider preflight
//! - Verify every chunk mesh can be conceptually constructed.
//! - Produce a pure preflight manifest without creating physics state.
//! - Transactional: no scene mutation before markers+resources pass.

use crate::geometry::build_chunk_geometry_plan;
use crate::layout::{ParsedLevel, Tile};

use super::config::NormalizedGeneratorConfig;
use super::error::{ErrorStage, GeneratorError};
use super::ir::IntendedTopology;

// ─── Resource counts ────────────────────────────────────────────────────────

/// Complete resource accounting for a generated level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceCounts {
    /// Total tile count (width × height × layers).
    pub total_tiles: u64,
    /// Number of floor tiles.
    pub floor_tiles: u64,
    /// Number of wall tiles.
    pub wall_tiles: u64,
    /// Number of void tiles.
    pub void_tiles: u64,
    /// Number of ramp tiles.
    pub ramp_tiles: u64,
    /// Number of non-empty chunks (floor + structure).
    pub non_empty_chunks: u32,
    /// Estimated vertex count.
    pub estimated_vertices: u64,
    /// Estimated index count.
    pub estimated_indices: u64,
    /// Number of placed light markers.
    pub light_count: u32,
    /// Number of placed model markers.
    pub model_count: u32,
    /// Number of generated static collider bodies.
    pub static_body_count: u32,
    /// Total body count (static + proof).
    pub total_body_count: u32,
}

/// Chunk preflight manifest entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ChunkPreflightEntry {
    pub(super) chunk_name: String,
    pub(super) vertex_count: usize,
    pub(super) index_count: usize,
    pub(super) triangle_count: usize,
}

/// Chunk/collider preflight manifest.
#[derive(Debug, Clone, Default)]
pub(super) struct ColliderPreflightManifest {
    pub(super) entries: Vec<ChunkPreflightEntry>,
    pub(super) total_vertices: u64,
    pub(super) total_indices: u64,
    pub(super) chunk_count: u32,
}

// ─── Resource counting ──────────────────────────────────────────────────────

/// Count resources from the materialized level using the exact geometry plan.
pub(super) fn count_resources(
    level: &ParsedLevel,
    _topology: &IntendedTopology,
    light_count: u32,
    model_count: u32,
    config: &NormalizedGeneratorConfig,
) -> Result<ResourceCounts, GeneratorError> {
    let width = u64::from(config.width());
    let height = u64::from(config.height());
    let layers = u64::from(config.layers().2);

    let total_tiles = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(layers))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "resource_total_tiles",
        })?;

    let mut floor_tiles = 0u64;
    let mut wall_tiles = 0u64;
    let mut void_tiles = 0u64;
    let mut ramp_tiles = 0u64;

    for layer in &level.layers {
        for tile in layer {
            match tile {
                Tile::Floor => floor_tiles = floor_tiles.saturating_add(1),
                Tile::Wall => wall_tiles = wall_tiles.saturating_add(1),
                Tile::Void => void_tiles = void_tiles.saturating_add(1),
                Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_) => {
                    ramp_tiles = ramp_tiles.saturating_add(1);
                }
            }
        }
    }

    // Build the exact geometry plan to get precise vertex/index/chunk counts.
    let plan = build_chunk_geometry_plan(level);

    let mut non_empty = 0u32;
    let mut exact_vertices = 0u64;
    let mut exact_indices = 0u64;

    for leaf in &plan.leaves {
        if !leaf.floor_verts.is_empty() {
            non_empty = non_empty.saturating_add(1);
            exact_vertices = exact_vertices.saturating_add(leaf.floor_verts.len() as u64);
            exact_indices = exact_indices.saturating_add(leaf.floor_indices.len() as u64);
        }
        if !leaf.wall_verts.is_empty() {
            non_empty = non_empty.saturating_add(1);
            exact_vertices = exact_vertices.saturating_add(leaf.wall_verts.len() as u64);
            exact_indices = exact_indices.saturating_add(leaf.wall_indices.len() as u64);
        }
    }

    // Static body count: one per non-empty output mesh.
    let static_body_count = non_empty;

    // Total body count = static bodies + 1 proof body.
    let total_body_count = static_body_count.saturating_add(1);

    Ok(ResourceCounts {
        total_tiles,
        floor_tiles,
        wall_tiles,
        void_tiles,
        ramp_tiles,
        non_empty_chunks: non_empty,
        estimated_vertices: exact_vertices,
        estimated_indices: exact_indices,
        light_count,
        model_count,
        static_body_count,
        total_body_count,
    })
}

// ─── Budget enforcement ─────────────────────────────────────────────────────

/// Validate resource counts against normalized config budgets.
pub(super) fn enforce_budgets(
    counts: &ResourceCounts,
    config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    // Tile count must match dimensions (already validated, but double-check).
    let expected_tiles = u64::from(config.width())
        .checked_mul(u64::from(config.height()))
        .and_then(|v| v.checked_mul(u64::from(config.layers().2)))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "budget_expected_tiles",
        })?;
    if counts.total_tiles != expected_tiles {
        return Err(GeneratorError::IrInvariant {
            stage: ErrorStage::Ir,
            detail: format!(
                "tile_count_mismatch expected={expected_tiles} actual={}",
                counts.total_tiles
            ),
        });
    }
    if counts.total_tiles > u64::from(config.max_tiles()) {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Ir,
            constraint: "max_tiles",
            required: u64::from(config.max_tiles()),
            available: counts.total_tiles,
        });
    }

    // Chunk budget.
    if counts.non_empty_chunks > config.max_chunks() {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Ir,
            constraint: "max_chunks",
            required: u64::from(config.max_chunks()),
            available: u64::from(counts.non_empty_chunks),
        });
    }

    // Static body budget.
    if counts.static_body_count > config.max_static_bodies() {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Ir,
            constraint: "max_static_bodies",
            required: u64::from(config.max_static_bodies()),
            available: u64::from(counts.static_body_count),
        });
    }

    // Total body budget.
    if counts.total_body_count > config.max_total_bodies() {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Ir,
            constraint: "max_total_bodies",
            required: u64::from(config.max_total_bodies()),
            available: u64::from(counts.total_body_count),
        });
    }

    // Vertex budget.
    if counts.estimated_vertices > u64::from(config.max_vertices()) {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Ir,
            constraint: "max_vertices",
            required: u64::from(config.max_vertices()),
            available: counts.estimated_vertices,
        });
    }

    // Index budget.
    if counts.estimated_indices > u64::from(config.max_indices()) {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Ir,
            constraint: "max_indices",
            required: u64::from(config.max_indices()),
            available: counts.estimated_indices,
        });
    }

    // Light budget (already enforced by placement, double-check).
    if counts.light_count > config.max_lights() {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Ir,
            constraint: "max_lights",
            required: u64::from(config.max_lights()),
            available: u64::from(counts.light_count),
        });
    }

    Ok(())
}

// ─── Collider preflight manifest ────────────────────────────────────────────

/// Build a pure preflight manifest from the exact geometry plan.
/// Does not create any renderer handle, physics body, or collider.
pub(super) fn build_preflight_manifest(
    level: &ParsedLevel,
    _config: &NormalizedGeneratorConfig,
) -> Result<ColliderPreflightManifest, GeneratorError> {
    let plan = build_chunk_geometry_plan(level);

    let mut entries = Vec::new();
    let mut total_vertices = 0u64;
    let mut total_indices = 0u64;
    let mut chunk_count = 0u32;

    for leaf in &plan.leaves {
        if !leaf.floor_verts.is_empty() {
            let name = format!("floor_{}", leaf.leaf_name);
            entries.push(ChunkPreflightEntry {
                chunk_name: name,
                vertex_count: leaf.floor_verts.len(),
                index_count: leaf.floor_indices.len(),
                triangle_count: leaf.floor_indices.len() / 3,
            });
            total_vertices = total_vertices.saturating_add(leaf.floor_verts.len() as u64);
            total_indices = total_indices.saturating_add(leaf.floor_indices.len() as u64);
            chunk_count = chunk_count.saturating_add(1);
        }
        if !leaf.wall_verts.is_empty() {
            let name = format!("struct_{}", leaf.leaf_name);
            entries.push(ChunkPreflightEntry {
                chunk_name: name,
                vertex_count: leaf.wall_verts.len(),
                index_count: leaf.wall_indices.len(),
                triangle_count: leaf.wall_indices.len() / 3,
            });
            total_vertices = total_vertices.saturating_add(leaf.wall_verts.len() as u64);
            total_indices = total_indices.saturating_add(leaf.wall_indices.len() as u64);
            chunk_count = chunk_count.saturating_add(1);
        }
    }

    // Validate index count is multiple of 3.
    for entry in &entries {
        if entry.index_count % 3 != 0 {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!(
                    "chunk_{}_indices_not_multiple_of_3 count={}",
                    entry.chunk_name, entry.index_count
                ),
            });
        }
    }

    Ok(ColliderPreflightManifest {
        entries,
        total_vertices,
        total_indices,
        chunk_count,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::GeneratorConfig;
    use super::*;
    use crate::layout::{ParsedLevel, Tile, TileCoord};

    fn make_level(w: usize, h: usize, tiles: Vec<Tile>) -> ParsedLevel {
        ParsedLevel {
            width: w,
            height: h,
            layers: vec![tiles],
            spawn: TileCoord {
                layer: 0,
                x: 1,
                y: 1,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        }
    }

    fn empty_topology(config: NormalizedGeneratorConfig) -> IntendedTopology {
        IntendedTopology {
            regions: vec![],
            edges: vec![],
            transitions: vec![],
            route_distance: 0,
            per_layer_cycles: vec![0; usize::from(config.layers().2)],
            max_branch_depth: 0,
            dead_end_count: 0,
            articulation_count: 0,
            crossing_count: 0,
            config,
        }
    }

    #[test]
    fn resource_count_total_tiles_matches_dimensions() {
        let config = GeneratorConfig::custom(64, 64, 2).normalize().unwrap();
        // Create a small level for counting — dimensions won't match config.
        // The count function uses config dimensions for total_tiles, not the level.
        // Create a properly sized level.
        let w = 64usize;
        let h = 64usize;
        let tiles = vec![Tile::Wall; w * h];
        let level = ParsedLevel {
            width: w,
            height: h,
            layers: vec![tiles.clone(), tiles],
            spawn: TileCoord {
                layer: 0,
                x: 1,
                y: 1,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        };
        let topology = empty_topology(config.clone());
        let counts = count_resources(&level, &topology, 0, 0, &config).unwrap();
        assert_eq!(counts.total_tiles, 64 * 64 * 2);
        assert_eq!(counts.wall_tiles, 64 * 64 * 2);
        assert_eq!(counts.floor_tiles, 0);
    }

    #[test]
    fn budget_enforcement_rejects_excess() {
        let config = GeneratorConfig::custom(64, 64, 2).normalize().unwrap();
        let mut counts = ResourceCounts {
            total_tiles: 64 * 64 * 2,
            floor_tiles: 0,
            wall_tiles: 0,
            void_tiles: 0,
            ramp_tiles: 0,
            non_empty_chunks: 999, // exceeds max_chunks=128 for minimum profile
            estimated_vertices: 0,
            estimated_indices: 0,
            light_count: 0,
            model_count: 0,
            static_body_count: 999,
            total_body_count: 1000,
        };
        assert!(enforce_budgets(&counts, &config).is_err());
    }

    #[test]
    fn budget_enforcement_passes_for_valid_counts() {
        let config = GeneratorConfig::custom(64, 64, 2).normalize().unwrap();
        let counts = ResourceCounts {
            total_tiles: 64 * 64 * 2,
            floor_tiles: 100,
            wall_tiles: 200,
            void_tiles: 7892,
            ramp_tiles: 0,
            non_empty_chunks: 4,
            estimated_vertices: 1600,
            estimated_indices: 2400,
            light_count: 8,
            model_count: 10,
            static_body_count: 4,
            total_body_count: 5,
        };
        assert!(enforce_budgets(&counts, &config).is_ok());
    }

    #[test]
    fn max_profile_budgets_are_correct() {
        let config = GeneratorConfig::qualified(super::super::config::QualifiedProfile::Maximum)
            .normalize()
            .unwrap();
        assert_eq!(config.max_tiles(), 65_536);
        assert_eq!(config.max_chunks(), 512);
        assert_eq!(config.max_static_bodies(), 512);
        assert_eq!(config.max_total_bodies(), 513);
        assert_eq!(config.max_vertices(), 1_600_000);
        assert_eq!(config.max_indices(), 2_400_000);
    }

    #[test]
    fn preflight_counts_vertices_and_indices() {
        let config = GeneratorConfig::custom(64, 64, 2).normalize().unwrap();
        let w = 64usize;
        let h = 64usize;
        let mut tiles = vec![Tile::Wall; w * h];
        // Add some floor tiles.
        for y in 1..5 {
            for x in 1..5 {
                tiles[y * w + x] = Tile::Floor;
            }
        }
        let level = ParsedLevel {
            width: w,
            height: h,
            layers: vec![tiles.clone(), tiles],
            spawn: TileCoord {
                layer: 0,
                x: 2,
                y: 2,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        };
        let manifest = build_preflight_manifest(&level, &config).unwrap();
        // Should have at least some geometry.
        assert!(manifest.total_vertices > 0 || manifest.entries.is_empty());
        // All entries must have index_count % 3 == 0.
        for entry in &manifest.entries {
            assert_eq!(
                entry.index_count % 3,
                0,
                "chunk {} has invalid indices",
                entry.chunk_name
            );
        }
    }

    #[test]
    fn preflight_chunk_count_bounded() {
        let config = GeneratorConfig::custom(64, 64, 2).normalize().unwrap();
        let w = 64usize;
        let h = 64usize;
        let mut tiles = vec![Tile::Wall; w * h];
        tiles[10 * w + 10] = Tile::Floor;
        let level = ParsedLevel {
            width: w,
            height: h,
            layers: vec![tiles.clone(), tiles],
            spawn: TileCoord {
                layer: 0,
                x: 10,
                y: 10,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        };
        let manifest = build_preflight_manifest(&level, &config).unwrap();
        // At most 2 layers × 2 material domains × (ceil(64/16))² = 64 meshes.
        assert!(manifest.chunk_count <= 64);
    }

    #[test]
    fn resource_preflight_exactly_matches_production_meshes() {
        let config = GeneratorConfig::custom(64, 64, 2).normalize().unwrap();
        let mut tiles = vec![Tile::Void; 64 * 64];
        tiles[1 * 64 + 1] = Tile::Floor;
        tiles[1 * 64 + 2] = Tile::RampEast(0);
        tiles[2 * 64 + 1] = Tile::Wall;
        let level = ParsedLevel {
            width: 64,
            height: 64,
            layers: vec![tiles.clone(), tiles],
            spawn: TileCoord {
                layer: 0,
                x: 1,
                y: 1,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        };
        let topology = empty_topology(config.clone());
        let counts = count_resources(&level, &topology, 0, 0, &config).unwrap();
        let manifest = build_preflight_manifest(&level, &config).unwrap();
        let material = renderer::MaterialHandle::new(0, 0);
        let chunks = crate::geometry::build_level_chunks(&level, material, material);

        assert_eq!(
            usize::try_from(counts.non_empty_chunks).unwrap(),
            chunks.len()
        );
        assert_eq!(usize::try_from(manifest.chunk_count).unwrap(), chunks.len());
        assert_eq!(
            counts.estimated_vertices,
            chunks
                .iter()
                .map(|chunk| chunk.mesh.vertices.len() as u64)
                .sum::<u64>()
        );
        assert_eq!(
            counts.estimated_indices,
            chunks
                .iter()
                .map(|chunk| chunk.mesh.indices.len() as u64)
                .sum::<u64>()
        );
        assert_eq!(manifest.total_vertices, counts.estimated_vertices);
        assert_eq!(manifest.total_indices, counts.estimated_indices);

        for (entry, chunk) in manifest.entries.iter().zip(&chunks) {
            assert_eq!(entry.chunk_name, chunk.name);
            assert_eq!(entry.vertex_count, chunk.mesh.vertices.len());
            assert_eq!(entry.index_count, chunk.mesh.indices.len());
            assert_eq!(entry.triangle_count, chunk.mesh.indices.len() / 3);
            assert!(chunk.mesh.material.is_some());
        }
    }
}
