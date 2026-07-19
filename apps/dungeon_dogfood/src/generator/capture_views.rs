//! Phase 07 — Deterministic capture-site camera views derived from
//! reconstructed movement graph and accepted topology metadata.
//!
//! Every view is validated against bounds, walkability, role, and
//! non-degenerate direction before conversion to finite world coordinates.
//! Missing or unavailable sites return typed errors; void and inaccessible
//! tiles are never synthesized.

use glam::Vec3;

use crate::collision::WALL_HEIGHT;
use crate::layout::{tile_to_world, ParsedLevel};
use crate::player::PLAYER_EYE_HEIGHT;

use super::error::{ErrorStage, GeneratorError};
use super::ir::{GridCoord, IntendedTopology, RegionRole};
use super::validation::{MovementGraph, MovementNode};

// ─── Capture view ───────────────────────────────────────────────────────────

/// A validated, deterministic capture-site camera specification.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureView {
    /// Unique label for this view (e.g. "spawn", "junction_3").
    pub label: String,
    /// Category for grouping and filtering.
    pub category: CaptureViewCategory,
    /// Camera eye position in world space.
    pub eye: Vec3,
    /// Camera look-at target in world space.
    pub look_at: Vec3,
}

/// Semantic category of a capture view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CaptureViewCategory {
    Spawn,
    RequiredRoute,
    DistantLandmark,
    Junction,
    RampApproach,
    RampCrest,
    UpperLanding,
    DarkBranch,
}

impl CaptureViewCategory {
    /// Canonical ordinal for stable sorting.
    const fn ordinal(self) -> u8 {
        match self {
            Self::Spawn => 0,
            Self::RequiredRoute => 1,
            Self::DistantLandmark => 2,
            Self::Junction => 3,
            Self::RampApproach => 4,
            Self::RampCrest => 5,
            Self::UpperLanding => 6,
            Self::DarkBranch => 7,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Spawn => "spawn",
            Self::RequiredRoute => "required_route",
            Self::DistantLandmark => "distant_landmark",
            Self::Junction => "junction",
            Self::RampApproach => "ramp_approach",
            Self::RampCrest => "ramp_crest",
            Self::UpperLanding => "upper_landing",
            Self::DarkBranch => "dark_branch",
        }
    }
}

impl CaptureView {
    /// Create a validated capture view.
    fn new(
        category: CaptureViewCategory,
        label_suffix: &str,
        eye_coord: GridCoord,
        look_at_coord: GridCoord,
        config: &super::config::NormalizedGeneratorConfig,
    ) -> Result<Self, GeneratorError> {
        // Validate both coordinates are within bounds.
        if eye_coord.layer >= config.layers().2
            || eye_coord.x >= config.width()
            || eye_coord.y >= config.height()
        {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!(
                    "capture_view_eye_oob category={} coord={eye_coord}",
                    category.label()
                ),
            });
        }
        if look_at_coord.layer >= config.layers().2
            || look_at_coord.x >= config.width()
            || look_at_coord.y >= config.height()
        {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!(
                    "capture_view_lookat_oob category={} coord={look_at_coord}",
                    category.label()
                ),
            });
        }

        // Non-degenerate: eye and look-at must differ.
        if eye_coord == look_at_coord {
            return Err(GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: format!(
                    "capture_view_degenerate category={} coord={eye_coord}",
                    category.label()
                ),
            });
        }

        let eye_world = grid_to_world(eye_coord);
        let look_at_world = grid_to_world(look_at_coord);

        let label = if label_suffix.is_empty() {
            category.label().to_string()
        } else {
            format!("{}_{}", category.label(), label_suffix)
        };

        Ok(Self {
            label,
            category,
            eye: eye_world,
            look_at: look_at_world,
        })
    }
}

// ─── Grid-to-world conversion ───────────────────────────────────────────────

fn grid_to_world(coord: GridCoord) -> Vec3 {
    let base = tile_to_world(
        usize::from(coord.x),
        usize::from(coord.y),
    );
    Vec3::new(
        base.x + 0.5,
        f32::from(coord.layer) * WALL_HEIGHT + PLAYER_EYE_HEIGHT,
        base.z - 0.5,
    )
}

// ─── View derivation ────────────────────────────────────────────────────────

/// Derive deterministic capture-site views from the reconstructed movement
/// graph and accepted topology metadata.
///
/// Views are emitted in this order: spawn, required-route, distant-landmark,
/// junction, ramp-approach, ramp-crest, upper-landing, dark-branch.
/// Missing or unavailable sites produce typed errors.
pub(crate) fn derive_capture_views(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    movement: &MovementGraph,
    config: &super::config::NormalizedGeneratorConfig,
) -> Result<Vec<CaptureView>, GeneratorError> {
    let mut views = Vec::new();

    // ── Spawn ──────────────────────────────────────────────────────────
    let spawn_region = find_region_by_role(topology, RegionRole::Spawn)
        .ok_or_else(|| GeneratorError::IrInvariant {
            stage: ErrorStage::Ir,
            detail: "capture_view_no_spawn_region".into(),
        })?;
    let spawn_coord = to_grid_coord(
        level.spawn.layer as u16,
        level.spawn.x as u16,
        level.spawn.y as u16,
        config,
    )?;
    let spawn_look = find_interior_cell(spawn_region, movement, level, config, spawn_coord)?;
    views.push(CaptureView::new(
        CaptureViewCategory::Spawn,
        "",
        spawn_coord,
        spawn_look,
        config,
    )?);

    // ── Required-route waypoint ────────────────────────────────────────
    if let Some(route_view) = derive_route_view(topology, movement, level, config)? {
        views.push(route_view);
    }

    // ── Distant landmark ───────────────────────────────────────────────
    if let Some(landmark_view) = derive_landmark_view(topology, movement, level, config)? {
        views.push(landmark_view);
    }

    // ── Junctions ──────────────────────────────────────────────────────
    let junction_views = derive_junction_views(topology, movement, level, config)?;
    views.extend(junction_views);

    // ── Ramp approach ──────────────────────────────────────────────────
    let ramp_approach_views = derive_ramp_approach_views(topology, movement, level, config)?;
    views.extend(ramp_approach_views);

    // ── Ramp crest ─────────────────────────────────────────────────────
    let ramp_crest_views = derive_ramp_crest_views(topology, movement, level, config)?;
    views.extend(ramp_crest_views);

    // ── Upper landing ──────────────────────────────────────────────────
    let upper_landing_views = derive_upper_landing_views(topology, movement, level, config)?;
    views.extend(upper_landing_views);

    // ── Dark branch ────────────────────────────────────────────────────
    if let Some(dark_view) = derive_dark_branch_view(topology, movement, level, config)? {
        views.push(dark_view);
    }

    Ok(views)
}

// ─── Per-category derivation helpers ────────────────────────────────────────

fn find_region_by_role(
    topology: &IntendedTopology,
    role: RegionRole,
) -> Option<&super::ir::PlacedRegion> {
    topology.regions.iter().find(|r| r.role == role)
}

fn to_grid_coord(
    layer: u16,
    x: u16,
    y: u16,
    config: &super::config::NormalizedGeneratorConfig,
) -> Result<GridCoord, GeneratorError> {
    GridCoord::new(layer, x, y, config.width(), config.height(), config.layers().2)
}

fn find_interior_cell(
    region: &super::ir::PlacedRegion,
    movement: &MovementGraph,
    _level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
    avoid: GridCoord,
) -> Result<GridCoord, GeneratorError> {
    // Find the walkable cell in this region farthest from `avoid`.
    let mut candidates: Vec<(MovementNode, u64)> = movement
        .nodes
        .iter()
        .filter(|n| {
            n.layer == region.layer
                && n.x >= region.footprint.0
                && n.x < region.footprint.0.saturating_add(region.footprint.2)
                && n.y >= region.footprint.1
                && n.y < region.footprint.1.saturating_add(region.footprint.3)
        })
        .map(|n| {
            let dist =
                u64::from(n.x.abs_diff(avoid.x)) + u64::from(n.y.abs_diff(avoid.y));
            (*n, dist)
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| a.0.y.cmp(&b.0.y))
            .then_with(|| a.0.x.cmp(&b.0.x))
    });

    candidates
        .first()
        .map(|(n, _)| to_grid_coord(n.layer, n.x, n.y, config))
        .transpose()?
        .ok_or_else(|| GeneratorError::IrInvariant {
            stage: ErrorStage::Ir,
            detail: format!(
                "capture_view_no_interior_cell region={}",
                region.id.raw()
            ),
        })
}

fn derive_route_view(
    topology: &IntendedTopology,
    movement: &MovementGraph,
    level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
) -> Result<Option<CaptureView>, GeneratorError> {
    // Pick the longest required edge (by path_witness length) and use its midpoint.
    let mut candidates: Vec<&super::ir::IntendedEdge> = topology
        .edges
        .iter()
        .filter(|e| e.required && e.transition.is_none() && !e.path_witness.is_empty())
        .collect();
    candidates.sort_by(|a, b| {
        b.path_witness
            .len()
            .cmp(&a.path_witness.len())
            .then_with(|| a.id.raw().cmp(&b.id.raw()))
    });

    let Some(edge) = candidates.first() else {
        return Ok(None);
    };

    let mid_idx = edge.path_witness.len() / 2;
    let eye_coord = edge.path_witness[mid_idx];
    ensure_walkable(eye_coord, movement, level, config, "required_route")?;

    // Look toward the goal end of the path.
    let look_idx = (edge.path_witness.len() - 1).min(mid_idx + 2);
    let look_coord = edge.path_witness[look_idx];

    let view = CaptureView::new(
        CaptureViewCategory::RequiredRoute,
        &edge.id.raw().to_string(),
        eye_coord,
        look_coord,
        config,
    )?;
    Ok(Some(view))
}

fn derive_landmark_view(
    topology: &IntendedTopology,
    movement: &MovementGraph,
    level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
) -> Result<Option<CaptureView>, GeneratorError> {
    // Distant landmark: center cell with look-at toward interior.
    let Some(region) = find_region_by_role(topology, RegionRole::DistantLandmark) else {
        return Ok(None);
    };

    let center = to_grid_coord(
        region.layer,
        region.footprint.0 + region.footprint.2 / 2,
        region.footprint.1 + region.footprint.3 / 2,
        config,
    )?;

    let center = ensure_walkable_or_in_region(
        center,
        region,
        movement,
        level,
        config,
        "distant_landmark",
    )?;

    let look = find_interior_cell(region, movement, level, config, center)?;

    let view = CaptureView::new(
        CaptureViewCategory::DistantLandmark,
        "",
        center,
        look,
        config,
    )?;
    Ok(Some(view))
}

fn derive_junction_views(
    topology: &IntendedTopology,
    movement: &MovementGraph,
    level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
) -> Result<Vec<CaptureView>, GeneratorError> {
    let mut views = Vec::new();
    for region in &topology.regions {
        if region.role != RegionRole::Junction {
            continue;
        }
        let center = to_grid_coord(
            region.layer,
            region.footprint.0 + region.footprint.2 / 2,
            region.footprint.1 + region.footprint.3 / 2,
            config,
        )?;
        let center =
            ensure_walkable_or_in_region(center, region, movement, level, config, "junction")?;

        let look = find_interior_cell(region, movement, level, config, center)?;

        let view = CaptureView::new(
            CaptureViewCategory::Junction,
            &region.id.raw().to_string(),
            center,
            look,
            config,
        )?;
        views.push(view);
    }

    // Sort by region ID for determinism.
    views.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(views)
}

fn derive_ramp_approach_views(
    topology: &IntendedTopology,
    movement: &MovementGraph,
    level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
) -> Result<Vec<CaptureView>, GeneratorError> {
    let mut views = Vec::new();
    for transition in &topology.transitions {
        // Lower approach cell: the first cell of lower_approach_cells if available.
        if let Some(approach) = transition.lower_approach_cells.first() {
            ensure_walkable(*approach, movement, level, config, "ramp_approach")?;
            // Look toward the first ramp cell.
            if let Some(ramp_cell) = transition.ramp_run_cells.first() {
                let view = CaptureView::new(
                    CaptureViewCategory::RampApproach,
                    &transition.id.raw().to_string(),
                    *approach,
                    *ramp_cell,
                    config,
                )?;
                views.push(view);
            }
        }
    }
    views.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(views)
}

fn derive_ramp_crest_views(
    topology: &IntendedTopology,
    movement: &MovementGraph,
    level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
) -> Result<Vec<CaptureView>, GeneratorError> {
    let mut views = Vec::new();
    for transition in &topology.transitions {
        // Ramp crest: last ramp cell (R2), look toward landing.
        if let Some(crest) = transition.ramp_run_cells.last() {
            ensure_walkable(*crest, movement, level, config, "ramp_crest")?;
            if let Some(landing) = transition.landing_cells.first() {
                let view = CaptureView::new(
                    CaptureViewCategory::RampCrest,
                    &transition.id.raw().to_string(),
                    *crest,
                    *landing,
                    config,
                )?;
                views.push(view);
            }
        }
    }
    views.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(views)
}

fn derive_upper_landing_views(
    topology: &IntendedTopology,
    movement: &MovementGraph,
    level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
) -> Result<Vec<CaptureView>, GeneratorError> {
    let mut views = Vec::new();
    for transition in &topology.transitions {
        // Upper landing: first landing cell, look back toward ramp crest.
        if let Some(landing) = transition.landing_cells.first() {
            ensure_walkable(*landing, movement, level, config, "upper_landing")?;
            if let Some(crest) = transition.ramp_run_cells.last() {
                let view = CaptureView::new(
                    CaptureViewCategory::UpperLanding,
                    &transition.id.raw().to_string(),
                    *landing,
                    *crest,
                    config,
                )?;
                views.push(view);
            }
        }
    }
    views.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(views)
}

fn derive_dark_branch_view(
    topology: &IntendedTopology,
    movement: &MovementGraph,
    level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
) -> Result<Option<CaptureView>, GeneratorError> {
    // Dark branch: an optional dead-end farthest from spawn or farthest from
    // any junction, representing the least-traversed part of the dungeon.
    let dead_end_regions: Vec<&super::ir::PlacedRegion> = topology
        .regions
        .iter()
        .filter(|r| r.role == RegionRole::DeadEnd || r.role == RegionRole::OptionalBranch)
        .collect();

    if dead_end_regions.is_empty() {
        return Ok(None);
    }

    // Find spawn for distance reference.
    let Some(spawn_region) = find_region_by_role(topology, RegionRole::Spawn) else {
        return Ok(None);
    };

    // Rank dead-end/optional regions by distance from spawn.
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct RankedRegion {
        dist: u64,
        region_id: u32,
    }

    let mut ranked: Vec<(RankedRegion, &super::ir::PlacedRegion)> = dead_end_regions
        .iter()
        .map(|region| {
            let dx = region
                .footprint
                .0
                .abs_diff(spawn_region.footprint.0)
                .max(
                    region
                        .footprint
                        .0
                        .saturating_add(region.footprint.2)
                        .abs_diff(
                            spawn_region
                                .footprint
                                .0
                                .saturating_add(spawn_region.footprint.2),
                        ),
                );
            let dy = region
                .footprint
                .1
                .abs_diff(spawn_region.footprint.1)
                .max(
                    region
                        .footprint
                        .1
                        .saturating_add(region.footprint.3)
                        .abs_diff(
                            spawn_region
                                .footprint
                                .1
                                .saturating_add(spawn_region.footprint.3),
                        ),
                );
            let dist = u64::from(dx) + u64::from(dy);
            (RankedRegion { dist, region_id: region.id.raw() }, *region)
        })
        .collect();

    ranked.sort_by(|a, b| {
        b.0.cmp(&a.0)
    });

    let (_, region) = ranked[0];
    let center = to_grid_coord(
        region.layer,
        region.footprint.0 + region.footprint.2 / 2,
        region.footprint.1 + region.footprint.3 / 2,
        config,
    )?;
    let center =
        ensure_walkable_or_in_region(center, region, movement, level, config, "dark_branch")?;

    let look = find_interior_cell(region, movement, level, config, center)?;

    let view = CaptureView::new(
        CaptureViewCategory::DarkBranch,
        &region.id.raw().to_string(),
        center,
        look,
        config,
    )?;
    Ok(Some(view))
}

// ─── Validation helpers ─────────────────────────────────────────────────────

fn ensure_walkable(
    coord: GridCoord,
    movement: &MovementGraph,
    level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
    category: &str,
) -> Result<(), GeneratorError> {
    let node = MovementNode {
        layer: coord.layer,
        x: coord.x,
        y: coord.y,
    };
    if !movement.nodes.contains(&node) {
        return Err(GeneratorError::IrInvariant {
            stage: ErrorStage::Ir,
            detail: format!(
                "capture_view_{category}_not_walkable coord={coord}"
            ),
        });
    }
    // Also check tile is walkable.
    if coord.layer >= config.layers().2
        || coord.x >= config.width()
        || coord.y >= config.height()
    {
        return Err(GeneratorError::IrInvariant {
            stage: ErrorStage::Ir,
            detail: format!(
                "capture_view_{category}_oob coord={coord}"
            ),
        });
    }
    let tile = level.tile_at_3d(
        usize::from(coord.layer),
        usize::from(coord.x),
        usize::from(coord.y),
    );
    if !matches!(
        tile,
        crate::layout::Tile::Floor
            | crate::layout::Tile::RampNorth(_)
            | crate::layout::Tile::RampEast(_)
            | crate::layout::Tile::RampSouth(_)
            | crate::layout::Tile::RampWest(_)
    ) {
        return Err(GeneratorError::IrInvariant {
            stage: ErrorStage::Ir,
            detail: format!(
                "capture_view_{category}_not_floor coord={coord} tile={tile:?}"
            ),
        });
    }
    Ok(())
}

fn ensure_walkable_or_in_region(
    coord: GridCoord,
    region: &super::ir::PlacedRegion,
    movement: &MovementGraph,
    level: &ParsedLevel,
    config: &super::config::NormalizedGeneratorConfig,
    category: &str,
) -> Result<GridCoord, GeneratorError> {
    // First try the exact coordinate.
    if movement.nodes.contains(&MovementNode {
        layer: coord.layer,
        x: coord.x,
        y: coord.y,
    }) {
        ensure_walkable(coord, movement, level, config, category)?;
        return Ok(coord);
    }

    // Fall back to any walkable cell in region.
    let alt = movement
        .nodes
        .iter()
        .find(|n| {
            n.layer == region.layer
                && n.x >= region.footprint.0
                && n.x < region.footprint.0.saturating_add(region.footprint.2)
                && n.y >= region.footprint.1
                && n.y < region.footprint.1.saturating_add(region.footprint.3)
        })
        .copied();

    if let Some(alt) = alt {
        let alt_coord = to_grid_coord(alt.layer, alt.x, alt.y, config)?;
        ensure_walkable(alt_coord, movement, level, config, category)?;
        return Ok(alt_coord);
    }

    Err(GeneratorError::IrInvariant {
        stage: ErrorStage::Ir,
        detail: format!(
            "capture_view_{category}_no_walkable_in_region region={}",
            region.id.raw()
        ),
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::GeneratorConfig;
    use crate::layout::{ParsedLevel, Tile, TileCoord};

    fn test_config() -> super::super::config::NormalizedGeneratorConfig {
        GeneratorConfig::custom(64, 64, 2).normalize().unwrap()
    }

    fn test_level() -> ParsedLevel {
        let w = 64usize;
        let h = 64usize;
        let mut tiles = vec![Tile::Wall; w * h];
        for y in 1..8 {
            for x in 1..8 {
                tiles[y * w + x] = Tile::Floor;
            }
        }
        let layer2 = vec![Tile::Wall; w * h];
        ParsedLevel {
            width: w,
            height: h,
            layers: vec![tiles, layer2],
            spawn: TileCoord {
                layer: 0,
                x: 3,
                y: 3,
            },
            model_markers: vec![],
            light_markers: vec![],
        }
    }

    #[test]
    fn grid_to_world_places_center_above_floor() {
        let config = test_config();
        let coord = GridCoord::new(1, 5, 5, 64, 64, 2).unwrap();
        let world = grid_to_world(coord);
        assert!((world.x - 5.5).abs() < 0.01);
        assert!((world.y - (WALL_HEIGHT + PLAYER_EYE_HEIGHT)).abs() < 0.01);
        assert!((world.z + 5.5).abs() < 0.01);
    }

    #[test]
    fn capture_view_rejects_degenerate_eye_equals_lookat() {
        let config = test_config();
        let coord = GridCoord::new(0, 5, 5, 64, 64, 2).unwrap();
        assert!(CaptureView::new(
            CaptureViewCategory::Spawn,
            "",
            coord,
            coord,
            &config,
        )
        .is_err());
    }
}
