//! Phase 06 — Marker classification, placement, and relocation.
//!
//! ## Marker pipeline
//! 1. Classify every walkable tile from the reconstructed movement graph.
//! 2. Place exactly one spawn on non-ramp reachable floor in the spawn region.
//! 3. Rank light candidates by topology priority and emit ≤16 in order.
//! 4. Fit model markers where prop envelopes clear. Reject co-location.
//! 5. Compose marker relocation with Phase 05 repair.
//! 6. Provide validator functions for the combined marker state.

use std::collections::{BTreeMap, BTreeSet};

use crate::content::{
    light_preset_for_marker_index, prop_for_marker_index, LightPresetId, PropPlacementEnvelope,
};
use crate::layout::{ParsedLevel, Tile, TileCoord};

use super::config::NormalizedGeneratorConfig;
use super::error::{ErrorStage, GeneratorError};
use super::ir::{Direction, GridCoord, IntendedTopology, PlacedRegion, RegionId, RegionRole};
use super::validation::{MovementGraph, MovementNode};

// ─── Tile classification ───────────────────────────────────────────────────

/// Tile class derived from the topology and movement graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum TileClass {
    /// Straight corridor segment.
    CorridorStraight,
    /// Corridor corner / turn.
    CorridorCorner,
    /// Junction where ≥3 corridors meet.
    Junction,
    /// Interior cell of a room region.
    RoomInterior,
    /// Edge cell of a room region (adjacent to wall or void).
    RoomEdge,
    /// Dead-end terminal cell.
    DeadEnd,
    /// Floor cell inside a ramp shaft.
    ShaftFloor,
    /// Edge cell adjacent to a ramp shaft.
    ShaftEdge,
    /// Cell immediately in front of a ramp entry.
    RampEntry,
    /// Unclassified walkable (fallback).
    Unclassified,
}

impl TileClass {
    fn label(self) -> &'static str {
        match self {
            Self::CorridorStraight => "corridor_straight",
            Self::CorridorCorner => "corridor_corner",
            Self::Junction => "junction",
            Self::RoomInterior => "room_interior",
            Self::RoomEdge => "room_edge",
            Self::DeadEnd => "dead_end",
            Self::ShaftFloor => "shaft_floor",
            Self::ShaftEdge => "shaft_edge",
            Self::RampEntry => "ramp_entry",
            Self::Unclassified => "unclassified",
        }
    }
}

// ─── Marker intent ──────────────────────────────────────────────────────────

/// A placed marker with intent metadata.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct PlacedMarker {
    pub(super) layer: u16,
    pub(super) x: u16,
    pub(super) y: u16,
    pub(super) class: TileClass,
    pub(super) region: Option<RegionId>,
}

/// Light marker with preset assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PlacedLight {
    pub(super) coord: GridCoord,
    pub(super) preset: LightPresetId,
    pub(super) required: bool,
    pub(super) intent_label: &'static str,
}

/// Model marker with resolved prop index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PlacedModel {
    pub(super) coord: GridCoord,
    pub(super) prop_index: usize,
    pub(super) required: bool,
}

/// Omitted optional marker record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MarkerOmission {
    pub(super) intent_label: String,
    pub(super) reason: String,
    pub(super) candidates_considered: usize,
}

/// Complete marker placement result.
#[derive(Debug, Clone)]
pub(super) struct MarkerPlacement {
    pub(super) spawn: GridCoord,
    pub(super) lights: Vec<PlacedLight>,
    pub(super) models: Vec<PlacedModel>,
    pub(super) omissions: Vec<MarkerOmission>,
    pub(super) tile_classes: BTreeMap<GridCoord, TileClass>,
}

// ─── Prop envelope fit ─────────────────────────────────────────────────────

/// Check whether a prop with the given world-space half-extents fits at a tile
/// coordinate. The envelope is centered on the tile center and must not
/// intersect walls, voids, ramps, other markers, or go out of bounds.
pub(super) fn prop_fits_at(
    layer: u16,
    x: u16,
    y: u16,
    half_extents: [f32; 3],
    level: &ParsedLevel,
    occupied: &BTreeSet<GridCoord>,
    config: &NormalizedGeneratorConfig,
) -> bool {
    // Convert tile center to world center, then expand by half-extents.
    // Tile size is 1.0 world unit.
    let tile_cx = x as f32 + 0.5;
    let tile_cz = y as f32 + 0.5;

    let min_x = tile_cx - half_extents[0];
    let max_x = tile_cx + half_extents[0];
    let min_z = tile_cz - half_extents[2];
    let max_z = tile_cz + half_extents[2];

    // Convert to tile intervals with epsilon policy: use 1e-4 outward expansion
    // to avoid false passes on exact boundary, then floor/ceil.
    const EPS: f32 = 1e-4;
    let tx_min = (min_x + EPS).floor() as i32;
    let tx_max = (max_x - EPS).ceil() as i32;
    let tz_min = (min_z + EPS).floor() as i32;
    let tz_max = (max_z - EPS).ceil() as i32;

    let w = config.width() as i32;
    let h = config.height() as i32;

    for ty in tz_min..tz_max {
        for tx in tx_min..tx_max {
            if tx < 0 || tx >= w || ty < 0 || ty >= h {
                return false;
            }
            let gx = tx as u16;
            let gy = ty as u16;
            let tile = level.tile_at_3d(layer as usize, gx as usize, gy as usize);
            if tile != Tile::Floor {
                return false;
            }
            let coord = match GridCoord::new(
                layer,
                gx,
                gy,
                config.width(),
                config.height(),
                config.layers().2,
            ) {
                Ok(c) => c,
                Err(_) => return false,
            };
            if occupied.contains(&coord) {
                return false;
            }
        }
    }
    true
}

// ─── Tile classification ───────────────────────────────────────────────────

/// Classify every walkable tile using the movement graph and topology.
pub(super) fn classify_tiles(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    movement: &MovementGraph,
    config: &NormalizedGeneratorConfig,
) -> Result<BTreeMap<GridCoord, TileClass>, GeneratorError> {
    let width = config.width();
    let height = config.height();
    let layers = config.layers().2;

    // Build node → region mapping for fast lookup.
    let node_regions: BTreeMap<MovementNode, BTreeSet<RegionId>> = movement.node_regions.clone();

    // Count adjacency degree for each node.
    let degree: BTreeMap<MovementNode, usize> = movement
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let deg = movement.adjacency.get(&i).map_or(0, |nbrs| nbrs.len());
            (*node, deg)
        })
        .collect();

    // Build region role lookup.
    let region_roles: BTreeMap<RegionId, RegionRole> =
        topology.regions.iter().map(|r| (r.id, r.role)).collect();

    // Precompute region edges for edge detection.
    // A node is on a room edge if any cardinal neighbor is non-walkable or in a different region.
    let tile_at = |layer: u16, x: u16, y: u16| -> Option<Tile> {
        if layer >= layers || x >= width || y >= height {
            return None;
        }
        Some(level.tile_at_3d(layer as usize, x as usize, y as usize))
    };

    let walkable_at = |layer: u16, x: u16, y: u16| -> bool {
        tile_at(layer, x, y).map_or(false, |t| {
            matches!(
                t,
                Tile::Floor
                    | Tile::RampNorth(_)
                    | Tile::RampEast(_)
                    | Tile::RampSouth(_)
                    | Tile::RampWest(_)
            )
        })
    };

    // Build ramp approach set: cells immediately in front of a ramp.
    let mut ramp_entry_set = BTreeSet::new();
    for layer in 0..layers {
        for y in 0..height {
            for x in 0..width {
                let tile = level.tile_at_3d(layer as usize, x as usize, y as usize);
                if matches!(
                    tile,
                    Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_)
                ) {
                    // The approach cell is one step back from the ramp entry.
                    let (dx, dy): (i32, i32) = match tile {
                        Tile::RampNorth(_) => (0, 1),
                        Tile::RampEast(_) => (-1, 0),
                        Tile::RampSouth(_) => (0, -1),
                        Tile::RampWest(_) => (1, 0),
                        _ => continue,
                    };
                    let ax = (x as i32 + dx).max(0).min(width as i32 - 1) as u16;
                    let ay = (y as i32 + dy).max(0).min(height as i32 - 1) as u16;
                    if walkable_at(layer, ax, ay) {
                        ramp_entry_set
                            .insert(GridCoord::new(layer, ax, ay, width, height, layers).ok());
                    }
                }
            }
        }
    }

    let mut classes = BTreeMap::new();

    for node in &movement.nodes {
        let coord = match GridCoord::new(node.layer, node.x, node.y, width, height, layers) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let tile = level.tile_at_3d(node.layer as usize, node.x as usize, node.y as usize);
        if !matches!(
            tile,
            Tile::Floor
                | Tile::RampNorth(_)
                | Tile::RampEast(_)
                | Tile::RampSouth(_)
                | Tile::RampWest(_)
        ) {
            continue;
        }

        let deg = degree.get(node).copied().unwrap_or(0);
        let regions = node_regions.get(node).cloned().unwrap_or_default();

        // Determine primary region and its role.
        let primary_role = regions
            .iter()
            .filter_map(|rid| region_roles.get(rid))
            .min_by_key(|r| r.ordinal())
            .copied();

        // Check if on a ramp tile.
        let is_ramp = matches!(
            tile,
            Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_)
        );

        // Check if adjacent to non-walkable (edge detection).
        let dirs = [(0i32, -1i32), (1, 0), (0, 1), (-1, 0)];
        let walkable_neighbors = dirs
            .iter()
            .filter(|&&(dx, dy)| {
                let nx = node.x as i32 + dx;
                let ny = node.y as i32 + dy;
                nx >= 0
                    && ny >= 0
                    && (nx as u16) < width
                    && (ny as u16) < height
                    && walkable_at(node.layer, nx as u16, ny as u16)
            })
            .count();

        let class = if is_ramp {
            TileClass::ShaftFloor
        } else if ramp_entry_set.contains(&Some(coord)) {
            TileClass::RampEntry
        } else if deg == 1 {
            // Dead end (only one walkable neighbor).
            match primary_role {
                Some(RegionRole::DeadEnd) => TileClass::DeadEnd,
                _ => TileClass::Unclassified,
            }
        } else if deg >= 3 {
            TileClass::Junction
        } else if deg == 2 {
            // Could be corridor straight or corner.
            // Check if the two neighbors are opposite (straight) or adjacent (corner).
            let neighbors: Vec<(i32, i32)> = dirs
                .iter()
                .filter(|&&(dx, dy)| {
                    let nx = node.x as i32 + dx;
                    let ny = node.y as i32 + dy;
                    nx >= 0
                        && ny >= 0
                        && (nx as u16) < width
                        && (ny as u16) < height
                        && walkable_at(node.layer, nx as u16, ny as u16)
                })
                .map(|&(dx, dy)| (dx, dy))
                .collect();
            if neighbors.len() == 2 {
                let (dx0, dy0) = neighbors[0];
                let (dx1, dy1) = neighbors[1];
                if dx0 + dx1 == 0 && dy0 + dy1 == 0 {
                    TileClass::CorridorStraight
                } else {
                    TileClass::CorridorCorner
                }
            } else {
                TileClass::CorridorStraight
            }
        } else {
            // deg == 0 or unclassified.
            match primary_role {
                Some(RegionRole::Spawn)
                | Some(RegionRole::DistantLandmark)
                | Some(RegionRole::MajorLandmark)
                | Some(RegionRole::Junction)
                | Some(RegionRole::DeadEnd)
                | Some(RegionRole::RequiredRoute)
                | Some(RegionRole::OptionalBranch)
                | Some(RegionRole::OrdinaryRoom) => {
                    if walkable_neighbors < 4 {
                        TileClass::RoomEdge
                    } else {
                        TileClass::RoomInterior
                    }
                }
                Some(RegionRole::VerticalHub) => TileClass::ShaftEdge,
                _ => TileClass::Unclassified,
            }
        };

        classes.insert(coord, class);
    }

    Ok(classes)
}

// ─── Spawn placement ────────────────────────────────────────────────────────

/// Place exactly one spawn on non-ramp reachable floor in the spawn region.
pub(super) fn place_spawn(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    movement: &MovementGraph,
    config: &NormalizedGeneratorConfig,
) -> Result<GridCoord, GeneratorError> {
    let spawn_regions: Vec<&PlacedRegion> = topology
        .regions
        .iter()
        .filter(|r| r.role == RegionRole::Spawn)
        .collect();
    if spawn_regions.len() != 1 {
        return Err(GeneratorError::IrInvariant {
            stage: ErrorStage::Ir,
            detail: format!("spawn_region_count={}", spawn_regions.len()),
        });
    }
    let spawn_region = spawn_regions[0];

    let mut candidates: Vec<(GridCoord, u32)> = Vec::new();
    let width = config.width();
    let height = config.height();

    for node in &movement.nodes {
        if node.layer != spawn_region.layer {
            continue;
        }
        let tile = level.tile_at_3d(node.layer as usize, node.x as usize, node.y as usize);
        if !matches!(tile, Tile::Floor) {
            continue;
        }
        // Must be inside spawn region footprint.
        if node.x < spawn_region.footprint.0
            || node.x
                >= spawn_region
                    .footprint
                    .0
                    .saturating_add(spawn_region.footprint.2)
            || node.y < spawn_region.footprint.1
            || node.y
                >= spawn_region
                    .footprint
                    .1
                    .saturating_add(spawn_region.footprint.3)
        {
            continue;
        }
        // Distance from walls: count walkable neighbors.
        let walkable = count_walkable_neighbors(node.layer, node.x, node.y, level, width, height);
        let dist_from_wall = 4u32.saturating_sub(walkable as u32);
        let center_dist =
            (node.x as i32 - spawn_region.footprint.0 as i32 - spawn_region.footprint.2 as i32 / 2)
                .unsigned_abs()
                + (node.y as i32
                    - spawn_region.footprint.1 as i32
                    - spawn_region.footprint.3 as i32 / 2)
                    .unsigned_abs();
        let score = dist_from_wall * 1000 + center_dist;
        let coord = GridCoord::new(node.layer, node.x, node.y, width, height, config.layers().2)?;
        candidates.push((coord, score));
    }

    if candidates.is_empty() {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Ir,
            constraint: "spawn_placement",
            required: 1,
            available: 0,
        });
    }

    // Sort by score descending (farther from walls first), then canonical tie-break.
    candidates.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| a.0.layer.cmp(&b.0.layer))
            .then_with(|| a.0.y.cmp(&b.0.y))
            .then_with(|| a.0.x.cmp(&b.0.x))
    });

    Ok(candidates[0].0)
}

fn count_walkable_neighbors(
    layer: u16,
    x: u16,
    y: u16,
    level: &ParsedLevel,
    width: u16,
    height: u16,
) -> usize {
    [(0i32, -1i32), (1, 0), (0, 1), (-1, 0)]
        .iter()
        .filter(|&&(dx, dy)| {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                return false;
            }
            let tile = level.tile_at_3d(layer as usize, nx as usize, ny as usize);
            matches!(
                tile,
                Tile::Floor
                    | Tile::RampNorth(_)
                    | Tile::RampEast(_)
                    | Tile::RampSouth(_)
                    | Tile::RampWest(_)
            )
        })
        .count()
}

// ─── Light placement ────────────────────────────────────────────────────────

/// A light candidate with ranking info.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LightCandidate {
    coord: GridCoord,
    /// Required intents rank first.
    required: bool,
    role: Option<RegionRole>,
    /// Semantic priority: lower = higher priority.
    semantic_rank: u8,
    /// Distance from spawn.
    distance_rank: u64,
    /// Tile class ordinal.
    class_ordinal: u8,
    /// Region ID for tie-breaks.
    region_id: u32,
    /// Intent label for diagnostics.
    intent_label: &'static str,
}

fn class_ordinal(c: TileClass) -> u8 {
    match c {
        TileClass::RampEntry => 0,
        TileClass::Junction => 1,
        TileClass::CorridorCorner => 2,
        TileClass::RoomInterior => 3,
        TileClass::RoomEdge => 4,
        TileClass::ShaftEdge => 5,
        TileClass::DeadEnd => 6,
        TileClass::CorridorStraight => 7,
        TileClass::ShaftFloor => 8,
        TileClass::Unclassified => 9,
    }
}

/// Rank and emit light markers. Returns at most 16 lights.
pub(super) fn place_lights(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    movement: &MovementGraph,
    tile_classes: &BTreeMap<GridCoord, TileClass>,
    spawn: GridCoord,
    config: &NormalizedGeneratorConfig,
) -> Result<(Vec<PlacedLight>, Vec<MarkerOmission>), GeneratorError> {
    let max_lights = usize::from(u16::try_from(config.max_lights()).unwrap_or(16));
    let width = config.width();
    let height = config.height();
    let layers = config.layers().2;

    // Build region role lookup.
    let region_roles: BTreeMap<RegionId, RegionRole> =
        topology.regions.iter().map(|r| (r.id, r.role)).collect();

    // Compute distances from spawn via BFS on movement graph.
    let distances = compute_distances(spawn, movement);

    // Collect candidates.
    let mut candidates: Vec<LightCandidate> = Vec::new();

    for node in &movement.nodes {
        let coord = match GridCoord::new(node.layer, node.x, node.y, width, height, layers) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Skip spawn cell, ramp tiles, walls, voids.
        if coord == spawn {
            continue;
        }
        let tile = level.tile_at_3d(node.layer as usize, node.x as usize, node.y as usize);
        if !matches!(tile, Tile::Floor) {
            continue;
        }

        let class = tile_classes
            .get(&coord)
            .copied()
            .unwrap_or(TileClass::Unclassified);
        let regions = movement.node_regions.get(node).cloned().unwrap_or_default();
        let primary_role = regions
            .iter()
            .filter_map(|rid| region_roles.get(rid))
            .min_by_key(|r| r.ordinal())
            .copied();

        let intent_label = primary_role.map_or("scenery", |r| r.label());

        // Semantic rank: required-route first, junctions, ramp entries, room interiors, dead ends, optional branches last.
        let semantic_rank = semantic_light_rank(primary_role, class);
        let distance_rank = distances.get(node).copied().unwrap_or(u64::MAX);

        candidates.push(LightCandidate {
            coord,
            required: false,
            role: primary_role,
            semantic_rank,
            distance_rank,
            class_ordinal: class_ordinal(class),
            region_id: regions.iter().next().map_or(u32::MAX, |r| r.raw()),
            intent_label,
        });
    }

    let candidate_order = |a: &LightCandidate, b: &LightCandidate| {
        b.required
            .cmp(&a.required)
            .then_with(|| a.semantic_rank.cmp(&b.semantic_rank))
            .then_with(|| a.distance_rank.cmp(&b.distance_rank))
            .then_with(|| a.class_ordinal.cmp(&b.class_ordinal))
            .then_with(|| a.region_id.cmp(&b.region_id))
            .then_with(|| a.coord.layer.cmp(&b.coord.layer))
            .then_with(|| a.coord.y.cmp(&b.coord.y))
            .then_with(|| a.coord.x.cmp(&b.coord.x))
    };
    candidates.sort_by(candidate_order);

    // Required lighting is one deterministic navigational intent per semantic
    // role class, not one light for every floor cell in matching regions.
    let required_roles: BTreeSet<RegionRole> = topology
        .regions
        .iter()
        .map(|region| region.role)
        .filter(|role| is_required_light_role(Some(*role)))
        .collect();
    if required_roles.len() > max_lights {
        let required = u64::try_from(required_roles.len()).map_err(|_| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "required_light_role_count",
            }
        })?;
        let available = u64::try_from(max_lights).map_err(|_| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Ir,
                operation: "required_light_capacity",
            }
        })?;
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Ir,
            constraint: "required_light_capacity",
            required,
            available,
        });
    }
    for role in &required_roles {
        let candidate = candidates
            .iter_mut()
            .find(|candidate| candidate.role == Some(*role))
            .ok_or(GeneratorError::MandatoryInfeasibility {
                stage: ErrorStage::Ir,
                constraint: "required_light_unplaced",
                required: 1,
                available: 0,
            })?;
        candidate.required = true;
    }
    candidates.sort_by(candidate_order);

    let mut lights = Vec::new();
    let mut omissions = Vec::new();
    let mut placed = BTreeSet::new();

    for (idx, candidate) in candidates.iter().enumerate() {
        if lights.len() >= max_lights {
            // Record omission for remaining optional candidates.
            if !candidate.required {
                omissions.push(MarkerOmission {
                    intent_label: candidate.intent_label.to_string(),
                    reason: "light_capacity_exhausted".to_string(),
                    candidates_considered: candidates.len().saturating_sub(idx),
                });
            }
            continue;
        }

        // Skip co-location with spawn or already-placed markers.
        if placed.contains(&candidate.coord) {
            continue;
        }

        let preset = light_preset_for_marker_index(lights.len());
        lights.push(PlacedLight {
            coord: candidate.coord,
            preset,
            required: candidate.required,
            intent_label: candidate.intent_label,
        });
        placed.insert(candidate.coord);
    }

    // Check for required intents that couldn't be placed.
    for candidate in &candidates {
        if candidate.required && !placed.contains(&candidate.coord) {
            return Err(GeneratorError::MandatoryInfeasibility {
                stage: ErrorStage::Ir,
                constraint: "required_light_unplaced",
                required: 1,
                available: 0,
            });
        }
    }

    // ParsedLevel marker arrays are canonical by coordinate. Reassign the
    // index-derived preset after sorting so runtime and serialized order agree.
    lights.sort_by_key(|light| (light.coord.layer, light.coord.y, light.coord.x));
    for (index, light) in lights.iter_mut().enumerate() {
        light.preset = light_preset_for_marker_index(index);
    }

    Ok((lights, omissions))
}

fn is_required_light_role(role: Option<RegionRole>) -> bool {
    matches!(
        role,
        Some(RegionRole::DistantLandmark)
            | Some(RegionRole::MajorLandmark)
            | Some(RegionRole::Junction)
            | Some(RegionRole::VerticalHub)
            | Some(RegionRole::RequiredRoute)
    )
}

fn semantic_light_rank(role: Option<RegionRole>, class: TileClass) -> u8 {
    match role {
        Some(RegionRole::RequiredRoute) => 0,
        Some(RegionRole::Junction) => 1,
        Some(RegionRole::VerticalHub) if matches!(class, TileClass::RampEntry) => 2,
        Some(RegionRole::DistantLandmark) | Some(RegionRole::MajorLandmark) => 3,
        Some(RegionRole::DeadEnd) => 4,
        Some(RegionRole::OptionalBranch) => 5,
        Some(RegionRole::OrdinaryRoom) => 6,
        Some(RegionRole::Spawn) => 7,
        Some(RegionRole::VerticalHub) => 8,
        _ => 9,
    }
}

// ─── Model placement ───────────────────────────────────────────────────────

/// Place model markers where prop envelopes fit.
pub(super) fn place_models(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    movement: &MovementGraph,
    tile_classes: &BTreeMap<GridCoord, TileClass>,
    spawn: GridCoord,
    placed_lights: &[PlacedLight],
    envelopes: &[Option<PropPlacementEnvelope>],
    config: &NormalizedGeneratorConfig,
) -> Result<(Vec<PlacedModel>, Vec<MarkerOmission>), GeneratorError> {
    if envelopes.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let width = config.width();
    let height = config.height();
    let layers = config.layers().2;
    let model_cap = usize::try_from(config.model_marker_cap()).unwrap_or(usize::MAX);

    // Build occupied set: spawn, lights, ramps, walls, voids.
    let mut occupied = BTreeSet::new();
    occupied.insert(spawn);
    for light in placed_lights {
        occupied.insert(light.coord);
    }

    // Collect model candidates.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ModelCandidate {
        coord: GridCoord,
        class: TileClass,
        distance_from_spawn: u64,
        region_id: u32,
    }

    let region_roles: BTreeMap<RegionId, RegionRole> =
        topology.regions.iter().map(|r| (r.id, r.role)).collect();
    let distances = compute_distances(spawn, movement);

    let mut candidates: Vec<ModelCandidate> = Vec::new();

    for node in &movement.nodes {
        let coord = match GridCoord::new(node.layer, node.x, node.y, width, height, layers) {
            Ok(c) => c,
            Err(_) => continue,
        };

        if occupied.contains(&coord) {
            continue;
        }
        if coord == spawn {
            continue;
        }

        let tile = level.tile_at_3d(node.layer as usize, node.x as usize, node.y as usize);
        if tile != Tile::Floor {
            continue;
        }

        let class = tile_classes
            .get(&coord)
            .copied()
            .unwrap_or(TileClass::Unclassified);
        let regions = movement.node_regions.get(node).cloned().unwrap_or_default();
        let distance_rank = distances.get(node).copied().unwrap_or(u64::MAX);

        candidates.push(ModelCandidate {
            coord,
            class,
            distance_from_spawn: distance_rank,
            region_id: regions.iter().next().map_or(u32::MAX, |r| r.raw()),
        });
    }

    // Sort: prioritize dead-end reward nooks, landmark side cells, off-route alcoves.
    candidates.sort_by(|a, b| {
        model_class_rank(a.class)
            .cmp(&model_class_rank(b.class))
            .then_with(|| b.distance_from_spawn.cmp(&a.distance_from_spawn))
            .then_with(|| a.region_id.cmp(&b.region_id))
            .then_with(|| a.coord.layer.cmp(&b.coord.layer))
            .then_with(|| a.coord.y.cmp(&b.coord.y))
            .then_with(|| a.coord.x.cmp(&b.coord.x))
    });

    let mut models = Vec::new();
    let mut omissions = Vec::new();

    for candidate in &candidates {
        if models.len() >= model_cap {
            omissions.push(MarkerOmission {
                intent_label: "model".to_string(),
                reason: "model_capacity_exhausted".to_string(),
                candidates_considered: candidates.len().saturating_sub(models.len()),
            });
            break;
        }

        // Check prop envelope fit.
        let marker_index = models.len();
        let prop_idx = prop_for_marker_index(marker_index, envelopes.len());
        if let Some(Some(envelope)) = envelopes.get(prop_idx) {
            if !prop_fits_at(
                candidate.coord.layer,
                candidate.coord.x,
                candidate.coord.y,
                envelope.world_half_extents(),
                level,
                &occupied,
                config,
            ) {
                continue;
            }
        } else {
            // If the prop has no valid envelope, skip.
            continue;
        }

        models.push(PlacedModel {
            coord: candidate.coord,
            prop_index: prop_idx,
            required: false,
        });
        occupied.insert(candidate.coord);
    }

    Ok((models, omissions))
}

fn model_class_rank(c: TileClass) -> u8 {
    match c {
        TileClass::DeadEnd => 0,
        TileClass::RoomEdge => 1,
        TileClass::RoomInterior => 2,
        TileClass::CorridorCorner => 3,
        TileClass::ShaftEdge => 4,
        TileClass::Junction => 5,
        TileClass::CorridorStraight => 6,
        TileClass::RampEntry => 7,
        TileClass::ShaftFloor => 8,
        TileClass::Unclassified => 9,
    }
}

// ─── Distance computation ───────────────────────────────────────────────────

fn compute_distances(source: GridCoord, movement: &MovementGraph) -> BTreeMap<MovementNode, u64> {
    let mut dists = BTreeMap::new();
    let source_node = MovementNode {
        layer: source.layer,
        x: source.x,
        y: source.y,
    };
    let source_idx = movement.nodes.iter().position(|n| *n == source_node);
    let Some(source_idx) = source_idx else {
        return dists;
    };

    let mut queue = std::collections::VecDeque::from([(source_idx, 0u64)]);
    dists.insert(source_node, 0);

    while let Some((idx, dist)) = queue.pop_front() {
        if let Some(neighbors) = movement.adjacency.get(&idx) {
            for (next_idx, _) in neighbors {
                let next_node = movement.nodes[*next_idx];
                if !dists.contains_key(&next_node) {
                    let nd = dist.saturating_add(1);
                    dists.insert(next_node, nd);
                    queue.push_back((*next_idx, nd));
                }
            }
        }
    }
    dists
}

// ─── Relocation support ─────────────────────────────────────────────────────

/// Relocate a marker to an equivalent valid candidate within the same region
/// and class. Returns the new coordinate if a legal candidate exists.
pub(super) fn relocate_marker(
    coord: GridCoord,
    _region: RegionId,
    class: TileClass,
    _level: &ParsedLevel,
    tile_classes: &BTreeMap<GridCoord, TileClass>,
    occupied: &BTreeSet<GridCoord>,
    _config: &NormalizedGeneratorConfig,
) -> Option<GridCoord> {
    // Collect alternate cells with same class and region.
    let mut alternates: Vec<(GridCoord, u32)> = Vec::new();
    for (candidate, &c) in tile_classes {
        if c != class || *candidate == coord || occupied.contains(candidate) {
            continue;
        }
        if candidate.layer != coord.layer {
            continue;
        }
        let dist = (candidate.x as i32 - coord.x as i32).unsigned_abs()
            + (candidate.y as i32 - coord.y as i32).unsigned_abs();
        alternates.push((*candidate, dist));
    }

    alternates.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| a.0.y.cmp(&b.0.y))
            .then_with(|| a.0.x.cmp(&b.0.x))
    });
    alternates.first().map(|a| a.0)
}

// ─── Validation ─────────────────────────────────────────────────────────────

/// Validate marker placement: spawn uniqueness, reachability, co-location, counts.
pub(super) fn validate_markers(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    movement: &MovementGraph,
    marker_placement: &MarkerPlacement,
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<GeneratorError>, GeneratorError> {
    let mut errors = Vec::new();

    // Spawn must be on walkable floor.
    let spawn_tile = level.tile_at_3d(
        marker_placement.spawn.layer as usize,
        marker_placement.spawn.x as usize,
        marker_placement.spawn.y as usize,
    );
    if spawn_tile != Tile::Floor {
        errors.push(marker_error(
            "spawn",
            format!("spawn_not_floor tile={spawn_tile:?}"),
        ));
    }

    // Spawn must be reachable.
    let spawn_node = MovementNode {
        layer: marker_placement.spawn.layer,
        x: marker_placement.spawn.x,
        y: marker_placement.spawn.y,
    };
    if !movement.nodes.contains(&spawn_node) {
        errors.push(marker_error("spawn", "spawn_not_reachable".into()));
    }

    // Light count ≤ 16.
    if marker_placement.lights.len() > 16 {
        errors.push(marker_error(
            "light",
            format!(
                "light_count_exceeds_max count={}",
                marker_placement.lights.len()
            ),
        ));
    }

    // Verify 7/2/1 preset distribution for emitted lights.
    for (idx, light) in marker_placement.lights.iter().enumerate() {
        let expected = light_preset_for_marker_index(idx);
        if light.preset != expected {
            errors.push(marker_error(
                "light",
                format!(
                    "light_{}_preset_mismatch expected={expected:?} actual={:?}",
                    idx, light.preset
                ),
            ));
        }
    }

    // No co-located markers.
    let mut placed = BTreeSet::new();
    for light in &marker_placement.lights {
        if !placed.insert(light.coord) {
            errors.push(marker_error(
                "light",
                format!("duplicate_light {}", light.coord),
            ));
        }
    }
    for model in &marker_placement.models {
        if !placed.insert(model.coord) {
            errors.push(marker_error(
                "model",
                format!("duplicate_or_co_located_model {}", model.coord),
            ));
        }
    }

    // No light/model on spawn.
    if placed.contains(&marker_placement.spawn) {
        errors.push(marker_error("co_location", "marker_on_spawn".into()));
    }

    // All lights must be on Floor tiles.
    for light in &marker_placement.lights {
        let tile = level.tile_at_3d(
            light.coord.layer as usize,
            light.coord.x as usize,
            light.coord.y as usize,
        );
        if tile != Tile::Floor {
            errors.push(marker_error(
                "light",
                format!("light_not_on_floor {} tile={tile:?}", light.coord),
            ));
        }
    }

    // All models must be on Floor tiles.
    for model in &marker_placement.models {
        let tile = level.tile_at_3d(
            model.coord.layer as usize,
            model.coord.x as usize,
            model.coord.y as usize,
        );
        if tile != Tile::Floor {
            errors.push(marker_error(
                "model",
                format!("model_not_on_floor {} tile={tile:?}", model.coord),
            ));
        }
    }

    Ok(errors)
}

fn marker_error(kind: &str, detail: String) -> GeneratorError {
    GeneratorError::IrInvariant {
        stage: ErrorStage::Ir,
        detail: format!("[marker.{kind}] {detail}"),
    }
}

// ─── Combined placement ─────────────────────────────────────────────────────

/// Run the complete marker placement pipeline.
pub(super) fn place_all_markers(
    level: &ParsedLevel,
    topology: &IntendedTopology,
    movement: &MovementGraph,
    envelopes: &[Option<PropPlacementEnvelope>],
    config: &NormalizedGeneratorConfig,
) -> Result<MarkerPlacement, GeneratorError> {
    let tile_classes = classify_tiles(level, topology, movement, config)?;
    let spawn = place_spawn(level, topology, movement, config)?;
    let (lights, light_omissions) =
        place_lights(level, topology, movement, &tile_classes, spawn, config)?;
    let (models, model_omissions) = place_models(
        level,
        topology,
        movement,
        &tile_classes,
        spawn,
        &lights,
        envelopes,
        config,
    )?;

    let mut omissions = light_omissions;
    omissions.extend(model_omissions);

    Ok(MarkerPlacement {
        spawn,
        lights,
        models,
        omissions,
        tile_classes,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::GeneratorConfig;
    use super::super::validation::{reconstruct_movement_graph, validate_structural};
    use super::*;
    use crate::content::PropPlacementEnvelope;

    fn tiny_config() -> NormalizedGeneratorConfig {
        GeneratorConfig::custom(64, 64, 2).normalize().unwrap()
    }

    /// Build a ParsedLevel programmatically to avoid parse_level SpawnCardinality requirement.
    fn make_level(width: usize, height: usize, tiles: Vec<Tile>) -> ParsedLevel {
        assert_eq!(tiles.len(), width * height);
        ParsedLevel {
            width,
            height,
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

    fn tiny_level() -> ParsedLevel {
        let w = 64usize;
        let h = 64usize;
        // Fill everything with Void, then make a small 6x6 floor region with borders.
        let mut tiles = vec![Tile::Void; w * h];
        // Wall border + floor interior in top-left corner.
        for y in 0..8 {
            for x in 0..8 {
                if x == 0 || y == 0 || x == 7 || y == 7 {
                    tiles[y * w + x] = Tile::Wall;
                } else {
                    tiles[y * w + x] = Tile::Floor;
                }
            }
        }
        let layer2 = vec![Tile::Void; w * h];
        ParsedLevel {
            width: w,
            height: h,
            layers: vec![tiles, layer2],
            spawn: TileCoord {
                layer: 0,
                x: 1,
                y: 1,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        }
    }

    fn tiny_topology() -> IntendedTopology {
        use super::super::ir::*;
        let mut alloc = IdAllocator::new();
        let r0 = alloc.next_region().unwrap();
        IntendedTopology {
            regions: vec![PlacedRegion {
                id: r0,
                role: RegionRole::Spawn,
                variant_index: 0,
                layer: 0,
                footprint: (1, 1, 6, 6),
                sockets: vec![],
                transitions: vec![],
                marker_variant_indices: vec![],
            }],
            edges: vec![],
            transitions: vec![],
            route_distance: 0,
            per_layer_cycles: vec![0; 2],
            max_branch_depth: 0,
            dead_end_count: 0,
            articulation_count: 0,
            crossing_count: 0,
            config: tiny_config(),
        }
    }

    #[test]
    fn spawn_placed_in_spawn_region_on_floor() {
        let config = tiny_config();
        let level = tiny_level();
        let topology = tiny_topology();
        let (movement, _inferred) = reconstruct_movement_graph(&level, &topology).unwrap();
        let spawn = place_spawn(&level, &topology, &movement, &config).unwrap();
        let tile = level.tile_at_3d(spawn.layer as usize, spawn.x as usize, spawn.y as usize);
        assert_eq!(tile, Tile::Floor);
        // Spawn must be inside region footprint [1,1]-[4,4)
        assert!(spawn.x >= 1 && spawn.x < 4 && spawn.y >= 1 && spawn.y < 4);
    }

    #[test]
    fn light_preset_7_2_1_distribution() {
        // Test the content-level function.
        let presets: Vec<LightPresetId> =
            (0..10).map(|i| light_preset_for_marker_index(i)).collect();
        let warm = presets
            .iter()
            .filter(|p| matches!(p, LightPresetId::Warm))
            .count();
        let cool = presets
            .iter()
            .filter(|p| matches!(p, LightPresetId::Cool))
            .count();
        let accent = presets
            .iter()
            .filter(|p| matches!(p, LightPresetId::Accent))
            .count();
        assert_eq!(warm, 7);
        assert_eq!(cool, 2);
        assert_eq!(accent, 1);
    }

    #[test]
    fn light_preset_sequence_stable_to_16() {
        let presets: Vec<LightPresetId> =
            (0..16).map(|i| light_preset_for_marker_index(i)).collect();
        // 0-6 warm, 7-8 cool, 9 accent, 10-16 warm (mod 10 wraps)
        for (i, &p) in presets.iter().enumerate() {
            match i % 10 {
                0..=6 => assert!(matches!(p, LightPresetId::Warm), "idx {i} should be warm"),
                7 | 8 => assert!(matches!(p, LightPresetId::Cool), "idx {i} should be cool"),
                9 => assert!(
                    matches!(p, LightPresetId::Accent),
                    "idx {i} should be accent"
                ),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn required_light_intent_is_per_role_not_per_floor_cell() {
        let config = tiny_config();
        let level = tiny_level();
        let mut topology = tiny_topology();
        topology.regions[0].role = RegionRole::RequiredRoute;
        let (movement, _inferred) = reconstruct_movement_graph(&level, &topology).unwrap();
        let spawn = GridCoord::new(0, 1, 1, config.width(), config.height(), config.layers().2)
            .unwrap();

        let (lights, _omissions) = place_lights(
            &level,
            &topology,
            &movement,
            &BTreeMap::new(),
            spawn,
            &config,
        )
        .expect("many floor cells in one required role must fit the light budget");

        assert_eq!(lights.iter().filter(|light| light.required).count(), 1);
        assert_eq!(lights.len(), usize::try_from(config.max_lights()).unwrap());
        assert!(lights.windows(2).all(|pair| {
            (pair[0].coord.layer, pair[0].coord.y, pair[0].coord.x)
                < (pair[1].coord.layer, pair[1].coord.y, pair[1].coord.x)
        }));
        for (index, light) in lights.iter().enumerate() {
            assert_eq!(light.preset, light_preset_for_marker_index(index));
        }
    }

    #[test]
    fn prop_envelope_world_half_extents_identity() {
        let env = PropPlacementEnvelope {
            half_extents_local: [0.5, 0.6, 0.4],
            scale: [1.0, 1.0, 1.0],
            yaw_degrees: 0.0,
        };
        let world = env.world_half_extents();
        assert!((world[0] - 0.5).abs() < 1e-6);
        assert!((world[1] - 0.6).abs() < 1e-6);
        assert!((world[2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn prop_envelope_world_half_extents_rotated_90() {
        let env = PropPlacementEnvelope {
            half_extents_local: [0.5, 0.6, 0.3],
            scale: [1.0, 1.0, 1.0],
            yaw_degrees: 90.0,
        };
        let world = env.world_half_extents();
        // cos(90)=0, sin(90)=1. So hx = sz*lz, hz = sx*lx.
        assert!((world[0] - 0.3).abs() < 1e-6); // sz * lz
        assert!((world[2] - 0.5).abs() < 1e-6); // sx * lx
    }

    #[test]
    fn prop_envelope_world_half_extents_scaled() {
        let env = PropPlacementEnvelope {
            half_extents_local: [0.5, 0.6, 0.4],
            scale: [2.0, 3.0, 1.5],
            yaw_degrees: 45.0,
        };
        let world = env.world_half_extents();
        let cos = 45f32.to_radians().cos().abs();
        let sin = 45f32.to_radians().sin().abs();
        let expected_hx = cos * 2.0 * 0.5 + sin * 1.5 * 0.4;
        let expected_hz = sin * 2.0 * 0.5 + cos * 1.5 * 0.4;
        assert!((world[0] - expected_hx).abs() < 1e-5);
        assert!((world[1] - 3.0 * 0.6).abs() < 1e-5);
        assert!((world[2] - expected_hz).abs() < 1e-5);
    }

    #[test]
    fn prop_envelope_finite_check() {
        // Non-finite half extents should return None.
        let spec = crate::content::PropSpec {
            id: "test".into(),
            enabled: true,
            path: Default::default(),
            prefer_unlit_fallback: false,
            scale: [1.0, 1.0, 1.0],
            yaw_degrees: 0.0,
            y_offset: 0.0,
            placement_half_extents: [f32::NAN, 0.5, 0.3],
        };
        assert!(spec.placement_envelope().is_none());

        let spec = crate::content::PropSpec {
            id: "test".into(),
            enabled: true,
            path: Default::default(),
            prefer_unlit_fallback: false,
            scale: [1.0, 1.0, 1.0],
            yaw_degrees: 0.0,
            y_offset: 0.0,
            placement_half_extents: [0.0, 0.5, 0.3],
        };
        assert!(spec.placement_envelope().is_none());
    }

    #[test]
    fn prop_fits_rejects_wall_intersection() {
        let config = tiny_config();
        let w = 64usize;
        let h = 64usize;
        let mut tiles = vec![Tile::Void; w * h];
        // Wall box around (1,1)-(3,3) with floor at (2,2).
        for y in 1..4 {
            for x in 1..4 {
                if x == 1 || y == 1 || x == 3 || y == 3 {
                    tiles[y * w + x] = Tile::Wall;
                } else {
                    tiles[y * w + x] = Tile::Floor;
                }
            }
        }
        let layer2 = vec![Tile::Void; w * h];
        let level = ParsedLevel {
            width: w,
            height: h,
            layers: vec![tiles, layer2],
            spawn: TileCoord {
                layer: 0,
                x: 1,
                y: 1,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        };
        let occupied = BTreeSet::new();
        let half_extents = [0.4, 0.4, 0.4];
        // Floor tile (2,2) should fit a small prop.
        assert!(prop_fits_at(
            0,
            2,
            2,
            half_extents,
            &level,
            &occupied,
            &config
        ));
        // Wall tile (1,1) should not fit.
        assert!(!prop_fits_at(
            0,
            1,
            1,
            half_extents,
            &level,
            &occupied,
            &config
        ));
        // Floor tile with large envelope that hits walls should not fit.
        let large_half = [1.5, 0.4, 1.5];
        assert!(!prop_fits_at(
            0, 2, 2, large_half, &level, &occupied, &config
        ));
    }

    #[test]
    fn tile_classification_detects_dead_end() {
        let config = tiny_config();
        let w = 64usize;
        let h = 64usize;
        let mut tiles = vec![Tile::Void; w * h];
        // A simple corridor: floor at (5,1)-(5,5) and (4,1).
        for y in 1..6 {
            tiles[y * w + 5] = Tile::Floor;
        }
        tiles[1 * w + 4] = Tile::Floor;
        let layer2 = vec![Tile::Void; w * h];
        let level = ParsedLevel {
            width: w,
            height: h,
            layers: vec![tiles, layer2],
            spawn: TileCoord {
                layer: 0,
                x: 4,
                y: 1,
            },
            model_markers: Vec::new(),
            light_markers: Vec::new(),
        };
        let mut topology = tiny_topology();
        topology.regions[0].footprint = (4, 1, 2, 2);
        let (movement, _inferred) = reconstruct_movement_graph(&level, &topology).unwrap();
        let classes = classify_tiles(&level, &topology, &movement, &config).unwrap();
        // Should have classified tiles.
        assert!(!classes.is_empty());
    }

    #[test]
    fn place_all_markers_produces_valid_spawn_and_lights() {
        let config = tiny_config();
        let level = tiny_level();
        let topology = tiny_topology();
        let (movement, _inferred) = reconstruct_movement_graph(&level, &topology).unwrap();

        let envelopes: Vec<Option<PropPlacementEnvelope>> = vec![
            Some(PropPlacementEnvelope {
                half_extents_local: [0.3, 0.5, 0.3],
                scale: [1.0, 1.0, 1.0],
                yaw_degrees: 0.0,
            }),
            Some(PropPlacementEnvelope {
                half_extents_local: [0.4, 0.4, 0.4],
                scale: [1.0, 1.0, 1.0],
                yaw_degrees: 0.0,
            }),
        ];

        let placement =
            place_all_markers(&level, &topology, &movement, &envelopes, &config).unwrap();

        // Spawn must be placed.
        let spawn_tile = level.tile_at_3d(
            placement.spawn.layer as usize,
            placement.spawn.x as usize,
            placement.spawn.y as usize,
        );
        assert_eq!(spawn_tile, Tile::Floor);

        // Lights must not exceed 16.
        assert!(placement.lights.len() <= 16);

        // No co-location between spawn and lights.
        for light in &placement.lights {
            assert_ne!(light.coord, placement.spawn);
        }
    }
}
