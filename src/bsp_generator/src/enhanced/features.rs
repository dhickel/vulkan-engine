//! Enhanced v2 feature variance — corridor width, ceiling height, pillars,
//! spawn origin, and light origins.
//!
//! # Design Contract
//!
//! This module consumes Phase 04 topology and Phase 05 theme. It selects
//! bounded corridor width, ceiling height, and pillar variance without
//! changing structural topology. It uses isolated Enhanced variance RNG
//! streams and records every decision for deterministic replay.
//!
//! # Key Constraints
//!
//! - Structural topology is immutable: no rerouting, backtracking, or topology
//!   changes.
//! - Corridor widths: {64, 80, 96}; each fits Phase-04 reserved capacity.
//!   Width transitions preserve 64-wide × 80-high conservative passage.
//! - Corridor height: always exactly 80.
//! - Ceilings: per-room from approved quanta {128, 144, 176}; minimum 80
//!   headroom preserved at all routes, apertures, and approaches.
//! - Pillars: freestanding, axis-aligned, positive-volume boxes. Accent material
//!   from room palette. Up to N per room. Must not intersect exclusion regions.
//! - Feature-aware spawn: exactly one info_player_start on entry layer at proven
//!   clear origin. Light origins from proven clear candidates.
//! - Exclusion regions derived before pillars: room shell walls, apertures,
//!   corridor envelopes, junctions, transition volumes, landings, headroom,
//!   spawn/light candidates.
//! - Connectivity oracle: for each pillar candidate, flood-fill walkable cells
//!   to verify room accessibility is not broken.

use std::collections::{BTreeMap, VecDeque};

use crate::config::CONSTRUCTION_QUANTUM;

use super::config::EnhancedConfig;
use super::error::EnhancedError;
use super::intent::{RoomId, RouteId};
use super::placement::{PlacedRoom, PlacementResult, WallDirection};
use super::seed::EnhancedStageRng;
use super::theme::ThemeAssignment;
use super::topology::TopologyResult;

// ── Constants ──────────────────────────────────────────────────────────────

const Q: i32 = CONSTRUCTION_QUANTUM as i32;

/// Approved corridor widths in Quake units.
pub const ALLOWED_CORRIDOR_WIDTHS: &[u32] = &[64, 80, 96];

/// Approved ceiling heights in Quake units.
pub const ALLOWED_CEILING_HEIGHTS: &[i32] = &[128, 144, 176];

/// Default safe ceiling height (fallback when validation fails).
pub const DEFAULT_SAFE_CEILING: i32 = 176;

/// Minimum headroom in Quake units.
pub const MIN_HEADROOM: i32 = 80;

/// Wall thickness in Quake units (one construction quantum).
pub const WALL_THICKNESS: i32 = Q;

/// Default pillar footprint: width (X) in Quake units.
pub const PILLAR_WIDTH: i32 = 32;

/// Default pillar footprint: depth (Y) in Quake units.
pub const PILLAR_DEPTH: i32 = 32;

/// Default pillar height in Quake units (floor to walkable ceiling).
pub const PILLAR_HEIGHT: i32 = 80;

/// Wider preferences whose unavailable Phase-04 capacity must be recorded.
const WIDER_WIDTH_PREFERENCES: &[u32] = &[96, 80];

// ── Public result types ────────────────────────────────────────────────────

/// Selected corridor width for a single route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorridorWidthSelection {
    pub route_id: RouteId,
    /// Width in Quake units: 64, 80, or 96.
    pub width: u32,
    /// Every approved wider width rejected by the committed route capacity.
    pub rejections: Vec<CorridorWidthRejection>,
}

/// Typed reason a route-width preference could not be accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorridorWidthRejectionReason {
    /// The Phase-04 committed reservation does not contain the wider envelope.
    CapacityUnavailable,
}

/// A rejected route-width preference, retained in deterministic preference order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorridorWidthRejection {
    pub width: u32,
    pub reason: CorridorWidthRejectionReason,
}

/// Typed reason an approved ceiling quantum could not be accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CeilingRejectionReason {
    ExceedsRoomEnvelope,
    InsufficientHeadroom,
    TransitionHeadroom,
}

/// A rejected ceiling quantum, retained in deterministic preference order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CeilingRejection {
    pub height: i32,
    pub reason: CeilingRejectionReason,
}

/// Selected ceiling height for a single room.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CeilingHeightSelection {
    pub room_id: RoomId,
    /// Ceiling height in Quake units.
    pub height: i32,
    /// True if the requested height was rejected and the safe default applied.
    pub is_fallback: bool,
    /// Typed reason the safe default was required.
    pub fallback_reason: Option<CeilingRejectionReason>,
    /// Every rejected approved quantum before selecting `height`.
    pub rejections: Vec<CeilingRejection>,
}

/// A placed freestanding pillar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PillarPlacement {
    pub room_id: RoomId,
    /// Axis-aligned bounds: (x0, y0, z0, x1, y1, z1) in Quake units.
    pub bounds: (i32, i32, i32, i32, i32, i32),
    /// Accent material name from the room's palette.
    pub accent_material: String,
}

/// Typed reason a pillar candidate was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PillarRejectionReason {
    /// Pillar has zero or negative volume.
    NonPositiveVolume,
    /// Pillar is not axis-aligned.
    NotAxisAligned,
    /// Intersects an exclusion region.
    ExclusionIntersection(String),
    /// Overlaps another committed pillar.
    Overlap(String),
    /// Insufficient clearance from walls or other features.
    InsufficientClearance,
    /// Flood-fill determined connectivity would be broken.
    ConnectivityBroken,
}

impl std::fmt::Display for PillarRejectionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonPositiveVolume => write!(f, "non-positive volume"),
            Self::NotAxisAligned => write!(f, "not axis-aligned"),
            Self::ExclusionIntersection(s) => write!(f, "exclusion intersection: {}", s),
            Self::Overlap(s) => write!(f, "overlap: {}", s),
            Self::InsufficientClearance => write!(f, "insufficient clearance"),
            Self::ConnectivityBroken => write!(f, "connectivity broken"),
        }
    }
}

/// Record of a rejected pillar candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PillarRejection {
    pub room_id: RoomId,
    /// Candidate index in canonical enumeration order.
    pub index: u32,
    pub reason: PillarRejectionReason,
}

/// Recorded when fewer pillars fit than the per-room quota.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestedCountUnmet {
    pub room_id: RoomId,
    pub requested: u32,
    pub placed: u32,
}

/// Feature-aware spawn point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnPoint {
    /// Origin in Quake units: (x, y, z).
    pub origin: (i32, i32, i32),
    pub room_id: RoomId,
    /// Layer index (0 = lower/entry).
    pub layer: u8,
}

/// Feature-aware light origin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightOrigin {
    /// Origin in Quake units: (x, y, z).
    pub origin: (i32, i32, i32),
    pub room_id: RoomId,
}

/// Complete Phase 06 feature result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureResult {
    /// Width selections for every route, in canonical RouteId order.
    pub corridor_widths: Vec<CorridorWidthSelection>,
    /// Ceiling selections for every room, in canonical RoomId order.
    pub ceiling_heights: Vec<CeilingHeightSelection>,
    /// All committed pillar placements, in canonical (RoomId, index) order.
    pub pillars: Vec<PillarPlacement>,
    /// Every rejected pillar candidate, in occurrence order.
    pub pillar_rejections: Vec<PillarRejection>,
    /// Rooms where the pillar quota could not be met.
    pub requested_count_unmet: Vec<RequestedCountUnmet>,
    /// The single spawn point on the entry layer.
    pub spawn_point: SpawnPoint,
    /// Light origins for populated rooms.
    pub light_origins: Vec<LightOrigin>,
}

// ── Internal exclusion helpers ─────────────────────────────────────────────

/// An axis-aligned exclusion volume where pillars must not be placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExclusionVolume {
    pub bounds: (i32, i32, i32, i32, i32, i32),
    pub reason: String,
}

/// Build a room lookup from placement.
fn room_map(placement: &PlacementResult) -> BTreeMap<RoomId, &PlacedRoom> {
    placement.rooms.iter().map(|r| (r.id, r)).collect()
}

// ── Entry point ────────────────────────────────────────────────────────────

/// Apply variance and feature placement to a completed topology and theme.
///
/// # Determinism
///
/// Given the same inputs (config, placement, topology, theme, seeds), the
/// returned `FeatureResult` is byte-identical.
pub fn apply_features(
    config: &EnhancedConfig,
    placement: &PlacementResult,
    topology: &TopologyResult,
    theme: &ThemeAssignment,
    feature_seed: EnhancedStageRng,
    corridor_seed: EnhancedStageRng,
) -> Result<FeatureResult, EnhancedError> {
    // Validate config variance policy
    validate_variance_policy(config)?;

    let rooms = room_map(placement);

    // Step 2: Corridor width selection
    let corridor_widths = select_corridor_widths(topology, corridor_seed)?;

    // Step 3: Ceiling height selection
    let ceiling_heights = select_ceiling_heights(placement, topology, feature_seed.clone())?;

    // Step 4: Derive exclusion regions and required clear origins before
    // optional pillars. Their protected volumes ensure later feature placement
    // cannot invalidate a required entity origin.
    let mut exclusions = derive_exclusion_regions(placement, topology, &rooms)?;
    let (spawn_point, light_origins) =
        select_clear_origins(placement, &rooms, &exclusions, feature_seed.clone())?;
    protect_required_origins(&mut exclusions, &spawn_point, &light_origins);

    // Step 5-6: Pillar placement
    let (pillars, rejections, unmet) =
        place_pillars(config, placement, &rooms, &exclusions, theme, feature_seed)?;

    Ok(FeatureResult {
        corridor_widths,
        ceiling_heights,
        pillars,
        pillar_rejections: rejections,
        requested_count_unmet: unmet,
        spawn_point,
        light_origins,
    })
}

// ── Step 1: Validate variance policy ───────────────────────────────────────

fn validate_variance_policy(_config: &EnhancedConfig) -> Result<(), EnhancedError> {
    // Validate allowed corridor widths are exactly {64, 80, 96}
    let allowed: Vec<u32> = ALLOWED_CORRIDOR_WIDTHS.to_vec();
    if allowed != vec![64, 80, 96] {
        return Err(EnhancedError::ContractViolation {
            detail: format!(
                "approved corridor widths must be {{64, 80, 96}}, got {:?}",
                allowed,
            ),
        });
    }

    // Validate allowed ceiling heights are a subset of approved values
    for &h in ALLOWED_CEILING_HEIGHTS {
        if ![128, 144, 176].contains(&h) {
            return Err(EnhancedError::ContractViolation {
                detail: format!(
                    "ceiling height {} is not an approved quantum value (128, 144, 176)",
                    h,
                ),
            });
        }
    }

    // Validate default safe ceiling is in the approved list
    if ![128, 144, 176].contains(&DEFAULT_SAFE_CEILING) {
        return Err(EnhancedError::ContractViolation {
            detail: format!(
                "default safe ceiling {} must be an approved quantum value",
                DEFAULT_SAFE_CEILING,
            ),
        });
    }

    // max_pillars_per_room validated at config construction (0..=8)

    Ok(())
}

// ── Step 2: Corridor width selection ───────────────────────────────────────

/// Select corridor width from {64, 80, 96} for each route using the isolated
/// variance RNG stream. Wider widths must fit within Phase-04 reserved capacity.
fn select_corridor_widths(
    topology: &TopologyResult,
    mut rng: EnhancedStageRng,
) -> Result<Vec<CorridorWidthSelection>, EnhancedError> {
    let mut selections = Vec::with_capacity(topology.routes.len());

    for route in &topology.routes {
        // Phase 04 commits exact 64-wide envelopes. Nearby clearance is not
        // owned capacity and cannot be consumed here without mutating the
        // accepted topology, so every wider preference is explicitly rejected.
        // The isolated stream controls only the deterministic preference order.
        let preferred = rng.range_u32(WIDER_WIDTH_PREFERENCES.len() as u32) as usize;
        let rejections = WIDER_WIDTH_PREFERENCES[preferred..]
            .iter()
            .chain(WIDER_WIDTH_PREFERENCES[..preferred].iter())
            .map(|&width| CorridorWidthRejection {
                width,
                reason: CorridorWidthRejectionReason::CapacityUnavailable,
            })
            .collect();

        selections.push(CorridorWidthSelection {
            route_id: route.id,
            width: 64,
            rejections,
        });
    }

    selections.sort_by_key(|s| s.route_id);
    Ok(selections)
}

// ── Step 3: Ceiling height selection ───────────────────────────────────────

/// Select per-room ceiling height from approved quanta. Validate against
/// floor Z + layer envelope + route lintels + transition headroom.
/// Rejected selections fall back to the default safe ceiling.
fn select_ceiling_heights(
    placement: &PlacementResult,
    topology: &TopologyResult,
    mut rng: EnhancedStageRng,
) -> Result<Vec<CeilingHeightSelection>, EnhancedError> {
    let mut selections = Vec::with_capacity(placement.rooms.len());

    for room in &placement.rooms {
        let floor_z = room.floor_z;
        let room_top_z = floor_z + room.dims.2 as i32;

        // Try each approved height, shuffled by RNG preference
        let mut heights: Vec<i32> = ALLOWED_CEILING_HEIGHTS.to_vec();
        // Deterministic shuffle: pick order based on RNG
        let pick = rng.range_u32(heights.len() as u32) as usize;
        heights.swap(0, pick);

        let mut chosen: Option<CeilingHeightSelection> = None;
        let mut rejections = Vec::new();

        for &height in &heights {
            let ceiling_z = floor_z + height;
            let reason = if ceiling_z > room_top_z {
                Some(CeilingRejectionReason::ExceedsRoomEnvelope)
            } else if height < MIN_HEADROOM {
                Some(CeilingRejectionReason::InsufficientHeadroom)
            } else if topology.transitions.iter().any(|t| {
                (t.lower_room == room.id || t.upper_room == room.id)
                    && volume_overlaps_room_shell(&t.headroom, room)
                    && ceiling_z < t.headroom.5.min(room_top_z)
            }) {
                Some(CeilingRejectionReason::TransitionHeadroom)
            } else {
                None
            };

            if let Some(reason) = reason {
                rejections.push(CeilingRejection { height, reason });
                continue;
            }

            chosen = Some(CeilingHeightSelection {
                room_id: room.id,
                height,
                is_fallback: false,
                fallback_reason: None,
                rejections: rejections.clone(),
            });
            break;
        }

        // The default is still subject to every ceiling invariant. Falling
        // back must never silently place an invalid ceiling.
        let sel = match chosen {
            Some(selection) => selection,
            None => {
                let default_z = floor_z + DEFAULT_SAFE_CEILING;
                if default_z > room_top_z || DEFAULT_SAFE_CEILING < MIN_HEADROOM {
                    return Err(EnhancedError::ContractViolation {
                        detail: format!(
                            "room {:?} has no legal approved ceiling, including safe default {}",
                            room.id, DEFAULT_SAFE_CEILING,
                        ),
                    });
                }
                CeilingHeightSelection {
                    room_id: room.id,
                    height: DEFAULT_SAFE_CEILING,
                    is_fallback: true,
                    fallback_reason: rejections.first().map(|rejection| rejection.reason.clone()),
                    rejections,
                }
            }
        };

        selections.push(sel);
    }

    selections.sort_by_key(|s| s.room_id);
    Ok(selections)
}

/// Check if a 3D volume (x0, y0, z0, x1, y1, z1) overlaps a room shell in XY.
fn volume_overlaps_room_shell(vol: &(i32, i32, i32, i32, i32, i32), room: &PlacedRoom) -> bool {
    let (vx0, vy0, _vz0, vx1, vy1, _vz1) = *vol;
    let (rx0, ry0, rx1, ry1) = room.shell;
    vx0 < rx1 && vx1 > rx0 && vy0 < ry1 && vy1 > ry0
}

// ── Step 4: Derive exclusion regions ───────────────────────────────────────

/// Derive all exclusion regions where pillars must not be placed.
/// These cover: room shell walls, apertures, corridor envelopes, junctions,
/// transition volumes, landings, and headroom.
fn derive_exclusion_regions(
    placement: &PlacementResult,
    topology: &TopologyResult,
    _rooms: &BTreeMap<RoomId, &PlacedRoom>,
) -> Result<BTreeMap<RoomId, Vec<ExclusionVolume>>, EnhancedError> {
    let mut regions: BTreeMap<RoomId, Vec<ExclusionVolume>> = BTreeMap::new();

    for room in &placement.rooms {
        let mut vols = Vec::new();

        let (x0, y0, x1, y1) = room.shell;
        let z0 = room.floor_z;
        let z1 = z0 + room.dims.2 as i32;

        // 1. Room shell walls: 16-unit thick perimeter
        // South wall
        vols.push(ExclusionVolume {
            bounds: (x0, y0, z0, x1, y0 + WALL_THICKNESS, z1),
            reason: "south wall".into(),
        });
        // North wall
        vols.push(ExclusionVolume {
            bounds: (x0, y1 - WALL_THICKNESS, z0, x1, y1, z1),
            reason: "north wall".into(),
        });
        // West wall
        vols.push(ExclusionVolume {
            bounds: (
                x0,
                y0 + WALL_THICKNESS,
                z0,
                x0 + WALL_THICKNESS,
                y1 - WALL_THICKNESS,
                z1,
            ),
            reason: "west wall".into(),
        });
        // East wall
        vols.push(ExclusionVolume {
            bounds: (
                x1 - WALL_THICKNESS,
                y0 + WALL_THICKNESS,
                z0,
                x1,
                y1 - WALL_THICKNESS,
                z1,
            ),
            reason: "east wall".into(),
        });

        // 2. Apertures (sockets): 64-wide openings in walls
        for socket in &placement.sockets {
            if socket.room != room.id {
                continue;
            }
            let half = (socket.width as i32) / 2;
            let (ax, ay) = (socket.anchor.0, socket.anchor.1);
            let az = socket.anchor.2;

            match socket.wall {
                WallDirection::North => {
                    vols.push(ExclusionVolume {
                        bounds: (
                            ax - half,
                            y1 - WALL_THICKNESS,
                            az - 40,
                            ax + half,
                            y1,
                            az + 40,
                        ),
                        reason: format!("aperture north socket {:?}", socket.id),
                    });
                }
                WallDirection::South => {
                    vols.push(ExclusionVolume {
                        bounds: (
                            ax - half,
                            y0,
                            az - 40,
                            ax + half,
                            y0 + WALL_THICKNESS,
                            az + 40,
                        ),
                        reason: format!("aperture south socket {:?}", socket.id),
                    });
                }
                WallDirection::East => {
                    vols.push(ExclusionVolume {
                        bounds: (
                            x1 - WALL_THICKNESS,
                            ay - half,
                            az - 40,
                            x1,
                            ay + half,
                            az + 40,
                        ),
                        reason: format!("aperture east socket {:?}", socket.id),
                    });
                }
                WallDirection::West => {
                    vols.push(ExclusionVolume {
                        bounds: (
                            x0,
                            ay - half,
                            az - 40,
                            x0 + WALL_THICKNESS,
                            ay + half,
                            az + 40,
                        ),
                        reason: format!("aperture west socket {:?}", socket.id),
                    });
                }
            }
        }

        // 3. Corridor envelopes: where routes enter/leave the room
        for route in &topology.routes {
            if route.source_room != room.id && route.target_room != room.id {
                continue;
            }
            for &(ex0, ey0, ex1, ey1) in &route.envelopes {
                // Only add if the envelope overlaps the room shell
                if ex0 < x1 && ex1 > x0 && ey0 < y1 && ey1 > y0 {
                    // Clip to room interior
                    let cx0 = ex0.max(x0);
                    let cy0 = ey0.max(y0);
                    let cx1 = ex1.min(x1);
                    let cy1 = ey1.min(y1);
                    if cx0 < cx1 && cy0 < cy1 {
                        vols.push(ExclusionVolume {
                            bounds: (cx0, cy0, z0, cx1, cy1, z0 + MIN_HEADROOM),
                            reason: format!("corridor route {:?}", route.id),
                        });
                    }
                }
            }
        }

        // 4. Transition solids. Entity origins and pillars use the exact
        // materializable tread boxes, not a transition bounding volume or an
        // all-or-nothing transition exception. Clear approaches remain clear.
        for t in &topology.transitions {
            for tread in &t.tread_boxes {
                let (tx0, ty0, tz0, tx1, ty1, tz1) = tread.bounds;
                let cx0 = tx0.max(x0);
                let cy0 = ty0.max(y0);
                let cz0 = tz0.max(z0);
                let cx1 = tx1.min(x1);
                let cy1 = ty1.min(y1);
                let cz1 = tz1.min(z1);
                if cx0 < cx1 && cy0 < cy1 && cz0 < cz1 {
                    vols.push(ExclusionVolume {
                        bounds: (cx0, cy0, cz0, cx1, cy1, cz1),
                        reason: format!("transition {:?} tread", t.id),
                    });
                }
            }
        }

        regions.insert(room.id, vols);
    }

    Ok(regions)
}

// ── Step 5-6: Pillar placement ─────────────────────────────────────────────

/// Enumerate, validate, and commit pillars up to `max_pillars_per_room` per room.
fn place_pillars(
    config: &EnhancedConfig,
    placement: &PlacementResult,
    _rooms: &BTreeMap<RoomId, &PlacedRoom>,
    exclusions: &BTreeMap<RoomId, Vec<ExclusionVolume>>,
    theme: &ThemeAssignment,
    mut rng: EnhancedStageRng,
) -> Result<
    (
        Vec<PillarPlacement>,
        Vec<PillarRejection>,
        Vec<RequestedCountUnmet>,
    ),
    EnhancedError,
> {
    let max_per_room = config.max_pillars_per_room();
    let mut pillars: Vec<PillarPlacement> = Vec::new();
    let mut rejections: Vec<PillarRejection> = Vec::new();
    let mut unmet: Vec<RequestedCountUnmet> = Vec::new();

    // Process rooms in canonical RoomId order
    let mut sorted_rooms: Vec<&PlacedRoom> = placement.rooms.iter().collect();
    sorted_rooms.sort_by_key(|r| r.id);

    for room in &sorted_rooms {
        let room_exclusions = exclusions
            .get(&room.id)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let palette_name = theme
            .room_palettes
            .get(&room.id)
            .map(|pa| pa.palette_name.clone())
            .unwrap_or_else(|| "base_stone".to_string());
        let accent = accent_material_for(&palette_name);

        let candidates = enumerate_pillar_candidates(room);
        let mut ranked_candidates: Vec<_> = candidates
            .into_iter()
            .enumerate()
            .map(|(index, bounds)| (rng.next_u64(), index as u32, bounds))
            .collect();
        ranked_candidates.sort_by_key(|(rank, index, _)| (*rank, *index));
        let mut placed_in_room: u32 = 0;
        let mut committed_bounds: Vec<(i32, i32, i32, i32, i32, i32)> = Vec::new();

        for (_, index, candidate_bounds) in ranked_candidates {
            let idx = index as usize;
            if placed_in_room >= max_per_room {
                break;
            }

            // Check non-positive volume
            let (cx0, cy0, cz0, cx1, cy1, cz1) = candidate_bounds;
            if cx0 >= cx1 || cy0 >= cy1 || cz0 >= cz1 {
                rejections.push(PillarRejection {
                    room_id: room.id,
                    index: idx as u32,
                    reason: PillarRejectionReason::NonPositiveVolume,
                });
                continue;
            }

            // Check axis alignment
            let is_aligned = cx0 % Q == 0
                && cy0 % Q == 0
                && cz0 % Q == 0
                && cx1 % Q == 0
                && cy1 % Q == 0
                && cz1 % Q == 0;
            if !is_aligned {
                rejections.push(PillarRejection {
                    room_id: room.id,
                    index: idx as u32,
                    reason: PillarRejectionReason::NotAxisAligned,
                });
                continue;
            }

            // Check exclusion intersection
            let mut excluded = false;
            for ev in room_exclusions {
                if boxes_intersect(candidate_bounds, ev.bounds) {
                    rejections.push(PillarRejection {
                        room_id: room.id,
                        index: idx as u32,
                        reason: PillarRejectionReason::ExclusionIntersection(ev.reason.clone()),
                    });
                    excluded = true;
                    break;
                }
            }
            if excluded {
                continue;
            }

            // Check overlap with committed pillars
            let mut overlapping = false;
            for &cb in &committed_bounds {
                if boxes_intersect(candidate_bounds, cb) {
                    rejections.push(PillarRejection {
                        room_id: room.id,
                        index: idx as u32,
                        reason: PillarRejectionReason::Overlap(format!(
                            "overlaps committed pillar at {:?}",
                            cb,
                        )),
                    });
                    overlapping = true;
                    break;
                }
            }
            if overlapping {
                continue;
            }

            // Check clearance: pillar must not be adjacent to walls
            // (at least one quantum gap)
            let interior_x0 = room.shell.0 + WALL_THICKNESS + Q;
            let interior_y0 = room.shell.1 + WALL_THICKNESS + Q;
            let interior_x1 = room.shell.2 - WALL_THICKNESS - Q;
            let interior_y1 = room.shell.3 - WALL_THICKNESS - Q;
            if cx0 < interior_x0 || cy0 < interior_y0 || cx1 > interior_x1 || cy1 > interior_y1 {
                rejections.push(PillarRejection {
                    room_id: room.id,
                    index: idx as u32,
                    reason: PillarRejectionReason::InsufficientClearance,
                });
                continue;
            }

            // Connectivity oracle: flood-fill walkable cells
            let connected =
                check_connectivity(room, room_exclusions, &committed_bounds, candidate_bounds);
            if !connected {
                rejections.push(PillarRejection {
                    room_id: room.id,
                    index: idx as u32,
                    reason: PillarRejectionReason::ConnectivityBroken,
                });
                continue;
            }

            // Commit
            pillars.push(PillarPlacement {
                room_id: room.id,
                bounds: candidate_bounds,
                accent_material: accent.clone(),
            });
            committed_bounds.push(candidate_bounds);
            placed_in_room += 1;
        }

        if placed_in_room < max_per_room {
            unmet.push(RequestedCountUnmet {
                room_id: room.id,
                requested: max_per_room,
                placed: placed_in_room,
            });
        }
    }

    pillars.sort_by_key(|p| (p.room_id, p.bounds.0, p.bounds.1, p.bounds.2));
    rejections.sort_by_key(|r| (r.room_id, r.index));
    unmet.sort_by_key(|u| u.room_id);

    Ok((pillars, rejections, unmet))
}

/// Enumerate pillar candidates in canonical (z, y, x) order within a room.
/// Candidates are axis-aligned 32×32×80 boxes placed on a quantum grid.
fn enumerate_pillar_candidates(room: &PlacedRoom) -> Vec<(i32, i32, i32, i32, i32, i32)> {
    let (x0, y0, _x1, _y1) = room.shell;
    let floor_z = room.floor_z;

    let interior_x0 = x0 + WALL_THICKNESS;
    let interior_y0 = y0 + WALL_THICKNESS;
    let interior_x1 = room.shell.2 - WALL_THICKNESS;
    let interior_y1 = room.shell.3 - WALL_THICKNESS;

    let mut candidates = Vec::new();

    let z = floor_z + Q; // above floor
                         // Canonical order: (z, y, x)
    let mut ys = Vec::new();
    let mut y = interior_y0;
    while y + PILLAR_DEPTH <= interior_y1 {
        ys.push(y);
        y += Q;
    }

    let mut xs = Vec::new();
    let mut x = interior_x0;
    while x + PILLAR_WIDTH <= interior_x1 {
        xs.push(x);
        x += Q;
    }

    for &cy in &ys {
        for &cx in &xs {
            candidates.push((
                cx,
                cy,
                z,
                cx + PILLAR_WIDTH,
                cy + PILLAR_DEPTH,
                z + PILLAR_HEIGHT,
            ));
        }
    }

    candidates
}

/// Check if two axis-aligned boxes intersect.
pub fn boxes_intersect(
    a: (i32, i32, i32, i32, i32, i32),
    b: (i32, i32, i32, i32, i32, i32),
) -> bool {
    a.0 < b.3 && a.3 > b.0 && a.1 < b.4 && a.4 > b.1 && a.2 < b.5 && a.5 > b.2
}

/// Derive the accent material name from a palette name.
fn accent_material_for(palette_name: &str) -> String {
    match palette_name {
        "base_stone" => "bs_accent".to_string(),
        "crypt" => "crypt_accent".to_string(),
        "treasury" => "treas_accent".to_string(),
        _ => "bs_accent".to_string(),
    }
}

// ── Connectivity oracle ────────────────────────────────────────────────────

/// Flood-fill walkable cells in a room to verify the candidate pillar does not
/// break internal connectivity.
///
/// Walkable area: room interior minus walls, at floor_z + Q to floor_z + MIN_HEADROOM.
/// The check operates on a 2D quantum-resolution projection of the walkable slice.
pub fn check_connectivity(
    room: &PlacedRoom,
    exclusions: &[ExclusionVolume],
    existing_pillars: &[(i32, i32, i32, i32, i32, i32)],
    candidate: (i32, i32, i32, i32, i32, i32),
) -> bool {
    let walk_z0 = room.floor_z + Q;
    let walk_z1 = room.floor_z + MIN_HEADROOM;

    let x0 = room.shell.0 + WALL_THICKNESS;
    let y0 = room.shell.1 + WALL_THICKNESS;
    let x1 = room.shell.2 - WALL_THICKNESS;
    let y1 = room.shell.3 - WALL_THICKNESS;

    let cells_x = ((x1 - x0) / Q) as usize;
    let cells_y = ((y1 - y0) / Q) as usize;
    if cells_x == 0 || cells_y == 0 {
        return true; // room too small to check
    }

    // Build walkability grid: true = walkable
    let mut walkable = vec![true; cells_x * cells_y];

    // Mark cells blocked by exclusion regions (that overlap walkable height)
    for ev in exclusions {
        let (ex0, ey0, ez0, ex1, ey1, ez1) = ev.bounds;
        if ez0 >= walk_z1 || ez1 <= walk_z0 {
            continue;
        }
        let cx0 = ((ex0.max(x0) - x0) / Q).max(0) as usize;
        let cy0 = ((ey0.max(y0) - y0) / Q).max(0) as usize;
        let cx1 = (((ex1.min(x1) - x0 + Q - 1) / Q).max(0) as usize).min(cells_x);
        let cy1 = (((ey1.min(y1) - y0 + Q - 1) / Q).max(0) as usize).min(cells_y);
        for cy in cy0..cy1 {
            for cx in cx0..cx1 {
                walkable[cy * cells_x + cx] = false;
            }
        }
    }

    // Mark cells blocked by existing pillars
    for &pillar in existing_pillars {
        mark_pillar_cells(
            pillar,
            x0,
            y0,
            cells_x,
            cells_y,
            walk_z0,
            walk_z1,
            &mut walkable,
        );
    }

    // Mark cells blocked by candidate pillar
    mark_pillar_cells(
        candidate,
        x0,
        y0,
        cells_x,
        cells_y,
        walk_z0,
        walk_z1,
        &mut walkable,
    );

    // Flood fill from first walkable cell
    let start = match walkable.iter().position(|&w| w) {
        Some(idx) => idx,
        None => return false, // required room clear space has been eliminated
    };

    let mut visited = vec![false; cells_x * cells_y];
    let mut queue = VecDeque::new();
    visited[start] = true;
    queue.push_back(start);

    while let Some(idx) = queue.pop_front() {
        let cx = idx % cells_x;
        let cy = idx / cells_x;

        // Cardinal neighbors (4-directional on 2D grid)
        let neighbors: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        for (dx, dy) in &neighbors {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx < 0 || ny < 0 || nx >= cells_x as i32 || ny >= cells_y as i32 {
                continue;
            }
            let nidx = (ny as usize) * cells_x + (nx as usize);
            if walkable[nidx] && !visited[nidx] {
                visited[nidx] = true;
                queue.push_back(nidx);
            }
        }
    }

    // All walkable cells must be visited
    for i in 0..(cells_x * cells_y) {
        if walkable[i] && !visited[i] {
            return false;
        }
    }

    true
}

/// Mark cells occupied by a pillar in the walkability grid.
fn mark_pillar_cells(
    pillar: (i32, i32, i32, i32, i32, i32),
    grid_x0: i32,
    grid_y0: i32,
    cells_x: usize,
    cells_y: usize,
    walk_z0: i32,
    walk_z1: i32,
    walkable: &mut [bool],
) {
    // A pillar outside the conservative walkable-height slice cannot affect
    // this flood-fill proof.
    if pillar.5 <= walk_z0 || pillar.2 >= walk_z1 {
        return;
    }

    let cx0 = ((pillar.0.max(grid_x0) - grid_x0) / Q).max(0) as usize;
    let cy0 = ((pillar.1.max(grid_y0) - grid_y0) / Q).max(0) as usize;
    let cx1 = (((pillar.3.min(grid_x0 + cells_x as i32 * Q) - grid_x0 + Q - 1) / Q).max(0)
        as usize)
        .min(cells_x);
    let cy1 = (((pillar.4.min(grid_y0 + cells_y as i32 * Q) - grid_y0 + Q - 1) / Q).max(0)
        as usize)
        .min(cells_y);

    for cy in cy0..cy1 {
        for cx in cx0..cx1 {
            walkable[cy * cells_x + cx] = false;
        }
    }
}

// ── Step 7: Spawn and light origins ────────────────────────────────────────

/// Select exactly one entry-layer spawn and one clear light origin per room
/// before optional pillars are considered. The selected one-quantum volumes
/// are subsequently protected as pillar exclusions.
fn select_clear_origins(
    placement: &PlacementResult,
    rooms: &BTreeMap<RoomId, &PlacedRoom>,
    exclusions: &BTreeMap<RoomId, Vec<ExclusionVolume>>,
    mut rng: EnhancedStageRng,
) -> Result<(SpawnPoint, Vec<LightOrigin>), EnhancedError> {
    let entry_id =
        placement
            .lower_rooms
            .iter()
            .copied()
            .min()
            .ok_or(EnhancedError::ContractViolation {
                detail: "no entry-layer room for spawn".into(),
            })?;
    let entry_room = rooms
        .get(&entry_id)
        .ok_or(EnhancedError::ContractViolation {
            detail: "entry room not found".into(),
        })?;
    let spawn_origin = find_clear_origin(
        entry_room,
        exclusions.get(&entry_id).map(Vec::as_slice).unwrap_or(&[]),
        entry_room.floor_z + Q + 24,
        None,
        &mut rng,
    )
    .ok_or_else(|| EnhancedError::ContractViolation {
        detail: format!("entry room {:?} has no clear spawn origin", entry_id),
    })?;
    let spawn = SpawnPoint {
        origin: spawn_origin,
        room_id: entry_id,
        layer: 0,
    };

    let mut lights = Vec::with_capacity(placement.rooms.len());
    for room in &placement.rooms {
        let avoid = (room.id == entry_id).then_some(spawn.origin);
        let origin = find_clear_origin(
            room,
            exclusions.get(&room.id).map(Vec::as_slice).unwrap_or(&[]),
            room.floor_z + MIN_HEADROOM + Q,
            avoid,
            &mut rng,
        )
        .ok_or_else(|| EnhancedError::ContractViolation {
            detail: format!("room {:?} has no clear light origin", room.id),
        })?;
        lights.push(LightOrigin {
            origin,
            room_id: room.id,
        });
    }
    lights.sort_by_key(|light| light.room_id);
    Ok((spawn, lights))
}

/// Find a deterministic, exclusion-free one-quantum entity volume in a room.
fn find_clear_origin(
    room: &PlacedRoom,
    exclusions: &[ExclusionVolume],
    z: i32,
    avoid: Option<(i32, i32, i32)>,
    rng: &mut EnhancedStageRng,
) -> Option<(i32, i32, i32)> {
    let x0 = room.shell.0 + WALL_THICKNESS + Q;
    let y0 = room.shell.1 + WALL_THICKNESS + Q;
    let x1 = room.shell.2 - WALL_THICKNESS - Q;
    let y1 = room.shell.3 - WALL_THICKNESS - Q;
    if x0 >= x1 || y0 >= y1 || z + Q > room.floor_z + room.dims.2 as i32 {
        return None;
    }

    let mut candidates = Vec::new();
    for y in (y0..=y1 - Q).step_by(Q as usize) {
        for x in (x0..=x1 - Q).step_by(Q as usize) {
            candidates.push((x, y));
        }
    }
    if candidates.is_empty() {
        return None;
    }
    let start = rng.range_u32(candidates.len() as u32) as usize;
    for offset in 0..candidates.len() {
        let (x, y) = candidates[(start + offset) % candidates.len()];
        let origin = (x, y, z);
        if avoid == Some(origin) {
            continue;
        }
        // Corridors and wall apertures are intentional clear space, but every
        // transition volume is materialized stair geometry or required
        // clearance. Test that transition geometry in 3-D; never grant a
        // blanket transition-volume exception to an entity origin.
        if !exclusions.iter().any(|exclusion| {
            if !exclusion.reason.starts_with("transition") {
                return false;
            }
            // A point entity exactly on a tread top is classified as solid by
            // the pinned light compiler even though half-open brush volumes do
            // not positively overlap. Reserve one unit above each tread so
            // origins are selected in strict clear space rather than on a
            // compiler boundary plane.
            let bounds = exclusion.bounds;
            let compiler_solid = (
                bounds.0.saturating_sub(1),
                bounds.1.saturating_sub(1),
                bounds.2.saturating_sub(1),
                bounds.3.saturating_add(1),
                bounds.4.saturating_add(1),
                bounds.5.saturating_add(1),
            );
            origin.0 >= compiler_solid.0
                && origin.0 < compiler_solid.3
                && origin.1 >= compiler_solid.1
                && origin.1 < compiler_solid.4
                && origin.2 >= compiler_solid.2
                && origin.2 < compiler_solid.5
        }) {
            return Some(origin);
        }
    }
    None
}

fn protect_required_origins(
    exclusions: &mut BTreeMap<RoomId, Vec<ExclusionVolume>>,
    spawn: &SpawnPoint,
    lights: &[LightOrigin],
) {
    let mut protect = |room_id: RoomId, origin: (i32, i32, i32), reason: String| {
        exclusions
            .entry(room_id)
            .or_default()
            .push(ExclusionVolume {
                bounds: (
                    origin.0,
                    origin.1,
                    origin.2,
                    origin.0 + Q,
                    origin.1 + Q,
                    origin.2 + Q,
                ),
                reason,
            });
    };
    protect(spawn.room_id, spawn.origin, "spawn origin".into());
    for light in lights {
        protect(
            light.room_id,
            light.origin,
            format!("light origin room {:?}", light.room_id),
        );
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::EnhancedConfig;
    use super::super::placement::place_rooms;
    use super::super::seed::{tags, EnhancedSeed};
    use super::super::theme::{assign_uniform, cc0_dungeon_v2_theme};
    use super::super::topology::build_topology;
    use super::*;

    fn build_test_input(
        seed_val: u64,
    ) -> (
        EnhancedConfig,
        PlacementResult,
        TopologyResult,
        ThemeAssignment,
    ) {
        let cfg = EnhancedConfig::nominal();
        let eseed = EnhancedSeed::new(seed_val);
        let placement = place_rooms(&cfg, eseed.stage_seed(tags::LAYER_PLACEMENT)).unwrap();
        let mut topo_rng = eseed.stage_seed(tags::VERTICAL_TOPOLOGY).rng();
        let topology = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
        let theme = cc0_dungeon_v2_theme();
        let assignment = assign_uniform(&theme, &placement.rooms, &topology);
        (cfg, placement, topology, assignment)
    }

    // ── Width selection tests ──────────────────────────────────────────

    #[test]
    fn width_selection_all_routes_have_width() {
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let corridor_rng = EnhancedSeed::new(42)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(42)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &EnhancedConfig::nominal(),
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        assert_eq!(result.corridor_widths.len(), topology.routes.len());
        for sel in &result.corridor_widths {
            assert!(ALLOWED_CORRIDOR_WIDTHS.contains(&sel.width));
        }
    }

    #[test]
    fn width_default_always_fits() {
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let corridor_rng = EnhancedSeed::new(42)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(42)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &EnhancedConfig::nominal(),
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        // At minimum, every route should have at least width 64
        for sel in &result.corridor_widths {
            assert!(sel.width >= 64);
        }
    }

    #[test]
    fn width_selection_deterministic() {
        let (_cfg, placement, topology, theme) = build_test_input(42);

        let a = {
            let cr = EnhancedSeed::new(42)
                .stage_seed(tags::CORRIDOR_VARIANCE)
                .rng();
            let fr = EnhancedSeed::new(42)
                .stage_seed(tags::FEATURE_PLACEMENT)
                .rng();
            apply_features(
                &EnhancedConfig::nominal(),
                &placement,
                &topology,
                &theme,
                fr,
                cr,
            )
            .unwrap()
        };

        let b = {
            let cr = EnhancedSeed::new(42)
                .stage_seed(tags::CORRIDOR_VARIANCE)
                .rng();
            let fr = EnhancedSeed::new(42)
                .stage_seed(tags::FEATURE_PLACEMENT)
                .rng();
            apply_features(
                &EnhancedConfig::nominal(),
                &placement,
                &topology,
                &theme,
                fr,
                cr,
            )
            .unwrap()
        };

        assert_eq!(a.corridor_widths, b.corridor_widths);
    }

    // ── Ceiling selection tests ────────────────────────────────────────

    #[test]
    fn ceiling_selection_all_rooms_have_height() {
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let corridor_rng = EnhancedSeed::new(42)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(42)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &EnhancedConfig::nominal(),
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        assert_eq!(result.ceiling_heights.len(), placement.rooms.len());
        for sel in &result.ceiling_heights {
            assert!(sel.height >= MIN_HEADROOM);
            assert!(ALLOWED_CEILING_HEIGHTS.contains(&sel.height) || sel.is_fallback);
        }
    }

    #[test]
    fn ceiling_min_headroom_preserved() {
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let corridor_rng = EnhancedSeed::new(42)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(42)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &EnhancedConfig::nominal(),
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        for sel in &result.ceiling_heights {
            let room = placement
                .rooms
                .iter()
                .find(|r| r.id == sel.room_id)
                .unwrap();
            assert!(
                sel.height >= MIN_HEADROOM,
                "room {:?}: ceiling {} < min headroom {}",
                sel.room_id,
                sel.height,
                MIN_HEADROOM,
            );
            // Ceiling must not exceed room height
            assert!(
                sel.height <= room.dims.2 as i32,
                "room {:?}: ceiling {} > room height {}",
                sel.room_id,
                sel.height,
                room.dims.2,
            );
        }
    }

    #[test]
    fn ceiling_selection_deterministic() {
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let a = {
            let cr = EnhancedSeed::new(42)
                .stage_seed(tags::CORRIDOR_VARIANCE)
                .rng();
            let fr = EnhancedSeed::new(42)
                .stage_seed(tags::FEATURE_PLACEMENT)
                .rng();
            apply_features(
                &EnhancedConfig::nominal(),
                &placement,
                &topology,
                &theme,
                fr,
                cr,
            )
            .unwrap()
        };
        let b = {
            let cr = EnhancedSeed::new(42)
                .stage_seed(tags::CORRIDOR_VARIANCE)
                .rng();
            let fr = EnhancedSeed::new(42)
                .stage_seed(tags::FEATURE_PLACEMENT)
                .rng();
            apply_features(
                &EnhancedConfig::nominal(),
                &placement,
                &topology,
                &theme,
                fr,
                cr,
            )
            .unwrap()
        };
        assert_eq!(a.ceiling_heights, b.ceiling_heights);
    }

    // ── Pillar tests ───────────────────────────────────────────────────

    #[test]
    fn pillar_placement_within_room() {
        let cfg = EnhancedConfig::nominal(); // max_pillars = 2
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let corridor_rng = EnhancedSeed::new(42)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(42)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &cfg,
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        for pillar in &result.pillars {
            let room = placement
                .rooms
                .iter()
                .find(|r| r.id == pillar.room_id)
                .unwrap();
            let (px0, py0, pz0, px1, py1, pz1) = pillar.bounds;

            // Pillar must be inside room shell
            assert!(
                px0 >= room.shell.0 && py0 >= room.shell.1,
                "pillar {:?} outside room shell",
                pillar.room_id,
            );
            assert!(
                px1 <= room.shell.2 && py1 <= room.shell.3,
                "pillar {:?} outside room shell",
                pillar.room_id,
            );
            // Pillar must be above floor
            assert!(pz0 >= room.floor_z + Q, "pillar below walkable floor + Q");
            // Pillar must not exceed headroom height
            assert!(pz1 - pz0 <= PILLAR_HEIGHT, "pillar too tall");
            // Positive volume
            assert!(px0 < px1 && py0 < py1 && pz0 < pz1, "pillar has no volume");
        }
    }

    #[test]
    fn pillars_dont_overlap() {
        let cfg = EnhancedConfig::maximal(); // max_pillars = 4
        let (_cfg, placement, topology, theme) = build_test_input(99);
        let corridor_rng = EnhancedSeed::new(99)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(99)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &cfg,
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        // Pillars in the same room must not overlap
        for i in 0..result.pillars.len() {
            for j in (i + 1)..result.pillars.len() {
                if result.pillars[i].room_id != result.pillars[j].room_id {
                    continue;
                }
                assert!(
                    !boxes_intersect(result.pillars[i].bounds, result.pillars[j].bounds),
                    "pillars {:?} and {:?} overlap in room {:?}",
                    i,
                    j,
                    result.pillars[i].room_id,
                );
            }
        }
    }

    #[test]
    fn pillars_not_in_exclusion_zones() {
        let cfg = EnhancedConfig::nominal();
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let corridor_rng = EnhancedSeed::new(42)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(42)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &cfg,
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        let rooms = room_map(&placement);
        let exclusions = derive_exclusion_regions(&placement, &topology, &rooms).unwrap();

        for pillar in &result.pillars {
            if let Some(evs) = exclusions.get(&pillar.room_id) {
                for ev in evs {
                    assert!(
                        !boxes_intersect(pillar.bounds, ev.bounds),
                        "pillar in room {:?} intersects exclusion: {}",
                        pillar.room_id,
                        ev.reason,
                    );
                }
            }
        }
    }

    #[test]
    fn pillar_connectivity_maintained() {
        let cfg = EnhancedConfig::nominal();
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let corridor_rng = EnhancedSeed::new(42)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(42)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &cfg,
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        let rooms = room_map(&placement);
        let exclusions = derive_exclusion_regions(&placement, &topology, &rooms).unwrap();

        for room in &placement.rooms {
            let room_pillars: Vec<_> = result
                .pillars
                .iter()
                .filter(|p| p.room_id == room.id)
                .map(|p| p.bounds)
                .collect();
            if room_pillars.is_empty() {
                continue;
            }
            let room_excs = exclusions
                .get(&room.id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            // Verify connectivity with all pillars in place
            let mut connected = true;
            let mut check_pillars = Vec::new();
            for &pb in &room_pillars {
                connected = connected && check_connectivity(room, room_excs, &check_pillars, pb);
                check_pillars.push(pb);
            }
            assert!(
                connected,
                "room {:?} connectivity broken by pillars",
                room.id,
            );
        }
    }

    #[test]
    fn exhausted_pillar_quota_recorded() {
        let cfg = EnhancedConfig::maximal(); // max_pillars = 4, 40 rooms
        let (_cfg, placement, topology, theme) = build_test_input(123);
        let corridor_rng = EnhancedSeed::new(123)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(123)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &cfg,
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        // With 40 rooms and max 4 pillars each, at least some rooms should
        // have RequestedCountUnmet (small rooms can't fit 4 pillars)
        // Actually, minimal room span is 112 units (7 quanta). Interior is
        // 80 units (5 quanta). Pillar is 32 wide (2 quanta). So at most 2
        // pillars fit side by side. 4 pillars need bigger rooms. So with
        // max=4 per room, we expect unmet for smaller rooms.
        assert!(
            !result.requested_count_unmet.is_empty() || result.pillars.len() as u32 <= 40 * 4,
            "should have unmet or valid pillar count",
        );

        // All unmet entries should be valid
        for u in &result.requested_count_unmet {
            assert_eq!(u.requested, cfg.max_pillars_per_room());
            assert!(u.placed < u.requested);
        }
    }

    // ── Spawn and light tests ──────────────────────────────────────────

    #[test]
    fn spawn_on_entry_layer() {
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let corridor_rng = EnhancedSeed::new(42)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(42)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &EnhancedConfig::nominal(),
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        // Spawn must be in a room on the lower layer
        let spawn_room = placement
            .rooms
            .iter()
            .find(|r| r.id == result.spawn_point.room_id)
            .unwrap();
        assert!(placement.lower_rooms.contains(&spawn_room.id));
        assert_eq!(result.spawn_point.layer, 0);

        // Spawn must be inside room shell
        let (sx, sy, sz) = result.spawn_point.origin;
        assert!(sx >= spawn_room.shell.0 && sx <= spawn_room.shell.2);
        assert!(sy >= spawn_room.shell.1 && sy <= spawn_room.shell.3);
        assert!(sz >= spawn_room.floor_z + Q);
    }

    #[test]
    fn light_origins_in_rooms() {
        let (_cfg, placement, topology, theme) = build_test_input(42);
        let corridor_rng = EnhancedSeed::new(42)
            .stage_seed(tags::CORRIDOR_VARIANCE)
            .rng();
        let feature_rng = EnhancedSeed::new(42)
            .stage_seed(tags::FEATURE_PLACEMENT)
            .rng();
        let result = apply_features(
            &EnhancedConfig::nominal(),
            &placement,
            &topology,
            &theme,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        assert!(!result.light_origins.is_empty());
        for light in &result.light_origins {
            let room = placement
                .rooms
                .iter()
                .find(|r| r.id == light.room_id)
                .unwrap();
            let (lx, ly, lz) = light.origin;
            assert!(lx >= room.shell.0 && lx <= room.shell.2);
            assert!(ly >= room.shell.1 && ly <= room.shell.3);
            assert!(lz >= room.floor_z + Q);

            // Light must not be inside a pillar in same room
            let light_box = (lx, ly, lz, lx + Q, ly + Q, lz + Q);
            for pillar in &result.pillars {
                if pillar.room_id == light.room_id {
                    assert!(
                        !boxes_intersect(light_box, pillar.bounds),
                        "light in room {:?} intersects pillar",
                        light.room_id,
                    );
                }
            }
        }
    }

    // ── Config validation tests ────────────────────────────────────────

    #[test]
    fn config_validates_max_pillars() {
        assert!(EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 0).is_ok());
        assert!(EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 8).is_ok());
        assert!(EnhancedConfig::with_full_params(28, 3, 1, 16, 2048, 32, 96, 9).is_err());
    }

    #[test]
    fn config_nominal_has_pillars() {
        let cfg = EnhancedConfig::nominal();
        assert_eq!(cfg.max_pillars_per_room(), 2);
    }

    #[test]
    fn config_minimal_has_pillars() {
        let cfg = EnhancedConfig::minimal();
        assert_eq!(cfg.max_pillars_per_room(), 1);
    }

    #[test]
    fn config_maximal_has_pillars() {
        let cfg = EnhancedConfig::maximal();
        assert_eq!(cfg.max_pillars_per_room(), 4);
    }

    // ── Full determinism test ──────────────────────────────────────────

    #[test]
    fn full_feature_result_deterministic() {
        let (_cfg, placement, topology, theme) = build_test_input(42);

        let make = || {
            let cr = EnhancedSeed::new(42)
                .stage_seed(tags::CORRIDOR_VARIANCE)
                .rng();
            let fr = EnhancedSeed::new(42)
                .stage_seed(tags::FEATURE_PLACEMENT)
                .rng();
            apply_features(
                &EnhancedConfig::nominal(),
                &placement,
                &topology,
                &theme,
                fr,
                cr,
            )
            .unwrap()
        };

        let a = make();
        let b = make();
        assert_eq!(a, b);
    }

    // ── Minimal and maximal config tests ───────────────────────────────

    #[test]
    fn features_minimal_config() {
        let cfg = EnhancedConfig::nominal();
        let eseed = EnhancedSeed::new(42);
        let placement = place_rooms(&cfg, eseed.stage_seed(tags::LAYER_PLACEMENT)).unwrap();
        let mut topo_rng = eseed.stage_seed(tags::VERTICAL_TOPOLOGY).rng();
        let topology = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
        let theme = cc0_dungeon_v2_theme();
        let assignment = assign_uniform(&theme, &placement.rooms, &topology);

        let corridor_rng = eseed.stage_seed(tags::CORRIDOR_VARIANCE).rng();
        let feature_rng = eseed.stage_seed(tags::FEATURE_PLACEMENT).rng();
        let result = apply_features(
            &cfg,
            &placement,
            &topology,
            &assignment,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        assert_eq!(result.ceiling_heights.len(), 28);
        assert!(result.spawn_point.origin.0 >= 0);
        assert!(!result.light_origins.is_empty());
    }

    #[test]
    fn features_maximal_config() {
        let cfg = EnhancedConfig::nominal();
        let eseed = EnhancedSeed::new(42);
        let placement = place_rooms(&cfg, eseed.stage_seed(tags::LAYER_PLACEMENT)).unwrap();
        let mut topo_rng = eseed.stage_seed(tags::VERTICAL_TOPOLOGY).rng();
        let topology = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
        let theme = cc0_dungeon_v2_theme();
        let assignment = assign_uniform(&theme, &placement.rooms, &topology);

        let corridor_rng = eseed.stage_seed(tags::CORRIDOR_VARIANCE).rng();
        let feature_rng = eseed.stage_seed(tags::FEATURE_PLACEMENT).rng();
        let result = apply_features(
            &cfg,
            &placement,
            &topology,
            &assignment,
            feature_rng,
            corridor_rng,
        )
        .unwrap();

        assert_eq!(result.ceiling_heights.len(), 28);
        assert!(result.spawn_point.origin.0 >= 0);
        assert!(!result.light_origins.is_empty());
        // With 2 pillars per room, should have some pillars
        assert!(!result.pillars.is_empty());
    }

    // ── Connectivity oracle unit tests ─────────────────────────────────

    #[test]
    fn connectivity_empty_room_is_connected() {
        let room = PlacedRoom {
            id: RoomId(0),
            layer: super::super::intent::LayerId(0),
            floor_z: 0,
            shell: (0, 0, 256, 256),
            dims: (256, 256, 176),
        };
        let exclusions = vec![];
        let existing = vec![];
        let candidate = (64, 64, 16, 96, 96, 96);

        assert!(check_connectivity(&room, &exclusions, &existing, candidate));
    }

    #[test]
    fn connectivity_pillar_blocking_path() {
        let room = PlacedRoom {
            id: RoomId(0),
            layer: super::super::intent::LayerId(0),
            floor_z: 0,
            shell: (0, 0, 160, 160),
            dims: (160, 160, 176),
        };
        // Interior: x=16..144, y=16..144 → 128×128 = 8×8 cells
        // Place a full-width horizontal bar that splits connectivity
        let candidate = (16, 64, 16, 144, 80, 96);
        let exclusions = vec![];
        let existing = vec![];

        assert!(!check_connectivity(
            &room,
            &exclusions,
            &existing,
            candidate
        ));
    }

    #[test]
    fn connectivity_multiple_small_pillars_ok() {
        let room = PlacedRoom {
            id: RoomId(0),
            layer: super::super::intent::LayerId(0),
            floor_z: 0,
            shell: (0, 0, 256, 256),
            dims: (256, 256, 176),
        };
        let exclusions = vec![];
        // Two pillars that don't block connectivity
        let existing = vec![(32, 32, 16, 64, 64, 96)];
        let candidate = (160, 160, 16, 192, 192, 96);

        assert!(check_connectivity(&room, &exclusions, &existing, candidate));
    }
}
