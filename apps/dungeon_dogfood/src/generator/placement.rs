use std::num::NonZeroU32;

use super::config::NormalizedGeneratorConfig;
use super::determinism::{Pcg32V1, SemanticComponent, SemanticStage, SemanticStreamFactory};
use super::error::{ErrorStage, GeneratorError};
use super::ir::{
    Direction, GridCoord, IdAllocator, IntendedTopology, OccupancyClass, OccupancyGrid,
    PlacedRegion, PlacedSocket, RegionRole, SocketRole,
    TransitionReservation,
};
use super::prefab::{
    PrefabCatalog, PrefabVariant, ReservationKind, ReservationOwner, SocketRole as PrefabSocketRole,
};

// ─── Role manifest ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct RoleManifest {
    spawn_count: u32,
    distant_landmark_count: u32,
    major_landmark_count: u32,
    junction_count: u32,
    dead_end_count: u32,
    vertical_hub_count: u32,
    required_route_count: u32,
    optional_branch_count: u32,
    ordinary_count: u32,
}

impl RoleManifest {
    fn from_config(
        config: &NormalizedGeneratorConfig,
        rng: &mut Pcg32V1,
    ) -> Result<Self, GeneratorError> {
        let upper = config.region_max().checked_add(1).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_region_upper",
            },
        )?;
        let total = rng.gen_range(config.region_min(), upper)?;
        let layer_pairs = u32::from(config.layers().2.checked_sub(1).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_layer_pairs",
            },
        )?);
        let transitions = layer_pairs
            .checked_mul(config.transitions_per_adjacent_pair())
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_transition_count",
            })?;
        let vertical_hub_count = transitions.checked_mul(2).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_transition_endpoint_count",
            },
        )?;
        let mandatory = 5u32
            .checked_add(vertical_hub_count)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_mandatory_count",
            })?;
        let remaining = total.checked_sub(mandatory).ok_or(
            GeneratorError::MandatoryInfeasibility {
                stage: ErrorStage::Placement,
                constraint: "mandatory_role_minimum",
                required: u64::from(mandatory),
                available: u64::from(total),
            },
        )?;

        // Extra major, junction, dead-end, required-route, optional, ordinary.
        let weights = [20u64, 20, 15, 15, 15, 15];
        let weight_sum = 100u64;
        let mut shares = [0u32; 6];
        let mut allocated = 0u32;
        for (index, weight) in weights.iter().copied().enumerate() {
            let value = u64::from(remaining)
                .checked_mul(weight)
                .ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "role_share_mul",
                })?
                / weight_sum;
            let share = u32::try_from(value).map_err(|_| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_share_convert",
            })?;
            shares[index] = share;
            allocated = allocated.checked_add(share).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "role_share_sum",
                },
            )?;
        }
        let mut remainder = remaining.checked_sub(allocated).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_share_remainder",
            },
        )?;
        let mut index = 0usize;
        while remainder > 0 {
            let share = shares.get_mut(index).ok_or(GeneratorError::IrInvariant {
                stage: ErrorStage::Placement,
                detail: "role_share_index_out_of_bounds".into(),
            })?;
            *share = share.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_share_remainder_add",
            })?;
            remainder = remainder.checked_sub(1).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "role_share_remainder_sub",
                },
            )?;
            index = index.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_share_index_add",
            })? % shares.len();
        }

        Ok(Self {
            spawn_count: 1,
            distant_landmark_count: 1,
            major_landmark_count: 1u32.checked_add(shares[0]).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "major_role_count",
                },
            )?,
            junction_count: 1u32.checked_add(shares[1]).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "junction_role_count",
                },
            )?,
            dead_end_count: 1u32.checked_add(shares[2]).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "dead_end_role_count",
                },
            )?,
            vertical_hub_count,
            required_route_count: shares[3],
            optional_branch_count: shares[4],
            ordinary_count: shares[5],
        })
    }

    fn total_regions(&self) -> Result<u32, GeneratorError> {
        [
            self.spawn_count,
            self.distant_landmark_count,
            self.major_landmark_count,
            self.junction_count,
            self.dead_end_count,
            self.vertical_hub_count,
            self.required_route_count,
            self.optional_branch_count,
            self.ordinary_count,
        ]
        .into_iter()
        .try_fold(0u32, |sum, value| {
            sum.checked_add(value).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "role_total",
            })
        })
    }

    fn role_counts(&self) -> [(RegionRole, u32); 9] {
        [
            (RegionRole::Spawn, self.spawn_count),
            (RegionRole::DistantLandmark, self.distant_landmark_count),
            (RegionRole::MajorLandmark, self.major_landmark_count),
            (RegionRole::Junction, self.junction_count),
            (RegionRole::DeadEnd, self.dead_end_count),
            (RegionRole::VerticalHub, self.vertical_hub_count),
            (RegionRole::RequiredRoute, self.required_route_count),
            (RegionRole::OptionalBranch, self.optional_branch_count),
            (RegionRole::OrdinaryRoom, self.ordinary_count),
        ]
    }
}

fn variant_supports_role(variant: &PrefabVariant, role: RegionRole) -> bool {
    let has_tag = |choices: &[&str]| {
        variant
            .tags
            .iter()
            .any(|tag| choices.iter().any(|choice| tag == choice))
    };
    match role {
        RegionRole::Spawn => has_tag(&["spawn", "ordinary", "room"]),
        RegionRole::DistantLandmark => has_tag(&["landmark", "major", "room"]),
        RegionRole::MajorLandmark => has_tag(&["landmark", "major", "room", "hall"]),
        RegionRole::Junction => has_tag(&["junction", "hall", "room"]),
        RegionRole::DeadEnd => has_tag(&["dead_end", "room"]),
        RegionRole::VerticalHub => {
            has_tag(&["ramp", "vertical", "hub"]) && !variant.transitions.is_empty()
        }
        RegionRole::RequiredRoute => has_tag(&["ordinary", "room", "hall", "corridor"]),
        RegionRole::OptionalBranch => has_tag(&["ordinary", "room"]),
        RegionRole::OrdinaryRoom => has_tag(&["ordinary", "room", "small"]),
    }
}

fn variants_for_role(
    catalog: &PrefabCatalog,
    role: RegionRole,
) -> Result<Vec<(u16, &PrefabVariant)>, GeneratorError> {
    catalog
        .variants()
        .iter()
        .enumerate()
        .filter(|(_, variant)| {
            variant_supports_role(variant, role)
                && (role == RegionRole::VerticalHub || variant.layer_count == 1)
        })
        .map(|(index, variant)| {
            let index = u16::try_from(index).map_err(|_| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "variant_index_convert",
            })?;
            Ok((index, variant))
        })
        .collect()
}

fn verify_mandatory_coverage(
    catalog: &PrefabCatalog,
    manifest: &RoleManifest,
) -> Result<(), GeneratorError> {
    for (role, count) in manifest.role_counts() {
        if count > 0 && variants_for_role(catalog, role)?.is_empty() {
            return Err(GeneratorError::MandatoryInfeasibility {
                stage: ErrorStage::Placement,
                constraint: "prefab_role_coverage",
                required: u64::from(count),
                available: 0,
            });
        }
    }
    Ok(())
}

// ─── Transition reservation and endpoint placement ─────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RampCandidate {
    variant_index: u16,
    transition_index: u16,
    lower_layer: u16,
    x: u16,
    y: u16,
}

#[derive(Debug, Clone)]
struct MaterializedTransition {
    variant_index: u16,
    lower_layer: u16,
    hub_footprint: (u16, u16, u16, u16),
    ramp_run_cells: Vec<GridCoord>,
    upper_opening_cells: Vec<GridCoord>,
    landing_cells: Vec<GridCoord>,
    headroom_cells: Vec<GridCoord>,
    lower_funnel_cells: Vec<GridCoord>,
    lower_approach_cells: Vec<GridCoord>,
    lower_socket_index: usize,
    upper_socket_index: usize,
}

fn enumerate_ramp_candidates(
    catalog: &PrefabCatalog,
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<RampCandidate>, GeneratorError> {
    let mut candidates = Vec::new();
    let layer_pairs = config.layers().2.checked_sub(1).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "ramp_candidate_layer_pairs",
        },
    )?;
    for (variant_index, variant) in catalog.variants().iter().enumerate() {
        if !variant_supports_role(variant, RegionRole::VerticalHub)
            || variant.width > config.width()
            || variant.height > config.height()
        {
            continue;
        }
        let variant_index = u16::try_from(variant_index).map_err(|_| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "ramp_variant_index_convert",
            }
        })?;
        let max_x = config.width().checked_sub(variant.width).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "ramp_candidate_max_x",
            },
        )?;
        let max_y = config.height().checked_sub(variant.height).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "ramp_candidate_max_y",
            },
        )?;
        for transition_index in 0..variant.transitions.len() {
            let transition_index = u16::try_from(transition_index).map_err(|_| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "ramp_transition_index_convert",
                }
            })?;
            for lower_layer in 0..layer_pairs {
                for y in 1..max_y {
                    for x in 1..max_x {
                        candidates.push(RampCandidate {
                            variant_index,
                            transition_index,
                            lower_layer,
                            x,
                            y,
                        });
                    }
                }
            }
        }
    }
    candidates.sort();
    Ok(candidates)
}

fn materialize_transition(
    candidate: RampCandidate,
    catalog: &PrefabCatalog,
    config: &NormalizedGeneratorConfig,
) -> Result<MaterializedTransition, GeneratorError> {
    let variant = catalog
        .variants()
        .get(usize::from(candidate.variant_index))
        .ok_or(GeneratorError::TransitionBinding {
            stage: ErrorStage::Placement,
            transition: u32::from(candidate.transition_index),
            reason: "variant_not_found",
        })?;
    let transition = variant
        .transitions
        .get(usize::from(candidate.transition_index))
        .ok_or(GeneratorError::TransitionBinding {
            stage: ErrorStage::Placement,
            transition: u32::from(candidate.transition_index),
            reason: "prefab_transition_not_found",
        })?;
    let lower_socket_index = variant
        .sockets
        .iter()
        .position(|socket| socket.id == transition.lower_approach_socket)
        .ok_or(GeneratorError::TransitionBinding {
            stage: ErrorStage::Placement,
            transition: u32::from(candidate.transition_index),
            reason: "lower_prefab_socket_not_found",
        })?;
    let upper_socket_index = variant
        .sockets
        .iter()
        .position(|socket| socket.id == transition.upper_landing_socket)
        .ok_or(GeneratorError::TransitionBinding {
            stage: ErrorStage::Placement,
            transition: u32::from(candidate.transition_index),
            reason: "upper_prefab_socket_not_found",
        })?;

    let mut ramp_run_cells = Vec::new();
    let mut upper_opening_cells = Vec::new();
    let mut landing_cells = Vec::new();
    let mut headroom_cells = Vec::new();
    let mut lower_funnel_cells = Vec::new();
    let mut lower_approach_cells = Vec::new();

    for reservation in &variant.reservations {
        if reservation.owner != ReservationOwner::Reference(transition.id.clone()) {
            continue;
        }
        let target = match reservation.kind {
            ReservationKind::RampVolume => &mut ramp_run_cells,
            ReservationKind::UpperOpening => &mut upper_opening_cells,
            ReservationKind::UpperLanding => &mut landing_cells,
            ReservationKind::Headroom => &mut headroom_cells,
            ReservationKind::SocketFunnel => &mut lower_funnel_cells,
            ReservationKind::CorridorApproach => &mut lower_approach_cells,
            ReservationKind::Footprint | ReservationKind::WallShell => continue,
        };
        for cell in &reservation.cells {
            let layer = candidate.lower_layer.checked_add(cell.layer).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "transition_cell_layer",
                },
            )?;
            let x = candidate.x.checked_add(cell.x).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "transition_cell_x",
                },
            )?;
            let y = candidate.y.checked_add(cell.y).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "transition_cell_y",
                },
            )?;
            target.push(GridCoord::new(
                layer,
                x,
                y,
                config.width(),
                config.height(),
                config.layers().2,
            )?);
        }
    }

    for cells in [
        &mut ramp_run_cells,
        &mut upper_opening_cells,
        &mut landing_cells,
        &mut headroom_cells,
        &mut lower_funnel_cells,
        &mut lower_approach_cells,
    ] {
        cells.sort();
        cells.dedup();
    }
    if ramp_run_cells.is_empty() || upper_opening_cells.is_empty() || landing_cells.is_empty() {
        return Err(GeneratorError::TransitionBinding {
            stage: ErrorStage::Placement,
            transition: u32::from(candidate.transition_index),
            reason: "required_transition_mask_empty",
        });
    }

    Ok(MaterializedTransition {
        variant_index: candidate.variant_index,
        lower_layer: candidate.lower_layer,
        hub_footprint: (candidate.x, candidate.y, variant.width, variant.height),
        ramp_run_cells,
        upper_opening_cells,
        landing_cells,
        headroom_cells,
        lower_funnel_cells,
        lower_approach_cells,
        lower_socket_index,
        upper_socket_index,
    })
}

fn projected_hub_cells(
    materialized: &MaterializedTransition,
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<GridCoord>, GeneratorError> {
    let upper = materialized.lower_layer.checked_add(1).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "hub_upper_layer",
        },
    )?;
    let (x, y, width, height) = materialized.hub_footprint;
    let capacity = usize::from(width)
        .checked_mul(usize::from(height))
        .and_then(|value| value.checked_mul(2))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "hub_cell_capacity",
        })?;
    let mut cells = Vec::with_capacity(capacity);
    for layer in [materialized.lower_layer, upper] {
        for dy in 0..height {
            for dx in 0..width {
                let cell_x = x.checked_add(dx).ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "hub_cell_x",
                })?;
                let cell_y = y.checked_add(dy).ok_or(GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "hub_cell_y",
                })?;
                cells.push(GridCoord::new(
                    layer,
                    cell_x,
                    cell_y,
                    config.width(),
                    config.height(),
                    config.layers().2,
                )?);
            }
        }
    }
    cells.sort();
    cells.dedup();
    Ok(cells)
}

fn transition_fits(
    materialized: &MaterializedTransition,
    grid: &OccupancyGrid,
    config: &NormalizedGeneratorConfig,
) -> Result<bool, GeneratorError> {
    for cell in projected_hub_cells(materialized, config)? {
        if grid.get(cell) != Some(OccupancyClass::Empty) {
            return Ok(false);
        }
    }
    for cell in materialized
        .ramp_run_cells
        .iter()
        .chain(&materialized.upper_opening_cells)
        .chain(&materialized.landing_cells)
        .chain(&materialized.headroom_cells)
    {
        if grid.get(*cell) != Some(OccupancyClass::Empty) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn placed_sockets_for_endpoint(
    variant: &PrefabVariant,
    variant_layer: u16,
    global_layer: u16,
    origin_x: u16,
    origin_y: u16,
    alloc: &mut IdAllocator,
    config: &NormalizedGeneratorConfig,
) -> Result<Vec<(usize, PlacedSocket)>, GeneratorError> {
    let mut sockets = Vec::new();
    for (index, socket) in variant.sockets.iter().enumerate() {
        if socket.anchor.layer != variant_layer {
            continue;
        }
        let x = origin_x.checked_add(socket.anchor.x).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "endpoint_socket_x",
            },
        )?;
        let y = origin_y.checked_add(socket.anchor.y).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "endpoint_socket_y",
            },
        )?;
        let variant_socket_index = u16::try_from(index).map_err(|_| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "endpoint_socket_index",
            }
        })?;
        sockets.push((
            index,
            PlacedSocket {
                id: alloc.next_socket()?,
                variant_socket_index,
                global_anchor: GridCoord::new(
                    global_layer,
                    x,
                    y,
                    config.width(),
                    config.height(),
                    config.layers().2,
                )?,
                direction: map_direction_from_variant(socket.direction),
                width: socket.width,
                role: map_socket_role_from_variant(socket.role),
                paired_socket_id: None,
            },
        ));
    }
    Ok(sockets)
}

fn commit_transition(
    materialized: &MaterializedTransition,
    catalog: &PrefabCatalog,
    grid: &mut OccupancyGrid,
    alloc: &mut IdAllocator,
    config: &NormalizedGeneratorConfig,
) -> Result<(TransitionReservation, PlacedRegion, PlacedRegion), GeneratorError> {
    let variant = catalog
        .variants()
        .get(usize::from(materialized.variant_index))
        .ok_or(GeneratorError::IrInvariant {
            stage: ErrorStage::Placement,
            detail: "selected_transition_variant_missing".into(),
        })?;
    let mut staged_grid = grid.clone();
    let mut staged_alloc = alloc.clone();
    let transition_id = staged_alloc.next_transition()?;
    let lower_region_id = staged_alloc.next_region()?;
    let upper_region_id = staged_alloc.next_region()?;
    let upper_layer = materialized.lower_layer.checked_add(1).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "commit_transition_upper_layer",
        },
    )?;
    let (origin_x, origin_y, width, height) = materialized.hub_footprint;

    let mut lower_sockets = placed_sockets_for_endpoint(
        variant,
        0,
        materialized.lower_layer,
        origin_x,
        origin_y,
        &mut staged_alloc,
        config,
    )?;
    let mut upper_sockets = placed_sockets_for_endpoint(
        variant,
        1,
        upper_layer,
        origin_x,
        origin_y,
        &mut staged_alloc,
        config,
    )?;
    let lower_socket = lower_sockets
        .iter()
        .find(|(index, _)| *index == materialized.lower_socket_index)
        .map(|(_, socket)| socket.id)
        .ok_or(GeneratorError::TransitionBinding {
            stage: ErrorStage::Placement,
            transition: transition_id.raw(),
            reason: "lower_endpoint_socket_missing",
        })?;
    let upper_socket = upper_sockets
        .iter()
        .find(|(index, _)| *index == materialized.upper_socket_index)
        .map(|(_, socket)| socket.id)
        .ok_or(GeneratorError::TransitionBinding {
            stage: ErrorStage::Placement,
            transition: transition_id.raw(),
            reason: "upper_endpoint_socket_missing",
        })?;
    for (_, socket) in &mut lower_sockets {
        if socket.id == lower_socket {
            socket.paired_socket_id = Some(upper_socket);
        }
    }
    for (_, socket) in &mut upper_sockets {
        if socket.id == upper_socket {
            socket.paired_socket_id = Some(lower_socket);
        }
    }

    let hub_class = OccupancyClass::TransitionHub(transition_id.raw());
    for cell in projected_hub_cells(materialized, config)? {
        if staged_grid.get(cell) != Some(OccupancyClass::Empty) {
            return Err(GeneratorError::OccupancyConflict {
                stage: ErrorStage::Placement,
                detail: format!("transition_hub_conflict {}", cell),
            });
        }
        staged_grid.set(cell, hub_class)?;
    }
    let transition_class = OccupancyClass::Transition(transition_id.raw());
    let mut protected_cells: Vec<GridCoord> = materialized
        .ramp_run_cells
        .iter()
        .chain(&materialized.upper_opening_cells)
        .chain(&materialized.landing_cells)
        .chain(&materialized.headroom_cells)
        .copied()
        .collect();
    protected_cells.sort();
    protected_cells.dedup();
    for cell in protected_cells {
        match staged_grid.get(cell) {
            Some(OccupancyClass::TransitionHub(owner)) if owner == transition_id.raw() => {
                staged_grid.set(cell, transition_class)?;
            }
            other => {
                return Err(GeneratorError::OccupancyConflict {
                    stage: ErrorStage::Placement,
                    detail: format!(
                        "transition_protected_cell_conflict {} existing={:?}",
                        cell, other
                    ),
                });
            }
        }
    }

    let lower_marker_indices = marker_indices_for_layer(variant, 0)?;
    let upper_marker_indices = marker_indices_for_layer(variant, 1)?;
    let lower_region = PlacedRegion {
        id: lower_region_id,
        role: RegionRole::VerticalHub,
        variant_index: materialized.variant_index,
        layer: materialized.lower_layer,
        footprint: (origin_x, origin_y, width, height),
        sockets: lower_sockets.into_iter().map(|(_, socket)| socket).collect(),
        transitions: vec![transition_id],
        marker_variant_indices: lower_marker_indices,
    };
    let upper_region = PlacedRegion {
        id: upper_region_id,
        role: RegionRole::VerticalHub,
        variant_index: materialized.variant_index,
        layer: upper_layer,
        footprint: (origin_x, origin_y, width, height),
        sockets: upper_sockets.into_iter().map(|(_, socket)| socket).collect(),
        transitions: vec![transition_id],
        marker_variant_indices: upper_marker_indices,
    };
    let reservation = TransitionReservation {
        id: transition_id,
        variant_index: materialized.variant_index,
        lower_layer: materialized.lower_layer,
        hub_footprint: materialized.hub_footprint,
        lower_region: lower_region_id,
        upper_region: upper_region_id,
        lower_socket,
        upper_socket,
        ramp_run_cells: materialized.ramp_run_cells.clone(),
        upper_opening_cells: materialized.upper_opening_cells.clone(),
        landing_cells: materialized.landing_cells.clone(),
        headroom_cells: materialized.headroom_cells.clone(),
        lower_funnel_cells: materialized.lower_funnel_cells.clone(),
        lower_approach_cells: materialized.lower_approach_cells.clone(),
    };

    *grid = staged_grid;
    *alloc = staged_alloc;
    Ok((reservation, lower_region, upper_region))
}

fn reserve_transitions_and_endpoints(
    candidates: &[RampCandidate],
    catalog: &PrefabCatalog,
    config: &NormalizedGeneratorConfig,
    grid: &mut OccupancyGrid,
    alloc: &mut IdAllocator,
    factory: SemanticStreamFactory,
) -> Result<(Vec<TransitionReservation>, Vec<PlacedRegion>), GeneratorError> {
    let layer_pairs = config.layers().2.checked_sub(1).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "reserve_transition_layer_pairs",
        },
    )?;
    let required = config.transitions_per_adjacent_pair();
    let mut transitions = Vec::new();
    let mut endpoints = Vec::new();

    for lower_layer in 0..layer_pairs {
        let upper_layer = lower_layer.checked_add(1).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "reserve_transition_upper_layer",
            },
        )?;
        let mut pair_candidates: Vec<RampCandidate> = candidates
            .iter()
            .copied()
            .filter(|candidate| candidate.lower_layer == lower_layer)
            .collect();
        let mut stream = factory.stream(
            SemanticStage::RampReservations,
            &[SemanticComponent::Index(u32::from(lower_layer))],
        );
        stream.shuffle(&mut pair_candidates)?;
        let mut placed = 0u32;
        let mut rejected = 0u64;
        for candidate in pair_candidates {
            if placed >= required {
                break;
            }
            let materialized = materialize_transition(candidate, catalog, config)?;
            if !transition_fits(&materialized, grid, config)? {
                rejected = rejected.checked_add(1).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Placement,
                        operation: "transition_rejection_count",
                    },
                )?;
                continue;
            }
            let (transition, lower, upper) =
                commit_transition(&materialized, catalog, grid, alloc, config)?;
            transitions.push(transition);
            endpoints.push(lower);
            endpoints.push(upper);
            placed = placed.checked_add(1).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "transition_placed_count",
            })?;
        }
        if placed < required {
            return Err(GeneratorError::TransitionInfeasible {
                stage: ErrorStage::Placement,
                lower_layer,
                upper_layer,
                required: u64::from(required),
                available: u64::from(placed),
                rejected,
            });
        }
    }
    transitions.sort_by_key(|transition| transition.id.raw());
    endpoints.sort_by_key(|region| region.id.raw());
    Ok((transitions, endpoints))
}

// ─── Ordinary region placement ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PlacementScore {
    separation_distance: u32,
    layer_population: u32,
    required_route_distance: u32,
    variant_usage: u32,
}

#[derive(Debug, Clone)]
struct PlacementCandidate {
    variant_index: u16,
    layer: u16,
    x: u16,
    y: u16,
    score: PlacementScore,
}

fn can_place_footprint(
    variant: &PrefabVariant,
    layer: u16,
    x: u16,
    y: u16,
    grid: &OccupancyGrid,
    config: &NormalizedGeneratorConfig,
) -> Result<bool, GeneratorError> {
    let end_x = x.checked_add(variant.width).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "footprint_end_x",
        },
    )?;
    let end_y = y.checked_add(variant.height).ok_or(
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "footprint_end_y",
        },
    )?;
    if layer >= config.layers().2
        || x == 0
        || y == 0
        || end_x >= config.width()
        || end_y >= config.height()
    {
        return Ok(false);
    }
    for dy in 0..variant.height {
        for dx in 0..variant.width {
            let cell_x = x.checked_add(dx).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "footprint_cell_x",
            })?;
            let cell_y = y.checked_add(dy).ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "footprint_cell_y",
            })?;
            let cell = GridCoord::new(
                layer,
                cell_x,
                cell_y,
                config.width(),
                config.height(),
                config.layers().2,
            )?;
            if grid.get(cell) != Some(OccupancyClass::Empty) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn placement_score(
    role: RegionRole,
    layer: u16,
    x: u16,
    y: u16,
    variant_index: u16,
    placed: &[PlacedRegion],
) -> Result<PlacementScore, GeneratorError> {
    let mut separation = u32::MAX;
    let mut required_distance = u32::MAX;
    let mut layer_population = 0u32;
    let mut variant_usage = 0u32;
    for region in placed {
        if region.layer == layer {
            layer_population = layer_population.checked_add(1).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "placement_layer_population",
                },
            )?;
        }
        if region.variant_index == variant_index {
            variant_usage = variant_usage.checked_add(1).ok_or(
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "placement_variant_usage",
                },
            )?;
        }
        if region.layer != layer {
            continue;
        }
        let distance = u32::from(x.abs_diff(region.footprint.0))
            .checked_add(u32::from(y.abs_diff(region.footprint.1)))
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "placement_distance_add",
            })?;
        separation = separation.min(distance);
        if matches!(
            region.role,
            RegionRole::Spawn
                | RegionRole::DistantLandmark
                | RegionRole::MajorLandmark
                | RegionRole::RequiredRoute
                | RegionRole::VerticalHub
        ) {
            required_distance = required_distance.min(distance);
        }
    }
    if separation == u32::MAX {
        separation = 0;
    }
    if required_distance == u32::MAX {
        required_distance = 0;
    }
    // Keep the distant landmark on the spawn layer. Otherwise a purely
    // population-balanced placement can put both at the same x/y on adjacent
    // layers and force every legal route through a remote transition, making
    // the configured route maximum unreachable before topology search starts.
    if role == RegionRole::DistantLandmark {
        let spawn_layer = placed
            .iter()
            .find(|region| region.role == RegionRole::Spawn)
            .map(|region| region.layer);
        if spawn_layer == Some(layer) {
            layer_population = 0;
            if let Some(spawn) = placed.iter().find(|region| region.role == RegionRole::Spawn) {
                separation = u32::from(x.abs_diff(spawn.footprint.0))
                    .checked_add(u32::from(y.abs_diff(spawn.footprint.1)))
                    .ok_or(GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Placement,
                        operation: "spawn_landmark_distance_add",
                    })?;
            }
        } else {
            separation = 0;
            layer_population = u32::MAX;
        }
    }
    Ok(PlacementScore {
        separation_distance: separation,
        layer_population,
        required_route_distance: required_distance,
        variant_usage,
    })
}

fn marker_indices_for_layer(
    variant: &PrefabVariant,
    layer: u16,
) -> Result<Vec<u16>, GeneratorError> {
    variant
        .markers
        .iter()
        .enumerate()
        .filter(|(_, marker)| marker.position.layer == layer)
        .map(|(index, _)| {
            u16::try_from(index).map_err(|_| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "marker_index_convert",
            })
        })
        .collect()
}

fn commit_region(
    variant: &PrefabVariant,
    variant_index: u16,
    layer: u16,
    origin_x: u16,
    origin_y: u16,
    role: RegionRole,
    grid: &mut OccupancyGrid,
    config: &NormalizedGeneratorConfig,
    alloc: &mut IdAllocator,
) -> Result<PlacedRegion, GeneratorError> {
    if !can_place_footprint(variant, layer, origin_x, origin_y, grid, config)? {
        return Err(GeneratorError::OccupancyConflict {
            stage: ErrorStage::Placement,
            detail: "region_footprint_unavailable".into(),
        });
    }
    let mut staged_grid = grid.clone();
    let mut staged_alloc = alloc.clone();
    let region_id = staged_alloc.next_region()?;
    let mut sockets = Vec::with_capacity(variant.sockets.len());
    for (index, socket) in variant.sockets.iter().enumerate() {
        let global_layer = layer.checked_add(socket.anchor.layer).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "region_socket_layer",
            },
        )?;
        let global_x = origin_x.checked_add(socket.anchor.x).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "region_socket_x",
            },
        )?;
        let global_y = origin_y.checked_add(socket.anchor.y).ok_or(
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "region_socket_y",
            },
        )?;
        sockets.push(PlacedSocket {
            id: staged_alloc.next_socket()?,
            variant_socket_index: u16::try_from(index).map_err(|_| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "region_socket_index",
                }
            })?,
            global_anchor: GridCoord::new(
                global_layer,
                global_x,
                global_y,
                config.width(),
                config.height(),
                config.layers().2,
            )?,
            direction: map_direction_from_variant(socket.direction),
            width: socket.width,
            role: map_socket_role_from_variant(socket.role),
            paired_socket_id: None,
        });
    }

    staged_grid.reserve_rect(
        layer,
        origin_x,
        origin_y,
        variant.width,
        variant.height,
        OccupancyClass::Region(region_id.raw()),
    )?;
    for socket in &sockets {
        match staged_grid.get(socket.global_anchor) {
            Some(OccupancyClass::Region(owner)) if owner == region_id.raw() => {
                staged_grid.set(
                    socket.global_anchor,
                    OccupancyClass::Socket(socket.id.raw()),
                )?;
            }
            other => {
                return Err(GeneratorError::OccupancyConflict {
                    stage: ErrorStage::Placement,
                    detail: format!(
                        "socket_anchor_not_owned {} existing={:?}",
                        socket.global_anchor, other
                    ),
                });
            }
        }
    }

    let spacing = i32::try_from(config.spacing()).map_err(|_| {
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "spacing_convert",
        }
    })?;
    let min_x = i32::from(origin_x)
        .checked_sub(spacing)
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "spacing_min_x",
        })?
        .max(0);
    let min_y = i32::from(origin_y)
        .checked_sub(spacing)
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "spacing_min_y",
        })?
        .max(0);
    let max_x = (i32::from(origin_x)
        .checked_add(i32::from(variant.width))
        .and_then(|value| value.checked_add(spacing))
        .and_then(|value| value.checked_sub(1))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "spacing_max_x",
        })?)
    .min(
        i32::from(config.width())
            .checked_sub(1)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "spacing_grid_max_x",
            })?,
    );
    let max_y = (i32::from(origin_y)
        .checked_add(i32::from(variant.height))
        .and_then(|value| value.checked_add(spacing))
        .and_then(|value| value.checked_sub(1))
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "spacing_max_y",
        })?)
    .min(
        i32::from(config.height())
            .checked_sub(1)
            .ok_or(GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "spacing_grid_max_y",
            })?,
    );
    for cell_y in min_y..=max_y {
        for cell_x in min_x..=max_x {
            let cell = GridCoord::new(
                layer,
                u16::try_from(cell_x).map_err(|_| GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "spacing_x_convert",
                })?,
                u16::try_from(cell_y).map_err(|_| GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "spacing_y_convert",
                })?,
                config.width(),
                config.height(),
                config.layers().2,
            )?;
            if staged_grid.get(cell) == Some(OccupancyClass::Empty) {
                staged_grid.set(cell, OccupancyClass::Spacing(region_id.raw()))?;
            }
        }
    }

    let region = PlacedRegion {
        id: region_id,
        role,
        variant_index,
        layer,
        footprint: (origin_x, origin_y, variant.width, variant.height),
        sockets,
        transitions: Vec::new(),
        marker_variant_indices: marker_indices_for_layer(variant, 0)?,
    };
    *grid = staged_grid;
    *alloc = staged_alloc;
    Ok(region)
}

fn place_role_regions(
    role: RegionRole,
    target_count: u32,
    candidates: &[(u16, &PrefabVariant)],
    config: &NormalizedGeneratorConfig,
    grid: &mut OccupancyGrid,
    placed: &mut Vec<PlacedRegion>,
    factory: SemanticStreamFactory,
    alloc: &mut IdAllocator,
) -> Result<(), GeneratorError> {
    for ordinal in 0..target_count {
        let mut scored = Vec::new();
        for layer in 0..config.layers().2 {
            for (variant_index, variant) in candidates.iter().copied() {
                if variant.width > config.width() || variant.height > config.height() {
                    continue;
                }
                let max_x = config.width().checked_sub(variant.width).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Placement,
                        operation: "placement_max_x",
                    },
                )?;
                let max_y = config.height().checked_sub(variant.height).ok_or(
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Placement,
                        operation: "placement_max_y",
                    },
                )?;
                for y in 0..=max_y {
                    for x in 0..=max_x {
                        if can_place_footprint(variant, layer, x, y, grid, config)? {
                            scored.push(PlacementCandidate {
                                variant_index,
                                layer,
                                x,
                                y,
                                score: placement_score(
                                    role,
                                    layer,
                                    x,
                                    y,
                                    variant_index,
                                    placed,
                                )?,
                            });
                        }
                    }
                }
            }
        }
        if scored.is_empty() {
            return Err(GeneratorError::PlacementExhausted {
                stage: ErrorStage::Placement,
                reason: "no_valid_placement",
                attempted: u64::from(config.placement_attempts()),
                placed: u64::from(ordinal),
                target: u64::from(target_count),
            });
        }
        scored.sort_by(|left, right| {
            right
                .score
                .separation_distance
                .cmp(&left.score.separation_distance)
                .then_with(|| left.score.layer_population.cmp(&right.score.layer_population))
                .then_with(|| {
                    right
                        .score
                        .required_route_distance
                        .cmp(&left.score.required_route_distance)
                })
                .then_with(|| left.score.variant_usage.cmp(&right.score.variant_usage))
                .then_with(|| left.variant_index.cmp(&right.variant_index))
                .then_with(|| left.layer.cmp(&right.layer))
                .then_with(|| left.y.cmp(&right.y))
                .then_with(|| left.x.cmp(&right.x))
        });
        let best_score = scored
            .first()
            .map(|candidate| candidate.score)
            .ok_or(GeneratorError::IrInvariant {
                stage: ErrorStage::Placement,
                detail: "sorted_placement_candidates_empty".into(),
            })?;
        let equal_count = scored
            .iter()
            .take_while(|candidate| candidate.score == best_score)
            .count();
        let chosen_index = if equal_count > 1 {
            let upper = NonZeroU32::new(u32::try_from(equal_count).map_err(|_| {
                GeneratorError::InvalidRngRange {
                    stage: ErrorStage::Placement,
                    reason: "placement_tie_count_unrepresentable",
                    lower: 0,
                    upper: u64::MAX,
                }
            })?)
            .ok_or(GeneratorError::InvalidRngRange {
                stage: ErrorStage::Placement,
                reason: "placement_tie_empty",
                lower: 0,
                upper: 0,
            })?;
            let mut stream = factory.stream(
                SemanticStage::Placement,
                &[
                    SemanticComponent::StableId(role.label().as_bytes()),
                    SemanticComponent::Index(ordinal),
                ],
            );
            usize::try_from(stream.gen_bounded(upper)).map_err(|_| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "placement_tie_index_convert",
                }
            })?
        } else {
            0
        };
        let chosen = scored
            .get(chosen_index)
            .ok_or(GeneratorError::IrInvariant {
                stage: ErrorStage::Placement,
                detail: "placement_choice_out_of_bounds".into(),
            })?;
        let variant = candidates
            .iter()
            .find(|(index, _)| *index == chosen.variant_index)
            .map(|(_, variant)| *variant)
            .ok_or(GeneratorError::IrInvariant {
                stage: ErrorStage::Placement,
                detail: "placement_variant_missing".into(),
            })?;
        let region = commit_region(
            variant,
            chosen.variant_index,
            chosen.layer,
            chosen.x,
            chosen.y,
            role,
            grid,
            config,
            alloc,
        )?;
        placed.push(region);
    }
    Ok(())
}

/// Place transition endpoints first, then ordinary role regions. Candidate
/// topology is deliberately not built here; it consumes this exact committed
/// occupancy grid through `topology::build_candidate_graph`.
pub(super) fn place_regions(
    config: &NormalizedGeneratorConfig,
    catalog: &PrefabCatalog,
    rng: &mut Pcg32V1,
    factory: SemanticStreamFactory,
) -> Result<(IntendedTopology, OccupancyGrid), GeneratorError> {
    let manifest = RoleManifest::from_config(config, rng)?;
    verify_mandatory_coverage(catalog, &manifest)?;
    let expected_total = manifest.total_regions()?;
    let mut grid = OccupancyGrid::new(config.width(), config.height(), config.layers().2)?;
    grid.reserve_borders();
    let mut alloc = IdAllocator::new();

    let ramp_candidates = enumerate_ramp_candidates(catalog, config)?;
    let (transitions, mut regions) = reserve_transitions_and_endpoints(
        &ramp_candidates,
        catalog,
        config,
        &mut grid,
        &mut alloc,
        factory,
    )?;
    let endpoint_count = u32::try_from(regions.len()).map_err(|_| {
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "endpoint_region_count_convert",
        }
    })?;
    if endpoint_count != manifest.vertical_hub_count {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Placement,
            constraint: "transition_endpoint_count",
            required: u64::from(manifest.vertical_hub_count),
            available: u64::from(endpoint_count),
        });
    }

    let mut roles = manifest.role_counts();
    roles.sort_by_key(|(role, _)| role.ordinal());
    for (role, count) in roles {
        if count == 0 || role == RegionRole::VerticalHub {
            continue;
        }
        let candidates = variants_for_role(catalog, role)?;
        place_role_regions(
            role,
            count,
            &candidates,
            config,
            &mut grid,
            &mut regions,
            factory,
            &mut alloc,
        )?;
    }
    regions.sort_by_key(|region| region.id.raw());
    let actual_total = u32::try_from(regions.len()).map_err(|_| {
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Placement,
            operation: "placed_region_count_convert",
        }
    })?;
    if actual_total != expected_total {
        return Err(GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Placement,
            constraint: "placed_region_total",
            required: u64::from(expected_total),
            available: u64::from(actual_total),
        });
    }

    let topology = IntendedTopology {
        regions,
        edges: Vec::new(),
        transitions,
        route_distance: 0,
        per_layer_cycles: vec![0; usize::from(config.layers().2)],
        max_branch_depth: 0,
        dead_end_count: 0,
        articulation_count: 0,
        crossing_count: 0,
        config: config.clone(),
    };
    topology.validate_unique_region_ids()?;
    topology.validate_transition_bindings()?;
    Ok((topology, grid))
}

fn map_direction_from_variant(direction: super::prefab::Direction) -> Direction {
    match direction {
        super::prefab::Direction::North => Direction::North,
        super::prefab::Direction::East => Direction::East,
        super::prefab::Direction::South => Direction::South,
        super::prefab::Direction::West => Direction::West,
    }
}

fn map_socket_role_from_variant(role: PrefabSocketRole) -> SocketRole {
    match role {
        PrefabSocketRole::Corridor => SocketRole::Corridor,
        PrefabSocketRole::Hall => SocketRole::Hall,
        PrefabSocketRole::Doorway => SocketRole::Doorway,
        PrefabSocketRole::Junction => SocketRole::Junction,
        PrefabSocketRole::DeadEnd => SocketRole::DeadEnd,
        PrefabSocketRole::LandmarkApproach => SocketRole::LandmarkApproach,
        PrefabSocketRole::LowerRampApproach => SocketRole::LowerRampApproach,
        PrefabSocketRole::UpperLanding => SocketRole::UpperLanding,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::*;
    use super::super::config::{GeneratorConfig, QualifiedProfile};
    use super::super::determinism::{AttemptIdentity, GeneratorIdentity};

    fn catalog() -> PrefabCatalog {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs");
        PrefabCatalog::load(&root).expect("bundled prefab catalog")
    }

    fn factory(
        config: &NormalizedGeneratorConfig,
        catalog: &PrefabCatalog,
        seed: u64,
    ) -> SemanticStreamFactory {
        let generator = GeneratorIdentity::new(config, catalog.identity_bytes(), seed);
        SemanticStreamFactory::new(AttemptIdentity::new(generator, 0))
    }

    #[test]
    fn role_manifest_allocates_two_regions_per_transition() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Primary)
            .normalize()
            .expect("primary config");
        let mut rng = Pcg32V1::new(1, 2);
        let manifest = RoleManifest::from_config(&config, &mut rng).expect("manifest");
        assert_eq!(manifest.vertical_hub_count, 8);
        let total = manifest.total_regions().expect("role total");
        assert!((config.region_min()..=config.region_max()).contains(&total));
    }

    #[test]
    fn transition_materialization_filters_by_transition_owner() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .expect("minimum config");
        let catalog = catalog();
        let variant_index = catalog
            .variants()
            .iter()
            .position(|variant| {
                variant.base_id == "ramp-hub-straight" && variant.rotation_degrees == 0
            })
            .expect("straight ramp variant");
        let candidate = RampCandidate {
            variant_index: u16::try_from(variant_index).expect("variant index"),
            transition_index: 0,
            lower_layer: 0,
            x: 10,
            y: 10,
        };
        let materialized =
            materialize_transition(candidate, &catalog, &config).expect("materialized transition");
        assert_eq!(materialized.ramp_run_cells.len(), 3);
        assert_eq!(materialized.upper_opening_cells.len(), 3);
        assert_eq!(materialized.landing_cells.len(), 1);
        assert!(materialized.lower_approach_cells.is_empty());
        assert!(materialized.lower_funnel_cells.is_empty());
    }

    #[test]
    fn transition_approach_is_hub_space_not_protected_transition_volume() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .expect("minimum config");
        let catalog = catalog();
        let variant_index = catalog
            .variants()
            .iter()
            .position(|variant| {
                variant.base_id == "ramp-hub-turn" && variant.rotation_degrees == 0
            })
            .expect("turn ramp variant");
        let materialized = materialize_transition(
            RampCandidate {
                variant_index: u16::try_from(variant_index).expect("variant index"),
                transition_index: 0,
                lower_layer: 0,
                x: 10,
                y: 10,
            },
            &catalog,
            &config,
        )
        .expect("materialized transition");
        assert!(!materialized.lower_approach_cells.is_empty());
        let approach = materialized.lower_approach_cells[0];
        let protected: BTreeMap<GridCoord, ()> = materialized
            .ramp_run_cells
            .iter()
            .chain(&materialized.upper_opening_cells)
            .chain(&materialized.landing_cells)
            .chain(&materialized.headroom_cells)
            .copied()
            .map(|cell| (cell, ()))
            .collect();
        assert!(!protected.contains_key(&approach));

        let mut grid = OccupancyGrid::new(
            config.width(),
            config.height(),
            config.layers().2,
        )
        .expect("grid");
        let mut allocator = IdAllocator::new();
        let (transition, _, _) = commit_transition(
            &materialized,
            &catalog,
            &mut grid,
            &mut allocator,
            &config,
        )
        .expect("transition commit");
        assert_eq!(
            grid.get(approach),
            Some(OccupancyClass::TransitionHub(transition.id.raw()))
        );
        for cell in materialized
            .ramp_run_cells
            .iter()
            .chain(&materialized.upper_opening_cells)
            .chain(&materialized.landing_cells)
            .chain(&materialized.headroom_cells)
        {
            assert_eq!(
                grid.get(*cell),
                Some(OccupancyClass::Transition(transition.id.raw()))
            );
        }
    }

    #[test]
    fn endpoint_regions_are_placed_first_and_keep_hub_ownership() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .expect("minimum config");
        let catalog = catalog();
        let factory = factory(&config, &catalog, 17);
        let mut role_rng = factory.stream(SemanticStage::Roles, &[]);
        let (topology, grid) =
            place_regions(&config, &catalog, &mut role_rng, factory).expect("placement");
        assert_eq!(topology.transitions.len(), 2);
        topology
            .validate_transition_bindings()
            .expect("transition bindings");
        for transition in &topology.transitions {
            assert!(transition.lower_region.raw() < 4);
            assert!(transition.upper_region.raw() < 4);
            let lower = topology
                .regions
                .iter()
                .find(|region| region.id == transition.lower_region)
                .expect("lower endpoint");
            let upper = topology
                .regions
                .iter()
                .find(|region| region.id == transition.upper_region)
                .expect("upper endpoint");
            assert_eq!(lower.transitions, vec![transition.id]);
            assert_eq!(upper.transitions, vec![transition.id]);
            let probe = GridCoord::new(
                transition.lower_layer,
                transition.hub_footprint.0,
                transition.hub_footprint.1,
                config.width(),
                config.height(),
                config.layers().2,
            )
            .expect("hub probe");
            assert!(matches!(
                grid.get(probe),
                Some(OccupancyClass::TransitionHub(owner))
                    | Some(OccupancyClass::Transition(owner))
                    if owner == transition.id.raw()
            ));
        }
    }

    #[test]
    fn placement_is_reproducible_through_real_entrypoint() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .expect("minimum config");
        let catalog = catalog();
        let factory = factory(&config, &catalog, 99);
        let mut rng_a = factory.stream(SemanticStage::Roles, &[]);
        let mut rng_b = factory.stream(SemanticStage::Roles, &[]);
        let a = place_regions(&config, &catalog, &mut rng_a, factory).expect("placement a");
        let b = place_regions(&config, &catalog, &mut rng_b, factory).expect("placement b");
        assert_eq!(a.0, b.0);
    }
}
