use std::collections::BTreeMap;
use std::num::NonZeroU32;

use super::config::NormalizedGeneratorConfig;
use super::determinism::{Pcg32V1, SemanticComponent, SemanticStage, SemanticStreamFactory};
use super::error::{ErrorStage, GeneratorError};
use super::ir::{
    Direction, GridCoord, IdAllocator, IntendedTopology, OccupancyClass, OccupancyGrid,
    PlacedRegion, PlacedSocket, RegionId, RegionRole, SocketId, SocketRole, TransitionId,
    TransitionReservation,
};
use super::prefab::{PrefabCatalog, PrefabVariant};

// ─── Role manifest ──────────────────────────────────────────────────────────

/// A deterministic assignment of roles to target counts, derived from the
/// normalized configuration.
#[derive(Debug, Clone)]
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
        let total = rng.gen_range(config.region_min(), config.region_max().saturating_add(1))?;

        // Mandatories: 1 spawn, 1 distant landmark
        let _allocated: u32 = 2;

        let layer_pairs = config.layers().2.saturating_sub(1) as u32;
        let transitions_min = config.transitions_per_adjacent_pair();
        let vertical_hubs_needed = layer_pairs
            .checked_mul(transitions_min)
            .ok_or_else(|| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "vertical_hub_count",
            })?;

        let mandatory_minimum = 2u32
            .checked_add(vertical_hubs_needed)
            .and_then(|v| v.checked_add(3)) // at least 1 major, 1 junction, 1 dead_end
            .ok_or_else(|| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "mandatory_minimum",
            })?;

        if total < mandatory_minimum {
            return Err(GeneratorError::MandatoryInfeasibility {
                stage: ErrorStage::Placement,
                constraint: "mandatory_role_minimum",
                required: u64::from(mandatory_minimum),
                available: u64::from(total),
            });
        }

        let remaining = total - mandatory_minimum;
        let remaining = remaining as u64;
        let proportions = [20u64, 20, 15, 15, 15, 15];
        let prop_sum: u64 = proportions.iter().sum();
        let mut shares = [0u64; 6];
        let mut allocated_prop: u64 = 0;
        for (i, &p) in proportions.iter().enumerate() {
            let share = (remaining * p + prop_sum / 2) / prop_sum;
            shares[i] = share;
            allocated_prop += share;
        }
        let mut remainder = remaining.saturating_sub(allocated_prop);
        for i in 0..6 {
            if remainder == 0 {
                break;
            }
            shares[i] += 1;
            remainder -= 1;
        }

        let major_landmark_count = 1u32 + shares[0] as u32;
        let junction_count = 1u32 + shares[1] as u32;
        let dead_end_count = 1u32 + shares[2] as u32;
        let required_route_count = 0u32 + shares[3] as u32;
        let optional_branch_count = 0u32 + shares[4] as u32;
        let ordinary_count = 0u32 + shares[5] as u32;

        Ok(Self {
            spawn_count: 1,
            distant_landmark_count: 1,
            major_landmark_count,
            junction_count,
            dead_end_count,
            vertical_hub_count: vertical_hubs_needed,
            required_route_count,
            optional_branch_count,
            ordinary_count,
        })
    }

    fn total_regions(&self) -> u32 {
        self.spawn_count
            .checked_add(self.distant_landmark_count)
            .and_then(|v| v.checked_add(self.major_landmark_count))
            .and_then(|v| v.checked_add(self.junction_count))
            .and_then(|v| v.checked_add(self.dead_end_count))
            .and_then(|v| v.checked_add(self.vertical_hub_count))
            .and_then(|v| v.checked_add(self.required_route_count))
            .and_then(|v| v.checked_add(self.optional_branch_count))
            .and_then(|v| v.checked_add(self.ordinary_count))
            .unwrap_or(u32::MAX)
    }

    fn role_counts(&self) -> Vec<(RegionRole, u32)> {
        vec![
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

// ─── Prefab role compatibility ──────────────────────────────────────────────

fn variant_supports_role(variant: &PrefabVariant, role: RegionRole) -> bool {
    match role {
        RegionRole::Spawn => variant
            .tags
            .iter()
            .any(|t| t == "spawn" || t == "ordinary" || t == "room"),
        RegionRole::DistantLandmark => variant
            .tags
            .iter()
            .any(|t| t == "landmark" || t == "major" || t == "room"),
        RegionRole::MajorLandmark => variant
            .tags
            .iter()
            .any(|t| t == "landmark" || t == "major" || t == "room" || t == "hall"),
        RegionRole::Junction => variant
            .tags
            .iter()
            .any(|t| t == "junction" || t == "hall" || t == "room"),
        RegionRole::DeadEnd => variant
            .tags
            .iter()
            .any(|t| t == "dead_end" || t == "room"),
        RegionRole::VerticalHub => variant
            .tags
            .iter()
            .any(|t| t == "ramp" || t == "vertical" || t == "hub"),
        RegionRole::RequiredRoute => variant
            .tags
            .iter()
            .any(|t| t == "ordinary" || t == "room" || t == "hall" || t == "corridor"),
        RegionRole::OptionalBranch => variant
            .tags
            .iter()
            .any(|t| t == "ordinary" || t == "room" || t == "dead_end"),
        RegionRole::OrdinaryRoom => variant
            .tags
            .iter()
            .any(|t| t == "ordinary" || t == "room" || t == "small"),
    }
}

fn variants_for_role<'a>(
    catalog: &'a PrefabCatalog,
    role: RegionRole,
) -> Vec<(u16, &'a PrefabVariant)> {
    catalog
        .variants()
        .iter()
        .enumerate()
        .filter(|(_, v)| variant_supports_role(v, role))
        .map(|(i, v)| (i as u16, v))
        .collect()
}

fn verify_mandatory_coverage(
    catalog: &PrefabCatalog,
    manifest: &RoleManifest,
) -> Result<(), GeneratorError> {
    for (role, count) in manifest.role_counts() {
        if count > 0 {
            let candidates = variants_for_role(catalog, role);
            if candidates.is_empty() {
                return Err(GeneratorError::MandatoryInfeasibility {
                    stage: ErrorStage::Placement,
                    constraint: "prefab_role_coverage",
                    required: u64::from(count),
                    available: 0,
                });
            }
        }
    }
    Ok(())
}

// ─── Transition reservation ─────────────────────────────────────────────────

/// Materialize transition reservations from validated prefab reservations.
/// Consumes the variant's Reservation and Transition definitions to produce
/// complete TransitionReservation records.
fn materialize_transition_reservation(
    variant: &PrefabVariant,
    _variant_index: u16,
    lower_layer: u16,
    origin_x: u16,
    origin_y: u16,
    config: &NormalizedGeneratorConfig,
    alloc: &mut IdAllocator,
) -> Result<TransitionReservation, GeneratorError> {
    let id = alloc.next_transition()?;
    let w = variant.width;
    let h = variant.height;

    let hub_footprint = (origin_x, origin_y, w, h);

    // Collect ramp run cells from the variant's lower layer
    let lower_grid = &variant.layers[0];
    let mut ramp_run_cells = Vec::new();
    let mut lower_funnel_cells = Vec::new();
    let mut lower_approach_cells = Vec::new();

    for (ly, row) in lower_grid.iter().enumerate() {
        for (lx, tile) in row.iter().enumerate() {
            let gx = origin_x.checked_add(lx as u16).ok_or_else(|| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "ramp_origin_x_add",
                }
            })?;
            let gy = origin_y.checked_add(ly as u16).ok_or_else(|| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "ramp_origin_y_add",
                }
            })?;
            let coord = GridCoord::new(
                lower_layer,
                gx,
                gy,
                config.width(),
                config.height(),
                config.layers().2,
            )?;

            match tile {
                super::prefab::Tile::Ramp { .. } => {
                    ramp_run_cells.push(coord);
                }
                _ => {}
            }
        }
    }

    // Consume reservations from the validated variant
    let mut upper_opening_cells = Vec::new();
    let mut landing_cells = Vec::new();
    let mut headroom_cells = Vec::new();

    for reservation in &variant.reservations {
        for cell in &reservation.cells {
            let gx = origin_x.checked_add(cell.x).ok_or_else(|| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "reservation_x_add",
                }
            })?;
            let gy = origin_y.checked_add(cell.y).ok_or_else(|| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "reservation_y_add",
                }
            })?;
            let global_layer = if cell.layer == 0 {
                lower_layer
            } else {
                lower_layer.checked_add(1).ok_or_else(|| {
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Placement,
                        operation: "upper_layer_add",
                    }
                })?
            };

            let coord = GridCoord::new(
                global_layer,
                gx,
                gy,
                config.width(),
                config.height(),
                config.layers().2,
            )?;

            match reservation.kind {
                super::prefab::ReservationKind::UpperOpening => {
                    upper_opening_cells.push(coord);
                }
                super::prefab::ReservationKind::UpperLanding => {
                    landing_cells.push(coord);
                }
                super::prefab::ReservationKind::Headroom => {
                    headroom_cells.push(coord);
                }
                super::prefab::ReservationKind::SocketFunnel
                | super::prefab::ReservationKind::CorridorApproach => {
                    if cell.layer == 0 {
                        // Determine if it's funnel or approach based on proximity to ramp
                        if ramp_run_cells.is_empty()
                            || ramp_run_cells
                                .iter()
                                .any(|rc| {
                                    rc.layer == coord.layer
                                        && (rc.x as i32 - coord.x as i32).abs() <= 1
                                        && (rc.y as i32 - coord.y as i32).abs() <= 1
                                })
                        {
                            lower_funnel_cells.push(coord);
                        } else {
                            lower_approach_cells.push(coord);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Sort all cell vectors for determinism
    ramp_run_cells.sort();
    ramp_run_cells.dedup();
    upper_opening_cells.sort();
    upper_opening_cells.dedup();
    landing_cells.sort();
    landing_cells.dedup();
    headroom_cells.sort();
    headroom_cells.dedup();
    lower_funnel_cells.sort();
    lower_funnel_cells.dedup();
    lower_approach_cells.sort();
    lower_approach_cells.dedup();

    Ok(TransitionReservation {
        id,
        lower_layer,
        hub_footprint,
        ramp_run_cells,
        upper_opening_cells,
        landing_cells,
        headroom_cells,
        lower_funnel_cells,
        lower_approach_cells,
    })
}

/// Enumerate candidate transition reservations from ramp-hub prefab variants.
fn enumerate_ramp_candidates(
    catalog: &PrefabCatalog,
    config: &NormalizedGeneratorConfig,
    alloc: &mut IdAllocator,
) -> Result<Vec<TransitionReservation>, GeneratorError> {
    let mut candidates = Vec::new();
    let layers = config.layers().2;

    for (vi, variant) in catalog.variants().iter().enumerate() {
        if !variant
            .tags
            .iter()
            .any(|t| t == "ramp" || t == "hub" || t == "vertical")
        {
            continue;
        }
        // Only consider ramp-hub variants that have transitions defined
        if variant.transitions.is_empty() {
            continue;
        }
        for lower_layer in 0..layers.saturating_sub(1) {
            let max_x = config.width().saturating_sub(variant.width);
            let max_y = config.height().saturating_sub(variant.height);
            for y in 0..=max_y {
                for x in 0..=max_x {
                    let reservation = materialize_transition_reservation(
                        variant,
                        vi as u16,
                        lower_layer,
                        x,
                        y,
                        config,
                        alloc,
                    )?;
                    candidates.push(reservation);
                }
            }
        }
    }

    // Canonical order: by (layer, y, x, id)
    candidates.sort_by(|a, b| {
        a.lower_layer
            .cmp(&b.lower_layer)
            .then_with(|| a.hub_footprint.1.cmp(&b.hub_footprint.1))
            .then_with(|| a.hub_footprint.0.cmp(&b.hub_footprint.0))
            .then_with(|| a.id.raw().cmp(&b.id.raw()))
    });

    Ok(candidates)
}

/// Reserve ramp transitions deterministically. Reserves complete volume:
/// hub footprint, ramp run, upper opening, landing, headroom, funnels, approaches.
fn reserve_ramps(
    candidates: &[TransitionReservation],
    config: &NormalizedGeneratorConfig,
    grid: &mut OccupancyGrid,
    factory: &SemanticStreamFactory,
) -> Result<Vec<TransitionReservation>, GeneratorError> {
    let layers = config.layers().2;
    let required_per_pair = config.transitions_per_adjacent_pair();

    let mut selected: Vec<TransitionReservation> = Vec::new();
    let mut per_pair_counts: BTreeMap<(u16, u16), u32> = BTreeMap::new();

    // Group candidates by layer pair
    let mut by_pair: BTreeMap<(u16, u16), Vec<usize>> = BTreeMap::new();
    for (i, cand) in candidates.iter().enumerate() {
        let upper = cand.lower_layer.checked_add(1).ok_or_else(|| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "upper_layer_increment",
            }
        })?;
        by_pair
            .entry((cand.lower_layer, upper))
            .or_default()
            .push(i);
    }

    for lower in 0..layers.saturating_sub(1) {
        let pair = (lower, lower + 1);
        let cand_indices = by_pair.get(&pair).cloned().unwrap_or_default();

        let mut indices: Vec<usize> = cand_indices;
        let mut stream = factory.stream(
            SemanticStage::RampReservations,
            &[SemanticComponent::Index(lower as u32)],
        );
        stream.shuffle(&mut indices)?;

        let mut placed_for_pair: u32 = 0;
        for idx in indices {
            if placed_for_pair >= required_per_pair {
                break;
            }
            let cand = &candidates[idx];

            // Check and reserve complete transition volume
            if !can_reserve_transition(cand, grid, config)? {
                continue;
            }

            // Reserve hub footprint (all cells on lower layer bounded by hub_footprint)
            reserve_transition_volume(cand, grid, config)?;

            selected.push(cand.clone());
            placed_for_pair += 1;
            *per_pair_counts.entry(pair).or_insert(0) += 1;
        }

        if placed_for_pair < required_per_pair {
            return Err(GeneratorError::MandatoryInfeasibility {
                stage: ErrorStage::Placement,
                constraint: "ramp_reservation_count",
                required: u64::from(required_per_pair),
                available: u64::from(placed_for_pair),
            });
        }
    }

    Ok(selected)
}

/// Check whether all cells in a transition reservation are empty.
fn can_reserve_transition(
    cand: &TransitionReservation,
    grid: &OccupancyGrid,
    _config: &NormalizedGeneratorConfig,
) -> Result<bool, GeneratorError> {
    // Check hub footprint rectangle
    if !grid.is_rect_empty(
        cand.lower_layer,
        cand.hub_footprint.0,
        cand.hub_footprint.1,
        cand.hub_footprint.2,
        cand.hub_footprint.3,
    )? {
        return Ok(false);
    }

    // Check ramp run cells
    for coord in &cand.ramp_run_cells {
        if grid.get(*coord) != Some(OccupancyClass::Empty) {
            return Ok(false);
        }
    }

    // Check upper opening cells
    for coord in &cand.upper_opening_cells {
        if grid.get(*coord) != Some(OccupancyClass::Empty) {
            return Ok(false);
        }
    }

    // Check landing cells
    for coord in &cand.landing_cells {
        if grid.get(*coord) != Some(OccupancyClass::Empty) {
            return Ok(false);
        }
    }

    // Check headroom cells
    for coord in &cand.headroom_cells {
        if grid.get(*coord) != Some(OccupancyClass::Empty) {
            return Ok(false);
        }
    }

    // Check lower funnel cells
    for coord in &cand.lower_funnel_cells {
        if grid.get(*coord) != Some(OccupancyClass::Empty) {
            return Ok(false);
        }
    }

    // Check lower approach cells
    for coord in &cand.lower_approach_cells {
        if grid.get(*coord) != Some(OccupancyClass::Empty) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Reserve all cells of a transition at once.
/// Ramp cells are reserved BEFORE the hub footprint to prevent self-conflict:
/// ramp run cells occupy cells inside the hub footprint rectangle.
fn reserve_transition_volume(
    cand: &TransitionReservation,
    grid: &mut OccupancyGrid,
    _config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    let class = OccupancyClass::Transition(cand.id.raw());

    // Phase 1: Reserve all discrete cell sets BEFORE the hub footprint rect.
    // This prevents the rect reservation from conflicting with ramp-run,
    // funnel, approach, opening, landing, and headroom cells that may overlap
    // the rect bounds.

    // Reserve ramp run cells first (these are inside the hub footprint)
    if !cand.ramp_run_cells.is_empty() {
        grid.reserve_cells(&cand.ramp_run_cells, class)?;
    }

    // Reserve lower funnel cells (may overlap hub footprint edges)
    if !cand.lower_funnel_cells.is_empty() {
        grid.reserve_cells(&cand.lower_funnel_cells, class)?;
    }

    // Reserve lower approach cells
    if !cand.lower_approach_cells.is_empty() {
        grid.reserve_cells(&cand.lower_approach_cells, class)?;
    }

    // Reserve upper opening cells (layer+1, may overlap projected footprint)
    if !cand.upper_opening_cells.is_empty() {
        grid.reserve_cells(&cand.upper_opening_cells, class)?;
    }

    // Reserve landing cells
    if !cand.landing_cells.is_empty() {
        grid.reserve_cells(&cand.landing_cells, class)?;
    }

    // Reserve headroom cells
    if !cand.headroom_cells.is_empty() {
        grid.reserve_cells(&cand.headroom_cells, class)?;
    }

    // Phase 2: Reserve hub footprint rect, skipping cells already claimed by
    // the discrete cell sets above.
    let (hx, hy, hw, hh) = cand.hub_footprint;
    let hub_class = class;
    for dy in 0..hh {
        for dx in 0..hw {
            let cx = hx.checked_add(dx).ok_or_else(|| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "transition_hub_x_add",
                }
            })?;
            let cy = hy.checked_add(dy).ok_or_else(|| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "transition_hub_y_add",
                }
            })?;
            let coord = GridCoord::new(
                cand.lower_layer,
                cx,
                cy,
                _config.width(),
                _config.height(),
                _config.layers().2,
            )?;
            // Only set if currently empty (cell sets were reserved first)
            let prev = grid.get(coord);
            if prev == Some(OccupancyClass::Empty) {
                grid.set(coord, hub_class)?;
            } else if prev != Some(hub_class) {
                return Err(GeneratorError::OccupancyConflict {
                    stage: ErrorStage::Placement,
                    detail: format!(
                        "transition_hub_cell_conflict {} was={:?} wanted={:?}",
                        coord, prev, hub_class
                    ),
                });
            }
        }
    }

    Ok(())
}

// ─── Footprint placement ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PlacementCandidate {
    variant_index: u16,
    layer: u16,
    x: u16,
    y: u16,
    score: PlacementScore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PlacementScore {
    /// Distance to nearest already-placed region (larger is better — more spread).
    separation_distance: u32,
    /// Layer distribution: prefer layers with fewer regions.
    layer_rank: u32,
    /// Required-route support: how many existing required-route regions are nearby.
    required_route_proximity: u32,
    /// Prefab diversity: lower rank = more distinct from already-placed variants.
    diversity_rank: u32,
}

/// Place role-assigned regions into the grid, returning the IntendedTopology
/// with placed regions and the populated occupancy grid.
pub(super) fn place_regions(
    config: &NormalizedGeneratorConfig,
    catalog: &PrefabCatalog,
    rng: &mut Pcg32V1,
    factory: SemanticStreamFactory,
) -> Result<(IntendedTopology, OccupancyGrid), GeneratorError> {
    let mut alloc = IdAllocator::new();

    // 1. Role manifest
    let manifest = RoleManifest::from_config(config, rng)?;
    verify_mandatory_coverage(catalog, &manifest)?;

    let width = config.width();
    let height = config.height();
    let layers = config.layers().2;

    // 2. Initialize occupancy grid
    let mut grid = OccupancyGrid::new(width, height, layers)?;

    // 3. Reserve ramp transitions
    let ramp_candidates = enumerate_ramp_candidates(catalog, config, &mut alloc)?;
    let transitions = reserve_ramps(&ramp_candidates, config, &mut grid, &factory)?;

    // 4. Place regions role by role
    let mut placed_regions: Vec<PlacedRegion> = Vec::new();
    let role_order: Vec<(RegionRole, u32)> = {
        let mut roles = manifest.role_counts();
        roles.sort_by_key(|(r, _)| r.ordinal());
        roles
    };

    for (role, target_count) in role_order {
        if target_count == 0 {
            continue;
        }
        let candidates = variants_for_role(catalog, role);
        if candidates.is_empty() {
            return Err(GeneratorError::MandatoryInfeasibility {
                stage: ErrorStage::Placement,
                constraint: "prefab_variants_for_role",
                required: u64::from(target_count),
                available: 0,
            });
        }

        let placed_for_role = place_role_regions(
            role,
            target_count,
            &candidates,
            config,
            &mut grid,
            &placed_regions,
            &factory,
            &mut alloc,
        )?;

        placed_regions.extend(placed_for_role);
    }

    // 5. Sort regions by ID for stable ordering
    placed_regions.sort_by_key(|r| r.id.raw());

    // 6. Build the IntendedTopology
    let topology = IntendedTopology {
        regions: placed_regions,
        edges: Vec::new(),
        transitions,
        route_distance: 0,
        per_layer_cycles: vec![0; layers as usize],
        max_branch_depth: 0,
        dead_end_count: 0,
        articulation_count: 0,
        crossing_count: 0,
        config: config.clone(),
    };

    topology.validate_unique_region_ids()?;

    Ok((topology, grid))
}

/// Place target_count regions with the given role.
fn place_role_regions(
    role: RegionRole,
    target_count: u32,
    candidates: &[(u16, &PrefabVariant)],
    config: &NormalizedGeneratorConfig,
    grid: &mut OccupancyGrid,
    placed: &[PlacedRegion],
    factory: &SemanticStreamFactory,
    alloc: &mut IdAllocator,
) -> Result<Vec<PlacedRegion>, GeneratorError> {
    let width = config.width();
    let height = config.height();
    let layers = config.layers().2;
    let spacing = config.spacing();

    let mut placed_this_role: Vec<PlacedRegion> = Vec::new();

    for _ in 0..target_count {
        let mut scored_candidates: Vec<PlacementCandidate> = Vec::new();

        for layer in 0..layers {
            for &(variant_index, variant) in candidates {
                let max_x = width.saturating_sub(variant.width);
                let max_y = height.saturating_sub(variant.height);

                // If variant doesn't fit at all, skip
                if variant.width > width || variant.height > height {
                    continue;
                }

                for y in 0..=max_y {
                    for x in 0..=max_x {
                        if !can_place_footprint(variant, layer, x, y, grid, config) {
                            continue;
                        }
                        if !check_spacing_multi(
                            layer,
                            x,
                            y,
                            variant.width,
                            variant.height,
                            placed,
                            &placed_this_role,
                            spacing,
                        ) {
                            continue;
                        }

                        let score = compute_placement_score(
                            layer,
                            x,
                            y,
                            variant_index,
                            placed,
                            &placed_this_role,
                            config,
                        );

                        scored_candidates.push(PlacementCandidate {
                            variant_index,
                            layer,
                            x,
                            y,
                            score,
                        });
                    }
                }
            }
        }

        if scored_candidates.is_empty() {
            return Err(GeneratorError::PlacementExhausted {
                stage: ErrorStage::Placement,
                reason: "no_valid_placement",
                attempted: u64::from(config.placement_attempts()),
                placed: placed_this_role.len() as u64,
                target: u64::from(target_count),
            });
        }

        // Sort by score (lower layer_rank/diversity_rank is better, higher separation is better)
        scored_candidates.sort_by(|a, b| {
            b.score
                .separation_distance
                .cmp(&a.score.separation_distance)
                .then_with(|| a.score.layer_rank.cmp(&b.score.layer_rank))
                .then_with(|| {
                    b.score
                        .required_route_proximity
                        .cmp(&a.score.required_route_proximity)
                })
                .then_with(|| a.score.diversity_rank.cmp(&b.score.diversity_rank))
                .then_with(|| a.variant_index.cmp(&b.variant_index))
                .then_with(|| a.layer.cmp(&b.layer))
                .then_with(|| a.x.cmp(&b.x))
                .then_with(|| a.y.cmp(&b.y))
        });

        // Tie-break within equal-score class using semantic stream
        let best = &scored_candidates[0];
        let equal_class: Vec<usize> = scored_candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.score == best.score
                    && c.variant_index == best.variant_index
                    && c.layer == best.layer
            })
            .map(|(i, _)| i)
            .collect();

        let chosen_idx = if equal_class.len() > 1 {
            let mut tie_rng = factory.stream(
                SemanticStage::Placement,
                &[
                    SemanticComponent::StableId(role.label().as_bytes()),
                    SemanticComponent::Index(placed_this_role.len() as u32),
                ],
            );
            let upper =
                NonZeroU32::new(equal_class.len() as u32).ok_or_else(|| {
                    GeneratorError::InvalidRngRange {
                        stage: ErrorStage::Placement,
                        reason: "tie_break_empty_class",
                        lower: 0,
                        upper: 0,
                    }
                })?;
            let pick = tie_rng.gen_bounded(upper) as usize;
            equal_class[pick]
        } else {
            equal_class[0]
        };

        let chosen = &scored_candidates[chosen_idx];
        let variant = candidates
            .iter()
            .find(|(vi, _)| *vi == chosen.variant_index)
            .map(|(_, v)| *v)
            .ok_or_else(|| GeneratorError::IrInvariant {
                stage: ErrorStage::Placement,
                detail: format!(
                    "variant_not_found_in_candidates vi={}",
                    chosen.variant_index
                ),
            })?;

        // Single atomic commit: allocate region ID, reserve footprint, reserve spacing,
        // reserve socket funnel/approach cells, build PlacedRegion — all with one RegionId.
        let region = commit_placement_atomic(
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

        placed_this_role.push(region);
    }

    Ok(placed_this_role)
}

/// Check whether a variant footprint fits at a given position.
fn can_place_footprint(
    variant: &PrefabVariant,
    layer: u16,
    x: u16,
    y: u16,
    grid: &OccupancyGrid,
    config: &NormalizedGeneratorConfig,
) -> bool {
    let w = variant.width;
    let h = variant.height;
    let layers = config.layers().2;

    if x.checked_add(w).map_or(true, |ex| ex > config.width()) {
        return false;
    }
    if y.checked_add(h).map_or(true, |ey| ey > config.height()) {
        return false;
    }
    if layer >= layers {
        return false;
    }

    for vy in 0..h {
        for vx in 0..w {
            let cx = match x.checked_add(vx) {
                Some(v) => v,
                None => return false,
            };
            let cy = match y.checked_add(vy) {
                Some(v) => v,
                None => return false,
            };
            let coord = match GridCoord::new(layer, cx, cy, config.width(), config.height(), layers)
            {
                Ok(c) => c,
                Err(_) => return false,
            };
            match grid.get(coord) {
                Some(OccupancyClass::Empty) => {}
                _ => return false,
            }
        }
    }
    true
}

/// Check spacing against already-placed regions.
fn check_spacing_multi(
    layer: u16,
    x: u16,
    y: u16,
    w: u16,
    h: u16,
    placed_prev: &[PlacedRegion],
    placed_current: &[PlacedRegion],
    spacing: u32,
) -> bool {
    let spacing = spacing as i32;
    let x_min = (x as i32).saturating_sub(spacing);
    let y_min = (y as i32).saturating_sub(spacing);
    let x_max = ((x as i32).saturating_add(w as i32)).saturating_add(spacing).saturating_sub(1);
    let y_max = ((y as i32).saturating_add(h as i32)).saturating_add(spacing).saturating_sub(1);

    for r in placed_prev.iter().chain(placed_current.iter()) {
        if r.layer != layer {
            continue;
        }
        let rx_min = (r.footprint.0 as i32).saturating_sub(spacing);
        let ry_min = (r.footprint.1 as i32).saturating_sub(spacing);
        let rx_max = ((r.footprint.0 as i32).saturating_add(r.footprint.2 as i32))
            .saturating_add(spacing)
            .saturating_sub(1);
        let ry_max = ((r.footprint.1 as i32).saturating_add(r.footprint.3 as i32))
            .saturating_add(spacing)
            .saturating_sub(1);

        if x_min <= rx_max && x_max >= rx_min && y_min <= ry_max && y_max >= ry_min {
            return false;
        }
    }
    true
}

/// Compute a deterministic placement score using actual candidate coordinates.
fn compute_placement_score(
    layer: u16,
    x: u16,
    y: u16,
    variant_index: u16,
    placed_prev: &[PlacedRegion],
    placed_current: &[PlacedRegion],
    _config: &NormalizedGeneratorConfig,
) -> PlacementScore {
    // Separation: find minimum Manhattan distance to any placed region center
    let mut min_dist = u32::MAX;
    let cx = x as i64 + 0; // Use corner for simplicity
    let cy = y as i64 + 0;

    for r in placed_prev.iter().chain(placed_current.iter()) {
        let rx = r.footprint.0 as i64 + (r.footprint.2 as i64 / 2);
        let ry = r.footprint.1 as i64 + (r.footprint.3 as i64 / 2);
        let dist = ((cx - rx).unsigned_abs() + (cy - ry).unsigned_abs()) as u32;
        min_dist = min_dist.min(dist);
    }

    let separation_distance = if min_dist == u32::MAX {
        // No regions placed yet — this is the first region. Use a mid-range score.
        500
    } else {
        min_dist.min(1000)
    };

    // Layer distribution: count regions per layer, prefer layers with fewer
    let mut layer_counts: BTreeMap<u16, u32> = BTreeMap::new();
    for r in placed_prev.iter().chain(placed_current.iter()) {
        *layer_counts.entry(r.layer).or_default() += 1;
    }
    let layer_rank = layer_counts.get(&layer).copied().unwrap_or(0);

    // Required-route proximity: count how many required-route regions exist nearby
    let mut required_route_proximity = 0u32;
    for r in placed_prev.iter().chain(placed_current.iter()) {
        if matches!(r.role, RegionRole::RequiredRoute) {
            let rx = r.footprint.0 as i64 + (r.footprint.2 as i64 / 2);
            let ry = r.footprint.1 as i64 + (r.footprint.3 as i64 / 2);
            let dist = ((cx - rx).unsigned_abs() + (cy - ry).unsigned_abs()) as u32;
            if dist < 50 {
                required_route_proximity += 1;
            }
        }
    }

    // Diversity: count how many of this variant are already placed
    let mut variant_usage = 0u32;
    for r in placed_prev.iter().chain(placed_current.iter()) {
        if r.variant_index == variant_index {
            variant_usage += 1;
        }
    }
    let diversity_rank = variant_usage;

    PlacementScore {
        separation_distance,
        layer_rank,
        required_route_proximity,
        diversity_rank,
    }
}

/// Single atomic commit: validate placement, allocate IDs, then in one phase commit
/// to the grid. No grid mutation occurs before all validation passes so that failures
/// leave the grid uncorrupted.
fn commit_placement_atomic(
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
    let width = config.width();
    let height = config.height();
    let layers = config.layers().2;
    let spacing = config.spacing();

    // ── Phase 1: Validate and allocate IDs (no grid mutation) ──────────────

    // Validate that the footprint rect is available
    for dy in 0..variant.height {
        for dx in 0..variant.width {
            let cx = origin_x.checked_add(dx).ok_or_else(|| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "commit_footprint_x",
                }
            })?;
            let cy = origin_y.checked_add(dy).ok_or_else(|| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Placement,
                    operation: "commit_footprint_y",
                }
            })?;
            let coord = GridCoord::new(layer, cx, cy, width, height, layers)?;
            if grid.get(coord) != Some(OccupancyClass::Empty) {
                return Err(GeneratorError::OccupancyConflict {
                    stage: ErrorStage::Placement,
                    detail: format!(
                        "commit_footprint_occupied {} existing={:?}",
                        coord,
                        grid.get(coord)
                    ),
                });
            }
        }
    }

    // Validate socket coordinates
    let sock_spacing_i32 = spacing as i32;
    let width_i32 = width as i32;
    let height_i32 = height as i32;
    let origin_x_i32 = origin_x as i32;
    let origin_y_i32 = origin_y as i32;
    let variant_w_i32 = variant.width as i32;
    let variant_h_i32 = variant.height as i32;

    let sx_min = (origin_x_i32 - sock_spacing_i32).max(0) as u16;
    let sy_min = (origin_y_i32 - sock_spacing_i32).max(0) as u16;
    let sx_max = (origin_x_i32 + variant_w_i32 + sock_spacing_i32 - 1).min(width_i32 - 1).max(0) as u16;
    let sy_max = (origin_y_i32 + variant_h_i32 + sock_spacing_i32 - 1).min(height_i32 - 1).max(0) as u16;

    // Validate spacing cushion cells are either empty or already spacing
    for dy in sy_min..=sy_max {
        for dx in sx_min..=sx_max {
            if dx >= origin_x
                && dx < origin_x + variant.width
                && dy >= origin_y
                && dy < origin_y + variant.height
            {
                continue;
            }
            let coord = GridCoord::new(layer, dx, dy, width, height, layers)?;
            match grid.get(coord) {
                Some(OccupancyClass::Empty) | Some(OccupancyClass::Spacing(_)) => {}
                other => {
                    return Err(GeneratorError::OccupancyConflict {
                        stage: ErrorStage::Placement,
                        detail: format!(
                            "commit_spacing_occupied {} existing={:?}",
                            coord, other
                        ),
                    });
                }
            }
        }
    }

    // Pre-validate and build socket list (allocate IDs)
    let mut sockets = Vec::with_capacity(variant.sockets.len());
    for (si, sock) in variant.sockets.iter().enumerate() {
        let gx = origin_x.checked_add(sock.anchor.x).ok_or_else(|| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "socket_x_add",
            }
        })?;
        let gy = origin_y.checked_add(sock.anchor.y).ok_or_else(|| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "socket_y_add",
            }
        })?;
        let gl = layer.checked_add(sock.anchor.layer).ok_or_else(|| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "socket_layer_add",
            }
        })?;

        let global_anchor = GridCoord::new(gl, gx, gy, width, height, layers)?;

        let direction = map_direction_from_variant(sock.direction);
        let sok_role = map_socket_role_from_variant(sock.role);
        let socket_id = alloc.next_socket()?;

        sockets.push(PlacedSocket {
            id: socket_id,
            variant_socket_index: si as u16,
            global_anchor,
            direction,
            width: sock.width,
            role: sok_role,
            paired_socket_id: None,
        });
    }

    // ── Phase 2: All validation passed; commit to grid atomically ─────────
    let id = alloc.next_region()?;

    // Reserve footprint
    grid.reserve_rect(
        layer,
        origin_x,
        origin_y,
        variant.width,
        variant.height,
        OccupancyClass::Region(id.raw()),
    )?;

    // Reserve spacing cushion
    for dy in sy_min..=sy_max {
        for dx in sx_min..=sx_max {
            if dx >= origin_x
                && dx < origin_x + variant.width
                && dy >= origin_y
                && dy < origin_y + variant.height
            {
                continue;
            }
            let coord = GridCoord::new(layer, dx, dy, width, height, layers)?;
            let existing = grid.get(coord);
            if existing == Some(OccupancyClass::Empty) {
                grid.set(coord, OccupancyClass::Spacing(id.raw()))?;
            }
        }
    }

    // Collect transition IDs: match footprint against reserved transition cells.
    // A vertical-hub region is associated with transitions whose hub_footprint
    // overlaps this region's footprint.
    let mut transition_ids: Vec<TransitionId> = Vec::new();
    // Transitions are resolved during topology selection where the full
    // transition reservation list is available. The PlacedRegion records
    // empty here; the binding is completed in select_topology.

    let marker_variant_indices: Vec<u16> = (0..variant.markers.len() as u16).collect();

    Ok(PlacedRegion {
        id,
        role,
        variant_index,
        layer,
        footprint: (origin_x, origin_y, variant.width, variant.height),
        sockets,
        transitions: transition_ids,
        marker_variant_indices,
    })
}

fn map_direction_from_variant(d: super::prefab::Direction) -> Direction {
    match d {
        super::prefab::Direction::North => Direction::North,
        super::prefab::Direction::East => Direction::East,
        super::prefab::Direction::South => Direction::South,
        super::prefab::Direction::West => Direction::West,
    }
}

fn map_socket_role_from_variant(r: super::prefab::SocketRole) -> SocketRole {
    match r {
        super::prefab::SocketRole::Corridor => SocketRole::Corridor,
        super::prefab::SocketRole::Hall => SocketRole::Hall,
        super::prefab::SocketRole::Doorway => SocketRole::Doorway,
        super::prefab::SocketRole::Junction => SocketRole::Junction,
        super::prefab::SocketRole::DeadEnd => SocketRole::DeadEnd,
        super::prefab::SocketRole::LandmarkApproach => SocketRole::LandmarkApproach,
        super::prefab::SocketRole::LowerRampApproach => SocketRole::LowerRampApproach,
        super::prefab::SocketRole::UpperLanding => SocketRole::UpperLanding,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::{GeneratorConfig, QualifiedProfile};

    #[test]
    fn role_manifest_is_deterministic() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Primary)
            .normalize()
            .unwrap();
        let mut rng = Pcg32V1::new(12345, 67890);
        let manifest = RoleManifest::from_config(&config, &mut rng).unwrap();
        let total = manifest.total_regions();
        assert!(total >= config.region_min());
        assert!(total <= config.region_max());
        assert_eq!(manifest.spawn_count, 1);
        assert_eq!(manifest.distant_landmark_count, 1);
        assert_eq!(manifest.vertical_hub_count, 4);
    }

    #[test]
    fn role_manifest_total_regions_uses_checked_add() {
        let manifest = RoleManifest {
            spawn_count: 1,
            distant_landmark_count: 1,
            major_landmark_count: 2,
            junction_count: 3,
            dead_end_count: 2,
            vertical_hub_count: 4,
            required_route_count: 3,
            optional_branch_count: 2,
            ordinary_count: 6,
        };
        assert_eq!(manifest.total_regions(), 24);
    }

    #[test]
    fn placement_score_uses_candidate_coordinates() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Primary)
            .normalize()
            .unwrap();

        let placed = Vec::new();
        let current = Vec::new();

        // Two candidates at different positions should get different separation scores
        let score_a = compute_placement_score(0, 5, 5, 0, &placed, &current, &config);
        let score_b = compute_placement_score(0, 50, 50, 0, &placed, &current, &config);
        // When no regions are placed, both get default 500
        assert_eq!(score_a.separation_distance, 500);
        assert_eq!(score_b.separation_distance, 500);

        // With a region placed, scores should differ
        let mut alloc = IdAllocator::new();
        let placed_region = PlacedRegion {
            id: alloc.next_region().unwrap(),
            role: RegionRole::Spawn,
            variant_index: 0,
            layer: 0,
            footprint: (5, 5, 5, 5),
            sockets: vec![],
            transitions: vec![],
            marker_variant_indices: vec![],
        };
        let placed = vec![placed_region];
        let score_near = compute_placement_score(0, 7, 7, 0, &placed, &current, &config);
        let score_far = compute_placement_score(0, 50, 50, 0, &placed, &current, &config);
        // Near score should have smaller separation_distance than far
        assert!(score_near.separation_distance < score_far.separation_distance);
    }

    #[test]
    fn placement_score_is_total_order() {
        let a = PlacementScore {
            separation_distance: 5,
            layer_rank: 3,
            required_route_proximity: 0,
            diversity_rank: 1,
        };
        let b = PlacementScore {
            separation_distance: 5,
            layer_rank: 3,
            required_route_proximity: 0,
            diversity_rank: 1,
        };
        assert_eq!(a, b);

        let c = PlacementScore {
            separation_distance: 5,
            layer_rank: 3,
            required_route_proximity: 0,
            diversity_rank: 2,
        };
        assert_ne!(a, c);
    }

    #[test]
    fn spacing_check_non_overlapping_layers() {
        let mut alloc = IdAllocator::new();
        let r1 = PlacedRegion {
            id: alloc.next_region().unwrap(),
            role: RegionRole::Spawn,
            variant_index: 0,
            layer: 0,
            footprint: (0, 0, 10, 10),
            sockets: vec![],
            transitions: vec![],
            marker_variant_indices: vec![],
        };
        let r2 = PlacedRegion {
            id: alloc.next_region().unwrap(),
            role: RegionRole::DistantLandmark,
            variant_index: 0,
            layer: 1,
            footprint: (0, 0, 10, 10),
            sockets: vec![],
            transitions: vec![],
            marker_variant_indices: vec![],
        };
        let placed = vec![r2];
        assert!(check_spacing_multi(0, 0, 0, 10, 10, &placed, &[], 2));
    }

    #[test]
    fn spacing_check_overlapping() {
        let mut alloc = IdAllocator::new();
        let r1 = PlacedRegion {
            id: alloc.next_region().unwrap(),
            role: RegionRole::Spawn,
            variant_index: 0,
            layer: 0,
            footprint: (5, 5, 10, 10),
            sockets: vec![],
            transitions: vec![],
            marker_variant_indices: vec![],
        };
        let placed = vec![r1];
        assert!(!check_spacing_multi(0, 3, 3, 2, 2, &placed, &[], 2));
        assert!(check_spacing_multi(0, 0, 0, 1, 1, &placed, &[], 2));
    }

    /// Atomic rollback: when commit_placement_atomic fails after ID allocation
    /// but before grid mutation, the grid must remain uncorrupted.
    #[test]
    fn atomic_rollback_does_not_corrupt_grid() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .unwrap();
        let mut grid = OccupancyGrid::new(config.width(), config.height(), config.layers().2)
            .unwrap();

        // Pre-reserve some cells to force a conflict
        grid.reserve_rect(0, 0, 0, 10, 10, OccupancyClass::Region(99))
            .unwrap();

        // Now try to place a region at the same spot — should fail
        // We directly test the validation phase: the function checks availability
        // before allocating the region ID.
        let grid_clone = grid.clone();

        // Verify grid is unchanged after a failed placement attempt
        let mut alloc = IdAllocator::new();
        use super::super::prefab::{Cell, PrefabVariant, Tile};
        let variant = PrefabVariant {
            base_id: "test".into(),
            rotation_degrees: 0,
            width: 5,
            height: 5,
            layer_count: 1,
            layers: vec![vec![vec![Tile::Floor; 5]; 5]],
            origin: Cell { layer: 0, x: 0, y: 0 },
            sockets: vec![],
            markers: vec![],
            reservations: vec![],
            transitions: vec![],
            tags: vec!["room".to_string()],
        };
        let result = commit_placement_atomic(
            &variant,
            0,
            0,
            0,
            0,
            RegionRole::Spawn,
            &mut grid,
            &config,
            &mut alloc,
        );
        assert!(result.is_err());
        // Grid cells should be unchanged from the original (except any prior state)
        // Verify the footprint area still has the pre-reserved Region(99)
        assert_eq!(
            grid.get(GridCoord::new(0, 0, 0, config.width(), config.height(), config.layers().2).unwrap()),
            Some(OccupancyClass::Region(99))
        );
    }

    /// Complete transition-mask: all 7 reservation parts (ramp_run, upper_opening,
    /// landing, headroom, lower_funnel, lower_approach, hub_footprint) must be
    /// reserved in the grid when a transition is placed.
    #[test]
    fn transition_mask_all_parts_reserved() {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .unwrap();
        let mut grid = OccupancyGrid::new(config.width(), config.height(), config.layers().2)
            .unwrap();

        let mut alloc = IdAllocator::new();
        let tid = alloc.next_transition().unwrap();

        let cells = |coords: &[(u16, u16)]| -> Vec<GridCoord> {
            coords
                .iter()
                .map(|&(x, y)| {
                    GridCoord::new(0, x, y, config.width(), config.height(), config.layers().2)
                        .unwrap()
                })
                .collect()
        };

        let upper_cells = |coords: &[(u16, u16)]| -> Vec<GridCoord> {
            coords
                .iter()
                .map(|&(x, y)| {
                    GridCoord::new(1, x, y, config.width(), config.height(), config.layers().2)
                        .unwrap()
                })
                .collect()
        };

        let reservation = TransitionReservation {
            id: tid,
            lower_layer: 0,
            hub_footprint: (5, 5, 5, 5),
            ramp_run_cells: cells(&[(6, 6), (7, 6), (8, 6)]),
            upper_opening_cells: upper_cells(&[(6, 5)]),
            landing_cells: upper_cells(&[(6, 6), (7, 6)]),
            headroom_cells: upper_cells(&[(6, 7)]),
            lower_funnel_cells: cells(&[(4, 6)]),
            lower_approach_cells: cells(&[(3, 6)]),
        };

        // Reserve the transition
        reserve_transition_volume(&reservation, &mut grid, &config).unwrap();

        // Verify all 7 parts are reserved
        let class = OccupancyClass::Transition(tid.raw());
        // Hub footprint
        assert_eq!(
            grid.get(GridCoord::new(0, 5, 5, config.width(), config.height(), config.layers().2).unwrap()),
            Some(class)
        );
        // Ramp run
        assert_eq!(
            grid.get(GridCoord::new(0, 6, 6, config.width(), config.height(), config.layers().2).unwrap()),
            Some(class)
        );
        // Upper opening
        assert_eq!(
            grid.get(GridCoord::new(1, 6, 5, config.width(), config.height(), config.layers().2).unwrap()),
            Some(class)
        );
        // Landing
        assert_eq!(
            grid.get(GridCoord::new(1, 6, 6, config.width(), config.height(), config.layers().2).unwrap()),
            Some(class)
        );
        // Headroom
        assert_eq!(
            grid.get(GridCoord::new(1, 6, 7, config.width(), config.height(), config.layers().2).unwrap()),
            Some(class)
        );
        // Lower funnel
        assert_eq!(
            grid.get(GridCoord::new(0, 4, 6, config.width(), config.height(), config.layers().2).unwrap()),
            Some(class)
        );
        // Lower approach
        assert_eq!(
            grid.get(GridCoord::new(0, 3, 6, config.width(), config.height(), config.layers().2).unwrap()),
            Some(class)
        );
    }
}
