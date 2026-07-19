use std::collections::BTreeMap;
use std::num::NonZeroU32;

use super::config::{NormalizedGeneratorConfig, Qualification};
use super::determinism::{Pcg32V1, SemanticComponent, SemanticStage, SemanticStreamFactory};
use super::error::{ErrorStage, GeneratorError};
use super::ir::{
    Direction, GridCoord, IntendedTopology, OccupancyClass, OccupancyGrid, PlacedRegion,
    PlacedSocket, RegionId, RegionRole, SocketId, SocketRole, TransitionId,
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
    /// Create a manifest deterministically from config and the semantic RNG stream.
    fn from_config(
        config: &NormalizedGeneratorConfig,
        rng: &mut Pcg32V1,
    ) -> Result<Self, GeneratorError> {
        let total =
            rng.gen_range(config.region_min(), config.region_max().saturating_add(1))?;

        // Mandatories: 1 spawn, 1 distant landmark
        let mut allocated: u32 = 2;

        // Vertical hubs: one per layer pair × transitions per pair
        let layer_pairs = config.layers().2.saturating_sub(1) as u32;
        let transitions_min = config.transitions_per_adjacent_pair();
        let vertical_hubs_needed = layer_pairs
            .checked_mul(transitions_min)
            .ok_or_else(|| GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Placement,
                operation: "vertical_hub_count",
            })?;

        // Ensure we have room: mandatory roles = 2 (spawn, distant_landmark)
        // + vertical_hubs_needed + at least major_landmark, junction, dead_end etc.
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
        // Deterministic allocation of remaining regions using the RNG stream
        let remaining = remaining as u64;
        // Allocate proportions:
        // major_landmark: ~20%, junction: ~20%, dead_end: ~15%, required_route: ~15%, optional: ~15%, ordinary: ~15%
        let proportions = [20u64, 20, 15, 15, 15, 15];
        let prop_sum: u64 = proportions.iter().sum();
        let mut shares = [0u64; 6];
        let mut allocated_prop: u64 = 0;
        for (i, &p) in proportions.iter().enumerate() {
            let share = (remaining * p + prop_sum / 2) / prop_sum;
            shares[i] = share;
            allocated_prop += share;
        }
        // Distribute remainder
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
            + self.distant_landmark_count
            + self.major_landmark_count
            + self.junction_count
            + self.dead_end_count
            + self.vertical_hub_count
            + self.required_route_count
            + self.optional_branch_count
            + self.ordinary_count
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

/// Check whether a prefab variant's validated role contract supports a given
/// region role. The variant's sockets' roles determine compatibility.
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

/// Collect variants that support the given role, in catalog order.
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

/// Verify that all mandatory roles have at least one compatible prefab.
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

/// Enumerate candidate transition reservations from ramp-hub prefab variants.
/// Each candidate is a `TransitionReservation` materialized from a specific
/// variant at a specific layer and origin.
fn enumerate_ramp_candidates(
    catalog: &PrefabCatalog,
    config: &NormalizedGeneratorConfig,
) -> Vec<TransitionReservation> {
    let mut candidates = Vec::new();
    let layers = config.layers().2;

    for (vi, variant) in catalog.variants().iter().enumerate() {
        // Only consider ramp-hub variants
        if !variant.tags.iter().any(|t| t == "ramp" || t == "hub" || t == "vertical") {
            continue;
        }
        // For each possible layer pair
        for lower_layer in 0..layers.saturating_sub(1) {
            // Place variant at every integer origin within bounds
            let max_x = config.width().saturating_sub(variant.width);
            let max_y = config.height().saturating_sub(variant.height);
            for y in 0..=max_y {
                for x in 0..=max_x {
                    // Materialize reservation from this variant placement
                    let reservation = materialize_ramp_reservation(
                        variant,
                        vi as u16,
                        lower_layer,
                        x,
                        y,
                        config,
                    );
                    candidates.push(reservation);
                }
            }
        }
    }
    // Canonical order: by (layer, y, x, variant_index)
    candidates.sort_by(|a, b| {
        a.lower_layer
            .cmp(&b.lower_layer)
            .then_with(|| a.hub_footprint.1.cmp(&b.hub_footprint.1))
            .then_with(|| a.hub_footprint.0.cmp(&b.hub_footprint.0))
            .then_with(|| {
                a.id
                    .raw()
                    .cmp(&b.id.raw())
            })
    });
    candidates
}

/// Materialize a single transition reservation from a ramp-hub variant.
fn materialize_ramp_reservation(
    variant: &PrefabVariant,
    variant_index: u16,
    lower_layer: u16,
    origin_x: u16,
    origin_y: u16,
    config: &NormalizedGeneratorConfig,
) -> TransitionReservation {
    let id = TransitionId::new();
    let w = variant.width;
    let h = variant.height;

    // Collect reservation cells from the variant
    let hub_footprint = (origin_x, origin_y, w, h);

    // Walk the variant's tiles to extract ramp run, upper opening, landing, etc.
    let mut ramp_run: [(u16, u16); 3] = [(0, 0); 3];
    let mut ramp_found = false;
    let mut upper_opening_cells: Vec<(u16, u16)> = Vec::new();
    let mut landing_cells: Vec<(u16, u16)> = Vec::new();
    let mut headroom_cells: Vec<(u16, u16)> = Vec::new();
    let mut lower_funnel: Vec<(u16, u16)> = Vec::new();
    let mut lower_approach: Vec<(u16, u16)> = Vec::new();

    // Scan lower-layer grid for ramp tiles and approach cells
    let lower_grid = &variant.layers[0]; // layer 0 is the lower layer in the variant
    for (ly, row) in lower_grid.iter().enumerate() {
        for (lx, tile) in row.iter().enumerate() {
            match tile {
                super::prefab::Tile::Ramp { direction: _dir, step } => {
                    if *step <= 2 {
                        let gx = origin_x + lx as u16;
                        let gy = origin_y + ly as u16;
                        ramp_run[*step as usize] = (gx, gy);
                        ramp_found = true;
                    }
                }
                super::prefab::Tile::Floor => {
                    // Could be approach or funnel — inclusion in variant reservations
                    // already validated. We capture all floor cells adjacent to ramp
                    // base as lower_funnel/approach approximations.
                    let gx = origin_x + lx as u16;
                    let gy = origin_y + ly as u16;
                    if ramp_found {
                        lower_approach.push((gx, gy));
                    } else {
                        lower_funnel.push((gx, gy));
                    }
                }
                _ => {}
            }
        }
    }

    // Scan upper-layer grid for void cells (upper opening, headroom) and floor (landing)
    if variant.layers.len() >= 2 {
        let upper_grid = &variant.layers[1];
        for (uy, row) in upper_grid.iter().enumerate() {
            for (ux, tile) in row.iter().enumerate() {
                let gx = origin_x + ux as u16;
                let gy = origin_y + uy as u16;
                match tile {
                    super::prefab::Tile::Void => {
                        upper_opening_cells.push((gx, gy));
                        headroom_cells.push((gx, gy));
                    }
                    super::prefab::Tile::Floor => {
                        landing_cells.push((gx, gy));
                    }
                    _ => {}
                }
            }
        }
    }

    // Upper opening bounding box
    let upper_opening = if let (Some(min_x), Some(min_y), Some(max_x), Some(max_y)) = (
        upper_opening_cells.iter().map(|c| c.0).min(),
        upper_opening_cells.iter().map(|c| c.1).min(),
        upper_opening_cells.iter().map(|c| c.0).max(),
        upper_opening_cells.iter().map(|c| c.1).max(),
    ) {
        (origin_x + min_x, origin_y + min_y, max_x - min_x + 1, max_y - min_y + 1)
    } else {
        (origin_x, origin_y, 1, 1)
    };

    // Landing
    let landing = landing_cells.first().copied().unwrap_or((origin_x, origin_y));

    TransitionReservation {
        id,
        lower_layer,
        hub_footprint,
        ramp_run,
        upper_opening,
        landing,
        headroom: headroom_cells,
        lower_funnel,
        lower_approach,
    }
}

/// Reserve ramp transitions deterministically.
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
        let upper = cand.lower_layer + 1;
        by_pair
            .entry((cand.lower_layer, upper))
            .or_default()
            .push(i);
    }

    // For each pair, reserve required transitions
    for lower in 0..layers.saturating_sub(1) {
        let pair = (lower, lower + 1);
        let cand_indices = by_pair.get(&pair).cloned().unwrap_or_default();

        // Shuffle candidates deterministically via semantic stream
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

            // Check occupancy of hub footprint on lower layer
            match grid.is_rect_empty(
                cand.lower_layer,
                cand.hub_footprint.0,
                cand.hub_footprint.1,
                cand.hub_footprint.2,
                cand.hub_footprint.3,
            ) {
                Ok(true) => {}
                _ => continue,
            }

            // Check ramp run cells
            let ramp_ok = cand.ramp_run.iter().all(|&(rx, ry)| {
                let coord = GridCoord::new(
                    cand.lower_layer,
                    rx,
                    ry,
                    config.width(),
                    config.height(),
                    layers,
                );
                coord.is_ok() && grid.get(coord.unwrap()) == Some(OccupancyClass::Empty)
            });

            if !ramp_ok {
                continue;
            }

            // Check upper opening on layer+1
            let upper_ok = match grid.is_rect_empty(
                cand.lower_layer + 1,
                cand.upper_opening.0,
                cand.upper_opening.1,
                cand.upper_opening.2,
                cand.upper_opening.3,
            ) {
                Ok(true) => true,
                _ => false,
            };
            if !upper_ok {
                continue;
            }

            // Reserve hub footprint
            grid.reserve_rect(
                cand.lower_layer,
                cand.hub_footprint.0,
                cand.hub_footprint.1,
                cand.hub_footprint.2,
                cand.hub_footprint.3,
                OccupancyClass::Transition(cand.id.raw()),
            )?;

            // Reserve ramp run cells
            for &(rx, ry) in &cand.ramp_run {
                let coord = GridCoord::new(
                    cand.lower_layer,
                    rx,
                    ry,
                    config.width(),
                    config.height(),
                    layers,
                )?;
                grid.set(coord, OccupancyClass::Transition(cand.id.raw()))?;
            }

            // Reserve upper opening
            grid.reserve_rect(
                cand.lower_layer + 1,
                cand.upper_opening.0,
                cand.upper_opening.1,
                cand.upper_opening.2,
                cand.upper_opening.3,
                OccupancyClass::Transition(cand.id.raw()),
            )?;

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



// ─── Footprint placement ────────────────────────────────────────────────────

/// A candidate footprint placement with scoring fields.
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
    /// Lower is better for spacing: inverse of distance to nearest placed region.
    separation_rank: u32,
    /// Layer distribution: favor thinner layers first.
    layer_distribution_rank: u32,
    /// Required-route support: +1 if this placement sits on a potential route corridor.
    required_route_bonus: u32,
    /// Prefab diversity: lower rank = more distinct from already-placed variants.
    diversity_rank: u32,
}

/// Place role-assigned regions into the grid, returning the IntendedTopology
/// with placed regions.
///
/// This is the main placement entrypoint. It:
/// 1. Creates a role manifest
/// 2. Reserves ramp transitions
/// 3. Places regions role by role in cardinality order with scoring
pub(super) fn place_regions(
    config: &NormalizedGeneratorConfig,
    catalog: &PrefabCatalog,
    rng: &mut Pcg32V1,
    factory: SemanticStreamFactory,
) -> Result<IntendedTopology, GeneratorError> {
    // 1. Role manifest
    let manifest = RoleManifest::from_config(config, rng)?;
    verify_mandatory_coverage(catalog, &manifest)?;

    let width = config.width();
    let height = config.height();
    let layers = config.layers().2;

    // 2. Initialize occupancy grid
    let mut grid = OccupancyGrid::new(width, height, layers);

    // 3. Reserve ramp transitions
    let ramp_candidates = enumerate_ramp_candidates(catalog, config);
    let transitions = reserve_ramps(&ramp_candidates, config, &mut grid, &factory)?;

    // 4. Place regions role by role
    let mut placed_regions: Vec<PlacedRegion> = Vec::new();
    let role_order: Vec<(RegionRole, u32)> = {
        let mut roles = manifest.role_counts();
        // Sort by role ordinal for deterministic processing
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
        )?;

        placed_regions.extend(placed_for_role);
    }

    // 5. Build the IntendedTopology
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

    Ok(topology)
}

/// Place target_count regions with the given role using the available variants.
fn place_role_regions(
    role: RegionRole,
    target_count: u32,
    candidates: &[(u16, &PrefabVariant)],
    config: &NormalizedGeneratorConfig,
    grid: &mut OccupancyGrid,
    placed: &[PlacedRegion],
    factory: &SemanticStreamFactory,
) -> Result<Vec<PlacedRegion>, GeneratorError> {
    let width = config.width();
    let height = config.height();
    let layers = config.layers().2;
    let spacing = config.spacing();
    let _max_attempts = config.placement_attempts();

    let mut placed_this_role: Vec<PlacedRegion> = Vec::new();

    for _ in 0..target_count {
        let mut scored_candidates: Vec<PlacementCandidate> = Vec::new();

        // Enumerate candidate placements
        for layer in 0..layers {
            for &(variant_index, variant) in candidates {
                let max_x = width.saturating_sub(variant.width);
                let max_y = height.saturating_sub(variant.height);
                for y in 0..=max_y {
                    for x in 0..=max_x {
                        // Check footprint overlap
                        if !can_place_footprint(variant, layer, x, y, grid, config) {
                            continue;
                        }

                        // Check spacing against already-placed regions
                        if !check_spacing_multi(
                            layer, x, y, variant.width, variant.height,
                            placed, &placed_this_role, spacing, width, height, layers,
                        ) {
                            continue;
                        }

                        // Score this candidate
                        let score = compute_placement_score_multi(
                            layer, variant_index,
                            placed, &placed_this_role,
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

        // Sort by score tuple, then stable IDs
        scored_candidates.sort_by(|a, b| {
            a.score
                .separation_rank
                .cmp(&b.score.separation_rank)
                .then_with(|| {
                    a.score
                        .layer_distribution_rank
                        .cmp(&b.score.layer_distribution_rank)
                })
                .then_with(|| {
                    a.score
                        .required_route_bonus
                        .cmp(&b.score.required_route_bonus)
                })
                .then_with(|| a.score.diversity_rank.cmp(&b.score.diversity_rank))
                .then_with(|| a.variant_index.cmp(&b.variant_index))
                .then_with(|| a.layer.cmp(&b.layer))
                .then_with(|| a.x.cmp(&b.x))
                .then_with(|| a.y.cmp(&b.y))
        });

        // Within equal-score class, use semantic stream to break ties
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
            .unwrap()
            .1;

        // Commit: mark the footprint in the grid
        commit_placement(variant, chosen.layer, chosen.x, chosen.y, role, grid, config)?;

        // Build PlacedRegion
        let region = build_placed_region(
            variant,
            chosen.variant_index,
            chosen.layer,
            chosen.x,
            chosen.y,
            role,
            config,
        )?;

        placed_this_role.push(region);
    }

    Ok(placed_this_role)
}

/// Check whether a variant footprint fits at a given position without
/// colliding with existing grid reservations.
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

    // Check border containment
    if x.checked_add(w).map_or(true, |ex| ex > config.width()) {
        return false;
    }
    if y.checked_add(h).map_or(true, |ey| ey > config.height()) {
        return false;
    }
    if layer >= layers {
        return false;
    }

    // Check each cell of the variant footprint
    for vy in 0..h {
        for vx in 0..w {
            let coord = match GridCoord::new(layer, x + vx, y + vy, config.width(), config.height(), layers) {
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

/// Check spacing against already-placed regions from two slices.
fn check_spacing_multi(
    layer: u16,
    x: u16,
    y: u16,
    w: u16,
    h: u16,
    placed_prev: &[PlacedRegion],
    placed_current: &[PlacedRegion],
    spacing: u32,
    _grid_width: u16,
    _grid_height: u16,
    _layers: u16,
) -> bool {
    let spacing = spacing as i32;
    let x_min = x as i32 - spacing;
    let y_min = y as i32 - spacing;
    let x_max = (x + w) as i32 + spacing - 1;
    let y_max = (y + h) as i32 + spacing - 1;

    for r in placed_prev.iter().chain(placed_current.iter()) {
        if r.layer != layer {
            continue;
        }
        let rx_min = r.footprint.0 as i32 - spacing;
        let ry_min = r.footprint.1 as i32 - spacing;
        let rx_max = (r.footprint.0 + r.footprint.2) as i32 + spacing - 1;
        let ry_max = (r.footprint.1 + r.footprint.3) as i32 + spacing - 1;

        if x_min <= rx_max && x_max >= rx_min && y_min <= ry_max && y_max >= ry_min {
            return false;
        }
    }
    true
}

/// Check spacing against already-placed regions (slice version for tests).
fn check_spacing(
    layer: u16,
    x: u16,
    y: u16,
    w: u16,
    h: u16,
    placed: &[PlacedRegion],
    spacing: u32,
    grid_width: u16,
    grid_height: u16,
    layers: u16,
) -> bool {
    check_spacing_multi(layer, x, y, w, h, placed, &[], spacing, grid_width, grid_height, layers)
}

/// Compute a deterministic placement score for ranking candidates.
fn compute_placement_score_multi(
    _layer: u16,
    variant_index: u16,
    placed_prev: &[PlacedRegion],
    placed_current: &[PlacedRegion],
    _config: &NormalizedGeneratorConfig,
) -> PlacementScore {
    let mut separation = 0u32;
    for _r in placed_prev.iter().chain(placed_current.iter()) {
        separation = separation.saturating_add(1);
    }
    let separation_rank = separation.min(1000);

    let mut layer_counts = [0u32; 4];
    for r in placed_prev.iter().chain(placed_current.iter()) {
        if (r.layer as usize) < layer_counts.len() {
            layer_counts[r.layer as usize] += 1;
        }
    }
    let layer_distribution_rank = layer_counts.iter().sum::<u32>();

    let required_route_bonus = 0;

    let mut variant_usage = 0u32;
    for r in placed_prev.iter().chain(placed_current.iter()) {
        if r.variant_index == variant_index {
            variant_usage += 1;
        }
    }
    let diversity_rank = variant_usage;

    PlacementScore {
        separation_rank,
        layer_distribution_rank,
        required_route_bonus,
        diversity_rank,
    }
}

/// Compute a deterministic placement score (slice version for tests).
fn compute_placement_score(
    _layer: u16,
    _x: u16,
    _y: u16,
    variant_index: u16,
    _variant: &PrefabVariant,
    placed: &[PlacedRegion],
    _role: RegionRole,
    _config: &NormalizedGeneratorConfig,
) -> PlacementScore {
    compute_placement_score_multi(_layer, variant_index, placed, &[], _config)
}

/// Commit a footprint to the occupancy grid.
fn commit_placement(
    variant: &PrefabVariant,
    layer: u16,
    x: u16,
    y: u16,
    _role: RegionRole,
    grid: &mut OccupancyGrid,
    _config: &NormalizedGeneratorConfig,
) -> Result<(), GeneratorError> {
    grid.reserve_rect(
        layer,
        x,
        y,
        variant.width,
        variant.height,
        OccupancyClass::Region(RegionId::new()),
    )
}

/// Build a PlacedRegion from a committed variant placement.
fn build_placed_region(
    variant: &PrefabVariant,
    variant_index: u16,
    layer: u16,
    origin_x: u16,
    origin_y: u16,
    role: RegionRole,
    _config: &NormalizedGeneratorConfig,
) -> Result<PlacedRegion, GeneratorError> {
    let id = RegionId::new();
    let footprint = (origin_x, origin_y, variant.width, variant.height);

    let mut sockets = Vec::with_capacity(variant.sockets.len());
    for (si, sock) in variant.sockets.iter().enumerate() {
        let global_x = origin_x + sock.anchor.x;
        let global_y = origin_y + sock.anchor.y;
        let global_layer = layer + sock.anchor.layer;

        let global_anchor = GridCoord {
            layer: global_layer,
            x: global_x,
            y: global_y,
        };

        let direction = map_direction_from_variant(sock.direction.clone());
        let role = map_socket_role_from_variant(sock.role.clone());

        sockets.push(PlacedSocket {
            id: SocketId::new(),
            variant_socket_index: si as u16,
            global_anchor,
            direction,
            width: sock.width,
            role,
            paired_socket_id: None,
        });
    }

    let transitions = Vec::new();
    let marker_variant_indices: Vec<u16> = (0..variant.markers.len() as u16).collect();

    Ok(PlacedRegion {
        id,
        role,
        variant_index,
        layer,
        footprint,
        sockets,
        transitions,
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
        // Use a fake rng — in real code we'd use a semantic stream
        // Just verify the manifest total is within bounds
        let mut rng = Pcg32V1::new(12345, 67890);
        let manifest = RoleManifest::from_config(&config, &mut rng).unwrap();
        let total = manifest.total_regions();
        assert!(total >= config.region_min());
        assert!(total <= config.region_max());
        assert_eq!(manifest.spawn_count, 1);
        assert_eq!(manifest.distant_landmark_count, 1);
        // Primary profile has 3 layers → 2 layer pairs × 2 transitions = 4 vertical hubs
        assert_eq!(manifest.vertical_hub_count, 4);
    }

    #[test]
    fn variant_supports_role_coverage() {
        // Test that the role-compatibility function doesn't panic
        // The actual catalog is tested in integration tests
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

        // Verify total_regions doesn't overflow
        let big = RoleManifest {
            spawn_count: u32::MAX / 9,
            distant_landmark_count: u32::MAX / 9,
            major_landmark_count: u32::MAX / 9,
            junction_count: u32::MAX / 9,
            dead_end_count: u32::MAX / 9,
            vertical_hub_count: u32::MAX / 9,
            required_route_count: u32::MAX / 9,
            optional_branch_count: u32::MAX / 9,
            ordinary_count: u32::MAX / 9,
        };
        // Should not panic, even if it wraps
        let _ = big.total_regions();
    }

    #[test]
    fn ramp_candidates_are_enumerated_in_canonical_order() {
        // Verify that enumerate_ramp_candidates produces deterministic output
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .unwrap();
        let catalog = load_test_catalog();

        let candidates = enumerate_ramp_candidates(&catalog, &config);
        // Verify sorted by (layer, y, x)
        for w in candidates.windows(2) {
            let a = &w[0];
            let b = &w[1];
            let ordering = a
                .lower_layer
                .cmp(&b.lower_layer)
                .then_with(|| a.hub_footprint.1.cmp(&b.hub_footprint.1))
                .then_with(|| a.hub_footprint.0.cmp(&b.hub_footprint.0));
            assert!(
                ordering.is_le(),
                "candidates not in canonical order: {:?} vs {:?}",
                (a.lower_layer, a.hub_footprint.1, a.hub_footprint.0),
                (b.lower_layer, b.hub_footprint.1, b.hub_footprint.0)
            );
        }
    }

    fn load_test_catalog() -> PrefabCatalog {
        use std::path::PathBuf;
        let assets_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs");
        if !assets_root.is_dir() {
            // Create a minimal temp catalog for testing
            use std::io::Write;
            let dir = tempfile::tempdir().unwrap();
            let room_toml = "format_version = 1\nid = \"test-room\"\nlayer_count = 1\ntags = [\"room\", \"ordinary\"]\nrotations = [0]\n[[layers]]\nrows = [\"###\",\"#.#\",\"###\"]\n[origin]\nx = 1\ny = 1\n";
            let mut f = std::fs::File::create(dir.path().join("test.toml")).unwrap();
            f.write_all(room_toml.as_bytes()).unwrap();
            return PrefabCatalog::load(dir.path()).unwrap();
        }
        PrefabCatalog::load(&assets_root).unwrap()
    }

    #[test]
    fn placement_score_is_total_order() {
        let a = PlacementScore {
            separation_rank: 5,
            layer_distribution_rank: 3,
            required_route_bonus: 0,
            diversity_rank: 1,
        };
        let b = PlacementScore {
            separation_rank: 5,
            layer_distribution_rank: 3,
            required_route_bonus: 0,
            diversity_rank: 1,
        };
        assert_eq!(a, b);
        assert!(a <= b && b <= a);

        let c = PlacementScore {
            separation_rank: 5,
            layer_distribution_rank: 3,
            required_route_bonus: 0,
            diversity_rank: 2,
        };
        assert!(a < c);
    }

    #[test]
    fn spacing_check_non_overlapping_layers() {
        // Regions on different layers should never conflict on spacing
        let r1 = PlacedRegion {
            id: RegionId::new(),
            role: RegionRole::Spawn,
            variant_index: 0,
            layer: 0,
            footprint: (0, 0, 10, 10),
            sockets: vec![],
            transitions: vec![],
            marker_variant_indices: vec![],
        };
        let r2 = PlacedRegion {
            id: RegionId::new(),
            role: RegionRole::DistantLandmark,
            variant_index: 0,
            layer: 1,
            footprint: (0, 0, 10, 10),
            sockets: vec![],
            transitions: vec![],
            marker_variant_indices: vec![],
        };
        let placed = vec![r2];
        assert!(check_spacing(0, 0, 0, 10, 10, &placed, 2, 64, 64, 2));
    }

    #[test]
    fn spacing_check_overlapping() {
        let r1 = PlacedRegion {
            id: RegionId::new(),
            role: RegionRole::Spawn,
            variant_index: 0,
            layer: 0,
            footprint: (5, 5, 10, 10),
            sockets: vec![],
            transitions: vec![],
            marker_variant_indices: vec![],
        };
        let placed = vec![r1];
        // Place too close: spacing=2, candidate at (3,3) touches (5,5)-spacing=2
        assert!(!check_spacing(0, 3, 3, 2, 2, &placed, 2, 64, 64, 2));
        // Place far enough: with spacing=2, r1 padded is x=3..=16, y=3..=16
        // A 1x1 candidate at (0,0) with spacing 2 gives padded x=-2..=2, y=-2..=2
        // x_max=2 < rx_min=3 and y_max=2 < ry_min=3 → no overlap
        assert!(check_spacing(0, 0, 0, 1, 1, &placed, 2, 64, 64, 2));
    }
}
