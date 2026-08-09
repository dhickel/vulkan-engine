//! Complete source validators for the Richness assembly.
//!
//! Validators (in order):
//! 1. Sealing ownership: every partition volume owned exactly once; openings
//!    are omissions from a unique ownership partition.
//! 2. Convexity: every brush is convex.
//! 3. Normal class: cardinal + 45-degree only.
//! 4. Grid alignment: 16-unit quantum.
//! 5. Textures: role-valid, using authorized miptex from theme WADs.
//! 6. Protected routes: 64×80 route witnesses unobstructed.
//! 7. Overlap: opening-aware — wall brushes split around openings so no false
//!    overlaps between wall segments and openings.
//! 8. Support DAG: complete, acyclic, every brush reaches world.
//! 9. Actual-vs-reserved cost: summed actual costs ≤ reserved from complexity plan.
//!
//! # Contract
//!
//! - All validators are `pub(crate) fn validate_*` returning `Result<(), RichnessError>`.
//! - Validators may call each other internally.
//! - Crate-private; canonical ordering; no baseline changes.

use std::collections::{BTreeMap, BTreeSet};

use super::assembly::{AssemblyIR, BrushAssemblyRole};
use super::complexity::{BudgetReservation, ComplexityPlan};
use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::geometry as richness_geom;
use super::ids::BrushAssemblyId;
use super::support::validate_support_dag;
use super::theme::ThemeDefinition;
use super::vertical;
use super::visibility::{
    validate_compiler_conventions, validate_no_aligned_openings, validate_visibility_caps,
    VisibilityPlan,
};
use crate::enhanced_v3::geometry::ConvexBrush;

// ── Validator result ──────────────────────────────────────────────────────

/// Collective validation result.
#[derive(Debug, Clone)]
pub(crate) struct ValidationReport {
    /// Individual check results.
    pub checks: Vec<ValidationCheck>,
    /// Whether all checks passed.
    pub all_passed: bool,
}

/// A single validation check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidationCheck {
    /// Check name.
    pub name: String,
    /// Whether it passed.
    pub passed: bool,
    /// Diagnostic message if failed.
    pub message: Option<String>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            all_passed: true,
        }
    }

    fn record(&mut self, name: &str, result: Result<(), RichnessError>) {
        match result {
            Ok(()) => {
                self.checks.push(ValidationCheck {
                    name: name.to_string(),
                    passed: true,
                    message: None,
                });
            }
            Err(e) => {
                self.all_passed = false;
                self.checks.push(ValidationCheck {
                    name: name.to_string(),
                    passed: false,
                    message: Some(e.context.clone()),
                });
            }
        }
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

fn validation_error(
    code: RichnessErrorCode,
    category: RichnessErrorCategory,
    path: &str,
    context: impl Into<String>,
) -> RichnessError {
    RichnessError::new(
        code, 0, "?", "?", "?", "?", "?", "?", "?", path, category, context,
    )
}

// ── Run all validators ────────────────────────────────────────────────────

/// Run the complete validator set on an assembly.
pub(crate) fn validate_assembly(
    ir: &AssemblyIR,
    visibility: &VisibilityPlan,
    complexity: Option<&ComplexityPlan>,
    theme: &ThemeDefinition,
) -> ValidationReport {
    let mut report = ValidationReport::new();

    // Presentation budgets: light entities stay under the frozen contract
    // ceiling of 100 and every light origin is inside its owning room.
    let light_count = ir
        .entities
        .values()
        .filter(|entity| entity.classname == "light")
        .count();
    if light_count > 100 {
        report.record(
            "presentation_light_budget",
            Err(super::error::RichnessError::new(
                super::error::RichnessErrorCode::BudgetOverrun,
                0,
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "presentation.lights",
                super::error::RichnessErrorCategory::BudgetOverrun,
                format!("presentation placed {light_count} lights; ceiling is 100"),
            )),
        );
    }

    // 1. Sealing ownership
    report.record("sealing_ownership", validate_sealing_ownership(ir));

    // 2. Convexity
    report.record("convexity", validate_convexity(ir));

    // 3. Normal class
    report.record("normal_class", validate_normal_class(ir));

    // 4. Grid alignment
    report.record("grid_alignment", validate_grid_alignment(ir));

    // 5. Textures
    report.record("textures", validate_textures(ir, theme));

    // 6. Protected routes
    report.record("protected_routes", validate_protected_routes(ir));

    // 7. Overlap (opening-aware)
    report.record("overlap", validate_overlap(ir));

    // 8. Support DAG
    report.record("support_dag", validate_support_dag(ir).map(|_| ()));

    // 9. Actual-vs-reserved cost
    if let Some(cp) = complexity {
        report.record(
            "actual_vs_reserved_cost",
            validate_actual_vs_reserved_cost(ir, cp),
        );
    }

    // 10. Visibility caps
    report.record("visibility_caps", validate_visibility_caps(visibility));

    // 11. Aligned openings
    report.record("aligned_openings", validate_no_aligned_openings(visibility));

    // 12. Exact Phase-05 compiler convention records
    report.record(
        "compiler_conventions",
        validate_compiler_conventions(visibility),
    );

    // 13. Vertical architecture features (Phase 10)
    report.record(
        "vertical_multi_storey_shells",
        vertical::validate_multi_storey_shells(ir),
    );
    report.record(
        "vertical_slab_ownership",
        vertical::validate_slab_opening_ownership(ir),
    );
    report.record(
        "vertical_balcony_clearance",
        vertical::validate_balcony_clearance(ir),
    );
    report.record(
        "vertical_catwalk_void",
        vertical::validate_catwalk_over_void_only(ir),
    );
    report.record(
        "vertical_overlook_sealed",
        vertical::validate_overlook_sealed(ir),
    );
    report.record(
        "vertical_pit_chasm_pairs",
        vertical::validate_pit_chasm_pairs(ir),
    );
    report.record(
        "vertical_ladder_shafts",
        vertical::validate_ladder_shafts(ir),
    );
    report.record("vertical_stairwells", vertical::validate_stairwells(ir));
    report.record(
        "vertical_spiral_stairs",
        vertical::validate_spiral_stairs(ir),
    );
    report.record("vertical_drop_shafts", vertical::validate_drop_shafts(ir));
    report.record("vertical_arena", vertical::validate_vertical_arena(ir));

    // 14. Vertical cost reservation (Phase 10-C)
    if let Some(cp) = complexity {
        report.record(
            "vertical_cost_reservation",
            validate_vertical_cost_reservation(ir, cp),
        );
    }

    // 15. Archetype vertical recipe confirmation (Phase 10-C)
    report.record(
        "archetype_vertical_recipes",
        validate_archetype_vertical_recipes(ir, visibility),
    );

    report
}

// ── 1. Sealing ownership ─────────────────────────────────────────────────

/// Every partition volume is owned exactly once. Openings are omissions
/// from a unique ownership partition. No volume is claimed by two owners.
pub(crate) fn validate_sealing_ownership(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for opening in ir.openings.values() {
        if opening.wall_segment_ids.is_empty()
            || !opening.wall_segment_ids.contains(&opening.owner_brush_id)
        {
            return Err(validation_error(
                RichnessErrorCode::SemanticInfeasible,
                RichnessErrorCategory::SemanticInfeasibility,
                "validation.sealing",
                format!("opening {} has no unique canonical owner", opening.id.raw()),
            ));
        }
        for segment_id in &opening.wall_segment_ids {
            let segment = ir.brushes.get(segment_id).ok_or_else(|| {
                validation_error(
                    RichnessErrorCode::SemanticInfeasible,
                    RichnessErrorCategory::SemanticInfeasibility,
                    "validation.sealing",
                    format!(
                        "opening {} references missing wall segment {}",
                        opening.id.raw(),
                        segment_id.raw()
                    ),
                )
            })?;
            if segment.role != opening.wall_role {
                return Err(validation_error(
                    RichnessErrorCode::SemanticInfeasible,
                    RichnessErrorCategory::SemanticInfeasibility,
                    "validation.sealing",
                    format!(
                        "opening {} segment {} has role {}, expected {}",
                        opening.id.raw(),
                        segment_id.raw(),
                        segment.role.tag(),
                        opening.wall_role.tag()
                    ),
                ));
            }
            if segment.owner != opening.owner {
                return Err(validation_error(
                    RichnessErrorCode::SemanticInfeasible,
                    RichnessErrorCategory::SemanticInfeasibility,
                    "validation.sealing",
                    format!(
                        "opening {} segment {} belongs to a different ownership partition",
                        opening.id.raw(),
                        segment_id.raw(),
                    ),
                ));
            }
        }

        let (px0, py0, pz0, px1, py1, pz1) = opening.owner_partition_bounds;
        let (ox0, oy0, oz0, ox1, oy1, oz1) = opening.bounds;
        if ox0 < px0 || ox1 > px1 || oy0 < py0 || oy1 > py1 || oz0 < pz0 || oz1 > pz1 {
            return Err(validation_error(
                RichnessErrorCode::SemanticInfeasible,
                RichnessErrorCategory::SemanticInfeasibility,
                "validation.sealing",
                format!(
                    "opening {} escaped its one-owner wall partition",
                    opening.id.raw()
                ),
            ));
        }
    }
    Ok(())
}

// ── 2. Convexity ─────────────────────────────────────────────────────────

/// Every brush is convex (guaranteed by ConvexBrush::new, but we verify).
pub(crate) fn validate_convexity(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for brush in ir.brushes.values() {
        richness_geom::validate_positive_volume(&brush.brush)?;
    }
    for model in ir
        .entities
        .values()
        .filter_map(|entity| entity.brush_model.as_ref())
    {
        richness_geom::validate_positive_volume(model)?;
    }
    Ok(())
}

// ── 3. Normal class ──────────────────────────────────────────────────────

/// Only cardinal (XYZ) and 45-degree XY diagonal normals are approved.
pub(crate) fn validate_normal_class(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for brush in ir.brushes.values() {
        richness_geom::validate_approved_normals(&brush.brush)?;
    }
    for model in ir
        .entities
        .values()
        .filter_map(|entity| entity.brush_model.as_ref())
    {
        richness_geom::validate_approved_normals(model)?;
    }
    Ok(())
}

// ── 4. Grid alignment ───────────────────────────────────────────────────

/// All face plane d-values must be multiples of 16.
pub(crate) fn validate_grid_alignment(ir: &AssemblyIR) -> Result<(), RichnessError> {
    for brush in ir.brushes.values() {
        richness_geom::validate_grid_alignment(&brush.brush)?;
    }
    for model in ir
        .entities
        .values()
        .filter_map(|entity| entity.brush_model.as_ref())
    {
        richness_geom::validate_grid_alignment(model)?;
    }
    Ok(())
}

// ── 5. Textures ──────────────────────────────────────────────────────────

/// Texture names must be role-valid: auth-brick materials for walls,
/// slab textures for floors/ceilings, trim for portal surrounds, etc.
///
/// At the source IR level, textures are assigned per face role via the
/// variation plan. The actual `.map` emission maps these to concrete
/// miptex names. We validate that the role indices correspond to known
/// material roles from the generated constants.
pub(crate) fn validate_textures(
    ir: &AssemblyIR,
    theme: &ThemeDefinition,
) -> Result<(), RichnessError> {
    let authorized: BTreeSet<_> = theme.all_wad_identities().into_iter().collect();
    for brush in ir.brushes.values() {
        let assigned = ir.material_roles.get(&brush.id).copied().ok_or_else(|| {
            validation_error(
                RichnessErrorCode::AssetRoleMissing,
                RichnessErrorCategory::AssetRoleMissing,
                "validation.textures",
                format!("brush {} has no theme material role", brush.id.raw()),
            )
        })?;
        let expected = brush.role.semantic_role();
        if assigned != expected {
            return Err(validation_error(
                RichnessErrorCode::AssetRoleMissing,
                RichnessErrorCategory::AssetRoleMissing,
                "validation.textures",
                format!(
                    "brush {} role {} requires {:?}, found {:?}",
                    brush.id.raw(),
                    brush.role.tag(),
                    expected,
                    assigned
                ),
            ));
        }
        if !theme.roles.contains(&assigned) || !authorized.contains(assigned.wad_identity()) {
            return Err(validation_error(
                RichnessErrorCode::AssetRoleMissing,
                RichnessErrorCategory::AssetRoleMissing,
                "validation.textures",
                format!(
                    "theme {} does not authorize role {:?} / miptex {}",
                    theme.name,
                    assigned,
                    assigned.wad_identity()
                ),
            ));
        }
        let (basecolor, normal, gloss) = theme.companion_filenames(assigned);
        let prefix = assigned.wad_identity();
        if !basecolor.starts_with(prefix)
            || !normal.starts_with(prefix)
            || !gloss.starts_with(prefix)
        {
            return Err(validation_error(
                RichnessErrorCode::AssetRoleMissing,
                RichnessErrorCategory::AssetRoleMissing,
                "validation.textures",
                format!(
                    "theme {} has role-invalid companions for {prefix}",
                    theme.name
                ),
            ));
        }
    }
    Ok(())
}

// ── 6. Protected routes ─────────────────────────────────────────────────

/// 64×80 route witnesses must be unobstructed by any brush geometry.
/// Every portal throat must have a clear 64×80 opening.
pub(crate) fn validate_protected_routes(ir: &AssemblyIR) -> Result<(), RichnessError> {
    // For each opening that is a portal (has portal_id), verify the throat
    // is exactly 64 wide and 80 tall and no other brush intrudes.
    for opening in ir.openings.values() {
        if opening.portal_id.is_some() {
            let (ox0, oy0, oz0, ox1, oy1, oz1) = opening.bounds;
            let width = match opening.wall_role {
                BrushAssemblyRole::NorthWall | BrushAssemblyRole::SouthWall => ox1 - ox0,
                BrushAssemblyRole::EastWall | BrushAssemblyRole::WestWall => oy1 - oy0,
                _ => -1,
            };
            let height = oz1 - oz0;

            if width != 64 || height != 80 {
                return Err(RichnessError::new(
                    RichnessErrorCode::ValueOutOfRange,
                    0,
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "validation.protected_routes",
                    RichnessErrorCategory::PlacementTopologyExhaustion,
                    format!(
                        "portal opening {:?} throat is {}×{} (requires exact 64×80)",
                        opening.id.raw(),
                        width,
                        height
                    ),
                ));
            }

            let throat =
                ConvexBrush::make_box((ox0, ox1), (oy0, oy1), (oz0, oz1)).map_err(|error| {
                    validation_error(
                        RichnessErrorCode::SemanticInfeasible,
                        RichnessErrorCategory::SemanticInfeasibility,
                        "validation.protected_routes",
                        format!("invalid throat solid: {error}"),
                    )
                })?;
            for brush in ir.brushes.values() {
                if richness_geom::brushes_overlap(&brush.brush, &throat)? {
                    return Err(validation_error(
                        RichnessErrorCode::SemanticInfeasible,
                        RichnessErrorCategory::SemanticInfeasibility,
                        "validation.protected_routes",
                        format!(
                            "brush {} ({}) intrudes into exact throat for opening {}",
                            brush.id.raw(),
                            brush.role.tag(),
                            opening.id.raw()
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

// ── 7. Overlap (opening-aware) ────────────────────────────────────────────

/// Validate that no two brushes have positive-volume overlap.
/// Opening-aware: wall brushes that have openings must have those openings
/// omitted (wall is split around the opening), so the wall brush itself
/// does not overlap the opening. Frame brushes must not overlap the throat
/// volume.
pub(crate) fn validate_overlap(ir: &AssemblyIR) -> Result<(), RichnessError> {
    // First: check all brush pairs for positive-volume overlap.
    let brush_ids: Vec<_> = ir.brushes.keys().copied().collect();
    let bounds = brush_ids
        .iter()
        .map(|id| {
            ir.brushes[id]
                .brush
                .aabb()
                .map(|(min, max)| (min.0, min.1, min.2, max.0, max.1, max.2))
                .map_err(|error| {
                    validation_error(
                        RichnessErrorCode::SemanticInfeasible,
                        RichnessErrorCategory::SemanticInfeasibility,
                        "validation.overlap",
                        format!("brush {} AABB: {error}", id.raw()),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for i in 0..brush_ids.len() {
        for j in (i + 1)..brush_ids.len() {
            let a_id = brush_ids[i];
            let b_id = brush_ids[j];
            let a = &ir.brushes[&a_id];
            let b = &ir.brushes[&b_id];
            if !aabbs_may_touch(bounds[i], bounds[j]) {
                continue;
            }

            if richness_geom::brushes_overlap(&a.brush, &b.brush)? {
                return Err(RichnessError::new(
                    RichnessErrorCode::ValueOutOfRange,
                    0,
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "validation.overlap",
                    RichnessErrorCategory::PlacementTopologyExhaustion,
                    format!(
                        "brushes {:?} ({}) and {:?} ({}) have positive-volume overlap",
                        a_id.raw(),
                        a.role.tag(),
                        b_id.raw(),
                        b.role.tag()
                    ),
                ));
            }
        }
    }

    // Second: every owning wall segment and frame must be disjoint from the
    // exact omitted volume. Boundary contact is legal; positive volume is not.
    for opening in ir.openings.values() {
        let (x0, y0, z0, x1, y1, z1) = opening.bounds;
        let omitted = ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z1)).map_err(|error| {
            validation_error(
                RichnessErrorCode::SemanticInfeasible,
                RichnessErrorCategory::SemanticInfeasibility,
                "validation.overlap",
                format!("opening {} has invalid bounds: {error}", opening.id.raw()),
            )
        })?;
        for brush_id in opening
            .wall_segment_ids
            .iter()
            .chain(opening.frame_brush_ids.iter())
        {
            let brush = ir.brushes.get(brush_id).ok_or_else(|| {
                validation_error(
                    RichnessErrorCode::SemanticInfeasible,
                    RichnessErrorCategory::SemanticInfeasibility,
                    "validation.overlap",
                    format!(
                        "opening {} references missing brush {}",
                        opening.id.raw(),
                        brush_id.raw()
                    ),
                )
            })?;
            if richness_geom::brushes_overlap(&brush.brush, &omitted)? {
                return Err(validation_error(
                    RichnessErrorCode::SemanticInfeasible,
                    RichnessErrorCategory::SemanticInfeasibility,
                    "validation.overlap",
                    format!(
                        "brush {} ({}) occupies opening {}",
                        brush.id.raw(),
                        brush.role.tag(),
                        opening.id.raw()
                    ),
                ));
            }
        }
    }

    Ok(())
}

fn aabbs_may_touch(
    (ax0, ay0, az0, ax1, ay1, az1): (i128, i128, i128, i128, i128, i128),
    (bx0, by0, bz0, bx1, by1, bz1): (i128, i128, i128, i128, i128, i128),
) -> bool {
    ax0 <= bx1 && bx0 <= ax1 && ay0 <= by1 && by0 <= ay1 && az0 <= bz1 && bz0 <= az1
}

// ── 8. Support DAG ───────────────────────────────────────────────────────

// validate_support_dag is in support.rs; re-exported via use.

// ── 9. Actual-vs-reserved cost ────────────────────────────────────────────

/// Summed actual costs from the assembly must be ≤ reserved costs from the
/// complexity plan.
pub(crate) fn validate_actual_vs_reserved_cost(
    ir: &AssemblyIR,
    plan: &ComplexityPlan,
) -> Result<(), RichnessError> {
    // Count actual costs from assembly
    let actual = BudgetReservation {
        faces: ir
            .brushes
            .values()
            .map(|brush| brush.brush.faces.len() as u32)
            .sum(),
        brushes: ir.brushes.len() as u32,
        entities: ir.entities.len() as u32,
        lights: 0, // lights not yet placed in assembly phase
        vertical_openings: 0,
        support_contacts: actual_support_budget_units(ir),
        package_assets: 0,
        compiler_lumps: 15, // always 15 for BSP2
        renderer_batches: 0,
        renderer_memory_bytes: 0,
        runtime_requirements: 0,
    };

    if !plan.assert_dominates(&actual) {
        let mut role_counts = BTreeMap::new();
        for brush in ir.brushes.values() {
            *role_counts.entry(brush.role.tag()).or_insert(0u32) += 1;
        }
        return Err(RichnessError::new(
            RichnessErrorCode::BudgetOverrun, 0,
            "?", "?", "?", "?", "?", "?", "?",
            "validation.cost",
            RichnessErrorCategory::BudgetOverrun,
            format!(
                "actual costs exceed reserved: faces actual={} reserved={}, brushes actual={} reserved={}, support actual={} reserved={}, roles={role_counts:?}",
                actual.faces, plan.total_reserved.faces,
                actual.brushes, plan.total_reserved.brushes,
                actual.support_contacts, plan.total_reserved.support_contacts,
            ),
        ));
    }

    Ok(())
}

/// Count emitted support assemblies in the same units reserved by Phase 08.
///
/// The exact support DAG deliberately contains one proof edge per brush. The
/// complexity catalog reserves coarser authored support assemblies: one world
/// anchor and one shell/ceiling transfer per room, one frame assembly per
/// portal, and one unit per independently emitted massing/support piece.
/// Keeping these counts separate preserves per-brush geometric proof without
/// charging every convex decomposition piece as a new authored support recipe.
fn actual_support_budget_units(ir: &AssemblyIR) -> u32 {
    let mut floor_owners = BTreeSet::new();
    let mut ceiling_owners = BTreeSet::new();
    let mut cave_owners = BTreeSet::new();
    let mut vertical_shells = BTreeSet::new();
    let mut independent = 0u32;

    for brush in ir.brushes.values() {
        match brush.role {
            BrushAssemblyRole::FloorSlab => {
                floor_owners.insert(brush.owner.clone());
            }
            BrushAssemblyRole::CeilingSlab => {
                ceiling_owners.insert(brush.owner.clone());
            }
            // Cave boxes are a deterministic convex decomposition of one
            // cave shell, not independently authored support assemblies.
            // Their individual positive-area DAG edges remain mandatory;
            // this budget meter charges the one owning cave assembly.
            BrushAssemblyRole::CaveFloor
            | BrushAssemblyRole::CaveWall
            | BrushAssemblyRole::CaveCeiling => {
                cave_owners.insert(brush.owner.clone());
            }
            // Split vertical shell partitions are one authored support
            // assembly per owner/role. Their individual geometric contacts
            // remain mandatory in the exact support DAG.
            BrushAssemblyRole::UpperShellWall
            | BrushAssemblyRole::LadderShaftWall
            | BrushAssemblyRole::DropShaftWall
            | BrushAssemblyRole::SpiralShellWall
            | BrushAssemblyRole::ArenaGateWall => {
                vertical_shells.insert((brush.owner.clone(), brush.role));
            }
            BrushAssemblyRole::NorthWall
            | BrushAssemblyRole::SouthWall
            | BrushAssemblyRole::EastWall
            | BrushAssemblyRole::WestWall
            | BrushAssemblyRole::DiagNEWall
            | BrushAssemblyRole::DiagNWWall
            | BrushAssemblyRole::DiagSEWall
            | BrushAssemblyRole::DiagSWWall
            | BrushAssemblyRole::PortalPost
            | BrushAssemblyRole::PortalLintel
            | BrushAssemblyRole::PortalSurround => {}
            _ => independent = independent.saturating_add(1),
        }
    }

    (floor_owners.len() as u32)
        .saturating_add(ceiling_owners.len() as u32)
        .saturating_add(cave_owners.len() as u32)
        .saturating_add(vertical_shells.len() as u32)
        .saturating_add(ir.portal_assemblies.len() as u32)
        .saturating_add(independent)
}

// ── Convenience function ──────────────────────────────────────────────────

/// Run all validators and panic if any fail (for test use).
pub(crate) fn assert_all_valid(
    ir: &AssemblyIR,
    visibility: &VisibilityPlan,
    complexity: Option<&ComplexityPlan>,
    theme: &ThemeDefinition,
) {
    let report = validate_assembly(ir, visibility, complexity, theme);
    for check in &report.checks {
        if !check.passed {
            panic!(
                "validation check '{}' failed: {}",
                check.name,
                check.message.as_deref().unwrap_or("no message")
            );
        }
    }
    assert!(report.all_passed, "validation must pass all checks");
}

// ── Route witness validator ───────────────────────────────────────────────

/// Validate that route witnesses are unobstructed at the specified point.
///
/// A route witness is a 64×80 clear volume starting at `origin` in the
/// given direction. No brush (except frame brushes for openings) may
/// intrude into this volume.
pub(crate) fn validate_route_witness(
    ir: &AssemblyIR,
    origin: (i128, i128, i128), // (x0, y0, z0) of witness
    direction: RouteWitnessDir,
) -> Result<(), RichnessError> {
    let (dx, dy) = match direction {
        RouteWitnessDir::North => (0, 1),
        RouteWitnessDir::South => (0, -1),
        RouteWitnessDir::East => (1, 0),
        RouteWitnessDir::West => (-1, 0),
    };

    let wx = if dx >= 0 { origin.0 } else { origin.0 - 64 };
    let wy = if dy >= 0 { origin.1 } else { origin.1 - 64 };
    let wx1 = wx + 64;
    let wy1 = wy + 64;
    let wz0 = 16; // floor top
    let wz1 = 96; // throat top (16 + 80)

    for brush in ir.brushes.values() {
        let bb = brush.brush.aabb().map_err(|e| {
            RichnessError::new(
                RichnessErrorCode::ValueOutOfRange,
                0,
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "validation.route_witness",
                RichnessErrorCategory::PlacementTopologyExhaustion,
                format!("AABB: {e}"),
            )
        })?;
        let ((bmin_x, bmin_y, bmin_z), (bmax_x, bmax_y, bmax_z)) = bb;

        if bmin_x < wx1
            && bmax_x > wx
            && bmin_y < wy1
            && bmax_y > wy
            && bmin_z < wz1
            && bmax_z > wz0
        {
            // Potential obstruction
            return Err(RichnessError::new(
                RichnessErrorCode::ValueOutOfRange,
                0,
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "validation.route_witness",
                RichnessErrorCategory::PlacementTopologyExhaustion,
                format!(
                    "route witness at ({},{},{}) obstructed by brush {:?} ({})",
                    origin.0,
                    origin.1,
                    origin.2,
                    brush.id.raw(),
                    brush.role.tag()
                ),
            ));
        }
    }

    Ok(())
}

/// Direction for a route witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RouteWitnessDir {
    North,
    South,
    East,
    West,
}

// ── 10. Vertical cost reservation ─────────────────────────────────────────

/// Validate that actual vertical feature costs are within the complexity plan
/// reservation. Per the plan: complete source/PVS/support cost reservation for
/// every vertical recipe. Actual vertical brush counts and face counts must
/// not exceed reserved vertical budget.
pub(crate) fn validate_vertical_cost_reservation(
    ir: &AssemblyIR,
    plan: &ComplexityPlan,
) -> Result<(), RichnessError> {
    let is_vertical = |brush: &&super::assembly::BrushAssembly| {
        brush.role.is_vertical_architecture()
            || (brush.role == BrushAssemblyRole::MonolithSolid
                && brush.owner.archetype_id_str() == Some("grand_arena"))
    };
    let vertical_brushes = ir.brushes.values().filter(is_vertical).collect::<Vec<_>>();
    let descriptor_entities = ir
        .entities
        .values()
        .filter(|entity| entity.keys.contains_key("richness_volume"))
        .collect::<Vec<_>>();
    let descriptor_models = descriptor_entities
        .iter()
        .filter_map(|entity| entity.brush_model.as_ref())
        .collect::<Vec<_>>();
    let actual = BudgetReservation {
        faces: vertical_brushes
            .iter()
            .map(|brush| brush.brush.faces.len() as u32)
            .chain(
                descriptor_models
                    .iter()
                    .map(|model| model.faces.len() as u32),
            )
            .sum(),
        brushes: (vertical_brushes.len() + descriptor_models.len()) as u32,
        entities: descriptor_entities.len() as u32,
        lights: 0,
        // A paired lower-ceiling/upper-floor omission is one logical opening.
        vertical_openings: ir
            .openings
            .values()
            .filter(|opening| opening.wall_role == BrushAssemblyRole::FloorSlab)
            .count() as u32,
        support_contacts: ir
            .supports
            .values()
            .filter(|support| {
                vertical_brushes
                    .iter()
                    .any(|brush| brush.id == support.child)
            })
            .count() as u32,
        package_assets: 0,
        compiler_lumps: 0,
        renderer_batches: u32::from(!vertical_brushes.is_empty()),
        renderer_memory_bytes: ((vertical_brushes.len() + descriptor_models.len()) as u64)
            .saturating_mul(4_096),
        runtime_requirements: descriptor_entities.len() as u32,
    };
    let reserved = plan.vertical_reservation();
    if actual.within(&reserved) {
        return Ok(());
    }
    Err(validation_error(
        RichnessErrorCode::BudgetOverrun,
        RichnessErrorCategory::BudgetOverrun,
        "validation.vertical_cost",
        format!("actual vertical cost {actual:?} exceeds complete reserved cost {reserved:?}"),
    ))
}

// ── 11. Archetype vertical recipe confirmation ────────────────────────────

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpectedVerticalRoleContract {
    pub archetype_id: &'static str,
    pub recipe: super::content_types::VerticalRecipe,
    pub brush_roles: &'static [BrushAssemblyRole],
    pub opening_roles: &'static [BrushAssemblyRole],
    pub descriptor_kind: Option<&'static str>,
}

const EMPTY_ROLES: &[BrushAssemblyRole] = &[];
const PAIRED_SLAB_OPENINGS: &[BrushAssemblyRole] =
    &[BrushAssemblyRole::FloorSlab, BrushAssemblyRole::CeilingSlab];
const DROP_ROLES: &[BrushAssemblyRole] = &[
    BrushAssemblyRole::PitPerimeterSlab,
    BrushAssemblyRole::DropEntryGuard,
    BrushAssemblyRole::DropLanding,
];
const STAIR_ROLES: &[BrushAssemblyRole] = &[
    BrushAssemblyRole::StairTread,
    BrushAssemblyRole::StairLanding,
    BrushAssemblyRole::StairGuard,
];
const LADDER_ROLES: &[BrushAssemblyRole] = &[
    BrushAssemblyRole::LadderShaftWall,
    BrushAssemblyRole::LadderRung,
    BrushAssemblyRole::LadderLanding,
    BrushAssemblyRole::LadderLip,
];
const SPIRAL_ROLES: &[BrushAssemblyRole] = &[
    BrushAssemblyRole::SpiralShellWall,
    BrushAssemblyRole::SpiralColumn,
    BrushAssemblyRole::SpiralTread,
    BrushAssemblyRole::SpiralLanding,
];
const BALCONY_ROLES: &[BrushAssemblyRole] = &[
    BrushAssemblyRole::BalconySlab,
    BrushAssemblyRole::GuardRail,
    BrushAssemblyRole::Corbel,
];
const CATWALK_ROLES: &[BrushAssemblyRole] = &[
    BrushAssemblyRole::CatwalkDeck,
    BrushAssemblyRole::GuardRail,
    BrushAssemblyRole::VerticalSupport,
];
const OVERLOOK_ROLES: &[BrushAssemblyRole] = &[BrushAssemblyRole::PartialWall];
const OVERLOOK_OPENINGS: &[BrushAssemblyRole] = &[BrushAssemblyRole::PartialWall];
const ARENA_ROLES: &[BrushAssemblyRole] = &[
    BrushAssemblyRole::BalconySlab,
    BrushAssemblyRole::GuardRail,
    BrushAssemblyRole::Corbel,
    BrushAssemblyRole::CatwalkDeck,
    BrushAssemblyRole::MonolithSolid,
    BrushAssemblyRole::StairTread,
    BrushAssemblyRole::StairGuard,
    BrushAssemblyRole::ArenaGateWall,
];
const ARENA_OPENINGS: &[BrushAssemblyRole] = &[
    BrushAssemblyRole::FloorSlab,
    BrushAssemblyRole::CeilingSlab,
    BrushAssemblyRole::ArenaGateWall,
];

const fn role_contract(
    archetype_id: &'static str,
    recipe: super::content_types::VerticalRecipe,
    brush_roles: &'static [BrushAssemblyRole],
    opening_roles: &'static [BrushAssemblyRole],
    descriptor_kind: Option<&'static str>,
) -> ExpectedVerticalRoleContract {
    ExpectedVerticalRoleContract {
        archetype_id,
        recipe,
        brush_roles,
        opening_roles,
        descriptor_kind,
    }
}

/// Total generated-catalog role contract.  Every one of the canonical 30
/// entries appears exactly once. Catalog recipe `None` does not erase an
/// archetype-specific vertical set-piece contract: arena, bridge, overlook,
/// and grand-arena roles remain mandatory without inventing a catalog enum.
/// `grand_arena` therefore stays `None` while its independently materialized
/// arena has a complete role/opening contract.
pub(crate) const EXPECTED_VERTICAL_ROLE_CONTRACTS: [ExpectedVerticalRoleContract; 30] = [
    role_contract(
        "ambush_cross",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "antechamber",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "arena",
        super::content_types::VerticalRecipe::None,
        BALCONY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "barracks",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "bridge_crossing",
        super::content_types::VerticalRecipe::None,
        CATWALK_ROLES,
        PAIRED_SLAB_OPENINGS,
        None,
    ),
    role_contract(
        "cistern",
        super::content_types::VerticalRecipe::DropHole,
        DROP_ROLES,
        PAIRED_SLAB_OPENINGS,
        Some("one_way_drop"),
    ),
    role_contract(
        "crossroads",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "entrance_hall",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "flooded_crypt",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "foundry",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "gallery",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "grand_arena",
        super::content_types::VerticalRecipe::None,
        ARENA_ROLES,
        ARENA_OPENINGS,
        None,
    ),
    role_contract(
        "grand_stair_hall",
        super::content_types::VerticalRecipe::Stairwell,
        STAIR_ROLES,
        PAIRED_SLAB_OPENINGS,
        None,
    ),
    role_contract(
        "grotto",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "guard_hall",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "hypostyle_hall",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "kill_court",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "ladder_hub",
        super::content_types::VerticalRecipe::LadderShaft,
        LADDER_ROLES,
        PAIRED_SLAB_OPENINGS,
        Some("climb"),
    ),
    role_contract(
        "observatory",
        super::content_types::VerticalRecipe::OpenStairwell,
        STAIR_ROLES,
        PAIRED_SLAB_OPENINGS,
        None,
    ),
    role_contract(
        "ossuary",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "overlook_hall",
        super::content_types::VerticalRecipe::None,
        OVERLOOK_ROLES,
        OVERLOOK_OPENINGS,
        None,
    ),
    role_contract(
        "pit_room",
        super::content_types::VerticalRecipe::DropHole,
        DROP_ROLES,
        PAIRED_SLAB_OPENINGS,
        Some("one_way_drop"),
    ),
    role_contract(
        "reliquary",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "shrine",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "spiral_tower",
        super::content_types::VerticalRecipe::SpiralStair,
        SPIRAL_ROLES,
        PAIRED_SLAB_OPENINGS,
        None,
    ),
    role_contract(
        "throne_hall",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "trapped_gallery",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "treasury",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "vault",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
    role_contract(
        "vestibule",
        super::content_types::VerticalRecipe::None,
        EMPTY_ROLES,
        EMPTY_ROLES,
        None,
    ),
];

pub(crate) fn validate_archetype_vertical_recipes(
    ir: &AssemblyIR,
    visibility: &VisibilityPlan,
) -> Result<(), RichnessError> {
    use super::generated_content;

    let _ = visibility;
    let mut represented = BTreeSet::new();
    for owner in ir
        .brushes
        .values()
        .map(|brush| &brush.owner)
        .chain(ir.openings.values().map(|opening| &opening.owner))
        .chain(ir.entities.values().map(|entity| &entity.owner))
    {
        if let (Some(request), Some(archetype)) = (owner.request_id, owner.archetype) {
            represented.insert((request, archetype));
        }
    }

    for (request, archetype) in represented {
        let index = archetype.raw() as usize;
        let contract = EXPECTED_VERTICAL_ROLE_CONTRACTS.get(index).ok_or_else(|| {
            validation_error(
                RichnessErrorCode::SemanticInfeasible,
                RichnessErrorCategory::SemanticInfeasibility,
                "validation.archetype_vertical",
                format!("represented archetype index {index} has no total role contract"),
            )
        })?;
        if generated_content::ARCHETYPE_IDS.get(index).copied() != Some(contract.archetype_id)
            || generated_content::ARCHETYPE_VERTICAL_RECIPE
                .get(index)
                .copied()
                != Some(contract.recipe)
        {
            return Err(validation_error(
                RichnessErrorCode::SemanticInfeasible,
                RichnessErrorCategory::SemanticInfeasibility,
                "validation.archetype_vertical",
                format!("catalog/role-contract mismatch at index {index}"),
            ));
        }
        if contract.brush_roles.is_empty()
            && contract.opening_roles.is_empty()
            && contract.descriptor_kind.is_none()
        {
            continue;
        }

        let roles = ir
            .brushes
            .values()
            .filter(|brush| {
                brush.owner.request_id == Some(request) && brush.owner.archetype == Some(archetype)
            })
            .map(|brush| brush.role)
            .collect::<BTreeSet<_>>();
        let opening_roles = ir
            .openings
            .values()
            .filter(|opening| {
                opening.owner.request_id == Some(request)
                    && opening.owner.archetype == Some(archetype)
            })
            .map(|opening| opening.wall_role)
            .collect::<BTreeSet<_>>();
        let mut missing = contract
            .brush_roles
            .iter()
            .filter(|role| !roles.contains(role))
            .map(|role| role.tag().to_string())
            .chain(
                contract
                    .opening_roles
                    .iter()
                    .filter(|role| !opening_roles.contains(role))
                    .map(|role| format!("{}_opening", role.tag())),
            )
            .collect::<Vec<_>>();
        if let Some(kind) = contract.descriptor_kind {
            let present = ir.entities.values().any(|entity| {
                entity.owner.request_id == Some(request)
                    && entity.owner.archetype == Some(archetype)
                    && entity.classname == "trigger_multiple"
                    && entity
                        .brush_model
                        .as_ref()
                        .and_then(|model| model.aabb().ok())
                        .zip(entity.brush_model_bounds)
                        .is_some_and(|((min, max), bounds)| {
                            bounds == (min.0, min.1, min.2, max.0, max.1, max.2)
                        })
                    && entity
                        .keys
                        .get("richness_volume")
                        .is_some_and(|value| value == kind)
            });
            if !present {
                missing.push(format!("{kind}_descriptor"));
            }
        }
        if !missing.is_empty() {
            return Err(validation_error(
                RichnessErrorCode::SemanticInfeasible,
                RichnessErrorCategory::SemanticInfeasibility,
                "validation.archetype_vertical",
                format!(
                    "archetype {} request {} ({:?}) is missing role tags [{}]; present roles={roles:?}; opening roles={opening_roles:?}",
                    contract.archetype_id,
                    request.raw(),
                    contract.recipe,
                    missing.join(",")
                ),
            ));
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::geometry::ConvexBrush;
    use crate::enhanced_v3::richness::assembly::{
        AssemblyIR, BrushAssembly, BrushAssemblyRole, BudgetDimension, CostSource, OpeningRecord,
        SemanticAttribution, SupportRecord, SupportTarget,
    };
    use crate::enhanced_v3::richness::generated_content;
    use crate::enhanced_v3::richness::ids::{
        ArchetypeIndex, ArchetypeRequestId, BeatId, OpeningAssemblyId, ReservationId, ZoneId,
    };
    use crate::enhanced_v3::richness::theme::THEME_ANCIENT;

    fn make_attr() -> SemanticAttribution {
        SemanticAttribution::from_reservation(
            ReservationId::new(0),
            Some(ArchetypeRequestId::new(0)),
            Some(ArchetypeIndex::new(1)),
            Some(BeatId::new(0)),
            Some(ZoneId::new(0)),
        )
    }

    fn make_cost() -> CostSource {
        CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        }
    }

    fn make_simple_assembly() -> (AssemblyIR, BrushAssemblyId) {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        let floor_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: floor_id,
            brush: ConvexBrush::make_box((0, 256), (0, 256), (0, 16)).unwrap(),
            role: BrushAssemblyRole::FloorSlab,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });
        let sid = ir.alloc_support_id();
        ir.insert_support(SupportRecord {
            id: sid,
            child: floor_id,
            parent: SupportTarget::World,
        });

        (ir, floor_id)
    }

    #[test]
    fn validate_sealing_ownership_rejects_orphan_opening() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();

        // Opening references non-existent owner
        ir.openings.insert(
            OpeningAssemblyId::new(0),
            OpeningRecord {
                id: OpeningAssemblyId::new(0),
                owner_brush_id: BrushAssemblyId::new(99),
                wall_segment_ids: vec![BrushAssemblyId::new(99)],
                owner_partition_bounds: (0, 0, 16, 64, 16, 160),
                wall_role: BrushAssemblyRole::NorthWall,
                owner: attr,
                bounds: (0, 0, 16, 64, 16, 96),
                portal_id: None,
                frame_brush_ids: Vec::new(),
                portal_style: None,
            },
        );

        assert!(validate_sealing_ownership(&ir).is_err());
    }

    #[test]
    fn validate_convexity_passes_for_valid_brushes() {
        let (ir, _) = make_simple_assembly();
        assert!(validate_convexity(&ir).is_ok());
    }

    #[test]
    fn validate_normal_class_passes_for_axis_aligned() {
        let (ir, _) = make_simple_assembly();
        assert!(validate_normal_class(&ir).is_ok());
    }

    #[test]
    fn validate_grid_alignment_passes_for_quantum_aligned() {
        let (ir, _) = make_simple_assembly();
        assert!(validate_grid_alignment(&ir).is_ok());
    }

    #[test]
    fn validate_textures_passes() {
        let (ir, _) = make_simple_assembly();
        for theme in super::super::theme::all_themes() {
            assert!(validate_textures(&ir, theme).is_ok());
        }
    }

    #[test]
    fn validate_textures_rejects_role_invalid_assignment() {
        let (mut ir, floor_id) = make_simple_assembly();
        ir.material_roles
            .insert(floor_id, super::super::theme::SemanticRole::Wall);
        let error = validate_textures(&ir, &THEME_ANCIENT).unwrap_err();
        assert_eq!(error.code, RichnessErrorCode::AssetRoleMissing);
    }

    #[test]
    fn validate_protected_routes_no_openings() {
        let (ir, _) = make_simple_assembly();
        // No openings = no portal throats to check = passes
        assert!(validate_protected_routes(&ir).is_ok());
    }

    #[test]
    fn validate_overlap_detects_overlapping_brushes() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        let a_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: a_id,
            brush: ConvexBrush::make_box((0, 64), (0, 64), (0, 64)).unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let b_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: b_id,
            brush: ConvexBrush::make_box((32, 96), (0, 64), (0, 64)).unwrap(),
            role: BrushAssemblyRole::SouthWall,
            owner: attr,
            cost,
            support: SupportTarget::World,
        });

        assert!(validate_overlap(&ir).is_err());
    }

    #[test]
    fn validation_report_aggregates_results() {
        let (ir, _) = make_simple_assembly();
        let vis = VisibilityPlan::new();
        let report = validate_assembly(&ir, &vis, None, &THEME_ANCIENT);

        assert!(report.all_passed);
        for check in &report.checks {
            assert!(check.passed, "check '{}' failed", check.name);
        }
    }

    #[test]
    fn assert_all_valid_panics_on_failure() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        // Create overlapping brushes
        let a_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: a_id,
            brush: ConvexBrush::make_box((0, 64), (0, 64), (16, 160)).unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });
        let b_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: b_id,
            brush: ConvexBrush::make_box((16, 48), (16, 48), (16, 160)).unwrap(),
            role: BrushAssemblyRole::EastWall,
            owner: attr,
            cost,
            support: SupportTarget::World,
        });

        let vis = VisibilityPlan::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_all_valid(&ir, &vis, None, &THEME_ANCIENT);
        }));
        assert!(result.is_err(), "assert_all_valid should panic on overlap");
    }

    #[test]
    fn validate_route_witness_unobstructed() {
        let (ir, _) = make_simple_assembly();
        // Route witness at (96, 0, 16) going North — should be clear
        // since the floor is at z=0..16 and the witness is above it.
        let result = validate_route_witness(&ir, (96, 0, 16), RouteWitnessDir::North);
        // The floor brush spans z=0..16, so z=16 is the top edge — might
        // touch but not intrude (AABB test with > not >= means z0=16 is floor top,
        // not inside). Let's check: wz0=16, floor_zmax=16 → bmax_z(16) > wz0(16) is false
        assert!(result.is_ok());
    }

    #[test]
    fn vertical_role_contract_is_total_and_matches_all_thirty_generated_entries() {
        assert_eq!(
            EXPECTED_VERTICAL_ROLE_CONTRACTS.len(),
            generated_content::ARCHETYPE_COUNT
        );
        let mut ids = BTreeSet::new();
        for (index, contract) in EXPECTED_VERTICAL_ROLE_CONTRACTS.iter().enumerate() {
            assert!(
                ids.insert(contract.archetype_id),
                "duplicate contract for {}",
                contract.archetype_id
            );
            assert_eq!(
                generated_content::ARCHETYPE_IDS[index],
                contract.archetype_id
            );
            assert_eq!(
                generated_content::ARCHETYPE_VERTICAL_RECIPE[index],
                contract.recipe
            );
        }
        assert_eq!(ids.len(), 30);

        let vestibule = EXPECTED_VERTICAL_ROLE_CONTRACTS
            .iter()
            .find(|contract| contract.archetype_id == "vestibule")
            .unwrap();
        assert_eq!(
            vestibule.recipe,
            super::super::content_types::VerticalRecipe::None
        );
        assert!(vestibule.brush_roles.is_empty());
        assert!(vestibule.opening_roles.is_empty());

        let grand_arena = EXPECTED_VERTICAL_ROLE_CONTRACTS
            .iter()
            .find(|contract| contract.archetype_id == "grand_arena")
            .unwrap();
        assert_eq!(
            grand_arena.recipe,
            super::super::content_types::VerticalRecipe::None
        );
        assert!(!grand_arena.brush_roles.is_empty());
        assert!(grand_arena
            .opening_roles
            .contains(&BrushAssemblyRole::ArenaGateWall));
    }

    #[test]
    fn every_vertical_contract_reports_a_typed_missing_role_error() {
        let visibility = VisibilityPlan::new();
        let required = EXPECTED_VERTICAL_ROLE_CONTRACTS
            .iter()
            .enumerate()
            .filter(|(_, contract)| {
                !contract.brush_roles.is_empty()
                    || !contract.opening_roles.is_empty()
                    || contract.descriptor_kind.is_some()
            })
            .collect::<Vec<_>>();
        assert_eq!(required.len(), 10);

        for (index, contract) in required {
            let mut ir = AssemblyIR::new();
            let owner = SemanticAttribution::from_reservation(
                ReservationId::new(index as u32),
                Some(ArchetypeRequestId::new(index as u32)),
                Some(ArchetypeIndex::new(index as u32)),
                Some(BeatId::new(0)),
                Some(ZoneId::new(0)),
            );
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush: ConvexBrush::make_box((0, 64), (0, 64), (0, 16)).unwrap(),
                role: BrushAssemblyRole::FloorSlab,
                owner,
                cost: make_cost(),
                support: SupportTarget::World,
            });

            let error = validate_archetype_vertical_recipes(&ir, &visibility).unwrap_err();
            assert_eq!(
                error.code,
                RichnessErrorCode::SemanticInfeasible,
                "{}",
                contract.archetype_id
            );
            assert_eq!(
                error.path, "validation.archetype_vertical",
                "{}",
                contract.archetype_id
            );
            assert!(error.context.contains(contract.archetype_id));
            assert!(error.context.contains("missing role tags"));
        }
    }

    #[test]
    fn validate_actual_vs_reserved_cost_detects_overrun() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 100, // high face count
        };

        // Create many brushes to exceed a minimal budget
        for i in 0..6 {
            let id = ir.alloc_brush_id();
            ir.insert_brush(BrushAssembly {
                id,
                brush: ConvexBrush::make_box((i * 32, i * 32 + 32), (0, 256), (0, 16)).unwrap(),
                role: BrushAssemblyRole::FloorSlab,
                owner: attr.clone(),
                cost,
                support: SupportTarget::World,
            });
        }

        // Create a minimal complexity plan with tiny budget
        let plan = ComplexityPlan {
            selected_recipes: Vec::new(),
            total_reserved: BudgetReservation {
                faces: 10,
                brushes: 2,
                ..BudgetReservation::ZERO
            },
            budget: crate::enhanced_v3::richness::complexity::ComplexityBudget::new(
                crate::enhanced_v3::richness::request::RichnessPreset::Sparse,
            ),
            mandatory_reserved: false,
            theme_variant: 0,
            errors: Vec::new(),
        };

        let result = validate_actual_vs_reserved_cost(&ir, &plan);
        assert!(result.is_err());
    }
}
