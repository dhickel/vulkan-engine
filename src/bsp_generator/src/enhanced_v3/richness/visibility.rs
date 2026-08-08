//! Structural occlusion intent and compiler convention management.
//!
//! Builds occlusion intent first (offset shafts, bent approaches, sills,
//! partial walls, columns as occluders), then applies only Phase-05-proven
//! compiler conventions (hint/hintskip/skip/func_detail). Limits semantic
//! vertical merges to ≤2 per room and ≤6 per planned cluster. Rejects
//! visibility caps and unintended aligned openings.
//!
//! Emits semantic room/opening attribution needed to map compiled leaves/PVS
//! back to intent (records, not runtime mapping).
//!
//! # Contract
//!
//! - Crate-private; canonical ordering; no baseline changes.
//! - No diagonal portals; no float geometry.
//! - All occluders use exact convex brush geometry.
//! - Convention assignments must be proven by Phase 05 evidence.

use std::collections::{BTreeMap, BTreeSet};

use super::assembly::{AssemblyIR, BrushAssembly, BrushAssemblyRole, SemanticAttribution};
use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::ids::{BrushAssemblyId, OpeningAssemblyId, PortalId, ReservationId};
use super::topology::CommittedRoute;

// ── Occluder kind ─────────────────────────────────────────────────────────

/// Structural elements that contribute intentional occlusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum OccluderKind {
    /// Offset shaft: vertical conduit creating axial occlusion.
    OffsetShaft,
    /// Bent approach: angled wall section blocking line-of-sight.
    BentApproach,
    /// Sill: low wall segment creating partial occlusion.
    Sill,
    /// Partial wall: segment blocking sight along a line.
    PartialWall,
    /// Column: pillar blocking diagonal sight lines.
    Column,
    /// Wall liner mass providing added occlusion.
    WallLiner,
    /// Pilaster creating lateral occlusion.
    Pilaster,
    /// Buttress mass creating external occlusion.
    Buttress,
    /// Portal post creating vertical frame occlusion.
    PortalPostOcclusion,
    /// Lintel creating horizontal frame occlusion.
    PortalLintelOcclusion,
    /// Surround mass creating portal frame occlusion.
    PortalSurroundOcclusion,
}

impl OccluderKind {
    pub fn tag(self) -> &'static str {
        match self {
            Self::OffsetShaft => "offset_shaft",
            Self::BentApproach => "bent_approach",
            Self::Sill => "sill",
            Self::PartialWall => "partial_wall",
            Self::Column => "column",
            Self::WallLiner => "wall_liner",
            Self::Pilaster => "pilaster",
            Self::Buttress => "buttress",
            Self::PortalPostOcclusion => "portal_post_occlusion",
            Self::PortalLintelOcclusion => "portal_lintel_occlusion",
            Self::PortalSurroundOcclusion => "portal_surround_occlusion",
        }
    }

    /// Convert from BrushAssemblyRole to OccluderKind.
    pub fn from_role(role: BrushAssemblyRole) -> Option<Self> {
        match role {
            BrushAssemblyRole::OffsetShaft => Some(Self::OffsetShaft),
            BrushAssemblyRole::BentApproach => Some(Self::BentApproach),
            BrushAssemblyRole::Sill => Some(Self::Sill),
            BrushAssemblyRole::PartialWall => Some(Self::PartialWall),
            BrushAssemblyRole::InteriorColumn => Some(Self::Column),
            BrushAssemblyRole::WallLiner => Some(Self::WallLiner),
            BrushAssemblyRole::Pilaster => Some(Self::Pilaster),
            BrushAssemblyRole::Buttress => Some(Self::Buttress),
            BrushAssemblyRole::PortalPost => Some(Self::PortalPostOcclusion),
            BrushAssemblyRole::PortalLintel => Some(Self::PortalLintelOcclusion),
            BrushAssemblyRole::PortalSurround => Some(Self::PortalSurroundOcclusion),
            _ => None,
        }
    }
}

// ── Compiler convention assignment ────────────────────────────────────────

/// A Phase-05-proven compiler convention applied to a brush.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum CompilerConvention {
    /// `hint` / `hintskip` split plane: guides BSP splits without adding
    /// draw surfaces. Only used on sloped hint wedges that survived Phase 05
    /// qualification.
    HintSplit,
    /// `skip` texture: adds solid collision volume with no visible faces.
    SkipSolid,
    /// `func_detail` entity: brush joins world model, classname is consumed.
    /// Does not create an inline model. Phase 05 proven by entity preservation
    /// test.
    FuncDetail,
}

impl CompilerConvention {
    pub fn tag(self) -> &'static str {
        match self {
            Self::HintSplit => "hint",
            Self::SkipSolid => "skip",
            Self::FuncDetail => "func_detail",
        }
    }
}

/// Concrete, Phase-05-qualified compiler emission for one brush.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompilerConventionRecord {
    pub brush_id: BrushAssemblyId,
    pub convention: CompilerConvention,
    /// Exact primary miptex or entity classname.
    pub primary: &'static str,
    /// Exact companion miptex (`hintskip` for a `hint` split), if required.
    pub secondary: Option<&'static str>,
}

impl CompilerConventionRecord {
    fn from_intent(intent: &OcclusionIntent) -> Option<Self> {
        let convention = intent.convention?;
        let (primary, secondary) = match convention {
            CompilerConvention::HintSplit => ("hint", Some("hintskip")),
            CompilerConvention::SkipSolid => ("skip", None),
            CompilerConvention::FuncDetail => ("func_detail", None),
        };
        Some(Self {
            brush_id: intent.brush_id,
            convention,
            primary,
            secondary,
        })
    }
}

// ── Occlusion intent record ───────────────────────────────────────────────

/// A single occlusion intent: a structural element + compiler convention.
#[derive(Debug, Clone)]
pub(crate) struct OcclusionIntent {
    /// The brush that creates occlusion.
    pub brush_id: BrushAssemblyId,
    /// What kind of structural occluder this is.
    pub kind: OccluderKind,
    /// The compiler convention applied (if any).
    pub convention: Option<CompilerConvention>,
    /// The semantic owner of this occluder.
    pub owner: SemanticAttribution,
    /// Whether this occluder is part of a portal frame (posts/lintel/surround).
    pub is_portal_frame: bool,
    /// The portal this occluder belongs to, if any.
    pub portal_id: Option<PortalId>,
    /// The opening this occluder frames, if any.
    pub opening_id: Option<OpeningAssemblyId>,
}

// ── Vertical merge record ─────────────────────────────────────────────────

/// A semantic vertical merge: two adjacent rooms merged into one leaf.
///
/// Capped at ≤2 per room and ≤6 per planned cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerticalMergeRecord {
    /// The two merged reservation IDs.
    pub merged: (ReservationId, ReservationId),
    /// The planned cluster this merge belongs to.
    pub cluster_id: u32,
    /// Why this merge is semantically safe (both rooms share identical
    /// structural height and are monotonic in the PVS tree).
    pub reason: String,
}

// ── Visibility plan ───────────────────────────────────────────────────────

/// Complete visibility strategy for an assembly.
#[derive(Debug, Clone)]
pub(crate) struct VisibilityPlan {
    /// All occlusion intents, ordered canonically by brush ID.
    pub occluders: BTreeMap<BrushAssemblyId, OcclusionIntent>,
    /// Exact compiler-emission records, keyed by brush ID.
    pub compiler_records: BTreeMap<BrushAssemblyId, CompilerConventionRecord>,
    /// Planned semantic leaf records keyed by room reservation.
    pub semantic_leaves: BTreeMap<ReservationId, SemanticLeafAttribution>,
    /// Planned semantic PVS adjacency keyed by room reservation.
    pub semantic_pvs: BTreeMap<ReservationId, SemanticPvsAttribution>,
    /// All vertical merges (capped at 2/room, 6/cluster).
    pub vertical_merges: Vec<VerticalMergeRecord>,
    /// Count of merges per room reservation.
    pub merge_count_per_room: BTreeMap<ReservationId, u32>,
    /// Count of merges per cluster.
    pub merge_count_per_cluster: BTreeMap<u32, u32>,
    /// Maximum merges per room cap (2).
    pub max_merges_per_room: u32,
    /// Maximum merges per cluster cap (6).
    pub max_merges_per_cluster: u32,
    /// Whether all caps are satisfied.
    pub caps_satisfied: bool,
    /// Rejection reasons.
    pub rejections: Vec<String>,
}

impl VisibilityPlan {
    /// Create an empty visibility plan.
    pub fn new() -> Self {
        Self {
            occluders: BTreeMap::new(),
            compiler_records: BTreeMap::new(),
            semantic_leaves: BTreeMap::new(),
            semantic_pvs: BTreeMap::new(),
            vertical_merges: Vec::new(),
            merge_count_per_room: BTreeMap::new(),
            merge_count_per_cluster: BTreeMap::new(),
            max_merges_per_room: 2,
            max_merges_per_cluster: 6,
            caps_satisfied: true,
            rejections: Vec::new(),
        }
    }

    /// Record an occlusion intent for a brush.
    pub fn record_occluder(&mut self, intent: OcclusionIntent) {
        if let Some(record) = CompilerConventionRecord::from_intent(&intent) {
            self.compiler_records.insert(intent.brush_id, record);
        }
        self.occluders.insert(intent.brush_id, intent);
    }

    /// Attempt to record a vertical merge. Rejects if caps would be exceeded.
    pub fn try_merge(
        &mut self,
        merged: (ReservationId, ReservationId),
        cluster_id: u32,
        reason: String,
    ) -> Result<(), String> {
        // Check per-room cap
        for &rid in &[merged.0, merged.1] {
            let count = self.merge_count_per_room.get(&rid).copied().unwrap_or(0);
            if count >= self.max_merges_per_room {
                let msg = format!(
                    "room {:?} has {} merges (cap {})",
                    rid.raw(),
                    count,
                    self.max_merges_per_room
                );
                self.rejections.push(msg.clone());
                self.caps_satisfied = false;
                return Err(msg);
            }
        }

        // Check per-cluster cap
        let cluster_count = self
            .merge_count_per_cluster
            .get(&cluster_id)
            .copied()
            .unwrap_or(0);
        if cluster_count >= self.max_merges_per_cluster {
            let msg = format!(
                "cluster {} has {} merges (cap {})",
                cluster_id, cluster_count, self.max_merges_per_cluster
            );
            self.rejections.push(msg.clone());
            self.caps_satisfied = false;
            return Err(msg);
        }

        // Record
        self.vertical_merges.push(VerticalMergeRecord {
            merged,
            cluster_id,
            reason,
        });
        *self.merge_count_per_room.entry(merged.0).or_insert(0) += 1;
        *self.merge_count_per_room.entry(merged.1).or_insert(0) += 1;
        *self.merge_count_per_cluster.entry(cluster_id).or_insert(0) += 1;

        Ok(())
    }

    /// Validate no unintended aligned openings.
    ///
    /// Two openings on opposite walls that are directly aligned create an
    /// unintended sightline that must be rejected unless it's an explicit
    /// planned feature.
    pub fn check_aligned_openings(&mut self, ir: &AssemblyIR) {
        let openings: Vec<_> = ir.openings.values().collect();
        for i in 0..openings.len() {
            for j in (i + 1)..openings.len() {
                let a = &openings[i];
                let b = &openings[j];

                if a.owner.reservation_id != b.owner.reservation_id {
                    continue;
                }
                let north_south = matches!(
                    (a.wall_role, b.wall_role),
                    (BrushAssemblyRole::NorthWall, BrushAssemblyRole::SouthWall)
                        | (BrushAssemblyRole::SouthWall, BrushAssemblyRole::NorthWall)
                );
                let east_west = matches!(
                    (a.wall_role, b.wall_role),
                    (BrushAssemblyRole::EastWall, BrushAssemblyRole::WestWall)
                        | (BrushAssemblyRole::WestWall, BrushAssemblyRole::EastWall)
                );
                let aligned = (north_south && a.bounds.0 < b.bounds.3 && b.bounds.0 < a.bounds.3)
                    || (east_west && a.bounds.1 < b.bounds.4 && b.bounds.1 < a.bounds.4);
                // Route portals are explicit planned sightlines. Only
                // unplanned openings on opposite walls are an unintended
                // alignment; route-owned portal pairs are retained in the
                // semantic PVS graph rather than rejected as decoration.
                if aligned && (a.portal_id.is_none() || b.portal_id.is_none()) {
                    self.rejections.push(format!(
                        "opposite-wall openings {} and {} align in reservation {}",
                        a.id.raw(),
                        b.id.raw(),
                        a.owner.reservation_id.raw()
                    ));
                }
            }
        }
    }

    /// Build all occlusion intents from an assembly IR.
    ///
    /// Every brush with an occluder role gets an intent record. Conventions
    /// are assigned based on the brush role.
    pub fn build_from_assembly(ir: &AssemblyIR) -> Self {
        let mut plan = Self::new();

        for brush in ir.brushes.values() {
            if let Some(kind) = OccluderKind::from_role(brush.role) {
                let convention = match brush.role {
                    BrushAssemblyRole::OffsetShaft
                    | BrushAssemblyRole::BentApproach
                    | BrushAssemblyRole::Sill
                    | BrushAssemblyRole::PartialWall
                    | BrushAssemblyRole::WallLiner
                    | BrushAssemblyRole::Pilaster
                    | BrushAssemblyRole::Buttress => {
                        // Structural occlusion: `skip` solid
                        Some(CompilerConvention::SkipSolid)
                    }
                    BrushAssemblyRole::InteriorColumn => {
                        // Columns may use skip for occlusion
                        Some(CompilerConvention::SkipSolid)
                    }
                    BrushAssemblyRole::PortalPost
                    | BrushAssemblyRole::PortalLintel
                    | BrushAssemblyRole::PortalSurround => {
                        // Portal frame: func_detail (consumed into world model)
                        Some(CompilerConvention::FuncDetail)
                    }
                    _ => None,
                };

                // Find opening/portal association
                let mut portal_id = None;
                let mut opening_id = None;
                for opening in ir.openings.values() {
                    if opening.frame_brush_ids.contains(&brush.id) {
                        portal_id = opening.portal_id;
                        opening_id = Some(opening.id);
                        break;
                    }
                }

                let is_portal_frame = matches!(
                    brush.role,
                    BrushAssemblyRole::PortalPost
                        | BrushAssemblyRole::PortalLintel
                        | BrushAssemblyRole::PortalSurround
                );

                plan.record_occluder(OcclusionIntent {
                    brush_id: brush.id,
                    kind,
                    convention,
                    owner: brush.owner.clone(),
                    is_portal_frame,
                    portal_id,
                    opening_id,
                });
            }
        }

        plan.check_aligned_openings(ir);
        plan.populate_semantic_records(ir, &[]);
        plan
    }

    /// Build visibility records from the assembly and the real committed
    /// horizontal/vertical route graph.
    pub fn build_from_assembly_and_routes(
        ir: &AssemblyIR,
        routes: &[CommittedRoute],
        vertical_routes: &[CommittedRoute],
    ) -> Self {
        let mut plan = Self::build_from_assembly(ir);
        let all_routes: Vec<_> = routes.iter().chain(vertical_routes.iter()).collect();
        plan.populate_semantic_records(ir, &all_routes);

        for route in vertical_routes {
            let cluster_id = ir
                .brushes_for_reservation(route.source)
                .first()
                .and_then(|brush| brush.owner.zone_id)
                .map_or(0, |zone| zone.raw());
            let source_count = plan
                .merge_count_per_room
                .get(&route.source)
                .copied()
                .unwrap_or(0);
            let target_count = plan
                .merge_count_per_room
                .get(&route.target)
                .copied()
                .unwrap_or(0);
            let cluster_count = plan
                .merge_count_per_cluster
                .get(&cluster_id)
                .copied()
                .unwrap_or(0);
            // Retain every vertical route in semantic PVS, but merge only
            // the canonical prefix that fits the leaf-cap budget. Later
            // routes remain explicit PVS edges instead of invalidating the
            // complete composition through an over-cap merge attempt.
            if source_count < plan.max_merges_per_room
                && target_count < plan.max_merges_per_room
                && cluster_count < plan.max_merges_per_cluster
            {
                let _ = plan.try_merge(
                    (route.source, route.target),
                    cluster_id,
                    format!("vertical route {}", route.id.raw()),
                );
            }
        }
        plan
    }

    fn populate_semantic_records(&mut self, ir: &AssemblyIR, routes: &[&CommittedRoute]) {
        self.semantic_leaves.clear();
        self.semantic_pvs.clear();

        let reservations: BTreeSet<_> = ir
            .brushes
            .values()
            .map(|brush| brush.owner.reservation_id)
            .collect();
        let mut adjacency: BTreeMap<ReservationId, BTreeSet<ReservationId>> = reservations
            .iter()
            .map(|reservation| (*reservation, BTreeSet::new()))
            .collect();
        let mut portals: BTreeMap<ReservationId, BTreeSet<PortalId>> = BTreeMap::new();
        for route in routes {
            adjacency
                .entry(route.source)
                .or_default()
                .insert(route.target);
            adjacency
                .entry(route.target)
                .or_default()
                .insert(route.source);
            portals
                .entry(route.source)
                .or_default()
                .insert(route.source_portal.id);
            portals
                .entry(route.target)
                .or_default()
                .insert(route.target_portal.id);
        }

        for reservation_id in reservations {
            let visible_openings: Vec<_> = ir
                .openings
                .values()
                .filter(|opening| opening.owner.reservation_id == reservation_id)
                .map(|opening| opening.id)
                .collect();
            let blocking_occluders: Vec<_> = self
                .occluders
                .values()
                .filter(|intent| intent.owner.reservation_id == reservation_id)
                .map(|intent| intent.brush_id)
                .collect();
            self.semantic_leaves.insert(
                reservation_id,
                SemanticLeafAttribution {
                    reservation_id,
                    visible_openings,
                    blocking_occluders,
                },
            );
            self.semantic_pvs.insert(
                reservation_id,
                SemanticPvsAttribution {
                    reservation_id,
                    potentially_visible_reservations: adjacency
                        .remove(&reservation_id)
                        .unwrap_or_default()
                        .into_iter()
                        .collect(),
                    through_portals: portals
                        .remove(&reservation_id)
                        .unwrap_or_default()
                        .into_iter()
                        .collect(),
                },
            );
        }
    }

    /// All brush IDs that contribute to occlusion.
    pub fn occluder_brush_ids(&self) -> BTreeSet<BrushAssemblyId> {
        self.occluders.keys().copied().collect()
    }

    /// Whether all visibility caps are satisfied.
    pub fn is_valid(&self) -> bool {
        self.caps_satisfied && self.rejections.is_empty()
    }
}

impl Default for VisibilityPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ── Visibility builder ─────────────────────────────────────────────────────

/// Builder that constructs a `VisibilityPlan` from assembly and validates caps.
pub(crate) struct VisibilityBuilder {
    plan: VisibilityPlan,
}

impl VisibilityBuilder {
    pub fn new() -> Self {
        Self {
            plan: VisibilityPlan::new(),
        }
    }

    /// Build from assembly IR.
    pub fn build_from(mut self, ir: &AssemblyIR) -> VisibilityPlan {
        self.plan = VisibilityPlan::build_from_assembly(ir);
        self.plan
    }
}

impl Default for VisibilityBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Validation helpers ─────────────────────────────────────────────────────

/// Validate visibility caps: merges ≤2 per room, ≤6 per cluster.
pub(crate) fn validate_visibility_caps(plan: &VisibilityPlan) -> Result<(), RichnessError> {
    if !plan.caps_satisfied {
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
            "visibility.caps",
            RichnessErrorCategory::PlacementTopologyExhaustion,
            format!("visibility caps exceeded: {}", plan.rejections.join("; ")),
        ));
    }
    for (rid, count) in &plan.merge_count_per_room {
        if *count > plan.max_merges_per_room {
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
                "visibility.caps",
                RichnessErrorCategory::PlacementTopologyExhaustion,
                format!(
                    "room {:?} has {} vertical merges (cap {})",
                    rid.raw(),
                    count,
                    plan.max_merges_per_room
                ),
            ));
        }
    }
    for (cluster, count) in &plan.merge_count_per_cluster {
        if *count > plan.max_merges_per_cluster {
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
                "visibility.caps",
                RichnessErrorCategory::PlacementTopologyExhaustion,
                format!(
                    "cluster {} has {} vertical merges (cap {})",
                    cluster, count, plan.max_merges_per_cluster
                ),
            ));
        }
    }
    Ok(())
}

/// Validate that every emitted convention record uses exactly one of the
/// Phase-05-qualified forms: `hint`+`hintskip`, `skip`, or `func_detail`.
pub(crate) fn validate_compiler_conventions(plan: &VisibilityPlan) -> Result<(), RichnessError> {
    for intent in plan.occluders.values() {
        let Some(convention) = intent.convention else {
            continue;
        };
        let record = plan.compiler_records.get(&intent.brush_id).ok_or_else(|| {
            visibility_error(
                "visibility.compiler",
                format!(
                    "brush {} has convention intent but no emission record",
                    intent.brush_id.raw()
                ),
            )
        })?;
        let valid = match convention {
            CompilerConvention::HintSplit => {
                record.primary == "hint" && record.secondary == Some("hintskip")
            }
            CompilerConvention::SkipSolid => record.primary == "skip" && record.secondary.is_none(),
            CompilerConvention::FuncDetail => {
                record.primary == "func_detail" && record.secondary.is_none()
            }
        };
        if record.convention != convention || !valid {
            return Err(visibility_error(
                "visibility.compiler",
                format!(
                    "brush {} has unqualified compiler emission {:?}/{:?}",
                    intent.brush_id.raw(),
                    record.primary,
                    record.secondary
                ),
            ));
        }
    }
    Ok(())
}

fn visibility_error(path: &str, context: impl Into<String>) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::UnsupportedConvention,
        0,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        path,
        RichnessErrorCategory::ConventionUnsupported,
        context,
    )
}

/// Validate no unintended aligned openings.
pub(crate) fn validate_no_aligned_openings(plan: &VisibilityPlan) -> Result<(), RichnessError> {
    for rejection in &plan.rejections {
        if rejection.contains("openings") && rejection.contains("aligned") {
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
                "visibility.aligned",
                RichnessErrorCategory::PlacementTopologyExhaustion,
                rejection.clone(),
            ));
        }
    }
    Ok(())
}

// ── Occlusion surface emission records ────────────────────────────────────

/// A record linking a compiled leaf or PVS cluster back to semantic intent.
///
/// Stored as metadata; not used for runtime mapping (runtime uses compiled PVS).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SemanticLeafAttribution {
    /// Which room reservation this leaf belongs to.
    pub reservation_id: ReservationId,
    /// Which opening(s) are visible from this leaf.
    pub visible_openings: Vec<OpeningAssemblyId>,
    /// Which occluders block sight from this leaf.
    pub blocking_occluders: Vec<BrushAssemblyId>,
}

/// Semantic adjacency that a later compiled PVS cluster must preserve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SemanticPvsAttribution {
    pub reservation_id: ReservationId,
    pub potentially_visible_reservations: Vec<ReservationId>,
    pub through_portals: Vec<PortalId>,
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::geometry::ConvexBrush;
    use crate::enhanced_v3::richness::assembly::{
        AssemblyIR, BrushAssembly, BrushAssemblyRole, BudgetDimension, CostSource, OpeningRecord,
        SemanticAttribution, SupportTarget,
    };
    use crate::enhanced_v3::richness::ids::{
        ArchetypeIndex, ArchetypeRequestId, BeatId, OpeningAssemblyId, PortalId, ReservationId,
        ZoneId,
    };

    fn make_attr() -> SemanticAttribution {
        SemanticAttribution::from_reservation(
            ReservationId::new(0),
            Some(ArchetypeRequestId::new(0)),
            Some(ArchetypeIndex::new(0)),
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

    #[test]
    fn occluder_kind_from_role_maps_all() {
        let roles = [
            (BrushAssemblyRole::InteriorColumn, OccluderKind::Column),
            (BrushAssemblyRole::WallLiner, OccluderKind::WallLiner),
            (BrushAssemblyRole::Pilaster, OccluderKind::Pilaster),
            (BrushAssemblyRole::Buttress, OccluderKind::Buttress),
            (BrushAssemblyRole::Sill, OccluderKind::Sill),
            (BrushAssemblyRole::PartialWall, OccluderKind::PartialWall),
            (BrushAssemblyRole::BentApproach, OccluderKind::BentApproach),
            (BrushAssemblyRole::OffsetShaft, OccluderKind::OffsetShaft),
            (
                BrushAssemblyRole::PortalPost,
                OccluderKind::PortalPostOcclusion,
            ),
            (
                BrushAssemblyRole::PortalLintel,
                OccluderKind::PortalLintelOcclusion,
            ),
            (
                BrushAssemblyRole::PortalSurround,
                OccluderKind::PortalSurroundOcclusion,
            ),
        ];
        for (role, expected) in &roles {
            assert_eq!(
                OccluderKind::from_role(*role),
                Some(*expected),
                "role {:?}",
                role.tag()
            );
        }
        // Non-occluder roles
        assert_eq!(OccluderKind::from_role(BrushAssemblyRole::FloorSlab), None);
        assert_eq!(OccluderKind::from_role(BrushAssemblyRole::NorthWall), None);
    }

    #[test]
    fn visibility_plan_empty_is_valid() {
        let plan = VisibilityPlan::new();
        assert!(plan.is_valid());
        assert!(plan.occluders.is_empty());
        assert!(plan.vertical_merges.is_empty());
    }

    #[test]
    fn vertical_merge_caps_enforced() {
        let mut plan = VisibilityPlan::new();
        let r0 = ReservationId::new(0);
        let r1 = ReservationId::new(1);
        let r2 = ReservationId::new(2);
        let r3 = ReservationId::new(3);
        let r4 = ReservationId::new(4);
        let r5 = ReservationId::new(5);

        // Room 0 merges with 1 and 2 — that's 2, at cap.
        assert!(plan.try_merge((r0, r1), 0, "adjacent_merge".into()).is_ok());
        assert!(plan.try_merge((r0, r2), 0, "adjacent_merge".into()).is_ok());
        // Room 0 tries to merge with 3 — should fail (would be 3rd merge for room 0)
        assert!(plan
            .try_merge((r0, r3), 0, "adjacent_merge".into())
            .is_err());
        assert!(!plan.caps_satisfied);
    }

    #[test]
    fn vertical_merge_cluster_cap_enforced() {
        let mut plan = VisibilityPlan::new();
        // Cluster 0 gets 6 merges (at cap)
        for i in 0..6 {
            let a = ReservationId::new(i * 2);
            let b = ReservationId::new(i * 2 + 1);
            assert!(plan.try_merge((a, b), 0, "merge".into()).is_ok());
        }
        // 7th merge to same cluster fails
        let a = ReservationId::new(12);
        let b = ReservationId::new(13);
        assert!(plan.try_merge((a, b), 0, "merge".into()).is_err());
        assert!(!plan.caps_satisfied);

        // Different cluster (1) should still work
        let mut plan2 = VisibilityPlan::new();
        assert!(plan2.try_merge((a, b), 1, "merge".into()).is_ok());
    }

    #[test]
    fn build_visibility_plan_from_assembly() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        let col_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: col_id,
            brush: ConvexBrush::make_box((128, 144), (128, 144), (16, 160)).unwrap(),
            role: BrushAssemblyRole::InteriorColumn,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let plan = VisibilityPlan::build_from_assembly(&ir);
        assert_eq!(plan.occluders.len(), 1);
        assert!(plan.occluders.contains_key(&col_id));
        assert_eq!(plan.occluders[&col_id].kind, OccluderKind::Column);
    }

    #[test]
    fn portal_frame_gets_func_detail_convention() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        let post_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: post_id,
            brush: ConvexBrush::make_box((80, 96), (240, 256), (16, 96)).unwrap(),
            role: BrushAssemblyRole::PortalPost,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let opening_id = OpeningAssemblyId::new(0);
        let dummy_brush_id = ir.alloc_brush_id();
        ir.openings.insert(
            opening_id,
            OpeningRecord {
                id: opening_id,
                owner_brush_id: dummy_brush_id,
                wall_segment_ids: vec![dummy_brush_id],
                owner_partition_bounds: (0, 240, 16, 256, 256, 160),
                wall_role: BrushAssemblyRole::NorthWall,
                owner: attr.clone(),
                bounds: (80, 240, 16, 176, 256, 112),
                portal_id: Some(PortalId::new(0)),
                frame_brush_ids: vec![post_id],
                portal_style: None,
            },
        );

        let plan = VisibilityPlan::build_from_assembly(&ir);
        assert_eq!(plan.occluders.len(), 1);
        let intent = &plan.occluders[&post_id];
        assert_eq!(intent.convention, Some(CompilerConvention::FuncDetail));
        assert!(intent.is_portal_frame);
        assert_eq!(intent.portal_id, Some(PortalId::new(0)));
        let record = &plan.compiler_records[&post_id];
        assert_eq!(record.primary, "func_detail");
        assert_eq!(record.secondary, None);
        validate_compiler_conventions(&plan).unwrap();
    }

    #[test]
    fn compiler_records_use_exact_phase05_miptex_names() {
        let mut plan = VisibilityPlan::new();
        let owner = make_attr();
        let hint_id = BrushAssemblyId::new(10);
        let skip_id = BrushAssemblyId::new(11);
        plan.record_occluder(OcclusionIntent {
            brush_id: hint_id,
            kind: OccluderKind::BentApproach,
            convention: Some(CompilerConvention::HintSplit),
            owner: owner.clone(),
            is_portal_frame: false,
            portal_id: None,
            opening_id: None,
        });
        plan.record_occluder(OcclusionIntent {
            brush_id: skip_id,
            kind: OccluderKind::PartialWall,
            convention: Some(CompilerConvention::SkipSolid),
            owner,
            is_portal_frame: false,
            portal_id: None,
            opening_id: None,
        });
        assert_eq!(plan.compiler_records[&hint_id].primary, "hint");
        assert_eq!(plan.compiler_records[&hint_id].secondary, Some("hintskip"));
        assert_eq!(plan.compiler_records[&skip_id].primary, "skip");
        assert_eq!(plan.compiler_records[&skip_id].secondary, None);
        validate_compiler_conventions(&plan).unwrap();
    }

    #[test]
    fn validate_visibility_caps_rejects_exceeded() {
        let mut plan = VisibilityPlan::new();
        let r0 = ReservationId::new(0);
        let r1 = ReservationId::new(1);
        let r2 = ReservationId::new(2);
        let r3 = ReservationId::new(3);
        // 2 merges for room 0 = at cap
        let _ = plan.try_merge((r0, r1), 0, "m1".into());
        let _ = plan.try_merge((r0, r2), 0, "m2".into());
        // This exceeded
        let _ = plan.try_merge((r0, r3), 0, "m3".into());

        // The plan is invalid because caps were exceeded during insertion.
        assert!(!plan.caps_satisfied);
        assert!(!plan.rejections.is_empty());
        assert!(plan.rejections.iter().any(|r| r.contains("room")));
    }

    #[test]
    fn semantic_leaf_attribution_roundtrip() {
        let attr = SemanticLeafAttribution {
            reservation_id: ReservationId::new(0),
            visible_openings: vec![OpeningAssemblyId::new(1)],
            blocking_occluders: vec![BrushAssemblyId::new(2)],
        };
        assert_eq!(attr.reservation_id, ReservationId::new(0));
        assert_eq!(attr.visible_openings.len(), 1);
        assert_eq!(attr.blocking_occluders.len(), 1);
    }
}
