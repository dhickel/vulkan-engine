//! Richness assembly IR: canonical brush, entity, interface, and support
//! records with semantic room/opening attribution, source-cost attribution,
//! and protected-volume ownership.
//!
//! Every brush knows its semantic owner (room/archetype/beat/zone) and cost
//! source (budget dimension + recipe priority). Interfaces record exact
//! positive-area face contacts between brushes. Support records track the
//! acyclic support DAG back to world surfaces.
//!
//! # Contract
//!
//! - All IDs are typed newtypes (Copy + Eq + Ord + Hash).
//! - Brushes are non-overlapping convex solids with approved normals.
//! - Every opening is an omission from a unique ownership partition.
//! - Floor/ceiling slabs own their full partition beneath/above walls.
//! - No floats; all geometry uses the baseline exact rational kernel.
//! - Crate-private; no baseline changes.

use std::collections::BTreeMap;

use crate::enhanced_v3::geometry::{CanonicalPlane, ConvexBrush};

use super::generated_content;
use super::ids::{
    ArchetypeIndex, ArchetypeRequestId, BeatId, BrushAssemblyId, EntityAssemblyId,
    InterfaceAssemblyId, OpeningAssemblyId, ReservationId, SupportAssemblyId, WallChainId, ZoneId,
};
use super::theme::SemanticRole;

// ── Semantic attribution ──────────────────────────────────────────────────

/// Which semantic entity owns a brush, entity, or opening.
///
/// Every assembly record carries this attribution so compiled leaves/PVS
/// can be mapped back to semantic intent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SemanticAttribution {
    /// The reservation this assembly element belongs to.
    pub reservation_id: ReservationId,
    /// The archetype request this fulfills.
    pub request_id: Option<ArchetypeRequestId>,
    /// The archetype identity.
    pub archetype: Option<ArchetypeIndex>,
    /// The pacing beat (if any).
    pub beat_id: Option<BeatId>,
    /// The semantic zone.
    pub zone_id: Option<ZoneId>,
}

impl SemanticAttribution {
    /// Create attribution from a reservation record's fields.
    pub fn from_reservation(
        reservation_id: ReservationId,
        request_id: Option<ArchetypeRequestId>,
        archetype: Option<ArchetypeIndex>,
        beat_id: Option<BeatId>,
        zone_id: Option<ZoneId>,
    ) -> Self {
        Self {
            reservation_id,
            request_id,
            archetype,
            beat_id,
            zone_id,
        }
    }

    /// The stable archetype ID string, if known and present in the frozen catalog.
    pub fn archetype_id_str(&self) -> Option<&'static str> {
        self.archetype.and_then(|archetype| {
            generated_content::ARCHETYPE_IDS
                .get(archetype.raw() as usize)
                .copied()
        })
    }
}

// ── Cost source ────────────────────────────────────────────────────────────

/// Which budget dimension and recipe priority a brush or entity was
/// sourced from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum BudgetDimension {
    /// Source `.map` faces.
    SourceFaces,
    /// Convex brushes.
    Brushes,
    /// Point / brush entities.
    Entities,
    /// Light entities.
    Lights,
    /// Support contacts.
    SupportContacts,
}

impl BudgetDimension {
    pub fn tag(self) -> &'static str {
        match self {
            Self::SourceFaces => "source_faces",
            Self::Brushes => "brushes",
            Self::Entities => "entities",
            Self::Lights => "lights",
            Self::SupportContacts => "support_contacts",
        }
    }
}

/// A complete cost-source attribution for a brush or entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct CostSource {
    /// The budget dimension this item consumes.
    pub dimension: BudgetDimension,
    /// Predicted face count for this item.
    pub face_count: u32,
}

// ── Brush role ────────────────────────────────────────────────────────────

/// The structural role of a brush in the assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum BrushAssemblyRole {
    /// Full-footprint floor slab (z=0..16).
    FloorSlab,
    /// Full-footprint ceiling slab (z=160..176).
    CeilingSlab,
    /// North cardinal wall (normal faces -Y).
    NorthWall,
    /// South cardinal wall (normal faces +Y).
    SouthWall,
    /// East cardinal wall (normal faces -X).
    EastWall,
    /// West cardinal wall (normal faces +X).
    WestWall,
    /// NE diagonal wall at chamfer corner.
    DiagNEWall,
    /// NW diagonal wall at chamfer corner.
    DiagNWWall,
    /// SE diagonal wall at chamfer corner.
    DiagSEWall,
    /// SW diagonal wall at chamfer corner.
    DiagSWWall,
    /// Interior column (pillar, post).
    InteriorColumn,
    /// Interior massing (buttress, dais, block).
    InteriorMass,
    /// Structural vault rib.
    VaultRib,
    /// Monolithic solid (for solid-filled kill courts, arenas).
    MonolithSolid,
    /// Portal post (vertical framing column).
    PortalPost,
    /// Portal lintel (horizontal beam above throat).
    PortalLintel,
    /// Portal surround (decorative frame around throat).
    PortalSurround,
    /// Wall liner (inward mass course).
    WallLiner,
    /// Engaged pilaster on a wall.
    Pilaster,
    /// Wall recess (carved inward opening).
    WallRecess,
    /// External buttress.
    Buttress,
    /// Window/overlook sill.
    Sill,
    /// Bent approach (angled wall section).
    BentApproach,
    /// Partial wall segment.
    PartialWall,
    /// Offset shaft (vertical conduit).
    OffsetShaft,
}

impl BrushAssemblyRole {
    pub fn tag(self) -> &'static str {
        match self {
            Self::FloorSlab => "floor_slab",
            Self::CeilingSlab => "ceiling_slab",
            Self::NorthWall => "north_wall",
            Self::SouthWall => "south_wall",
            Self::EastWall => "east_wall",
            Self::WestWall => "west_wall",
            Self::DiagNEWall => "diag_ne_wall",
            Self::DiagNWWall => "diag_nw_wall",
            Self::DiagSEWall => "diag_se_wall",
            Self::DiagSWWall => "diag_sw_wall",
            Self::InteriorColumn => "interior_column",
            Self::InteriorMass => "interior_mass",
            Self::VaultRib => "vault_rib",
            Self::MonolithSolid => "monolith_solid",
            Self::PortalPost => "portal_post",
            Self::PortalLintel => "portal_lintel",
            Self::PortalSurround => "portal_surround",
            Self::WallLiner => "wall_liner",
            Self::Pilaster => "pilaster",
            Self::WallRecess => "wall_recess",
            Self::Buttress => "buttress",
            Self::Sill => "sill",
            Self::BentApproach => "bent_approach",
            Self::PartialWall => "partial_wall",
            Self::OffsetShaft => "offset_shaft",
        }
    }

    /// Whether this role is a wall (cardinal or diagonal).
    pub fn is_wall(self) -> bool {
        matches!(
            self,
            Self::NorthWall
                | Self::SouthWall
                | Self::EastWall
                | Self::WestWall
                | Self::DiagNEWall
                | Self::DiagNWWall
                | Self::DiagSEWall
                | Self::DiagSWWall
        )
    }

    /// Whether this role is a floor or ceiling slab.
    pub fn is_slab(self) -> bool {
        matches!(self, Self::FloorSlab | Self::CeilingSlab)
    }

    /// Whether this role owns the full footprint partition beneath/above walls.
    pub fn owns_full_partition(self) -> bool {
        matches!(self, Self::FloorSlab | Self::CeilingSlab)
    }

    /// The authored theme role that is legal for this structural brush role.
    pub fn semantic_role(self) -> SemanticRole {
        match self {
            Self::FloorSlab => SemanticRole::Floor,
            Self::CeilingSlab => SemanticRole::Ceiling,
            Self::NorthWall
            | Self::SouthWall
            | Self::EastWall
            | Self::WestWall
            | Self::DiagNEWall
            | Self::DiagNWWall
            | Self::DiagSEWall
            | Self::DiagSWWall
            | Self::WallLiner
            | Self::WallRecess
            | Self::PartialWall => SemanticRole::Wall,
            Self::PortalPost | Self::PortalLintel | Self::PortalSurround => SemanticRole::Portal,
            Self::OffsetShaft => SemanticRole::Vertical,
            Self::InteriorColumn
            | Self::InteriorMass
            | Self::VaultRib
            | Self::MonolithSolid
            | Self::Pilaster
            | Self::Buttress
            | Self::Sill
            | Self::BentApproach => SemanticRole::Accent,
        }
    }
}

// ── Assembly brush ────────────────────────────────────────────────────────

/// A single convex brush in the assembly IR.
///
/// Carries its exact convex geometry (via the baseline `ConvexBrush`),
/// structural role, semantic ownership, cost source, and support target.
#[derive(Debug, Clone)]
pub(crate) struct BrushAssembly {
    /// Unique brush ID.
    pub id: BrushAssemblyId,
    /// The exact convex brush geometry.
    pub brush: ConvexBrush,
    /// Structural role.
    pub role: BrushAssemblyRole,
    /// Semantic owner.
    pub owner: SemanticAttribution,
    /// Cost source.
    pub cost: CostSource,
    /// Support target (which brush supports this one, or World).
    pub support: SupportTarget,
}

/// What supports a brush.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum SupportTarget {
    /// Supported by the world (ground / bedrock).
    World,
    /// Supported by another assembly brush.
    Brush(BrushAssemblyId),
}

impl SupportTarget {
    pub fn is_world(&self) -> bool {
        matches!(self, Self::World)
    }
}

// ── Assembly entity ───────────────────────────────────────────────────────

/// A point or brush entity in the assembly IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EntityAssembly {
    /// Unique entity ID.
    pub id: EntityAssemblyId,
    /// Entity classname (e.g. "info_player_start", "light").
    pub classname: String,
    /// Origin in Quake coordinates.
    pub origin: (i128, i128, i128),
    /// Semantic owner.
    pub owner: SemanticAttribution,
    /// Cost source.
    pub cost: CostSource,
}

// ── Interface record ──────────────────────────────────────────────────────

/// An exact positive-area face contact between two brushes.
///
/// Interfaces are derived from positive-area contacts and explicit omissions.
/// Every contact between brushes that is not at a declared opening must have
/// an interface record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InterfaceRecord {
    /// Unique interface ID.
    pub id: InterfaceAssemblyId,
    /// First brush.
    pub brush_a: BrushAssemblyId,
    /// Second brush.
    pub brush_b: BrushAssemblyId,
    /// The kind of contact.
    pub kind: InterfaceKind,
}

/// The structural kind of face contact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum InterfaceKind {
    /// Wall bottom contacts floor top.
    WallToFloor,
    /// Wall top contacts ceiling bottom.
    WallToCeiling,
    /// Cardinal wall meets cardinal wall at a corner.
    WallToWallCorner,
    /// Cardinal wall meets diagonal wall at a chamfer joint.
    WallToDiagJoint,
    /// Column bottom contacts floor top or slab.
    ColumnToFloor,
    /// Massing contacts wall.
    MassToWall,
    /// Massing contacts floor.
    MassToFloor,
    /// Portal post contacts floor.
    PostToFloor,
    /// Portal post contacts wall.
    PostToWall,
    /// Portal lintel contacts post.
    LintelToPost,
    /// Portal lintel contacts wall.
    LintelToWall,
    /// Portal surround contacts wall.
    SurroundToWall,
    /// Liner contacts wall.
    LinerToWall,
    /// Pilaster contacts wall.
    PilasterToWall,
    /// Buttress contacts wall.
    ButtressToWall,
    /// Sill contacts wall.
    SillToWall,
    /// Sill contacts floor.
    SillToFloor,
    /// Bent approach contacts wall.
    BentApproachToWall,
    /// Partial wall meets cardinal wall.
    PartialWallToWall,
    /// Offset shaft contacts floor.
    ShaftToFloor,
    /// Offset shaft contacts wall.
    ShaftToWall,
    /// Two split segments belonging to one cardinal wall partition.
    WallSegmentJoint,
    /// Adjacent floor or ceiling slab runs.
    SlabRunJoint,
    /// Two pieces of one portal frame contact.
    PortalFrameJoint,
    /// Portal surround rests on a floor slab.
    SurroundToFloor,
    /// Shared wall contacts shared wall (same chain, adjacent rooms).
    SharedWallContact,
}

// ── Support record ────────────────────────────────────────────────────────

/// A directed support edge in the support DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SupportRecord {
    /// Unique support ID.
    pub id: SupportAssemblyId,
    /// The dependent brush (the one being supported).
    pub child: BrushAssemblyId,
    /// The supporting brush or world.
    pub parent: SupportTarget,
}

// ── Opening record ────────────────────────────────────────────────────────

/// An opening omission from a unique ownership partition.
///
/// Openings are NOT brushes — they are omitted volumes in the wall/floor/ceiling
/// that an owner brush is responsible for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpeningRecord {
    /// Unique opening ID.
    pub id: OpeningAssemblyId,
    /// Canonical brush identity for the one logical wall partition that owns
    /// this omission. It is one of `wall_segment_ids` after splitting.
    pub owner_brush_id: BrushAssemblyId,
    /// All emitted wall segments that realize the owning partition.
    pub wall_segment_ids: Vec<BrushAssemblyId>,
    /// Bounds of the unsplit owning wall partition.
    pub owner_partition_bounds: (i128, i128, i128, i128, i128, i128),
    /// Cardinal wall containing the omission.
    pub wall_role: BrushAssemblyRole,
    /// The semantic owner of the opening.
    pub owner: SemanticAttribution,
    /// Exact clear-throat bounds in Quake coordinates
    /// `(x0, y0, z0, x1, y1, z1)`.
    pub bounds: (i128, i128, i128, i128, i128, i128),
    /// The portal this opening connects to.
    pub portal_id: Option<super::ids::PortalId>,
    /// Brush IDs for portal frame elements (posts, lintel, surround).
    pub frame_brush_ids: Vec<BrushAssemblyId>,
    /// The portal style used for this opening.
    pub portal_style: Option<PortalStyle>,
}

// ── Portal style ──────────────────────────────────────────────────────────

/// Reserved portal construction styles.
///
/// All portals are CARDINAL. No diagonal portals are permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum PortalStyle {
    /// Ancient post-and-lintel: two 16u posts + 16u lintel framing a 64×80 throat.
    AncientPostLintel,
    /// Egyptian stepped surround: stepped batter courses framing the throat.
    EgyptianSteppedSurround,
    /// Brutalist reveal/surround: reveal channels + surround mass.
    BrutalistRevealSurround,
}

impl PortalStyle {
    pub fn tag(self) -> &'static str {
        match self {
            Self::AncientPostLintel => "ancient_post_lintel",
            Self::EgyptianSteppedSurround => "egyptian_stepped_surround",
            Self::BrutalistRevealSurround => "brutalist_reveal_surround",
        }
    }
}

// ── Portal assembly ───────────────────────────────────────────────────────

/// A constructed portal with all frame brushes.
#[derive(Debug, Clone)]
pub(crate) struct PortalAssembly {
    /// The portal ID this assembly realizes.
    pub portal_id: super::ids::PortalId,
    /// The portal style.
    pub style: PortalStyle,
    /// Post brush IDs (typically 2 for post-and-lintel).
    pub post_ids: Vec<BrushAssemblyId>,
    /// Lintel brush ID(s).
    pub lintel_ids: Vec<BrushAssemblyId>,
    /// Surround brush IDs.
    pub surround_ids: Vec<BrushAssemblyId>,
    /// Canonical segment of the wall partition that owns the omission.
    pub wall_brush_id: BrushAssemblyId,
    /// The opening omission ID.
    pub opening_id: OpeningAssemblyId,
    /// Witness bounds of the clear portal throat (exactly 64×80).
    pub throat_bounds: (i128, i128, i128, i128, i128, i128),
}

/// One physical wall run shared by two touching room reservations.
///
/// The owner brush is emitted once. Both rooms consume `shared_plane`; no
/// second wall brush or independently rounded plane is permitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SharedWallChainAssembly {
    pub id: WallChainId,
    pub owner_reservation_id: ReservationId,
    pub sharing_reservation_id: ReservationId,
    pub owner_brush_id: BrushAssemblyId,
    pub shared_plane: CanonicalPlane,
    pub span: (i128, i128),
}

// ── Assembly IR ────────────────────────────────────────────────────────────

/// The complete assembly intermediate representation.
///
/// Produced by macro composition and consumed by portal construction (session B)
/// and visibility/validation (session C).
#[derive(Debug, Clone)]
pub(crate) struct AssemblyIR {
    /// All assembly brushes, canonically ordered by ID.
    pub brushes: BTreeMap<BrushAssemblyId, BrushAssembly>,
    /// All entities.
    pub entities: BTreeMap<EntityAssemblyId, EntityAssembly>,
    /// All interfaces between brushes.
    pub interfaces: BTreeMap<InterfaceAssemblyId, InterfaceRecord>,
    /// All support edges.
    pub supports: BTreeMap<SupportAssemblyId, SupportRecord>,
    /// All opening omissions (populated in session B).
    pub openings: BTreeMap<OpeningAssemblyId, OpeningRecord>,
    /// All constructed portal assemblies (session B).
    pub portal_assemblies: Vec<PortalAssembly>,
    /// Canonical one-owner shared-wall runs.
    pub shared_wall_chains: BTreeMap<WallChainId, SharedWallChainAssembly>,
    /// Concrete theme-role assignment for every emitted brush.
    pub material_roles: BTreeMap<BrushAssemblyId, SemanticRole>,

    // ── ID allocators ──────────────────────────────────────────────────
    next_brush_id: u32,
    next_entity_id: u32,
    next_interface_id: u32,
    next_support_id: u32,
    next_opening_id: u32,
    next_wall_chain_id: u32,
}

impl AssemblyIR {
    /// Create an empty assembly IR.
    pub fn new() -> Self {
        Self {
            brushes: BTreeMap::new(),
            entities: BTreeMap::new(),
            interfaces: BTreeMap::new(),
            supports: BTreeMap::new(),
            openings: BTreeMap::new(),
            portal_assemblies: Vec::new(),
            shared_wall_chains: BTreeMap::new(),
            material_roles: BTreeMap::new(),
            next_brush_id: 0,
            next_entity_id: 0,
            next_interface_id: 0,
            next_support_id: 0,
            next_opening_id: 0,
            next_wall_chain_id: 0,
        }
    }

    // ── ID allocation ──────────────────────────────────────────────────

    pub fn alloc_brush_id(&mut self) -> BrushAssemblyId {
        let id = BrushAssemblyId::new(self.next_brush_id);
        self.next_brush_id += 1;
        id
    }

    pub fn alloc_entity_id(&mut self) -> EntityAssemblyId {
        let id = EntityAssemblyId::new(self.next_entity_id);
        self.next_entity_id += 1;
        id
    }

    pub fn alloc_interface_id(&mut self) -> InterfaceAssemblyId {
        let id = InterfaceAssemblyId::new(self.next_interface_id);
        self.next_interface_id += 1;
        id
    }

    pub fn alloc_support_id(&mut self) -> SupportAssemblyId {
        let id = SupportAssemblyId::new(self.next_support_id);
        self.next_support_id += 1;
        id
    }

    pub fn alloc_opening_id(&mut self) -> OpeningAssemblyId {
        let id = OpeningAssemblyId::new(self.next_opening_id);
        self.next_opening_id += 1;
        id
    }

    pub fn alloc_wall_chain_id(&mut self) -> WallChainId {
        let id = WallChainId::new(self.next_wall_chain_id);
        self.next_wall_chain_id += 1;
        id
    }

    // ── Insertion ──────────────────────────────────────────────────────

    /// Insert a brush, bind its role-valid theme material, and return its ID.
    pub fn insert_brush(&mut self, mut brush: BrushAssembly) -> BrushAssemblyId {
        let id = brush.id;
        brush.cost.face_count = brush.brush.faces.len() as u32;
        self.material_roles.insert(id, brush.role.semantic_role());
        self.brushes.insert(id, brush);
        id
    }

    /// Remove a brush and every directly keyed assignment for it.
    pub fn remove_brush(&mut self, id: BrushAssemblyId) -> Option<BrushAssembly> {
        self.material_roles.remove(&id);
        self.brushes.remove(&id)
    }

    /// Insert an entity and return its ID.
    pub fn insert_entity(&mut self, entity: EntityAssembly) -> EntityAssemblyId {
        let id = entity.id;
        self.entities.insert(id, entity);
        id
    }

    /// Insert an interface and return its ID.
    pub fn insert_interface(&mut self, interface: InterfaceRecord) -> InterfaceAssemblyId {
        let id = interface.id;
        self.interfaces.insert(id, interface);
        id
    }

    /// Insert a support record and return its ID.
    pub fn insert_support(&mut self, support: SupportRecord) -> SupportAssemblyId {
        let id = support.id;
        self.supports.insert(id, support);
        id
    }

    /// Insert an opening record and return its ID.
    pub fn insert_opening(&mut self, opening: OpeningRecord) -> OpeningAssemblyId {
        let id = opening.id;
        self.openings.insert(id, opening);
        id
    }

    pub fn insert_shared_wall_chain(&mut self, chain: SharedWallChainAssembly) -> WallChainId {
        let id = chain.id;
        self.shared_wall_chains.insert(id, chain);
        id
    }

    // ── Queries ────────────────────────────────────────────────────────

    /// All brushes belonging to a given reservation.
    pub fn brushes_for_reservation(&self, reservation_id: ReservationId) -> Vec<&BrushAssembly> {
        self.brushes
            .values()
            .filter(|b| b.owner.reservation_id == reservation_id)
            .collect()
    }

    /// Total brush count.
    pub fn brush_count(&self) -> usize {
        self.brushes.len()
    }

    /// Total entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Total interface count.
    pub fn interface_count(&self) -> usize {
        self.interfaces.len()
    }

    /// Total support count.
    pub fn support_count(&self) -> usize {
        self.supports.len()
    }
}

impl Default for AssemblyIR {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembly_ids_are_distinct_types() {
        let b = BrushAssemblyId::new(1);
        let e = EntityAssemblyId::new(1);
        let i = InterfaceAssemblyId::new(1);
        let s = SupportAssemblyId::new(1);
        let o = OpeningAssemblyId::new(1);

        assert_eq!(b.raw(), 1);
        assert_eq!(e.raw(), 1);
        assert_eq!(i.raw(), 1);
        assert_eq!(s.raw(), 1);
        assert_eq!(o.raw(), 1);
    }

    #[test]
    fn semantic_attribution_from_reservation() {
        let attr = SemanticAttribution::from_reservation(
            ReservationId::new(0),
            Some(ArchetypeRequestId::new(2)),
            Some(ArchetypeIndex::new(2)),
            Some(BeatId::new(1)),
            Some(ZoneId::new(0)),
        );
        assert_eq!(attr.reservation_id, ReservationId::new(0));
        assert!(attr.archetype.is_some());
        assert!(attr.archetype_id_str().is_some());
    }

    #[test]
    fn semantic_attribution_no_request_has_no_archetype() {
        let attr = SemanticAttribution::from_reservation(
            ReservationId::new(1),
            None,
            None,
            Some(BeatId::new(2)),
            None,
        );
        assert_eq!(attr.reservation_id, ReservationId::new(1));
        assert!(attr.archetype.is_none());
        assert!(attr.archetype_id_str().is_none());
    }

    #[test]
    fn brush_role_tags_are_unique() {
        let roles = [
            BrushAssemblyRole::FloorSlab,
            BrushAssemblyRole::CeilingSlab,
            BrushAssemblyRole::NorthWall,
            BrushAssemblyRole::SouthWall,
            BrushAssemblyRole::EastWall,
            BrushAssemblyRole::WestWall,
            BrushAssemblyRole::DiagNEWall,
            BrushAssemblyRole::DiagNWWall,
            BrushAssemblyRole::DiagSEWall,
            BrushAssemblyRole::DiagSWWall,
            BrushAssemblyRole::InteriorColumn,
            BrushAssemblyRole::InteriorMass,
            BrushAssemblyRole::VaultRib,
            BrushAssemblyRole::MonolithSolid,
            BrushAssemblyRole::PortalPost,
            BrushAssemblyRole::PortalLintel,
            BrushAssemblyRole::PortalSurround,
            BrushAssemblyRole::WallLiner,
            BrushAssemblyRole::Pilaster,
            BrushAssemblyRole::WallRecess,
            BrushAssemblyRole::Buttress,
            BrushAssemblyRole::Sill,
            BrushAssemblyRole::BentApproach,
            BrushAssemblyRole::PartialWall,
            BrushAssemblyRole::OffsetShaft,
        ];
        let tags: Vec<_> = roles.iter().map(|r| r.tag()).collect();
        let set: std::collections::BTreeSet<_> = tags.iter().collect();
        assert_eq!(set.len(), tags.len(), "duplicate role tags");
    }

    #[test]
    fn wall_roles_identify_correctly() {
        assert!(BrushAssemblyRole::NorthWall.is_wall());
        assert!(BrushAssemblyRole::DiagSWWall.is_wall());
        assert!(!BrushAssemblyRole::FloorSlab.is_wall());
        assert!(!BrushAssemblyRole::InteriorColumn.is_wall());
    }

    #[test]
    fn slab_roles_own_full_partition() {
        assert!(BrushAssemblyRole::FloorSlab.owns_full_partition());
        assert!(BrushAssemblyRole::CeilingSlab.owns_full_partition());
        assert!(!BrushAssemblyRole::NorthWall.owns_full_partition());
    }

    #[test]
    fn assembly_ir_default_is_empty() {
        let ir = AssemblyIR::new();
        assert_eq!(ir.brush_count(), 0);
        assert_eq!(ir.entity_count(), 0);
        assert_eq!(ir.interface_count(), 0);
        assert_eq!(ir.support_count(), 0);
    }

    #[test]
    fn assembly_ir_id_allocation_is_sequential() {
        let mut ir = AssemblyIR::new();
        let b0 = ir.alloc_brush_id();
        let b1 = ir.alloc_brush_id();
        assert_eq!(b0.raw(), 0);
        assert_eq!(b1.raw(), 1);

        let e0 = ir.alloc_entity_id();
        assert_eq!(e0.raw(), 0);
    }

    #[test]
    fn assembly_ir_insert_brush_roundtrip() {
        let mut ir = AssemblyIR::new();

        let attr =
            SemanticAttribution::from_reservation(ReservationId::new(0), None, None, None, None);

        let cb = ConvexBrush::make_box((0, 256), (0, 256), (0, 16)).unwrap();
        let brush_id = ir.alloc_brush_id();
        let brush = BrushAssembly {
            id: brush_id,
            brush: cb,
            role: BrushAssemblyRole::FloorSlab,
            owner: attr,
            cost: CostSource {
                dimension: BudgetDimension::SourceFaces,
                face_count: 6,
            },
            support: SupportTarget::World,
        };
        let id = ir.insert_brush(brush);
        assert_eq!(id.raw(), 0);
        assert_eq!(ir.brush_count(), 1);
        assert!(ir.brushes.get(&id).is_some());
    }

    #[test]
    fn brushes_for_reservation_filters_correctly() {
        let mut ir = AssemblyIR::new();

        let attr_a =
            SemanticAttribution::from_reservation(ReservationId::new(0), None, None, None, None);
        let attr_b =
            SemanticAttribution::from_reservation(ReservationId::new(1), None, None, None, None);

        let cb = ConvexBrush::make_box((0, 64), (0, 64), (0, 16)).unwrap();

        let bid_a = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: bid_a,
            brush: cb.clone(),
            role: BrushAssemblyRole::FloorSlab,
            owner: attr_a,
            cost: CostSource {
                dimension: BudgetDimension::SourceFaces,
                face_count: 6,
            },
            support: SupportTarget::World,
        });

        let bid_b = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: bid_b,
            brush: cb,
            role: BrushAssemblyRole::CeilingSlab,
            owner: attr_b,
            cost: CostSource {
                dimension: BudgetDimension::SourceFaces,
                face_count: 6,
            },
            support: SupportTarget::World,
        });

        assert_eq!(ir.brushes_for_reservation(ReservationId::new(0)).len(), 1);
        assert_eq!(ir.brushes_for_reservation(ReservationId::new(1)).len(), 1);
        assert_eq!(ir.brushes_for_reservation(ReservationId::new(99)).len(), 0);
    }
}
