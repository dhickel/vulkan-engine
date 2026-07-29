//! Enhanced v2 theme package — typed, deterministic palette assignment.
//!
//! Every palette, texture identity, role binding, and CC0 provenance record
//! is checked-in typed data. No TOML parsing, filesystem discovery, or
//! runtime asset loading lives in this module.
//!
//! # Responsibilities
//!
//! - Declare the frozen CC0 Dungeon v2 theme package with exact-case ASCII
//!   texture identities, palette closure, and connector palette.
//! - Derive topology-fact room roles (Entry / Hub / DeadEnd / Side) from
//!   committed Phase‑04 topology alone.
//! - Derive zones from ordered topology facts with stable typed IDs.
//! - Assign palettes to rooms, routes, and transitions via Uniform and
//!   ByZone strategies; record every fallback with a reason.

use std::collections::{BTreeMap, VecDeque};

use super::intent::{PaletteId, RoomId, RouteId, TransitionId, ZoneId};
use super::placement::PlacedRoom;
use super::seed::EnhancedStageRng;
use super::topology::TopologyResult;

// ── Texture roles ──────────────────────────────────────────────────────────

/// Visible surface roles that a palette must bind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureRole {
    Floor,
    Wall,
    Ceiling,
    Accent,
}

impl TextureRole {
    /// All four visible roles in canonical order.
    pub const ALL: &[TextureRole] = &[
        TextureRole::Floor,
        TextureRole::Wall,
        TextureRole::Ceiling,
        TextureRole::Accent,
    ];

    /// Roles required for route / stair connector surfaces (no accent).
    pub const ROUTE_ROLES: &[TextureRole] =
        &[TextureRole::Floor, TextureRole::Wall, TextureRole::Ceiling];
}

// ── Room roles (topology facts) ────────────────────────────────────────────

/// Topology-fact role assigned to each room after Phase‑04 commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoomRole {
    /// Designated entry room (lowest RoomId in canonical order).
    Entry,
    /// Highest committed graph degree; RoomId breaks ties.
    Hub,
    /// Graph degree = 1 and not Entry or Hub.
    DeadEnd,
    /// All remaining rooms.
    Side,
}

// ── Assignment strategy ────────────────────────────────────────────────────

/// How palettes are distributed across rooms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignmentStrategy {
    /// Every room receives the theme's base palette.
    Uniform,
    /// Zones receive distinct palettes; cross-zone routes use the connector.
    ByZone,
}

// ── Palette definition ─────────────────────────────────────────────────────

/// Static definition of one CC0 palette.
///
/// Every visible texture identity is exact-case ASCII and carries a
/// 1024×1024 basecolor + normal + gloss companion in the theme's WAD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteDefinition {
    pub id: PaletteId,
    /// Palette name for diagnostics and theme.toml identities.
    pub name: &'static str,
    /// Exact-case WAD miptex name for the floor role.
    pub floor: &'static str,
    /// Exact-case WAD miptex name for the wall role.
    pub wall: &'static str,
    /// Exact-case WAD miptex name for the ceiling role.
    pub ceiling: &'static str,
    /// Exact-case WAD miptex name for the accent role; [`None`] for the
    /// connector palette (cross-zone surfaces carry no accent).
    pub accent: Option<&'static str>,
    /// Whether this palette is the explicit connector palette.
    pub is_connector: bool,
    /// Whether this palette is the declared base fallback.
    pub is_base: bool,
}

impl PaletteDefinition {
    /// Look up the exact-case texture name for a role.
    pub fn texture_for(&self, role: TextureRole) -> Option<&'static str> {
        match role {
            TextureRole::Floor => Some(self.floor),
            TextureRole::Wall => Some(self.wall),
            TextureRole::Ceiling => Some(self.ceiling),
            TextureRole::Accent => self.accent,
        }
    }
}

// ── Theme package ──────────────────────────────────────────────────────────

/// The frozen CC0 Dungeon v2 theme — checked-in typed data.
///
/// All textures are project-authored CC0. The WAD carries exactly one
/// `skip` entry (compiler-only, 64×64, no PBR companions) and every
/// visible texture has matching basecolor / normal / gloss at 1024×1024.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThemePackage {
    /// WAD basename written into generated .map files.
    pub wad_basename: &'static str,
    /// SHA-256 of the canonical WAD2 file (populated post-build).
    pub wad_sha256: [u8; 32],
    /// Every palette including the connector, in canonical order.
    pub palettes: Vec<PaletteDefinition>,
    /// Exact-case name of the compiler-only skip texture.
    pub skip_name: &'static str,
    /// CC0 provenance / attribution text.
    pub cc0_provenance: &'static str,
}

impl ThemePackage {
    /// Return the palette with the given ID.
    pub fn palette(&self, id: PaletteId) -> Option<&PaletteDefinition> {
        self.palettes.iter().find(|p| p.id == id)
    }

    /// Return the base palette (the one with `is_base` set).
    pub fn base_palette(&self) -> &PaletteDefinition {
        self.palettes
            .iter()
            .find(|p| p.is_base)
            .expect("theme package must declare a base palette")
    }

    /// Return the connector palette.
    pub fn connector_palette(&self) -> &PaletteDefinition {
        self.palettes
            .iter()
            .find(|p| p.is_connector)
            .expect("theme package must declare a connector palette")
    }

    /// Number of room-eligible palettes (non-connector).
    pub fn room_palette_count(&self) -> usize {
        self.palettes.iter().filter(|p| !p.is_connector).count()
    }

    /// Resolve a palette ID + role to the exact-case WAD texture name.
    pub fn texture_for(&self, palette_id: PaletteId, role: TextureRole) -> Option<&'static str> {
        self.palette(palette_id).and_then(|p| p.texture_for(role))
    }

    /// Collect every visible (non-skip) WAD texture name in canonical order.
    pub fn visible_texture_names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = Vec::new();
        for palette in &self.palettes {
            names.push(palette.floor);
            names.push(palette.wall);
            names.push(palette.ceiling);
            if let Some(accent) = palette.accent {
                if !names.contains(&accent) {
                    names.push(accent);
                }
            }
        }
        names.sort();
        names.dedup();
        names
    }
}

// ── Frozen CC0 Dungeon v2 theme ────────────────────────────────────────────

/// CC0 provenance text.
const CC0_PROVENANCE: &str =
    "CC0 1.0 Universal — project-authored procedural textures for the Vulkan Engine project.";

/// SHA-256 of `cc0_dungeon_v2.wad` produced by the checked-in deterministic
/// builder. The evidence suite regenerates the WAD and checks this value.
const CC0_DUNGEON_V2_WAD_SHA256: [u8; 32] = [
    0xb8, 0x99, 0xae, 0x81, 0x65, 0x74, 0x4f, 0xdc, 0x2d, 0x92, 0xf9, 0xfe, 0x4d, 0x03, 0xd5, 0xb5,
    0x2c, 0x33, 0xeb, 0xfd, 0xdc, 0x22, 0x88, 0x22, 0xc7, 0x07, 0x90, 0xe1, 0x20, 0xcd, 0x34, 0x96,
];

// Palette IDs are allocated in canonical declaration order.
const PID_BASE_STONE: u32 = 0;
const PID_CRYPT: u32 = 1;
const PID_TREASURY: u32 = 2;
const PID_CONNECTOR: u32 = 3;

/// Build the const palettes vector for the frozen theme.
const fn build_palettes() -> [PaletteDefinition; 4] {
    [
        // base_stone — the declared base fallback palette.
        PaletteDefinition {
            id: PaletteId(PID_BASE_STONE),
            name: "base_stone",
            floor: "bs_floor",
            wall: "bs_wall",
            ceiling: "bs_ceil",
            accent: Some("bs_accent"),
            is_connector: false,
            is_base: true,
        },
        // crypt — dark, damp stone.
        PaletteDefinition {
            id: PaletteId(PID_CRYPT),
            name: "crypt",
            floor: "crypt_floor",
            wall: "crypt_wall",
            ceiling: "crypt_ceil",
            accent: Some("crypt_accent"),
            is_connector: false,
            is_base: false,
        },
        // treasury — warm, dressed stone.
        PaletteDefinition {
            id: PaletteId(PID_TREASURY),
            name: "treasury",
            floor: "treas_floor",
            wall: "treas_wall",
            ceiling: "treas_ceil",
            accent: Some("treas_accent"),
            is_connector: false,
            is_base: false,
        },
        // connector — cross-zone routes and stairs (no accent).
        PaletteDefinition {
            id: PaletteId(PID_CONNECTOR),
            name: "connector",
            floor: "conn_floor",
            wall: "conn_wall",
            ceiling: "conn_ceil",
            accent: None,
            is_connector: true,
            is_base: false,
        },
    ]
}

/// Return a fully populated CC0 Dungeon v2 theme.
///
/// This function exists because `const` Vec construction is not stable;
/// callers should use this once and treat the result as immutable.
pub fn cc0_dungeon_v2_theme() -> ThemePackage {
    ThemePackage {
        wad_basename: "cc0_dungeon_v2.wad",
        wad_sha256: CC0_DUNGEON_V2_WAD_SHA256,
        palettes: build_palettes().to_vec(),
        skip_name: "skip",
        cc0_provenance: CC0_PROVENANCE,
    }
}

// ── Assignment records ─────────────────────────────────────────────────────

/// A single palette assignment for an owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteAssignment {
    pub palette_id: PaletteId,
    pub palette_name: String,
    /// True if this assignment used the base fallback.
    pub is_fallback: bool,
    /// Reason when fallback was applied.
    pub fallback_reason: Option<String>,
}

/// A recorded fallback event.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FallbackRecord {
    /// "room", "route", or "transition".
    pub owner_kind: String,
    /// The typed owner ID as a u32.
    pub owner_id: u32,
    /// The palette that was requested.
    pub requested_palette: String,
    /// The palette that was actually assigned.
    pub fallback_palette: String,
    /// Human-readable reason.
    pub reason: String,
}

/// Complete immutable palette assignment for a generated dungeon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThemeAssignment {
    /// WAD basename written into the .map.
    pub wad_basename: String,
    /// Which strategy produced this assignment.
    pub strategy: AssignmentStrategy,
    /// Room → palette binding (canonical RoomId order).
    pub room_palettes: BTreeMap<RoomId, PaletteAssignment>,
    /// Route → palette binding (canonical RouteId order).
    pub route_palettes: BTreeMap<RouteId, PaletteAssignment>,
    /// Transition → palette binding (canonical TransitionId order).
    pub transition_palettes: BTreeMap<TransitionId, PaletteAssignment>,
    /// Zones produced by topology derivation.
    pub zones: BTreeMap<RoomId, ZoneId>,
    /// Topology-fact roles for every room.
    pub room_roles: BTreeMap<RoomId, RoomRole>,
    /// Every fallback event, ordered by occurrence.
    pub fallbacks: Vec<FallbackRecord>,
}

// ── Role derivation ────────────────────────────────────────────────────────

/// Derive per-room topology-fact roles from committed Phase‑04 topology.
///
/// # Classification rules (priority order)
///
/// 1. **Entry** — room with the lowest `RoomId` in canonical sorted order.
/// 2. **Hub** — among remaining rooms, the one with the highest committed
///    graph degree (incident routes + transitions); lowest `RoomId` breaks
///    ties.
/// 3. **DeadEnd** — remaining rooms with degree = 1.
/// 4. **Side** — all other remaining rooms.
pub fn derive_roles(rooms: &[PlacedRoom], topology: &TopologyResult) -> BTreeMap<RoomId, RoomRole> {
    let degrees = compute_degrees(rooms, topology);

    // Entry: lowest RoomId
    let entry_id = rooms.iter().map(|r| r.id).min().expect("at least one room");

    let mut roles: BTreeMap<RoomId, RoomRole> = BTreeMap::new();
    roles.insert(entry_id, RoomRole::Entry);

    // Hub: highest degree among remaining rooms; RoomId tiebreak
    let mut remaining: Vec<(RoomId, usize)> = rooms
        .iter()
        .filter(|r| r.id != entry_id)
        .map(|r| (r.id, degrees[&r.id]))
        .collect();
    remaining.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    if let Some(&(hub_id, _)) = remaining.first() {
        roles.insert(hub_id, RoomRole::Hub);
    }

    // DeadEnd: degree-1 rooms not yet classified
    for (room_id, &deg) in &degrees {
        if deg == 1 && !roles.contains_key(room_id) {
            roles.insert(*room_id, RoomRole::DeadEnd);
        }
    }

    // Side: everything else
    for room in rooms {
        if !roles.contains_key(&room.id) {
            roles.insert(room.id, RoomRole::Side);
        }
    }

    roles
}

/// Compute the committed graph degree for every room.
fn compute_degrees(rooms: &[PlacedRoom], topology: &TopologyResult) -> BTreeMap<RoomId, usize> {
    let mut degrees: BTreeMap<RoomId, usize> = rooms.iter().map(|r| (r.id, 0)).collect();

    for route in &topology.routes {
        *degrees.get_mut(&route.source_room).unwrap() += 1;
        *degrees.get_mut(&route.target_room).unwrap() += 1;
    }
    for t in &topology.transitions {
        *degrees.get_mut(&t.lower_room).unwrap() += 1;
        *degrees.get_mut(&t.upper_room).unwrap() += 1;
    }

    degrees
}

// ── Zone derivation ────────────────────────────────────────────────────────

/// Derive zones from committed topology facts.
///
/// # Algorithm
///
/// 1. BFS from the entry room (lowest `RoomId`) over the undirected
///    committed graph (routes + transitions).
/// 2. Order rooms by `(BFS distance, RoomId)`.
/// 3. Partition the ordered list into `palette_count` contiguous groups
///    of roughly equal size.
/// 4. Assign a fresh `ZoneId` to each group.
///
/// This is deterministic given the same topology and `palette_count`.
pub fn derive_zones(
    rooms: &[PlacedRoom],
    topology: &TopologyResult,
    _rng: &mut EnhancedStageRng,
    palette_count: usize,
) -> BTreeMap<RoomId, ZoneId> {
    assert!(palette_count > 0, "palette_count must be positive");

    if rooms.is_empty() {
        return BTreeMap::new();
    }

    let entry_id = rooms.iter().map(|r| r.id).min().expect("at least one room");

    // Build adjacency
    let mut adj: BTreeMap<RoomId, Vec<RoomId>> = rooms.iter().map(|r| (r.id, Vec::new())).collect();
    for route in &topology.routes {
        adj.get_mut(&route.source_room)
            .unwrap()
            .push(route.target_room);
        adj.get_mut(&route.target_room)
            .unwrap()
            .push(route.source_room);
    }
    for t in &topology.transitions {
        adj.get_mut(&t.lower_room).unwrap().push(t.upper_room);
        adj.get_mut(&t.upper_room).unwrap().push(t.lower_room);
    }

    // BFS from entry
    let mut distances: BTreeMap<RoomId, u32> = BTreeMap::new();
    let mut queue = VecDeque::new();
    distances.insert(entry_id, 0);
    queue.push_back(entry_id);

    while let Some(current) = queue.pop_front() {
        let dist = distances[&current];
        for &neighbor in &adj[&current] {
            if !distances.contains_key(&neighbor) {
                distances.insert(neighbor, dist + 1);
                queue.push_back(neighbor);
            }
        }
    }

    // Sort by (distance, RoomId)
    let mut ordered: Vec<(u32, RoomId)> = rooms
        .iter()
        .map(|r| (distances.get(&r.id).copied().unwrap_or(u32::MAX), r.id))
        .collect();
    ordered.sort();

    // Partition
    let room_count = ordered.len();
    let per_zone = room_count.div_ceil(palette_count);
    let mut zones: BTreeMap<RoomId, ZoneId> = BTreeMap::new();

    for (i, &(_, room_id)) in ordered.iter().enumerate() {
        let zone_index = (i / per_zone) as u32;
        // Cap at palette_count - 1
        let zone_index = zone_index.min(palette_count as u32 - 1);
        zones.insert(room_id, ZoneId(zone_index));
    }

    zones
}

// ── Palette assignment ─────────────────────────────────────────────────────

/// Assign palettes using the Uniform strategy.
///
/// Every room, route, and transition receives the base palette.
/// No fallbacks are recorded.
pub fn assign_uniform(
    theme: &ThemePackage,
    rooms: &[PlacedRoom],
    topology: &TopologyResult,
) -> ThemeAssignment {
    let base = theme.base_palette();
    let base_assignment = PaletteAssignment {
        palette_id: base.id,
        palette_name: base.name.to_string(),
        is_fallback: false,
        fallback_reason: None,
    };

    let mut room_palettes: BTreeMap<RoomId, PaletteAssignment> = BTreeMap::new();
    for room in rooms {
        room_palettes.insert(room.id, base_assignment.clone());
    }

    let mut route_palettes: BTreeMap<RouteId, PaletteAssignment> = BTreeMap::new();
    for route in &topology.routes {
        route_palettes.insert(route.id, base_assignment.clone());
    }

    let mut transition_palettes: BTreeMap<TransitionId, PaletteAssignment> = BTreeMap::new();
    for t in &topology.transitions {
        transition_palettes.insert(t.id, base_assignment.clone());
    }

    // Under Uniform, zones are trivial: all rooms share one zone.
    let mut zones: BTreeMap<RoomId, ZoneId> = BTreeMap::new();
    for room in rooms {
        zones.insert(room.id, ZoneId(0));
    }

    let room_roles = derive_roles(rooms, topology);

    ThemeAssignment {
        wad_basename: theme.wad_basename.to_string(),
        strategy: AssignmentStrategy::Uniform,
        room_palettes,
        route_palettes,
        transition_palettes,
        zones,
        room_roles,
        fallbacks: Vec::new(),
    }
}

/// Assign palettes using the ByZone strategy.
///
/// # Rules
///
/// 1. Derive zones from topology.
/// 2. Assign each zone a distinct palette from the theme's room palette
///    list (ordered by `PaletteId`).
/// 3. Same-zone routes use the zone's palette.
/// 4. Cross-zone routes use the connector palette.
/// 5. Transitions where both rooms share a zone use the zone palette;
///    otherwise they use the connector palette.
/// 6. If a zone index exceeds the available room palette count, the
///    assignment falls back to the base palette and records the reason.
pub fn assign_by_zone(
    theme: &ThemePackage,
    rooms: &[PlacedRoom],
    topology: &TopologyResult,
    rng: &mut EnhancedStageRng,
) -> ThemeAssignment {
    let room_palette_count = theme.room_palette_count();
    let zones = derive_zones(rooms, topology, rng, room_palette_count);
    let room_roles = derive_roles(rooms, topology);

    // Build a ZoneId → PaletteDefinition map (ordered by PaletteId).
    let room_palettes: Vec<&PaletteDefinition> =
        theme.palettes.iter().filter(|p| !p.is_connector).collect();

    let connector = theme.connector_palette();
    let base = theme.base_palette();
    let mut fallbacks: Vec<FallbackRecord> = Vec::new();

    // ── Room assignments ───────────────────────────────────────────────
    let mut room_palettes_map: BTreeMap<RoomId, PaletteAssignment> = BTreeMap::new();
    for room in rooms {
        let zone_id = zones.get(&room.id).copied().unwrap_or(ZoneId(0));
        let palette = if (zone_id.0 as usize) < room_palettes.len() {
            room_palettes[zone_id.0 as usize]
        } else {
            // Fallback: zone index out of palette range
            fallbacks.push(FallbackRecord {
                owner_kind: "room".into(),
                owner_id: room.id.0,
                requested_palette: format!("zone_{}", zone_id.0),
                fallback_palette: base.name.to_string(),
                reason: format!(
                    "zone {} exceeds {} room palettes; falling back to base",
                    zone_id.0,
                    room_palettes.len()
                ),
            });
            base
        };

        room_palettes_map.insert(
            room.id,
            PaletteAssignment {
                palette_id: palette.id,
                palette_name: palette.name.to_string(),
                is_fallback: palette.id
                    != room_palettes
                        .get(zone_id.0 as usize)
                        .map(|p| p.id)
                        .unwrap_or(base.id),
                fallback_reason: if palette.id == base.id
                    && (zone_id.0 as usize) >= room_palettes.len()
                {
                    Some(format!(
                        "zone {} out of range [0..{})",
                        zone_id.0,
                        room_palettes.len()
                    ))
                } else {
                    None
                },
            },
        );
    }

    // ── Route assignments ──────────────────────────────────────────────
    let mut route_palettes_map: BTreeMap<RouteId, PaletteAssignment> = BTreeMap::new();
    for route in &topology.routes {
        let src_zone = zones.get(&route.source_room).copied();
        let tgt_zone = zones.get(&route.target_room).copied();

        let palette = if src_zone == tgt_zone {
            // Same zone — use the zone palette
            let zid = src_zone.unwrap_or(ZoneId(0));
            if (zid.0 as usize) < room_palettes.len() {
                room_palettes[zid.0 as usize]
            } else {
                fallbacks.push(FallbackRecord {
                    owner_kind: "route".into(),
                    owner_id: route.id.0,
                    requested_palette: format!("zone_{}", zid.0),
                    fallback_palette: base.name.to_string(),
                    reason: format!("zone {} out of range", zid.0),
                });
                base
            }
        } else {
            // Cross-zone — use connector
            connector
        };

        let is_fallback = palette.id == base.id
            && src_zone == tgt_zone
            && src_zone.map(|z| z.0 as usize).unwrap_or(0) >= room_palettes.len();
        route_palettes_map.insert(
            route.id,
            PaletteAssignment {
                palette_id: palette.id,
                palette_name: palette.name.to_string(),
                is_fallback,
                fallback_reason: is_fallback.then(|| {
                    format!(
                        "zone {} out of range [0..{})",
                        src_zone.unwrap_or(ZoneId(0)).0,
                        room_palettes.len()
                    )
                }),
            },
        );
    }

    // ── Transition assignments ─────────────────────────────────────────
    let mut transition_palettes_map: BTreeMap<TransitionId, PaletteAssignment> = BTreeMap::new();
    for t in &topology.transitions {
        let lower_zone = zones.get(&t.lower_room).copied();
        let upper_zone = zones.get(&t.upper_room).copied();

        let palette = if lower_zone == upper_zone {
            let zid = lower_zone.unwrap_or(ZoneId(0));
            if (zid.0 as usize) < room_palettes.len() {
                room_palettes[zid.0 as usize]
            } else {
                fallbacks.push(FallbackRecord {
                    owner_kind: "transition".into(),
                    owner_id: t.id.0,
                    requested_palette: format!("zone_{}", zid.0),
                    fallback_palette: base.name.to_string(),
                    reason: format!("zone {} out of range", zid.0),
                });
                base
            }
        } else {
            connector
        };

        let is_fallback = palette.id == base.id
            && lower_zone == upper_zone
            && lower_zone.map(|z| z.0 as usize).unwrap_or(0) >= room_palettes.len();
        transition_palettes_map.insert(
            t.id,
            PaletteAssignment {
                palette_id: palette.id,
                palette_name: palette.name.to_string(),
                is_fallback,
                fallback_reason: is_fallback.then(|| {
                    format!(
                        "zone {} out of range [0..{})",
                        lower_zone.unwrap_or(ZoneId(0)).0,
                        room_palettes.len()
                    )
                }),
            },
        );
    }

    ThemeAssignment {
        wad_basename: theme.wad_basename.to_string(),
        strategy: AssignmentStrategy::ByZone,
        room_palettes: room_palettes_map,
        route_palettes: route_palettes_map,
        transition_palettes: transition_palettes_map,
        zones,
        room_roles,
        fallbacks,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::config::EnhancedConfig;
    use super::super::placement::place_rooms;
    use super::super::seed::{tags, EnhancedSeed};
    use super::super::topology::build_topology;
    use super::*;

    fn build_nominal_topology(
        seed_val: u64,
    ) -> (
        EnhancedConfig,
        Vec<PlacedRoom>,
        TopologyResult,
        EnhancedStageRng,
    ) {
        let cfg = EnhancedConfig::nominal();
        let eseed = EnhancedSeed::new(seed_val);
        let placement = place_rooms(&cfg, eseed.stage_seed(tags::LAYER_PLACEMENT)).unwrap();
        let mut topo_rng = eseed.stage_seed(tags::VERTICAL_TOPOLOGY).rng();
        let topology = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
        let theme_rng = eseed.stage_seed(tags::THEME_ASSIGNMENT).rng();
        (cfg, placement.rooms, topology, theme_rng)
    }

    #[test]
    fn theme_package_palette_count() {
        let theme = cc0_dungeon_v2_theme();
        assert_eq!(theme.palettes.len(), 4); // 3 room + 1 connector
        assert_eq!(theme.room_palette_count(), 3);
    }

    #[test]
    fn theme_package_base_and_connector() {
        let theme = cc0_dungeon_v2_theme();
        let base = theme.base_palette();
        assert_eq!(base.name, "base_stone");
        assert!(base.is_base);
        assert!(!base.is_connector);

        let conn = theme.connector_palette();
        assert_eq!(conn.name, "connector");
        assert!(conn.is_connector);
        assert!(!conn.is_base);
    }

    #[test]
    fn theme_package_texture_lookup_exact_case() {
        let theme = cc0_dungeon_v2_theme();
        let base = theme.base_palette();
        assert_eq!(
            theme.texture_for(base.id, TextureRole::Floor),
            Some("bs_floor")
        );
        assert_eq!(
            theme.texture_for(base.id, TextureRole::Wall),
            Some("bs_wall")
        );
        assert_eq!(
            theme.texture_for(base.id, TextureRole::Ceiling),
            Some("bs_ceil")
        );
        assert_eq!(
            theme.texture_for(base.id, TextureRole::Accent),
            Some("bs_accent")
        );

        // Connector has no accent
        let conn = theme.connector_palette();
        assert_eq!(theme.texture_for(conn.id, TextureRole::Accent), None);
        assert_eq!(
            theme.texture_for(conn.id, TextureRole::Floor),
            Some("conn_floor")
        );
    }

    #[test]
    fn theme_package_visible_texture_names() {
        let theme = cc0_dungeon_v2_theme();
        let names = theme.visible_texture_names();
        // 3 room palettes × 4 roles + 1 connector × 3 roles = 15 unique names
        assert_eq!(names.len(), 15);
        // All must be lowercase ASCII
        for name in &names {
            assert!(
                name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "texture name {name} not lowercase ASCII"
            );
        }
    }

    // ── Role derivation tests ──────────────────────────────────────────

    #[test]
    fn role_derivation_entry_is_lowest_room_id() {
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let roles = derive_roles(&rooms, &topology);
        let min_id = rooms.iter().map(|r| r.id).min().unwrap();
        assert_eq!(roles[&min_id], RoomRole::Entry);
    }

    #[test]
    fn role_derivation_all_rooms_classified() {
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let roles = derive_roles(&rooms, &topology);
        assert_eq!(roles.len(), rooms.len());
        for room in &rooms {
            assert!(
                roles.contains_key(&room.id),
                "room {:?} not classified",
                room.id
            );
        }
    }

    #[test]
    fn role_derivation_dead_end_degree_one() {
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let degrees = compute_degrees(&rooms, &topology);
        let roles = derive_roles(&rooms, &topology);
        for (room_id, &deg) in &degrees {
            if deg == 1 {
                let role = roles[room_id];
                assert!(
                    role == RoomRole::DeadEnd || role == RoomRole::Entry,
                    "degree-1 room {:?} role {:?}",
                    room_id,
                    role
                );
            }
        }
    }

    #[test]
    fn role_derivation_deterministic() {
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let a = derive_roles(&rooms, &topology);
        let b = derive_roles(&rooms, &topology);
        assert_eq!(a, b);
    }

    #[test]
    fn role_derivation_hub_is_highest_degree() {
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let degrees = compute_degrees(&rooms, &topology);
        let roles = derive_roles(&rooms, &topology);

        let entry_id = rooms.iter().map(|r| r.id).min().unwrap();
        let hub_id = roles
            .iter()
            .find(|(_, &r)| r == RoomRole::Hub)
            .map(|(id, _)| *id)
            .unwrap();

        let hub_deg = degrees[&hub_id];
        for (room_id, &deg) in &degrees {
            if *room_id != entry_id && *room_id != hub_id {
                assert!(
                    deg <= hub_deg,
                    "room {:?} has degree {} > hub degree {}",
                    room_id,
                    deg,
                    hub_deg
                );
            }
        }
    }

    // ── Zone derivation tests ──────────────────────────────────────────

    #[test]
    fn zone_derivation_covers_all_rooms() {
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        let zones = derive_zones(&rooms, &topology, &mut rng, 3);
        assert_eq!(zones.len(), rooms.len());
        for room in &rooms {
            assert!(zones.contains_key(&room.id));
        }
    }

    #[test]
    fn zone_derivation_deterministic() {
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let mut rng_a = EnhancedSeed::new(42)
            .stage_seed(tags::THEME_ASSIGNMENT)
            .rng();
        let mut rng_b = EnhancedSeed::new(42)
            .stage_seed(tags::THEME_ASSIGNMENT)
            .rng();
        let a = derive_zones(&rooms, &topology, &mut rng_a, 3);
        let b = derive_zones(&rooms, &topology, &mut rng_b, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn zone_derivation_palette_count_bounds() {
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        // With 3 palettes, zone IDs should be 0, 1, 2
        let zones = derive_zones(&rooms, &topology, &mut rng, 3);
        let mut seen: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for z in zones.values() {
            seen.insert(z.0);
        }
        assert!(seen.len() <= 3);
        for z in seen {
            assert!(z < 3);
        }
    }

    #[test]
    fn zone_derivation_single_palette() {
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        let zones = derive_zones(&rooms, &topology, &mut rng, 1);
        // All rooms in zone 0
        for z in zones.values() {
            assert_eq!(z.0, 0);
        }
    }

    // ── Uniform assignment tests ───────────────────────────────────────

    #[test]
    fn uniform_assignment_all_rooms_base() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let assignment = assign_uniform(&theme, &rooms, &topology);

        let base = theme.base_palette();
        for room in &rooms {
            let pa = &assignment.room_palettes[&room.id];
            assert_eq!(pa.palette_id, base.id);
            assert!(!pa.is_fallback);
        }

        for route in &topology.routes {
            let pa = &assignment.route_palettes[&route.id];
            assert_eq!(pa.palette_id, base.id);
            assert!(!pa.is_fallback);
        }

        for t in &topology.transitions {
            let pa = &assignment.transition_palettes[&t.id];
            assert_eq!(pa.palette_id, base.id);
            assert!(!pa.is_fallback);
        }

        assert!(assignment.fallbacks.is_empty());
        assert_eq!(assignment.strategy, AssignmentStrategy::Uniform);
    }

    #[test]
    fn uniform_assignment_deterministic() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let a = assign_uniform(&theme, &rooms, &topology);
        let b = assign_uniform(&theme, &rooms, &topology);
        assert_eq!(a, b);
    }

    #[test]
    fn uniform_assignment_no_fallback_closure() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let assignment = assign_uniform(&theme, &rooms, &topology);

        // Every assigned palette must exist in the theme
        for pa in assignment.room_palettes.values() {
            assert!(theme.palette(pa.palette_id).is_some());
        }
        for pa in assignment.route_palettes.values() {
            assert!(theme.palette(pa.palette_id).is_some());
        }
        for pa in assignment.transition_palettes.values() {
            assert!(theme.palette(pa.palette_id).is_some());
        }
    }

    // ── ByZone assignment tests ────────────────────────────────────────

    #[test]
    fn byzone_assignment_all_rooms_have_palette() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        let assignment = assign_by_zone(&theme, &rooms, &topology, &mut rng);

        assert_eq!(assignment.strategy, AssignmentStrategy::ByZone);
        for room in &rooms {
            assert!(assignment.room_palettes.contains_key(&room.id));
        }
        for route in &topology.routes {
            assert!(assignment.route_palettes.contains_key(&route.id));
        }
        for t in &topology.transitions {
            assert!(assignment.transition_palettes.contains_key(&t.id));
        }
    }

    #[test]
    fn byzone_assignment_deterministic() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, _) = build_nominal_topology(42);
        let mut rng_a = EnhancedSeed::new(42)
            .stage_seed(tags::THEME_ASSIGNMENT)
            .rng();
        let mut rng_b = EnhancedSeed::new(42)
            .stage_seed(tags::THEME_ASSIGNMENT)
            .rng();
        let a = assign_by_zone(&theme, &rooms, &topology, &mut rng_a);
        let b = assign_by_zone(&theme, &rooms, &topology, &mut rng_b);
        assert_eq!(a, b);
    }

    #[test]
    fn byzone_cross_zone_routes_use_connector() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        let assignment = assign_by_zone(&theme, &rooms, &topology, &mut rng);
        let connector = theme.connector_palette();

        // At least some routes should be cross-zone
        let mut has_cross_zone = false;
        for route in &topology.routes {
            let src_zone = assignment.zones[&route.source_room];
            let tgt_zone = assignment.zones[&route.target_room];
            if src_zone != tgt_zone {
                has_cross_zone = true;
                let pa = &assignment.route_palettes[&route.id];
                assert_eq!(
                    pa.palette_id, connector.id,
                    "cross-zone route {:?} (zones {:?}→{:?}) must use connector",
                    route.id, src_zone, tgt_zone
                );
            }
        }
        // With multiple zones, it's extremely likely some routes are cross-zone
        // But if all rooms end up in one zone, that's OK too
        if !has_cross_zone {
            // All rooms in one zone → no cross-zone routes → all routes use zone palette
            for route in &topology.routes {
                let pa = &assignment.route_palettes[&route.id];
                assert_ne!(pa.palette_id, connector.id);
            }
        }
    }

    #[test]
    fn byzone_same_zone_routes_use_zone_palette() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        let assignment = assign_by_zone(&theme, &rooms, &topology, &mut rng);

        for route in &topology.routes {
            let src_zone = assignment.zones[&route.source_room];
            let tgt_zone = assignment.zones[&route.target_room];
            let pa = &assignment.route_palettes[&route.id];

            if src_zone == tgt_zone {
                // Same zone — should use a room palette, not connector
                assert!(
                    theme
                        .palette(pa.palette_id)
                        .map(|p| !p.is_connector)
                        .unwrap_or(false),
                    "same-zone route {:?} should not use connector",
                    route.id
                );
            }
        }
    }

    #[test]
    fn byzone_transition_connector_ownership() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        let assignment = assign_by_zone(&theme, &rooms, &topology, &mut rng);
        let connector = theme.connector_palette();

        for t in &topology.transitions {
            let lower_zone = assignment.zones[&t.lower_room];
            let upper_zone = assignment.zones[&t.upper_room];
            let pa = &assignment.transition_palettes[&t.id];

            if lower_zone != upper_zone {
                assert_eq!(
                    pa.palette_id, connector.id,
                    "cross-zone transition {:?} must use connector",
                    t.id
                );
            }
        }
    }

    #[test]
    fn byzone_fallback_records_when_zone_out_of_range() {
        let theme = cc0_dungeon_v2_theme();
        // Use only 1 room palette => zones beyond 0 must fallback
        let small_theme = ThemePackage {
            palettes: vec![
                theme.base_palette().clone(),
                theme.connector_palette().clone(),
            ],
            ..theme.clone()
        };
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        let assignment = assign_by_zone(&small_theme, &rooms, &topology, &mut rng);

        // All assignments should resolve to a palette that exists
        for pa in assignment.room_palettes.values() {
            assert!(small_theme.palette(pa.palette_id).is_some());
        }
        // Check fallbacks recorded
        let room_fallbacks: Vec<_> = assignment
            .fallbacks
            .iter()
            .filter(|f| f.owner_kind == "room")
            .collect();
        // If any zone ≥ 1 exists, there should be room fallbacks
        let has_high_zone = assignment
            .zones
            .values()
            .any(|z| z.0 >= small_theme.room_palette_count() as u32);
        if has_high_zone {
            assert!(
                !room_fallbacks.is_empty(),
                "expected room fallbacks when zones exceed palette count"
            );
        }
    }

    #[test]
    fn byzone_no_connector_fallback_on_cross_zone() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        let assignment = assign_by_zone(&theme, &rooms, &topology, &mut rng);
        let connector = theme.connector_palette();

        // Cross-zone routes/stair should use connector directly, never fallback
        for route in &topology.routes {
            let src_zone = assignment.zones[&route.source_room];
            let tgt_zone = assignment.zones[&route.target_room];
            if src_zone != tgt_zone {
                let pa = &assignment.route_palettes[&route.id];
                assert_eq!(pa.palette_id, connector.id);
                assert!(!pa.is_fallback);
            }
        }
    }

    #[test]
    fn byzone_room_roles_present() {
        let theme = cc0_dungeon_v2_theme();
        let (_, rooms, topology, mut rng) = build_nominal_topology(42);
        let assignment = assign_by_zone(&theme, &rooms, &topology, &mut rng);

        assert_eq!(assignment.room_roles.len(), rooms.len());
        // At minimum we should see Entry role
        let has_entry = assignment
            .room_roles
            .values()
            .any(|r| *r == RoomRole::Entry);
        assert!(has_entry);
    }
}
