//! Immutable intermediate representations produced by the dungeon generator
//! pipeline.
//!
//! Each struct is a pure data record — construction and validation happen at
//! the pipeline stage that produces it. None of these types contain generation
//! logic.

/// A placed room whose position and dimensions are snapped to the 16-unit
/// construction quantum (see [`CONSTRUCTION_QUANTUM`]).
///
/// [`CONSTRUCTION_QUANTUM`]: crate::config::CONSTRUCTION_QUANTUM
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RoomIntent {
    /// Minimum-corner position of the room in Quake units `(x, y, z)`.
    pub position: (i32, i32, i32),
    /// Room dimensions in Quake units `(dx, dy, dz)`; all are multiples of
    /// the construction quantum.
    pub dimensions: (u32, u32, u32),
}

/// The placed-room layout: rooms, their connectivity graph, and the target
/// number of spatial loops.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayoutIntent {
    /// All placed rooms in placement order.
    pub rooms: Vec<RoomIntent>,
    /// Undirected connectivity edges: each `(usize, usize)` indexes into
    /// `rooms`.
    pub edges: Vec<(usize, usize)>,
    /// Target number of redundant spatial loops.
    pub loop_count: u32,
}

/// An axis-aligned straight corridor segment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Corridor {
    /// Start position in Quake units `(x, y, z)` — one end of the corridor
    /// center-line.
    pub start: (i32, i32, i32),
    /// End position in Quake units `(x, y, z)` — the opposite end of the
    /// corridor center-line.
    pub end: (i32, i32, i32),
    /// Clear interior width in Quake units (≥ 64 for walkable).
    pub width: u32,
    /// Clear interior height in Quake units (≥ 80 for walkable).
    pub height: u32,
}

/// A junction point where two or more corridors meet.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Junction {
    /// Junction center position in Quake units `(x, y, z)`.
    pub position: (i32, i32, i32),
}

/// The routed corridor network: materialized corridors, junction nodes, and
/// room-corridor portal references.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RoutedIntent {
    /// All corridor segments.
    pub corridors: Vec<Corridor>,
    /// All junction nodes.
    pub junctions: Vec<Junction>,
}

/// A single face of a convex brush, defined by three non-collinear points on
/// its plane and associated texture mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BrushFace {
    /// Three non-collinear integer points `(x, y, z)` defining the plane.
    pub plane_points: [(i32, i32, i32); 3],
    /// Texture name (references a WAD entry).
    pub texture: String,
    /// U-axis texture mapping `[u_x, u_y, u_z, u_offset]`.
    pub u_axis: [i32; 4],
    /// V-axis texture mapping `[v_x, v_y, v_z, v_offset]`.
    pub v_axis: [i32; 4],
}

/// A convex brush: an ordered set of faces.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Brush {
    /// Faces in creation-index order.
    pub faces: Vec<BrushFace>,
}

/// A Quake entity with classname, origin, key-value properties, and optional
/// brush geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EntityIntent {
    /// Entity classname (e.g. `"worldspawn"`, `"info_player_start"`).
    pub classname: String,
    /// Origin in Quake units `(x, y, z)`.
    pub origin: (i32, i32, i32),
    /// Additional key-value properties (alphabetically ordered).
    pub properties: Vec<(String, String)>,
    /// Brushes owned by this entity (empty for point entities).
    pub brushes: Vec<Brush>,
}

/// The final emission-ready representation: all brushes, entities, and the
/// WAD texture reference.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmissionIntent {
    /// Brushes belonging to the `worldspawn` entity.
    pub brushes: Vec<Brush>,
    /// All non-worldspawn entities in creation-index order.
    pub entities: Vec<EntityIntent>,
    /// WAD texture archive basename referenced by the emitted `.map`.
    pub wad: String,
}
