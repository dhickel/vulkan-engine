use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::determinism::lowercase_hex;
use super::error::{ErrorStage, GeneratorError};

// ─── Constants ──────────────────────────────────────────────────────────────

const CATALOG_DOMAIN: &[u8] = b"dungeon-generator/prefab-catalog/v1";
const CATALOG_VERSION: u32 = 1;
const FORMAT_VERSION: u32 = 1;
const ALLOWED_ROTATIONS: [u16; 4] = [0, 90, 180, 270];

// ─── Direction ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    fn rotate_cw(self) -> Self {
        match self {
            Self::North => Self::East,
            Self::East => Self::South,
            Self::South => Self::West,
            Self::West => Self::North,
        }
    }

    pub(super) fn rotate(self, quarter_turns: u8) -> Self {
        let mut d = self;
        for _ in 0..(quarter_turns % 4) {
            d = d.rotate_cw();
        }
        d
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "north" => Some(Self::North),
            "east" => Some(Self::East),
            "south" => Some(Self::South),
            "west" => Some(Self::West),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::North => "north",
            Self::East => "east",
            Self::South => "south",
            Self::West => "west",
        }
    }

    fn as_ramp_char(self) -> char {
        match self {
            Self::North => '^',
            Self::East => '>',
            Self::South => 'v',
            Self::West => '<',
        }
    }

    fn from_ramp_char(c: char) -> Option<Self> {
        match c {
            '^' => Some(Self::North),
            '>' => Some(Self::East),
            'v' => Some(Self::South),
            '<' => Some(Self::West),
            _ => None,
        }
    }

    /// Unit delta for advancing one cell in this direction.
    fn delta(self) -> (i32, i32) {
        match self {
            Self::North => (0, -1),
            Self::East => (1, 0),
            Self::South => (0, 1),
            Self::West => (-1, 0),
        }
    }
}

// ─── Tile ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tile {
    Wall,
    Floor,
    Void,
    Ramp { direction: Direction, step: u8 },
}

impl Tile {
    pub(super) fn is_walkable(self) -> bool {
        matches!(self, Self::Floor | Self::Ramp { .. })
    }

    pub(super) fn is_void(self) -> bool {
        matches!(self, Self::Void)
    }

    pub(super) fn is_ramp(self) -> bool {
        matches!(self, Self::Ramp { .. })
    }

    /// Parse a single token from the beginning of a row string.
    /// Returns (tile, bytes_consumed).
    fn parse_token(s: &str) -> Option<(Self, usize)> {
        let bytes = s.as_bytes();
        if bytes.is_empty() {
            return None;
        }
        // Try ramp tokens: R[0-2][^>v<]
        if bytes.len() >= 3 && bytes[0] == b'R' && bytes[1].is_ascii_digit() && bytes[1] <= b'2' {
            if let Some(dir) = Direction::from_ramp_char(bytes[2] as char) {
                let step = bytes[1] - b'0';
                return Some((Self::Ramp { direction: dir, step }, 3));
            }
        }
        // Single-char tokens
        match bytes[0] {
            b'#' => Some((Self::Wall, 1)),
            b'.' => Some((Self::Floor, 1)),
            b'_' => Some((Self::Void, 1)),
            _ => None,
        }
    }

    /// Tokenize an entire row string.
    fn tokenize_row(row: &str) -> Option<Vec<Self>> {
        let mut tiles = Vec::new();
        let mut pos = 0;
        while pos < row.len() {
            let (tile, consumed) = Self::parse_token(&row[pos..])?;
            tiles.push(tile);
            pos += consumed;
        }
        Some(tiles)
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wall => write!(f, "#"),
            Self::Floor => write!(f, "."),
            Self::Void => write!(f, "_"),
            Self::Ramp { direction, step } => {
                write!(f, "R{}{}", step, direction.as_ramp_char())
            }
        }
    }
}

// ─── Enums from TOML strings ────────────────────────────────────────────────

macro_rules! enum_from_str {
    ($vis:vis enum $name:ident { $($variant:ident => $str:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        $vis enum $name { $($variant),+ }

        impl $name {
            fn from_str(s: &str) -> Option<Self> {
                match s {
                    $($str => Some(Self::$variant),)+
                    _ => None,
                }
            }

            fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $str,)+
                }
            }
        }
    };
}

enum_from_str!(pub(crate) enum SocketRole {
    Corridor => "corridor",
    Hall => "hall",
    Doorway => "doorway",
    Junction => "junction",
    DeadEnd => "dead_end",
    LandmarkApproach => "landmark_approach",
    LowerRampApproach => "lower_ramp_approach",
    UpperLanding => "upper_landing",
});

enum_from_str!(enum MarkerKind {
    Prop => "prop",
    Light => "light",
    Spawn => "spawn",
    Loot => "loot",
    Hazard => "hazard",
    Landmark => "landmark",
});

enum_from_str!(enum ReservationKind {
    Footprint => "footprint",
    WallShell => "wall_shell",
    SocketFunnel => "socket_funnel",
    CorridorApproach => "corridor_approach",
    RampVolume => "ramp_volume",
    UpperOpening => "upper_opening",
    UpperLanding => "upper_landing",
    Headroom => "headroom",
});

impl ReservationKind {
    /// Returns the required tile for this reservation kind, if one is mandated.
    fn required_tile(self) -> Option<Tile> {
        match self {
            Self::WallShell => Some(Tile::Wall),
            Self::UpperOpening | Self::Headroom => Some(Tile::Void),
            Self::Footprint | Self::SocketFunnel | Self::CorridorApproach | Self::UpperLanding => {
                Some(Tile::Floor)
            }
            Self::RampVolume => None, // Must be Ramp of any direction/step
        }
    }
}

// ─── Coordinates ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Cell {
    pub(crate) layer: u16,
    pub(crate) x: u16,
    pub(crate) y: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CellBox {
    layer: u16,
    x_min: u16,
    y_min: u16,
    x_max: u16,
    y_max: u16,
}

impl CellBox {
    fn cells(self) -> Vec<Cell> {
        let box_width = self.x_max as usize - self.x_min as usize + 1;
        let box_height = self.y_max as usize - self.y_min as usize + 1;
        let mut out = Vec::with_capacity(box_width.saturating_mul(box_height));
        for y in self.y_min..=self.y_max {
            for x in self.x_min..=self.x_max {
                out.push(Cell { layer: self.layer, x, y });
            }
        }
        out
    }
}

// ─── Reservation owner ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ReservationOwner {
    Self_,
    Reference(String),
}

impl ReservationOwner {
    fn from_str(s: &str) -> Self {
        if s == "self" {
            Self::Self_
        } else {
            // Socket and transition owners share one globally unique stable-ID
            // namespace, so preserving the reference string is sufficient.
            Self::Reference(s.to_owned())
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Self_ => "self",
            Self::Reference(id) => id,
        }
    }
}

// ─── TOML deserialization types ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CellDef {
    layer: u16,
    x: u16,
    y: u16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoxDef {
    layer: u16,
    x_min: u16,
    y_min: u16,
    x_max: u16,
    y_max: u16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginDef {
    x: u16,
    y: u16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerDef {
    rows: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SocketDef {
    id: String,
    anchor: CellDef,
    direction: String,
    width: u16,
    role: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MarkerDef {
    id: String,
    position: CellDef,
    facing: String,
    kind: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReservationDef {
    owner: String,
    kind: String,
    #[serde(default)]
    cells: Option<Vec<CellDef>>,
    #[serde(default, rename = "box")]
    box_def: Option<BoxDef>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransitionDef {
    id: String,
    lower_approach_socket: String,
    upper_landing_socket: String,
    upper_layer: u16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefabToml {
    format_version: u32,
    id: String,
    #[serde(default)]
    layer_count: Option<u16>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    layers: Vec<LayerDef>,
    #[serde(default)]
    origin: Option<OriginDef>,
    #[serde(default)]
    rotations: Vec<u16>,
    #[serde(default)]
    sockets: Vec<SocketDef>,
    #[serde(default)]
    markers: Vec<MarkerDef>,
    #[serde(default)]
    reservations: Vec<ReservationDef>,
    #[serde(default)]
    transitions: Vec<TransitionDef>,
}

// ─── Validated domain types ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Socket {
    pub(crate) id: String,
    pub(crate) anchor: Cell,
    pub(crate) direction: Direction,
    pub(crate) width: u16,
    pub(crate) role: SocketRole,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Marker {
    pub(crate) id: String,
    pub(crate) position: Cell,
    pub(crate) facing: Direction,
    pub(crate) kind: MarkerKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reservation {
    pub(crate) owner: ReservationOwner,
    pub(crate) kind: ReservationKind,
    pub(crate) cells: Vec<Cell>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Transition {
    pub(crate) id: String,
    pub(crate) lower_approach_socket: String,
    pub(crate) upper_landing_socket: String,
    pub(crate) upper_layer: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Prefab {
    pub(crate) id: String,
    pub(crate) format_version: u32,
    pub(crate) layer_count: u16,
    pub(crate) tags: Vec<String>,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) layers: Vec<Vec<Vec<Tile>>>,
    pub(crate) origin: Cell,
    pub(crate) rotations: Vec<u16>,
    pub(crate) sockets: Vec<Socket>,
    pub(crate) markers: Vec<Marker>,
    pub(crate) reservations: Vec<Reservation>,
    pub(crate) transitions: Vec<Transition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PrefabVariant {
    pub(crate) base_id: String,
    pub(crate) rotation_degrees: u16,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) layer_count: u16,
    pub(crate) layers: Vec<Vec<Vec<Tile>>>,
    pub(crate) origin: Cell,
    pub(crate) sockets: Vec<Socket>,
    pub(crate) markers: Vec<Marker>,
    pub(crate) reservations: Vec<Reservation>,
    pub(crate) transitions: Vec<Transition>,
    pub(crate) tags: Vec<String>,
}

// ─── Ramp inference types ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct InferredRamp {
    /// (x, y) of R0, R1, R2 on the lower layer
    cells: [(u16, u16); 3],
    lower_layer: u16,
}

// ─── Helper: prefab error ───────────────────────────────────────────────────

fn prefab_err(context: &str, reason: &'static str) -> GeneratorError {
    GeneratorError::PrefabIntegrity {
        stage: ErrorStage::Prefab,
        context: context.to_owned(),
        reason,
    }
}

// ─── Parsing layers ─────────────────────────────────────────────────────────

fn parse_layers(
    defs: &[LayerDef],
    explicit_count: Option<u16>,
    context: &str,
) -> Result<(u16, u16, u16, Vec<Vec<Vec<Tile>>>), GeneratorError> {
    if defs.is_empty() {
        return Err(prefab_err(context, "no_layers"));
    }
    let parsed_count = defs.len();
    if let Some(n) = explicit_count {
        if n as usize != parsed_count {
            return Err(prefab_err(context, "layer_count_mismatch"));
        }
    }
    let layer_count_u16 = u16::try_from(parsed_count).map_err(|_| {
        GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Prefab,
            operation: "layer_count_conversion",
        }
    })?;

    let mut layers: Vec<Vec<Vec<Tile>>> = Vec::with_capacity(parsed_count);
    let mut width: Option<u16> = None;
    let mut height: Option<u16> = None;

    for (li, layer_def) in defs.iter().enumerate() {
        if layer_def.rows.is_empty() {
            return Err(prefab_err(context, "empty_layer"));
        }
        let h = u16::try_from(layer_def.rows.len()).map_err(|_| {
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::Prefab,
                operation: "row_count_conversion",
            }
        })?;
        if h == 0 {
            return Err(prefab_err(context, "zero_height"));
        }

        let mut grid: Vec<Vec<Tile>> = Vec::with_capacity(h as usize);
        let mut w: Option<u16> = None;

        for (ri, row_str) in layer_def.rows.iter().enumerate() {
            let tiles = Tile::tokenize_row(row_str).ok_or_else(|| {
                GeneratorError::PrefabIntegrity {
                    stage: ErrorStage::Prefab,
                    context: format!("{} layer={} row={}", context, li, ri),
                    reason: "invalid_token",
                }
            })?;
            if tiles.is_empty() {
                return Err(GeneratorError::PrefabIntegrity {
                    stage: ErrorStage::Prefab,
                    context: format!("{} layer={} row={}", context, li, ri),
                    reason: "empty_row",
                });
            }
            let row_w = u16::try_from(tiles.len()).map_err(|_| {
                GeneratorError::ArithmeticOverflow {
                    stage: ErrorStage::Prefab,
                    operation: "column_count_conversion",
                }
            })?;
            match w {
                None => w = Some(row_w),
                Some(prev) if prev != row_w => {
                    return Err(GeneratorError::PrefabIntegrity {
                        stage: ErrorStage::Prefab,
                        context: format!("{} layer={} row={}", context, li, ri),
                        reason: "ragged_row",
                    });
                }
                _ => {}
            }
            grid.push(tiles);
        }

        let w = w.unwrap_or(0);
        match width {
            None => width = Some(w),
            Some(prev) if prev != w => {
                return Err(prefab_err(context, "layer_width_mismatch"));
            }
            _ => {}
        }
        match height {
            None => height = Some(h),
            Some(prev) if prev != h => {
                return Err(prefab_err(context, "layer_height_mismatch"));
            }
            _ => {}
        }

        layers.push(grid);
    }

    let w = width.unwrap_or(0);
    let h = height.unwrap_or(0);
    if w == 0 || h == 0 {
        return Err(prefab_err(context, "zero_dimensions"));
    }
    Ok((w, h, layer_count_u16, layers))
}

// ─── Validation ─────────────────────────────────────────────────────────────

fn validate_tags(tags: &[String], context: &str) -> Result<(), GeneratorError> {
    let mut seen = BTreeSet::new();
    for tag in tags {
        if tag.is_empty() {
            return Err(prefab_err(context, "empty_tag"));
        }
        if !seen.insert(tag) {
            return Err(prefab_err(context, "duplicate_tag"));
        }
    }
    Ok(())
}

fn validate_rotations(rots: &[u16], context: &str) -> Result<Vec<u16>, GeneratorError> {
    if rots.is_empty() {
        return Err(prefab_err(context, "no_rotations"));
    }
    let allowed: BTreeSet<u16> = ALLOWED_ROTATIONS.iter().copied().collect();
    let mut seen = BTreeSet::new();
    let mut sorted: Vec<u16> = Vec::with_capacity(rots.len());
    for &r in rots {
        if !allowed.contains(&r) {
            return Err(prefab_err(context, "invalid_rotation"));
        }
        if !seen.insert(r) {
            return Err(prefab_err(context, "duplicate_rotation"));
        }
    }
    // Sort numerically; caller's order is presentation only; we canonicalize.
    for r in ALLOWED_ROTATIONS {
        if seen.contains(&r) {
            sorted.push(r);
        }
    }
    Ok(sorted)
}

fn validate_origin(
    origin_def: Option<&OriginDef>,
    width: u16,
    height: u16,
    context: &str,
) -> Result<Cell, GeneratorError> {
    let origin = origin_def.map_or(Cell { layer: 0, x: 0, y: 0 }, |o| Cell {
        layer: 0,
        x: o.x,
        y: o.y,
    });
    if origin.x >= width || origin.y >= height {
        return Err(prefab_err(context, "origin_out_of_bounds"));
    }
    Ok(origin)
}

fn validate_sockets(
    socket_defs: &[SocketDef],
    width: u16,
    height: u16,
    layer_count: u16,
    layers: &[Vec<Vec<Tile>>],
    context: &str,
) -> Result<Vec<Socket>, GeneratorError> {
    let mut ids = BTreeSet::new();
    let mut sockets = Vec::with_capacity(socket_defs.len());

    for sd in socket_defs {
        if sd.id.is_empty() {
            return Err(prefab_err(context, "empty_socket_id"));
        }
        if !ids.insert(&sd.id) {
            return Err(prefab_err(context, "duplicate_socket_id"));
        }

        let direction = Direction::from_str(&sd.direction)
            .ok_or_else(|| prefab_err(context, "invalid_socket_direction"))?;
        let role = SocketRole::from_str(&sd.role)
            .ok_or_else(|| prefab_err(context, "invalid_socket_role"))?;
        if sd.width == 0 {
            return Err(prefab_err(context, "socket_width_zero"));
        }

        let anchor = Cell { layer: sd.anchor.layer, x: sd.anchor.x, y: sd.anchor.y };

        if anchor.layer >= layer_count {
            return Err(prefab_err(context, "socket_layer_out_of_bounds"));
        }
        if anchor.x >= width || anchor.y >= height {
            return Err(prefab_err(context, "socket_anchor_out_of_bounds"));
        }

        // Socket aperture must be on the boundary.
        let on_boundary = match direction {
            Direction::North => anchor.y == 0,
            Direction::South => anchor.y == height.saturating_sub(1),
            Direction::West => anchor.x == 0,
            Direction::East => anchor.x == width.saturating_sub(1),
        };
        if !on_boundary {
            return Err(prefab_err(context, "socket_not_on_boundary"));
        }

        // Width must fit along the boundary.
        let (fit, _) = match direction {
            Direction::North | Direction::South => {
                let end = anchor.x.checked_add(sd.width).ok_or_else(|| {
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Prefab,
                        operation: "socket_width_fit",
                    }
                })?;
                (end <= width, end)
            }
            Direction::West | Direction::East => {
                let end = anchor.y.checked_add(sd.width).ok_or_else(|| {
                    GeneratorError::ArithmeticOverflow {
                        stage: ErrorStage::Prefab,
                        operation: "socket_width_fit",
                    }
                })?;
                (end <= height, end)
            }
        };
        if !fit {
            return Err(prefab_err(context, "socket_width_exceeds_boundary"));
        }

        // Aperture cells on boundary must be walkable. Verify inward cell
        // exists (is in-bounds) but do not require walkability — ramp
        // landing sockets may have void openings on their inward side.
        let tile_at_anchor = layers[anchor.layer as usize][anchor.y as usize][anchor.x as usize];
        if !tile_at_anchor.is_walkable() {
            return Err(prefab_err(context, "socket_anchor_not_walkable"));
        }

        let (dx, dy) = direction.delta();
        let inward_x = anchor.x as i32 - dx;
        let inward_y = anchor.y as i32 - dy;
        if inward_x < 0 || inward_y < 0 {
            return Err(prefab_err(context, "socket_no_inward_cell"));
        }
        let ix = inward_x as u16;
        let iy = inward_y as u16;
        if ix >= width || iy >= height {
            return Err(prefab_err(context, "socket_no_inward_cell"));
        }
        if layers[anchor.layer as usize][iy as usize][ix as usize] == Tile::Wall {
            return Err(prefab_err(context, "socket_inward_blocked"));
        }

        // For width > 1, check the additional aperture cells are walkable
        // and have inward cells in-bounds.
        for w_off in 1..sd.width {
            let (ax, ay) = match direction {
                Direction::North | Direction::South => (anchor.x + w_off, anchor.y),
                Direction::West | Direction::East => (anchor.x, anchor.y + w_off),
            };
            let at = layers[anchor.layer as usize][ay as usize][ax as usize];
            if !at.is_walkable() {
                return Err(prefab_err(context, "socket_aperture_not_walkable"));
            }
            let iix = ax as i32 - dx;
            let iiy = ay as i32 - dy;
            if iix < 0 || iiy < 0 || iix as u16 >= width || iiy as u16 >= height {
                return Err(prefab_err(context, "socket_no_inward_cell"));
            }
            if layers[anchor.layer as usize][iiy as usize][iix as usize] == Tile::Wall {
                return Err(prefab_err(context, "socket_inward_blocked"));
            }
        }

        sockets.push(Socket { id: sd.id.clone(), anchor, direction, width: sd.width, role });
    }

    Ok(sockets)
}

fn validate_markers(
    marker_defs: &[MarkerDef],
    width: u16,
    height: u16,
    layer_count: u16,
    context: &str,
) -> Result<Vec<Marker>, GeneratorError> {
    let mut ids = BTreeSet::new();
    let mut markers = Vec::with_capacity(marker_defs.len());

    for md in marker_defs {
        if md.id.is_empty() {
            return Err(prefab_err(context, "empty_marker_id"));
        }
        if !ids.insert(&md.id) {
            return Err(prefab_err(context, "duplicate_marker_id"));
        }
        let facing = Direction::from_str(&md.facing)
            .ok_or_else(|| prefab_err(context, "invalid_marker_facing"))?;
        let kind = MarkerKind::from_str(&md.kind)
            .ok_or_else(|| prefab_err(context, "invalid_marker_kind"))?;
        let position = Cell { layer: md.position.layer, x: md.position.x, y: md.position.y };
        if position.layer >= layer_count || position.x >= width || position.y >= height {
            return Err(prefab_err(context, "marker_out_of_bounds"));
        }
        markers.push(Marker { id: md.id.clone(), position, facing, kind });
    }
    Ok(markers)
}

fn expand_reservation(
    def: &ReservationDef,
    width: u16,
    height: u16,
    layer_count: u16,
    context: &str,
) -> Result<Vec<Cell>, GeneratorError> {
    match (&def.cells, &def.box_def) {
        (Some(_), Some(_)) => Err(prefab_err(context, "reservation_cells_and_box")),
        (None, None) => Err(prefab_err(context, "reservation_no_cells_or_box")),
        (Some(cells), None) => {
            if cells.is_empty() {
                return Err(prefab_err(context, "reservation_empty_cells"));
            }
            let mut out = Vec::with_capacity(cells.len());
            for c in cells {
                out.push(Cell { layer: c.layer, x: c.x, y: c.y });
            }
            // Sort and deduplicate
            out.sort();
            out.dedup();
            Ok(out)
        }
        (None, Some(b)) => {
            if b.x_min > b.x_max || b.y_min > b.y_max {
                return Err(prefab_err(context, "reservation_invalid_box"));
            }
            // Reject before expansion so malformed boxes cannot trigger huge
            // allocations or integer overflow.
            if b.layer >= layer_count || b.x_max >= width || b.y_max >= height {
                return Err(prefab_err(context, "reservation_out_of_bounds"));
            }
            Ok(CellBox {
                layer: b.layer,
                x_min: b.x_min,
                y_min: b.y_min,
                x_max: b.x_max,
                y_max: b.y_max,
            }
            .cells())
        }
    }
}

fn validate_reservations(
    reservation_defs: &[ReservationDef],
    width: u16,
    height: u16,
    layer_count: u16,
    layers: &[Vec<Vec<Tile>>],
    context: &str,
) -> Result<Vec<Reservation>, GeneratorError> {
    let mut reservations = Vec::with_capacity(reservation_defs.len());

    for rd in reservation_defs {
        let kind = ReservationKind::from_str(&rd.kind)
            .ok_or_else(|| prefab_err(context, "invalid_reservation_kind"))?;
        let owner = ReservationOwner::from_str(&rd.owner);
        let cells = expand_reservation(rd, width, height, layer_count, context)?;

        // Check bounds and tile types
        for c in &cells {
            if c.layer >= layer_count || c.x >= width || c.y >= height {
                return Err(prefab_err(context, "reservation_out_of_bounds"));
            }
            let tile = layers[c.layer as usize][c.y as usize][c.x as usize];
            if let Some(required) = kind.required_tile() {
                if tile != required {
                    return Err(prefab_err(context, "reservation_wrong_tile"));
                }
            }
            // For RampVolume, must be a ramp tile of any kind
            if kind == ReservationKind::RampVolume && !tile.is_ramp() {
                return Err(prefab_err(context, "reservation_wrong_tile"));
            }
        }

        reservations.push(Reservation { owner, kind, cells });
    }

    // Check for incompatible overlaps between different owners
    // Map from (layer, x, y) to owner index
    let mut occupancy: BTreeMap<(u16, u16, u16), usize> = BTreeMap::new();
    for (ri, res) in reservations.iter().enumerate() {
        for c in &res.cells {
            let key = (c.layer, c.x, c.y);
            if let Some(&prev_ri) = occupancy.get(&key) {
                // Same owner is OK (overlapping self-reservations allowed)
                if reservations[prev_ri].owner != res.owner {
                    return Err(prefab_err(context, "reservation_owner_overlap"));
                }
            }
            occupancy.insert(key, ri);
        }
    }

    Ok(reservations)
}

// ─── Ramp inference ─────────────────────────────────────────────────────────

fn infer_ramps(layers: &[Vec<Vec<Tile>>]) -> Vec<InferredRamp> {
    let mut ramps = Vec::new();
    for (li, grid) in layers.iter().enumerate() {
        let li = li as u16;
        let h = grid.len();
        if h == 0 {
            continue;
        }
        let w = grid[0].len();
        for y in 0..h {
            for x in 0..w {
                if let Tile::Ramp { direction, step } = grid[y][x] {
                    if step != 0 {
                        continue;
                    }
                    // Found R0. Check for R1 and R2 in the same direction.
                    let (dx, dy) = direction.delta();
                    let x1 = x as i32 + dx;
                    let y1 = y as i32 + dy;
                    let x2 = x1 + dx;
                    let y2 = y1 + dy;
                    if x1 < 0 || y1 < 0 || x1 >= w as i32 || y1 >= h as i32 {
                        continue;
                    }
                    if x2 < 0 || y2 < 0 || x2 >= w as i32 || y2 >= h as i32 {
                        continue;
                    }
                    match (
                        grid[y1 as usize][x1 as usize],
                        grid[y2 as usize][x2 as usize],
                    ) {
                        (
                            Tile::Ramp { direction: d1, step: 1 },
                            Tile::Ramp { direction: d2, step: 2 },
                        ) if d1 == direction && d2 == direction => {
                            ramps.push(InferredRamp {
                                cells: [
                                    (x as u16, y as u16),
                                    (x1 as u16, y1 as u16),
                                    (x2 as u16, y2 as u16),
                                ],
                                lower_layer: li,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    ramps
}

fn validate_ramp_patterns(
    layers: &[Vec<Vec<Tile>>],
    context: &str,
) -> Result<Vec<InferredRamp>, GeneratorError> {
    let inferred = infer_ramps(layers);
    let mut covered = BTreeSet::new();
    for ramp in &inferred {
        for &(x, y) in &ramp.cells {
            if !covered.insert((ramp.lower_layer, x, y)) {
                return Err(prefab_err(context, "overlapping_ramp_patterns"));
            }
        }
    }
    for (layer, grid) in layers.iter().enumerate() {
        for (y, row) in grid.iter().enumerate() {
            for (x, tile) in row.iter().enumerate() {
                if tile.is_ramp() && !covered.contains(&(layer as u16, x as u16, y as u16)) {
                    return Err(prefab_err(context, "incomplete_ramp_pattern"));
                }
            }
        }
    }
    Ok(inferred)
}

fn reservation_connects_approach(
    reservations: &[Reservation],
    transition_id: &str,
    layer: u16,
    inward: (i32, i32),
    ramp_start: (i32, i32),
) -> bool {
    let cells: BTreeSet<(i32, i32)> = reservations
        .iter()
        .filter(|reservation| {
            reservation.kind == ReservationKind::CorridorApproach
                && reservation.owner.as_str() == transition_id
        })
        .flat_map(|reservation| reservation.cells.iter())
        .filter(|cell| cell.layer == layer)
        .map(|cell| (cell.x as i32, cell.y as i32))
        .collect();
    if !cells.contains(&inward) {
        return false;
    }
    let mut frontier = vec![inward];
    let mut visited = BTreeSet::from([inward]);
    while let Some(cell) = frontier.pop() {
        if (cell.0 - ramp_start.0).abs() + (cell.1 - ramp_start.1).abs() <= 1 {
            return true;
        }
        for delta in [(0, -1), (1, 0), (0, 1), (-1, 0)] {
            let next = (cell.0 + delta.0, cell.1 + delta.1);
            if cells.contains(&next) && visited.insert(next) {
                frontier.push(next);
            }
        }
    }
    false
}

fn validate_transitions(
    transition_defs: &[TransitionDef],
    sockets: &[Socket],
    reservations: &[Reservation],
    layers: &[Vec<Vec<Tile>>],
    _width: u16,
    _height: u16,
    _layer_count: u16,
    context: &str,
) -> Result<Vec<Transition>, GeneratorError> {
    let inferred = validate_ramp_patterns(layers, context)?;
    let mut ids = BTreeSet::new();
    let mut used_ramps = BTreeSet::new();
    let mut transitions = Vec::with_capacity(transition_defs.len());
    let socket_map: BTreeMap<&str, &Socket> = sockets.iter().map(|s| (s.id.as_str(), s)).collect();

    for td in transition_defs {
        if td.id.is_empty() {
            return Err(prefab_err(context, "empty_transition_id"));
        }
        if !ids.insert(&td.id) {
            return Err(prefab_err(context, "duplicate_transition_id"));
        }
        if td.upper_layer as usize >= layers.len() {
            return Err(prefab_err(context, "transition_upper_layer_out_of_bounds"));
        }

        let lower_sock = socket_map
            .get(td.lower_approach_socket.as_str())
            .ok_or_else(|| prefab_err(context, "transition_lower_socket_not_found"))?;
        let upper_sock = socket_map
            .get(td.upper_landing_socket.as_str())
            .ok_or_else(|| prefab_err(context, "transition_upper_socket_not_found"))?;
        let lower_layer = lower_sock.anchor.layer;
        if lower_layer.checked_add(1) != Some(td.upper_layer) {
            return Err(prefab_err(context, "transition_layer_order"));
        }
        if lower_sock.role != SocketRole::LowerRampApproach {
            return Err(prefab_err(context, "transition_lower_socket_role"));
        }
        if upper_sock.anchor.layer != td.upper_layer {
            return Err(prefab_err(context, "transition_upper_socket_layer"));
        }
        if upper_sock.role != SocketRole::UpperLanding {
            return Err(prefab_err(context, "transition_upper_socket_role"));
        }

        let (dx, dy) = lower_sock.direction.delta();
        let inward = (
            lower_sock.anchor.x as i32 - dx,
            lower_sock.anchor.y as i32 - dy,
        );
        let candidates: Vec<usize> = inferred
            .iter()
            .enumerate()
            .filter_map(|(index, ramp)| {
                let r0 = (ramp.cells[0].0 as i32, ramp.cells[0].1 as i32);
                let approach_distance = (r0.0 - inward.0).abs() + (r0.1 - inward.1).abs();
                let connected = approach_distance <= 1
                    || reservation_connects_approach(
                        reservations,
                        &td.id,
                        lower_layer,
                        inward,
                        r0,
                    );
                (ramp.lower_layer == lower_layer && connected).then_some(index)
            })
            .collect();
        if candidates.is_empty() {
            return Err(prefab_err(context, "transition_no_matching_ramp"));
        }
        if candidates.len() != 1 {
            return Err(prefab_err(context, "transition_ambiguous_ramp"));
        }
        let ramp_index = candidates[0];
        if !used_ramps.insert(ramp_index) {
            return Err(prefab_err(context, "transition_ramp_reused"));
        }
        let ramp = &inferred[ramp_index];
        let upper_grid = &layers[td.upper_layer as usize];
        if ramp
            .cells
            .iter()
            .any(|&(x, y)| !upper_grid[y as usize][x as usize].is_void())
        {
            return Err(prefab_err(context, "transition_upper_opening_not_void"));
        }
        let anchor = upper_sock.anchor;
        if upper_grid[anchor.y as usize][anchor.x as usize] != Tile::Floor {
            return Err(prefab_err(context, "transition_upper_landing_not_floor"));
        }

        transitions.push(Transition {
            id: td.id.clone(),
            lower_approach_socket: td.lower_approach_socket.clone(),
            upper_landing_socket: td.upper_landing_socket.clone(),
            upper_layer: td.upper_layer,
        });
    }

    if used_ramps.len() != inferred.len() {
        return Err(prefab_err(context, "ramp_without_transition"));
    }
    Ok(transitions)
}

fn validate_component_ids_and_owners(
    sockets: &[Socket],
    markers: &[Marker],
    reservations: &[Reservation],
    transitions: &[Transition],
    context: &str,
) -> Result<(), GeneratorError> {
    let mut component_ids = BTreeSet::new();
    for id in sockets
        .iter()
        .map(|socket| socket.id.as_str())
        .chain(markers.iter().map(|marker| marker.id.as_str()))
        .chain(transitions.iter().map(|transition| transition.id.as_str()))
    {
        if !component_ids.insert(id) {
            return Err(prefab_err(context, "duplicate_component_id"));
        }
    }
    let valid_owners: BTreeSet<&str> = sockets
        .iter()
        .map(|socket| socket.id.as_str())
        .chain(transitions.iter().map(|transition| transition.id.as_str()))
        .collect();
    for reservation in reservations {
        if let ReservationOwner::Reference(owner) = &reservation.owner {
            if !valid_owners.contains(owner.as_str()) {
                return Err(prefab_err(context, "reservation_owner_not_found"));
            }
        }
    }
    Ok(())
}

// ─── Full prefab validation ─────────────────────────────────────────────────

fn validate_prefab(toml: PrefabToml, context: &str) -> Result<Prefab, GeneratorError> {
    if toml.format_version != FORMAT_VERSION {
        return Err(prefab_err(context, "unsupported_format_version"));
    }
    if toml.id.is_empty() {
        return Err(prefab_err(context, "empty_id"));
    }

    validate_tags(&toml.tags, context)?;

    let (width, height, layer_count, layers) =
        parse_layers(&toml.layers, toml.layer_count, context)?;

    let rotations = validate_rotations(&toml.rotations, context)?;
    let origin = validate_origin(toml.origin.as_ref(), width, height, context)?;
    let sockets = validate_sockets(&toml.sockets, width, height, layer_count, &layers, context)?;
    let markers = validate_markers(&toml.markers, width, height, layer_count, context)?;
    let reservations =
        validate_reservations(&toml.reservations, width, height, layer_count, &layers, context)?;
    let transitions = validate_transitions(
        &toml.transitions,
        &sockets,
        &reservations,
        &layers,
        width,
        height,
        layer_count,
        context,
    )?;
    validate_component_ids_and_owners(
        &sockets,
        &markers,
        &reservations,
        &transitions,
        context,
    )?;

    Ok(Prefab {
        id: toml.id,
        format_version: toml.format_version,
        layer_count,
        tags: toml.tags,
        width,
        height,
        layers,
        origin,
        rotations,
        sockets,
        markers,
        reservations,
        transitions,
    })
}

// ─── Quarter-turn rotation ──────────────────────────────────────────────────

/// Rotate a cell clockwise `quarter_turns` times within a grid of
/// dimensions `(width, height)`.
fn rotate_cell(c: Cell, width: u16, height: u16, quarter_turns: u8) -> Cell {
    let q = quarter_turns % 4;
    if q == 0 {
        return c;
    }
    let (mut x, mut y) = (c.x, c.y);
    let (mut w, mut h) = (width, height);
    for _ in 0..q {
        let new_x = h - 1 - y;
        let new_y = x;
        x = new_x;
        y = new_y;
        // Swap w, h
        std::mem::swap(&mut w, &mut h);
    }
    Cell { layer: c.layer, x, y }
}

fn rotate_direction(d: Direction, quarter_turns: u8) -> Direction {
    d.rotate(quarter_turns)
}

fn rotate_tile(tile: Tile, quarter_turns: u8) -> Tile {
    match tile {
        Tile::Ramp { direction, step } => Tile::Ramp {
            direction: direction.rotate(quarter_turns),
            step,
        },
        other => other,
    }
}

fn rotate_cells_sorted(
    cells: &[Cell],
    width: u16,
    height: u16,
    quarter_turns: u8,
) -> Vec<Cell> {
    let mut out: Vec<Cell> = cells
        .iter()
        .map(|c| rotate_cell(*c, width, height, quarter_turns))
        .collect();
    out.sort();
    out.dedup();
    out
}

fn rotate_grid(
    grid: &[Vec<Tile>],
    quarter_turns: u8,
) -> (Vec<Vec<Tile>>, u16, u16) {
    let q = quarter_turns % 4;
    if q == 0 {
        let h = grid.len() as u16;
        let w = if h == 0 { 0 } else { grid[0].len() as u16 };
        return (grid.to_vec(), w, h);
    }
    let h = grid.len();
    let w = if h == 0 { 0 } else { grid[0].len() };

    let (new_w, new_h) = if q % 2 == 1 { (h, w) } else { (w, h) };
    let mut new_grid = vec![vec![Tile::Wall; new_w]; new_h];

    for y in 0..h {
        for x in 0..w {
            let cell = Cell { layer: 0, x: x as u16, y: y as u16 };
            let rotated = rotate_cell(cell, w as u16, h as u16, q);
            new_grid[rotated.y as usize][rotated.x as usize] =
                rotate_tile(grid[y][x], q);
        }
    }

    (new_grid, new_w as u16, new_h as u16)
}

fn rotate_socket(socket: &Socket, old_w: u16, old_h: u16, q: u8) -> Socket {
    // A socket anchor is the minimum coordinate of its boundary aperture, not
    // an oriented endpoint. Rotate every aperture cell and normalize the
    // anchor so widths greater than one remain correct at 180° and 270°.
    let rotated_aperture: Vec<Cell> = (0..socket.width)
        .map(|offset| {
            let source = match socket.direction {
                Direction::North | Direction::South => Cell {
                    x: socket.anchor.x + offset,
                    ..socket.anchor
                },
                Direction::East | Direction::West => Cell {
                    y: socket.anchor.y + offset,
                    ..socket.anchor
                },
            };
            rotate_cell(source, old_w, old_h, q)
        })
        .collect();
    let anchor = Cell {
        layer: socket.anchor.layer,
        x: rotated_aperture.iter().map(|cell| cell.x).min().unwrap_or(socket.anchor.x),
        y: rotated_aperture.iter().map(|cell| cell.y).min().unwrap_or(socket.anchor.y),
    };
    Socket {
        id: socket.id.clone(),
        anchor,
        direction: rotate_direction(socket.direction, q),
        width: socket.width,
        role: socket.role,
    }
}

fn rotate_marker(marker: &Marker, old_w: u16, old_h: u16, q: u8) -> Marker {
    Marker {
        id: marker.id.clone(),
        position: rotate_cell(marker.position, old_w, old_h, q),
        facing: rotate_direction(marker.facing, q),
        kind: marker.kind,
    }
}

fn rotate_reservation(res: &Reservation, old_w: u16, old_h: u16, q: u8) -> Reservation {
    Reservation {
        owner: res.owner.clone(), // socket/transition refs stay as strings
        kind: res.kind,
        cells: rotate_cells_sorted(&res.cells, old_w, old_h, q),
    }
}

/// Generate a single rotated variant and run the same domain validation used
/// for source prefabs over every transformed component.
fn generate_variant(prefab: &Prefab, rotation_degrees: u16) -> Result<PrefabVariant, GeneratorError> {
    let quarter_turns = match rotation_degrees {
        0 => 0,
        90 => 1,
        180 => 2,
        270 => 3,
        _ => return Err(prefab_err(&prefab.id, "invalid_rotation")),
    };

    let mut rotated_layers = Vec::with_capacity(prefab.layers.len());
    let mut new_w = prefab.width;
    let mut new_h = prefab.height;
    for layer_grid in &prefab.layers {
        let (grid, width, height) = rotate_grid(layer_grid, quarter_turns);
        new_w = width;
        new_h = height;
        rotated_layers.push(grid);
    }
    let origin = rotate_cell(prefab.origin, prefab.width, prefab.height, quarter_turns);
    if origin.x >= new_w || origin.y >= new_h {
        return Err(prefab_err(&prefab.id, "origin_out_of_bounds_after_rotation"));
    }

    let rotated_sockets: Vec<Socket> = prefab
        .sockets
        .iter()
        .map(|socket| rotate_socket(socket, prefab.width, prefab.height, quarter_turns))
        .collect();
    let socket_defs: Vec<SocketDef> = rotated_sockets
        .iter()
        .map(|socket| SocketDef {
            id: socket.id.clone(),
            anchor: CellDef {
                layer: socket.anchor.layer,
                x: socket.anchor.x,
                y: socket.anchor.y,
            },
            direction: socket.direction.as_str().to_owned(),
            width: socket.width,
            role: socket.role.as_str().to_owned(),
        })
        .collect();
    let sockets = validate_sockets(
        &socket_defs,
        new_w,
        new_h,
        prefab.layer_count,
        &rotated_layers,
        &prefab.id,
    )?;

    let rotated_markers: Vec<Marker> = prefab
        .markers
        .iter()
        .map(|marker| rotate_marker(marker, prefab.width, prefab.height, quarter_turns))
        .collect();
    let marker_defs: Vec<MarkerDef> = rotated_markers
        .iter()
        .map(|marker| MarkerDef {
            id: marker.id.clone(),
            position: CellDef {
                layer: marker.position.layer,
                x: marker.position.x,
                y: marker.position.y,
            },
            facing: marker.facing.as_str().to_owned(),
            kind: marker.kind.as_str().to_owned(),
        })
        .collect();
    let markers = validate_markers(
        &marker_defs,
        new_w,
        new_h,
        prefab.layer_count,
        &prefab.id,
    )?;

    let rotated_reservations: Vec<Reservation> = prefab
        .reservations
        .iter()
        .map(|reservation| {
            rotate_reservation(reservation, prefab.width, prefab.height, quarter_turns)
        })
        .collect();
    let reservation_defs: Vec<ReservationDef> = rotated_reservations
        .iter()
        .map(|reservation| ReservationDef {
            owner: reservation.owner.as_str().to_owned(),
            kind: reservation.kind.as_str().to_owned(),
            cells: Some(
                reservation
                    .cells
                    .iter()
                    .map(|cell| CellDef { layer: cell.layer, x: cell.x, y: cell.y })
                    .collect(),
            ),
            box_def: None,
        })
        .collect();
    let reservations = validate_reservations(
        &reservation_defs,
        new_w,
        new_h,
        prefab.layer_count,
        &rotated_layers,
        &prefab.id,
    )?;

    let transition_defs: Vec<TransitionDef> = prefab
        .transitions
        .iter()
        .map(|transition| TransitionDef {
            id: transition.id.clone(),
            lower_approach_socket: transition.lower_approach_socket.clone(),
            upper_landing_socket: transition.upper_landing_socket.clone(),
            upper_layer: transition.upper_layer,
        })
        .collect();
    let transitions = validate_transitions(
        &transition_defs,
        &sockets,
        &reservations,
        &rotated_layers,
        new_w,
        new_h,
        prefab.layer_count,
        &prefab.id,
    )?;
    validate_component_ids_and_owners(
        &sockets,
        &markers,
        &reservations,
        &transitions,
        &prefab.id,
    )?;

    Ok(PrefabVariant {
        base_id: prefab.id.clone(),
        rotation_degrees,
        width: new_w,
        height: new_h,
        layer_count: prefab.layer_count,
        layers: rotated_layers,
        origin,
        sockets,
        markers,
        reservations,
        transitions,
        tags: prefab.tags.clone(),
    })
}

// ─── Catalog ────────────────────────────────────────────────────────────────

struct CatalogFile {
    relative_path: String,
    bytes: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct PrefabCatalog {
    identity_hex: String,
    identity_bytes: [u8; 32],
    variants: Vec<PrefabVariant>,
    _base_prefabs: Vec<Prefab>,
}

impl PrefabCatalog {
    pub(crate) fn load(root: &Path) -> Result<Self, GeneratorError> {
        let root_metadata = fs::symlink_metadata(root)
            .map_err(|_| prefab_err("catalog_root", "metadata_failed"))?;
        if root_metadata.file_type().is_symlink() {
            return Err(prefab_err("catalog_root", "symlink_rejected"));
        }
        let canonical_root = root
            .canonicalize()
            .map_err(|_| prefab_err("catalog_root", "canonicalize_failed"))?;

        let mut entries = Vec::new();
        collect_toml_files(&canonical_root, &canonical_root, &mut entries)?;

        if entries.is_empty() {
            return Err(prefab_err("catalog", "empty_catalog"));
        }

        // Sort by relative path bytes (canonical order)
        entries.sort_by(|a, b| {
            a.relative_path.as_bytes().cmp(b.relative_path.as_bytes())
        });

        // Check for duplicate relative paths (shouldn't happen after canonicalization)
        for w in entries.windows(2) {
            if w[0].relative_path == w[1].relative_path {
                return Err(prefab_err("catalog", "duplicate_path"));
            }
        }

        // Parse all files
        let mut base_prefabs: Vec<Prefab> = Vec::with_capacity(entries.len());
        let mut seen_ids = BTreeSet::new();

        for entry in &entries {
            let content = std::str::from_utf8(&entry.bytes).map_err(|_| {
                GeneratorError::PrefabIntegrity {
                    stage: ErrorStage::Prefab,
                    context: entry.relative_path.clone(),
                    reason: "toml_parse_failed",
                }
            })?;
            let toml: PrefabToml = toml::from_str(content).map_err(|_| {
                GeneratorError::PrefabIntegrity {
                    stage: ErrorStage::Prefab,
                    context: entry.relative_path.clone(),
                    reason: "toml_parse_failed",
                }
            })?;

            // Check for id duplicates across the catalog
            if !seen_ids.insert(toml.id.clone()) {
                return Err(GeneratorError::PrefabIntegrity {
                    stage: ErrorStage::Prefab,
                    context: entry.relative_path.clone(),
                    reason: "duplicate_prefab_id",
                });
            }

            let prefab = validate_prefab(toml, &entry.relative_path)?;
            base_prefabs.push(prefab);
        }

        // Sort base prefabs by id for deterministic order
        base_prefabs.sort_by(|a, b| a.id.cmp(&b.id));

        // Compute catalog identity
        let identity_bytes = compute_catalog_hash(&entries);
        let identity_hex = lowercase_hex(&identity_bytes);

        // Generate all variants
        let mut variants = Vec::new();
        for prefab in &base_prefabs {
            for &rot in &prefab.rotations {
                let variant = generate_variant(prefab, rot)?;
                variants.push(variant);
            }
        }

        Ok(Self {
            identity_hex,
            identity_bytes,
            variants,
            _base_prefabs: base_prefabs,
        })
    }

    pub(crate) fn identity_bytes(&self) -> [u8; 32] {
        self.identity_bytes
    }

    pub(crate) fn identity_hex(&self) -> &str {
        &self.identity_hex
    }

    pub(crate) fn variants(&self) -> &[PrefabVariant] {
        &self.variants
    }

    #[allow(dead_code)]
    pub(crate) fn variants_by_role(&self, role: SocketRole) -> Vec<&PrefabVariant> {
        self.variants
            .iter()
            .filter(|v| v.sockets.iter().any(|s| s.role == role))
            .collect()
    }

    #[allow(dead_code)]
    pub(crate) fn variants_by_tag(&self, tag: &str) -> Vec<&PrefabVariant> {
        self.variants
            .iter()
            .filter(|v| v.tags.iter().any(|t| t == tag))
            .collect()
    }
}

// ─── File collection ────────────────────────────────────────────────────────

fn catalog_relative_path(canonical_root: &Path, path: &Path) -> Result<String, GeneratorError> {
    let relative = path
        .strip_prefix(canonical_root)
        .map_err(|_| prefab_err("catalog_entry", "path_escape"))?;
    let text = relative
        .to_str()
        .ok_or_else(|| prefab_err("catalog_entry", "non_utf8_path"))?;
    Ok(text.replace(std::path::MAIN_SEPARATOR, "/"))
}

fn collect_toml_files(
    canonical_root: &Path,
    dir: &Path,
    entries: &mut Vec<CatalogFile>,
) -> Result<(), GeneratorError> {
    let read_dir = dir.read_dir().map_err(|_| prefab_err("catalog", "read_dir_failed"))?;

    for entry in read_dir {
        let entry = entry.map_err(|_| prefab_err("catalog", "read_entry_failed"))?;
        let path = entry.path();
        let context = catalog_relative_path(canonical_root, &path)?;
        let file_type = entry.file_type().map_err(|_| {
            prefab_err(&context, "file_type_failed")
        })?;
        if file_type.is_symlink() {
            return Err(prefab_err(&context, "symlink_rejected"));
        }
        if file_type.is_dir() {
            collect_toml_files(canonical_root, &path, entries)?;
        } else if file_type.is_file() {
            if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
                return Err(prefab_err(&context, "bad_extension"));
            }
            let absolute_path = path
                .canonicalize()
                .map_err(|_| prefab_err(&context, "canonicalize_failed"))?;
            if !absolute_path.starts_with(canonical_root) {
                return Err(prefab_err(&context, "path_escape"));
            }
            let relative_path = catalog_relative_path(canonical_root, &absolute_path)?;
            let bytes = fs::read(&absolute_path)
                .map_err(|_| prefab_err(&relative_path, "read_failed"))?;
            entries.push(CatalogFile { relative_path, bytes });
        } else {
            return Err(prefab_err(&context, "unsupported_file_type"));
        }
    }
    Ok(())
}

// ─── Catalog hash ───────────────────────────────────────────────────────────

fn compute_catalog_hash(entries: &[CatalogFile]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(CATALOG_DOMAIN);
    hasher.update(CATALOG_VERSION.to_be_bytes());
    hasher.update((entries.len() as u64).to_be_bytes());
    for entry in entries {
        let path_bytes = entry.relative_path.as_bytes();
        hasher.update((path_bytes.len() as u64).to_be_bytes());
        hasher.update(path_bytes);
        hasher.update((entry.bytes.len() as u64).to_be_bytes());
        hasher.update(&entry.bytes);
    }
    hasher.finalize().into()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    // ── Helpers ─────────────────────────────────────────────────────────

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("create temp dir")
    }

    fn write_prefab(root: &Path, rel_path: &str, content: &str) {
        let full = root.join(rel_path);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(&full).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    fn basic_room_toml(id: &str, w: u16, h: u16) -> String {
        let mut rows = Vec::new();
        let mid_x = w / 2;
        let mid_y = h / 2;
        // Build rows with door openings at mid positions on each wall
        for y in 0..h {
            let mut row_chars: Vec<char> = vec!['#'; w as usize];
            if y == 0 {
                // Top wall: opening at mid_x
                row_chars[mid_x as usize] = '.';
            } else if y == h - 1 {
                // Bottom wall: opening at mid_x
                row_chars[mid_x as usize] = '.';
            } else if y == mid_y {
                // Middle row: open sides for east/west doors
                row_chars[0] = '.';
                row_chars[w as usize - 1] = '.';
                for x in 1..w as usize - 1 {
                    row_chars[x] = '.';
                }
            } else {
                // Interior rows: walls at edges, floor inside
                row_chars[0] = '#';
                row_chars[w as usize - 1] = '#';
                for x in 1..w as usize - 1 {
                    row_chars[x] = '.';
                }
            }
            rows.push(row_chars.into_iter().collect::<String>());
        }
        let rows_toml = rows
            .iter()
            .map(|r| format!("\"{}\"", r))
            .collect::<Vec<_>>()
            .join(",\n    ");
        let mid_x = w / 2;
        let mid_y = h / 2;
        format!(
            r#"format_version = 1
id = "{id}"
layer_count = 1
tags = ["room", "small", "ordinary"]

rotations = [0, 90, 180, 270]

[[layers]]
rows = [
    {rows_toml}
]

[[sockets]]
id = "north"
anchor = {{ layer = 0, x = {mx}, y = 0 }}
direction = "north"
width = 1
role = "corridor"

[[sockets]]
id = "south"
anchor = {{ layer = 0, x = {mx}, y = {mh} }}
direction = "south"
width = 1
role = "corridor"

[[sockets]]
id = "east"
anchor = {{ layer = 0, x = {ew}, y = {my} }}
direction = "east"
width = 1
role = "corridor"

[[sockets]]
id = "west"
anchor = {{ layer = 0, x = 0, y = {my} }}
direction = "west"
width = 1
role = "corridor"

[origin]
x = {mid_x}
y = {mid_y}
"#,
            id = id,
            mx = mid_x,
            my = mid_y,
            mh = h - 1,
            ew = w - 1,
        )
    }

    // ── Tokenizer tests ────────────────────────────────────────────────

    #[test]
    fn tokenize_simple_row() {
        let tiles = Tile::tokenize_row("##..__").unwrap();
        assert_eq!(tiles, vec![
            Tile::Wall, Tile::Wall, Tile::Floor, Tile::Floor, Tile::Void, Tile::Void
        ]);
    }

    #[test]
    fn tokenize_with_ramps() {
        let tiles = Tile::tokenize_row("#R0^R1>R2vR0<#").unwrap();
        assert_eq!(tiles, vec![
            Tile::Wall,
            Tile::Ramp { direction: Direction::North, step: 0 },
            Tile::Ramp { direction: Direction::East, step: 1 },
            Tile::Ramp { direction: Direction::South, step: 2 },
            Tile::Ramp { direction: Direction::West, step: 0 },
            Tile::Wall,
        ]);
    }

    #[test]
    fn tokenize_invalid_rejected() {
        assert!(Tile::tokenize_row("#X#").is_none());
        assert!(Tile::tokenize_row("R3^").is_none());
        assert!(Tile::tokenize_row("R0x").is_none());
    }

    // ── Direction rotation ─────────────────────────────────────────────

    #[test]
    fn direction_rotation_cycle() {
        let dirs = [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ];
        for (i, &d) in dirs.iter().enumerate() {
            assert_eq!(d.rotate(1), dirs[(i + 1) % 4]);
            assert_eq!(d.rotate(2), dirs[(i + 2) % 4]);
            assert_eq!(d.rotate(4), d);
        }
    }

    // ── Cell rotation ─────────────────────────────────────────────────

    #[test]
    fn cell_rotation_square() {
        // 5 wide, 3 tall grid
        let c = Cell { layer: 0, x: 1, y: 0 };
        // 90° CW: (x,y)=(1,0) w=5,h=3 → new x = 3-1-0 = 2, new y = 1
        let r1 = rotate_cell(c, 5, 3, 1);
        assert_eq!(r1, Cell { layer: 0, x: 2, y: 1 });
        // 180°: (1,0) w=5,h=3 → new x = 5-1-1=3, new y = 3-1-0=2
        let r2 = rotate_cell(c, 5, 3, 2);
        assert_eq!(r2, Cell { layer: 0, x: 3, y: 2 });
        // 270°: (1,0) w=5,h=3 → (0,3)
        let r3 = rotate_cell(c, 5, 3, 3);
        assert_eq!(r3, Cell { layer: 0, x: 0, y: 3 });
    }

    #[test]
    fn grid_rotation_preserves_tiles() {
        let grid = vec![
            vec![Tile::Wall, Tile::Wall, Tile::Wall],
            vec![Tile::Wall, Tile::Floor, Tile::Wall],
            vec![Tile::Wall, Tile::Wall, Tile::Wall],
        ];
        let (rot, w, h) = rotate_grid(&grid, 1);
        assert_eq!((w, h), (3, 3));
        assert_eq!(rot[1][1], Tile::Floor);
        // Rotate 4 times to get back
        let (back, _, _) = rotate_grid(&grid, 4);
        assert_eq!(back, grid);
    }

    // ── Catalog hash stability ─────────────────────────────────────────

    #[test]
    fn catalog_hash_deterministic_and_byte_sensitive() {
        let dir = temp_dir();
        write_prefab(dir.path(), "a.toml", &basic_room_toml("a", 5, 5));
        write_prefab(dir.path(), "b.toml", &basic_room_toml("b", 3, 3));

        let cat1 = PrefabCatalog::load(dir.path()).unwrap();
        let cat2 = PrefabCatalog::load(dir.path()).unwrap();
        assert_eq!(cat1.identity_bytes(), cat2.identity_bytes());
        assert_eq!(cat1.identity_hex(), cat2.identity_hex());

        // Modify a file byte → hash changes
        write_prefab(dir.path(), "a.toml", &basic_room_toml("a", 7, 5));
        let cat3 = PrefabCatalog::load(dir.path()).unwrap();
        assert_ne!(cat1.identity_bytes(), cat3.identity_bytes());
    }

    #[test]
    fn catalog_hash_file_order_independent() {
        let dir = temp_dir();
        write_prefab(dir.path(), "z.toml", &basic_room_toml("z", 5, 5));
        write_prefab(dir.path(), "a.toml", &basic_room_toml("a", 5, 5));

        let cat = PrefabCatalog::load(dir.path()).unwrap();
        // The canonical sorted order is a.toml before z.toml
        let h1 = cat.identity_bytes();

        // Same files, same content → same hash
        let dir2 = temp_dir();
        write_prefab(dir2.path(), "a.toml", &basic_room_toml("a", 5, 5));
        write_prefab(dir2.path(), "z.toml", &basic_room_toml("z", 5, 5));
        let cat2 = PrefabCatalog::load(dir2.path()).unwrap();
        assert_eq!(h1, cat2.identity_bytes());
    }

    #[test]
    fn repeated_load_idempotent() {
        let dir = temp_dir();
        write_prefab(dir.path(), "r.toml", &basic_room_toml("r", 5, 5));
        let cat1 = PrefabCatalog::load(dir.path()).unwrap();
        let cat2 = PrefabCatalog::load(dir.path()).unwrap();
        let cat3 = PrefabCatalog::load(dir.path()).unwrap();
        assert_eq!(cat1.identity_bytes(), cat2.identity_bytes());
        assert_eq!(cat2.identity_bytes(), cat3.identity_bytes());
        assert_eq!(cat1.variants().len(), cat2.variants().len());
    }

    // ── Malformed content ──────────────────────────────────────────────

    /// Build a simple 3x3 TOML string with optional extra lines appended.
    fn simple_toml(id: &str, extra: &str) -> String {
        format!(
            "format_version = 1\nid = \"{}\"\n[[layers]]\nrows = [\"###\",\"#.#\",\"###\"]\nrotations = [0]\norigin = {{ x = 1, y = 1 }}\n{}",
            id, extra
        )
    }

    #[test]
    fn malformed_unsupported_version() {
        let dir = temp_dir();
        write_prefab(dir.path(), "bad.toml", "format_version = 99\nid = \"x\"\n[[layers]]\nrows = [\"###\"]\nrotations = [0]\n");
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_unknown_field() {
        let dir = temp_dir();
        write_prefab(dir.path(), "bad.toml", "format_version = 1\nid = \"x\"\nfoobar = 1\n[[layers]]\nrows = [\"###\",\"#.#\",\"###\"]\nrotations = [0]\n");
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_duplicate_id_in_catalog() {
        let dir = temp_dir();
        write_prefab(dir.path(), "a.toml", &basic_room_toml("same", 5, 5));
        write_prefab(dir.path(), "b.toml", &basic_room_toml("same", 3, 3));
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_empty_catalog() {
        let dir = temp_dir();
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_bad_extension() {
        let dir = temp_dir();
        let full = dir.path().join("bad.txt");
        fs::write(&full, "hello").unwrap();
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_ragged_rows() {
        let dir = temp_dir();
        let toml = "format_version = 1\nid = \"ragged\"\n[[layers]]\nrows = [\"###\", \"##\", \"###\"]\nrotations = [0]\norigin = { x = 1, y = 1 }\n";
        write_prefab(dir.path(), "bad.toml", toml);
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_invalid_token() {
        let dir = temp_dir();
        let toml = "format_version = 1\nid = \"bad-token\"\n[[layers]]\nrows = [\"#X#\", \"###\", \"###\"]\nrotations = [0]\n";
        write_prefab(dir.path(), "bad.toml", toml);
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_origin_out_of_bounds() {
        let dir = temp_dir();
        let toml = simple_toml("oob", "origin = { x = 99, y = 99 }\n");
        write_prefab(dir.path(), "bad.toml", &toml);
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_bad_socket_not_on_boundary() {
        let dir = temp_dir();
        let extra = "[[sockets]]\nid = \"s\"\nanchor = { layer = 0, x = 1, y = 1 }\ndirection = \"north\"\nwidth = 1\nrole = \"corridor\"\n";
        write_prefab(dir.path(), "bad.toml", &simple_toml("bad-sock", extra));
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_invalid_socket_role() {
        let dir = temp_dir();
        let extra = "[[sockets]]\nid = \"s\"\nanchor = { layer = 0, x = 1, y = 0 }\ndirection = \"north\"\nwidth = 1\nrole = \"garbage\"\n";
        write_prefab(dir.path(), "bad.toml", &simple_toml("bad-role", extra));
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_socket_width_and_aperture_clearance() {
        let cases = [
            "[[sockets]]\nid = \"s\"\nanchor = { layer = 0, x = 1, y = 0 }\ndirection = \"north\"\nwidth = 0\nrole = \"corridor\"\n",
            "[[sockets]]\nid = \"s\"\nanchor = { layer = 0, x = 1, y = 0 }\ndirection = \"north\"\nwidth = 3\nrole = \"corridor\"\n",
        ];
        for (index, extra) in cases.iter().enumerate() {
            let dir = temp_dir();
            let content = format!(
                "format_version = 1\nid = \"socket-{index}\"\nrotations = [0]\n[[layers]]\nrows = [\"###\",\"#.#\",\"###\"]\n{extra}"
            );
            write_prefab(dir.path(), "bad.toml", &content);
            let error = PrefabCatalog::load(dir.path()).unwrap_err();
            assert!(matches!(
                error.reason_code(),
                "socket_width_zero" | "socket_width_exceeds_boundary"
            ));
        }

        let dir = temp_dir();
        let blocked = "format_version = 1\nid = \"blocked\"\nrotations = [0]\n[[layers]]\nrows = [\"#.\",\"##\"]\n[[sockets]]\nid = \"s\"\nanchor = { layer = 0, x = 1, y = 0 }\ndirection = \"north\"\nwidth = 1\nrole = \"corridor\"\n";
        write_prefab(dir.path(), "bad.toml", blocked);
        let error = PrefabCatalog::load(dir.path()).unwrap_err();
        assert_eq!(error.reason_code(), "socket_inward_blocked");
    }

    #[test]
    fn malformed_bad_marker() {
        let dir = temp_dir();
        let extra = "[[markers]]\nid = \"m\"\nposition = { layer = 0, x = 99, y = 99 }\nfacing = \"north\"\nkind = \"prop\"\n";
        write_prefab(dir.path(), "bad.toml", &simple_toml("bad-marker", extra));
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_overlapping_reservations_different_owners() {
        let dir = temp_dir();
        let extra = "[[reservations]]\nowner = \"self\"\nkind = \"footprint\"\ncells = [{ layer = 0, x = 1, y = 1 }]\n[[reservations]]\nowner = \"other\"\nkind = \"footprint\"\ncells = [{ layer = 0, x = 1, y = 1 }]\n";
        write_prefab(dir.path(), "bad.toml", &simple_toml("overlap", extra));
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_unknown_reservation_owner_and_cross_kind_id_collision() {
        let dir = temp_dir();
        let unknown_owner = "format_version = 1\nid = \"owner\"\nrotations = [0]\n[[layers]]\nrows = [\"###\",\"#.#\",\"###\"]\n[[reservations]]\nowner = \"missing\"\nkind = \"footprint\"\ncells = [{ layer = 0, x = 1, y = 1 }]\n";
        write_prefab(dir.path(), "bad.toml", unknown_owner);
        let error = PrefabCatalog::load(dir.path()).unwrap_err();
        assert_eq!(error.reason_code(), "reservation_owner_not_found");

        let dir = temp_dir();
        let duplicate = "format_version = 1\nid = \"ids\"\nrotations = [0]\n[[layers]]\nrows = [\"#.\",\"#.\"]\n[[sockets]]\nid = \"shared\"\nanchor = { layer = 0, x = 1, y = 0 }\ndirection = \"north\"\nwidth = 1\nrole = \"corridor\"\n[[markers]]\nid = \"shared\"\nposition = { layer = 0, x = 1, y = 1 }\nfacing = \"north\"\nkind = \"prop\"\n";
        write_prefab(dir.path(), "bad.toml", duplicate);
        let error = PrefabCatalog::load(dir.path()).unwrap_err();
        assert_eq!(error.reason_code(), "duplicate_component_id");
    }

    #[test]
    fn malformed_broken_ramp_sequence() {
        let dir = temp_dir();
        let toml = "format_version = 1\nid = \"bad-ramp\"\nlayer_count = 2\n\
[[layers]]\nrows = [\"#####\", \"##R2^##\", \"##R1v##\", \"##R0^##\", \"#####\"]\n\
[[layers]]\nrows = [\"#####\", \"#___#\", \"#___#\", \"#___#\", \"#####\"]\n\
rotations = [0]\norigin = { x = 1, y = 2 }\n\
[[sockets]]\nid = \"s-south\"\nanchor = { layer = 0, x = 1, y = 4 }\ndirection = \"south\"\nwidth = 1\nrole = \"lower_ramp_approach\"\n\
[[sockets]]\nid = \"s-north\"\nanchor = { layer = 1, x = 1, y = 0 }\ndirection = \"north\"\nwidth = 1\nrole = \"upper_landing\"\n\
[[transitions]]\nid = \"ramp\"\nlower_approach_socket = \"s-south\"\nupper_landing_socket = \"s-north\"\nupper_layer = 1\n";
        write_prefab(dir.path(), "bad.toml", toml);
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn malformed_overflow_dimension() {
        let dir = temp_dir();
        let big = "#".repeat(u16::MAX as usize + 1);
        let toml = format!("format_version = 1\nid = \"big\"\n[[layers]]\nrows = [\"{}\"]\nrotations = [0]\n", big);
        write_prefab(dir.path(), "big.toml", &toml);
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    // ── Rotation variants ──────────────────────────────────────────────

    #[test]
    fn rectangular_all_four_rotations() {
        let dir = temp_dir();
        // 5 wide × 3 tall rectangle
        write_prefab(dir.path(), "rect.toml", &basic_room_toml("rect-room", 5, 3));
        let cat = PrefabCatalog::load(dir.path()).unwrap();
        assert_eq!(cat.variants().len(), 4);
        // Check dimensions: width × height rotates
        let v0 = &cat.variants()[0];
        assert_eq!(v0.rotation_degrees, 0);
        assert_eq!((v0.width, v0.height), (5, 3));

        let v90 = &cat.variants()[1];
        assert_eq!(v90.rotation_degrees, 90);
        assert_eq!((v90.width, v90.height), (3, 5));

        let v180 = &cat.variants()[2];
        assert_eq!(v180.rotation_degrees, 180);
        assert_eq!((v180.width, v180.height), (5, 3));

        let v270 = &cat.variants()[3];
        assert_eq!(v270.rotation_degrees, 270);
        assert_eq!((v270.width, v270.height), (3, 5));
    }

    #[test]
    fn rotation_preserves_socket_count_and_ids() {
        let dir = temp_dir();
        write_prefab(dir.path(), "r.toml", &basic_room_toml("r", 5, 5));
        let cat = PrefabCatalog::load(dir.path()).unwrap();
        for v in cat.variants() {
            assert_eq!(v.sockets.len(), 4);
            let ids: BTreeSet<&str> = v.sockets.iter().map(|s| s.id.as_str()).collect();
            assert!(ids.contains("north"));
            assert!(ids.contains("south"));
            assert!(ids.contains("east"));
            assert!(ids.contains("west"));
        }
    }

    #[test]
    fn rotation_transforms_markers() {
        let dir = temp_dir();
        let toml = "format_version = 1\nid = \"m\"\nrotations = [0, 90]\n\n[[layers]]\nrows = [\"###\",\"#.#\",\"###\"]\n\n[origin]\nx = 1\ny = 1\n\n[[markers]]\nid = \"mk\"\nposition = { layer = 0, x = 1, y = 0 }\nfacing = \"north\"\nkind = \"prop\"\n";
        write_prefab(dir.path(), "m.toml", toml);
        let cat = PrefabCatalog::load(dir.path()).unwrap();
        let v0 = &cat.variants()[0];
        assert_eq!(v0.markers[0].facing, Direction::North);
        let v90 = &cat.variants()[1];
        assert_eq!(v90.markers[0].facing, Direction::East);
    }

    #[test]
    fn rotation_transforms_reservation_boxes() {
        let dir = temp_dir();
        // Use a room with floor on the first row (y=0 has floor at x=0,1)
        let toml = "format_version = 1\nid = \"res\"\nrotations = [0, 90]\n\n[[layers]]\nrows = [\"...\",\"...\",\"...\"]\n\n[origin]\nx = 1\ny = 1\n\n[[reservations]]\nowner = \"self\"\nkind = \"footprint\"\nbox = { layer = 0, x_min = 0, y_min = 0, x_max = 1, y_max = 0 }\n";
        write_prefab(dir.path(), "res.toml", toml);
        let cat = PrefabCatalog::load(dir.path()).unwrap();
        let v0 = &cat.variants()[0];
        // Box expanded to cells (0,0) and (1,0) — both Floor
        assert_eq!(v0.reservations[0].cells.len(), 2);
        let v90 = &cat.variants()[1];
        assert_eq!(v90.reservations[0].cells.len(), 2);
        // After 90° CW rotation of 3×3 grid:
        // (0,0) → (2,0), (1,0) → (2,1)
        let expected: BTreeSet<Cell> = [
            Cell { layer: 0, x: 2, y: 0 },
            Cell { layer: 0, x: 2, y: 1 },
        ].iter().copied().collect();
        let actual: BTreeSet<Cell> = v90.reservations[0].cells.iter().copied().collect();
        assert_eq!(actual, expected);
    }

    // ── Ramp inference ─────────────────────────────────────────────────

    #[test]
    fn ramp_inference_valid_passes() {
        let dir = temp_dir();
        let toml = concat!(
            "format_version = 1\n",
            "id = \"ramp\"\n",
            "layer_count = 2\n",
            "rotations = [0]\n",
            "\n",
            "[[layers]]\n",
            "rows = [\"#####\",\"##R2^##\",\"##R1^##\",\"##R0^##\",\"##.##\"]\n",
            "[[layers]]\n",
            "rows = [\"##.##\",\"#___#\",\"#___#\",\"#___#\",\"#####\"]\n",
            "\n",
            "[origin]\n",
            "x = 2\n",
            "y = 2\n",
            "\n",
            "[[sockets]]\n",
            "id = \"s-south\"\n",
            "anchor = { layer = 0, x = 2, y = 4 }\n",
            "direction = \"south\"\n",
            "width = 1\n",
            "role = \"lower_ramp_approach\"\n",
            "[[sockets]]\n",
            "id = \"s-north\"\n",
            "anchor = { layer = 1, x = 2, y = 0 }\n",
            "direction = \"north\"\n",
            "width = 1\n",
            "role = \"upper_landing\"\n",
            "\n",
            "[[transitions]]\n",
            "id = \"ramp-north\"\n",
            "lower_approach_socket = \"s-south\"\n",
            "upper_landing_socket = \"s-north\"\n",
            "upper_layer = 1\n",
        );
        write_prefab(dir.path(), "ramp.toml", toml);
        let cat = PrefabCatalog::load(dir.path()).unwrap();
        assert_eq!(cat.variants().len(), 1);
        let v = &cat.variants()[0];
        assert_eq!(v.transitions.len(), 1);
        assert_eq!(v.transitions[0].id, "ramp-north");
    }

    #[test]
    fn ramp_upper_opening_requires_void_above_all_three_substeps() {
        let assets_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs");
        let content = fs::read_to_string(assets_root.join("ramp-hub-straight.toml")).unwrap();
        let malformed = content.replacen("#___#", "#_._#", 1);
        let dir = temp_dir();
        write_prefab(dir.path(), "bad.toml", &malformed);
        let error = PrefabCatalog::load(dir.path()).unwrap_err();
        assert!(matches!(
            error.reason_code(),
            "reservation_wrong_tile" | "transition_upper_opening_not_void"
        ));
    }

    #[test]
    fn ramp_inference_reversed_fails() {
        let dir = temp_dir();
        let toml = concat!(
            "format_version = 1\n",
            "id = \"bad-ramp-rev\"\n",
            "layer_count = 2\n",
            "rotations = [0]\n",
            "\n",
            "[[layers]]\n",
            "rows = [\"#####\",\"##R0^##\",\"##R1^##\",\"##R2^##\",\"##.##\"]\n",
            "[[layers]]\n",
            "rows = [\"#####\",\"#___#\",\"#___#\",\"#___#\",\"##.##\"]\n",
            "\n",
            "[origin]\n",
            "x = 2\n",
            "y = 2\n",
            "\n",
            "[[sockets]]\n",
            "id = \"s-south\"\n",
            "anchor = { layer = 0, x = 2, y = 4 }\n",
            "direction = \"south\"\n",
            "width = 1\n",
            "role = \"lower_ramp_approach\"\n",
            "[[sockets]]\n",
            "id = \"s-north\"\n",
            "anchor = { layer = 1, x = 2, y = 0 }\n",
            "direction = \"north\"\n",
            "width = 1\n",
            "role = \"upper_landing\"\n",
            "\n",
            "[[transitions]]\n",
            "id = \"ramp\"\n",
            "lower_approach_socket = \"s-south\"\n",
            "upper_landing_socket = \"s-north\"\n",
            "upper_layer = 1\n",
        );
        write_prefab(dir.path(), "bad.toml", toml);
        // R0 is at y=1, R1 at y=2, R2 at y=3 — going south. The ramp should go north for the sockets.
        // This will fail because no valid ramp (R0,R1,R2) matches the transition direction.
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn ramp_inference_gapped_fails() {
        let dir = temp_dir();
        let toml = concat!(
            "format_version = 1\n",
            "id = \"bad-ramp-gap\"\n",
            "layer_count = 2\n",
            "rotations = [0]\n",
            "\n",
            "[[layers]]\n",
            "rows = [\"#####\",\"##R2^##\",\"##.##\",\"##R0^##\",\"##.##\"]\n",
            "[[layers]]\n",
            "rows = [\"##.##\",\"#___#\",\"#___#\",\"#___#\",\"#####\"]\n",
            "\n",
            "[origin]\n",
            "x = 2\n",
            "y = 2\n",
            "\n",
            "[[sockets]]\n",
            "id = \"s-south\"\n",
            "anchor = { layer = 0, x = 2, y = 4 }\n",
            "direction = \"south\"\n",
            "width = 1\n",
            "role = \"lower_ramp_approach\"\n",
            "[[sockets]]\n",
            "id = \"s-north\"\n",
            "anchor = { layer = 1, x = 2, y = 0 }\n",
            "direction = \"north\"\n",
            "width = 1\n",
            "role = \"upper_landing\"\n",
            "\n",
            "[[transitions]]\n",
            "id = \"ramp\"\n",
            "lower_approach_socket = \"s-south\"\n",
            "upper_landing_socket = \"s-north\"\n",
            "upper_layer = 1\n",
        );
        write_prefab(dir.path(), "bad.toml", toml);
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn ramp_inference_generic_void_no_transition() {
        let dir = temp_dir();
        let toml = concat!(
            "format_version = 1\n",
            "id = \"no-ramp\"\n",
            "layer_count = 2\n",
            "rotations = [0]\n",
            "\n",
            "[[layers]]\n",
            "rows = [\"###\",\"#.#\",\"###\"]\n",
            "[[layers]]\n",
            "rows = [\"###\",\"#_#\",\"###\"]\n",
            "\n",
            "[origin]\n",
            "x = 1\n",
            "y = 1\n",
            "\n",
            "[[sockets]]\n",
            "id = \"s-south\"\n",
            "anchor = { layer = 0, x = 1, y = 2 }\n",
            "direction = \"south\"\n",
            "width = 1\n",
            "role = \"lower_ramp_approach\"\n",
            "[[sockets]]\n",
            "id = \"s-north\"\n",
            "anchor = { layer = 1, x = 1, y = 0 }\n",
            "direction = \"north\"\n",
            "width = 1\n",
            "role = \"upper_landing\"\n",
            "\n",
            "[[transitions]]\n",
            "id = \"ramp\"\n",
            "lower_approach_socket = \"s-south\"\n",
            "upper_landing_socket = \"s-north\"\n",
            "upper_layer = 1\n",
        );
        write_prefab(dir.path(), "bad.toml", toml);
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    #[test]
    fn ramp_inference_wrong_direction_fails() {
        let dir = temp_dir();
        let toml = concat!(
            "format_version = 1\n",
            "id = \"wrong-dir\"\n",
            "layer_count = 2\n",
            "rotations = [0]\n",
            "\n",
            "[[layers]]\n",
            "rows = [\"#####\",\"#####\",\"R2>R1>R0>##\",\"#####\",\"#####\"]\n",
            "[[layers]]\n",
            "rows = [\"#####\",\"#___#\",\"#___#\",\"#___#\",\"#####\"]\n",
            "\n",
            "[origin]\n",
            "x = 2\n",
            "y = 2\n",
            "\n",
            "[[sockets]]\n",
            "id = \"s-south\"\n",
            "anchor = { layer = 0, x = 2, y = 4 }\n",
            "direction = \"south\"\n",
            "width = 1\n",
            "role = \"lower_ramp_approach\"\n",
            "[[sockets]]\n",
            "id = \"s-north\"\n",
            "anchor = { layer = 1, x = 2, y = 0 }\n",
            "direction = \"north\"\n",
            "width = 1\n",
            "role = \"upper_landing\"\n",
            "\n",
            "[[transitions]]\n",
            "id = \"ramp\"\n",
            "lower_approach_socket = \"s-south\"\n",
            "upper_landing_socket = \"s-north\"\n",
            "upper_layer = 1\n",
        );
        write_prefab(dir.path(), "bad.toml", toml);
        // Ramp direction east doesn't match socket arrangement → no matching ramp
        assert!(PrefabCatalog::load(dir.path()).is_err());
    }

    // ── Variant ordering ───────────────────────────────────────────────

    #[test]
    fn variant_ordering_is_deterministic() {
        let dir = temp_dir();
        write_prefab(dir.path(), "a.toml", &basic_room_toml("a", 5, 5));
        write_prefab(dir.path(), "b.toml", &basic_room_toml("b", 3, 3));

        let cat1 = PrefabCatalog::load(dir.path()).unwrap();
        let cat2 = PrefabCatalog::load(dir.path()).unwrap();
        for (v1, v2) in cat1.variants().iter().zip(cat2.variants().iter()) {
            assert_eq!(v1.base_id, v2.base_id);
            assert_eq!(v1.rotation_degrees, v2.rotation_degrees);
        }
    }

    #[test]
    fn catalog_hash_matches_canonical_framing_golden() {
        let entries = vec![CatalogFile {
            relative_path: "a.toml".to_owned(),
            bytes: b"abc".to_vec(),
        }];
        assert_eq!(
            lowercase_hex(&compute_catalog_hash(&entries)),
            "8f20ad22edd7c0ccfbb9434599161cec52387a1da3b42a23e1fa6f1168044ccc"
        );
    }

    #[test]
    fn legacy_grid_tokens_are_rejected() {
        for token in ["S", "M", "L"] {
            let dir = temp_dir();
            let toml = format!(
                "format_version = 1\nid = \"legacy-{token}\"\n[[layers]]\nrows = [\"{token}\"]\nrotations = [0]\n"
            );
            write_prefab(dir.path(), "bad.toml", &toml);
            assert!(PrefabCatalog::load(dir.path()).is_err());
        }
    }

    #[test]
    fn malformed_layer_and_reservation_classes_are_rejected_without_panicking() {
        let cases = [
            "format_version = 1\nid = \"layers\"\nlayer_count = 2\n[[layers]]\nrows = [\".\"]\nrotations = [0]\n",
            "format_version = 1\nid = \"box\"\n[[layers]]\nrows = [\".\"]\nrotations = [0]\n[[reservations]]\nowner = \"self\"\nkind = \"footprint\"\nbox = { layer = 0, x_min = 0, y_min = 0, x_max = 65535, y_max = 65535 }\n",
            "format_version = 1\nid = \"wall\"\n[[layers]]\nrows = [\".\"]\nrotations = [0]\n[[reservations]]\nowner = \"self\"\nkind = \"wall_shell\"\ncells = [{ layer = 0, x = 0, y = 0 }]\n",
            "format_version = 1\nid = \"opening\"\n[[layers]]\nrows = [\".\"]\nrotations = [0]\n[[reservations]]\nowner = \"self\"\nkind = \"upper_opening\"\ncells = [{ layer = 0, x = 0, y = 0 }]\n",
            "format_version = 1\nid = \"landing\"\n[[layers]]\nrows = [\"#\"]\nrotations = [0]\n[[reservations]]\nowner = \"self\"\nkind = \"upper_landing\"\ncells = [{ layer = 0, x = 0, y = 0 }]\n",
            "format_version = 1\nid = \"approach\"\n[[layers]]\nrows = [\"#\"]\nrotations = [0]\n[[reservations]]\nowner = \"self\"\nkind = \"corridor_approach\"\ncells = [{ layer = 0, x = 0, y = 0 }]\n",
        ];
        for (index, content) in cases.iter().enumerate() {
            let dir = temp_dir();
            write_prefab(dir.path(), &format!("bad-{index}.toml"), content);
            let result = std::panic::catch_unwind(|| PrefabCatalog::load(dir.path()));
            assert!(result.is_ok(), "malformed input panicked: {index}");
            assert!(result.unwrap().is_err(), "malformed input loaded: {index}");
        }
    }

    #[test]
    fn wide_socket_rotation_normalizes_aperture_anchor() {
        let socket = Socket {
            id: "wide".to_owned(),
            anchor: Cell { layer: 0, x: 1, y: 0 },
            direction: Direction::North,
            width: 2,
            role: SocketRole::Hall,
        };
        let rotated = rotate_socket(&socket, 5, 3, 2);
        assert_eq!(rotated.direction, Direction::South);
        assert_eq!(rotated.anchor, Cell { layer: 0, x: 2, y: 2 });
        assert_eq!(rotated.width, 2);
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_rejected_without_absolute_host_paths_in_errors() {
        use std::os::unix::fs::symlink;

        let dir = temp_dir();
        let outside = temp_dir();
        write_prefab(outside.path(), "outside.toml", &basic_room_toml("outside", 3, 3));
        symlink(outside.path().join("outside.toml"), dir.path().join("linked.toml")).unwrap();
        let error = PrefabCatalog::load(dir.path()).unwrap_err();
        let rendered = error.to_string();
        assert_eq!(error.reason_code(), "symlink_rejected");
        assert!(rendered.contains("context=linked.toml"));
        assert!(!rendered.contains(&dir.path().display().to_string()));
        assert!(!rendered.contains(&outside.path().display().to_string()));
    }

    #[test]
    fn lexical_path_escape_is_rejected() {
        let root = Path::new("catalog");
        let error = catalog_relative_path(root, Path::new("outside/file.toml")).unwrap_err();
        assert_eq!(error.reason_code(), "path_escape");
    }

    // ── Asset validation ──────────────────────────────────────────────

    #[test]
    fn bundled_prefab_assets_load_and_produce_variants() {
        let assets_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs");
        if !assets_root.is_dir() {
            // Asset directory not present; skip.
            return;
        }
        let cat = PrefabCatalog::load(&assets_root).unwrap();
        // We expect exactly 8 base prefabs
        let base_ids: BTreeSet<&str> = cat.variants().iter().map(|v| v.base_id.as_str()).collect();
        let expected: BTreeSet<&str> = [
            "small-room-square",
            "small-room-offset",
            "major-hall",
            "doorway-arch",
            "junction-cross",
            "dead-end-reward-nook",
            "ramp-hub-straight",
            "ramp-hub-turn",
        ]
        .into_iter()
        .collect();
        assert_eq!(base_ids, expected);
        let small_shapes: BTreeSet<(u16, u16)> = cat
            .variants()
            .iter()
            .filter(|variant| {
                variant.rotation_degrees == 0
                    && (variant.base_id == "small-room-square"
                        || variant.base_id == "small-room-offset")
            })
            .map(|variant| (variant.width, variant.height))
            .collect();
        assert_eq!(small_shapes.len(), 2, "small rooms must be distinct assets");
        // Every variant must have the tags from its base prefab
        for v in cat.variants() {
            assert!(!v.tags.is_empty(), "variant {} rotation={} has no tags", v.base_id, v.rotation_degrees);
        }
    }
}
