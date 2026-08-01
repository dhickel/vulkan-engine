//! EnhancedV3 explorer GUI state, deterministic input handling, and overlay drawing.
//!
//! This module deliberately has no generation, filesystem, or renderer-state side
//! effects.  The application loop owns those effects after receiving [`GuiAction`].

use crate::generation::GenConfig;
use bsp_generator::enhanced_v3::{
    config::{
        CONSTRUCTION_QUANTUM, DEFAULT_ROOM_SPAN_MIN, GRAMMAR_FAMILIES, LOOP_COUNT_MAX,
        ROOM_COUNT_MAX, ROOM_COUNT_MIN, VERTICAL_EDGE_MAX, XY_MAX, XY_MIN,
    },
    ArchType, FeatureFlags, GrammarMode, V3Preset,
};
use std::time::Instant;
use winit::{event::MouseButton, keyboard::KeyCode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiMode {
    None,
    Keyboard,
    Mouse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Press,
    Release,
    Repeat,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GuiAction {
    None,
    Close,
    Generate(GenConfig),
    ApplyAndClose(GenConfig),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldKind {
    Action,
    Bool,
    U64,
    U32,
    F32,
    Enum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Quick,
    Topology,
    Geometry,
    Grammar,
    Features,
    Lighting,
    Actions,
}

impl Section {
    const ALL: [Self; 7] = [
        Self::Quick,
        Self::Topology,
        Self::Geometry,
        Self::Grammar,
        Self::Features,
        Self::Lighting,
        Self::Actions,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Quick => "Quick Presets",
            Self::Topology => "Room & Topology",
            Self::Geometry => "Geometry",
            Self::Grammar => "Grammar Families",
            Self::Features => "Features",
            Self::Lighting => "Lighting",
            Self::Actions => "Actions",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldId {
    Sparse,
    Moderate,
    Rich,
    Seed,
    RandomSeed,
    Extent,
    Rooms,
    Corridors,
    Loops,
    VerticalEdges,
    Chamfer,
    Arch,
    Stairs,
    WallThickness,
    RoomSpanMin,
    RoomSpanMax,
    GrammarMode,
    PortalChamber,
    ButtressedHall,
    ColumnGrove,
    FracturedVault,
    TerracedShrine,
    MonolithicChamber,
    Pillars,
    Buttresses,
    Blades,
    VaultRibs,
    Monoliths,
    Density,
    Minlight,
    LightCount,
    Generate,
    ApplyClose,
    RandomizeAll,
    ResetDefaults,
}

#[derive(Debug, Clone, Copy)]
struct Field {
    id: FieldId,
    label: &'static str,
    section: Section,
    kind: FieldKind,
    editable: bool,
}

macro_rules! field {
    ($id:ident, $label:literal, $section:ident, $kind:ident) => {
        Field {
            id: FieldId::$id,
            label: $label,
            section: Section::$section,
            kind: FieldKind::$kind,
            editable: true,
        }
    };
    ($id:ident, $label:literal, $section:ident, $kind:ident, disabled) => {
        Field {
            id: FieldId::$id,
            label: $label,
            section: Section::$section,
            kind: FieldKind::$kind,
            editable: false,
        }
    };
}

const FIELDS: &[Field] = &[
    field!(Sparse, "Sparse", Quick, Action),
    field!(Moderate, "Moderate", Quick, Action),
    field!(Rich, "Rich", Quick, Action),
    field!(Seed, "Seed", Quick, U64),
    field!(RandomSeed, "Random Seed", Quick, Action),
    field!(Extent, "Extent", Topology, U32),
    field!(Rooms, "Rooms", Topology, U32),
    field!(Corridors, "Corridors", Topology, U32),
    field!(Loops, "Loops", Topology, U32),
    field!(VerticalEdges, "Vertical Edges", Topology, U32),
    field!(Chamfer, "Chamfer", Geometry, Bool),
    field!(Arch, "Arch Type", Geometry, Enum),
    field!(Stairs, "Stairs", Geometry, Bool),
    field!(WallThickness, "Wall Thickness", Geometry, U32, disabled),
    field!(RoomSpanMin, "Room Span Min", Geometry, U32),
    field!(RoomSpanMax, "Room Span Max", Geometry, U32),
    field!(GrammarMode, "Mode", Grammar, Enum),
    field!(PortalChamber, "portal-chamber", Grammar, Bool),
    field!(ButtressedHall, "buttressed-hall", Grammar, Bool),
    field!(ColumnGrove, "column-grove", Grammar, Bool),
    field!(FracturedVault, "fractured-vault", Grammar, Bool),
    field!(TerracedShrine, "terraced-shrine", Grammar, Bool),
    field!(MonolithicChamber, "monolithic-chamber", Grammar, Bool),
    field!(Pillars, "Pillars", Features, Bool),
    field!(Buttresses, "Buttresses", Features, Bool),
    field!(Blades, "Blades", Features, Bool),
    field!(VaultRibs, "Vault Ribs", Features, Bool),
    field!(Monoliths, "Monoliths", Features, Bool),
    field!(Density, "Density", Features, F32),
    field!(Minlight, "Minlight", Lighting, U32),
    field!(LightCount, "Light Count", Lighting, U32),
    field!(Generate, "Generate", Actions, Action),
    field!(ApplyClose, "Apply & Close", Actions, Action),
    field!(RandomizeAll, "Randomize All", Actions, Action),
    field!(ResetDefaults, "Reset Defaults", Actions, Action),
];

const PANEL_TOP: f32 = 16.0;
const PANEL_MARGIN: f32 = 16.0;
const PANEL_WIDTH: f32 = 480.0;
const HEADER_H: f32 = 26.0;
const ROW_H: f32 = 22.0;
const SECTION_PAD: f32 = 6.0;

/// In-game EnhancedV3 draft editor. Public state is intentionally restricted
/// to the requested config, selection, scroll, and input mode.
pub struct M3Gui {
    pub config: GenConfig,
    pub selected_field: usize,
    pub scroll_offset: f32,
    pub mode: GuiMode,
    /// Last validation, parse, or entropy failure in user-actionable text.
    pub status: Option<String>,
    shift_held: bool,
    editing_field: Option<usize>,
    edit_buffer: String,
    generated_flash: Option<Instant>,
    dropdown_open: Option<usize>,
    viewport: (u32, u32),
}

impl Default for M3Gui {
    fn default() -> Self {
        Self::new()
    }
}

impl M3Gui {
    pub fn new() -> Self {
        Self {
            config: GenConfig::default_config(),
            selected_field: 0,
            scroll_offset: 0.0,
            mode: GuiMode::None,
            status: None,
            shift_held: false,
            editing_field: None,
            edit_buffer: String::new(),
            generated_flash: None,
            dropdown_open: None,
            viewport: (1280, 720),
        }
    }

    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.viewport = (width.max(1), height.max(1));
        self.clamp_scroll();
    }

    pub fn flash_generated(&mut self) {
        self.generated_flash = Some(Instant::now());
    }

    fn selected(&self) -> Field {
        FIELDS[self.selected_field.min(FIELDS.len() - 1)]
    }
    #[cfg(test)]
    fn field_index(id: FieldId) -> usize {
        FIELDS.iter().position(|field| field.id == id).unwrap()
    }

    fn move_selection(&mut self, forward: bool) {
        for _ in 0..FIELDS.len() {
            self.selected_field = if forward {
                (self.selected_field + 1) % FIELDS.len()
            } else {
                (self.selected_field + FIELDS.len() - 1) % FIELDS.len()
            };
            if FIELDS[self.selected_field].editable {
                return;
            }
        }
    }

    fn move_section(&mut self, forward: bool) {
        let current = self.selected().section;
        let position = Section::ALL
            .iter()
            .position(|section| *section == current)
            .unwrap();
        for offset in 1..=Section::ALL.len() {
            let index = if forward {
                (position + offset) % Section::ALL.len()
            } else {
                (position + Section::ALL.len() - offset) % Section::ALL.len()
            };
            if let Some(field) = FIELDS
                .iter()
                .position(|field| field.section == Section::ALL[index] && field.editable)
            {
                self.selected_field = field;
                return;
            }
        }
    }

    fn grammar_id(id: FieldId) -> Option<&'static str> {
        Some(match id {
            FieldId::PortalChamber => "portal-chamber",
            FieldId::ButtressedHall => "buttressed-hall",
            FieldId::ColumnGrove => "column-grove",
            FieldId::FracturedVault => "fractured-vault",
            FieldId::TerracedShrine => "terraced-shrine",
            FieldId::MonolithicChamber => "monolithic-chamber",
            _ => return None,
        })
    }

    fn feature_id(id: FieldId) -> Option<FeatureFlags> {
        Some(match id {
            FieldId::Pillars => FeatureFlags::PILLARS,
            FieldId::Buttresses => FeatureFlags::BUTTRESSES,
            FieldId::Blades => FeatureFlags::BLADES,
            FieldId::VaultRibs => FeatureFlags::VAULT_RIBS,
            FieldId::Monoliths => FeatureFlags::MONOLITHS,
            _ => return None,
        })
    }

    fn bool_value(&self, id: FieldId) -> bool {
        match id {
            FieldId::Chamfer => self.config.chamfer,
            FieldId::Stairs => self.config.stairs,
            id if Self::grammar_id(id).is_some() => {
                let family = Self::grammar_id(id).unwrap();
                self.config.grammar_families.is_empty()
                    || self
                        .config
                        .grammar_families
                        .iter()
                        .any(|item| item == family)
            }
            id if Self::feature_id(id).is_some() => {
                self.config.features.contains(Self::feature_id(id).unwrap())
            }
            _ => false,
        }
    }

    fn set_bool(&mut self, id: FieldId, value: bool) {
        match id {
            FieldId::Chamfer => self.config.chamfer = value,
            FieldId::Stairs => {
                self.config.stairs = value;
                self.config.vertical_edges = Some(if value { 1 } else { 0 });
            }
            id if Self::grammar_id(id).is_some() => {
                self.set_grammar_enabled(Self::grammar_id(id).unwrap(), value)
            }
            id if Self::feature_id(id).is_some() => {
                self.set_feature_enabled(Self::feature_id(id).unwrap(), value)
            }
            _ => {}
        }
        self.reconcile_draft();
    }

    fn set_grammar_enabled(&mut self, family: &str, enabled: bool) {
        // Empty is V3's all-family representation, so materialize all six
        // before changing one checkbox. This makes the default UI truthful.
        let mut selected: Vec<String> = if self.config.grammar_families.is_empty() {
            GRAMMAR_FAMILIES
                .iter()
                .map(|item| (*item).to_owned())
                .collect()
        } else {
            self.config.grammar_families.clone()
        };
        if enabled {
            if !selected.iter().any(|item| item == family) {
                selected.push(family.to_owned());
            }
        } else {
            selected.retain(|item| item != family);
        }
        self.store_grammar_allowlist(selected);
    }

    fn store_grammar_allowlist(&mut self, selected: Vec<String>) {
        let canonical: Vec<String> = GRAMMAR_FAMILIES
            .iter()
            .filter(|family| selected.iter().any(|item| item == *family))
            .map(|family| (*family).to_owned())
            .collect();
        self.config.grammar_families = if canonical.len() == GRAMMAR_FAMILIES.len() {
            Vec::new()
        } else {
            canonical
        };
        let has_enabled_family = GRAMMAR_FAMILIES.iter().any(|family| {
            (self.config.grammar_families.is_empty()
                || self
                    .config
                    .grammar_families
                    .iter()
                    .any(|item| item == family))
                && self.config.features.enables_family(family)
        });
        if self.config.feature_density > 0.0 && !has_enabled_family {
            self.config.feature_density = 0.0;
            self.status = Some("Feature density set to 0: no grammar family is enabled.".into());
        }
    }

    fn set_feature_enabled(&mut self, feature: FeatureFlags, enabled: bool) {
        self.config.features = if enabled {
            self.config.features | feature
        } else {
            flags_from_bits(self.config.features.bits() & !feature.bits())
        };
        // Explicit family allowlists must not name a family whose required flag
        // was just disabled. Empty remains V3's valid all-eligible form.
        if !self.config.grammar_families.is_empty() {
            let allowed = self
                .config
                .grammar_families
                .iter()
                .filter(|family| self.config.features.enables_family(family))
                .cloned()
                .collect();
            self.store_grammar_allowlist(allowed);
        }
    }

    fn numeric(&self, id: FieldId) -> u64 {
        match id {
            FieldId::Seed => self.config.seed,
            FieldId::Extent => self.config.extent as u64,
            FieldId::WallThickness => CONSTRUCTION_QUANTUM as u64,
            FieldId::Minlight => self.config.minlight as u64,
            FieldId::Rooms => {
                self.config
                    .rooms
                    .unwrap_or_else(|| self.config.effective_rooms()) as u64
            }
            FieldId::Corridors => {
                self.config
                    .corridors
                    .unwrap_or_else(|| self.config.effective_corridors()) as u64
            }
            FieldId::Loops => {
                self.config
                    .loops
                    .unwrap_or_else(|| self.config.effective_loops()) as u64
            }
            FieldId::VerticalEdges => self
                .config
                .vertical_edges
                .unwrap_or_else(|| self.config.effective_vertical_edges())
                as u64,
            FieldId::RoomSpanMin => {
                self.config
                    .room_span_min
                    .unwrap_or_else(|| self.config.effective_room_span_min()) as u64
            }
            FieldId::RoomSpanMax => {
                self.config
                    .room_span_max
                    .unwrap_or_else(|| self.config.effective_room_span_max()) as u64
            }
            FieldId::LightCount => {
                self.config
                    .light_count
                    .unwrap_or_else(|| self.config.effective_light_count()) as u64
            }
            _ => 0,
        }
    }

    fn numeric_range(&self, id: FieldId) -> (u64, u64) {
        match id {
            FieldId::Seed => (0, u64::MAX),
            FieldId::Extent => (XY_MIN as u64, XY_MAX as u64),
            FieldId::Rooms => (ROOM_COUNT_MIN as u64, ROOM_COUNT_MAX as u64),
            FieldId::Corridors => {
                let routes = self.config.effective_routes() as u64;
                (routes, routes * 3)
            }
            FieldId::Loops => (0, LOOP_COUNT_MAX as u64),
            FieldId::VerticalEdges => (
                0,
                VERTICAL_EDGE_MAX.min(self.config.effective_rooms() / 2) as u64,
            ),
            FieldId::RoomSpanMin | FieldId::RoomSpanMax => {
                (DEFAULT_ROOM_SPAN_MIN as u64, self.config.extent as u64)
            }
            FieldId::Minlight => (0, 255),
            FieldId::LightCount => (0, self.config.effective_rooms() as u64),
            FieldId::WallThickness => (CONSTRUCTION_QUANTUM as u64, CONSTRUCTION_QUANTUM as u64),
            _ => (0, 0),
        }
    }

    fn set_numeric(&mut self, id: FieldId, value: u64) {
        let (min, max) = self.numeric_range(id);
        let value = value.clamp(min, max);
        let value32 = value as u32;
        match id {
            FieldId::Seed => self.config.seed = value,
            FieldId::Extent => self.config.extent = value32,
            FieldId::Rooms => self.config.rooms = Some(value32),
            FieldId::Corridors => self.config.corridors = Some(value32),
            FieldId::Loops => self.config.loops = Some(value32),
            FieldId::VerticalEdges => {
                self.config.vertical_edges = Some(if self.config.stairs { value32 } else { 0 })
            }
            FieldId::RoomSpanMin => self.config.room_span_min = Some(value32),
            FieldId::RoomSpanMax => self.config.room_span_max = Some(value32),
            FieldId::Minlight => self.config.minlight = value32,
            FieldId::LightCount => self.config.light_count = Some(value32),
            _ => {}
        }
        self.reconcile_draft();
    }

    fn enum_label(&self, id: FieldId) -> &'static str {
        match id {
            FieldId::Arch => self.config.arch_type.tag(),
            FieldId::GrammarMode => self.config.grammar_mode.tag(),
            _ => "",
        }
    }
    fn enum_options(id: FieldId) -> &'static [&'static str] {
        match id {
            FieldId::Arch => &["none", "pointed", "segmented"],
            FieldId::GrammarMode => &["single", "mixed"],
            _ => &[],
        }
    }
    fn set_enum(&mut self, id: FieldId, choice: usize) {
        match (id, choice) {
            (FieldId::Arch, 0) => self.config.arch_type = ArchType::None,
            (FieldId::Arch, 1) => self.config.arch_type = ArchType::Pointed,
            (FieldId::Arch, 2) => self.config.arch_type = ArchType::Segmented,
            (FieldId::GrammarMode, 0) => self.config.grammar_mode = GrammarMode::Single,
            (FieldId::GrammarMode, 1) => self.config.grammar_mode = GrammarMode::Mixed,
            _ => return,
        }
        self.reconcile_draft();
    }
    fn cycle_enum(&mut self, id: FieldId) {
        let options = Self::enum_options(id);
        let current = options
            .iter()
            .position(|value| *value == self.enum_label(id))
            .unwrap_or(0);
        self.set_enum(id, (current + 1) % options.len());
    }

    fn reconcile_draft(&mut self) {
        self.config.normalize();
        if let Err(error) = self.config.to_v3_config() {
            self.status = Some(format!("Invalid draft: {error}"));
        }
        self.clamp_scroll();
    }

    fn select_preset(&mut self, preset: V3Preset) {
        let seed = self.config.seed;
        self.config.reset_defaults();
        self.config.seed = seed;
        self.config.preset = preset;
        self.config.extent = if preset == V3Preset::Rich { 3072 } else { 2048 };
        self.status = Some(format!("{} preset selected.", preset.tag()));
        self.reconcile_draft();
    }

    fn execute_action(&mut self, id: FieldId) -> GuiAction {
        match id {
            FieldId::Sparse => {
                self.select_preset(V3Preset::Sparse);
                GuiAction::None
            }
            FieldId::Moderate => {
                self.select_preset(V3Preset::Moderate);
                GuiAction::None
            }
            FieldId::Rich => {
                self.select_preset(V3Preset::Rich);
                GuiAction::None
            }
            FieldId::RandomSeed => {
                self.random_seed_with(|| random_u64().map_err(|error| error.to_string()))
            }
            FieldId::Generate => match self.config.to_v3_config() {
                Ok(_) => GuiAction::Generate(self.config.clone()),
                Err(error) => {
                    self.status = Some(format!("Cannot generate: {error}"));
                    GuiAction::None
                }
            },
            FieldId::ApplyClose => match self.config.to_v3_config() {
                Ok(_) => GuiAction::ApplyAndClose(self.config.clone()),
                Err(error) => {
                    self.status = Some(format!("Cannot apply: {error}"));
                    GuiAction::None
                }
            },
            FieldId::RandomizeAll => {
                self.randomize_with(|| random_u64().map_err(|error| error.to_string()))
            }
            FieldId::ResetDefaults => {
                self.config.reset_defaults();
                self.selected_field = 0;
                self.status = Some("Defaults restored.".into());
                self.reconcile_draft();
                GuiAction::None
            }
            _ => GuiAction::None,
        }
    }

    fn random_seed_with<E: std::fmt::Display>(
        &mut self,
        entropy: impl FnOnce() -> Result<u64, E>,
    ) -> GuiAction {
        match entropy() {
            Ok(seed) => {
                self.config.seed = seed;
                self.status = Some("Seed randomized.".into());
            }
            Err(error) => self.status = Some(format!("Could not randomize seed: {error}")),
        }
        GuiAction::None
    }

    fn randomize_with<E: std::fmt::Display>(
        &mut self,
        mut entropy: impl FnMut() -> Result<u64, E>,
    ) -> GuiAction {
        match self.config.randomize_with(&mut entropy) {
            Ok(()) => {
                self.status = Some("All configuration categories randomized.".into());
                self.reconcile_draft();
            }
            Err(error) => self.status = Some(format!("Could not randomize configuration: {error}")),
        }
        GuiAction::None
    }

    pub fn handle_keyboard_input(&mut self, key: KeyCode, action: Action) -> GuiAction {
        match (key, action) {
            (KeyCode::ShiftLeft | KeyCode::ShiftRight, Action::Press) => self.shift_held = true,
            (KeyCode::ShiftLeft | KeyCode::ShiftRight, Action::Release) => self.shift_held = false,
            _ => {}
        }
        if action != Action::Press {
            return GuiAction::None;
        }
        // Escape is deliberately unconditional: final user rule beats the
        // former edit-cancel behavior.
        if key == KeyCode::Escape {
            self.editing_field = None;
            self.edit_buffer.clear();
            self.dropdown_open = None;
            return GuiAction::Close;
        }
        if let Some(index) = self.editing_field {
            return self.handle_edit_key(key, index);
        }
        match key {
            KeyCode::ArrowUp | KeyCode::ArrowLeft => {
                self.move_selection(false);
                GuiAction::None
            }
            KeyCode::ArrowDown | KeyCode::ArrowRight => {
                self.move_selection(true);
                GuiAction::None
            }
            KeyCode::Tab => {
                self.move_section(!self.shift_held);
                GuiAction::None
            }
            KeyCode::Enter | KeyCode::NumpadEnter => self.activate_selected(),
            KeyCode::Space => self.toggle_selected(),
            KeyCode::Equal | KeyCode::NumpadAdd => self.adjust_selected(true),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.adjust_selected(false),
            _ => key_to_digit(key).map_or(GuiAction::None, |digit| self.begin_edit(digit)),
        }
    }

    fn handle_edit_key(&mut self, key: KeyCode, index: usize) -> GuiAction {
        match key {
            KeyCode::Enter | KeyCode::NumpadEnter => {
                self.commit_edit(index);
                GuiAction::None
            }
            KeyCode::Backspace => {
                self.edit_buffer.pop();
                if self.edit_buffer.is_empty() {
                    self.editing_field = None;
                }
                GuiAction::None
            }
            KeyCode::Tab => {
                self.commit_edit(index);
                self.move_section(!self.shift_held);
                GuiAction::None
            }
            _ => {
                if let Some(digit) = key_to_digit(key) {
                    self.edit_buffer.push(digit);
                }
                GuiAction::None
            }
        }
    }

    fn begin_edit(&mut self, digit: char) -> GuiAction {
        let field = self.selected();
        if matches!(field.kind, FieldKind::U64 | FieldKind::U32 | FieldKind::F32) && field.editable
        {
            self.editing_field = Some(self.selected_field);
            self.edit_buffer = digit.to_string();
        }
        GuiAction::None
    }

    fn commit_edit(&mut self, index: usize) {
        let field = FIELDS[index];
        let attempted = self.edit_buffer.clone();
        let valid = match field.kind {
            FieldKind::U64 | FieldKind::U32 => match attempted.parse::<u64>() {
                Ok(value) => {
                    self.set_numeric(field.id, value);
                    true
                }
                Err(_) => false,
            },
            FieldKind::F32 => match attempted.parse::<f32>() {
                Ok(value) => {
                    self.config.feature_density = value.clamp(0.0, 1.0);
                    self.reconcile_draft();
                    true
                }
                Err(_) => false,
            },
            _ => true,
        };
        if !attempted.is_empty() && !valid {
            self.status = Some(format!("Invalid {} value '{attempted}'.", field.label));
        }
        self.editing_field = None;
        self.edit_buffer.clear();
    }

    fn activate_selected(&mut self) -> GuiAction {
        let field = self.selected();
        if !field.editable {
            return GuiAction::None;
        }
        match field.kind {
            FieldKind::Action => self.execute_action(field.id),
            FieldKind::Bool => {
                self.set_bool(field.id, !self.bool_value(field.id));
                GuiAction::None
            }
            FieldKind::Enum => {
                self.cycle_enum(field.id);
                GuiAction::None
            }
            FieldKind::U64 | FieldKind::U32 | FieldKind::F32 => {
                self.editing_field = Some(self.selected_field);
                self.edit_buffer.clear();
                GuiAction::None
            }
        }
    }

    fn toggle_selected(&mut self) -> GuiAction {
        let field = self.selected();
        if field.kind == FieldKind::Bool && field.editable {
            self.set_bool(field.id, !self.bool_value(field.id));
            GuiAction::None
        } else {
            self.activate_selected()
        }
    }

    fn adjust_selected(&mut self, increase: bool) -> GuiAction {
        let field = self.selected();
        if !field.editable {
            return GuiAction::None;
        }
        match field.kind {
            FieldKind::U64 | FieldKind::U32 => {
                let step = if matches!(
                    field.id,
                    FieldId::Extent | FieldId::RoomSpanMin | FieldId::RoomSpanMax
                ) {
                    CONSTRUCTION_QUANTUM as u64
                } else {
                    1
                };
                let (min, max) = self.numeric_range(field.id);
                let current = self.numeric(field.id);
                self.set_numeric(
                    field.id,
                    if increase {
                        current.saturating_add(step).min(max)
                    } else {
                        current.saturating_sub(step).max(min)
                    },
                );
            }
            FieldKind::F32 => {
                let delta = if increase { 0.05 } else { -0.05 };
                self.config.feature_density = (self.config.feature_density + delta).clamp(0.0, 1.0);
                self.reconcile_draft();
            }
            _ => {}
        }
        GuiAction::None
    }

    pub fn handle_mouse_input(
        &mut self,
        x: f32,
        y: f32,
        button: MouseButton,
        action: Action,
    ) -> GuiAction {
        if action != Action::Press || button != MouseButton::Left {
            return GuiAction::None;
        }
        let layout = self.layout();
        match layout.hit(x, y) {
            Some(Target::Dropdown { index, choice }) => {
                self.selected_field = index;
                self.set_enum(FIELDS[index].id, choice);
                self.dropdown_open = None;
                GuiAction::None
            }
            Some(Target::Field { index, part }) => {
                self.selected_field = index;
                self.editing_field = None;
                self.edit_buffer.clear();
                let field = FIELDS[index];
                if !field.editable {
                    return GuiAction::None;
                }
                match part {
                    Part::Plus => self.adjust_selected(true),
                    Part::Minus => self.adjust_selected(false),
                    Part::Dropdown => {
                        self.dropdown_open = (self.dropdown_open != Some(index)).then_some(index);
                        GuiAction::None
                    }
                    Part::Main => match field.kind {
                        FieldKind::Action => self.execute_action(field.id),
                        FieldKind::Bool => {
                            self.set_bool(field.id, !self.bool_value(field.id));
                            GuiAction::None
                        }
                        FieldKind::Enum => {
                            self.dropdown_open =
                                (self.dropdown_open != Some(index)).then_some(index);
                            GuiAction::None
                        }
                        _ => {
                            self.editing_field = Some(index);
                            GuiAction::None
                        }
                    },
                }
            }
            None => GuiAction::None,
        }
    }

    /// Scrolls in the same screen/content coordinate system used by hit tests.
    pub fn scroll_by(&mut self, delta: f32) {
        self.scroll_offset = (self.scroll_offset + delta).clamp(0.0, self.max_scroll());
    }
    fn panel_height(&self) -> f32 {
        (self.viewport.1 as f32 - PANEL_TOP - 8.0).max(1.0)
    }
    fn content_height(&self) -> f32 {
        Section::ALL
            .iter()
            .map(|section| {
                HEADER_H
                    + SECTION_PAD
                    + FIELDS
                        .iter()
                        .filter(|field| field.section == *section)
                        .count() as f32
                        * ROW_H
                    + SECTION_PAD
            })
            .sum()
    }
    fn max_scroll(&self) -> f32 {
        (self.content_height() - self.panel_height()).max(0.0)
    }
    fn clamp_scroll(&mut self) {
        self.scroll_offset = self.scroll_offset.clamp(0.0, self.max_scroll());
    }

    fn value_text(&self, field: Field) -> String {
        match field.kind {
            FieldKind::Action => field.label.to_owned(),
            FieldKind::Bool => {
                if self.bool_value(field.id) {
                    "[x]".into()
                } else {
                    "[ ]".into()
                }
            }
            FieldKind::U64 | FieldKind::U32 => self.numeric(field.id).to_string(),
            FieldKind::F32 => format!("{:.2}", self.config.feature_density),
            FieldKind::Enum => self.enum_label(field.id).to_owned(),
        }
    }

    pub fn render(&self) -> String {
        let mut output = String::from("EnhancedV3 Explorer — M3 Config\n\n");
        for section in Section::ALL {
            output.push_str(&format!("── {} ──\n", section.label()));
            for (index, field) in FIELDS
                .iter()
                .enumerate()
                .filter(|(_, field)| field.section == section)
            {
                let selected = if index == self.selected_field {
                    ">"
                } else {
                    " "
                };
                let disabled = if field.editable {
                    ""
                } else {
                    " (disabled, fixed)"
                };
                output.push_str(&format!(
                    "{selected} {:30} {}{disabled}\n",
                    field.label,
                    self.value_text(*field)
                ));
            }
            output.push('\n');
        }
        output.push_str(&format!(
            "Draft valid: {}\n",
            if self.config.is_valid() { "YES" } else { "NO" }
        ));
        if let Some(status) = &self.status {
            output.push_str(&format!("Status: {status}\n"));
        }
        if self
            .generated_flash
            .is_some_and(|flash| flash.elapsed().as_secs_f32() < 2.0)
        {
            output.push_str("Status: Generated\n");
        }
        output
    }

    /// Draws a clipped, full-screen, ~80%-dark overlay. This callback only
    /// reads GUI state and emits draw commands; it performs no I/O or mutation
    /// of generation/renderer state beyond synchronizing local viewport data.
    pub fn render_imgui(&mut self, ui: &imgui::Ui, _ctx: &renderer::prelude::DebugUiFrameContext) {
        let display = ui.io().display_size;
        self.set_viewport(display[0].max(1.0) as u32, display[1].max(1.0) as u32);
        let background = ui.get_background_draw_list();
        background
            .add_rect(
                [0.0, 0.0],
                display,
                imgui::ImColor32::from_rgba(0, 0, 0, 204),
            )
            .filled(true)
            .build();
        let foreground = ui.get_foreground_draw_list();
        let layout = self.layout();
        foreground
            .add_rect(
                [layout.panel.x, layout.panel.y],
                [
                    layout.panel.x + layout.panel.w,
                    layout.panel.y + layout.panel.h,
                ],
                imgui::ImColor32::from_rgba(20, 20, 28, 255),
            )
            .filled(true)
            .build();
        foreground.with_clip_rect(
            [layout.panel.x, layout.panel.y],
            [
                layout.panel.x + layout.panel.w,
                layout.panel.y + layout.panel.h,
            ],
            || {
                for section in &layout.sections {
                    foreground
                        .add_rect(
                            [section.header.x, section.header.y],
                            [
                                section.header.x + section.header.w,
                                section.header.y + section.header.h,
                            ],
                            imgui::ImColor32::from_rgba(48, 52, 64, 255),
                        )
                        .filled(true)
                        .build();
                    foreground.add_text(
                        [section.header.x + 6.0, section.header.y + 4.0],
                        imgui::ImColor32::from_rgba(255, 255, 255, 255),
                        section.label.label(),
                    );
                    for row in &section.rows {
                        let field = FIELDS[row.index];
                        let selected = row.index == self.selected_field;
                        let background = if field.kind == FieldKind::Action {
                            imgui::ImColor32::from_rgba(46, 64, 82, 255)
                        } else if selected {
                            imgui::ImColor32::from_rgba(0, 110, 0, 220)
                        } else if !field.editable {
                            imgui::ImColor32::from_rgba(64, 64, 64, 180)
                        } else {
                            imgui::ImColor32::from_rgba(30, 34, 42, 220)
                        };
                        foreground
                            .add_rect(
                                [row.rect.x, row.rect.y],
                                [row.rect.x + row.rect.w, row.rect.y + row.rect.h],
                                background,
                            )
                            .filled(true)
                            .build();
                        let label_color = if field.editable {
                            imgui::ImColor32::from_rgba(255, 255, 255, 255)
                        } else {
                            imgui::ImColor32::from_rgba(145, 145, 145, 255)
                        };
                        foreground.add_text(
                            [row.rect.x + 6.0, row.rect.y + 3.0],
                            label_color,
                            field.label,
                        );
                        if field.kind != FieldKind::Action {
                            let value = self.value_text(field);
                            foreground.add_text(
                                [row.rect.x + row.rect.w - 100.0, row.rect.y + 3.0],
                                imgui::ImColor32::from_rgba(255, 235, 90, 255),
                                &value,
                            );
                        } else {
                            // Action labels are deliberately repeated inside the button affordance.
                            foreground.add_text(
                                [row.rect.x + row.rect.w / 2.0 - 40.0, row.rect.y + 3.0],
                                imgui::ImColor32::from_rgba(255, 255, 255, 255),
                                field.label,
                            );
                        }
                        if matches!(field.kind, FieldKind::U64 | FieldKind::U32 | FieldKind::F32)
                            && field.editable
                        {
                            foreground.add_text(
                                [row.rect.x + row.rect.w - 48.0, row.rect.y + 3.0],
                                imgui::ImColor32::from_rgba(220, 220, 220, 255),
                                "−  +",
                            );
                        }
                    }
                }
                for dropdown in &layout.dropdowns {
                    foreground
                        .add_rect(
                            [dropdown.rect.x, dropdown.rect.y],
                            [
                                dropdown.rect.x + dropdown.rect.w,
                                dropdown.rect.y + dropdown.rect.h,
                            ],
                            imgui::ImColor32::from_rgba(32, 36, 46, 255),
                        )
                        .filled(true)
                        .build();
                    foreground.add_text(
                        [dropdown.rect.x + 5.0, dropdown.rect.y + 3.0],
                        imgui::ImColor32::from_rgba(255, 235, 90, 255),
                        dropdown.label,
                    );
                }
            },
        );
        if self
            .generated_flash
            .is_some_and(|flash| flash.elapsed().as_secs_f32() < 2.0)
        {
            foreground.add_text(
                [layout.panel.x + 8.0, layout.panel.y + 4.0],
                imgui::ImColor32::from_rgba(100, 255, 100, 255),
                "Generated",
            );
        }
    }

    fn layout(&self) -> Layout {
        let panel_w = PANEL_WIDTH.min((self.viewport.0 as f32 - PANEL_MARGIN * 2.0).max(1.0));
        let panel = Rect {
            x: (self.viewport.0 as f32 - panel_w) / 2.0,
            y: PANEL_TOP,
            w: panel_w,
            h: self.panel_height(),
        };
        let mut content_y = PANEL_TOP;
        let mut sections = Vec::new();
        let mut rows_by_index: Vec<Option<Row>> = vec![None; FIELDS.len()];
        for section in Section::ALL {
            let header = Rect {
                x: panel.x + 2.0,
                y: content_y - self.scroll_offset,
                w: panel.w - 4.0,
                h: HEADER_H,
            };
            content_y += HEADER_H + SECTION_PAD;
            let mut rows = Vec::new();
            for (index, _) in FIELDS
                .iter()
                .enumerate()
                .filter(|(_, field)| field.section == section)
            {
                let rect = Rect {
                    x: panel.x + 4.0,
                    y: content_y - self.scroll_offset,
                    w: panel.w - 8.0,
                    h: ROW_H,
                };
                let row = Row { index, rect };
                rows_by_index[index] = Some(row);
                rows.push(row);
                content_y += ROW_H;
            }
            content_y += SECTION_PAD;
            sections.push(LayoutSection {
                label: section,
                header,
                rows,
            });
        }
        let mut dropdowns = Vec::new();
        if let Some(index) = self.dropdown_open {
            if let Some(row) = rows_by_index[index] {
                for (choice, label) in Self::enum_options(FIELDS[index].id).iter().enumerate() {
                    dropdowns.push(Dropdown {
                        index,
                        choice,
                        label,
                        rect: Rect {
                            x: row.rect.x + row.rect.w - 150.0,
                            y: row.rect.y + row.rect.h + choice as f32 * ROW_H,
                            w: 146.0,
                            h: ROW_H,
                        },
                    });
                }
            }
        }
        Layout {
            panel,
            sections,
            dropdowns,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}
impl Rect {
    fn contains(self, x: f32, y: f32) -> bool {
        x >= self.x && x <= self.x + self.w && y >= self.y && y <= self.y + self.h
    }
}
#[derive(Debug, Clone, Copy)]
struct Row {
    index: usize,
    rect: Rect,
}
struct LayoutSection {
    label: Section,
    header: Rect,
    rows: Vec<Row>,
}
struct Dropdown {
    index: usize,
    choice: usize,
    label: &'static str,
    rect: Rect,
}
struct Layout {
    panel: Rect,
    sections: Vec<LayoutSection>,
    dropdowns: Vec<Dropdown>,
}
#[derive(Debug, Clone, Copy)]
enum Part {
    Main,
    Plus,
    Minus,
    Dropdown,
}
#[derive(Debug, Clone, Copy)]
enum Target {
    Field { index: usize, part: Part },
    Dropdown { index: usize, choice: usize },
}
impl Layout {
    fn hit(&self, x: f32, y: f32) -> Option<Target> {
        if !self.panel.contains(x, y) {
            return None;
        }
        // Open menus own their visible rectangles over underlying rows.
        if let Some(menu) = self.dropdowns.iter().find(|menu| menu.rect.contains(x, y)) {
            return Some(Target::Dropdown {
                index: menu.index,
                choice: menu.choice,
            });
        }
        for section in &self.sections {
            for row in &section.rows {
                if row.rect.contains(x, y) {
                    let field = FIELDS[row.index];
                    let part = if field.kind == FieldKind::Enum {
                        Part::Dropdown
                    } else if matches!(field.kind, FieldKind::U64 | FieldKind::U32 | FieldKind::F32)
                        && x >= row.rect.x + row.rect.w - 28.0
                    {
                        Part::Plus
                    } else if matches!(field.kind, FieldKind::U64 | FieldKind::U32 | FieldKind::F32)
                        && x >= row.rect.x + row.rect.w - 56.0
                    {
                        Part::Minus
                    } else {
                        Part::Main
                    };
                    return Some(Target::Field {
                        index: row.index,
                        part,
                    });
                }
            }
        }
        None
    }
}

fn flags_from_bits(bits: u32) -> FeatureFlags {
    [
        FeatureFlags::PILLARS,
        FeatureFlags::BUTTRESSES,
        FeatureFlags::BLADES,
        FeatureFlags::VAULT_RIBS,
        FeatureFlags::MONOLITHS,
    ]
    .into_iter()
    .filter(|flag| bits & flag.bits() != 0)
    .fold(FeatureFlags::empty(), |all, flag| all | flag)
}
fn random_u64() -> Result<u64, getrandom::Error> {
    let mut bytes = [0; 8];
    getrandom::getrandom(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}
fn key_to_digit(key: KeyCode) -> Option<char> {
    Some(match key {
        KeyCode::Digit0 | KeyCode::Numpad0 => '0',
        KeyCode::Digit1 | KeyCode::Numpad1 => '1',
        KeyCode::Digit2 | KeyCode::Numpad2 => '2',
        KeyCode::Digit3 | KeyCode::Numpad3 => '3',
        KeyCode::Digit4 | KeyCode::Numpad4 => '4',
        KeyCode::Digit5 | KeyCode::Numpad5 => '5',
        KeyCode::Digit6 | KeyCode::Numpad6 => '6',
        KeyCode::Digit7 | KeyCode::Numpad7 => '7',
        KeyCode::Digit8 | KeyCode::Numpad8 => '8',
        KeyCode::Digit9 | KeyCode::Numpad9 => '9',
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn press() -> Action {
        Action::Press
    }
    fn field(id: FieldId) -> usize {
        M3Gui::field_index(id)
    }

    #[test]
    fn default_grammar_shows_all_six_checked_and_unchecking_materializes_other_five() {
        let mut gui = M3Gui::new();
        assert!(GRAMMAR_FAMILIES
            .iter()
            .all(|family| gui.config.grammar_families.is_empty()
                || gui
                    .config
                    .grammar_families
                    .iter()
                    .any(|item| item == family)));
        assert!(gui.bool_value(FieldId::PortalChamber));
        gui.selected_field = field(FieldId::PortalChamber);
        gui.handle_keyboard_input(KeyCode::Space, press());
        assert_eq!(gui.config.grammar_families.len(), 5);
        assert!(!gui
            .config
            .grammar_families
            .iter()
            .any(|family| family == "portal-chamber"));
        assert!(!gui.bool_value(FieldId::PortalChamber));
        assert!(gui.config.is_valid());
    }

    #[test]
    fn feature_toggle_removes_conflicting_explicit_family_and_preserves_validity() {
        let mut gui = M3Gui::new();
        gui.config.grammar_families = vec!["portal-chamber".into(), "terraced-shrine".into()];
        gui.selected_field = field(FieldId::Blades);
        gui.handle_keyboard_input(KeyCode::Space, press());
        assert!(!gui
            .config
            .grammar_families
            .iter()
            .any(|family| family == "portal-chamber"));
        assert!(gui.config.is_valid());
    }

    #[test]
    fn escape_always_closes_while_editing() {
        let mut gui = M3Gui::new();
        gui.selected_field = field(FieldId::Seed);
        gui.handle_keyboard_input(KeyCode::Digit9, press());
        assert_eq!(
            gui.handle_keyboard_input(KeyCode::Escape, press()),
            GuiAction::Close
        );
        assert!(gui.editing_field.is_none());
    }

    #[test]
    fn releases_and_repeats_cannot_activate() {
        let mut gui = M3Gui::new();
        gui.selected_field = field(FieldId::Chamfer);
        let before = gui.config.chamfer;
        gui.handle_keyboard_input(KeyCode::Enter, Action::Release);
        gui.handle_keyboard_input(KeyCode::Enter, Action::Repeat);
        assert_eq!(gui.config.chamfer, before);
    }

    #[test]
    fn invalid_numeric_parse_sets_actionable_status_without_discarding_value() {
        let mut gui = M3Gui::new();
        gui.selected_field = field(FieldId::Seed);
        let before = gui.config.seed;
        gui.editing_field = Some(gui.selected_field);
        gui.edit_buffer = "nope".into();
        gui.commit_edit(gui.selected_field);
        assert_eq!(gui.config.seed, before);
        assert!(gui
            .status
            .as_deref()
            .unwrap()
            .contains("Invalid Seed value"));
    }

    #[test]
    fn entropy_failures_are_reported_without_mutating_draft() {
        let mut gui = M3Gui::new();
        let before = gui.config.clone();
        gui.random_seed_with(|| Err::<u64, _>("entropy unavailable"));
        assert_eq!(gui.config, before);
        assert!(gui
            .status
            .as_deref()
            .unwrap()
            .contains("Could not randomize seed"));
        gui.randomize_with(|| Err::<u64, _>("entropy unavailable"));
        assert_eq!(gui.config, before);
        assert!(gui
            .status
            .as_deref()
            .unwrap()
            .contains("Could not randomize configuration"));
    }

    #[test]
    fn dropdown_selects_exact_option_using_raw_hitboxes() {
        let mut gui = M3Gui::new();
        gui.set_viewport(1280, 720);
        let index = field(FieldId::Arch);
        let row = gui
            .layout()
            .sections
            .iter()
            .flat_map(|section| section.rows.iter())
            .find(|row| row.index == index)
            .copied()
            .unwrap();
        gui.handle_mouse_input(
            row.rect.x + 8.0,
            row.rect.y + 5.0,
            MouseButton::Left,
            press(),
        );
        let layout = gui.layout();
        let option = layout
            .dropdowns
            .iter()
            .find(|option| option.label == "none")
            .unwrap();
        gui.handle_mouse_input(
            option.rect.x + 3.0,
            option.rect.y + 3.0,
            MouseButton::Left,
            press(),
        );
        assert_eq!(gui.config.arch_type, ArchType::None);
    }

    #[test]
    fn raw_mouse_hitboxes_own_plus_checkbox_and_action_behavior() {
        let mut gui = M3Gui::new();
        gui.set_viewport(1280, 720);
        let row_for = |gui: &M3Gui, index| {
            gui.layout()
                .sections
                .iter()
                .flat_map(|section| section.rows.iter())
                .find(|row| row.index == index)
                .copied()
                .unwrap()
        };
        let extent = row_for(&gui, field(FieldId::Extent));
        let before_extent = gui.config.extent;
        gui.handle_mouse_input(
            extent.rect.x + extent.rect.w - 8.0,
            extent.rect.y + 4.0,
            MouseButton::Left,
            press(),
        );
        assert!(gui.config.extent > before_extent);
        let chamfer = row_for(&gui, field(FieldId::Chamfer));
        let before_chamfer = gui.config.chamfer;
        gui.handle_mouse_input(
            chamfer.rect.x + 8.0,
            chamfer.rect.y + 4.0,
            MouseButton::Left,
            press(),
        );
        assert_ne!(gui.config.chamfer, before_chamfer);
        gui.scroll_by(10_000.0);
        let generate = row_for(&gui, field(FieldId::Generate));
        assert!(matches!(
            gui.handle_mouse_input(
                generate.rect.x + 8.0,
                generate.rect.y + 4.0,
                MouseButton::Left,
                press()
            ),
            GuiAction::Generate(_)
        ));
    }

    #[test]
    fn scroll_clamps_to_real_content_maximum_and_layout_uses_it() {
        let mut gui = M3Gui::new();
        gui.set_viewport(640, 180);
        let max = gui.max_scroll();
        assert!(max > 0.0);
        gui.scroll_by(10_000.0);
        assert_eq!(gui.scroll_offset, max);
        gui.scroll_by(-10_000.0);
        assert_eq!(gui.scroll_offset, 0.0);
        gui.scroll_by(max);
        let layout = gui.layout();
        assert_eq!(layout.panel.y, PANEL_TOP);
        assert!(
            layout.sections.last().unwrap().rows.last().unwrap().rect.y
                < layout.panel.y + layout.panel.h + ROW_H
        );
    }

    #[test]
    fn render_contains_visible_action_button_text_and_all_sections() {
        let gui = M3Gui::new();
        let rendered = gui.render();
        for label in [
            "Quick Presets",
            "Room & Topology",
            "Geometry",
            "Grammar Families",
            "Features",
            "Lighting",
            "Actions",
            "Generate",
            "Apply & Close",
            "Randomize All",
            "Reset Defaults",
        ] {
            assert!(rendered.contains(label), "missing {label}");
        }
    }

    #[test]
    fn randomization_is_explicit_valid_and_pipeline_safe_for_deterministic_samples() {
        let sequences = [
            [0, 42, 0, 0, 0, 0, 1, 0, 0, 3],
            [1, 99, 1, 1, 1, 3, 20, 16, 4, 5],
            [2, 255, 0, 2, 0, 31, 50, 64, 7, 8],
        ];
        for sequence in sequences {
            let mut values = sequence.into_iter();
            let mut gui = M3Gui::new();
            gui.randomize_with(|| Ok::<_, String>(values.next().unwrap()));
            assert!(gui.config.is_valid());
            assert!(
                gui.config.rooms.is_some()
                    && gui.config.corridors.is_some()
                    && gui.config.loops.is_some()
            );
            assert!(
                gui.config.vertical_edges.is_some()
                    && gui.config.room_span_min.is_some()
                    && gui.config.room_span_max.is_some()
            );
            assert!(!gui.config.grammar_families.is_empty());
            assert!(gui
                .config
                .grammar_families
                .iter()
                .all(|family| gui.config.features.enables_family(family)));
            assert!(gui.config.light_count.is_some());
            assert!(
                bsp_generator::enhanced_v3::run_pipeline(&gui.config.to_v3_config().unwrap())
                    .is_ok()
            );
        }
    }
}
