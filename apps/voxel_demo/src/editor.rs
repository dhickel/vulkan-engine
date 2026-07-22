//! In-app imgui editor for complete voxel-demo preset documents.
//!
//! The callback only mutates the draft and enqueues owned commands. File I/O,
//! OS randomness, regeneration submission, and callback registration changes
//! are performed by the event-loop owner after the callback borrow has ended.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use imgui::{Condition, TreeNodeFlags, Ui};
use renderer::prelude::DebugUiFrameContext;

use crate::config::{
    self, compute_geometry_identity, compute_scene_config_identity, get_embedded_preset,
    known_catalog_ids, known_preset_names, load_preset, normalize_document, resolve_asset_ref,
    AssetRef, DocumentSource, GeometryIdentity, MaterialTheme, PresetDocument, ResolvedAppConfig,
    ResolvedAssetRef, RuntimeOptions, SceneConfigIdentity,
};
use crate::regeneration::RegenerationState;
use crate::scene_package::CpuScenePackage;
use crate::validate::validate_preset_document;

pub const EDITOR_VIEW_ID: &str = "voxel_editor";
const MAX_EDITOR_COMMANDS: usize = 32;
const ASSET_FIELD_COUNT: usize = 8;
const EDITOR_DEFAULT_WIDTH: f32 = 400.0;
const EDITOR_DEFAULT_HEIGHT: f32 = 600.0;
const EDITOR_MIN_WIDTH: f32 = 280.0;
const EDITOR_MIN_HEIGHT: f32 = 180.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DraftSource {
    BuiltIn(String),
    External(String),
    PreviouslySaved(String),
}

impl DraftSource {
    pub fn display(&self) -> String {
        match self {
            Self::BuiltIn(name) => format!("Built-in: {name}"),
            Self::External(path) => format!("External: {path}"),
            Self::PreviouslySaved(path) => format!("Saved: {path}"),
        }
    }

    fn document_source(&self) -> DocumentSource {
        match self {
            Self::BuiltIn(name) => DocumentSource::Preset { name: name.clone() },
            Self::External(path) | Self::PreviouslySaved(path) => DocumentSource::ConfigFile {
                path: PathBuf::from(path),
            },
        }
    }
}

impl From<&DocumentSource> for DraftSource {
    fn from(source: &DocumentSource) -> Self {
        match source {
            DocumentSource::Embedded { name } | DocumentSource::Preset { name } => {
                Self::BuiltIn(name.clone())
            }
            DocumentSource::ConfigFile { path } => Self::External(path.display().to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorPhase {
    Idle,
    Queued,
    Generating {
        request_id: u64,
    },
    Failed {
        request_id: Option<u64>,
        message: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ActiveStats {
    pub wall_triangles: usize,
    pub floor_triangles: usize,
    pub total_voxels: u64,
    pub generation_time_ms: u64,
    pub mesh_time_ms: u64,
    pub partition_time_ms: u64,
}

impl ActiveStats {
    pub fn from_package(package: &CpuScenePackage) -> Self {
        Self {
            wall_triangles: package.wall_triangles,
            floor_triangles: package.floor_triangles,
            total_voxels: package.total_voxels,
            generation_time_ms: package.generation_time_ms,
            mesh_time_ms: package.mesh_time_ms,
            partition_time_ms: package.partition_time_ms,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EditorCommand {
    LoadPreset(String),
    LoadConfig(String),
    Save {
        path: String,
        document: PresetDocument,
        source_dir: PathBuf,
    },
    RandomizeSeed,
    Regenerate {
        config: ResolvedAppConfig,
        draft_identity: SceneConfigIdentity,
    },
    Hide,
}

pub struct EditorModel {
    pub draft: PresetDocument,
    pub draft_source: DraftSource,
    pub source_dir: PathBuf,
    pub dirty: bool,
    pub validation_errors: Vec<String>,
    pub active_identity: SceneConfigIdentity,
    pub draft_identity: Option<SceneConfigIdentity>,
    pub latest_accepted_request_id: Option<u64>,
    pub phase: EditorPhase,
    pub active_stats: ActiveStats,
    pub cache_entries: usize,
    pub retirement_entries: usize,
    pub config_path: String,
    pub save_path: String,
    pub visible: bool,
    pub status_message: Option<String>,

    runtime: RuntimeOptions,
    command_queue: VecDeque<EditorCommand>,
    seed_input: String,
    asset_inputs: [String; ASSET_FIELD_COUNT],
    input_errors: Vec<String>,
}

impl EditorModel {
    pub fn new(
        draft: PresetDocument,
        source: &DocumentSource,
        source_dir: PathBuf,
        runtime: RuntimeOptions,
        active_identity: SceneConfigIdentity,
        active_stats: ActiveStats,
    ) -> Self {
        let seed_input = draft.generator.seed.to_string();
        let asset_inputs = asset_input_strings(&draft);
        let mut model = Self {
            draft,
            draft_source: DraftSource::from(source),
            source_dir,
            dirty: false,
            validation_errors: Vec::new(),
            active_identity,
            draft_identity: None,
            latest_accepted_request_id: None,
            phase: EditorPhase::Idle,
            active_stats,
            cache_entries: 0,
            retirement_entries: 0,
            config_path: String::new(),
            save_path: String::new(),
            visible: true,
            status_message: None,
            runtime,
            command_queue: VecDeque::new(),
            seed_input,
            asset_inputs,
            input_errors: Vec::new(),
        };
        model.revalidate();
        model
    }

    pub fn is_draft_valid(&self) -> bool {
        self.validation_errors.is_empty() && self.draft_identity.is_some()
    }

    pub fn queued_command_count(&self) -> usize {
        self.command_queue.len()
    }

    pub fn drain_commands(&mut self) -> Vec<EditorCommand> {
        self.command_queue.drain(..).collect()
    }

    pub fn enqueue(&mut self, command: EditorCommand) -> bool {
        if self.command_queue.len() >= MAX_EDITOR_COMMANDS {
            self.status_message = Some(format!(
                "Editor command queue is full ({MAX_EDITOR_COMMANDS}); action was not queued"
            ));
            return false;
        }
        self.command_queue.push_back(command);
        true
    }

    pub fn enqueue_regenerate(&mut self) -> bool {
        self.revalidate();
        let Some(draft_identity) = self.draft_identity.clone() else {
            self.status_message = Some("Cannot regenerate: fix all validation errors".into());
            return false;
        };
        let config = match resolved_config_from_draft(self) {
            Ok(config) => config,
            Err(errors) => {
                self.validation_errors = errors;
                self.status_message = Some("Cannot regenerate: draft is not resolvable".into());
                return false;
            }
        };
        if self.enqueue(EditorCommand::Regenerate {
            config,
            draft_identity,
        }) {
            self.phase = EditorPhase::Queued;
            true
        } else {
            false
        }
    }

    pub fn enqueue_save(&mut self, path: String) -> bool {
        self.revalidate();
        if !self.is_draft_valid() {
            self.status_message = Some("Cannot save: fix all validation errors".into());
            return false;
        }
        self.enqueue(EditorCommand::Save {
            path,
            document: self.draft.clone(),
            source_dir: self.source_dir.clone(),
        })
    }

    pub fn replace_draft(
        &mut self,
        document: PresetDocument,
        source: DraftSource,
        source_dir: PathBuf,
    ) {
        self.draft = document;
        self.draft_source = source;
        self.source_dir = source_dir;
        self.seed_input = self.draft.generator.seed.to_string();
        self.asset_inputs = asset_input_strings(&self.draft);
        self.input_errors.clear();
        self.dirty = false;
        self.status_message = None;
        self.revalidate();
    }

    fn mark_edited(&mut self) {
        self.dirty = true;
        self.status_message = None;
        self.revalidate();
    }

    pub fn revalidate(&mut self) {
        let mut errors = self.input_errors.clone();
        if let Err(error) = normalize_document(&mut self.draft) {
            errors.push(error);
        }
        errors.extend(
            validate_preset_document(&self.draft)
                .into_iter()
                .map(|error| error.to_string()),
        );

        match resolve_document_assets(&self.draft, &self.source_dir) {
            Ok(assets) if errors.is_empty() => {
                let geometry = compute_geometry_identity(
                    self.draft.generator_version,
                    self.draft.rng_version,
                    &self.draft.generator,
                );
                self.draft_identity = Some(scene_identity(&self.draft, &geometry, &assets));
            }
            Ok(_) => self.draft_identity = None,
            Err(asset_errors) => {
                errors.extend(asset_errors);
                self.draft_identity = None;
            }
        }
        self.validation_errors = errors;
    }

    pub fn sync_from_regen_state(&mut self, state: &RegenerationState) {
        self.cache_entries = state.material_cache.len();
        self.retirement_entries = state.retired_materials.len();
        if state.has_pending_work() {
            self.latest_accepted_request_id = Some(state.latest_request_id);
            self.phase = EditorPhase::Generating {
                request_id: state.latest_request_id,
            };
        }
        if let Some(active) = state.active.as_ref() {
            self.active_identity = active.identity.clone();
        }
    }

    pub fn record_request_accepted(&mut self, request_id: u64) {
        self.latest_accepted_request_id = Some(request_id);
        self.phase = EditorPhase::Generating { request_id };
        self.status_message = Some(format!("Regeneration request {request_id} accepted"));
    }

    pub fn record_failure(&mut self, request_id: u64, message: String) {
        if self.latest_accepted_request_id == Some(request_id) {
            self.phase = EditorPhase::Failed {
                request_id: Some(request_id),
                message: message.clone(),
            };
            self.status_message = Some(format!("Regeneration failed: {message}"));
        }
    }

    pub fn record_success(
        &mut self,
        request_id: u64,
        identity: SceneConfigIdentity,
        stats: ActiveStats,
    ) {
        if self.latest_accepted_request_id == Some(request_id) {
            self.phase = EditorPhase::Idle;
            self.active_identity = identity;
            self.active_stats = stats;
            self.status_message = Some(format!("Regeneration request {request_id} presented"));
        }
    }
}

#[derive(Clone)]
struct ResolvedAssets {
    wall_albedo: ResolvedAssetRef,
    wall_normal: ResolvedAssetRef,
    wall_roughness: ResolvedAssetRef,
    wall_ao: ResolvedAssetRef,
    floor_albedo: ResolvedAssetRef,
    floor_normal: ResolvedAssetRef,
    floor_roughness: ResolvedAssetRef,
    floor_ao: ResolvedAssetRef,
}

fn resolve_document_assets(
    doc: &PresetDocument,
    source_dir: &Path,
) -> Result<ResolvedAssets, Vec<String>> {
    let refs = document_asset_refs(doc);
    let mut resolved = Vec::with_capacity(ASSET_FIELD_COUNT);
    let mut errors = Vec::new();
    for (label, asset_ref) in ASSET_LABELS.iter().zip(refs) {
        match resolve_asset_ref(asset_ref, source_dir, known_catalog_ids()) {
            Ok(value) => resolved.push(value),
            Err(error) => errors.push(format!("{label}: {error}")),
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let mut values = resolved.into_iter();
    Ok(ResolvedAssets {
        wall_albedo: values.next().unwrap(),
        wall_normal: values.next().unwrap(),
        wall_roughness: values.next().unwrap(),
        wall_ao: values.next().unwrap(),
        floor_albedo: values.next().unwrap(),
        floor_normal: values.next().unwrap(),
        floor_roughness: values.next().unwrap(),
        floor_ao: values.next().unwrap(),
    })
}

fn scene_identity(
    doc: &PresetDocument,
    geometry: &GeometryIdentity,
    assets: &ResolvedAssets,
) -> SceneConfigIdentity {
    compute_scene_config_identity(
        geometry,
        &doc.generator,
        &doc.materials.wall,
        &doc.materials.floor,
        &assets.wall_albedo,
        &assets.wall_normal,
        &assets.wall_roughness,
        &assets.wall_ao,
        &assets.floor_albedo,
        &assets.floor_normal,
        &assets.floor_roughness,
        &assets.floor_ao,
    )
}

fn resolved_config_from_draft(model: &EditorModel) -> Result<ResolvedAppConfig, Vec<String>> {
    let assets = resolve_document_assets(&model.draft, &model.source_dir)?;
    let geometry_identity = compute_geometry_identity(
        model.draft.generator_version,
        model.draft.rng_version,
        &model.draft.generator,
    );
    let scene_config_identity = scene_identity(&model.draft, &geometry_identity, &assets);
    Ok(ResolvedAppConfig {
        document: model.draft.clone(),
        runtime: model.runtime.clone(),
        source: model.draft_source.document_source(),
        resolved_wall_albedo: assets.wall_albedo,
        resolved_wall_normal: assets.wall_normal,
        resolved_wall_roughness: assets.wall_roughness,
        resolved_wall_ao: assets.wall_ao,
        resolved_floor_albedo: assets.floor_albedo,
        resolved_floor_normal: assets.floor_normal,
        resolved_floor_roughness: assets.floor_roughness,
        resolved_floor_ao: assets.floor_ao,
        geometry_identity,
        scene_config_identity,
        asset_digests: Vec::new(),
    })
}

const ASSET_LABELS: [&str; ASSET_FIELD_COUNT] = [
    "Wall albedo",
    "Wall normal",
    "Wall roughness",
    "Wall AO",
    "Floor albedo",
    "Floor normal",
    "Floor roughness",
    "Floor AO",
];

fn document_asset_refs(doc: &PresetDocument) -> [&AssetRef; ASSET_FIELD_COUNT] {
    [
        &doc.materials.wall.albedo,
        &doc.materials.wall.normal,
        &doc.materials.wall.roughness,
        &doc.materials.wall.ao,
        &doc.materials.floor.albedo,
        &doc.materials.floor.normal,
        &doc.materials.floor.roughness,
        &doc.materials.floor.ao,
    ]
}

fn asset_input_strings(doc: &PresetDocument) -> [String; ASSET_FIELD_COUNT] {
    document_asset_refs(doc).map(asset_ref_to_input)
}

fn asset_ref_to_input(asset_ref: &AssetRef) -> String {
    match asset_ref {
        AssetRef::Catalog { id } => format!("catalog:{id}"),
        AssetRef::Filesystem { path, .. } => format!("file:{}", path.display()),
    }
}

fn parse_asset_input(input: &str) -> Result<AssetRef, String> {
    if let Some(id) = input.strip_prefix("catalog:") {
        if id.is_empty() {
            return Err("catalog ID is empty".into());
        }
        return Ok(AssetRef::Catalog { id: id.into() });
    }
    if let Some(path) = input.strip_prefix("file:") {
        if path.is_empty() {
            return Err("file path is empty".into());
        }
        let path = PathBuf::from(path);
        let non_portable = path.is_absolute();
        return Ok(AssetRef::Filesystem { path, non_portable });
    }
    Err("use catalog:<id> or file:<path>".into())
}

fn set_asset_ref(doc: &mut PresetDocument, index: usize, value: AssetRef) {
    let target = match index {
        0 => &mut doc.materials.wall.albedo,
        1 => &mut doc.materials.wall.normal,
        2 => &mut doc.materials.wall.roughness,
        3 => &mut doc.materials.wall.ao,
        4 => &mut doc.materials.floor.albedo,
        5 => &mut doc.materials.floor.normal,
        6 => &mut doc.materials.floor.roughness,
        7 => &mut doc.materials.floor.ao,
        _ => unreachable!(),
    };
    *target = value;
}

pub fn render_editor_ui(ui: &Ui, _ctx: &DebugUiFrameContext, model: &Rc<RefCell<EditorModel>>) {
    let Ok(mut model) = model.try_borrow_mut() else {
        return;
    };

    let display_size = ui.io().display_size;
    let max_width = display_size[0].max(1.0);
    let max_height = display_size[1].max(1.0);
    let min_width = EDITOR_MIN_WIDTH.min(max_width);
    let min_height = EDITOR_MIN_HEIGHT.min(max_height);
    let initial_width = EDITOR_DEFAULT_WIDTH.min(max_width);
    let initial_height = EDITOR_DEFAULT_HEIGHT.min(max_height);

    ui.window("Voxel Editor###voxel_editor")
        .position([0.0, 0.0], Condition::Always)
        .size([initial_width, initial_height], Condition::FirstUseEver)
        .size_constraints([min_width, min_height], [max_width, max_height])
        .resizable(true)
        .movable(false)
        .build(|| {
            ui.child_window("voxel_editor_scroll_content")
                .size([0.0, 0.0])
                .scroll_bar(true)
                .scrollable(true)
                .build(|| {
                    render_file_section(ui, &mut model);
                    ui.separator();
                    render_generator_section(ui, &mut model);
                    ui.separator();
                    render_materials_section(ui, &mut model);
                    ui.separator();
                    render_status_section(ui, &model);
                    ui.separator();
                    render_actions(ui, &mut model);
                });
        });
}

fn render_file_section(ui: &Ui, model: &mut EditorModel) {
    if !ui.collapsing_header("File", TreeNodeFlags::DEFAULT_OPEN) {
        return;
    }
    ui.text(format!("Source: {}", model.draft_source.display()));
    let presets = known_preset_names();
    let mut selected = match &model.draft_source {
        DraftSource::BuiltIn(name) => presets
            .iter()
            .position(|candidate| candidate == name)
            .unwrap_or(0),
        _ => 0,
    };
    if ui.combo_simple_string("Load Preset", &mut selected, presets) {
        if let Some(name) = presets.get(selected) {
            model.enqueue(EditorCommand::LoadPreset((*name).to_string()));
        }
    }
    ui.input_text("Load Config", &mut model.config_path).build();
    if ui.button("Load File") && !model.config_path.is_empty() {
        model.enqueue(EditorCommand::LoadConfig(model.config_path.clone()));
    }
    ui.input_text("Save As", &mut model.save_path).build();
    if ui.button("Save") && !model.save_path.is_empty() {
        model.enqueue_save(model.save_path.clone());
    }
}

fn render_generator_section(ui: &Ui, model: &mut EditorModel) {
    if !ui.collapsing_header("Generator", TreeNodeFlags::DEFAULT_OPEN) {
        return;
    }
    ui.text(format!(
        "Schema {} / Generator {} / RNG {}",
        model.draft.schema_version, model.draft.generator_version, model.draft.rng_version
    ));

    let mut changed = false;
    if ui.input_text("Seed", &mut model.seed_input).build() {
        model
            .input_errors
            .retain(|error| !error.starts_with("Seed input:"));
        match model.seed_input.parse::<u64>() {
            Ok(seed) => {
                model.draft.generator.seed = seed;
                changed = true;
            }
            Err(_) => model
                .input_errors
                .push("Seed input: expected a decimal u64".into()),
        }
    }

    let gen = &mut model.draft.generator;
    let resolutions = [64_u32, 96, 128];
    let labels = ["64", "96", "128"];
    let mut resolution_index = resolutions
        .iter()
        .position(|value| *value == gen.resolution)
        .unwrap_or(0);
    if ui.combo_simple_string("Resolution", &mut resolution_index, &labels) {
        gen.resolution = resolutions[resolution_index];
        changed = true;
    }
    changed |= input_u32(ui, "Shell Thickness", &mut gen.shell_thickness);
    changed |= input_u32(ui, "Cavern Count", &mut gen.cavern_count);
    changed |= input_u32(ui, "Tunnel Count", &mut gen.tunnel_count);
    changed |= input_f32(ui, "Tunnel Radius Min", &mut gen.tunnel_radius_min);
    changed |= input_f32(ui, "Tunnel Radius Max", &mut gen.tunnel_radius_max);
    changed |= input_f32(ui, "Cavern Radius Min", &mut gen.cavern_radius_min);
    changed |= input_f32(ui, "Cavern Radius Max", &mut gen.cavern_radius_max);
    changed |= input_f32(ui, "Spline Tension", &mut gen.spline_tension);
    changed |= input_f32(ui, "Surface Roughness", &mut gen.roughness);
    changed |= input_f32(ui, "Maze Density", &mut gen.maze_density);
    changed |= input_f32(ui, "Maze Twistiness", &mut gen.maze_twistiness);
    changed |= input_f32(ui, "Maze Radius", &mut gen.maze_radius);
    changed |= input_u32(ui, "Maze Retries", &mut gen.maze_retries);
    changed |= input_u32(ui, "Maze Search Budget", &mut gen.maze_search_budget);
    changed |= input_f32(ui, "Floor Threshold", &mut gen.floor_threshold);
    changed |= input_f32(ui, "Wall UV Scale", &mut gen.wall_uv_scale);
    changed |= input_f32(ui, "Floor UV Scale", &mut gen.floor_uv_scale);
    if changed {
        model.mark_edited();
    } else if !model.input_errors.is_empty() {
        model.revalidate();
    }
}

fn input_u32(ui: &Ui, label: &str, value: &mut u32) -> bool {
    let mut input = i64::from(*value);
    if ui
        .input_scalar(label, &mut input)
        .step(1)
        .step_fast(10)
        .build()
        && input >= 0
    {
        if let Ok(converted) = u32::try_from(input) {
            *value = converted;
            return true;
        }
    }
    false
}

fn input_f32(ui: &Ui, label: &str, value: &mut f32) -> bool {
    ui.input_scalar(label, value)
        .step(0.05)
        .step_fast(0.5)
        .build()
}

fn render_materials_section(ui: &Ui, model: &mut EditorModel) {
    if !ui.collapsing_header("Materials", TreeNodeFlags::DEFAULT_OPEN) {
        return;
    }
    ui.text("Asset syntax: catalog:<id> or file:<path>");
    let mut changed = false;
    for index in 0..ASSET_FIELD_COUNT {
        let label = format!("{}##asset{index}", ASSET_LABELS[index]);
        if ui
            .input_text(&label, &mut model.asset_inputs[index])
            .build()
        {
            model
                .input_errors
                .retain(|error| !error.starts_with(&format!("{} input:", ASSET_LABELS[index])));
            match parse_asset_input(&model.asset_inputs[index]) {
                Ok(asset_ref) => {
                    set_asset_ref(&mut model.draft, index, asset_ref);
                    changed = true;
                }
                Err(error) => model
                    .input_errors
                    .push(format!("{} input: {error}", ASSET_LABELS[index])),
            }
        }
    }

    changed |= render_material_theme(ui, "Wall", &mut model.draft.materials.wall);
    changed |= render_material_theme(ui, "Floor", &mut model.draft.materials.floor);
    if changed {
        model.mark_edited();
    } else if !model.input_errors.is_empty() {
        model.revalidate();
    }
}

fn render_material_theme(ui: &Ui, prefix: &str, theme: &mut MaterialTheme) -> bool {
    ui.text(prefix);
    let mut changed = false;
    changed |= input_f32(
        ui,
        &format!("Base Color R##{prefix}"),
        &mut theme.base_color_r,
    );
    changed |= input_f32(
        ui,
        &format!("Base Color G##{prefix}"),
        &mut theme.base_color_g,
    );
    changed |= input_f32(
        ui,
        &format!("Base Color B##{prefix}"),
        &mut theme.base_color_b,
    );
    changed |= input_f32(
        ui,
        &format!("Roughness Factor##{prefix}"),
        &mut theme.roughness_factor,
    );
    changed |= input_f32(
        ui,
        &format!("Metallic Factor##{prefix}"),
        &mut theme.metallic_factor,
    );
    changed
}

fn render_status_section(ui: &Ui, model: &EditorModel) {
    if !ui.collapsing_header("Status", TreeNodeFlags::DEFAULT_OPEN) {
        return;
    }
    if model.validation_errors.is_empty() {
        ui.text_colored([0.3, 1.0, 0.3, 1.0], "Draft valid");
    } else {
        ui.text_colored([1.0, 0.3, 0.3, 1.0], "Validation errors:");
        for error in &model.validation_errors {
            ui.text_wrapped(format!("• {error}"));
        }
    }
    ui.text(format!("Phase: {:?}", model.phase));
    ui.text(format!(
        "Latest accepted request: {}",
        model
            .latest_accepted_request_id
            .map_or_else(|| "none".into(), |id| id.to_string())
    ));
    ui.text(format!(
        "Active identity: {}",
        identity_hex(&model.active_identity)
    ));
    ui.text(format!(
        "Draft identity: {}",
        model
            .draft_identity
            .as_ref()
            .map_or_else(|| "invalid".into(), identity_hex)
    ));
    ui.text(format!(
        "Triangles: wall={} floor={}",
        model.active_stats.wall_triangles, model.active_stats.floor_triangles
    ));
    ui.text(format!(
        "Measurements: generation={}ms mesh={}ms partition={}ms voxels={}",
        model.active_stats.generation_time_ms,
        model.active_stats.mesh_time_ms,
        model.active_stats.partition_time_ms,
        model.active_stats.total_voxels
    ));
    ui.text(format!(
        "Material cache={} retirement={}",
        model.cache_entries, model.retirement_entries
    ));
    ui.text(if model.dirty {
        "Draft modified"
    } else {
        "Draft clean"
    });
    if let Some(message) = &model.status_message {
        ui.text_wrapped(message);
    }
}

fn identity_hex(identity: &SceneConfigIdentity) -> String {
    identity
        .0
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn render_actions(ui: &Ui, model: &mut EditorModel) {
    if ui.button("Randomize Seed") {
        model.enqueue(EditorCommand::RandomizeSeed);
    }
    ui.same_line();
    let can_regenerate = model.is_draft_valid()
        && !matches!(
            model.phase,
            EditorPhase::Generating { .. } | EditorPhase::Queued
        );
    if can_regenerate {
        if ui.button("Regenerate") {
            model.enqueue_regenerate();
        }
    } else {
        let disabled = ui.begin_disabled(true);
        ui.button("Regenerate");
        disabled.end();
    }
    ui.same_line();
    if ui.button("Hide Editor") {
        model.enqueue(EditorCommand::Hide);
    }
}

pub fn handle_command(
    command: EditorCommand,
    model: &Rc<RefCell<EditorModel>>,
    regen_state: &mut RegenerationState,
) -> bool {
    match command {
        EditorCommand::Hide => return true,
        EditorCommand::RandomizeSeed => {
            let result = random_seed().map_err(|error| error.to_string());
            apply_random_seed_result(model, result);
        }
        EditorCommand::LoadPreset(name) => handle_load_preset(&name, model),
        EditorCommand::LoadConfig(path) => handle_load_config(&path, model),
        EditorCommand::Save {
            path,
            document,
            source_dir,
        } => handle_save(&path, document, source_dir, model),
        EditorCommand::Regenerate {
            config,
            draft_identity,
        } => {
            if config.scene_config_identity != draft_identity {
                model.borrow_mut().status_message =
                    Some("Regeneration snapshot identity mismatch; request rejected".into());
            } else {
                regen_state.submit_request(config);
                model
                    .borrow_mut()
                    .record_request_accepted(regen_state.latest_request_id);
            }
        }
    }
    false
}

fn prepare_loaded_document(
    mut document: PresetDocument,
    source_dir: &Path,
) -> Result<PresetDocument, Vec<String>> {
    let mut errors = Vec::new();
    if let Err(error) = normalize_document(&mut document) {
        errors.push(error);
    }
    errors.extend(
        validate_preset_document(&document)
            .into_iter()
            .map(|error| error.to_string()),
    );
    if let Err(asset_errors) = resolve_document_assets(&document, source_dir) {
        errors.extend(asset_errors);
    }
    if errors.is_empty() {
        Ok(document)
    } else {
        Err(errors)
    }
}

fn handle_load_preset(name: &str, model: &Rc<RefCell<EditorModel>>) {
    let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let result = get_embedded_preset(name)
        .ok_or_else(|| vec![format!("Unknown built-in preset '{name}'")])
        .and_then(|(canonical_name, document)| {
            prepare_loaded_document(document, &source_dir)
                .map(|document| (canonical_name.to_string(), document))
        });
    match result {
        Ok((canonical_name, document)) => {
            let mut model = model.borrow_mut();
            model.replace_draft(
                document,
                DraftSource::BuiltIn(canonical_name.clone()),
                source_dir,
            );
            model.status_message = Some(format!("Loaded built-in preset '{canonical_name}'"));
        }
        Err(errors) => {
            let mut model = model.borrow_mut();
            model.status_message = Some(format!(
                "Failed to load preset '{name}': {}",
                errors.join("; ")
            ));
        }
    }
}

fn absolute_user_path(path: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        Ok(path)
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .map_err(|error| error.to_string())
    }
}

fn handle_load_config(path: &str, model: &Rc<RefCell<EditorModel>>) {
    let result = absolute_user_path(path).and_then(|absolute| {
        let source_dir = absolute
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "config path has no parent".to_string())?;
        let document = load_preset(&absolute)?;
        prepare_loaded_document(document, &source_dir)
            .map(|document| (absolute, source_dir, document))
            .map_err(|errors| errors.join("; "))
    });
    match result {
        Ok((absolute, source_dir, document)) => {
            let display = absolute.display().to_string();
            let mut model = model.borrow_mut();
            model.replace_draft(document, DraftSource::External(display.clone()), source_dir);
            model.status_message = Some(format!("Loaded config '{display}'"));
        }
        Err(error) => {
            model.borrow_mut().status_message =
                Some(format!("Failed to load config '{path}': {error}"));
        }
    }
}

fn handle_save(
    path: &str,
    snapshot: PresetDocument,
    source_dir: PathBuf,
    model: &Rc<RefCell<EditorModel>>,
) {
    let result = absolute_user_path(path).and_then(|absolute| {
        config::save_preset(&snapshot, &source_dir, &absolute)?;
        let saved = load_preset(&absolute)?;
        let saved_dir = absolute
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "save path has no parent".to_string())?;
        prepare_loaded_document(saved, &saved_dir)
            .map(|document| (absolute, saved_dir, document))
            .map_err(|errors| errors.join("; "))
    });
    match result {
        Ok((absolute, saved_dir, saved_document)) => {
            let mut model = model.borrow_mut();
            if model.draft == snapshot {
                model.replace_draft(
                    saved_document,
                    DraftSource::PreviouslySaved(absolute.display().to_string()),
                    saved_dir,
                );
                model.status_message = Some(format!("Saved config '{}'", absolute.display()));
            } else {
                model.status_message = Some(format!(
                    "Saved snapshot to '{}'; newer draft edits remain unsaved",
                    absolute.display()
                ));
                model.dirty = true;
            }
        }
        Err(error) => {
            model.borrow_mut().status_message =
                Some(format!("Failed to save config '{path}': {error}"));
        }
    }
}

fn random_seed() -> Result<u64, getrandom::Error> {
    let mut bytes = [0_u8; 8];
    getrandom::getrandom(&mut bytes)?;
    Ok(u64::from_be_bytes(bytes))
}

fn apply_random_seed_result(model: &Rc<RefCell<EditorModel>>, result: Result<u64, String>) {
    let mut model = model.borrow_mut();
    match result {
        Ok(seed) => {
            model.draft.generator.seed = seed;
            model.seed_input = seed.to_string();
            model.mark_edited();
            model.status_message = Some(format!(
                "Randomized seed to {seed}; regenerate explicitly to apply"
            ));
        }
        Err(error) => {
            model.status_message = Some(format!("Failed to randomize seed: {error}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Rc<RefCell<EditorModel>> {
        let (_, document) = get_embedded_preset("default").unwrap();
        let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let assets = resolve_document_assets(&document, &source_dir).unwrap();
        let geometry = compute_geometry_identity(
            document.generator_version,
            document.rng_version,
            &document.generator,
        );
        let identity = scene_identity(&document, &geometry, &assets);
        Rc::new(RefCell::new(EditorModel::new(
            document,
            &DocumentSource::Embedded {
                name: "default".into(),
            },
            source_dir,
            RuntimeOptions {
                light_budget: 12,
                headless: false,
                capture_dir: Some("ignored".into()),
                env_path: Some("env.exr".into()),
            },
            identity,
            ActiveStats::default(),
        )))
    }

    #[test]
    fn initial_model_is_complete_clean_and_valid() {
        let model = model();
        let model = model.borrow();
        assert!(!model.dirty);
        assert!(model.is_draft_valid());
        assert_eq!(model.draft.generator.seed.to_string(), model.seed_input);
        assert_eq!(model.asset_inputs.len(), ASSET_FIELD_COUNT);
    }

    #[test]
    fn invalid_draft_cannot_enqueue_regeneration() {
        let model = model();
        model.borrow_mut().draft.generator.resolution = 65;
        assert!(!model.borrow_mut().enqueue_regenerate());
        assert_eq!(model.borrow().queued_command_count(), 0);
    }

    #[test]
    fn regeneration_command_owns_snapshot_and_preserves_runtime_options() {
        let model = model();
        model.borrow_mut().draft.generator.seed = 77;
        assert!(model.borrow_mut().enqueue_regenerate());
        model.borrow_mut().draft.generator.seed = 88;
        let command = model.borrow_mut().drain_commands().pop().unwrap();
        match command {
            EditorCommand::Regenerate { config, .. } => {
                assert_eq!(config.document.generator.seed, 77);
                assert_eq!(config.runtime.light_budget, 12);
                assert_eq!(config.runtime.env_path, Some("env.exr".into()));
            }
            _ => panic!("unexpected command"),
        }
    }

    #[test]
    fn randomize_is_separate_and_failure_preserves_seed() {
        let model = model();
        let original = model.borrow().draft.generator.seed;
        apply_random_seed_result(&model, Err("entropy unavailable".into()));
        assert_eq!(model.borrow().draft.generator.seed, original);
        assert!(!model.borrow().dirty);
        apply_random_seed_result(&model, Ok(0));
        assert_eq!(model.borrow().draft.generator.seed, 0);
        assert!(model.borrow().dirty);
        assert_eq!(model.borrow().queued_command_count(), 0);
    }

    #[test]
    fn queue_is_bounded_and_reports_saturation() {
        let model = model();
        for _ in 0..MAX_EDITOR_COMMANDS {
            assert!(model.borrow_mut().enqueue(EditorCommand::Hide));
        }
        assert!(!model.borrow_mut().enqueue(EditorCommand::Hide));
        assert_eq!(model.borrow().queued_command_count(), MAX_EDITOR_COMMANDS);
        assert!(model
            .borrow()
            .status_message
            .as_deref()
            .unwrap()
            .contains("full"));
    }

    #[test]
    fn failed_load_preserves_complete_draft_and_dirty_state() {
        let model = model();
        model.borrow_mut().dirty = true;
        let before = model.borrow().draft.clone();
        handle_load_config("/definitely/not/a/voxel-demo-config.toml", &model);
        assert_eq!(model.borrow().draft, before);
        assert!(model.borrow().dirty);
        assert!(model
            .borrow()
            .status_message
            .as_deref()
            .unwrap()
            .contains("Failed to load"));
    }

    #[test]
    fn failed_save_preserves_draft_and_dirty_state() {
        let model = model();
        let before = model.borrow().draft.clone();
        let source_dir = model.borrow().source_dir.clone();
        model.borrow_mut().dirty = true;
        handle_save(
            "/definitely/not/a/directory/config.toml",
            before.clone(),
            source_dir,
            &model,
        );
        assert_eq!(model.borrow().draft, before);
        assert!(model.borrow().dirty);
    }

    #[test]
    fn save_snapshot_does_not_clear_intervening_edit() {
        let model = model();
        let snapshot = model.borrow().draft.clone();
        model.borrow_mut().draft.generator.seed = snapshot.generator.seed.wrapping_add(1);
        model.borrow_mut().dirty = true;
        let path = std::env::temp_dir().join(format!(
            "voxel-editor-save-{}-{}.toml",
            std::process::id(),
            snapshot.generator.seed
        ));
        let source_dir = model.borrow().source_dir.clone();
        handle_save(path.to_str().unwrap(), snapshot, source_dir, &model);
        assert!(model.borrow().dirty);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn saved_document_excludes_runtime_and_presentation_state() {
        let model = model();
        let text = config::save_preset_canonical(&model.borrow().draft).unwrap();
        for excluded in [
            "light_budget",
            "headless",
            "capture_dir",
            "env_path",
            "visible",
        ] {
            assert!(!text.contains(excluded));
        }
    }

    #[test]
    fn stale_status_does_not_replace_current_editor_status() {
        let model = model();
        model.borrow_mut().record_request_accepted(2);
        let old_identity = model.borrow().active_identity.clone();
        model.borrow_mut().record_failure(1, "stale".into());
        model
            .borrow_mut()
            .record_success(1, SceneConfigIdentity([9; 32]), ActiveStats::default());
        assert_eq!(model.borrow().active_identity, old_identity);
        assert!(matches!(
            model.borrow().phase,
            EditorPhase::Generating { request_id: 2 }
        ));
    }

    #[test]
    fn hide_command_is_drained_as_owned_main_loop_action() {
        let model = model();
        model.borrow_mut().enqueue(EditorCommand::Hide);
        let commands = model.borrow_mut().drain_commands();
        assert_eq!(commands.len(), 1);
        assert!(matches!(commands[0], EditorCommand::Hide));
        assert_eq!(model.borrow().queued_command_count(), 0);
    }

    #[test]
    fn asset_input_round_trips_catalog_and_absolute_file() {
        assert_eq!(
            parse_asset_input("catalog:abc").unwrap(),
            AssetRef::Catalog { id: "abc".into() }
        );
        let parsed = parse_asset_input("file:/tmp/a.png").unwrap();
        assert!(matches!(
            parsed,
            AssetRef::Filesystem {
                non_portable: true,
                ..
            }
        ));
        assert!(parse_asset_input("abc").is_err());
    }
}
