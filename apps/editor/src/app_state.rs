use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};

use glam::{Mat4, Quat, Vec3};
use renderer::{AssetKind, DurableAssetRecord, SceneAssetReference, SceneNodeId, SceneNodeSummary};

const STATUS_LIMIT: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct EditorSession {
    project_path: Option<PathBuf>,
    project_name: Option<String>,
    active_scene: Option<PathBuf>,
    active_scene_text: String,
    dirty: bool,
    package_count: usize,
    assets: Vec<DurableAssetRecord>,
    asset_search: String,
    asset_kind_filter: Option<AssetKind>,
    selected_asset_id: Option<String>,
    placement: Option<PlacementState>,
    next_placement_index: u64,
    selection: Option<EditorSelection>,
    tool_mode: ToolMode,
    hierarchy: Vec<EditorSceneNode>,
    transform_edit: Option<TransformEdit>,
    pending_actions: VecDeque<EditorAction>,
    viewport_rect: Option<ViewportRect>,
    panels: PanelVisibility,
    status_messages: VecDeque<String>,
}

impl EditorSession {
    pub fn new(project_path: Option<PathBuf>, active_scene: Option<PathBuf>) -> Self {
        let mut session = Self {
            project_path,
            project_name: None,
            active_scene_text: active_scene
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            active_scene,
            dirty: false,
            package_count: 0,
            assets: Vec::new(),
            asset_search: String::new(),
            asset_kind_filter: None,
            selected_asset_id: None,
            placement: None,
            next_placement_index: 1,
            selection: None,
            tool_mode: ToolMode::Select,
            hierarchy: Vec::new(),
            transform_edit: None,
            pending_actions: VecDeque::new(),
            viewport_rect: None,
            panels: PanelVisibility::default(),
            status_messages: VecDeque::new(),
        };

        session.push_status("Editor workspace initialized");
        session
    }

    pub fn project_path(&self) -> Option<&Path> {
        self.project_path.as_deref()
    }

    pub fn active_scene(&self) -> Option<&Path> {
        self.active_scene.as_deref()
    }

    pub fn active_scene_text(&self) -> &str {
        &self.active_scene_text
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn project_name(&self) -> Option<&str> {
        self.project_name.as_deref()
    }

    pub fn package_count(&self) -> usize {
        self.package_count
    }

    pub fn assets(&self) -> &[DurableAssetRecord] {
        &self.assets
    }

    pub fn asset_search(&self) -> &str {
        &self.asset_search
    }

    pub fn asset_kind_filter(&self) -> Option<&AssetKind> {
        self.asset_kind_filter.as_ref()
    }

    pub fn selected_asset_id(&self) -> Option<&str> {
        self.selected_asset_id.as_deref()
    }

    pub fn selected_asset(&self) -> Option<&DurableAssetRecord> {
        let selected = self.selected_asset_id.as_deref()?;
        self.assets.iter().find(|asset| asset.asset_id == selected)
    }

    pub fn placement(&self) -> Option<&PlacementState> {
        self.placement.as_ref()
    }

    pub fn selection(&self) -> Option<&EditorSelection> {
        self.selection.as_ref()
    }

    pub fn tool_mode(&self) -> ToolMode {
        self.tool_mode
    }

    pub fn hierarchy(&self) -> &[EditorSceneNode] {
        &self.hierarchy
    }

    pub fn transform_edit(&self) -> Option<TransformEdit> {
        self.transform_edit
    }

    pub fn viewport_rect(&self) -> Option<ViewportRect> {
        self.viewport_rect
    }

    pub fn panels(&self) -> &PanelVisibility {
        &self.panels
    }

    pub fn status_messages(&self) -> impl DoubleEndedIterator<Item = &str> {
        self.status_messages.iter().map(String::as_str)
    }

    pub fn set_selection(&mut self, selection: Option<EditorSelection>) {
        self.selection = selection;
    }

    pub fn clear_selection(&mut self) {
        self.selection = None;
        self.transform_edit = None;
    }

    pub fn set_tool_mode(&mut self, mode: ToolMode) {
        self.tool_mode = mode;
    }

    pub fn set_viewport_rect(&mut self, rect: ViewportRect) {
        self.viewport_rect = Some(rect);
    }

    pub fn queue_action(&mut self, action: EditorAction) {
        self.pending_actions.push_back(action);
    }

    pub fn drain_actions(&mut self) -> Vec<EditorAction> {
        self.pending_actions.drain(..).collect()
    }

    pub fn refresh_scene_nodes(&mut self, summaries: Vec<SceneNodeSummary>) {
        self.hierarchy = summaries
            .into_iter()
            .map(EditorSceneNode::from_summary)
            .collect();

        if let Some(selection) = self.selection.as_ref() {
            let selected = self
                .hierarchy
                .iter()
                .find(|node| node.id == selection.runtime_id)
                .or_else(|| {
                    selection.stable_id.as_ref().and_then(|stable_id| {
                        self.hierarchy
                            .iter()
                            .find(|node| node.stable_id.as_deref() == Some(stable_id.as_str()))
                    })
                })
                .cloned();

            if let Some(node) = selected {
                self.selection = Some(EditorSelection::from_node(&node));
                self.transform_edit = Some(TransformEdit::from_mat4(node.local_transform));
            } else {
                self.clear_selection();
                self.push_status("Selection cleared after node removal");
            }
        }
    }

    pub fn push_status(&mut self, message: impl Into<String>) {
        if self.status_messages.len() == STATUS_LIMIT {
            self.status_messages.pop_front();
        }
        self.status_messages.push_back(message.into());
    }

    pub fn set_project_summary(
        &mut self,
        project_path: Option<PathBuf>,
        project_name: Option<String>,
        active_scene: Option<PathBuf>,
        package_count: usize,
    ) {
        self.project_path = project_path;
        self.project_name = project_name;
        self.active_scene = active_scene;
        self.active_scene_text = self
            .active_scene
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        self.package_count = package_count;
    }

    pub fn set_active_scene_path(&mut self, path: Option<PathBuf>) {
        self.active_scene = path;
        self.active_scene_text = self
            .active_scene
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
    }

    pub fn set_active_scene_text(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.active_scene = text
            .trim()
            .is_empty()
            .then_some(None)
            .unwrap_or_else(|| Some(PathBuf::from(text.trim())));
        self.active_scene_text = text;
    }

    pub fn mark_dirty(&mut self, message: impl Into<String>) {
        self.dirty = true;
        self.push_status(message);
    }

    pub fn mark_clean(&mut self, message: impl Into<String>) {
        self.dirty = false;
        self.push_status(message);
    }

    pub fn set_assets(&mut self, mut assets: Vec<DurableAssetRecord>) {
        assets.sort_by(|a, b| a.asset_id.cmp(&b.asset_id));
        if let Some(selected) = self.selected_asset_id.as_deref() {
            if !assets.iter().any(|asset| asset.asset_id == selected) {
                self.selected_asset_id = None;
                self.placement = None;
            }
        }
        self.assets = assets;
    }

    pub fn set_asset_search(&mut self, search: impl Into<String>) {
        self.asset_search = search.into();
    }

    pub fn set_asset_kind_filter(&mut self, filter: Option<AssetKind>) {
        self.asset_kind_filter = filter;
    }

    pub fn filtered_assets(&self) -> Vec<DurableAssetRecord> {
        let needle = self.asset_search.trim().to_lowercase();
        self.assets
            .iter()
            .filter(|asset| {
                self.asset_kind_filter
                    .as_ref()
                    .map_or(true, |kind| &asset.kind == kind)
            })
            .filter(|asset| {
                if needle.is_empty() {
                    return true;
                }
                asset.asset_id.to_lowercase().contains(&needle)
                    || asset.display_name.to_lowercase().contains(&needle)
                    || asset.kind.as_str().contains(&needle)
                    || asset
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&needle))
            })
            .cloned()
            .collect()
    }

    pub fn select_asset(&mut self, asset_id: impl Into<String>) {
        self.selected_asset_id = Some(asset_id.into());
    }

    pub fn start_placement(&mut self, asset_id: impl Into<String>) {
        let asset_id = asset_id.into();
        self.selected_asset_id = Some(asset_id.clone());
        self.placement = Some(PlacementState {
            asset_id,
            transform: TransformEdit::identity(),
        });
        self.tool_mode = ToolMode::Place;
    }

    pub fn cancel_placement(&mut self) {
        self.placement = None;
        if self.tool_mode == ToolMode::Place {
            self.tool_mode = ToolMode::Select;
        }
    }

    pub fn set_placement_transform(&mut self, transform: TransformEdit) {
        if let Some(placement) = self.placement.as_mut() {
            placement.transform = transform;
        }
    }

    pub fn take_placement(&mut self) -> Option<PlacementState> {
        let placement = self.placement.take();
        if self.tool_mode == ToolMode::Place {
            self.tool_mode = ToolMode::Select;
        }
        placement
    }

    pub fn next_placement_stable_id(&mut self, asset_id: &str) -> String {
        let index = self.next_placement_index;
        self.next_placement_index += 1;
        let sanitized: String = asset_id
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    ch
                } else {
                    '_'
                }
            })
            .collect();
        format!("node.placed.{sanitized}.{index:06}")
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ToolMode {
    Select,
    Translate,
    Rotate,
    Scale,
    Place,
}

impl ToolMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Select => "Select",
            Self::Translate => "Translate",
            Self::Rotate => "Rotate",
            Self::Scale => "Scale",
            Self::Place => "Place",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditorSelection {
    pub runtime_id: SceneNodeId,
    pub stable_id: Option<String>,
    pub label: String,
}

impl EditorSelection {
    pub fn from_node(node: &EditorSceneNode) -> Self {
        Self {
            runtime_id: node.id,
            stable_id: node.stable_id.clone(),
            label: node.name.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditorSceneNode {
    pub id: SceneNodeId,
    pub parent: Option<SceneNodeId>,
    pub stable_id: Option<String>,
    pub name: String,
    pub local_transform: Mat4,
    pub asset: Option<SceneAssetReference>,
    pub material_overrides: BTreeMap<String, String>,
    pub tags: Vec<String>,
    pub child_count: usize,
    pub mesh_count: usize,
}

impl EditorSceneNode {
    fn from_summary(summary: SceneNodeSummary) -> Self {
        Self {
            id: summary.id,
            parent: summary.parent,
            stable_id: summary.stable_id,
            name: summary.name,
            local_transform: summary.local_transform,
            asset: summary.asset,
            material_overrides: summary.material_overrides,
            tags: summary.tags,
            child_count: summary.child_count,
            mesh_count: summary.mesh_count,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TransformEdit {
    pub translation: [f32; 3],
    pub rotation_degrees: [f32; 3],
    pub scale: [f32; 3],
}

impl TransformEdit {
    pub fn identity() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_degrees: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }

    pub fn from_mat4(transform: Mat4) -> Self {
        let (scale, rotation, translation) = transform.to_scale_rotation_translation();
        let (x, y, z) = rotation.to_euler(glam::EulerRot::XYZ);
        Self {
            translation: translation.to_array(),
            rotation_degrees: [x.to_degrees(), y.to_degrees(), z.to_degrees()],
            scale: scale.to_array(),
        }
    }

    pub fn to_mat4(self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::from_array(self.scale),
            Quat::from_euler(
                glam::EulerRot::XYZ,
                self.rotation_degrees[0].to_radians(),
                self.rotation_degrees[1].to_radians(),
                self.rotation_degrees[2].to_radians(),
            ),
            Vec3::from_array(self.translation),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlacementState {
    pub asset_id: String,
    pub transform: TransformEdit,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ViewportRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ViewportRect {
    pub fn contains(self, x: f32, y: f32) -> bool {
        x >= self.x && y >= self.y && x <= self.x + self.width && y <= self.y + self.height
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum EditorAction {
    SelectNode(SceneNodeId),
    PickViewport {
        x: f32,
        y: f32,
    },
    SetTool(ToolMode),
    ApplyTransform {
        node: SceneNodeId,
        transform: TransformEdit,
    },
    SetNodeName {
        node: SceneNodeId,
        name: String,
    },
    SetNodeTags {
        node: SceneNodeId,
        tags: Vec<String>,
    },
    SetMaterialOverride {
        node: SceneNodeId,
        slot: String,
        override_id: String,
    },
    ClearMaterialOverride {
        node: SceneNodeId,
        slot: String,
    },
    SetActiveScenePath(String),
    SelectAsset(String),
    StartPlacement(String),
    CancelPlacement,
    ConfirmPlacement,
    SaveScene,
    LoadScene,
    DeleteSelection,
    Undo,
    Redo,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelVisibility {
    pub asset_browser: bool,
    pub scene_hierarchy: bool,
    pub inspector: bool,
    pub status_log: bool,
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            asset_browser: true,
            scene_hierarchy: true,
            inspector: true,
            status_log: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn session_keeps_project_scene_and_default_panels() {
        let session = EditorSession::new(
            Some(PathBuf::from("engine.project.toml")),
            Some(PathBuf::from("scenes/start.engine.scene.json")),
        );

        assert_eq!(
            session.project_path(),
            Some(Path::new("engine.project.toml"))
        );
        assert_eq!(
            session.active_scene(),
            Some(Path::new("scenes/start.engine.scene.json"))
        );
        assert_eq!(
            session.active_scene_text(),
            "scenes/start.engine.scene.json"
        );
        assert!(!session.is_dirty());
        assert!(session.panels().asset_browser);
        assert!(session.panels().scene_hierarchy);
        assert!(session.panels().inspector);
        assert!(session.panels().status_log);
    }

    #[test]
    fn session_tracks_dirty_and_active_scene_path_controls() {
        let mut session = EditorSession::new(None, None);

        session.set_active_scene_text(" scenes/edited.engine.scene.json ");
        assert_eq!(
            session.active_scene(),
            Some(Path::new("scenes/edited.engine.scene.json"))
        );

        session.mark_dirty("changed transform");
        assert!(session.is_dirty());

        session.mark_clean("saved");
        assert!(!session.is_dirty());

        session.set_active_scene_text("   ");
        assert!(session.active_scene().is_none());
    }

    #[test]
    fn status_messages_are_bounded() {
        let mut session = EditorSession::new(None, None);
        for index in 0..80 {
            session.push_status(format!("message {index}"));
        }

        let messages: Vec<&str> = session.status_messages().collect();
        assert_eq!(messages.len(), STATUS_LIMIT);
        assert_eq!(messages.first(), Some(&"message 16"));
        assert_eq!(messages.last(), Some(&"message 79"));
    }

    #[test]
    fn stale_selection_clears_when_hierarchy_no_longer_contains_node() {
        let mut session = EditorSession::new(None, None);
        let selected = SceneNodeId::new(1, 0);
        session.set_selection(Some(EditorSelection {
            runtime_id: selected,
            stable_id: Some("node.gone".to_string()),
            label: "Gone".to_string(),
        }));

        session.refresh_scene_nodes(Vec::new());

        assert!(session.selection().is_none());
        assert!(session.transform_edit().is_none());
    }

    #[test]
    fn asset_browser_filters_records_by_kind_tag_and_search() {
        let mut session = EditorSession::new(None, None);
        session.set_assets(vec![
            test_asset("sample.model.block", AssetKind::Model, "Block", &["prop"]),
            test_asset(
                "sample.wall.stone_2m",
                AssetKind::WallChunk,
                "Stone Wall",
                &["wall", "chunk"],
            ),
        ]);

        session.set_asset_kind_filter(Some(AssetKind::WallChunk));
        session.set_asset_search("wall");

        let filtered = session.filtered_assets();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].asset_id, "sample.wall.stone_2m");
    }

    #[test]
    fn placement_state_has_explicit_start_cancel_and_stable_ids() {
        let mut session = EditorSession::new(None, None);
        session.start_placement("sample.wall.stone_2m");
        assert_eq!(session.tool_mode(), ToolMode::Place);
        assert_eq!(
            session
                .placement()
                .map(|placement| placement.asset_id.as_str()),
            Some("sample.wall.stone_2m")
        );

        let stable_id = session.next_placement_stable_id("sample.wall.stone_2m");
        assert_eq!(stable_id, "node.placed.sample_wall_stone_2m.000001");

        session.cancel_placement();
        assert_eq!(session.tool_mode(), ToolMode::Select);
        assert!(session.placement().is_none());
    }

    fn test_asset(
        asset_id: &str,
        kind: AssetKind,
        display_name: &str,
        tags: &[&str],
    ) -> DurableAssetRecord {
        DurableAssetRecord {
            package_id: "sample".to_string(),
            package_display_name: "Sample".to_string(),
            package_version: "0.1.0".to_string(),
            asset_id: asset_id.to_string(),
            kind,
            source_path: PathBuf::from("assets/sample.obj"),
            package_relative_path: PathBuf::from("sample.obj"),
            display_name: display_name.to_string(),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            material: None,
            materials: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}
