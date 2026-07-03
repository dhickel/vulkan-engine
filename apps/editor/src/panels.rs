use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use imgui::{Condition, Ui};
use renderer::{AssetKind, DebugUiFrameContext};

use crate::app_state::{EditorAction, EditorSelection, EditorSession, ToolMode, ViewportRect};

const MENU_HEIGHT: f32 = 26.0;
const TOOLBAR_HEIGHT: f32 = 38.0;
const STATUS_HEIGHT: f32 = 112.0;
const LEFT_WIDTH: f32 = 286.0;
const RIGHT_WIDTH: f32 = 320.0;
const MIN_CENTER_WIDTH: f32 = 360.0;
const MIN_CENTER_HEIGHT: f32 = 260.0;

pub fn render_editor_workspace(
    session: &Rc<RefCell<EditorSession>>,
    ui: &Ui,
    ctx: &DebugUiFrameContext,
) {
    let display_size = ui.io().display_size;
    let width = display_size[0].max(900.0);
    let height = display_size[1].max(600.0);
    let center_left = LEFT_WIDTH;
    let center_width = (width - LEFT_WIDTH - RIGHT_WIDTH).max(MIN_CENTER_WIDTH);
    let work_top = MENU_HEIGHT + TOOLBAR_HEIGHT;
    let work_height = (height - work_top - STATUS_HEIGHT).max(MIN_CENTER_HEIGHT);
    let right_x = center_left + center_width;

    render_menu_bar(session, ui, width);
    render_toolbar(session, ui, width, MENU_HEIGHT);
    render_left_column(session, ui, work_top, work_height);
    render_viewport(
        session,
        ui,
        ctx,
        center_left,
        work_top,
        center_width,
        work_height,
    );
    render_inspector(session, ui, right_x, work_top, RIGHT_WIDTH, work_height);
    render_status_log(session, ui, height - STATUS_HEIGHT, width, STATUS_HEIGHT);
}

fn render_menu_bar(session: &Rc<RefCell<EditorSession>>, ui: &Ui, width: f32) {
    ui.window("Editor Menu###editor_menu")
        .position([0.0, 0.0], Condition::Always)
        .size([width, MENU_HEIGHT], Condition::Always)
        .movable(false)
        .resizable(false)
        .collapsible(false)
        .title_bar(false)
        .build(|| {
            let dirty_marker = if session.borrow().is_dirty() {
                "Unsaved"
            } else {
                "Saved"
            };
            ui.text(dirty_marker);
            ui.same_line();
            if ui.small_button("Project") {
                session.borrow_mut().push_status("Project menu opened");
            }
            ui.same_line();
            if ui.small_button("Scene") {
                session.borrow_mut().push_status("Scene menu opened");
            }
            ui.same_line();
            if ui.small_button("Save") {
                session.borrow_mut().queue_action(EditorAction::SaveScene);
            }
            ui.same_line();
            if ui.small_button("Load") {
                session.borrow_mut().queue_action(EditorAction::LoadScene);
            }
        });
}

fn render_toolbar(session: &Rc<RefCell<EditorSession>>, ui: &Ui, width: f32, y: f32) {
    ui.window("Editor Toolbar###editor_toolbar")
        .position([0.0, y], Condition::Always)
        .size([width, TOOLBAR_HEIGHT], Condition::Always)
        .movable(false)
        .resizable(false)
        .collapsible(false)
        .title_bar(false)
        .build(|| {
            let active = session.borrow().tool_mode();
            ui.text("Tool");
            for mode in [
                ToolMode::Select,
                ToolMode::Translate,
                ToolMode::Rotate,
                ToolMode::Scale,
                ToolMode::Place,
            ] {
                ui.same_line();
                let label = if active == mode {
                    format!("[{}]##tool_{mode:?}", mode.label())
                } else {
                    format!("{}##tool_{mode:?}", mode.label())
                };
                if ui.small_button(label) {
                    let mut session = session.borrow_mut();
                    session.set_tool_mode(mode);
                    session.queue_action(EditorAction::SetTool(mode));
                }
            }
            ui.same_line();
            if ui.small_button("Undo##toolbar_undo") {
                session.borrow_mut().queue_action(EditorAction::Undo);
            }
            ui.same_line();
            if ui.small_button("Redo##toolbar_redo") {
                session.borrow_mut().queue_action(EditorAction::Redo);
            }
            ui.same_line();
            if ui.small_button("Delete##toolbar_delete") {
                session
                    .borrow_mut()
                    .queue_action(EditorAction::DeleteSelection);
            }
        });
}

fn render_left_column(session: &Rc<RefCell<EditorSession>>, ui: &Ui, y: f32, height: f32) {
    let half_height = (height * 0.5).max(180.0);
    if session.borrow().panels().asset_browser {
        ui.window("Asset Browser###editor_asset_browser")
            .position([0.0, y], Condition::Always)
            .size([LEFT_WIDTH, half_height], Condition::Always)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .build(|| {
                ui.text("Packages");
                ui.separator();
                let (project_path, project_name, package_count, total_assets) = {
                    let session = session.borrow();
                    (
                        session.project_path().map(Path::to_path_buf),
                        session.project_name().map(str::to_string),
                        session.package_count(),
                        session.assets().len(),
                    )
                };
                if let Some(project_name) = project_name {
                    ui.text(project_name);
                }
                if let Some(project_path) = project_path.as_deref() {
                    ui.text_wrapped(format_path("Project", project_path));
                    ui.text_disabled(format!(
                        "{package_count} package(s), {total_assets} asset(s)"
                    ));
                } else {
                    ui.text_disabled("No project asset registry loaded");
                }
                ui.separator();
                render_asset_filters(session, ui);
                ui.separator();
                render_asset_records(session, ui);
                ui.separator();
                render_placement_controls(session, ui);
            });
    }

    if session.borrow().panels().scene_hierarchy {
        ui.window("Scene Hierarchy###editor_scene_hierarchy")
            .position([0.0, y + half_height], Condition::Always)
            .size([LEFT_WIDTH, height - half_height], Condition::Always)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .build(|| {
                ui.text("Active Scene");
                ui.separator();
                if let Some(scene_path) = session.borrow().active_scene() {
                    ui.text(format_path("Scene", scene_path));
                } else {
                    ui.text_disabled("Unsaved startup scene");
                }
                let mut scene_path_text = session.borrow().active_scene_text().to_string();
                if ui
                    .input_text("Path##active_scene_path", &mut scene_path_text)
                    .build()
                {
                    session
                        .borrow_mut()
                        .queue_action(EditorAction::SetActiveScenePath(scene_path_text));
                }
                ui.separator();
                let nodes = session.borrow().hierarchy().to_vec();
                if nodes.is_empty() {
                    ui.text_disabled("No scene nodes");
                }
                for node in nodes {
                    let indent = if node.parent.is_some() { "  " } else { "" };
                    let selected = session
                        .borrow()
                        .selection()
                        .is_some_and(|selection| selection.runtime_id == node.id);
                    let marker = if selected { ">" } else { " " };
                    let label = format!(
                        "{marker}{indent}{}  [{}:{}]##node_{}_{}",
                        node.name,
                        node.id.slot,
                        node.id.generation,
                        node.id.slot,
                        node.id.generation
                    );
                    if ui.small_button(label) {
                        let mut session = session.borrow_mut();
                        session.set_selection(Some(EditorSelection::from_node(&node)));
                        session.queue_action(EditorAction::SelectNode(node.id));
                    }
                    if node.child_count > 0 || node.mesh_count > 0 {
                        ui.same_line();
                        ui.text_disabled(format!(
                            "{} child / {} mesh",
                            node.child_count, node.mesh_count
                        ));
                    }
                }
                ui.text_disabled("Single-root editor scene authoring only");
            });
    }
}

fn render_asset_filters(session: &Rc<RefCell<EditorSession>>, ui: &Ui) {
    let mut search = session.borrow().asset_search().to_string();
    if ui.input_text("Search##asset_search", &mut search).build() {
        session.borrow_mut().set_asset_search(search);
    }

    ui.text("Kind");
    let active = session.borrow().asset_kind_filter().cloned();
    for (label, kind) in [
        ("All", None),
        ("Model", Some(AssetKind::Model)),
        ("Prefab", Some(AssetKind::Prefab)),
        ("Wall", Some(AssetKind::WallChunk)),
    ] {
        ui.same_line();
        let selected = active == kind;
        let button = if selected {
            format!("[{label}]##asset_kind_{label}")
        } else {
            format!("{label}##asset_kind_{label}")
        };
        if ui.small_button(button) {
            session.borrow_mut().set_asset_kind_filter(kind);
        }
    }
}

fn render_asset_records(session: &Rc<RefCell<EditorSession>>, ui: &Ui) {
    let assets = session.borrow().filtered_assets();
    if assets.is_empty() {
        ui.text_disabled("No assets match the current filter");
        return;
    }

    for asset in assets {
        let selected = session.borrow().selected_asset_id() == Some(asset.asset_id.as_str());
        let marker = if selected { ">" } else { " " };
        let tag_text = if asset.tags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", asset.tags.join(","))
        };
        let label = format!(
            "{marker} {} ({}){}##asset_{}",
            asset.display_name, asset.kind, tag_text, asset.asset_id
        );
        if ui.small_button(label) {
            session
                .borrow_mut()
                .queue_action(EditorAction::SelectAsset(asset.asset_id.clone()));
        }
        ui.text_disabled(asset.asset_id);
    }
}

fn render_placement_controls(session: &Rc<RefCell<EditorSession>>, ui: &Ui) {
    if let Some(asset) = session.borrow().selected_asset().cloned() {
        if is_placeable_kind(&asset.kind) {
            if ui.small_button("Place Selected##place_selected_asset") {
                session
                    .borrow_mut()
                    .queue_action(EditorAction::StartPlacement(asset.asset_id.clone()));
            }
        } else {
            ui.text_disabled("Selected asset is not placeable");
        }
    } else {
        ui.text_disabled("Select a model, prefab, or wall chunk to place");
    }

    let placement = session.borrow().placement().cloned();
    if let Some(mut placement) = placement {
        ui.separator();
        ui.text("Active Placement");
        ui.text_disabled(&placement.asset_id);
        let mut changed = false;
        changed |= ui
            .input_float3(
                "Translate##placement_translate",
                &mut placement.transform.translation,
            )
            .build();
        changed |= ui
            .input_float3(
                "Rotate##placement_rotate",
                &mut placement.transform.rotation_degrees,
            )
            .build();
        changed |= ui
            .input_float3("Scale##placement_scale", &mut placement.transform.scale)
            .build();
        if changed {
            session
                .borrow_mut()
                .set_placement_transform(placement.transform);
        }
        if ui.small_button("Confirm##confirm_placement") {
            session
                .borrow_mut()
                .queue_action(EditorAction::ConfirmPlacement);
        }
        ui.same_line();
        if ui.small_button("Cancel##cancel_placement") {
            session
                .borrow_mut()
                .queue_action(EditorAction::CancelPlacement);
        }
    }
}

fn is_placeable_kind(kind: &AssetKind) -> bool {
    matches!(
        kind,
        AssetKind::Model | AssetKind::Prefab | AssetKind::WallChunk
    )
}

fn render_viewport(
    session: &Rc<RefCell<EditorSession>>,
    ui: &Ui,
    ctx: &DebugUiFrameContext,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
) {
    session.borrow_mut().set_viewport_rect(ViewportRect {
        x,
        y,
        width,
        height,
    });
    ui.window("Viewport###editor_viewport")
        .position([x, y], Condition::Always)
        .size([width, height], Condition::Always)
        .movable(false)
        .resizable(false)
        .collapsible(false)
        .build(|| {
            ui.text("Live Viewport");
            ui.separator();
            ui.text(format!(
                "Renderer: {} x {}",
                ctx.viewport_size.0, ctx.viewport_size.1
            ));
            ui.text(format!("Frame: {}", ctx.frame_index));
            ui.text(format!("FPS: {:.1}", ctx.fps));
            ui.text(format!("Tool: {}", session.borrow().tool_mode().label()));
            ui.text_disabled("Click in this panel to pick the nearest scene proxy bound");
        });
}

fn render_inspector(
    session: &Rc<RefCell<EditorSession>>,
    ui: &Ui,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
) {
    if !session.borrow().panels().inspector {
        return;
    }

    ui.window("Inspector###editor_inspector")
        .position([x, y], Condition::Always)
        .size([width, height], Condition::Always)
        .movable(false)
        .resizable(false)
        .collapsible(false)
        .build(|| {
            ui.text("Selection");
            ui.separator();
            let selected_node = {
                let session = session.borrow();
                session.selection().and_then(|selection| {
                    session
                        .hierarchy()
                        .iter()
                        .find(|node| node.id == selection.runtime_id)
                        .cloned()
                })
            };
            if let Some(node) = selected_node.as_ref() {
                ui.text(&node.name);
                ui.text_disabled(format!("Runtime [{}:{}]", node.id.slot, node.id.generation));
                if let Some(stable_id) = node.stable_id.as_ref() {
                    ui.text_disabled(stable_id);
                }
                if let Some(asset) = node.asset.as_ref() {
                    ui.text_wrapped(format!("Asset: {}", asset.id));
                    if let Some(path_hint) = asset.path_hint.as_ref() {
                        ui.text_disabled(format!("Path hint: {}", path_hint.display()));
                    }
                } else {
                    ui.text_disabled("Asset: none");
                }
            } else {
                ui.text_disabled("Nothing selected");
            }
            ui.separator();
            ui.text("Settings");
            if let Some(node) = selected_node.as_ref() {
                let mut name = node.name.clone();
                if ui.input_text("Name##selected_node_name", &mut name).build() {
                    session.borrow_mut().queue_action(EditorAction::SetNodeName {
                        node: node.id,
                        name,
                    });
                }
                let mut tags = node.tags.join(", ");
                if ui.input_text("Tags##selected_node_tags", &mut tags).build() {
                    session.borrow_mut().queue_action(EditorAction::SetNodeTags {
                        node: node.id,
                        tags: parse_tag_list(&tags),
                    });
                }
            } else {
                ui.text_disabled("Select a node to edit name and tags");
            }
            ui.separator();
            ui.text("Transform");
            let selection = session.borrow().selection().cloned();
            let transform = session.borrow().transform_edit();
            if let (Some(selection), Some(mut edit)) = (selection, transform) {
                let mut changed = false;
                changed |= ui.input_float3("Translate", &mut edit.translation).build();
                changed |= ui
                    .input_float3("Rotate", &mut edit.rotation_degrees)
                    .build();
                changed |= ui.input_float3("Scale", &mut edit.scale).build();
                if changed {
                    session
                        .borrow_mut()
                        .queue_action(EditorAction::ApplyTransform {
                            node: selection.runtime_id,
                            transform: edit,
                        });
                }
            } else {
                ui.text_disabled("Select a node to edit numeric transform values");
            }
            ui.separator();
            ui.text("Material");
            if let Some(node) = selected_node.as_ref() {
                ui.text_disabled("Scene metadata only: slot override IDs");
                let current = node
                    .material_overrides
                    .get("0")
                    .cloned()
                    .unwrap_or_default();
                let mut override_id = current.clone();
                if ui
                    .input_text("Slot 0 override##material_slot_0", &mut override_id)
                    .build()
                {
                    let trimmed = override_id.trim().to_string();
                    if trimmed.is_empty() {
                        session
                            .borrow_mut()
                            .queue_action(EditorAction::ClearMaterialOverride {
                                node: node.id,
                                slot: "0".to_string(),
                            });
                    } else {
                        session
                            .borrow_mut()
                            .queue_action(EditorAction::SetMaterialOverride {
                                node: node.id,
                                slot: "0".to_string(),
                                override_id: trimmed,
                            });
                    }
                }
                if node.material_overrides.is_empty() {
                    ui.text_disabled("No material overrides assigned");
                } else {
                    for (slot, material) in &node.material_overrides {
                        ui.text_disabled(format!("slot {slot} -> {material}"));
                    }
                }
                ui.text_disabled("PBR factors, textures, shader graphs, and material asset documents are deferred");
            } else {
                ui.text_disabled("Select a node to edit material metadata");
            }
        });
}

fn render_status_log(
    session: &Rc<RefCell<EditorSession>>,
    ui: &Ui,
    y: f32,
    width: f32,
    height: f32,
) {
    if !session.borrow().panels().status_log {
        return;
    }

    ui.window("Status###editor_status")
        .position([0.0, y], Condition::Always)
        .size([width, height], Condition::Always)
        .movable(false)
        .resizable(false)
        .collapsible(false)
        .build(|| {
            ui.text("Status");
            ui.separator();
            for message in session.borrow().status_messages().rev().take(5) {
                ui.bullet_text(message);
            }
        });
}

fn format_path(label: &str, path: &Path) -> String {
    format!("{label}: {}", path.display())
}

fn parse_tag_list(tags: &str) -> Vec<String> {
    tags.split(',')
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(str::to_string)
        .collect()
}
