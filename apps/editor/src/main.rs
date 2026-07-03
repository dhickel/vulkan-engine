mod app_state;
mod launch;
mod panels;

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use app_state::{EditorAction, EditorSelection, EditorSession};
use launch::LaunchOptions;
use log::{error, info};
use renderer::{
    AssetManifestMode, AssetPolicyConfig, CommandHistory, DebugRuntimeMode, DurableAssetRecord,
    FrameRenderOutcome, PlaceAssetCommand, Project, RemoveNodeCommand, Renderer, RendererConfig,
    RendererError, RendererInitError, Scene, SceneAssetReference, SceneError, SceneNodeId,
    SetTransformCommand,
};
use winit::dpi::PhysicalSize;
use winit::event::MouseButton;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowBuilder};

const APP_NAME: &str = "Engine Editor";
const DEFAULT_SAMPLE_PROJECT: &str = "apps/editor/sample_project/engine.project.toml";
const COMMAND_HISTORY_DEPTH: usize = 128;

fn main() {
    init_logging();

    let launch_options = match LaunchOptions::parse_env() {
        Ok(options) => options,
        Err(err) => {
            error!("Failed to parse launch arguments: {err}");
            return;
        }
    };

    if let Err(err) = run(launch_options) {
        error!("{err}");
    }
}

fn run(launch_options: LaunchOptions) -> Result<(), String> {
    let event_loop =
        EventLoop::new().map_err(|err| format!("failed to create event loop: {err}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let config = RendererConfig {
        app_name: APP_NAME.to_string(),
        window_width: 1440,
        window_height: 900,
        compile_shaders: false,
        shader_debug_mode: DebugRuntimeMode::Default,
        asset_policy: AssetPolicyConfig {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
            ..AssetPolicyConfig::default()
        },
        ..RendererConfig::default()
    };

    let window = WindowBuilder::new()
        .with_title(APP_NAME)
        .with_inner_size(PhysicalSize::new(config.window_width, config.window_height))
        .build(&event_loop)
        .map_err(|err| format!("failed to create window: {err}"))?;

    let mut renderer = Renderer::new(config, &window)
        .map_err(|err| format!("renderer initialization failed: {err}"))?;
    renderer.install_default_fps_input();

    apply_debug_record_launch_options(&mut renderer, &launch_options)
        .map_err(|err| format!("failed to configure debug timing recording: {err}"))?;

    let mut project_context = load_project_context(launch_options.project_path.clone())?;
    let package_records = load_enabled_project_packages(&mut renderer, &mut project_context)?;

    let scene_path = launch_options
        .scene_path
        .clone()
        .or_else(|| project_context.startup_scene_path.clone());
    let mut scene = load_startup_scene(&mut renderer, scene_path.clone())
        .map_err(|err| format!("failed to initialize editor scene: {err}"))?;
    ensure_editor_root(&mut scene).map_err(|err| format!("failed to seed editor scene: {err}"))?;
    let mut history = CommandHistory::new(COMMAND_HISTORY_DEPTH);

    let session = Rc::new(RefCell::new(EditorSession::new(
        project_context.project_path.clone(),
        scene_path.clone(),
    )));
    {
        let mut session = session.borrow_mut();
        session.set_project_summary(
            project_context.project_path.clone(),
            project_context
                .project
                .as_ref()
                .map(|project| project.name.clone()),
            scene_path.clone(),
            project_context.enabled_package_count,
        );
        session.set_assets(package_records);
    }
    if let Some(project_path) = project_context.project_path.as_ref() {
        session
            .borrow_mut()
            .push_status(format!("Project path: {}", project_path.display()));
    }
    if let Some(project) = project_context.project.as_ref() {
        session
            .borrow_mut()
            .push_status(format!("Loaded project: {}", project.name));
    }
    if let Some(scene_path) = scene_path.as_ref() {
        session
            .borrow_mut()
            .push_status(format!("Scene path: {}", scene_path.display()));
    }

    let panel_session = session.clone();
    renderer
        .register_app_ui(
            "editor.workspace",
            Box::new(move |ui, ctx| {
                panels::render_editor_workspace(&panel_session, ui, ctx);
            }),
        )
        .map_err(|err| format!("failed to register editor UI: {err}"))?;

    info!("Editor initialized, starting event loop");
    window.request_redraw();

    let mut fps_timer = Instant::now();
    let mut frame_counter: u32 = 0;
    let mut modifiers = ModifiersState::default();
    let mut last_window_size = window.inner_size();
    let mut last_cursor_position: Option<(f32, f32)> = None;

    event_loop
        .run(move |event, control_flow| {
            if let Err(err) = renderer.update_input(&window, &event) {
                error!("Input update failed: {err}");
                control_flow.exit();
                return;
            }

            match event {
                Event::WindowEvent { window_id, event } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            control_flow.exit();
                        }
                        WindowEvent::KeyboardInput {
                            event: key_event, ..
                        } => {
                            if handle_fullscreen_toggle(&window, &key_event, modifiers) {
                                return;
                            }
                            if handle_editor_shortcut(
                                &session,
                                &key_event,
                                modifiers,
                                renderer.imgui_wants_keyboard_capture(),
                            ) {
                                return;
                            }
                            if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                                control_flow.exit();
                            }
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            last_cursor_position = Some((position.x as f32, position.y as f32));
                        }
                        WindowEvent::MouseInput {
                            state: ElementState::Pressed,
                            button: MouseButton::Left,
                            ..
                        } => {
                            if let Some((x, y)) = last_cursor_position {
                                queue_viewport_pick(&session, x, y);
                            }
                        }
                        WindowEvent::ModifiersChanged(next_modifiers) => {
                            modifiers = next_modifiers.state();
                        }
                        WindowEvent::Resized(new_size) => {
                            last_window_size = new_size;
                            if let Err(err) = renderer.resize(new_size.width, new_size.height) {
                                error!("Resize failed: {err}");
                                control_flow.exit();
                            }
                        }
                        WindowEvent::ScaleFactorChanged {
                            mut inner_size_writer,
                            ..
                        } => {
                            let new_size = window.inner_size();
                            if let Err(err) = inner_size_writer.request_inner_size(new_size) {
                                error!("Scale factor size request failed: {err}");
                                control_flow.exit();
                                return;
                            }
                            last_window_size = new_size;
                            if let Err(err) = renderer.resize(new_size.width, new_size.height) {
                                error!("Resize failed after scale change: {err}");
                                control_flow.exit();
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            session
                                .borrow_mut()
                                .refresh_scene_nodes(scene.node_summaries());
                            if let Err(err) = process_editor_actions(
                                &session,
                                &mut renderer,
                                &mut scene,
                                &mut history,
                                &window,
                            ) {
                                session
                                    .borrow_mut()
                                    .push_status(format!("Editor action failed: {err}"));
                            }

                            let current_size = window.inner_size();
                            if current_size != last_window_size {
                                last_window_size = current_size;
                                if let Err(err) =
                                    renderer.resize(current_size.width, current_size.height)
                                {
                                    error!("Resize failed while redrawing: {err}");
                                    control_flow.exit();
                                    return;
                                }
                            }

                            let outcome = match renderer.render_scene(&window, &mut scene) {
                                Ok(outcome) => outcome,
                                Err(err) => {
                                    error!("Render failed: {err}");
                                    control_flow.exit();
                                    return;
                                }
                            };

                            if outcome == FrameRenderOutcome::Rendered {
                                frame_counter = frame_counter.wrapping_add(1);
                                if fps_timer.elapsed() >= Duration::from_secs(1) {
                                    window.set_title(
                                        format!("{APP_NAME} - FPS: {frame_counter}").as_str(),
                                    );
                                    fps_timer = Instant::now();
                                    frame_counter = 0;
                                }
                            }

                            if outcome == FrameRenderOutcome::SkippedResizePending {
                                window.set_title(format!("{APP_NAME} - resizing...").as_str());
                            }

                            window.request_redraw();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        })
        .map_err(|err| format!("editor event loop failed: {err}"))?;

    Ok(())
}

fn load_startup_scene(
    renderer: &mut Renderer,
    scene_path: Option<PathBuf>,
) -> Result<Scene, RendererError> {
    if let Some(scene_path) = scene_path {
        if !scene_path.exists() {
            return Err(RendererError::Init(RendererInitError::StartupScene(
                format!("startup scene '{}' does not exist", scene_path.display()),
            )));
        }
        let mut assets = renderer.assets();
        return Scene::load(scene_path, &mut assets);
    }

    Ok(renderer.take_startup_scene().unwrap_or_else(Scene::new))
}

#[derive(Clone, Debug)]
struct ProjectContext {
    project_path: Option<PathBuf>,
    project: Option<Project>,
    project_root: PathBuf,
    startup_scene_path: Option<PathBuf>,
    enabled_package_count: usize,
}

fn load_project_context(requested_path: Option<PathBuf>) -> Result<ProjectContext, String> {
    let project_path = requested_path.or_else(|| {
        let default_path = PathBuf::from(DEFAULT_SAMPLE_PROJECT);
        default_path.exists().then_some(default_path)
    });

    let Some(project_path) = project_path else {
        return Ok(ProjectContext {
            project_path: None,
            project: None,
            project_root: PathBuf::new(),
            startup_scene_path: None,
            enabled_package_count: 0,
        });
    };

    let project = Project::load(&project_path)
        .map_err(|err| format!("failed to load project '{}': {err}", project_path.display()))?;
    let project_root = project_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_path_buf();
    let startup_scene_path = project
        .startup_scene
        .as_ref()
        .map(|scene| project_root.join(scene));
    let enabled_package_count = project
        .packages
        .iter()
        .filter(|package| package.enabled)
        .count();

    Ok(ProjectContext {
        project_path: Some(project_path),
        project: Some(project),
        project_root,
        startup_scene_path,
        enabled_package_count,
    })
}

fn load_enabled_project_packages(
    renderer: &mut Renderer,
    context: &mut ProjectContext,
) -> Result<Vec<DurableAssetRecord>, String> {
    let Some(project) = context.project.as_ref() else {
        return Ok(Vec::new());
    };

    let mut assets = renderer.assets();
    for package in project.packages.iter().filter(|package| package.enabled) {
        let manifest_path = context.project_root.join(&package.manifest);
        assets
            .load_package_manifest_with_expected_id(&manifest_path, &package.package_id)
            .map_err(|err| {
                format!(
                    "failed to load package '{}' from '{}': {err}",
                    package.package_id,
                    manifest_path.display()
                )
            })?;
    }

    Ok(assets.list_assets())
}

fn ensure_editor_root(scene: &mut Scene) -> Result<(), SceneError> {
    if scene.root().is_none() {
        let root = scene.create_node_default(None)?;
        scene.set_node_name(root, "Startup scene root")?;
    }
    Ok(())
}

fn process_editor_actions(
    session: &Rc<RefCell<EditorSession>>,
    renderer: &mut Renderer,
    scene: &mut Scene,
    history: &mut CommandHistory,
    window: &Window,
) -> Result<(), SceneError> {
    let actions = session.borrow_mut().drain_actions();
    for action in actions {
        match action {
            EditorAction::SelectNode(node) => select_node(session, scene, node),
            EditorAction::PickViewport { x, y } => {
                let size = window.inner_size();
                let picked = scene.pick_last_camera(x, y, size.width.max(1), size.height.max(1));
                if let Some(node) = picked {
                    select_node(session, scene, node);
                } else {
                    session.borrow_mut().clear_selection();
                    session
                        .borrow_mut()
                        .push_status("Viewport pick found no node");
                }
            }
            EditorAction::SetTool(mode) => {
                session.borrow_mut().set_tool_mode(mode);
                session
                    .borrow_mut()
                    .push_status(format!("{} tool active", mode.label()));
            }
            EditorAction::SetActiveScenePath(path) => {
                let trimmed = path.trim().to_string();
                let path = (!trimmed.is_empty()).then(|| PathBuf::from(trimmed));
                session.borrow_mut().set_active_scene_path(path);
                session
                    .borrow_mut()
                    .push_status("Active scene path updated");
            }
            EditorAction::SelectAsset(asset_id) => {
                let label = session
                    .borrow()
                    .assets()
                    .iter()
                    .find(|asset| asset.asset_id == asset_id)
                    .map(|asset| asset.display_name.clone())
                    .unwrap_or_else(|| asset_id.clone());
                session.borrow_mut().select_asset(asset_id);
                session
                    .borrow_mut()
                    .push_status(format!("Selected asset {label}"));
            }
            EditorAction::StartPlacement(asset_id) => {
                session.borrow_mut().start_placement(asset_id.clone());
                session
                    .borrow_mut()
                    .push_status(format!("Placement active for {asset_id}"));
            }
            EditorAction::CancelPlacement => {
                session.borrow_mut().cancel_placement();
                session.borrow_mut().push_status("Placement canceled");
            }
            EditorAction::ConfirmPlacement => {
                confirm_asset_placement(session, renderer, scene, history)?;
            }
            EditorAction::ApplyTransform { node, transform } => {
                if scene.is_valid_node(node) {
                    let result = scene.execute_command(
                        history,
                        Box::new(SetTransformCommand::new(node, transform.to_mat4())),
                    )?;
                    session
                        .borrow_mut()
                        .mark_dirty(format!("Command: {}", result.description));
                    select_node(session, scene, node);
                } else {
                    session.borrow_mut().clear_selection();
                }
            }
            EditorAction::SetNodeName { node, name } => {
                if scene.is_valid_node(node) {
                    scene.set_node_name(node, name.trim())?;
                    session
                        .borrow_mut()
                        .mark_dirty("Updated selected node name");
                    select_node(session, scene, node);
                } else {
                    session.borrow_mut().clear_selection();
                    session
                        .borrow_mut()
                        .push_status("Name edit ignored because selection is stale");
                }
            }
            EditorAction::SetNodeTags { node, tags } => {
                if scene.is_valid_node(node) {
                    scene.set_node_tags(node, tags)?;
                    session
                        .borrow_mut()
                        .mark_dirty("Updated selected node tags");
                    select_node(session, scene, node);
                } else {
                    session.borrow_mut().clear_selection();
                    session
                        .borrow_mut()
                        .push_status("Tag edit ignored because selection is stale");
                }
            }
            EditorAction::SetMaterialOverride {
                node,
                slot,
                override_id,
            } => {
                if scene.is_valid_node(node) {
                    scene.set_node_material_override(node, slot, override_id)?;
                    session
                        .borrow_mut()
                        .mark_dirty("Updated selected node material override");
                    select_node(session, scene, node);
                } else {
                    session.borrow_mut().clear_selection();
                    session
                        .borrow_mut()
                        .push_status("Material edit ignored because selection is stale");
                }
            }
            EditorAction::ClearMaterialOverride { node, slot } => {
                if scene.is_valid_node(node) {
                    scene.clear_node_material_override(node, slot)?;
                    session
                        .borrow_mut()
                        .mark_dirty("Cleared selected node material override");
                    select_node(session, scene, node);
                } else {
                    session.borrow_mut().clear_selection();
                    session
                        .borrow_mut()
                        .push_status("Material edit ignored because selection is stale");
                }
            }
            EditorAction::DeleteSelection => {
                let Some(selection) = session.borrow().selection().cloned() else {
                    continue;
                };
                if scene.is_valid_node(selection.runtime_id) {
                    let result = scene.execute_command(
                        history,
                        Box::new(RemoveNodeCommand::new(selection.runtime_id)),
                    )?;
                    session.borrow_mut().clear_selection();
                    session
                        .borrow_mut()
                        .mark_dirty(format!("Command: {}", result.description));
                } else {
                    session.borrow_mut().clear_selection();
                }
            }
            EditorAction::Undo => {
                let result = scene.undo_command(history)?;
                apply_command_result_to_selection(session, scene, &result.node_remap);
                session
                    .borrow_mut()
                    .mark_dirty(format!("Undo: {}", result.description));
            }
            EditorAction::Redo => {
                let result = scene.redo_command(history)?;
                apply_command_result_to_selection(session, scene, &result.node_remap);
                if let Some(node) = result.created_node {
                    select_node(session, scene, node);
                }
                cleanup_invalid_selection(session, scene);
                session
                    .borrow_mut()
                    .mark_dirty(format!("Redo: {}", result.description));
            }
            EditorAction::SaveScene => {
                let Some(path) = session.borrow().active_scene().map(Path::to_path_buf) else {
                    session
                        .borrow_mut()
                        .push_status("Save requires an active scene path");
                    continue;
                };
                match scene.save(&path) {
                    Ok(()) => {
                        session
                            .borrow_mut()
                            .mark_clean(format!("Saved scene {}", path.display()));
                    }
                    Err(err) => {
                        session
                            .borrow_mut()
                            .push_status(format!("Save failed: {err}"));
                    }
                }
            }
            EditorAction::LoadScene => {
                let Some(path) = session.borrow().active_scene().map(Path::to_path_buf) else {
                    session
                        .borrow_mut()
                        .push_status("Load requires an active scene path");
                    continue;
                };
                let mut assets = renderer.assets();
                match Scene::load(&path, &mut assets) {
                    Ok(loaded) => {
                        *scene = loaded;
                        *history = CommandHistory::new(COMMAND_HISTORY_DEPTH);
                        session.borrow_mut().clear_selection();
                        session
                            .borrow_mut()
                            .mark_clean(format!("Loaded scene {}", path.display()));
                    }
                    Err(err) => {
                        session
                            .borrow_mut()
                            .push_status(format!("Load failed: {err}"));
                    }
                }
            }
        }
    }

    session
        .borrow_mut()
        .refresh_scene_nodes(scene.node_summaries());
    cleanup_invalid_selection(session, scene);
    Ok(())
}

fn confirm_asset_placement(
    session: &Rc<RefCell<EditorSession>>,
    renderer: &mut Renderer,
    scene: &mut Scene,
    history: &mut CommandHistory,
) -> Result<(), SceneError> {
    let Some(placement) = session.borrow_mut().take_placement() else {
        session
            .borrow_mut()
            .push_status("No active placement to confirm");
        return Ok(());
    };

    let Some(record) = session
        .borrow()
        .assets()
        .iter()
        .find(|asset| asset.asset_id == placement.asset_id)
        .cloned()
    else {
        session
            .borrow_mut()
            .push_status(format!("Asset {} is not loaded", placement.asset_id));
        return Ok(());
    };

    let fragment = {
        let mut assets = renderer.assets();
        assets
            .load_model_asset(&record.asset_id)
            .map_err(|err| SceneError::MergeFailed(format!("asset placement failed: {err}")))?
    };
    let stable_id = session
        .borrow_mut()
        .next_placement_stable_id(&record.asset_id);
    let asset_ref = SceneAssetReference::new(
        record.asset_id.clone(),
        Some(record.package_relative_path.clone()),
    );
    let result = scene.execute_command(
        history,
        Box::new(PlaceAssetCommand::new(
            scene.root(),
            placement.transform.to_mat4(),
            fragment,
            asset_ref,
            record.display_name.clone(),
            record.tags.clone(),
            stable_id,
        )),
    )?;
    if let Some(node) = result.created_node {
        select_node(session, scene, node);
    }
    session.borrow_mut().push_status(format!(
        "Command: {} {}",
        result.description, record.asset_id
    ));
    session.borrow_mut().mark_dirty("Placed asset");
    Ok(())
}

fn select_node(session: &Rc<RefCell<EditorSession>>, scene: &Scene, node: SceneNodeId) {
    if let Some(summary) = scene
        .node_summaries()
        .into_iter()
        .find(|summary| summary.id == node)
    {
        let selection = EditorSelection {
            runtime_id: summary.id,
            stable_id: summary.stable_id.clone(),
            label: summary.name.clone(),
        };
        session.borrow_mut().set_selection(Some(selection));
        session
            .borrow_mut()
            .push_status(format!("Selected {}", summary.name));
    }
}

fn apply_command_result_to_selection(
    session: &Rc<RefCell<EditorSession>>,
    scene: &Scene,
    remap: &Option<renderer::SceneNodeRemap>,
) {
    let Some(remap) = remap else {
        return;
    };
    let selection = session.borrow().selection().cloned();
    if selection
        .as_ref()
        .is_some_and(|selection| selection.runtime_id == remap.old)
    {
        select_node(session, scene, remap.new);
    }
}

fn cleanup_invalid_selection(session: &Rc<RefCell<EditorSession>>, scene: &Scene) {
    let selection = session.borrow().selection().cloned();
    let Some(selection) = selection else {
        return;
    };
    if scene.is_valid_node(selection.runtime_id) {
        return;
    }
    if let Some(stable_id) = selection.stable_id.as_deref() {
        if let Some(remapped) = scene.find_node_by_stable_id(stable_id) {
            select_node(session, scene, remapped);
            return;
        }
    }
    session.borrow_mut().clear_selection();
}

fn queue_viewport_pick(session: &Rc<RefCell<EditorSession>>, x: f32, y: f32) {
    let viewport = session.borrow().viewport_rect();
    if viewport.is_some_and(|rect| rect.contains(x, y)) {
        session
            .borrow_mut()
            .queue_action(EditorAction::PickViewport { x, y });
    }
}

fn handle_editor_shortcut(
    session: &Rc<RefCell<EditorSession>>,
    key_event: &KeyEvent,
    modifiers: ModifiersState,
    keyboard_capture_active: bool,
) -> bool {
    if let Some(action) = editor_shortcut_action(
        key_event.state,
        key_event.repeat,
        key_event.physical_key,
        modifiers,
        keyboard_capture_active,
    ) {
        session.borrow_mut().queue_action(action);
        return true;
    }

    false
}

fn editor_shortcut_action(
    state: ElementState,
    repeat: bool,
    physical_key: PhysicalKey,
    modifiers: ModifiersState,
    keyboard_capture_active: bool,
) -> Option<EditorAction> {
    if keyboard_capture_active || state != ElementState::Pressed || repeat {
        return None;
    }

    match physical_key {
        PhysicalKey::Code(KeyCode::KeyZ) if modifiers.control_key() => Some(EditorAction::Undo),
        PhysicalKey::Code(KeyCode::KeyY) if modifiers.control_key() => Some(EditorAction::Redo),
        PhysicalKey::Code(KeyCode::KeyS) if modifiers.control_key() => {
            Some(EditorAction::SaveScene)
        }
        PhysicalKey::Code(KeyCode::KeyO) if modifiers.control_key() => {
            Some(EditorAction::LoadScene)
        }
        PhysicalKey::Code(KeyCode::Delete) | PhysicalKey::Code(KeyCode::Backspace) => {
            Some(EditorAction::DeleteSelection)
        }
        PhysicalKey::Code(KeyCode::KeyQ) => {
            Some(EditorAction::SetTool(app_state::ToolMode::Select))
        }
        PhysicalKey::Code(KeyCode::KeyT) => {
            Some(EditorAction::SetTool(app_state::ToolMode::Translate))
        }
        PhysicalKey::Code(KeyCode::KeyR) => {
            Some(EditorAction::SetTool(app_state::ToolMode::Rotate))
        }
        PhysicalKey::Code(KeyCode::KeyS) if !modifiers.control_key() => {
            Some(EditorAction::SetTool(app_state::ToolMode::Scale))
        }
        _ => None,
    }
}

fn apply_debug_record_launch_options(
    renderer: &mut Renderer,
    options: &LaunchOptions,
) -> Result<(), RendererError> {
    if options.record_debug_secs.is_none()
        && options.record_debug_interval_ms.is_none()
        && options.record_debug_path.is_none()
    {
        return Ok(());
    }

    renderer.configure_debug_timing_recording(
        options.record_debug_secs,
        options.record_debug_interval_ms,
        options.record_debug_path.clone(),
    )?;

    if options.record_debug_secs.is_some() {
        let path = renderer.start_debug_timing_recording()?;
        info!("Debug timing recording active -> {path}");
    }

    Ok(())
}

fn handle_fullscreen_toggle(
    window: &Window,
    key_event: &KeyEvent,
    modifiers: ModifiersState,
) -> bool {
    if key_event.state != ElementState::Pressed || key_event.repeat {
        return false;
    }

    if key_event.physical_key != PhysicalKey::Code(KeyCode::KeyF) || !modifiers.control_key() {
        return false;
    }

    let next_mode = if window.fullscreen().is_some() {
        None
    } else {
        Some(Fullscreen::Borderless(window.current_monitor()))
    };
    window.set_fullscreen(next_mode);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_shortcuts_are_suppressed_when_imgui_captures_keyboard() {
        let action = editor_shortcut_action(
            ElementState::Pressed,
            false,
            PhysicalKey::Code(KeyCode::KeyS),
            ModifiersState::default(),
            true,
        );

        assert_eq!(action, None);
    }

    #[test]
    fn editor_shortcuts_still_work_without_keyboard_capture() {
        let action = editor_shortcut_action(
            ElementState::Pressed,
            false,
            PhysicalKey::Code(KeyCode::KeyT),
            ModifiersState::default(),
            false,
        );

        assert_eq!(
            action,
            Some(EditorAction::SetTool(app_state::ToolMode::Translate))
        );
    }

    #[test]
    fn editor_shortcuts_ignore_repeated_key_events() {
        let action = editor_shortcut_action(
            ElementState::Pressed,
            true,
            PhysicalKey::Code(KeyCode::Delete),
            ModifiersState::default(),
            false,
        );

        assert_eq!(action, None);
    }
}

fn init_logging() {
    let _ = env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .try_init();

    info!("Starting engine editor");
}
