use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(test)]
use engine_events::EventEnvelope;
use engine_events::{
    AssetEvent, EngineEvent, EventBus, EventRecorder, EventStage, LifecycleEvent, PackageId,
    ProjectId, SceneId,
};
use log::{info, warn};
use renderer::{
    default_capture_run_dir, single_capture_path, AssetManifestMode, AssetPolicyConfig,
    DurableAssetRecord, FrameCaptureRequest, FrameCaptureSequence, FrameCaptureStatus,
    FrameRenderOutcome, Project, ProjectPackage, ProjectValidationOptions, Renderer,
    RendererConfig, Scene, SceneValidationOptions,
};
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::WindowBuilder;

use crate::launch::LaunchOptions;

const FALLBACK_APP_NAME: &str = "engine";
const DEFAULT_HEADLESS_SMOKE_FRAMES: u32 = 3;
const RUNTIME_EVENT_RECORDER_CAPACITY: usize = 512;

#[derive(Clone, Debug)]
struct RuntimeProject {
    project_path: PathBuf,
    project_root: PathBuf,
    project: Project,
    startup_scene_path: PathBuf,
    enabled_package_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HeadlessCapturePlan {
    expected_captures: usize,
    frame_budget: u32,
}

struct RuntimeEvents {
    bus: EventBus,
}

impl RuntimeEvents {
    fn new() -> Self {
        Self {
            bus: EventBus::with_recorder(EventRecorder::bounded(RUNTIME_EVENT_RECORDER_CAPACITY)),
        }
    }

    fn emit(&mut self, stage: EventStage, event: EngineEvent) {
        self.bus.emit(stage, None, event);
        let report = self.bus.drain_stage(stage);
        for failure in report.failures {
            warn!(
                "runtime event listener {:?} failed for event {:?}: {}",
                failure.listener, failure.sequence, failure.message
            );
        }
    }

    #[cfg(test)]
    fn recorded(&self) -> Vec<EventEnvelope> {
        self.bus
            .recorder()
            .map(|recorder| recorder.entries().cloned().collect())
            .unwrap_or_default()
    }
}

pub fn run(options: LaunchOptions) -> Result<(), String> {
    let mut runtime_events = RuntimeEvents::new();
    runtime_events.emit(
        EventStage::Startup,
        EngineEvent::Lifecycle(LifecycleEvent::AppStarting {
            app_name: FALLBACK_APP_NAME.to_string(),
        }),
    );
    runtime_events.emit(
        EventStage::ProjectLoad,
        EngineEvent::Lifecycle(LifecycleEvent::ProjectLoading {
            path: options.project_path.display().to_string(),
        }),
    );

    let runtime_project = load_runtime_project(&options)?;
    let config = renderer_config(&runtime_project.project, options.headless);
    let capture_run_dir = default_capture_run_dir(config.app_name.as_str());
    emit_runtime_project_loaded(&mut runtime_events, &runtime_project);
    runtime_events.emit(
        EventStage::Startup,
        EngineEvent::Lifecycle(LifecycleEvent::AppStarted {
            app_name: config.app_name.clone(),
        }),
    );

    if options.headless {
        run_headless(
            runtime_project,
            config,
            options,
            &capture_run_dir,
            &mut runtime_events,
        )
    } else {
        run_windowed(
            runtime_project,
            config,
            options,
            &capture_run_dir,
            &mut runtime_events,
        )
    }
}

fn load_runtime_project(options: &LaunchOptions) -> Result<RuntimeProject, String> {
    let project_path = resolve_project_path(&options.project_path)?;
    if !project_path.exists() {
        return Err(format!(
            "project file '{}' does not exist",
            project_path.display()
        ));
    }
    if !project_path.is_file() {
        return Err(format!(
            "project path '{}' is not a file",
            project_path.display()
        ));
    }

    let project = renderer::validate_project_file(
        &project_path,
        &ProjectValidationOptions::default().check_files(true),
    )
    .map_err(|err| {
        format!(
            "project validation failed for '{}': {err}",
            project_path.display()
        )
    })?;

    let project_root = project_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_path_buf();
    let startup_scene_path = resolve_startup_scene_path(
        &project_root,
        options.scene_path.as_deref(),
        project.startup_scene.as_deref(),
    )?;
    if !startup_scene_path.exists() {
        return Err(format!(
            "startup scene '{}' does not exist",
            startup_scene_path.display()
        ));
    }

    let enabled_package_count = project
        .packages
        .iter()
        .filter(|package| package.enabled)
        .count();

    Ok(RuntimeProject {
        project_path,
        project_root,
        project,
        startup_scene_path,
        enabled_package_count,
    })
}

fn resolve_project_path(path: &Path) -> Result<PathBuf, String> {
    if path.as_os_str().is_empty() {
        return Err("project path must not be empty".to_string());
    }
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .map_err(|err| format!("failed to resolve current directory: {err}"))
    }
}

fn resolve_startup_scene_path(
    project_root: &Path,
    cli_scene: Option<&Path>,
    project_scene: Option<&Path>,
) -> Result<PathBuf, String> {
    let scene = cli_scene.or(project_scene).ok_or_else(|| {
        "no startup scene supplied by --scene or project startup_scene".to_string()
    })?;
    if scene.as_os_str().is_empty() {
        return Err("startup scene path must not be empty".to_string());
    }
    if scene.is_absolute() {
        Ok(scene.to_path_buf())
    } else {
        Ok(project_root.join(scene))
    }
}

fn renderer_config(project: &Project, headless: bool) -> RendererConfig {
    RendererConfig {
        app_name: if project.name.trim().is_empty() {
            FALLBACK_APP_NAME.to_string()
        } else {
            project.name.clone()
        },
        window_width: project.settings.window_width,
        window_height: project.settings.window_height,
        headless,
        preload_startup_scene: false,
        asset_policy: AssetPolicyConfig {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
            ..AssetPolicyConfig::default()
        },
        ..RendererConfig::default()
    }
}

fn run_headless(
    runtime_project: RuntimeProject,
    config: RendererConfig,
    options: LaunchOptions,
    capture_run_dir: &Path,
    runtime_events: &mut RuntimeEvents,
) -> Result<(), String> {
    let mut renderer = Renderer::new_headless(config)
        .map_err(|err| format!("headless renderer initialization failed: {err}"))?;

    apply_frame_capture_launch_options(&mut renderer, &options, capture_run_dir)
        .map_err(|err| format!("failed to configure frame capture: {err}"))?;
    apply_debug_record_launch_options(&mut renderer, &options)
        .map_err(|err| format!("failed to configure debug timing recording: {err}"))?;

    let package_records =
        load_enabled_project_packages(&mut renderer, &runtime_project, runtime_events)?;
    info!(
        "Loaded {} package-backed asset record(s) from {} enabled package(s)",
        package_records.len(),
        runtime_project.enabled_package_count
    );
    emit_runtime_scene_loading(runtime_events, &runtime_project);
    validate_startup_scene(&runtime_project.startup_scene_path, &package_records)?;
    let mut scene = load_startup_scene(&mut renderer, &runtime_project.startup_scene_path)?;
    emit_runtime_scene_loaded(runtime_events, &runtime_project);

    let plan = headless_capture_plan(&options);
    let mut succeeded_paths = HashSet::new();

    for _ in 0..plan.frame_budget {
        match renderer.render_scene_headless(&mut scene) {
            Ok(FrameRenderOutcome::Rendered) | Ok(FrameRenderOutcome::SkippedResizePending) => {}
            Err(err) => return Err(format!("headless render failed: {err}")),
        }

        match renderer.last_frame_capture_status() {
            Some(FrameCaptureStatus::Succeeded { output_path, .. }) => {
                succeeded_paths.insert(output_path.clone());
                if succeeded_paths.len() >= plan.expected_captures && plan.expected_captures > 0 {
                    info!(
                        "Headless runtime capture completed: {} capture(s) written",
                        succeeded_paths.len()
                    );
                    runtime_events.emit(
                        EventStage::Shutdown,
                        EngineEvent::Lifecycle(LifecycleEvent::ShutdownCompleted),
                    );
                    return Ok(());
                }
            }
            Some(FrameCaptureStatus::Failed { message, .. }) => {
                return Err(format!("headless capture failed: {message}"));
            }
            Some(FrameCaptureStatus::BackendNotImplemented { target, .. }) => {
                return Err(format!(
                    "headless capture target '{}' is not implemented",
                    target.as_label()
                ));
            }
            _ => {}
        }
    }

    if plan.expected_captures == 0 {
        runtime_events.emit(
            EventStage::Shutdown,
            EngineEvent::Lifecycle(LifecycleEvent::ShutdownCompleted),
        );
        Ok(())
    } else {
        Err(format!(
            "headless capture did not complete within {} frames ({} of {} capture(s) written)",
            plan.frame_budget,
            succeeded_paths.len(),
            plan.expected_captures
        ))
    }
}

fn run_windowed(
    runtime_project: RuntimeProject,
    config: RendererConfig,
    options: LaunchOptions,
    capture_run_dir: &Path,
    mut runtime_events: &mut RuntimeEvents,
) -> Result<(), String> {
    let event_loop =
        EventLoop::new().map_err(|err| format!("failed to create event loop: {err}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = WindowBuilder::new()
        .with_title(config.app_name.as_str())
        .with_inner_size(PhysicalSize::new(config.window_width, config.window_height))
        .build(&event_loop)
        .map_err(|err| format!("failed to create window: {err}"))?;

    let mut renderer = Renderer::new(config, &window)
        .map_err(|err| format!("renderer initialization failed: {err}"))?;
    renderer.install_default_fps_input();

    apply_frame_capture_launch_options(&mut renderer, &options, capture_run_dir)
        .map_err(|err| format!("failed to configure frame capture: {err}"))?;
    apply_debug_record_launch_options(&mut renderer, &options)
        .map_err(|err| format!("failed to configure debug timing recording: {err}"))?;

    let package_records =
        load_enabled_project_packages(&mut renderer, &runtime_project, runtime_events)?;
    info!(
        "Loaded {} package-backed asset record(s) from {} enabled package(s)",
        package_records.len(),
        runtime_project.enabled_package_count
    );
    emit_runtime_scene_loading(runtime_events, &runtime_project);
    validate_startup_scene(&runtime_project.startup_scene_path, &package_records)?;
    let mut scene = load_startup_scene(&mut renderer, &runtime_project.startup_scene_path)?;
    emit_runtime_scene_loaded(runtime_events, &runtime_project);

    info!(
        "Launching project '{}' from '{}' with scene '{}'",
        runtime_project.project.name,
        runtime_project.project_path.display(),
        runtime_project.startup_scene_path.display()
    );

    let mut fps_timer = Instant::now();
    let mut frame_counter: u32 = 0;
    let mut last_window_size = window.inner_size();
    let app_name = runtime_project.project.name.clone();
    window.request_redraw();

    event_loop
        .run(move |event, control_flow| {
            if let Err(err) = renderer.update_input(&window, &event) {
                log::error!("Input update failed: {err}");
                control_flow.exit();
                return;
            }

            match event {
                Event::WindowEvent { window_id, event } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            emit_runtime_shutdown_requested(
                                &mut runtime_events,
                                "window close requested",
                            );
                            control_flow.exit();
                        }
                        WindowEvent::KeyboardInput { event, .. } => {
                            if !event.repeat
                                && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Escape))
                            {
                                emit_runtime_shutdown_requested(
                                    &mut runtime_events,
                                    "escape key requested shutdown",
                                );
                                control_flow.exit();
                            }
                        }
                        WindowEvent::Resized(new_size) => {
                            last_window_size = new_size;
                            if let Err(err) = renderer.resize(new_size.width, new_size.height) {
                                log::error!("Resize failed: {err}");
                                control_flow.exit();
                            }
                        }
                        WindowEvent::ScaleFactorChanged {
                            mut inner_size_writer,
                            ..
                        } => {
                            let new_size = window.inner_size();
                            if let Err(err) = inner_size_writer.request_inner_size(new_size) {
                                log::error!("Scale factor size request failed: {err}");
                                control_flow.exit();
                                return;
                            }
                            last_window_size = new_size;
                            if let Err(err) = renderer.resize(new_size.width, new_size.height) {
                                log::error!("Resize failed after scale change: {err}");
                                control_flow.exit();
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let current_size = window.inner_size();
                            if current_size != last_window_size {
                                last_window_size = current_size;
                                if let Err(err) =
                                    renderer.resize(current_size.width, current_size.height)
                                {
                                    log::error!("Resize failed while redrawing: {err}");
                                    control_flow.exit();
                                    return;
                                }
                            }

                            let outcome = match renderer.render_scene(&window, &mut scene) {
                                Ok(outcome) => outcome,
                                Err(err) => {
                                    log::error!("Render failed: {err}");
                                    control_flow.exit();
                                    return;
                                }
                            };

                            if outcome == FrameRenderOutcome::Rendered {
                                frame_counter = frame_counter.wrapping_add(1);
                                if fps_timer.elapsed() >= Duration::from_secs(1) {
                                    window.set_title(
                                        format!("{app_name} - FPS: {frame_counter}").as_str(),
                                    );
                                    fps_timer = Instant::now();
                                    frame_counter = 0;
                                }
                            }

                            if outcome == FrameRenderOutcome::SkippedResizePending {
                                window.set_title(format!("{app_name} - resizing...").as_str());
                            }

                            window.request_redraw();
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        })
        .map_err(|err| format!("runtime event loop failed: {err}"))?;

    Ok(())
}

fn load_enabled_project_packages(
    renderer: &mut Renderer,
    runtime_project: &RuntimeProject,
    runtime_events: &mut RuntimeEvents,
) -> Result<Vec<DurableAssetRecord>, String> {
    let mut all_records = Vec::new();
    let mut assets = renderer.assets();
    for package in runtime_project
        .project
        .packages
        .iter()
        .filter(|package| package.enabled)
    {
        let manifest_path = runtime_project.project_root.join(&package.manifest);
        runtime_events.emit(
            EventStage::ProjectLoad,
            EngineEvent::Asset(AssetEvent::PackageLoading {
                package: PackageId::new(package.package_id.clone()),
                path: manifest_path.display().to_string(),
            }),
        );
        let records = assets
            .load_package_manifest_with_expected_id(&manifest_path, &package.package_id)
            .map_err(|err| {
                let message = format!(
                    "failed to load package '{}' from '{}': {err}",
                    package.package_id,
                    manifest_path.display()
                );
                emit_runtime_package_failed(runtime_events, package, message.clone());
                message
            })?;
        runtime_events.emit(
            EventStage::ProjectLoad,
            EngineEvent::Asset(AssetEvent::PackageLoaded {
                package: PackageId::new(package.package_id.clone()),
                path: manifest_path.display().to_string(),
            }),
        );
        all_records.extend(records);
    }
    Ok(all_records)
}

fn emit_runtime_project_loaded(
    runtime_events: &mut RuntimeEvents,
    runtime_project: &RuntimeProject,
) {
    runtime_events.emit(
        EventStage::ProjectLoad,
        EngineEvent::Lifecycle(LifecycleEvent::ProjectLoaded {
            project: ProjectId::new(runtime_project.project.project_id.clone()),
            path: runtime_project.project_path.display().to_string(),
        }),
    );
}

fn emit_runtime_scene_loading(
    runtime_events: &mut RuntimeEvents,
    runtime_project: &RuntimeProject,
) {
    runtime_events.emit(
        EventStage::SceneLoad,
        EngineEvent::Lifecycle(LifecycleEvent::SceneLoading {
            scene: SceneId::new(runtime_scene_id(runtime_project)),
            path: runtime_project.startup_scene_path.display().to_string(),
        }),
    );
}

fn emit_runtime_scene_loaded(runtime_events: &mut RuntimeEvents, runtime_project: &RuntimeProject) {
    runtime_events.emit(
        EventStage::SceneLoad,
        EngineEvent::Lifecycle(LifecycleEvent::SceneLoaded {
            scene: SceneId::new(runtime_scene_id(runtime_project)),
            path: runtime_project.startup_scene_path.display().to_string(),
        }),
    );
}

fn emit_runtime_shutdown_requested(runtime_events: &mut RuntimeEvents, reason: impl Into<String>) {
    runtime_events.emit(
        EventStage::Shutdown,
        EngineEvent::Lifecycle(LifecycleEvent::ShutdownRequested {
            reason: reason.into(),
        }),
    );
}

fn emit_runtime_package_failed(
    runtime_events: &mut RuntimeEvents,
    package: &ProjectPackage,
    message: impl Into<String>,
) {
    runtime_events.emit(
        EventStage::ProjectLoad,
        EngineEvent::Asset(AssetEvent::PackageFailed {
            package: PackageId::new(package.package_id.clone()),
            message: message.into(),
        }),
    );
}

fn runtime_scene_id(runtime_project: &RuntimeProject) -> String {
    runtime_project
        .project
        .startup_scene
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "startup".to_string())
}

fn validate_startup_scene(
    scene_path: &Path,
    package_records: &[DurableAssetRecord],
) -> Result<(), String> {
    let known_asset_ids = package_records
        .iter()
        .map(|record| record.asset_id.clone())
        .collect::<Vec<_>>();
    renderer::validate_scene_file_with_options(
        scene_path,
        &SceneValidationOptions::default().with_known_asset_ids(known_asset_ids),
    )
    .map_err(|err| {
        format!(
            "startup scene validation failed for '{}': {err}",
            scene_path.display()
        )
    })
}

fn load_startup_scene(renderer: &mut Renderer, scene_path: &Path) -> Result<Scene, String> {
    let mut assets = renderer.assets();
    Scene::load(scene_path, &mut assets).map_err(|err| {
        format!(
            "failed to load startup scene '{}': {err}",
            scene_path.display()
        )
    })
}

fn apply_debug_record_launch_options(
    renderer: &mut Renderer,
    options: &LaunchOptions,
) -> Result<(), renderer::RendererError> {
    if options.record_debug_secs.is_none()
        && options.record_debug_interval_ms.is_none()
        && options.record_debug_path.is_none()
    {
        return Ok(());
    }

    renderer.configure_debug_timing_recording(
        options.record_debug_secs,
        options.record_debug_interval_ms,
        options
            .record_debug_path
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned()),
    )?;

    if options.record_debug_secs.is_some() {
        let path = renderer.start_debug_timing_recording()?;
        info!("Debug timing recording active -> {path}");
    } else {
        warn!(
            "Debug timing recording configured but not started because --record_debug was absent"
        );
    }

    Ok(())
}

fn apply_frame_capture_launch_options(
    renderer: &mut Renderer,
    options: &LaunchOptions,
    capture_run_dir: &Path,
) -> Result<(), renderer::RendererError> {
    renderer.configure_manual_frame_capture_dir(
        options
            .manual_capture_dir
            .clone()
            .or_else(|| Some(capture_run_dir.to_path_buf())),
    )?;

    if let Some(frame_number) = options.capture_frame {
        let output_path = options.capture_frame_path.clone().unwrap_or_else(|| {
            single_capture_path(
                capture_run_dir,
                FALLBACK_APP_NAME,
                frame_number,
                options.capture_target,
            )
        });
        renderer.request_frame_capture_at(
            frame_number,
            FrameCaptureRequest::new(options.capture_target, output_path),
        )?;
    }

    if let Some(count) = options.capture_frames {
        let output_dir = options
            .capture_dir
            .clone()
            .unwrap_or_else(|| capture_run_dir.to_path_buf());
        let sequence = FrameCaptureSequence::new(
            options.capture_target,
            output_dir,
            options.capture_frame_start.unwrap_or(0),
            options.capture_frame_interval.unwrap_or(1),
            count,
        )?;
        renderer.configure_frame_capture_sequence(sequence)?;
    }

    Ok(())
}

fn headless_capture_plan(options: &LaunchOptions) -> HeadlessCapturePlan {
    let expected_captures = options
        .capture_frames
        .or_else(|| options.capture_frame.map(|_| 1))
        .unwrap_or(0) as usize;

    if expected_captures == 0 {
        return HeadlessCapturePlan {
            expected_captures,
            frame_budget: DEFAULT_HEADLESS_SMOKE_FRAMES,
        };
    }

    let last_requested_frame = if let Some(frame) = options.capture_frame {
        frame
    } else {
        let count = options.capture_frames.unwrap_or(1);
        let start = options.capture_frame_start.unwrap_or(0);
        let interval = options.capture_frame_interval.unwrap_or(1);
        start.saturating_add(interval.saturating_mul(count.saturating_sub(1)))
    };

    HeadlessCapturePlan {
        expected_captures,
        frame_budget: last_requested_frame.saturating_add(120).max(180),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use renderer::CaptureTarget;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn options(project_path: impl Into<PathBuf>) -> LaunchOptions {
        LaunchOptions {
            project_path: project_path.into(),
            scene_path: None,
            record_debug_secs: None,
            record_debug_interval_ms: None,
            record_debug_path: None,
            capture_frame: None,
            capture_frame_path: None,
            capture_frames: None,
            capture_frame_start: None,
            capture_frame_interval: None,
            capture_dir: None,
            capture_target: CaptureTarget::Present,
            headless: true,
            manual_capture_dir: None,
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("engine-runtime-{label}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_project_fixture(root: &Path) -> PathBuf {
        fs::create_dir_all(root.join("assets/models")).unwrap();
        fs::create_dir_all(root.join("scenes")).unwrap();
        fs::write(root.join("assets/models/block.obj"), b"placeholder").unwrap();
        fs::write(
            root.join("assets/sample.package.toml"),
            r#"
format_version = 1
package_id = "sample"
display_name = "Sample"

[[assets]]
id = "sample.model.block"
kind = "model"
path = "models/block.obj"
"#,
        )
        .unwrap();
        fs::write(
            root.join("scenes/start.engine.scene.json"),
            r#"{
  "format_version": 1,
  "scene_id": "scene.sample.start",
  "root_nodes": [],
  "nodes": [],
  "lights": [],
  "environment": null,
  "materials": {},
  "editor": {}
}"#,
        )
        .unwrap();
        fs::write(
            root.join("engine.project.toml"),
            r#"
format_version = 1
project_id = "project.sample"
name = "Sample Runtime"
asset_root = "assets"
startup_scene = "scenes/start.engine.scene.json"

[[packages]]
package_id = "sample"
manifest = "assets/sample.package.toml"
enabled = true

[settings]
window_width = 800
window_height = 600
fullscreen = false
vsync = true
"#,
        )
        .unwrap();
        root.join("engine.project.toml")
    }

    #[test]
    fn resolves_relative_project_path_against_cwd() {
        let resolved =
            resolve_project_path(Path::new("apps/editor/sample_project/engine.project.toml"))
                .expect("relative path should resolve");
        assert!(resolved.is_absolute());
        assert!(resolved.ends_with("apps/editor/sample_project/engine.project.toml"));
    }

    #[test]
    fn loads_project_root_packages_and_startup_scene_path_without_vulkan() {
        let dir = temp_dir("valid-project");
        let project_path = write_project_fixture(&dir);
        let loaded = load_runtime_project(&options(&project_path)).expect("project should load");

        assert_eq!(loaded.project_root, dir);
        assert_eq!(loaded.project.name, "Sample Runtime");
        assert_eq!(loaded.enabled_package_count, 1);
        assert_eq!(
            loaded.startup_scene_path,
            loaded.project_root.join("scenes/start.engine.scene.json")
        );
    }

    #[test]
    fn missing_project_is_a_controlled_runtime_error() {
        let dir = temp_dir("missing-project");
        let err = load_runtime_project(&options(dir.join("engine.project.toml")))
            .expect_err("missing project should fail");

        assert!(err.contains("project file"));
        assert!(err.contains("does not exist"));
    }

    #[test]
    fn missing_startup_scene_is_a_controlled_runtime_error() {
        let dir = temp_dir("missing-scene");
        let project_path = write_project_fixture(&dir);
        fs::remove_file(dir.join("scenes/start.engine.scene.json")).unwrap();

        let err = load_runtime_project(&options(project_path)).expect_err("missing scene fails");

        assert!(err.contains("project validation failed"));
        assert!(err.contains("missing_startup_scene"));
    }

    #[test]
    fn package_id_mismatch_is_a_controlled_runtime_error() {
        let dir = temp_dir("package-mismatch");
        let project_path = write_project_fixture(&dir);
        let manifest = fs::read_to_string(dir.join("assets/sample.package.toml")).unwrap();
        fs::write(
            dir.join("assets/sample.package.toml"),
            manifest.replace("package_id = \"sample\"", "package_id = \"wrong\""),
        )
        .unwrap();

        let err = load_runtime_project(&options(project_path)).expect_err("package id fails");

        assert!(err.contains("project validation failed"));
        assert!(err.contains("package.id_mismatch"));
    }

    #[test]
    fn runtime_lifecycle_helpers_record_project_and_scene_order_without_vulkan() {
        let dir = temp_dir("lifecycle-order");
        let project_path = write_project_fixture(&dir);
        let loaded = load_runtime_project(&options(&project_path)).expect("project should load");
        let mut runtime_events = RuntimeEvents::new();

        runtime_events.emit(
            EventStage::Startup,
            EngineEvent::Lifecycle(LifecycleEvent::AppStarting {
                app_name: FALLBACK_APP_NAME.to_string(),
            }),
        );
        runtime_events.emit(
            EventStage::ProjectLoad,
            EngineEvent::Lifecycle(LifecycleEvent::ProjectLoading {
                path: project_path.display().to_string(),
            }),
        );
        emit_runtime_project_loaded(&mut runtime_events, &loaded);
        emit_runtime_scene_loading(&mut runtime_events, &loaded);
        emit_runtime_scene_loaded(&mut runtime_events, &loaded);

        let labels = runtime_events
            .recorded()
            .into_iter()
            .map(|event| match event.event {
                EngineEvent::Lifecycle(LifecycleEvent::AppStarting { .. }) => "app_starting",
                EngineEvent::Lifecycle(LifecycleEvent::ProjectLoading { .. }) => "project_loading",
                EngineEvent::Lifecycle(LifecycleEvent::ProjectLoaded { .. }) => "project_loaded",
                EngineEvent::Lifecycle(LifecycleEvent::SceneLoading { .. }) => "scene_loading",
                EngineEvent::Lifecycle(LifecycleEvent::SceneLoaded { .. }) => "scene_loaded",
                _ => "other",
            })
            .collect::<Vec<_>>();

        assert_eq!(
            labels,
            [
                "app_starting",
                "project_loading",
                "project_loaded",
                "scene_loading",
                "scene_loaded"
            ]
        );
    }

    #[test]
    fn runtime_shutdown_requested_is_recorded_without_vulkan() {
        let mut runtime_events = RuntimeEvents::new();

        emit_runtime_shutdown_requested(&mut runtime_events, "window close requested");

        let recorded = runtime_events.recorded();
        assert_eq!(recorded.len(), 1);
        match &recorded[0].event {
            EngineEvent::Lifecycle(LifecycleEvent::ShutdownRequested { reason }) => {
                assert_eq!(reason, "window close requested");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn runtime_package_failure_is_recorded_without_vulkan() {
        let mut runtime_events = RuntimeEvents::new();
        let package = ProjectPackage {
            package_id: "sample".to_string(),
            manifest: PathBuf::from("assets/sample.package.toml"),
            enabled: true,
        };

        emit_runtime_package_failed(
            &mut runtime_events,
            &package,
            "failed to load package 'sample'",
        );

        let recorded = runtime_events.recorded();
        assert_eq!(recorded.len(), 1);
        match &recorded[0].event {
            EngineEvent::Asset(AssetEvent::PackageFailed { package, message }) => {
                assert_eq!(package.as_str(), "sample");
                assert_eq!(message, "failed to load package 'sample'");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn headless_budget_covers_last_sequence_frame() {
        let mut launch = options("engine.project.toml");
        launch.capture_frames = Some(3);
        launch.capture_frame_start = Some(5);
        launch.capture_frame_interval = Some(5);

        let plan = headless_capture_plan(&launch);

        assert_eq!(plan.expected_captures, 3);
        assert!(plan.frame_budget >= 15);
        assert_eq!(plan.frame_budget, 180);
    }

    #[test]
    fn headless_budget_covers_single_capture_frame() {
        let mut launch = options("engine.project.toml");
        launch.capture_frame = Some(250);

        let plan = headless_capture_plan(&launch);

        assert_eq!(plan.expected_captures, 1);
        assert_eq!(plan.frame_budget, 370);
    }
}
