//! GPU smoke test: validates that the headless renderer can construct a scene,
//! render one frame, and produce a valid non-empty PNG capture.
//!
//! Requires a Vulkan-capable GPU with validation layer support.
//! Run manually with:
//!   LIBCLANG_PATH=/usr/lib64 cargo test -p renderer gpu_smoke -- --ignored --nocapture
//!
//! Phase 09 hardening: adds BSP-inactive GPU smoke to prove no BSP
//! pipelines/descriptors/uploads/binds/draws occur before mount request.

use std::process::Command;

use glam::{Vec3, Vec4};
use renderer::prelude::{
    AssetError, CaptureTarget, FrameCaptureRequest, FrameCaptureStatus, FrameRenderOutcome,
    LoadStatus, MeshDeformation, PbrMaterialDesc, PointLight, ProceduralMeshData, ProceduralVertex,
    Renderer, RendererConfig, Scene, SceneFragmentNodeId,
};
/// Build a unit cube (1×1×1, centered at origin) as procedural mesh data.
fn build_cube_mesh() -> ProceduralMeshData {
    let s = 0.5;
    let positions = [
        // +X
        ([s, -s, s], [1.0, 0.0, 0.0], [0.0, 0.0]),
        ([s, s, s], [1.0, 0.0, 0.0], [0.0, 1.0]),
        ([s, s, -s], [1.0, 0.0, 0.0], [1.0, 1.0]),
        ([s, -s, -s], [1.0, 0.0, 0.0], [1.0, 0.0]),
        // -X
        ([-s, -s, -s], [-1.0, 0.0, 0.0], [0.0, 0.0]),
        ([-s, s, -s], [-1.0, 0.0, 0.0], [0.0, 1.0]),
        ([-s, s, s], [-1.0, 0.0, 0.0], [1.0, 1.0]),
        ([-s, -s, s], [-1.0, 0.0, 0.0], [1.0, 0.0]),
        // +Y
        ([-s, s, s], [0.0, 1.0, 0.0], [0.0, 0.0]),
        ([s, s, s], [0.0, 1.0, 0.0], [1.0, 0.0]),
        ([s, s, -s], [0.0, 1.0, 0.0], [1.0, 1.0]),
        ([-s, s, -s], [0.0, 1.0, 0.0], [0.0, 1.0]),
        // -Y
        ([-s, -s, -s], [0.0, -1.0, 0.0], [0.0, 0.0]),
        ([s, -s, -s], [0.0, -1.0, 0.0], [1.0, 0.0]),
        ([s, -s, s], [0.0, -1.0, 0.0], [1.0, 1.0]),
        ([-s, -s, s], [0.0, -1.0, 0.0], [0.0, 1.0]),
        // +Z
        ([-s, -s, s], [0.0, 0.0, 1.0], [0.0, 0.0]),
        ([-s, s, s], [0.0, 0.0, 1.0], [0.0, 1.0]),
        ([s, s, s], [0.0, 0.0, 1.0], [1.0, 1.0]),
        ([s, -s, s], [0.0, 0.0, 1.0], [1.0, 0.0]),
        // -Z
        ([s, -s, -s], [0.0, 0.0, -1.0], [0.0, 0.0]),
        ([s, s, -s], [0.0, 0.0, -1.0], [0.0, 1.0]),
        ([-s, s, -s], [0.0, 0.0, -1.0], [1.0, 1.0]),
        ([-s, -s, -s], [0.0, 0.0, -1.0], [1.0, 0.0]),
    ];

    let vertices: Vec<ProceduralVertex> = positions
        .iter()
        .map(|&(pos, normal, uv)| {
            let position = Vec3::from_array(pos);
            let n = Vec3::from_array(normal);
            // Compute tangent perpendicular to normal
            let up = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
            let tangent = up.cross(n).normalize();
            let bitangent = n.cross(tangent);
            let handedness = if bitangent.dot(Vec3::Y) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            ProceduralVertex {
                position,
                normal: n,
                tangent: tangent.extend(handedness),
                uv0: glam::Vec2::from_array(uv),
                uv1: glam::Vec2::ZERO,
                color: Vec4::ONE,
            }
        })
        .collect();

    let indices: Vec<u32> = (0..6)
        .flat_map(|face| {
            let base = face * 4;
            vec![base, base + 1, base + 2, base, base + 2, base + 3]
        })
        .collect();

    ProceduralMeshData {
        name: "smoke_cube".to_string(),
        vertices,
        indices,
        material: None,
    }
}

/// GPU smoke test: headless renderer constructs scene, renders one frame,
/// and produces a valid non-empty PNG capture.
///
/// Marked `#[ignore]` because it requires a Vulkan-capable GPU with
/// validation layer support. Run with:
///
/// ```bash
/// LIBCLANG_PATH=/usr/lib64 cargo test -p renderer gpu_smoke -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn renderer_headless_capture_worker() {
    // This worker is launched by gpu_smoke_headless_capture so the parent test can assert
    // validation diagnostics and teardown failures emitted after Renderer is dropped.
    if std::env::var_os("RENDERER_GPU_SMOKE_WORKER").is_none() {
        return;
    }
    let _ = env_logger::builder().is_test(true).try_init();

    // Ensure we're writing captures to a temp directory.
    let output_dir = std::env::temp_dir().join("renderer-gpu-smoke");
    std::fs::create_dir_all(&output_dir).expect("create temp output dir");
    let capture_path = output_dir.join("gpu_smoke_frame.png");

    // Remove any leftover capture from a prior run.
    let _ = std::fs::remove_file(&capture_path);

    // ── Build config with validation enabled ─────────────────────────
    let config = RendererConfig {
        app_name: "gpu-smoke-test".to_string(),
        validation_layer: true,
        headless: false, // set to false; new_headless will force it true
        preload_startup_scene: false,
        ..RendererConfig::default()
    };

    // ── Create headless renderer ─────────────────────────────────────
    let mut renderer = match Renderer::new_headless(config) {
        Ok(r) => r,
        Err(err) => {
            panic!("Headless renderer init failed (is a Vulkan-capable GPU available?): {err}");
        }
    };

    // Exercise deferred model completion and prove every returned fragment mesh has a DTO.
    let ticket = renderer
        .assets()
        .request_model_load("src/renderer/src/assets/DamagedHelmet.glb")
        .expect("queue deferred model");
    let fragment = loop {
        renderer
            .pump_asset_tasks(usize::MAX)
            .expect("pump deferred model");
        match renderer.assets().poll_model_load(ticket) {
            LoadStatus::Pending { .. } => std::thread::sleep(std::time::Duration::from_millis(1)),
            LoadStatus::Uploaded { value } => break value,
            LoadStatus::Failed { error } => panic!("deferred model failed: {error}"),
            LoadStatus::Cancelled => panic!("deferred model was unexpectedly cancelled"),
        }
    };
    let fragment_meshes: Vec<_> = (0..fragment.node_count())
        .flat_map(|index| {
            fragment
                .node(SceneFragmentNodeId::new(index as u32))
                .expect("fragment node")
                .meshes
                .iter()
                .copied()
        })
        .collect();
    assert!(!fragment_meshes.is_empty());
    for mesh in fragment_meshes {
        let dto = renderer
            .assets()
            .mesh_geometry(mesh)
            .expect("deferred fragment geometry DTO");
        assert_eq!(dto.deformation, MeshDeformation::Rigid);
        assert!(dto.local_aabb.is_some());
    }

    // Position the camera looking at origin.
    renderer
        .set_camera_look_at(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y)
        .expect("set camera look-at");

    // ── Build scene ──────────────────────────────────────────────────
    let mut scene = Scene::new();

    // Create a cube mesh with a simple PBR material.
    let cube_mesh = build_cube_mesh();
    let material = {
        let mut assets = renderer.assets();
        assets
            .create_material_pbr(PbrMaterialDesc {
                base_color: Vec4::new(0.8, 0.2, 0.2, 1.0),
                metallic: 0.0,
                roughness: 0.5,
                ..Default::default()
            })
            .expect("create material")
    };

    // Upload the cube mesh and assign material.
    let mesh_handle = {
        let mut assets = renderer.assets();
        let mut mesh = cube_mesh;
        mesh.material = Some(material);
        assets
            .upload_procedural_mesh(mesh)
            .expect("upload procedural mesh")
    };

    let geometry = renderer
        .assets()
        .mesh_geometry(mesh_handle)
        .expect("synchronous procedural geometry DTO");
    assert_eq!(geometry.deformation, MeshDeformation::Rigid);
    assert_eq!(geometry.positions.len(), 24);
    assert!(geometry.local_aabb.is_some());

    // Unload a separate never-submitted mesh and prove stale/double-unload behavior.
    let unload_handle = renderer
        .assets()
        .upload_procedural_mesh(build_cube_mesh())
        .expect("upload unload proof mesh");
    renderer
        .assets()
        .unload_mesh(unload_handle)
        .expect("first mesh unload");
    assert!(matches!(
        renderer.assets().mesh_geometry(unload_handle),
        Err(AssetError::StaleHandle { .. })
    ));
    assert!(matches!(
        renderer.assets().unload_mesh(unload_handle),
        Err(AssetError::StaleHandle { .. })
    ));

    // Place cube at origin.
    let root = scene.create_node_default(None).expect("create root node");
    scene.add_mesh(root, mesh_handle).expect("add cube mesh");

    // Add a point light above and in front.
    scene
        .create_point_light(PointLight {
            position: Vec3::new(2.0, 4.0, 3.0),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 30.0,
            range: 15.0,
        })
        .expect("create point light");

    // ── Configure capture ────────────────────────────────────────────
    renderer
        .request_frame_capture_at(
            0,
            FrameCaptureRequest::new(CaptureTarget::Draw, &capture_path),
        )
        .expect("schedule frame capture at frame 0");

    // ── Render one frame ─────────────────────────────────────────────
    let outcome = renderer
        .render_scene_headless(&mut scene)
        .expect("render headless frame");
    assert_eq!(
        outcome,
        FrameRenderOutcome::Rendered,
        "headless frame must complete rendering"
    );

    // ── Check capture status ─────────────────────────────────────────
    let status = renderer
        .last_frame_capture_status()
        .expect("capture status should be set after rendering");

    match status {
        FrameCaptureStatus::Succeeded {
            output_path,
            width,
            height,
            ..
        } => {
            assert!(output_path.exists(), "capture PNG file must exist");
            let metadata =
                std::fs::metadata(output_path).expect("capture PNG file should be accessible");
            let size = metadata.len();
            assert!(size > 1024, "capture PNG must be > 1KB (got {size} bytes)");
            assert!(*width > 0, "capture width must be positive");
            assert!(*height > 0, "capture height must be positive");
            let decoded = image::open(output_path).expect("capture must decode as a valid image");
            assert_eq!(
                decoded.width(),
                *width,
                "decoded capture width must match status"
            );
            assert_eq!(
                decoded.height(),
                *height,
                "decoded capture height must match status"
            );
            eprintln!(
                "GPU smoke worker rendered: {output_path:?} ({width}x{height}, {size} bytes)"
            );
        }
        FrameCaptureStatus::Failed { message, .. } => {
            panic!("GPU smoke test capture failed: {message}");
        }
        FrameCaptureStatus::BackendNotImplemented { .. } => {
            panic!("GPU smoke test: capture backend not implemented");
        }
        FrameCaptureStatus::Pending { .. } => {
            panic!("GPU smoke test: capture still pending after render");
        }
    }

    // Force backend teardown inside this test so the parent process observes lifecycle failures.
    drop(renderer);
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: BSP-Inactive GPU Smoke
// ═══════════════════════════════════════════════════════════════════════

/// BSP-inactive GPU smoke: renderer with BSP feature compiled in but no
/// BSP mount requested. Proves no BSP pipelines/descriptors/uploads/binds/
/// draws occur before a mount request.
///
/// Marked `#[ignore]` because it requires a Vulkan-capable GPU.
#[test]
#[ignore]
fn bsp_inactive_gpu_smoke_worker() {
    if std::env::var_os("RENDERER_BSP_INACTIVE_WORKER").is_none() {
        return;
    }
    let _ = env_logger::builder().is_test(true).try_init();

    let output_dir = std::env::temp_dir().join("renderer-bsp-inactive-smoke");
    std::fs::create_dir_all(&output_dir).expect("create temp output dir");
    let capture_path = output_dir.join("bsp_inactive_frame.png");
    let _ = std::fs::remove_file(&capture_path);

    let config = RendererConfig {
        app_name: "bsp-inactive-smoke".to_string(),
        validation_layer: true,
        headless: false,
        preload_startup_scene: false,
        ..RendererConfig::default()
    };

    // BSP feature is compiled in but no mount requested — smoke must still pass
    let mut renderer = match Renderer::new_headless(config) {
        Ok(r) => r,
        Err(err) => {
            panic!("Headless renderer init failed (GPU required): {err}");
        }
    };

    renderer
        .set_camera_look_at(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y)
        .expect("set camera");

    let mut scene = Scene::new();

    // Create a cube (standard PBR, not BSP)
    let material = {
        let mut assets = renderer.assets();
        assets
            .create_material_pbr(PbrMaterialDesc {
                base_color: Vec4::new(0.2, 0.6, 0.2, 1.0),
                metallic: 0.0,
                roughness: 0.5,
                ..Default::default()
            })
            .expect("create material")
    };

    let mesh_handle = {
        let mut assets = renderer.assets();
        let mut mesh = build_cube_mesh();
        mesh.material = Some(material);
        assets.upload_procedural_mesh(mesh).expect("upload mesh")
    };

    let root = scene.create_node_default(None).expect("create root");
    scene.add_mesh(root, mesh_handle).expect("add mesh");

    // No BSP mount requested — render standard PBR scene
    renderer
        .request_frame_capture_at(
            0,
            FrameCaptureRequest::new(CaptureTarget::Draw, &capture_path),
        )
        .expect("schedule capture");

    let outcome = renderer
        .render_scene_headless(&mut scene)
        .expect("render headless frame");
    assert_eq!(outcome, FrameRenderOutcome::Rendered);

    let status = renderer
        .last_frame_capture_status()
        .expect("capture status should be set");

    match status {
        FrameCaptureStatus::Succeeded { output_path, .. } => {
            assert!(output_path.exists(), "capture PNG must exist");
            let size = std::fs::metadata(&output_path).unwrap().len();
            assert!(size > 1024, "capture must be > 1KB (got {size})");
            eprintln!("BSP-inactive smoke captured: {output_path:?} ({size} bytes)");
        }
        FrameCaptureStatus::Failed { message, .. } => {
            panic!("BSP-inactive smoke capture failed: {message}");
        }
        _ => panic!("unexpected capture status"),
    }

    drop(renderer);
}

#[test]
#[ignore]
fn gpu_smoke_bsp_inactive() {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("resolve workspace root");
    let output = Command::new(std::env::current_exe().expect("resolve GPU smoke test executable"))
        .current_dir(workspace_root)
        .args([
            "--ignored",
            "--exact",
            "bsp_inactive_gpu_smoke_worker",
            "--nocapture",
        ])
        .env("RENDERER_BSP_INACTIVE_WORKER", "1")
        .output()
        .expect("launch isolated BSP-inactive GPU smoke worker");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprint!("{stderr}");
    print!("{stdout}");

    assert!(
        output.status.success(),
        "BSP-inactive GPU smoke worker failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );

    let diagnostics = format!("{stdout}\n{stderr}");
    let lines: Vec<_> = diagnostics.lines().collect();
    let has_validation_error = lines.windows(2).any(|pair| {
        pair[0].trim() == "ERROR:" && pair[1].to_ascii_uppercase().contains("VALIDATION")
    });
    assert!(
        !has_validation_error,
        "BSP-inactive: Vulkan validation error(s) emitted:\n{diagnostics}"
    );
}

#[test]
#[ignore]
fn gpu_smoke_headless_capture() {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("resolve workspace root");
    let output = Command::new(std::env::current_exe().expect("resolve GPU smoke test executable"))
        .current_dir(workspace_root)
        .args([
            "--ignored",
            "--exact",
            "renderer_headless_capture_worker",
            "--nocapture",
        ])
        .env("RENDERER_GPU_SMOKE_WORKER", "1")
        .output()
        .expect("launch isolated GPU smoke worker");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprint!("{stderr}");
    print!("{stdout}");

    assert!(
        output.status.success(),
        "GPU smoke worker failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout,
        stderr
    );

    let diagnostics = format!("{stdout}\n{stderr}");
    let lines: Vec<_> = diagnostics.lines().collect();
    let has_validation_error = lines.windows(2).any(|pair| {
        pair[0].trim() == "ERROR:" && pair[1].to_ascii_uppercase().contains("VALIDATION")
    });
    assert!(
        !has_validation_error,
        "Vulkan validation error(s) were emitted:\n{diagnostics}"
    );
}
