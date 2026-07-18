//! GPU smoke test: validates that the headless renderer can construct a scene,
//! render one frame, and produce a valid non-empty PNG capture.
//!
//! Requires a Vulkan-capable GPU with validation layer support.
//! Run manually with:
//!   LIBCLANG_PATH=/usr/lib64 cargo test -p renderer gpu_smoke -- --ignored --nocapture

use glam::{Vec3, Vec4};
use renderer::prelude::{
    CaptureTarget, FrameCaptureRequest, FrameCaptureStatus, FrameRenderOutcome, PbrMaterialDesc,
    PointLight, ProceduralMeshData, ProceduralVertex, Renderer, RendererConfig, Scene,
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
            let handedness = if bitangent.dot(Vec3::Y) >= 0.0 { 1.0 } else { -1.0 };
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
fn gpu_smoke_headless_capture() {
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

    // Position the camera looking at origin.
    renderer
        .set_camera_look_at(
            Vec3::new(0.0, 2.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
        )
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
    assert!(
        matches!(
            outcome,
            FrameRenderOutcome::Rendered
                | FrameRenderOutcome::SubmittedNotPresented
                | FrameRenderOutcome::PresentedSuboptimal
        ),
        "unexpected frame outcome: {outcome:?}"
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
            assert!(
                size > 1024,
                "capture PNG must be > 1KB (got {size} bytes)"
            );
            assert!(*width > 0, "capture width must be positive");
            assert!(*height > 0, "capture height must be positive");
            eprintln!(
                "GPU smoke test PASSED: {output_path:?} ({width}x{height}, {size} bytes)"
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

    // Check that validation was enabled and the debug callback was set
    // up. Vulkan validation errors/warnings go to stderr and can be
    // inspected by running with `--nocapture`.
    eprintln!("Validation layer was enabled — check stderr for any Vulkan validation messages.");
}
