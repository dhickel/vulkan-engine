//! Shared infrastructure for headless capture validation test scenes.
//!
//! Provides mesh builders (cube, sphere, plane), a minimal arg parser,
//! and a headless capture runner that eliminates boilerplate across test binaries.
//!
//! Each capture test binary should:
//! 1. Call `parse_capture_test_args()`
//! 2. Define a scene builder closure
//! 3. Call `run_headless_capture_test()`

use glam::{Mat4, Vec2, Vec3, Vec4};
use log::{error, info};
use renderer::prelude::{
    default_capture_run_dir, CaptureTarget, FrameCaptureRequest, FrameCaptureSequence,
    FrameCaptureStatus, FrameRenderOutcome, ProceduralMeshData, ProceduralVertex, Renderer,
    RendererConfig, Scene,
};
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

// ── Arg parsing ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CaptureTestArgs {
    pub headless: bool,
    pub capture_target: CaptureTarget,
    pub capture_frames: Option<u32>,
    pub capture_frame_start: Option<u32>,
    pub capture_frame_interval: Option<u32>,
    pub capture_dir: Option<PathBuf>,
    pub env_path: Option<PathBuf>,
}

impl Default for CaptureTestArgs {
    fn default() -> Self {
        Self {
            headless: false,
            capture_target: CaptureTarget::Draw,
            capture_frames: None,
            capture_frame_start: None,
            capture_frame_interval: None,
            capture_dir: None,
            env_path: None,
        }
    }
}

pub fn parse_capture_test_args() -> CaptureTestArgs {
    parse_capture_test_args_from(env::args().skip(1))
}

pub fn parse_capture_test_args_from(
    args: impl IntoIterator<Item = impl Into<String>>,
) -> CaptureTestArgs {
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut opts = CaptureTestArgs::default();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "--headless" => {
                opts.headless = true;
                i += 1;
            }
            "--capture_target" => {
                if let Some(v) = args.get(i + 1) {
                    opts.capture_target = parse_target(v);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--capture_frames" => {
                if let Some(v) = args.get(i + 1) {
                    opts.capture_frames = v.parse::<u32>().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--capture_frame_start" => {
                if let Some(v) = args.get(i + 1) {
                    opts.capture_frame_start = v.parse::<u32>().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--capture_frame_interval" => {
                if let Some(v) = args.get(i + 1) {
                    opts.capture_frame_interval = v.parse::<u32>().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--capture_dir" => {
                if let Some(v) = args.get(i + 1) {
                    opts.capture_dir = Some(PathBuf::from(v));
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--env" => {
                if let Some(v) = args.get(i + 1) {
                    opts.env_path = Some(PathBuf::from(v));
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ if arg.starts_with("--capture_target=") => {
                opts.capture_target = parse_target(&arg["--capture_target=".len()..]);
                i += 1;
            }
            _ if arg.starts_with("--capture_frames=") => {
                opts.capture_frames = arg["--capture_frames=".len()..].parse::<u32>().ok();
                i += 1;
            }
            _ if arg.starts_with("--capture_frame_start=") => {
                opts.capture_frame_start =
                    arg["--capture_frame_start=".len()..].parse::<u32>().ok();
                i += 1;
            }
            _ if arg.starts_with("--capture_frame_interval=") => {
                opts.capture_frame_interval =
                    arg["--capture_frame_interval=".len()..].parse::<u32>().ok();
                i += 1;
            }
            _ if arg.starts_with("--capture_dir=") => {
                opts.capture_dir = Some(PathBuf::from(&arg["--capture_dir=".len()..]));
                i += 1;
            }
            _ if arg.starts_with("--env=") => {
                opts.env_path = Some(PathBuf::from(&arg["--env=".len()..]));
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    opts
}

fn parse_target(s: &str) -> CaptureTarget {
    match s {
        "draw" | "Draw" => CaptureTarget::Draw,
        "present" | "Present" => CaptureTarget::Present,
        _ => CaptureTarget::Draw,
    }
}

// ── Headless runner ─────────────────────────────────────────────────────────

/// Run a scene builder in headless capture mode.
///
/// `app_name` is used for capture directory naming.
/// `build_scene` receives `&mut Renderer` and returns a `Scene`.
pub fn run_headless_capture_test(
    app_name: &str,
    args: &CaptureTestArgs,
    build_scene: impl FnOnce(&mut Renderer) -> Scene,
) {
    init_logging();

    let validation_layer = std::env::var("ENGINE_CAPTURE_VALIDATION")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "on"));
    let config = RendererConfig {
        app_name: app_name.to_string(),
        headless: true,
        validation_layer,
        ..RendererConfig::default()
    };

    let capture_run_dir = default_capture_run_dir(app_name);

    let mut renderer = match Renderer::new_headless(config.clone()) {
        Ok(r) => r,
        Err(err) => {
            error!("Headless renderer init failed: {err}");
            return;
        }
    };

    // Configure capture
    let output_dir = args
        .capture_dir
        .clone()
        .unwrap_or_else(|| capture_run_dir.to_path_buf());

    if let Some(count) = args.capture_frames {
        let start = args.capture_frame_start.unwrap_or(5);
        let interval = args.capture_frame_interval.unwrap_or(5);
        match FrameCaptureSequence::new(args.capture_target, output_dir, start, interval, count) {
            Ok(seq) => {
                if let Err(err) = renderer.configure_frame_capture_sequence(seq) {
                    error!("Failed to configure capture sequence: {err}");
                    return;
                }
            }
            Err(err) => {
                error!("Invalid capture sequence: {err}");
                return;
            }
        }
    } else {
        // Default: single capture at frame 5
        let output_path = output_dir.join(format!("{}_frame_5.png", app_name));
        let req = FrameCaptureRequest::new(args.capture_target, output_path);
        if let Err(err) = renderer.request_frame_capture_at(5, req) {
            error!("Failed to schedule frame capture: {err}");
            return;
        }
    }

    let expected_captures = args.capture_frames.unwrap_or(1) as usize;
    let frame_budget = args
        .capture_frame_start
        .unwrap_or(5)
        .saturating_add(
            args.capture_frame_interval
                .unwrap_or(5)
                .saturating_mul(args.capture_frames.unwrap_or(1).saturating_add(2)),
        )
        .max(180);

    let mut scene = build_scene(&mut renderer);
    let mut succeeded_paths = HashSet::new();

    for _ in 0..frame_budget {
        match renderer.render_scene_headless(&mut scene) {
            Ok(FrameRenderOutcome::Rendered)
            | Ok(FrameRenderOutcome::SkippedResizePending)
            | Ok(FrameRenderOutcome::SubmittedNotPresented)
            | Ok(FrameRenderOutcome::PresentedSuboptimal) => {}
            Err(err) => {
                error!("Headless render failed: {err}");
                return;
            }
        }

        match renderer.last_frame_capture_status() {
            Some(FrameCaptureStatus::Succeeded { output_path, .. }) => {
                succeeded_paths.insert(output_path.clone());
                if succeeded_paths.len() >= expected_captures {
                    info!(
                        "Headless capture completed: {} capture(s) written",
                        succeeded_paths.len()
                    );
                    return;
                }
            }
            Some(FrameCaptureStatus::Failed { message, .. }) => {
                error!("Headless capture failed: {message}");
                return;
            }
            _ => {}
        }
    }

    if expected_captures > 0 {
        error!(
            "Capture did not complete within {} frames ({} of {} written)",
            frame_budget,
            succeeded_paths.len(),
            expected_captures
        );
    }
}

// ── Camera helper ───────────────────────────────────────────────────────────

/// Set a default perspective camera on the renderer and scene.
///
/// Camera positioned at `eye`, looking at `target`, with Y-up.
pub fn set_default_camera(
    renderer: &mut Renderer,
    scene: &mut Scene,
    eye: Vec3,
    target: Vec3,
    fov_deg: f32,
) {
    renderer
        .set_camera_look_at(eye, target, Vec3::Y)
        .expect("set renderer camera look-at");
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    let aspect = 1920.0 / 1080.0;
    let projection = Mat4::perspective_rh(fov_deg.to_radians(), aspect, 0.1, 100.0);
    scene.set_camera(view, projection, eye);
}

// ── Mesh builders ───────────────────────────────────────────────────────────

/// Build a unit cube (1×1×1, centered at origin) as procedural mesh data.
///
/// Returns vertices and indices forming 12 triangles (2 per face × 6 faces).
/// Each face gets its own vertices for correct face normals.
#[allow(
    dead_code,
    reason = "shared mesh helper is not used by every capture binary"
)]
pub fn build_cube_mesh(name: &str) -> ProceduralMeshData {
    let s = 0.5; // half-size
                 // 6 faces × 4 vertices = 24 vertices for correct face normals
    let vertices = vec![
        // +X face (normal = +X)
        vtx([s, -s, s], [1.0, 0.0, 0.0], [0.0, 0.0]),
        vtx([s, s, s], [1.0, 0.0, 0.0], [0.0, 1.0]),
        vtx([s, s, -s], [1.0, 0.0, 0.0], [1.0, 1.0]),
        vtx([s, -s, -s], [1.0, 0.0, 0.0], [1.0, 0.0]),
        // -X face (normal = -X)
        vtx([-s, -s, -s], [-1.0, 0.0, 0.0], [0.0, 0.0]),
        vtx([-s, s, -s], [-1.0, 0.0, 0.0], [0.0, 1.0]),
        vtx([-s, s, s], [-1.0, 0.0, 0.0], [1.0, 1.0]),
        vtx([-s, -s, s], [-1.0, 0.0, 0.0], [1.0, 0.0]),
        // +Y face (normal = +Y)
        vtx([-s, s, s], [0.0, 1.0, 0.0], [0.0, 0.0]),
        vtx([s, s, s], [0.0, 1.0, 0.0], [1.0, 0.0]),
        vtx([s, s, -s], [0.0, 1.0, 0.0], [1.0, 1.0]),
        vtx([-s, s, -s], [0.0, 1.0, 0.0], [0.0, 1.0]),
        // -Y face (normal = -Y)
        vtx([-s, -s, -s], [0.0, -1.0, 0.0], [0.0, 0.0]),
        vtx([s, -s, -s], [0.0, -1.0, 0.0], [1.0, 0.0]),
        vtx([s, -s, s], [0.0, -1.0, 0.0], [1.0, 1.0]),
        vtx([-s, -s, s], [0.0, -1.0, 0.0], [0.0, 1.0]),
        // +Z face (normal = +Z)
        vtx([-s, -s, s], [0.0, 0.0, 1.0], [0.0, 0.0]),
        vtx([-s, s, s], [0.0, 0.0, 1.0], [0.0, 1.0]),
        vtx([s, s, s], [0.0, 0.0, 1.0], [1.0, 1.0]),
        vtx([s, -s, s], [0.0, 0.0, 1.0], [1.0, 0.0]),
        // -Z face (normal = -Z)
        vtx([s, -s, -s], [0.0, 0.0, -1.0], [0.0, 0.0]),
        vtx([s, s, -s], [0.0, 0.0, -1.0], [0.0, 1.0]),
        vtx([-s, s, -s], [0.0, 0.0, -1.0], [1.0, 1.0]),
        vtx([-s, -s, -s], [0.0, 0.0, -1.0], [1.0, 0.0]),
    ];

    let indices: Vec<u32> = (0..6)
        .flat_map(|face| {
            let base = face * 4;
            vec![base, base + 1, base + 2, base, base + 2, base + 3]
        })
        .collect();

    ProceduralMeshData {
        name: name.to_string(),
        vertices,
        indices,
        material: None,
    }
}

/// Build a UV sphere as procedural mesh data.
///
/// `segments` controls the number of longitudinal slices (min 3).
/// The sphere has radius 1.0 and is centered at origin.
#[allow(
    dead_code,
    reason = "shared mesh helper is not used by every capture binary"
)]
pub fn build_sphere_mesh(name: &str, segments: u32) -> ProceduralMeshData {
    let segs = segments.max(4);
    let rings = segs / 2; // latitude rings (half the segments)

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Generate vertices: rings+1 rows, segs+1 columns
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32; // 0.0 (top) to 1.0 (bottom)
        let theta = v * std::f32::consts::PI; // 0 to PI
        let y = theta.cos();
        let r = theta.sin();

        for seg in 0..=segs {
            let u = seg as f32 / segs as f32; // 0.0 to 1.0 around
            let phi = u * 2.0 * std::f32::consts::PI;

            let nx = r * phi.cos();
            let nz = r * phi.sin();
            let normal = Vec3::new(nx, y, nz);
            let position = normal; // unit sphere
            let tangent = compute_tangent(normal);
            let uv0 = Vec2::new(u, v);

            vertices.push(ProceduralVertex {
                position,
                normal,
                tangent,
                uv0,
                uv1: Vec2::ZERO,
                color: Vec4::ONE,
            });
        }
    }

    // Generate indices: two triangles per quad
    let cols = segs + 1;
    for ring in 0..rings {
        for seg in 0..segs {
            let a = ring * cols + seg;
            let b = a + cols;
            let c = a + 1;
            let d = b + 1;

            indices.push(a);
            indices.push(b);
            indices.push(c);

            indices.push(c);
            indices.push(b);
            indices.push(d);
        }
    }

    ProceduralMeshData {
        name: name.to_string(),
        vertices,
        indices,
        material: None,
    }
}

/// Build a flat quad in the XZ plane as procedural mesh data.
///
/// Centered at origin, normal pointing +Y.
#[allow(
    dead_code,
    reason = "shared mesh helper is not used by every capture binary"
)]
pub fn build_plane_mesh(name: &str, size: f32) -> ProceduralMeshData {
    let s = size * 0.5;
    let vertices = vec![
        vtx([-s, 0.0, -s], [0.0, 1.0, 0.0], [0.0, 0.0]),
        vtx([s, 0.0, -s], [0.0, 1.0, 0.0], [1.0, 0.0]),
        vtx([s, 0.0, s], [0.0, 1.0, 0.0], [1.0, 1.0]),
        vtx([-s, 0.0, s], [0.0, 1.0, 0.0], [0.0, 1.0]),
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];

    ProceduralMeshData {
        name: name.to_string(),
        vertices,
        indices,
        material: None,
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

#[allow(
    dead_code,
    reason = "shared mesh helper is not used by every capture binary"
)]
fn vtx(pos: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> ProceduralVertex {
    let position = Vec3::from_array(pos);
    let n = Vec3::from_array(normal);
    ProceduralVertex {
        position,
        normal: n,
        tangent: compute_tangent(n),
        uv0: Vec2::from_array(uv),
        uv1: Vec2::ZERO,
        color: Vec4::ONE,
    }
}

/// Compute a tangent vector perpendicular to the given normal.
/// For +Y normal (0,1,0), returns (1,0,0,1) (tangent in +X, handedness +1).
/// For -Y normal (0,-1,0), returns (1,0,0,-1) (tangent in +X, handedness -1).
/// For other normals, uses a cross-product with an up vector.
#[allow(
    dead_code,
    reason = "shared mesh helper is not used by every capture binary"
)]
pub fn compute_tangent(normal: Vec3) -> Vec4 {
    let up = if normal.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Z
    };
    let tangent = up.cross(normal).normalize();
    // handedness: cross(normal, tangent) should align with bitangent direction
    let bitangent = normal.cross(tangent);
    let handedness = if bitangent.dot(Vec3::Y) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    tangent.extend(handedness)
}

fn init_logging() {
    let _ = env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_mesh_is_valid() {
        let mesh = build_cube_mesh("test_cube");
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36); // 12 triangles × 3
        assert!(mesh.indices.iter().all(|&i| (i as usize) < 24));
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn sphere_mesh_is_valid() {
        let mesh = build_sphere_mesh("test_sphere", 16);
        let n_verts = mesh.vertices.len();
        assert!(n_verts > 0);
        assert_eq!(mesh.indices.len() % 3, 0);
        assert!(mesh.indices.iter().all(|&i| (i as usize) < n_verts));
    }

    #[test]
    fn plane_mesh_is_valid() {
        let mesh = build_plane_mesh("test_plane", 10.0);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices.len(), 6);
        assert!(mesh.indices.iter().all(|&i| (i as usize) < 4));
    }

    #[test]
    fn parse_capture_test_args_defaults() {
        let args = parse_capture_test_args_from(Vec::<&str>::new());
        assert!(!args.headless);
        assert_eq!(args.capture_target, CaptureTarget::Draw);
        assert!(args.capture_frames.is_none());
    }

    #[test]
    fn parse_capture_test_args_headless() {
        let args = parse_capture_test_args_from(["--headless", "--capture_frames=3"]);
        assert!(args.headless);
        assert_eq!(args.capture_frames, Some(3));
    }
}
