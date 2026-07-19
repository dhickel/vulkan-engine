//! Headless capture test: CSM (Cascaded Shadow Maps) validation.
//!
//! Renders a deterministic scene designed to exercise cascade boundaries,
//! off-camera casters, per-cascade culling, and shadow stability across frames.
//!
//! Scene layout (camera at origin, looking down -Z):
//!   - Near cascade (0-10m): small occluder casting shadow on ground.
//!   - Mid cascade (10-30m): taller occluder, partially occluded.
//!   - Far cascade (30-80m): large background wall, lit from above.
//!   - Off-camera caster: tall pillar behind camera that still casts shadow
//!     into the light's footprint (visible on the ground in front).
//!   - Cascade-boundary pillar: placed near the mid/far split to exercise
//!     the blend band.
//!
//! Run with CSM:
//!   cargo run -p renderer --example capture_csm --features csm -- --headless
//!
//! Run legacy (no CSM):
//!   cargo run -p renderer --example capture_csm -- --headless

mod common;

use common::*;
use glam::{Mat4, Quat, Vec3, Vec4};
use renderer::prelude::{DirectionalLight, PbrMaterialDesc, Scene};

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_csm", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();

    // ── Camera ──────────────────────────────────────────────────────────
    // Position camera looking down a long corridor to span near/far depths.
    let camera_offset_x = std::env::var("CAPTURE_CSM_CAMERA_OFFSET_X")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    let eye = Vec3::new(camera_offset_x, 3.5, 8.0);
    let target = Vec3::new(camera_offset_x, 1.0, -20.0);
    set_default_camera(renderer, &mut scene, eye, target, 55.0);

    let root = scene.create_node_default(None).expect("csm root");
    scene.set_node_name(root, "CSM Capture Root").unwrap();

    let mut assets = renderer.assets();

    // ── Materials ───────────────────────────────────────────────────────
    let ground_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.65, 0.65, 0.68, 1.0),
            metallic: 0.0,
            roughness: 0.75,
            ..Default::default()
        })
        .expect("ground material");

    let near_caster_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.85, 0.25, 0.15, 1.0),
            metallic: 0.05,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("near caster material");

    let mid_caster_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.2, 0.7, 0.25, 1.0),
            metallic: 0.05,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("mid caster material");

    let far_wall_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.75, 0.7, 0.65, 1.0),
            metallic: 0.0,
            roughness: 0.8,
            ..Default::default()
        })
        .expect("far wall material");

    let pillar_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.5, 0.5, 0.85, 1.0),
            metallic: 0.1,
            roughness: 0.45,
            ..Default::default()
        })
        .expect("pillar material");

    let occluder_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.92, 0.8, 0.3, 1.0),
            metallic: 0.05,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("occluder material");

    // ── Ground plane ────────────────────────────────────────────────────
    {
        let mut plane = build_plane_mesh("csm_ground", 70.0);
        plane.material = Some(ground_mat);
        let plane_handle = assets.upload_procedural_mesh(plane).expect("ground");
        let ground = scene
            .create_node(Some(root), Mat4::IDENTITY)
            .expect("ground node");
        scene.add_mesh(ground, plane_handle).expect("add ground");
        scene.set_node_name(ground, "Ground").unwrap();
    }

    // ── Far background wall ─────────────────────────────────────────────
    {
        let mut wall = build_cube_mesh("far_wall");
        wall.material = Some(far_wall_mat);
        let wall_handle = assets.upload_procedural_mesh(wall).expect("wall mesh");
        let wall_node = scene
            .create_node(
                Some(root),
                Mat4::from_scale_rotation_translation(
                    Vec3::new(18.0, 5.0, 0.5),
                    Quat::IDENTITY,
                    Vec3::new(0.0, 2.5, -35.0),
                ),
            )
            .expect("wall node");
        scene.add_mesh(wall_node, wall_handle).expect("add wall");
        scene.set_node_name(wall_node, "Far Wall").unwrap();
    }

    // ── Near-cascade caster (close to camera, ~3m) ─────────────────────
    {
        let mut cube = build_cube_mesh("near_caster");
        cube.material = Some(near_caster_mat);
        let cube_handle = assets.upload_procedural_mesh(cube).expect("near caster mesh");
        let node = scene
            .create_node(
                Some(root),
                Mat4::from_scale_rotation_translation(
                    Vec3::new(0.8, 1.6, 0.8),
                    Quat::IDENTITY,
                    Vec3::new(1.5, 0.8, 2.0),
                ),
            )
            .expect("near caster node");
        scene.add_mesh(node, cube_handle).expect("add near caster");
        scene.set_node_name(node, "Near Caster").unwrap();
    }

    // ── Mid-cascade caster (~12m from camera) ──────────────────────────
    {
        let mut cube = build_cube_mesh("mid_caster");
        cube.material = Some(mid_caster_mat);
        let cube_handle = assets.upload_procedural_mesh(cube).expect("mid caster mesh");
        let node = scene
            .create_node(
                Some(root),
                Mat4::from_scale_rotation_translation(
                    Vec3::new(1.2, 2.2, 1.2),
                    Quat::IDENTITY,
                    Vec3::new(-2.0, 1.1, -10.0),
                ),
            )
            .expect("mid caster node");
        scene.add_mesh(node, cube_handle).expect("add mid caster");
        scene.set_node_name(node, "Mid Caster").unwrap();
    }

    // ── Far-cascade occluder (~40m from camera) ─────────────────────────
    {
        let mut cube = build_cube_mesh("far_occluder");
        cube.material = Some(occluder_mat);
        let cube_handle = assets.upload_procedural_mesh(cube).expect("far occluder mesh");
        let node = scene
            .create_node(
                Some(root),
                Mat4::from_scale_rotation_translation(
                    Vec3::new(3.0, 3.5, 3.0),
                    Quat::IDENTITY,
                    Vec3::new(0.0, 1.75, -38.0),
                ),
            )
            .expect("far occluder node");
        scene.add_mesh(node, cube_handle).expect("add far occluder");
        scene.set_node_name(node, "Far Occluder").unwrap();
    }

    // ── Cascade-boundary pillar (near mid/far split ~25m) ──────────────
    {
        let mut cube = build_cube_mesh("boundary_pillar");
        cube.material = Some(pillar_mat);
        let cube_handle = assets.upload_procedural_mesh(cube).expect("boundary pillar mesh");
        let node = scene
            .create_node(
                Some(root),
                Mat4::from_scale_rotation_translation(
                    Vec3::new(0.6, 3.0, 0.6),
                    Quat::IDENTITY,
                    Vec3::new(3.5, 1.5, -22.0),
                ),
            )
            .expect("boundary pillar node");
        scene.add_mesh(node, cube_handle).expect("add boundary pillar");
        scene.set_node_name(node, "Cascade Boundary Pillar").unwrap();
    }

    // ── Off-camera caster (behind camera, still inside light footprint) ─
    // Placed behind and above the camera. The directional light (coming from
    // upper-right) will cast its shadow forward onto the ground plane even
    // though this pillar is outside the camera frustum.
    {
        let mut cube = build_cube_mesh("off_camera_caster");
        cube.material = Some(pillar_mat);
        let cube_handle = assets
            .upload_procedural_mesh(cube)
            .expect("off-camera caster mesh");
        let node = scene
            .create_node(
                Some(root),
                Mat4::from_scale_rotation_translation(
                    Vec3::new(0.7, 4.0, 0.7),
                    Quat::IDENTITY,
                    Vec3::new(0.0, 2.0, 12.0), // behind camera (camera at z=8, this at z=12)
                ),
            )
            .expect("off-camera caster node");
        scene.add_mesh(node, cube_handle).expect("add off-camera caster");
        scene.set_node_name(node, "Off-Camera Caster").unwrap();
    }

    // ── Lit receiver spheres (to verify correct shadow absence) ─────────
    {
        let mut sphere = build_sphere_mesh("lit_sphere", 24);
        sphere.material = Some(
            assets
                .create_material_pbr(PbrMaterialDesc {
                    base_color: Vec4::new(0.9, 0.9, 0.9, 1.0),
                    metallic: 0.0,
                    roughness: 0.4,
                    ..Default::default()
                })
                .expect("lit sphere material"),
        );
        let sphere_handle = assets.upload_procedural_mesh(sphere).expect("sphere mesh");

        for (i, (x, z)) in [(-4.0, 0.0), (4.0, -5.0), (-3.5, -15.0)].iter().enumerate() {
            let node = scene
                .create_node(
                    Some(root),
                    Mat4::from_scale_rotation_translation(
                        Vec3::splat(0.5),
                        Quat::IDENTITY,
                        Vec3::new(*x, 0.5, *z),
                    ),
                )
                .expect("lit sphere node");
            scene
                .add_mesh(node, sphere_handle)
                .expect("add lit sphere");
            scene
                .set_node_name(node, format!("Lit Sphere {}", i))
                .unwrap();
        }
    }

    drop(assets);

    // ── Directional light with CSM shadows enabled ──────────────────────
    // Light comes from upper-right casting shadows toward lower-left.
    let dir_light_id = scene
        .create_directional_light(DirectionalLight {
            direction: Vec3::new(-0.5, 1.2, 0.3),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 8.0,
        })
        .expect("directional light");

    // Enable shadow casting on this directional light.
    #[cfg(feature = "csm")]
    {
        use renderer::prelude::DirectionalShadowConfig;
        scene
            .set_directional_shadow_config(
                dir_light_id,
                DirectionalShadowConfig { enabled: true },
            )
            .expect("enable CSM shadows");
    }

    scene
}
