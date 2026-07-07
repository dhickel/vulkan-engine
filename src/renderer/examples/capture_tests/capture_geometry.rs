//! Headless capture test: Procedural Geometry Verification
//!
//! Renders a cube, sphere, pyramid, and plane with distinct colored materials.
//! Tests the procedural mesh creation pipeline end-to-end.
//!
//! Run: cargo run -p renderer --example capture_geometry -- --headless

mod common;

use glam::{Mat4, Vec3, Vec4};
use renderer::prelude::{PbrMaterialDesc, PointLight, ProceduralMeshData, ProceduralVertex, Scene};

use common::*;

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_geometry", &args, build_scene);
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();

    // ── Camera ──────────────────────────────────────────────────────────
    let eye = Vec3::new(0.0, 4.0, 10.0);
    set_default_camera(renderer, &mut scene, eye, Vec3::new(0.0, 0.5, 0.0), 55.0);

    let root = scene.create_node_default(None).expect("capture root");
    scene
        .set_node_name(root, "Capture Root")
        .expect("root name");

    let mut assets = renderer.assets();

    // ── Materials ───────────────────────────────────────────────────────
    let red_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.9, 0.15, 0.15, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("red");

    let green_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.15, 0.85, 0.2, 1.0),
            metallic: 0.0,
            roughness: 0.4,
            ..Default::default()
        })
        .expect("green");

    let blue_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.15, 0.3, 0.95, 1.0),
            metallic: 0.3,
            roughness: 0.3,
            ..Default::default()
        })
        .expect("blue");

    let gray_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.5, 0.5, 0.5, 1.0),
            metallic: 0.0,
            roughness: 0.7,
            ..Default::default()
        })
        .expect("gray");

    let gold_mat = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(1.0, 0.85, 0.1, 1.0),
            metallic: 1.0,
            roughness: 0.25,
            ..Default::default()
        })
        .expect("gold");

    // ── Ground plane ────────────────────────────────────────────────────
    {
        let mut plane = build_plane_mesh("ground", 14.0);
        plane.material = Some(gray_mat);
        let plane_handle = assets.upload_procedural_mesh(plane).expect("plane");
        let ground = scene
            .create_node(Some(root), Mat4::IDENTITY)
            .expect("ground");
        scene.add_mesh(ground, plane_handle).expect("add plane");
    }

    // ── Cube (red) ──────────────────────────────────────────────────────
    {
        let mut cube = build_cube_mesh("cube");
        cube.material = Some(red_mat);
        let handle = assets.upload_procedural_mesh(cube).expect("cube");
        let node = scene
            .create_node(
                Some(root),
                Mat4::from_scale_rotation_translation(
                    Vec3::splat(1.0),
                    glam::Quat::from_rotation_y(0.3),
                    Vec3::new(-3.0, 0.5, 0.0),
                ),
            )
            .expect("cube node");
        scene.add_mesh(node, handle).expect("add cube");
    }

    // ── Sphere (green) ──────────────────────────────────────────────────
    {
        let mut sphere = build_sphere_mesh("sphere", 32);
        sphere.material = Some(green_mat);
        let handle = assets.upload_procedural_mesh(sphere).expect("sphere");
        let node = scene
            .create_node(Some(root), Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)))
            .expect("sphere node");
        scene.add_mesh(node, handle).expect("add sphere");
    }

    // ── Pyramid/tetrahedron (blue) ──────────────────────────────────────
    {
        let pyramid = build_pyramid_mesh("pyramid");
        let mut mesh = pyramid;
        mesh.material = Some(blue_mat);
        let handle = assets.upload_procedural_mesh(mesh).expect("pyramid");
        let node = scene
            .create_node(
                Some(root),
                Mat4::from_scale_rotation_translation(
                    Vec3::splat(1.2),
                    glam::Quat::from_rotation_y(-0.4),
                    Vec3::new(3.0, 0.0, 0.0),
                ),
            )
            .expect("pyramid node");
        scene.add_mesh(node, handle).expect("add pyramid");
    }

    // ── Small metallic sphere (gold) ────────────────────────────────────
    {
        let mut sphere = build_sphere_mesh("gold_sphere", 32);
        sphere.material = Some(gold_mat);
        let handle = assets.upload_procedural_mesh(sphere).expect("gold sphere");
        let node = scene
            .create_node(Some(root), Mat4::from_translation(Vec3::new(0.0, 1.0, 3.0)))
            .expect("gold node");
        scene.add_mesh(node, handle).expect("add gold sphere");
    }

    // ── Lighting ────────────────────────────────────────────────────────
    scene
        .create_point_light(PointLight {
            position: Vec3::new(2.0, 6.0, 4.0),
            color: Vec3::new(1.0, 0.95, 0.85),
            intensity: 60.0,
            range: 20.0,
        })
        .expect("key light");

    scene
        .create_point_light(PointLight {
            position: Vec3::new(-4.0, 3.0, -2.0),
            color: Vec3::new(0.3, 0.4, 0.9),
            intensity: 25.0,
            range: 12.0,
        })
        .expect("fill light");

    scene
}

/// Build a simple tetrahedron (pyramid) mesh.
fn build_pyramid_mesh(name: &str) -> ProceduralMeshData {
    // Tetrahedron vertices: 4 corners, each with its own face normal
    let h = 1.0_f32; // height
    let r = 0.57735_f32; // base radius (1/sqrt(3))

    // Base triangle vertices (CCW from above)
    let b0 = Vec3::new(0.0, 0.0, r * 2.0 / 3.0);
    let b1 = Vec3::new(-r, 0.0, -r / 3.0);
    let b2 = Vec3::new(r, 0.0, -r / 3.0);
    let apex = Vec3::new(0.0, h, 0.0);

    // Build with face-specific vertices for correct normals
    let mut vertices = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Helper to add a triangle face
    let mut add_face = |a: Vec3, b: Vec3, c: Vec3| {
        let base = vertices.len() as u32;
        let normal = (b - a).cross(c - a).normalize();
        let tangent = compute_tangent(normal);
        vertices.push(ProceduralVertex {
            position: a,
            normal,
            tangent,
            uv0: glam::Vec2::new(0.0, 0.0),
            uv1: glam::Vec2::ZERO,
            color: Vec4::ONE,
        });
        vertices.push(ProceduralVertex {
            position: b,
            normal,
            tangent,
            uv0: glam::Vec2::new(0.5, 1.0),
            uv1: glam::Vec2::ZERO,
            color: Vec4::ONE,
        });
        vertices.push(ProceduralVertex {
            position: c,
            normal,
            tangent,
            uv0: glam::Vec2::new(1.0, 0.0),
            uv1: glam::Vec2::ZERO,
            color: Vec4::ONE,
        });
        indices.push(base);
        indices.push(base + 1);
        indices.push(base + 2);
    };

    // Base (two triangles, facing down)
    add_face(b2, b1, b0);
    add_face(b0, b1, b2);
    // Sides
    add_face(b0, b1, apex);
    add_face(b1, b2, apex);
    add_face(b2, b0, apex);

    ProceduralMeshData {
        name: name.to_string(),
        vertices,
        indices,
        material: None,
    }
}
