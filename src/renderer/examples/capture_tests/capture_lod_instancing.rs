//! Headless capture test: LOD fixture and instanced-draw candidate groups.
//!
//! Scene layout:
//! - Row of red cubes at varying depths for LOD selection (when scene-bvh enabled).
//! - Row of blue cubes for instanced-group verification (when instancing enabled).
//! - Reference plane for background.
//! - Feature-gated: LOD/BVH behind `scene-bvh`, instancing behind `instancing`.
//!   Falls back to standard rendering when features are disabled.

mod common;

use common::*;
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use renderer::prelude::{PbrMaterialDesc, PointLight, ProceduralMeshData, ProceduralVertex, Scene};

fn main() {
    let args = parse_capture_test_args();
    run_headless_capture_test("capture_lod_instancing", &args, move |renderer| {
        build_scene(renderer)
    });
}

fn build_scene(renderer: &mut renderer::prelude::Renderer) -> Scene {
    let mut scene = Scene::new();
    scene.set_frustum_culling(true);

    set_default_camera(
        renderer,
        &mut scene,
        Vec3::new(0.0, 2.5, 12.0),
        Vec3::new(0.0, 0.5, 0.0),
        55.0,
    );

    let root = scene.create_node_default(None).expect("root");
    let mut assets = renderer.assets();

    // ---- Materials ----
    let opaque_red = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.85, 0.15, 0.15, 1.0),
            metallic: 0.05,
            roughness: 0.45,
            ..Default::default()
        })
        .expect("opaque red");

    let opaque_blue = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.15, 0.25, 0.85, 1.0),
            metallic: 0.1,
            roughness: 0.35,
            ..Default::default()
        })
        .expect("opaque blue");

    let opaque_green = assets
        .create_material_pbr(PbrMaterialDesc {
            base_color: Vec4::new(0.15, 0.75, 0.25, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            ..Default::default()
        })
        .expect("opaque green");

    // ---- Meshes ----
    let cube_red = build_cube_mesh_with_mat("lod_red_cube", Some(opaque_red));
    let cube_red_mesh = assets.upload_procedural_mesh(cube_red).expect("cube red");
    let cube_red_bounds = assets
        .mesh_scene_bounds(cube_red_mesh)
        .expect("cube red bounds");

    let cube_blue = build_cube_mesh_with_mat("lod_blue_cube", Some(opaque_blue));
    let cube_blue_mesh = assets.upload_procedural_mesh(cube_blue).expect("cube blue");
    let cube_blue_bounds = assets
        .mesh_scene_bounds(cube_blue_mesh)
        .expect("cube blue bounds");

    let cube_green = build_cube_mesh_with_mat("lod_green_cube", Some(opaque_green));
    let cube_green_mesh = assets
        .upload_procedural_mesh(cube_green)
        .expect("cube green");
    let cube_green_bounds = assets
        .mesh_scene_bounds(cube_green_mesh)
        .expect("cube green bounds");

    // ---- Control draws (singleton, always present) ----
    let control_node = scene
        .create_node(
            Some(root),
            Mat4::from_translation(Vec3::new(-3.5, 2.0, 0.0)),
        )
        .expect("control node");
    scene
        .add_mesh_with_bounds(control_node, cube_green_mesh, cube_green_bounds)
        .expect("control mesh");

    // ---- Instanced group candidates: repeated same-mesh/material opaque cubes ----
    // Row of red cubes at varying depths.
    let red_positions = [
        Vec3::new(-2.0, 0.0, -2.0),
        Vec3::new(-2.0, 0.0, 0.0),
        Vec3::new(-2.0, 0.0, 2.0),
        Vec3::new(-2.0, 0.0, 4.0),
        Vec3::new(-2.0, 0.0, 6.0),
    ];
    for &pos in &red_positions {
        let node = scene
            .create_node(Some(root), Mat4::from_translation(pos))
            .expect("red cube node");
        scene
            .add_mesh_with_bounds(node, cube_red_mesh, cube_red_bounds)
            .expect("red cube mesh");
    }

    // Row of blue cubes (separate group from red — different material).
    let blue_positions = [
        Vec3::new(2.0, 0.0, -2.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 2.0),
        Vec3::new(2.0, 0.0, 4.0),
        Vec3::new(2.0, 0.0, 6.0),
    ];
    for &pos in &blue_positions {
        let node = scene
            .create_node(Some(root), Mat4::from_translation(pos))
            .expect("blue cube node");
        scene
            .add_mesh_with_bounds(node, cube_blue_mesh, cube_blue_bounds)
            .expect("blue cube mesh");
    }

    // ---- Background reference plane ----
    let plane_node = scene
        .create_node(
            Some(root),
            Mat4::from_scale_rotation_translation(
                Vec3::new(20.0, 0.01, 20.0),
                Quat::IDENTITY,
                Vec3::new(0.0, -0.7, 0.0),
            ),
        )
        .expect("plane node");
    let plane = build_plane_mesh("lod_plane", None);
    let plane_mesh = assets.upload_procedural_mesh(plane).expect("plane mesh");
    let plane_bounds = assets.mesh_scene_bounds(plane_mesh).expect("plane bounds");
    scene
        .add_mesh_with_bounds(plane_node, plane_mesh, plane_bounds)
        .expect("plane mesh add");

    // ---- Point light ----
    let _ = scene.create_point_light(PointLight {
        position: Vec3::new(0.0, 4.0, 0.0),
        color: Vec3::new(1.0, 0.95, 0.8),
        intensity: 80.0,
        range: 15.0,
    });

    scene
}

fn build_cube_mesh_with_mat(
    name: &str,
    material: Option<renderer::prelude::MaterialHandle>,
) -> ProceduralMeshData {
    let s = 0.5;
    let positions: [Vec3; 24] = [
        Vec3::new(s, -s, -s),
        Vec3::new(s, -s, s),
        Vec3::new(s, s, s),
        Vec3::new(s, s, -s),
        Vec3::new(-s, -s, s),
        Vec3::new(-s, -s, -s),
        Vec3::new(-s, s, -s),
        Vec3::new(-s, s, s),
        Vec3::new(-s, s, -s),
        Vec3::new(s, s, -s),
        Vec3::new(s, s, s),
        Vec3::new(-s, s, s),
        Vec3::new(-s, -s, s),
        Vec3::new(s, -s, s),
        Vec3::new(s, -s, -s),
        Vec3::new(-s, -s, -s),
        Vec3::new(-s, -s, s),
        Vec3::new(-s, s, s),
        Vec3::new(s, s, s),
        Vec3::new(s, -s, s),
        Vec3::new(s, -s, -s),
        Vec3::new(s, s, -s),
        Vec3::new(-s, s, -s),
        Vec3::new(-s, -s, -s),
    ];
    let normals: [Vec3; 24] = [
        Vec3::X,
        Vec3::X,
        Vec3::X,
        Vec3::X,
        Vec3::NEG_X,
        Vec3::NEG_X,
        Vec3::NEG_X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::Y,
        Vec3::Y,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::NEG_Y,
        Vec3::NEG_Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::Z,
        Vec3::Z,
        Vec3::Z,
        Vec3::NEG_Z,
        Vec3::NEG_Z,
        Vec3::NEG_Z,
        Vec3::NEG_Z,
    ];
    let indices: Vec<u32> = vec![
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17,
        18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    ];
    let vertices: Vec<ProceduralVertex> = positions
        .iter()
        .zip(normals.iter())
        .map(|(&pos, &nrm)| ProceduralVertex {
            position: pos,
            normal: nrm,
            uv0: Vec2::ZERO,
            ..Default::default()
        })
        .collect();
    ProceduralMeshData {
        name: name.to_string(),
        vertices,
        indices,
        material,
    }
}

fn build_plane_mesh(
    name: &str,
    material: Option<renderer::prelude::MaterialHandle>,
) -> ProceduralMeshData {
    let s = 0.5;
    let positions = [
        Vec3::new(-s, 0.0, -s),
        Vec3::new(s, 0.0, -s),
        Vec3::new(s, 0.0, s),
        Vec3::new(-s, 0.0, s),
    ];
    let normal = Vec3::Y;
    let vertices: Vec<ProceduralVertex> = positions
        .iter()
        .map(|&pos| ProceduralVertex {
            position: pos,
            normal,
            uv0: Vec2::ZERO,
            ..Default::default()
        })
        .collect();
    ProceduralMeshData {
        name: name.to_string(),
        vertices,
        indices: vec![0u32, 1, 2, 0, 2, 3],
        material,
    }
}
