//! Tests for BSP mesh upload and lightmap atlas creation from ExtractedBsp DTOs.

#![cfg(feature = "bsp")]

use bsp::geometry::FaceGeometry;
use bsp::lightmaps::{AtlasPage, FaceLightmapLayout, LightmapAtlas};
use bsp::materials::{BspMaterial, SurfaceClass};
use bsp::resources::{ExtractedTexture, PbrTextureCompanions, TextureCompanion};
use glam::{Vec2, Vec3};
use renderer::api::bsp::{
    build_bsp_material_descs, build_face_meshes, face_to_procedural_mesh, BspLightmapAtlasPage,
    BspSurfaceClass,
};
use renderer::api::BspTextureHandle;

// ── Face mesh triangulation tests ───────────────────────────────────────

fn make_face(vertices: Vec<Vec3>, uv0: Vec<Vec2>) -> FaceGeometry {
    let n = vertices.len();
    FaceGeometry {
        face_index: 0,
        vertices,
        uv0,
        uv1: vec![Vec2::ZERO; n],
        normal: Vec3::Z,
        bounds: (Vec3::ZERO, Vec3::ONE),
        luxel_extents: (16, 16),
        is_valid: true,
    }
}

#[test]
fn triangle_face_converts_directly() {
    let face_geo = make_face(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        ],
    );

    let mesh = face_to_procedural_mesh(&face_geo, 1.0, 1.0, None);
    assert!(mesh.is_some());
    let m = mesh.unwrap();
    assert_eq!(m.vertices.len(), 3);
    assert_eq!(m.indices.len(), 3); // single triangle
}

#[test]
fn quad_face_triangulates_to_two_triangles() {
    let face_geo = make_face(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ],
        vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ],
    );

    let mesh = face_to_procedural_mesh(&face_geo, 1.0, 1.0, None);
    assert!(mesh.is_some());
    let m = mesh.unwrap();
    assert_eq!(m.vertices.len(), 4);
    assert_eq!(m.indices.len(), 6); // fan: [0,1,2, 0,2,3]
    assert_eq!(m.indices, vec![0, 1, 2, 0, 2, 3]);
}

#[test]
fn degenerate_face_rejected() {
    let face_geo = FaceGeometry {
        face_index: 0,
        vertices: vec![Vec3::new(0.0, 0.0, 0.0)],
        uv0: vec![Vec2::ZERO],
        uv1: vec![Vec2::ZERO],
        normal: Vec3::Z,
        bounds: (Vec3::ZERO, Vec3::ZERO),
        luxel_extents: (0, 0),
        is_valid: true,
    };

    let mesh = face_to_procedural_mesh(&face_geo, 1.0, 1.0, None);
    assert!(mesh.is_none());
}

#[test]
fn normal_is_shared_across_all_vertices() {
    let face_geo = make_face(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![Vec2::ZERO; 3],
    );

    let mesh = face_to_procedural_mesh(&face_geo, 1.0, 1.0, None);
    assert!(mesh.is_some());
    let m = mesh.unwrap();
    let n0 = m.vertices[0].normal;
    let n1 = m.vertices[1].normal;
    let n2 = m.vertices[2].normal;
    assert!((n0 - n1).length() < 0.001);
    assert!((n1 - n2).length() < 0.001);
    assert!((n0.length() - 1.0).abs() < 0.001);
}

// ── Lightmap atlas tests ────────────────────────────────────────────────

#[test]
fn empty_atlas_produces_empty_pages() {
    let atlas = LightmapAtlas::new();
    let extracted = make_extracted_with_atlas(atlas);

    let pages = BspLightmapAtlasPage::from_extracted(&extracted);
    assert!(pages.is_empty());
}

#[test]
fn atlas_page_allocation_handle_increments() {
    let page = BspLightmapAtlasPage {
        width: 256,
        height: 256,
        layer_count: 1,
        pixels: vec![0u8; 256 * 256 * 4],
    };

    let mut slot = 0u32;
    let h0 = page.allocate_handle(&mut slot);
    assert_eq!(h0.slot, 0);
    assert_eq!(slot, 1);

    let h1 = page.allocate_handle(&mut slot);
    assert_eq!(h1.slot, 1);
    assert_eq!(slot, 2);
}

// ── BSP material descriptor tests ────────────────────────────────────────

fn make_extracted_with_atlas(atlas: LightmapAtlas) -> bsp::extract::ExtractedBsp {
    bsp::extract::ExtractedBsp {
        transform: bsp::coords::QuakeToEngine::new(0.0254),
        profile_tag: "bsp29",
        textures: vec![],
        face_geometries: vec![],
        face_materials: vec![],
        render_batches: vec![],
        lightmap_atlas: atlas,
        face_lightmap_layouts: vec![],
        has_pvs: false,
        camera_pvs: None,
        visibility: bsp::extract::ExtractedVisibility::default(),
        leaf_membership: vec![],
        entity_descriptors: vec![],
        entity_identities: vec![],
        light_descriptors: vec![],
        inline_models: vec![],
        world_collision_planes: vec![],
        collision_recipes: vec![],
        content_hash: [0u8; 32],
        source_identity: String::new(),
        diagnostics: vec![],
    }
}

#[test]
fn nodraw_faces_produce_no_material_desc() {
    let face_geo = make_face(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![Vec2::ZERO; 3],
    );

    let atlas = LightmapAtlas::new();
    let layout = FaceLightmapLayout {
        page_index: 0,
        atlas_offset: (0, 0),
        luxel_extents: (16, 16),
        has_data: false,
        style_layers: vec![],
    };

    let extracted = bsp::extract::ExtractedBsp {
        face_geometries: vec![face_geo],
        face_materials: vec![BspMaterial {
            surface_class: SurfaceClass::NoDraw,
            ..BspMaterial::default()
        }],
        face_lightmap_layouts: vec![layout],
        lightmap_atlas: atlas,
        ..make_extracted_with_atlas(LightmapAtlas::new())
    };

    let dummy_albedo = vec![BspTextureHandle::new(0, 0)];
    let dummy_lightmap = BspTextureHandle::new(1, 0);
    let descs = build_bsp_material_descs(&extracted, &dummy_albedo, dummy_lightmap);
    assert_eq!(descs.len(), 1);
    assert!(descs[0].is_none());
}

#[test]
fn opaque_face_produces_lightmapped_desc() {
    let face_geo = make_face(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![Vec2::ZERO; 3],
    );

    let atlas_page = AtlasPage::new(0, 256, 256);
    let mut atlas = LightmapAtlas::new();
    atlas.pages = vec![atlas_page];

    let layout = FaceLightmapLayout {
        page_index: 0,
        atlas_offset: (0, 0),
        luxel_extents: (16, 16),
        has_data: true,
        style_layers: vec![],
    };

    let extracted = bsp::extract::ExtractedBsp {
        face_geometries: vec![face_geo],
        face_materials: vec![BspMaterial {
            surface_class: SurfaceClass::Opaque,
            ..BspMaterial::default()
        }],
        face_lightmap_layouts: vec![layout],
        lightmap_atlas: atlas,
        ..make_extracted_with_atlas(LightmapAtlas::new())
    };

    let dummy_albedo = vec![BspTextureHandle::new(0, 0)];
    let dummy_lightmap = BspTextureHandle::new(1, 0);
    let descs = build_bsp_material_descs(&extracted, &dummy_albedo, dummy_lightmap);
    assert_eq!(descs.len(), 1);
    assert!(descs[0].is_some());
    let desc = descs[0].as_ref().unwrap();
    assert_eq!(desc.surface_class, BspSurfaceClass::Lightmapped);
    assert_eq!(desc.textures.lightmap_atlas.slot, 1);
}

// ── Face mesh build for extracted BSP ───────────────────────────────────

#[test]
fn pbr_companion_face_produces_pbr_lightmapped_desc() {
    let face_geo = make_face(
        vec![Vec3::ZERO, Vec3::X, Vec3::Y],
        vec![Vec2::ZERO, Vec2::X, Vec2::Y],
    );
    let mut atlas = LightmapAtlas::new();
    atlas.pages.push(AtlasPage::new(0, 16, 16));
    let extracted = bsp::extract::ExtractedBsp {
        textures: vec![ExtractedTexture {
            identity: "brick1_2".into(),
            width: 1,
            height: 1,
            albedo: vec![255; 4],
            fullbright_mask: vec![0],
            pbr_companions: PbrTextureCompanions {
                normal: Some(TextureCompanion::new("brick1_2_norm.png", vec![1])),
                gloss: Some(TextureCompanion::new("brick1_2_gloss.png", vec![2])),
            },
            ..ExtractedTexture::default()
        }],
        face_geometries: vec![face_geo],
        face_materials: vec![BspMaterial {
            material_index: 0,
            texture_identity: "brick1_2".into(),
            surface_class: SurfaceClass::Opaque,
            ..BspMaterial::default()
        }],
        face_lightmap_layouts: vec![FaceLightmapLayout {
            page_index: 0,
            atlas_offset: (0, 0),
            luxel_extents: (1, 1),
            has_data: true,
            style_layers: vec![],
        }],
        lightmap_atlas: atlas,
        ..make_extracted_with_atlas(LightmapAtlas::new())
    };

    let descs = build_bsp_material_descs(
        &extracted,
        &[BspTextureHandle::new(0, 0)],
        BspTextureHandle::new(1, 0),
    );
    let desc = descs[0].as_ref().unwrap();
    assert_eq!(desc.surface_class, BspSurfaceClass::PbrLightmapped);
    assert_ne!(
        desc.surface_params.surface_flags & renderer::api::bsp::bsp_surface_flags::SURF_PBR,
        0
    );
    assert_ne!(
        desc.surface_params.receive_mask & renderer::api::bsp::bsp_surface_flags::RECEIVE_IBL,
        0
    );
}

#[test]
fn build_face_meshes_skips_invalid_faces() {
    let valid_face = make_face(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![Vec2::ZERO; 3],
    );
    let invalid_face = FaceGeometry {
        face_index: 1,
        is_valid: false,
        ..valid_face.clone()
    };
    let nodraw_face = make_face(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![Vec2::ZERO; 3],
    );

    let extracted = bsp::extract::ExtractedBsp {
        face_geometries: vec![valid_face, invalid_face, nodraw_face],
        face_materials: vec![
            BspMaterial {
                surface_class: SurfaceClass::Opaque,
                ..BspMaterial::default()
            },
            BspMaterial {
                surface_class: SurfaceClass::Opaque,
                ..BspMaterial::default()
            },
            BspMaterial {
                surface_class: SurfaceClass::NoDraw,
                ..BspMaterial::default()
            },
        ],
        ..make_extracted_with_atlas(LightmapAtlas::new())
    };

    let meshes = build_face_meshes(&extracted);
    assert_eq!(meshes.len(), 3);
    assert!(meshes[0].is_some()); // valid
    assert!(meshes[1].is_none()); // invalid
    assert!(meshes[2].is_none()); // nodraw
}
