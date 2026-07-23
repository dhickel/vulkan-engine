//! Golden tests for approved lumps, extensions, companions, entities, and resources.
//!
//! Tests assert stable diagnostic codes/severity and deterministic ordering.
//! Uses programmatically-constructed BSP data since real compiled BSPs are
//! pending ericw-tools compilation (see fixture-manifest.toml placeholder status).

use bsp::*;

/// Construct a minimal valid BSP29 binary with configurable content.
fn make_bsp29_header(lump_entries: &[(usize, u32, u32)]) -> Vec<u8> {
    let mut data = vec![0u8; 124];
    data[0..4].copy_from_slice(&29u32.to_le_bytes());

    let mut lumps = [(0u32, 0u32); 15];
    for &(idx, off, sz) in lump_entries {
        lumps[idx] = (off, sz);
    }
    for (i, &(off, sz)) in lumps.iter().enumerate() {
        let base = 4 + i * 8;
        data[base..base + 4].copy_from_slice(&off.to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&sz.to_le_bytes());
    }
    data
}

fn append_lump(data: &mut Vec<u8>, content: &[u8]) -> (u32, u32) {
    let offset = data.len() as u32;
    let size = content.len() as u32;
    data.extend_from_slice(content);
    (offset, size)
}

#[test]
fn golden_bsp29_minimal_worldspawn() {
    let mut data = make_bsp29_header(&[]);
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let (entity_off, entity_sz) = append_lump(&mut data, entities);

    // Patch lump table
    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&entity_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&entity_sz.to_le_bytes());

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.profile, profile::BspProfile::Bsp29);
    assert_eq!(world.entities.len(), 1);
    assert!(world.worldspawn().is_some());
    assert_eq!(world.planes.len(), 0);
    assert_eq!(world.vertices.len(), 0);
}

#[test]
fn golden_bsp29_with_planes_and_vertices() {
    let mut data = make_bsp29_header(&[]);

    // Entities
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let (e_off, e_sz) = append_lump(&mut data, entities);

    // Planes: 2 planes
    let mut plane_data = Vec::new();
    for i in 0..2u8 {
        plane_data.extend_from_slice(&0.0f32.to_le_bytes());
        plane_data.extend_from_slice(&0.0f32.to_le_bytes());
        plane_data.extend_from_slice(&1.0f32.to_le_bytes());
        plane_data.extend_from_slice(&(i as f32 * 100.0).to_le_bytes());
        plane_data.extend_from_slice(&0i32.to_le_bytes());
    }
    let (p_off, p_sz) = append_lump(&mut data, &plane_data);

    // Vertices: 3 vertices
    let mut vert_data = Vec::new();
    for i in 0..3 {
        vert_data.extend_from_slice(&(i as f32).to_le_bytes());
        vert_data.extend_from_slice(&(i as f32 * 2.0).to_le_bytes());
        vert_data.extend_from_slice(&(i as f32 * 3.0).to_le_bytes());
    }
    let (v_off, v_sz) = append_lump(&mut data, &vert_data);

    // Patch lump table
    let lump_data: [(usize, u32, u32); 3] = [(0, e_off, e_sz), (1, p_off, p_sz), (3, v_off, v_sz)];
    for &(idx, off, sz) in &lump_data {
        let base = 4 + idx * 8;
        data[base..base + 4].copy_from_slice(&off.to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&sz.to_le_bytes());
    }

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.planes.len(), 2);
    assert_eq!(world.vertices.len(), 3);
    assert!((world.planes[0].dist - 0.0).abs() < 0.01);
    assert!((world.planes[1].dist - 100.0).abs() < 0.01);
    assert!((world.vertices[0].x - 0.0).abs() < 0.01);
    assert!((world.vertices[2].z - 6.0).abs() < 0.01);
}

#[test]
fn golden_bsp29_with_light_entity() {
    let mut data = make_bsp29_header(&[]);
    let entities = b"{\"classname\" \"worldspawn\"}\0{\"classname\" \"light\" \"origin\" \"0 0 64\" \"light\" \"200\"}\0";
    let (e_off, e_sz) = append_lump(&mut data, entities);

    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&e_sz.to_le_bytes());

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.entities.len(), 2);
    assert_eq!(world.entities[0].class, entities::EntityClass::Worldspawn);
    assert_eq!(world.entities[1].class, entities::EntityClass::Light);
    assert_eq!(
        entities::get_singleton(&world.entities[1], "light"),
        Some("200")
    );
}

#[test]
fn golden_bsp29_with_duplicate_keys() {
    let mut data = make_bsp29_header(&[]);
    let entities = b"{\"classname\" \"light\" \"light\" \"100\" \"light\" \"200\"}\0";
    let (e_off, e_sz) = append_lump(&mut data, entities);

    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&e_sz.to_le_bytes());

    let options = LoadOptions::default();
    let world = BspLoader::load(&data, &options).unwrap();
    // Last value wins
    assert_eq!(
        entities::get_singleton(&world.entities[0], "light"),
        Some("200")
    );
    // Duplicate key diagnostic present
    assert!(world
        .diagnostics
        .iter()
        .any(|d| d.code == DiagnosticCode::EntityDuplicateKey));
}

#[test]
fn golden_palette_loading() {
    let mut data = make_bsp29_header(&[]);
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let (e_off, e_sz) = append_lump(&mut data, entities);

    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&e_sz.to_le_bytes());

    let palette: Vec<u8> = (0..768).map(|i| (i % 256) as u8).collect();
    let options = LoadOptions {
        palette: Some(palette),
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&data, &options).unwrap();
    assert!(world.palette.is_some());
    let pal = world.palette.unwrap();
    // Check first and last entries
    assert_eq!(pal[0], [0, 1, 2]);
    // Check last entry
    assert_eq!(pal[255], [253, 254, 255]);
}

#[test]
fn golden_bsp29_model_references() {
    let mut data = make_bsp29_header(&[]);

    // Entities
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let (e_off, e_sz) = append_lump(&mut data, entities);

    // One plane
    let mut plane_data = Vec::new();
    plane_data.extend_from_slice(&0.0f32.to_le_bytes());
    plane_data.extend_from_slice(&0.0f32.to_le_bytes());
    plane_data.extend_from_slice(&1.0f32.to_le_bytes());
    plane_data.extend_from_slice(&0.0f32.to_le_bytes());
    plane_data.extend_from_slice(&0i32.to_le_bytes());
    let (p_off, p_sz) = append_lump(&mut data, &plane_data);

    // One model (worldspawn model 0): 64 bytes
    let mut model_data = Vec::new();
    // mins (12), maxs (12), origin (12)
    for _ in 0..3 {
        model_data.extend_from_slice(&0.0f32.to_le_bytes()); // mins
    }
    for _ in 0..3 {
        model_data.extend_from_slice(&100.0f32.to_le_bytes()); // maxs
    }
    for _ in 0..3 {
        model_data.extend_from_slice(&0.0f32.to_le_bytes()); // origin
    }
    // headnode[4] = -1 (no hull)
    for _ in 0..4 {
        model_data.extend_from_slice(&(-1i32).to_le_bytes());
    }
    model_data.extend_from_slice(&0i32.to_le_bytes()); // visleafs
    model_data.extend_from_slice(&0i32.to_le_bytes()); // face_id: i32
    model_data.extend_from_slice(&0i32.to_le_bytes()); // face_num: i32
    let (m_off, m_sz) = append_lump(&mut data, &model_data);

    let lump_data: [(usize, u32, u32); 3] = [(0, e_off, e_sz), (1, p_off, p_sz), (14, m_off, m_sz)];
    for &(idx, off, sz) in &lump_data {
        let base = 4 + idx * 8;
        data[base..base + 4].copy_from_slice(&off.to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&sz.to_le_bytes());
    }

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.models.len(), 1);
    assert!((world.models[0].maxs.x - 100.0).abs() < 0.01);
}

#[test]
fn golden_deterministic_entity_ordering() {
    // Entities are parsed in source order and must maintain that ordering
    let mut data = make_bsp29_header(&[]);
    let entities = concat!(
        "{\"classname\" \"worldspawn\"}\0",
        "{\"classname\" \"light\" \"origin\" \"0 0 0\"}\0",
        "{\"classname\" \"info_player_start\" \"origin\" \"0 128 0\"}\0",
        "{\"classname\" \"trigger_once\" \"target\" \"door1\"}\0",
    );
    let (e_off, e_sz) = append_lump(&mut data, entities.as_bytes());

    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&e_sz.to_le_bytes());

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.entities.len(), 4);
    assert_eq!(world.entities[0].source_index, 0);
    assert_eq!(world.entities[1].source_index, 1);
    assert_eq!(world.entities[2].source_index, 2);
    assert_eq!(world.entities[3].source_index, 3);
    assert_eq!(world.entities[0].class, entities::EntityClass::Worldspawn);
    assert_eq!(world.entities[1].class, entities::EntityClass::Light);
    assert_eq!(world.entities[2].class, entities::EntityClass::SpawnMarker);
    assert_eq!(world.entities[3].class, entities::EntityClass::Trigger);
}

#[test]
fn golden_strict_mode_rejects_unknown_extension() {
    let mut data = make_bsp29_header(&[]);
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let (e_off, e_sz) = append_lump(&mut data, entities);
    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&e_sz.to_le_bytes());

    // Append a BSPX directory with an unknown extension.
    // The entry's data must be after the standard lumps and BSPX directory.
    // We place dummy data at the end for the entry to reference.
    let bspx_data_start = data.len() as u32;
    data.extend_from_slice(&[0u8; 10]); // dummy extension data

    let mut name_bytes = [0u8; 24];
    name_bytes[..11].copy_from_slice(b"MYEXTENSION");
    data.extend_from_slice(&name_bytes);
    data.extend_from_slice(&bspx_data_start.to_le_bytes()); // offset (after standard lumps)
    data.extend_from_slice(&10u32.to_le_bytes()); // size
    data.extend_from_slice(&1u32.to_le_bytes()); // count
    data.extend_from_slice(&bsp::bspx::BSPX_MAGIC);

    // In dev mode, unknown extension is a warning
    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert!(world
        .diagnostics
        .iter()
        .any(|d| d.code == DiagnosticCode::UnsupportedExtension));

    // In strict mode, unknown extensions are fatal unsupported compatibility.
    let options = LoadOptions {
        strict: true,
        ..LoadOptions::default()
    };
    let r = BspLoader::load(&data, &options);
    assert!(r.is_err());
    let report = r.unwrap_err();
    assert_eq!(report.code, DiagnosticCode::UnsupportedExtension);
    assert_eq!(report.severity, Severity::Error);
}

#[test]
fn golden_lit_validation() {
    let mut data = make_bsp29_header(&[]);
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let (e_off, e_sz) = append_lump(&mut data, entities);

    // Lightmaps: 30 bytes (10 luxels * 1 byte each)
    let lightmaps = vec![128u8; 30];
    let (lm_off, lm_sz) = append_lump(&mut data, &lightmaps);

    let lump_data: [(usize, u32, u32); 2] = [(0, e_off, e_sz), (8, lm_off, lm_sz)];
    for &(idx, off, sz) in &lump_data {
        let base = 4 + idx * 8;
        data[base..base + 4].copy_from_slice(&off.to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&sz.to_le_bytes());
    }

    // Valid .lit: QLIT + version 1 + 90 bytes RGB (30 luxels * 3)
    let mut lit = Vec::new();
    lit.extend_from_slice(b"QLIT");
    lit.extend_from_slice(&1u32.to_le_bytes());
    lit.extend_from_slice(&vec![0u8; 90]);

    let options = LoadOptions {
        lit_data: Some(lit),
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&data, &options).unwrap();
    assert_eq!(
        world.colored_light_source,
        companions::ColoredLightSource::LitFile
    );
}

#[test]
fn golden_lit_mismatch_diagnosed() {
    let mut data = make_bsp29_header(&[]);
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let (e_off, e_sz) = append_lump(&mut data, entities);

    // Lightmaps: 30 bytes
    let lightmaps = vec![128u8; 30];
    let (lm_off, lm_sz) = append_lump(&mut data, &lightmaps);

    let lump_data: [(usize, u32, u32); 2] = [(0, e_off, e_sz), (8, lm_off, lm_sz)];
    for &(idx, off, sz) in &lump_data {
        let base = 4 + idx * 8;
        data[base..base + 4].copy_from_slice(&off.to_le_bytes());
        data[base + 4..base + 8].copy_from_slice(&sz.to_le_bytes());
    }

    // Mismatched .lit: 93 bytes RGB (31 luxels, not 30)
    let mut lit = Vec::new();
    lit.extend_from_slice(b"QLIT");
    lit.extend_from_slice(&1u32.to_le_bytes());
    lit.extend_from_slice(&vec![0u8; 93]);

    let options = LoadOptions {
        lit_data: Some(lit),
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&data, &options).unwrap();
    // Should fall back to monochrome and diagnose the mismatch
    assert_eq!(
        world.colored_light_source,
        companions::ColoredLightSource::Monochrome
    );
    assert!(world
        .diagnostics
        .iter()
        .any(|d| d.code == DiagnosticCode::CompanionContentMismatch));
}

#[test]
fn golden_wad_loading() {
    let mut data = make_bsp29_header(&[]);
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let (e_off, e_sz) = append_lump(&mut data, entities);

    let base = 4 + 0 * 8;
    data[base..base + 4].copy_from_slice(&e_off.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&e_sz.to_le_bytes());

    // Build a WAD2 archive with one texture
    let mut wad = Vec::new();
    wad.extend_from_slice(b"WAD2");
    wad.extend_from_slice(&1u32.to_le_bytes()); // num_entries
    let dir_offset: u32 = 12 + 100; // after header + dummy texture data
    wad.extend_from_slice(&dir_offset.to_le_bytes());
    // Dummy texture data
    wad.extend_from_slice(&vec![0u8; 100]);
    // Directory entry
    wad.extend_from_slice(&12u32.to_le_bytes()); // offset
    wad.extend_from_slice(&100u32.to_le_bytes()); // disk_size
    wad.extend_from_slice(&100u32.to_le_bytes()); // size
    wad.push(0x44); // type = miptex
    wad.push(0); // compression
    wad.extend_from_slice(&[0u8; 2]); // padding
    let mut name_bytes = [0u8; 16];
    name_bytes[..7].copy_from_slice(b"TESTTEX");
    wad.extend_from_slice(&name_bytes);

    let options = LoadOptions {
        wad_archives: vec![("test.wad".into(), wad)],
        ..LoadOptions::default()
    };
    let world = BspLoader::load(&data, &options).unwrap();
    assert_eq!(world.wad_archives.len(), 1);
    assert_eq!(world.wad_archives[0].1.entries.len(), 1);
    assert_eq!(world.wad_archives[0].1.entries[0].name, "TESTTEX");
}
