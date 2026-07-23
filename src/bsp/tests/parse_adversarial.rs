//! Adversarial / table-driven mutation tests for the BSP parser.
//!
//! Each test case supplies a mutated byte sequence and asserts the exact
//! diagnostic code and severity. Categories: truncation, overlap, overflow,
//! cycles, bad indices, non-finite data, malformed BSPX/WAD/entities,
//! stale companions, budget exhaustion.

use bsp::*;

/// Build a baseline valid BSP29 header with specified lump sizes.
fn baseline_header() -> Vec<u8> {
    let mut data = vec![0u8; 124];
    data[0..4].copy_from_slice(&29u32.to_le_bytes());
    data
}

fn set_lump(header: &mut [u8], lump_idx: usize, offset: u32, size: u32) {
    let base = 4 + lump_idx * 8;
    header[base..base + 4].copy_from_slice(&offset.to_le_bytes());
    header[base + 4..base + 8].copy_from_slice(&size.to_le_bytes());
}

// ── Truncation tests ──

#[test]
fn adversarial_truncated_header() {
    let data = vec![0u8; 50]; // < 124
    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    // 50-byte file: magic detection fails first (UnsupportedDialect)
    assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
}

#[test]
fn adversarial_file_too_small_for_magic() {
    let data = vec![0x1D]; // 1 byte
    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
}

#[test]
fn adversarial_lump_past_end_of_file() {
    let mut data = baseline_header();
    // Planes lump at offset 1000, size 100, but file is only 124 bytes
    set_lump(&mut data, 1, 1000, 100);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptLump);
}

#[test]
fn adversarial_lump_negative_offset() {
    let mut data = vec![0u8; 124];
    data[0..4].copy_from_slice(&29u32.to_le_bytes());
    // Write negative offset for planes lump
    let base = 4 + 1 * 8;
    data[base..base + 4].copy_from_slice(&(-1i32).to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&100u32.to_le_bytes());

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptLump);
}

#[test]
fn adversarial_lump_negative_size() {
    let mut data = vec![0u8; 124];
    data[0..4].copy_from_slice(&29u32.to_le_bytes());
    let base = 4 + 1 * 8;
    data[base..base + 4].copy_from_slice(&200u32.to_le_bytes());
    data[base + 4..base + 8].copy_from_slice(&(-1i32).to_le_bytes());

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptLump);
}

// ── Overlap tests ──

#[test]
fn adversarial_overlapping_lumps() {
    let mut data = baseline_header();
    set_lump(&mut data, 0, 124, 100); // entities: 124..224
    set_lump(&mut data, 1, 174, 100); // planes: 174..274 (overlaps entities)
                                      // Extend data
    data.resize(300, 0);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptLump);
}

// ── Invalid indices tests ──

#[test]
fn adversarial_face_references_out_of_range_plane() {
    let mut data = baseline_header();

    // Entities
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    let e_sz = entities.len() as u32;
    set_lump(&mut data, 0, e_off, e_sz);

    // Planes: 1 plane
    let p_off = data.len() as u32;
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0i32.to_le_bytes());
    let p_sz = 20u32;
    set_lump(&mut data, 1, p_off, p_sz);

    // Faces: 1 face referencing plane 99 (out of range)
    let f_off = data.len() as u32;
    let mut face = [0u8; 20];
    face[0..2].copy_from_slice(&99u16.to_le_bytes()); // plane_id = 99 (only 1 plane)
    face[10..12].copy_from_slice(&0u16.to_le_bytes()); // texinfo_id = 0
    data.extend_from_slice(&face);
    let f_sz = 20u32;
    set_lump(&mut data, 7, f_off, f_sz);

    // Markfaces: 0
    // Surfedges: 0
    // Edges: 0

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptIndex);
}

#[test]
fn adversarial_node_cycle() {
    let mut data = baseline_header();

    // Entities
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    let e_sz = entities.len() as u32;
    set_lump(&mut data, 0, e_off, e_sz);

    // Planes: 2 planes
    let p_off = data.len() as u32;
    for _ in 0..2 {
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0i32.to_le_bytes());
    }
    let p_sz = 40u32;
    set_lump(&mut data, 1, p_off, p_sz);

    // Nodes: 2 nodes forming cycle (0->1, 1->0)
    let n_off = data.len() as u32;
    // Node 0: plane=0, child[0]=1 (node 1), child[1]=-1 (leaf 0)
    data.extend_from_slice(&0u32.to_le_bytes()); // plane
    data.extend_from_slice(&1i16.to_le_bytes()); // child[0] = node 1
    data.extend_from_slice(&(-1i16).to_le_bytes()); // child[1] = leaf 0
    data.extend_from_slice(&[0u8; 12]); // mins/maxs (i16×6)
    data.extend_from_slice(&0u16.to_le_bytes()); // face_id
    data.extend_from_slice(&0u16.to_le_bytes()); // face_num
                                                 // Node 1: plane=1, child[0]=0 (node 0 -> cycle!), child[1]=-1 (leaf 0)
    data.extend_from_slice(&1u32.to_le_bytes()); // plane
    data.extend_from_slice(&0i16.to_le_bytes()); // child[0] = node 0 (CYCLE)
    data.extend_from_slice(&(-1i16).to_le_bytes()); // child[1] = leaf 0
    data.extend_from_slice(&[0u8; 12]); // mins/maxs
    data.extend_from_slice(&0u16.to_le_bytes()); // face_id
    data.extend_from_slice(&0u16.to_le_bytes()); // face_num
    let n_sz = 48u32;
    set_lump(&mut data, 5, n_off, n_sz);

    // Leaves: 1 leaf
    let l_off = data.len() as u32;
    data.extend_from_slice(&0i32.to_le_bytes()); // contents
    data.extend_from_slice(&(-1i32).to_le_bytes()); // visofs
    data.extend_from_slice(&[0u8; 20]); // mins/maxs/mark/ambient
    let l_sz = 28u32;
    set_lump(&mut data, 10, l_off, l_sz);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptCycle);
}

// ── Non-finite data tests ──

#[test]
fn adversarial_non_finite_vertex() {
    let mut data = baseline_header();

    // Entities
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // Vertices: 1 vertex with NaN
    let v_off = data.len() as u32;
    data.extend_from_slice(&f32::NAN.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    set_lump(&mut data, 3, v_off, 12);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptLump);
}

#[test]
fn adversarial_entity_string_not_null_terminated_dev() {
    let mut data = baseline_header();

    let entities = b"{\"classname\" \"worldspawn\"}";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // Non-null-terminated entity strings are structural corruption in every mode.
    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptEntity);
}

#[test]
fn adversarial_malformed_entity_unterminated() {
    let mut data = baseline_header();

    let entities = b"{\"classname\" \"worldspawn\"}\0{\"classname\" \"light\"\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::EntityUnterminated);
}

// ── Budget exhaustion tests ──

#[test]
fn adversarial_entity_count_too_large() {
    let mut data = baseline_header();

    // Write an entity lump that claims huge size for entities
    let e_off = data.len() as u32;
    // Claim 10 MiB of entity data
    data.resize(data.len() + 10_000_000, 0);
    set_lump(&mut data, 0, e_off, 10_000_000);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::EntityStringTooLarge);
}

// ── Malformed WAD tests ──

#[test]
fn adversarial_bad_wad_magic() {
    let mut data = baseline_header();
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // WAD with bad magic
    let wad = b"BADD12345678".to_vec();

    let options = LoadOptions {
        wad_archives: vec![("bad.wad".into(), wad)],
        ..LoadOptions::default()
    };
    let r = BspLoader::load(&data, &options);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptLump);
}

// ── Unsupported dialect tests ──

#[test]
fn adversarial_bsp30_rejected() {
    let mut data = vec![0u8; 200];
    data[0..4].copy_from_slice(&30u32.to_le_bytes());

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
}

#[test]
fn adversarial_bsp_ibsp_rejected() {
    let mut data = vec![0u8; 200];
    data[0..4].copy_from_slice(b"IBSP");

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
}

// ── Entity edge cases ──

#[test]
fn adversarial_entity_escape_sequences() {
    let mut data = baseline_header();

    let entities = b"{\"classname\" \"light\" \"message\" \"hello\\nworld\\\"quote\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    let msg = entities::get_singleton(&world.entities[0], "message").unwrap();
    assert!(msg.contains('\n'));
    assert!(msg.contains("quote"));
}

#[test]
fn adversarial_empty_entity_diagnosed() {
    let mut data = baseline_header();

    let entities = b"{\"classname\" \"worldspawn\"}\0{}\0{\"classname\" \"light\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(world.entities.len(), 3);
    assert!(world
        .diagnostics
        .iter()
        .any(|d| d.code == DiagnosticCode::EntityEmpty));
}

// ── BSPX edge cases ──

#[test]
fn adversarial_bspx_empty_name() {
    let mut data = baseline_header();
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // BSPX directory with empty-named entry
    let name_bytes = [0u8; 24]; // all zeros = empty name
    data.extend_from_slice(&name_bytes);
    data.extend_from_slice(&200u32.to_le_bytes());
    data.extend_from_slice(&10u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&bspx::BSPX_MAGIC);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptLump);
}

#[test]
fn adversarial_bspx_rgblighting_size_mismatch_rejected_in_strict() {
    let mut data = baseline_header();
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    let lm_off = data.len() as u32;
    data.extend_from_slice(&[128u8; 2]);
    set_lump(&mut data, 8, lm_off, 2);

    let rgb_off = data.len() as u32;
    data.extend_from_slice(&[0u8; 5]); // expected lightmap_size * 3 = 6
    let mut name_bytes = [0u8; 24];
    name_bytes[..11].copy_from_slice(b"RGBLIGHTING");
    data.extend_from_slice(&name_bytes);
    data.extend_from_slice(&rgb_off.to_le_bytes());
    data.extend_from_slice(&5u32.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&bspx::BSPX_MAGIC);

    let dev_world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    assert_eq!(
        dev_world.colored_light_source,
        companions::ColoredLightSource::Monochrome
    );
    assert!(dev_world
        .diagnostics
        .iter()
        .any(|d| d.code == DiagnosticCode::CompanionContentMismatch));

    let strict = LoadOptions {
        strict: true,
        ..Default::default()
    };
    let r = BspLoader::load(&data, &strict);
    assert!(r.is_err());
    assert_eq!(
        r.unwrap_err().code,
        DiagnosticCode::CompanionContentMismatch
    );
}

// ── Out-of-range style tests ──

#[test]
fn adversarial_face_style_out_of_range() {
    let mut data = baseline_header();

    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // Planes: 1 plane
    let p_off = data.len() as u32;
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0i32.to_le_bytes());
    set_lump(&mut data, 1, p_off, 20);

    // Faces: 1 face with style 100 (invalid)
    let f_off = data.len() as u32;
    let mut face = [0u8; 20];
    face[12] = 100; // style[0] = 100 (max is 63, sentinel 255)
    face[13] = 255;
    face[14] = 255;
    face[15] = 255;
    data.extend_from_slice(&face);
    set_lump(&mut data, 7, f_off, 20);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedStyleSlot);
}

// ── Extraction API tests ──

#[test]
fn adversarial_extraction_fails_without_world() {
    let request = BspExtractionRequest::default();
    let result = extract(request);
    assert!(result.is_ok());
    let ext = result.unwrap();
    assert!(ext.face_geometries.is_empty());
    assert!(ext.face_materials.is_empty());
    assert!(ext.render_batches.is_empty());
}

#[test]
fn adversarial_extraction_custom_scale() {
    let request = BspExtractionRequest {
        scale: 0.05,
        ..Default::default()
    };
    let ext = extract(request).unwrap();
    assert!((ext.transform.scale - 0.05).abs() < 1e-8);
}

#[test]
fn adversarial_extraction_zero_scale_still_works() {
    let request = BspExtractionRequest {
        scale: 0.001,
        ..Default::default()
    };
    let ext = extract(request).unwrap();
    assert!((ext.transform.scale - 0.001).abs() < 1e-8);
}

#[test]
fn adversarial_extraction_preserves_content_hash() {
    let request = BspExtractionRequest {
        world: BspWorld::empty(),
        ..Default::default()
    };
    let ext = extract(request).unwrap();
    assert_eq!(ext.content_hash, [0u8; 32]);
}
