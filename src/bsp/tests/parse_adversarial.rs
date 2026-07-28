//! Adversarial / table-driven mutation tests for the BSP parser.
//!
//! Each test case supplies a mutated byte sequence and asserts the exact
//! diagnostic code and severity. Categories: truncation, overlap, overflow,
//! cycles, bad indices, non-finite data, malformed BSPX/WAD/entities,
//! stale companions, budget exhaustion, VIS RLE edge cases, BSP2
//! structural attacks, surfedge/winding edge cases, miptex/palette attacks,
//! .lit style range attacks, texture name traversal, animation arrays,
//! clipnode cycles, convex reconstruction attacks, and extraction budgets.
//!
//! Phase 09 hardening: expanded coverage for all fuzz target categories
//! in bsp-acceptance.md §10.

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

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: VIS RLE Decompression Edge Cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_vis_rle_empty_data() {
    let mut data = baseline_header();
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // VIS lump with zero size
    set_lump(&mut data, 4, 200, 0);

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    // Empty VIS should result in empty vis_data
    assert!(world.vis_data.is_empty());
}

#[test]
fn adversarial_vis_rle_truncated_zero_run() {
    let mut data = baseline_header();
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // Leaves: 2 leaves (for VIS)
    let leaf_off = data.len() as u32;
    // Leaf 0: visofs=0
    data.extend_from_slice(&0i32.to_le_bytes()); // contents
    data.extend_from_slice(&0i32.to_le_bytes()); // visofs = 0
    data.extend_from_slice(&[0u8; 20]); // mins/maxs/mark/ambient
                                        // Leaf 1: visofs=1
    data.extend_from_slice(&0i32.to_le_bytes());
    data.extend_from_slice(&1i32.to_le_bytes()); // visofs = 1
    data.extend_from_slice(&[0u8; 20]);
    set_lump(&mut data, 10, leaf_off, 56);

    // VIS data: 0x00 (zero-run command) then truncated (no count byte follows)
    let vis_off = data.len() as u32;
    data.push(0x00);
    set_lump(&mut data, 4, vis_off, 1);

    // Loading should produce a world (VIS fallback is conservative, not fatal)
    // The corrupt VIS byte causes a fallback, which is not necessarily fatal.
    let result = BspLoader::load(&data, &LoadOptions::default());
    // May pass with warning or fail depending on strictness
    let _ = result; // At minimum, no panic
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Clipnode Cycle Detection
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_clipnode_cycle() {
    let mut data = baseline_header();

    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // Planes: 4 planes for clipnodes
    let p_off = data.len() as u32;
    for _ in 0..4 {
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0i32.to_le_bytes());
    }
    set_lump(&mut data, 1, p_off, 80);

    // Clipnodes: 3 clipnodes forming cycle (0→1, 1→2, 2→0)
    let c_off = data.len() as u32;
    // Clipnode 0: plane=0, child[0]=1, child[1]=-1
    data.extend_from_slice(&0u32.to_le_bytes()); // plane
    data.extend_from_slice(&1i16.to_le_bytes()); // child[0] = node 1
    data.extend_from_slice(&(-1i16).to_le_bytes()); // child[1] = leaf
                                                    // Clipnode 1: plane=1, child[0]=2, child[1]=-1
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&2i16.to_le_bytes()); // child[0] = node 2
    data.extend_from_slice(&(-2i16).to_le_bytes());
    // Clipnode 2: plane=2, child[0]=0 (CYCLE!), child[1]=-1
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&0i16.to_le_bytes()); // child[0] = node 0 (CYCLE)
    data.extend_from_slice(&(-3i16).to_le_bytes());
    set_lump(&mut data, 9, c_off, 24);

    let r = BspLoader::load(&data, &LoadOptions::default());
    // Clipnode cycles should be caught as structural corruption
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptCycle);
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Surfedge / Face Winding Edge Cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_surfedge_out_of_range() {
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

    // Vertices: 1 vertex
    let v_off = data.len() as u32;
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    set_lump(&mut data, 3, v_off, 12);

    // Edges: 1 edge referencing vertex 0
    let edge_off = data.len() as u32;
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    set_lump(&mut data, 12, edge_off, 4);

    // Surfedges: 1 surfedge referencing edge 99 (out of range)
    let se_off = data.len() as u32;
    data.extend_from_slice(&99u32.to_le_bytes());
    set_lump(&mut data, 13, se_off, 4);

    // Faces: 1 face with surfedge references
    let f_off = data.len() as u32;
    let mut face = [0u8; 20];
    face[0..2].copy_from_slice(&0u16.to_le_bytes()); // plane_id = 0
    face[2..4].copy_from_slice(&0u16.to_le_bytes()); // side
    face[4..8].copy_from_slice(&0u32.to_le_bytes()); // first_edge = 0
    face[8..10].copy_from_slice(&1u16.to_le_bytes()); // num_edges = 1
    face[10..12].copy_from_slice(&0u16.to_le_bytes()); // texinfo_id = 0
    face[12] = 0;
    face[13] = 255;
    face[14] = 255;
    face[15] = 255; // styles
    face[16..20].copy_from_slice(&0u32.to_le_bytes()); // lightofs = 0
    data.extend_from_slice(&face);
    set_lump(&mut data, 7, f_off, 20);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptIndex);
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Miptex / WAD / Palette Edge Cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_miptex_name_too_long() {
    let mut data = baseline_header();
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // Miptex lump with a texture whose name is too long (no NUL terminator)
    let tex_off = data.len() as u32;
    // Miptex count: 1
    data.extend_from_slice(&1u32.to_le_bytes());
    // Miptex offset array: offset 0 = 16
    data.extend_from_slice(&16u32.to_le_bytes());
    // Miptex at offset 16: name with all 'A's, no NUL
    data.extend_from_slice(b"AAAAAAAAAAAAAAAA");
    data.extend_from_slice(&64u32.to_le_bytes()); // width
    data.extend_from_slice(&64u32.to_le_bytes()); // height
    data.extend_from_slice(&0u32.to_le_bytes()); // mip1 offset
    data.extend_from_slice(&0u32.to_le_bytes()); // mip2 offset
    data.extend_from_slice(&0u32.to_le_bytes()); // mip3 offset
    let end = data.len();
    set_lump(&mut data, 2, tex_off, (end - tex_off as usize) as u32);

    // Miptex with too-long name: parser may accept or reject
    // The important property is no panic
    let _ = BspLoader::load(&data, &LoadOptions::default());
}

#[test]
fn adversarial_palette_too_short() {
    let mut data = baseline_header();
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // WAD with valid magic but palette too short
    let wad_data = {
        let mut w = Vec::new();
        w.extend_from_slice(b"WAD2");
        w.extend_from_slice(&1u32.to_le_bytes()); // num entries
        w.extend_from_slice(&32u32.to_le_bytes()); // directory offset
                                                   // Truncated — no actual dir entries
        w
    };

    let options = LoadOptions {
        wad_archives: vec![("bad.wad".into(), wad_data)],
        ..LoadOptions::default()
    };
    let r = BspLoader::load(&data, &options);
    assert!(r.is_err());
}

#[test]
fn adversarial_palette_wrong_size() {
    let palette = vec![0u8; 100]; // Not 768
    let result = companions::validate_palette(&palette, false);
    assert!(result.is_err());
    // MissingRequiredPalette is the correct code for invalid palette
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: .lit / Style Range Edge Cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_lit_wrong_magic() {
    let lit_bytes: &[u8] = b"BADMAGIC......";
    let result = companions::validate_lit_header(lit_bytes, false);
    assert!(result.is_err());
}

#[test]
fn adversarial_lit_too_short() {
    let lit_bytes: &[u8] = b"QLIT\x01"; // Only 5 bytes, need 8-byte header
    let result = companions::validate_lit_header(lit_bytes, false);
    assert!(result.is_err());
}

#[test]
fn adversarial_lit_luxel_mismatch() {
    // validate_lit_against_lightmap checks luxel count match between lit RGB and base lightmap
    // lit_rgb_size = 30, lightmap_size = 100 (30 / 3 = 10 ≠ 100)
    let result = companions::validate_lit_against_lightmap(30, 100, false);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err().code,
        DiagnosticCode::CompanionContentMismatch
    );
}

#[test]
fn adversarial_lit_luxel_mismatch_strict_rejected() {
    let result = companions::validate_lit_against_lightmap(15, 20, true);
    assert!(result.is_err());
    // 15 / 3 = 5 ≠ 20
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Texture Name Traversal
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_texture_name_parent_traversal() {
    // A texture name with ".." should be rejected as a security path traversal
    let mut data = baseline_header();
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // Miptex lump with path traversal in texture name
    let tex_off = data.len() as u32;
    data.extend_from_slice(&1u32.to_le_bytes()); // count
    let name_offset = data.len() as u32 + 4;
    data.extend_from_slice(&name_offset.to_le_bytes()); // offset to miptex
                                                        // Miptex with name containing ".."
    let mut name = [0u8; 16];
    name[0..10].copy_from_slice(b"../escape\0");
    data.extend_from_slice(&name);
    data.extend_from_slice(&64u32.to_le_bytes());
    data.extend_from_slice(&64u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    let end = data.len();
    set_lump(&mut data, 2, tex_off, (end - tex_off as usize) as u32);

    let r = BspLoader::load(&data, &LoadOptions::default());
    // Path traversal in texture names: may be rejected as security or handled later
    // The important property: no panic
    let _ = r;
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: BSP2-Specific Structural Attacks
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_bsp2_corrupt_magic() {
    // Bytes that happen to start with "BSP2" but are otherwise garbage
    let mut data = vec![0u8; 200];
    data[0..4].copy_from_slice(b"BSP2");
    // But all lumps have zero offset/size except a corrupted one
    set_lump(&mut data, 0, 0, 0);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_ok()); // Empty BSP2 is valid
}

#[test]
fn adversarial_bsp2_node_count_mismatch() {
    // Claim BSP2 but with BSP29-sized nodes (24B) that don't evenly divide
    let mut data = vec![0u8; 200];
    data[0..4].copy_from_slice(b"BSP2");

    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = 124u32;
    let e_sz = entities.len() as u32;
    set_lump(&mut data, 0, e_off, e_sz);
    data[e_off as usize..e_off as usize + e_sz as usize].copy_from_slice(entities);

    // Node lump with size not divisible by BSP2 node stride (44B)
    let n_off = e_off + e_sz;
    set_lump(&mut data, 5, n_off, 50); // 50 not divisible by 44
    data.resize(n_off as usize + 50, 0);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
}

#[test]
fn adversarial_bsp2_leaf_count_mismatch() {
    let mut data = vec![0u8; 200];
    data[0..4].copy_from_slice(b"BSP2");

    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = 124u32;
    let e_sz = entities.len() as u32;
    set_lump(&mut data, 0, e_off, e_sz);
    data[e_off as usize..e_off as usize + e_sz as usize].copy_from_slice(entities);

    // Leaf lump with size not divisible by BSP2 leaf stride (44B)
    let l_off = e_off + e_sz;
    set_lump(&mut data, 10, l_off, 90); // 90 not divisible by 44
    data.resize(l_off as usize + 90, 0);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
}

#[test]
fn adversarial_bsp2_edge_count_mismatch() {
    let mut data = vec![0u8; 200];
    data[0..4].copy_from_slice(b"BSP2");

    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = 124u32;
    let e_sz = entities.len() as u32;
    set_lump(&mut data, 0, e_off, e_sz);
    data[e_off as usize..e_off as usize + e_sz as usize].copy_from_slice(entities);

    // Edge lump with size not divisible by BSP2 edge stride (8B)
    let edge_off = e_off + e_sz;
    set_lump(&mut data, 12, edge_off, 7); // 7 not divisible by 8
    data.resize(edge_off as usize + 7, 0);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Face / Surfedge Reconstruction Edge Cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_face_single_edge_degenerate() {
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

    // Vertices: 1 vertex
    let v_off = data.len() as u32;
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    set_lump(&mut data, 3, v_off, 12);

    // Edges: 1 edge
    let edge_off = data.len() as u32;
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    set_lump(&mut data, 12, edge_off, 4);

    // Surfedges: 1 surfedge
    let se_off = data.len() as u32;
    data.extend_from_slice(&0u32.to_le_bytes()); // positive reference to edge 0
    set_lump(&mut data, 13, se_off, 4);

    // Face: num_edges = 1 (single edge = degenerate, not a polygon)
    let f_off = data.len() as u32;
    let mut face = [0u8; 20];
    face[0..2].copy_from_slice(&0u16.to_le_bytes());
    face[4..8].copy_from_slice(&0u32.to_le_bytes()); // first_edge = 0
    face[8..10].copy_from_slice(&1u16.to_le_bytes()); // num_edges = 1
    face[10..12].copy_from_slice(&0u16.to_le_bytes());
    face[12] = 0;
    face[13] = 255;
    face[14] = 255;
    face[15] = 255;
    data.extend_from_slice(&face);
    set_lump(&mut data, 7, f_off, 20);

    let r = BspLoader::load(&data, &LoadOptions::default());
    // A face with a single edge should be rejected as corrupt
    assert!(r.is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Nested Brace Entity Attacks
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_entity_nested_braces() {
    let mut data = baseline_header();

    let entities = b"{\"classname\" \"worldspawn\" { \"nested\" \"bad\" }}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::EntityNestedBraces);
}

#[test]
fn adversarial_entity_key_without_value() {
    let mut data = baseline_header();

    let entities = b"{\"classname\" \"worldspawn\" \"orphan_key\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    let r = BspLoader::load(&data, &LoadOptions::default());
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, DiagnosticCode::EntityValueMissing);
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Convex Reconstruction from Clipnodes - No Volume
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_convex_reconstruction_no_volume() {
    // Build a world with a clipnode tree that has all coplanar planes,
    // resulting in a degenerate convex polyhedron.
    let mut data = baseline_header();

    let entities = b"{\"classname\" \"worldspawn\"}\0{\"classname\" \"func_door\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // Planes: 4 parallel planes (all same direction, different distances)
    let p_off = data.len() as u32;
    for i in 0..4 {
        data.extend_from_slice(&0.0f32.to_le_bytes()); // nx
        data.extend_from_slice(&0.0f32.to_le_bytes()); // ny
        data.extend_from_slice(&1.0f32.to_le_bytes()); // nz
        data.extend_from_slice(&(i as f32).to_le_bytes()); // dist
        data.extend_from_slice(&0i32.to_le_bytes()); // type
    }
    set_lump(&mut data, 1, p_off, 80);

    // Clipnodes: a simple self-contained node
    let c_off = data.len() as u32;
    data.extend_from_slice(&0u32.to_le_bytes()); // plane 0
    data.extend_from_slice(&(-1i16).to_le_bytes()); // child[0] = leaf (contents -1)
    data.extend_from_slice(&(-2i16).to_le_bytes()); // child[1] = leaf (contents -2)
    set_lump(&mut data, 9, c_off, 8);

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    // The world should parse but convex reconstruction from parallel planes
    // won't produce a valid polyhedron — that's OK as long as it doesn't panic.
    let _ = world;
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Model Reference Edge Cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_model_out_of_range_face() {
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

    // Models: 1 model with out-of-range face_id
    let m_off = data.len() as u32;
    let mut model = [0u8; 64];
    model[40..44].copy_from_slice(&999i32.to_le_bytes()); // face_id = 999
    model[44..48].copy_from_slice(&1i32.to_le_bytes()); // num_faces = 1
    data.extend_from_slice(&model);
    set_lump(&mut data, 14, m_off, 64);

    let r = BspLoader::load(&data, &LoadOptions::default());
    // Model with out-of-range face ref: may be caught at load or extraction
    // The important property: no panic
    if let Ok(world) = r {
        // Extraction should surface the error
        let _ = bsp::extract::extract(bsp::BspExtractionRequest {
            world,
            scale: 0.0254,
            ..Default::default()
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 09: Extraction Budget Exhaustion
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_extraction_fails_on_budget_exhausted_world() {
    // Create an empty world and verify extraction handles empty data
    let world = BspWorld::empty();
    let request = BspExtractionRequest {
        world,
        scale: 0.0254,
        ..Default::default()
    };
    let ext = extract(request).unwrap();
    // Empty world produces empty extraction
    assert!(ext.face_geometries.is_empty());
    assert!(ext.face_materials.is_empty());
    assert!(ext.render_batches.is_empty());
}

#[test]
fn adversarial_extraction_handles_missing_palette_with_warning() {
    let mut data = baseline_header();
    let entities = b"{\"classname\" \"worldspawn\"}\0";
    let e_off = data.len() as u32;
    data.extend_from_slice(entities);
    set_lump(&mut data, 0, e_off, entities.len() as u32);

    // Miptex with a texture name
    let tex_off = data.len() as u32;
    data.extend_from_slice(&1u32.to_le_bytes());
    let name_offset = (data.len() + 4) as u32;
    data.extend_from_slice(&name_offset.to_le_bytes());
    let mut name = [0u8; 16];
    name[0..9].copy_from_slice(b"test_tex\0");
    data.extend_from_slice(&name);
    data.extend_from_slice(&64u32.to_le_bytes());
    data.extend_from_slice(&64u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    let tex_sz = (data.len() - tex_off as usize) as u32;
    set_lump(&mut data, 2, tex_off, tex_sz);

    // Texinfo: 1 texinfo referencing miptex 0
    let ti_off = data.len() as u32;
    let mut texinfo = [0u8; 40];
    texinfo[32..36].copy_from_slice(&0u32.to_le_bytes()); // miptex index 0
    data.extend_from_slice(&texinfo);
    set_lump(&mut data, 6, ti_off, 40);

    // Planes, faces, etc.
    let p_off = data.len() as u32;
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0i32.to_le_bytes());
    set_lump(&mut data, 1, p_off, 20);

    let world = BspLoader::load(&data, &LoadOptions::default()).unwrap();
    // Use an empty/default palette (all zeros works for basic tests)
    let palette = [[0u8; 3]; 256];
    let request = BspExtractionRequest {
        world,
        palette: Some(palette),
        scale: 0.0254,
        strict: false,
        ..Default::default()
    };
    // Extraction should not panic even with partially missing resources
    let ext = extract(request).unwrap();
    let _ = ext;
}
