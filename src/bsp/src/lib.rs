//! Pure byte-level BSP29/BSP2/WAD/entity parser trust boundary.
//!
//! The `bsp` crate produces owned, immutable [`BspWorld`] records from raw BSP
//! bytes. It has zero renderer, Vulkan, physics, app, windowing, async, or
//! filesystem-watcher dependencies.
//!
//! # Entry Point
//!
//! ```ignore
//! use bsp::{BspLoader, LoadOptions};
//!
//! let data = std::fs::read("maps/mylevel.bsp")?;
//! let options = LoadOptions::default();
//! let world = BspLoader::load(&data, &options)?;
//! ```

pub mod bspx;
pub mod companions;
pub mod decode;
pub mod diagnostic;
pub mod entities;
pub mod limits;
pub mod lumps;
pub mod profile;
pub mod resources;
pub mod wad;
pub mod world;

use world::BspWorld;

/// Options for loading a BSP.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Whether to use strict/release severity policy.
    pub strict: bool,
    /// Palette data (768 bytes). Required for texture/lightmap decoding.
    pub palette: Option<Vec<u8>>,
    /// .lit colored light data (optional).
    pub lit_data: Option<Vec<u8>>,
    /// WAD archives to search for textures, keyed by sanitized basename.
    pub wad_archives: Vec<(String, Vec<u8>)>,
    /// Explicit texture name overrides.
    pub texture_overrides: Vec<(String, String)>,
    /// Package-relative source identity for diagnostics.
    pub source_identity: String,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            strict: false,
            palette: None,
            lit_data: None,
            wad_archives: Vec::new(),
            texture_overrides: Vec::new(),
            source_identity: String::new(),
        }
    }
}

/// The BSP loader — the only public entry point for parsing.
pub struct BspLoader;

impl BspLoader {
    /// Load and validate a BSP from raw bytes and options.
    ///
    /// Returns `Ok(BspWorld)` if all structural validation passes and the
    /// profile is recognized. Returns `Err(BspReport)` for fatal errors
    /// (structural corruption, unsupported dialect, missing required resources
    /// in strict mode).
    ///
    /// Non-fatal diagnostics are accumulated in `BspWorld::diagnostics`.
    pub fn load(data: &[u8], options: &LoadOptions) -> Result<BspWorld, BspReport> {
        // 1. Profile detection
        if data.len() < 4 {
            return Err(BspReport::fatal(
                DiagnosticCode::UnsupportedDialect,
                format!("file too small: {} bytes", data.len()),
            ));
        }
        let profile = profile::detect_profile(&data[0..4])?;

        // 2. Header validation
        profile::validate_header_size(data.len())?;

        // 3. Parse lump table
        let lumps = lumps::parse_lump_table(data)?;

        // 4. Discover BSPX directory (before standard lump validation so we know the boundary)
        let bspx_opt = bspx::discover_bspx(data)?;
        let bspx_boundary = bspx_opt.as_ref().map(|d| d.directory_offset);

        // 5. Validate standard lump ranges (non-overlapping, within file, before BSPX)
        lumps::validate_lump_ranges(&lumps, data.len(), bspx_boundary)?;

        // 6. Parse standard lumps
        let entity_raw = lumps::parse_entities(data, &lumps[lumps::LUMP_ENTITIES], options.strict)?;
        let planes = lumps::parse_planes(data, &lumps[lumps::LUMP_PLANES])?;
        let vertices = lumps::parse_vertices(data, &lumps[lumps::LUMP_VERTICES], profile)?;
        let nodes = lumps::parse_nodes(data, &lumps[lumps::LUMP_NODES], profile)?;
        let leaves = lumps::parse_leaves(data, &lumps[lumps::LUMP_LEAVES], profile)?;
        let faces = lumps::parse_faces(data, &lumps[lumps::LUMP_FACES], profile)?;
        let models = lumps::parse_models(data, &lumps[lumps::LUMP_MODELS], profile)?;
        let texinfos = lumps::parse_texinfo(data, &lumps[lumps::LUMP_TEXINFO])?;
        let edges = lumps::parse_edges(data, &lumps[lumps::LUMP_EDGES], profile)?;
        let surfedges = lumps::parse_surfedges(data, &lumps[lumps::LUMP_SURFEDGES], profile)?;
        let markfaces = lumps::parse_markfaces(data, &lumps[lumps::LUMP_MARKFACES], profile)?;
        let clipnodes = lumps::parse_clipnodes(data, &lumps[lumps::LUMP_CLIPNODES], profile)?;

        // Raw data lumps
        let miptex_data = extract_lump_bytes(data, &lumps[lumps::LUMP_MIPTEX]);
        let lightmap_data = extract_lump_bytes(data, &lumps[lumps::LUMP_LIGHTMAPS]);
        let vis_data = extract_lump_bytes(data, &lumps[lumps::LUMP_VISINFO]);

        // 7. Parse entities from raw string
        let (entities, entity_diags) = entities::parse_entities(&entity_raw, options.strict)?;

        // 8. Process BSPX extensions
        let mut bspx_diags = Vec::new();
        let mut bspx_rgb = if let Some(ref bspx_dir) = bspx_opt {
            let standard_lump_end = compute_standard_lump_end(&lumps);
            bspx_diags = bspx::validate_bspx_entries(bspx_dir, options.strict, standard_lump_end);
            bspx::read_bspx_lump(data, bspx_dir, bspx::known_names::RGBLIGHTING).map(|d| d.to_vec())
        } else {
            None
        };
        if let Some(ref rgb) = bspx_rgb {
            if let Err(report) = companions::validate_lit_against_lightmap(
                rgb.len() as u32,
                lightmap_data.len() as u32,
                options.strict,
            ) {
                bspx_diags.push(report);
                bspx_rgb = None;
            }
        }

        // 9. Process companion files
        let mut companion_diags = Vec::new();
        let palette =
            options.palette.as_ref().and_then(|data| {
                match companions::validate_palette(data, options.strict) {
                    Ok(()) => Some(resources::decode_palette(data)),
                    Err(e) => {
                        companion_diags.push(e);
                        None
                    }
                }
            });

        // .lit validation
        let lit_valid = options.lit_data.as_ref().map_or(false, |lit_data| {
            companions::validate_lit_header(lit_data, options.strict).is_ok()
        });
        let mut lit_data = options.lit_data.clone();
        if let Some(ref ld) = lit_data {
            if let Ok(rgb_size) = companions::validate_lit_header(ld, options.strict) {
                if let Err(e) = companions::validate_lit_against_lightmap(
                    rgb_size,
                    lightmap_data.len() as u32,
                    options.strict,
                ) {
                    companion_diags.push(e);
                    lit_data = None;
                }
            } else {
                lit_data = None;
            }
        }

        // Resolve colored light source
        let has_bspx_rgb = bspx_rgb.is_some();
        let has_lit = lit_data.is_some();
        let (colored_light_source, colored_diags) = companions::resolve_colored_light_source(
            has_bspx_rgb,
            has_lit,
            lit_valid,
            options.strict,
        );

        // 10. Parse WAD archives
        let mut wad_diags = Vec::new();
        let mut wad_archives = Vec::new();
        for (name, wad_bytes) in &options.wad_archives {
            match wad::parse_wad(wad_bytes.clone()) {
                Ok(archive) => wad_archives.push((name.clone(), archive)),
                Err(e) => wad_diags.push(e),
            }
        }

        // 11. Compute content hash
        let content_hash = compute_content_hash(data);

        // 12. Build the world (this validates cross-references)
        let mut builder = world::BspWorldBuilder::new(profile);
        builder.set_entity_raw(entity_raw);
        builder.set_entities(entities);
        builder.set_planes(planes);
        builder.set_vertices(vertices);
        builder.set_nodes(nodes);
        builder.set_leaves(leaves);
        builder.set_faces(faces);
        builder.set_models(models);
        builder.set_texinfos(texinfos);
        builder.set_edges(edges);
        builder.set_surfedges(surfedges);
        builder.set_markfaces(markfaces);
        builder.set_clipnodes(clipnodes);
        builder.set_miptex_data(miptex_data);
        builder.set_lightmap_data(lightmap_data);
        builder.set_vis_data(vis_data);
        builder.set_bspx(bspx_opt);
        builder.set_bspx_rgb_lighting(bspx_rgb);
        builder.set_palette(palette);
        builder.set_colored_light_source(colored_light_source);
        builder.set_lit_data(lit_data);
        builder.set_wad_archives(wad_archives);
        builder.set_content_hash(content_hash);
        builder.set_source_identity(options.source_identity.clone());
        builder.add_diagnostics(entity_diags);
        builder.add_diagnostics(bspx_diags);
        builder.add_diagnostics(companion_diags);
        builder.add_diagnostics(colored_diags);
        builder.add_diagnostics(wad_diags);

        builder.build()
    }
}

/// Extract a lump's raw bytes from the file.
fn extract_lump_bytes(data: &[u8], lump: &lumps::LumpRange) -> Vec<u8> {
    if lump.size == 0 {
        return Vec::new();
    }
    let start = lump.offset as usize;
    let end = start + lump.size as usize;
    data.get(start..end).map(|s| s.to_vec()).unwrap_or_default()
}

/// Compute the end of the standard lump region (highest offset + size).
fn compute_standard_lump_end(lumps: &[lumps::LumpRange; 15]) -> usize {
    let mut max_end = 0usize;
    for lump in lumps {
        if lump.size > 0 {
            let end = lump.offset as usize + lump.size as usize;
            if end > max_end {
                max_end = end;
            }
        }
    }
    max_end
}

/// Compute a deterministic content fingerprint for Phase 02 metadata.
fn compute_content_hash(data: &[u8]) -> [u8; 32] {
    let mut lanes = [
        0xcbf2_9ce4_8422_2325u64,
        0x9e37_79b9_7f4a_7c15u64,
        0x94d0_49bb_1331_11ebu64,
        0x2545_f491_4f6c_dd1du64,
    ];
    for (i, &byte) in data.iter().enumerate() {
        let lane = i & 3;
        lanes[lane] ^= byte as u64;
        lanes[lane] = lanes[lane].wrapping_mul(0x100_0000_01b3);
        lanes[lane] ^= (i as u64).rotate_left((lane as u32) + 1);
    }
    let mut arr = [0u8; 32];
    for (i, lane) in lanes.iter().enumerate() {
        arr[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
    }
    arr
}

// Re-export key types for convenience
pub use diagnostic::{BspReport, DiagnosticCode, Severity};
pub use entities::{Entity, EntityClass, KeyValue};

#[cfg(test)]
mod tests {
    use super::*;

    use crate::profile::BspProfile;

    /// Build a minimal valid BSP29 file for testing.
    pub fn make_minimal_bsp29() -> Vec<u8> {
        let mut data = Vec::new();

        // Header: version (4 bytes) + 15 lump descriptors (120 bytes) = 124 bytes
        data.extend_from_slice(&29u32.to_le_bytes());

        // Lump table: all lumps empty except entities and a single plane
        // We'll place data after the header
        let mut current_offset: u32 = 124;

        // Entities: a minimal entity string (null-terminated)
        let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
        let entity_offset = current_offset;
        let entity_size = entity_bytes.len() as u32;
        current_offset += entity_size;

        // Plane: one plane for the world
        let plane_offset = current_offset;
        let plane_size = 20u32; // one plane: 20 bytes
        current_offset += plane_size;
        let _ = current_offset; // reserved for future expansion

        // Vertices: 0
        // Nodes: 0
        // Leaves: 0
        // ... all others empty

        // Build lump table
        let lumps: [(u32, u32); 15] = [
            (entity_offset, entity_size), // entities
            (plane_offset, plane_size),   // planes
            (0, 0),                       // miptex
            (0, 0),                       // vertices
            (0, 0),                       // visinfo
            (0, 0),                       // nodes
            (0, 0),                       // texinfo
            (0, 0),                       // faces
            (0, 0),                       // lightmaps
            (0, 0),                       // clipnodes
            (0, 0),                       // leaves
            (0, 0),                       // markfaces
            (0, 0),                       // edges
            (0, 0),                       // surfedges
            (0, 0),                       // models
        ];

        for (off, sz) in &lumps {
            data.extend_from_slice(&off.to_le_bytes());
            data.extend_from_slice(&sz.to_le_bytes());
        }

        // Now write the actual data
        // Entities
        data.extend_from_slice(entity_bytes);

        // Planes: (0, 0, 1), dist=0, type=0
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0i32.to_le_bytes());

        data
    }

    #[test]
    fn load_minimal_bsp29() {
        let data = make_minimal_bsp29();
        let options = LoadOptions::default();
        let world = BspLoader::load(&data, &options).unwrap();
        assert_eq!(world.profile, BspProfile::Bsp29);
        assert_eq!(world.planes.len(), 1);
        assert_eq!(world.entities.len(), 1);
        assert!(world.worldspawn().is_some());
    }

    #[test]
    fn load_rejects_unknown_magic() {
        let data = b"XXXX_rest_of_file_data...";
        let options = LoadOptions::default();
        let r = BspLoader::load(data, &options);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedDialect);
    }

    #[test]
    fn load_rejects_truncated_header() {
        let data = vec![0u8; 50]; // < 124
        let r = BspLoader::load(&data, &LoadOptions::default());
        assert!(r.is_err());
    }

    #[test]
    fn load_with_palette() {
        let data = make_minimal_bsp29();
        let palette = vec![0u8; 768];
        let options = LoadOptions {
            palette: Some(palette.clone()),
            ..Default::default()
        };
        let world = BspLoader::load(&data, &options).unwrap();
        assert!(world.palette.is_some());
    }
}
