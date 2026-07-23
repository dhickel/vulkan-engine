//! Standard 15-lump BSP parsing.
//!
//! Validates header size, lump ranges, alignment, overlap, count divisibility,
//! cross-lump indices, and graph acyclicity. Structural corruption triggers
//! whole-asset rejection — individual faces/models/nodes are never silently dropped.

use glam::Vec3;

use crate::decode;
use crate::diagnostic::{BspReport, DiagnosticCode, SourceSpan};
use crate::limits;
use crate::profile::BspProfile;

// ── Lump indices ──
pub const LUMP_ENTITIES: usize = 0;
pub const LUMP_PLANES: usize = 1;
pub const LUMP_MIPTEX: usize = 2;
pub const LUMP_VERTICES: usize = 3;
pub const LUMP_VISINFO: usize = 4;
pub const LUMP_NODES: usize = 5;
pub const LUMP_TEXINFO: usize = 6;
pub const LUMP_FACES: usize = 7;
pub const LUMP_LIGHTMAPS: usize = 8;
pub const LUMP_CLIPNODES: usize = 9;
pub const LUMP_LEAVES: usize = 10;
pub const LUMP_MARKFACES: usize = 11;
pub const LUMP_EDGES: usize = 12;
pub const LUMP_SURFEDGES: usize = 13;
pub const LUMP_MODELS: usize = 14;

pub const LUMP_NAMES: [&str; 15] = [
    "entities",
    "planes",
    "miptex",
    "vertices",
    "visinfo",
    "nodes",
    "texinfo",
    "faces",
    "lightmaps",
    "clipnodes",
    "leaves",
    "markfaces",
    "edges",
    "surfedges",
    "models",
];

// ── Raw lump descriptor from header ──
#[derive(Debug, Clone, Copy)]
pub struct LumpRange {
    pub offset: u32,
    pub size: u32,
}

impl LumpRange {
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

fn checked_vec_capacity(
    count: u32,
    stride: u32,
    max_count: u32,
    code: DiagnosticCode,
    lump_name: &str,
) -> Result<usize, BspReport> {
    let bytes = limits::checked_count_stride_usize(count, stride as usize, max_count, code)?;
    if bytes as u64 > limits::TOTAL_LUMP_ALLOCATION {
        return Err(BspReport::fatal(
            DiagnosticCode::AllocationExceeded,
            format!(
                "lump {} allocation {} exceeds budget {}",
                lump_name,
                bytes,
                limits::TOTAL_LUMP_ALLOCATION
            ),
        ));
    }
    Ok(count as usize)
}

fn read_non_negative_i32_as_u32(
    data: &[u8],
    offset: usize,
    context: &str,
) -> Result<u32, BspReport> {
    let value = decode::read_i32_le(data, offset)?;
    if value < 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptIndex,
            format!("{} is negative ({})", context, value),
        ));
    }
    Ok(value as u32)
}

// ── Parsed BSP types ──

#[derive(Debug, Clone)]
pub struct Plane {
    pub normal: Vec3,
    pub dist: f32,
    pub plane_type: i32,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub plane_id: u32,
    pub children: [i32; 2],
    pub mins: [i16; 3],
    pub maxs: [i16; 3],
    pub face_id: u32,
    pub face_num: u32,
}

#[derive(Debug, Clone)]
pub struct Leaf {
    pub contents: i32,
    pub visofs: i32,
    pub mins: [i16; 3],
    pub maxs: [i16; 3],
    pub mark_id: u32,
    pub mark_num: u32,
    pub ambient: [u8; 4],
}

#[derive(Debug, Clone)]
pub struct Face {
    pub plane_id: u32,
    pub side: u32,
    pub ledge_id: u32,
    pub ledge_num: u32,
    pub texinfo_id: u32,
    pub styles: [u8; 4],
    pub lightofs: i32,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub mins: Vec3,
    pub maxs: Vec3,
    pub origin: Vec3,
    pub headnode: [i32; 4],
    pub visleafs: i32,
    pub face_id: u32,
    pub face_num: u32,
}

#[derive(Debug, Clone)]
pub struct Texinfo {
    pub vec_s: Vec3,
    pub dist_s: f32,
    pub vec_t: Vec3,
    pub dist_t: f32,
    pub miptex: u32,
    pub flags: u32,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub v: [u32; 2],
}

#[derive(Debug, Clone)]
pub struct Clipnode {
    pub plane: u32,
    pub children: [i32; 2],
}

// ── Lump table parsing ──

/// Parse all 15 lump descriptors from the header (bytes 4..124).
pub fn parse_lump_table(data: &[u8]) -> Result<[LumpRange; 15], BspReport> {
    let mut lumps = [LumpRange { offset: 0, size: 0 }; 15];
    for i in 0..15 {
        let off = 4 + i * 8;
        let offset = decode::read_i32_le(data, off)?;
        let size = decode::read_i32_le(data, off + 4)?;

        // Validate non-negative
        if offset < 0 || size < 0 {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptLump,
                format!(
                    "lump {} ({}) has negative offset ({}) or size ({})",
                    i, LUMP_NAMES[i], offset, size
                ),
            )
            .with_span(SourceSpan::Lump {
                index: i,
                offset: None,
            }));
        }

        lumps[i] = LumpRange {
            offset: offset as u32,
            size: size as u32,
        };
    }
    Ok(lumps)
}

/// Validate lump ranges: each must fit within the file, and lumps must not overlap.
pub fn validate_lump_ranges(
    lumps: &[LumpRange; 15],
    file_len: usize,
    bspx_data_offset: Option<usize>, // BSPX directory starts after standard lumps
) -> Result<(), BspReport> {
    // Track occupied byte ranges for overlap detection and enforce the aggregate allocation budget.
    let mut ranges: Vec<(usize, usize, usize)> = Vec::new(); // (start, end, lump_idx)
    let mut cumulative_allocation = 0u64;

    for i in 0..15 {
        let lump = &lumps[i];
        if lump.size == 0 {
            continue;
        }
        let start = lump.offset as usize;
        let end = start.checked_add(lump.size as usize).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptOverflow,
                format!("lump {} ({}) offset+size overflow", i, LUMP_NAMES[i]),
            )
        })?;

        if end > file_len {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptLump,
                format!(
                    "lump {} ({}) range [{}, {}) exceeds file length {}",
                    i, LUMP_NAMES[i], start, end, file_len
                ),
            )
            .with_span(SourceSpan::Lump {
                index: i,
                offset: None,
            }));
        }

        if i == LUMP_ENTITIES && start % 4 != 0 {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptAlignment,
                format!("entities lump offset {} is not 4-byte aligned", start),
            )
            .with_span(SourceSpan::Lump {
                index: i,
                offset: None,
            }));
        }

        // Check against BSPX boundary if applicable
        if let Some(bspx_start) = bspx_data_offset {
            if start >= bspx_start || end > bspx_start {
                return Err(BspReport::fatal(
                    DiagnosticCode::StructuralCorruptLump,
                    format!(
                        "lump {} ({}) range [{}, {}) overlaps BSPX directory region (starts at {})",
                        i, LUMP_NAMES[i], start, end, bspx_start
                    ),
                )
                .with_span(SourceSpan::Lump {
                    index: i,
                    offset: None,
                }));
            }
        }

        cumulative_allocation = limits::check_cumulative_allocation(
            cumulative_allocation,
            lump.size as u64,
            limits::TOTAL_LUMP_ALLOCATION,
        )?;

        ranges.push((start, end, i));
    }

    // Sort by offset and check for overlaps
    ranges.sort_by_key(|r| r.0);
    for w in ranges.windows(2) {
        let (_s1, e1, i1) = w[0];
        let (s2, _e2, i2) = w[1];
        if e1 > s2 {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptLump,
                format!(
                    "lump {} ({}) overlaps lump {} ({}): end {} > start {}",
                    i1, LUMP_NAMES[i1], i2, LUMP_NAMES[i2], e1, s2
                ),
            ));
        }
    }

    Ok(())
}

// ── Individual lump parsers ──

/// Parse entities lump (lump 0): null-terminated UTF-8/Latin-1 string.
/// Returns the raw entity string bytes.
pub fn parse_entities(data: &[u8], lump: &LumpRange, strict: bool) -> Result<Vec<u8>, BspReport> {
    if lump.size == 0 {
        return Ok(Vec::new());
    }
    if lump.size > limits::MAX_ENTITY_STRING_LENGTH {
        return Err(BspReport::fatal(
            DiagnosticCode::EntityStringTooLarge,
            format!(
                "entity string size {} exceeds limit {}",
                lump.size,
                limits::MAX_ENTITY_STRING_LENGTH
            ),
        ));
    }
    let offset = lump.offset as usize;
    let size = lump.size as usize;
    let raw = data.get(offset..offset + size).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            "entities lump out of bounds",
        )
    })?;

    // The entity string is structural data; non-null-terminated strings are rejected in every mode.
    let _ = strict;
    if raw.last() != Some(&0) {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptEntity,
            "entity string is not null-terminated",
        ));
    }

    Ok(raw.to_vec())
}

/// Parse planes lump (lump 1): array of (normal: vec3 f32, dist: f32, type: i32) = 20 bytes.
pub fn parse_planes(data: &[u8], lump: &LumpRange) -> Result<Vec<Plane>, BspReport> {
    const STRIDE: u32 = 20;
    let count = lump.size / STRIDE;
    if lump.size % STRIDE != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("planes lump size {} not divisible by 20", lump.size),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        STRIDE,
        u32::MAX / STRIDE,
        DiagnosticCode::StructuralCorruptLump,
        "planes",
    )?;

    let mut planes = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * 20;
        let normal = decode::read_vec3_finite(data, off, "plane.normal")?;
        let dist = decode::read_f32_finite(data, off + 12, "plane.dist")?;
        let plane_type = decode::read_i32_le(data, off + 16)?;
        planes.push(Plane {
            normal,
            dist,
            plane_type,
        });
    }
    Ok(planes)
}

/// Parse vertices lump (lump 3): array of (x, y, z: f32) = 12 bytes.
pub fn parse_vertices(
    data: &[u8],
    lump: &LumpRange,
    profile: BspProfile,
) -> Result<Vec<Vec3>, BspReport> {
    const STRIDE: u32 = 12;
    let count = lump.size / STRIDE;
    if lump.size % STRIDE != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("vertices lump size {} not divisible by 12", lump.size),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        STRIDE,
        profile.max_vertices(),
        DiagnosticCode::StructuralVertexCount,
        "vertices",
    )?;

    let mut vertices = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * 12;
        let v = decode::read_vec3_finite(data, off, "vertex")?;
        vertices.push(v);
    }
    Ok(vertices)
}

/// Parse nodes lump (lump 5): 28 bytes in BSP29, 32 bytes in BSP2.
pub fn parse_nodes(
    data: &[u8],
    lump: &LumpRange,
    profile: BspProfile,
) -> Result<Vec<Node>, BspReport> {
    let stride = if profile.uses_32bit_indices() {
        32u32
    } else {
        28u32
    };
    let count = lump.size / stride;
    if lump.size % stride != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("nodes lump size {} not divisible by {}", lump.size, stride),
        ));
    }

    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        stride,
        profile.max_nodes(),
        DiagnosticCode::StructuralNodeCount,
        "nodes",
    )?;
    let mut nodes = Vec::with_capacity(capacity);

    for i in 0..count {
        let off = offset + (i as usize) * stride as usize;
        let plane_id = decode::read_u32_le(data, off)?;
        let child0 = decode::read_i32_le(data, off + 4)?;
        let child1 = decode::read_i32_le(data, off + 8)?;
        let mins = decode::read_i16x3(data, off + 12)?;
        let maxs = decode::read_i16x3(data, off + 18)?;
        let face_id = if profile.uses_32bit_indices() {
            decode::read_u32_le(data, off + 24)?
        } else {
            decode::read_u16_le(data, off + 24)? as u32
        };
        let face_num = if profile.uses_32bit_indices() {
            decode::read_u32_le(data, off + 28)?
        } else {
            decode::read_u16_le(data, off + 26)? as u32
        };

        nodes.push(Node {
            plane_id,
            children: [child0, child1],
            mins,
            maxs,
            face_id,
            face_num,
        });
    }
    Ok(nodes)
}

/// Parse leaves lump (lump 10): 28 bytes in BSP29, 32 bytes in BSP2.
pub fn parse_leaves(
    data: &[u8],
    lump: &LumpRange,
    profile: BspProfile,
) -> Result<Vec<Leaf>, BspReport> {
    let stride = if profile.uses_32bit_indices() {
        32u32
    } else {
        28u32
    };
    let count = lump.size / stride;
    if lump.size % stride != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("leaves lump size {} not divisible by {}", lump.size, stride),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        stride,
        profile.max_leaves(),
        DiagnosticCode::StructuralLeafCount,
        "leaves",
    )?;

    let mut leaves = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * stride as usize;
        let contents = decode::read_i32_le(data, off)?;
        let visofs = decode::read_i32_le(data, off + 4)?;

        let (mins_arr, maxs_arr, mark_id, mark_num, ambient_off) = if profile.uses_32bit_indices() {
            // ericw-tools BSP2 leaves keep BSP29 i16 bounds and expand mark ranges to u32.
            // Layout: contents i32, visofs i32, mins i16x3, maxs i16x3, mark u32, markleaf u32, ambient u8[4].
            let mins_arr = decode::read_i16x3(data, off + 8)?;
            let maxs_arr = decode::read_i16x3(data, off + 14)?;
            let mark_id = decode::read_u32_le(data, off + 20)?;
            let mark_num = decode::read_u32_le(data, off + 24)?;
            let ambient_off = off + 28;
            (mins_arr, maxs_arr, mark_id, mark_num, ambient_off)
        } else {
            let mins_arr = decode::read_i16x3(data, off + 8)?;
            let maxs_arr = decode::read_i16x3(data, off + 14)?;
            let mark_id = decode::read_u16_le(data, off + 20)? as u32;
            let mark_num = decode::read_u16_le(data, off + 22)? as u32;
            let ambient_off = off + 24;
            (mins_arr, maxs_arr, mark_id, mark_num, ambient_off)
        };

        let ambient = [
            decode::read_u8(data, ambient_off)?,
            decode::read_u8(data, ambient_off + 1)?,
            decode::read_u8(data, ambient_off + 2)?,
            decode::read_u8(data, ambient_off + 3)?,
        ];

        leaves.push(Leaf {
            contents,
            visofs,
            mins: mins_arr,
            maxs: maxs_arr,
            mark_id,
            mark_num,
            ambient,
        });
    }
    Ok(leaves)
}

/// Parse faces lump (lump 7): 20 bytes in BSP29, 28 bytes in BSP2.
pub fn parse_faces(
    data: &[u8],
    lump: &LumpRange,
    profile: BspProfile,
) -> Result<Vec<Face>, BspReport> {
    let stride = if profile.uses_32bit_indices() {
        28u32
    } else {
        20u32
    };
    let count = lump.size / stride;
    if lump.size % stride != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("faces lump size {} not divisible by {}", lump.size, stride),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        stride,
        profile.max_faces(),
        DiagnosticCode::StructuralFaceCount,
        "faces",
    )?;

    let mut faces = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * stride as usize;
        let plane_id = if profile.uses_32bit_indices() {
            decode::read_u32_le(data, off)?
        } else {
            decode::read_u16_le(data, off)? as u32
        };
        let side = if profile.uses_32bit_indices() {
            decode::read_u32_le(data, off + 4)?
        } else {
            decode::read_u16_le(data, off + 2)? as u32
        };
        let ledge_id = if profile.uses_32bit_indices() {
            decode::read_u32_le(data, off + 8)?
        } else {
            decode::read_u32_le(data, off + 4)?
        };
        let ledge_num = if profile.uses_32bit_indices() {
            decode::read_u32_le(data, off + 12)?
        } else {
            decode::read_u16_le(data, off + 8)? as u32
        };
        let texinfo_id = if profile.uses_32bit_indices() {
            decode::read_u32_le(data, off + 16)?
        } else {
            decode::read_u16_le(data, off + 10)? as u32
        };
        let styles = if profile.uses_32bit_indices() {
            [
                decode::read_u8(data, off + 20)?,
                decode::read_u8(data, off + 21)?,
                decode::read_u8(data, off + 22)?,
                decode::read_u8(data, off + 23)?,
            ]
        } else {
            [
                decode::read_u8(data, off + 12)?,
                decode::read_u8(data, off + 13)?,
                decode::read_u8(data, off + 14)?,
                decode::read_u8(data, off + 15)?,
            ]
        };
        let lightofs = if profile.uses_32bit_indices() {
            decode::read_i32_le(data, off + 24)?
        } else {
            decode::read_i32_le(data, off + 16)?
        };

        faces.push(Face {
            plane_id,
            side,
            ledge_id,
            ledge_num,
            texinfo_id,
            styles,
            lightofs,
        });
    }
    Ok(faces)
}

/// Parse models lump (lump 14): 64 bytes per model.
pub fn parse_models(
    data: &[u8],
    lump: &LumpRange,
    profile: BspProfile,
) -> Result<Vec<Model>, BspReport> {
    const STRIDE: u32 = 64;
    let count = lump.size / STRIDE;
    if lump.size % STRIDE != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("models lump size {} not divisible by 64", lump.size),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        STRIDE,
        profile.max_models(),
        DiagnosticCode::StructuralModelCount,
        "models",
    )?;

    let mut models = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * 64;
        let mins = decode::read_vec3_finite(data, off, "model.mins")?;
        let maxs = decode::read_vec3_finite(data, off + 12, "model.maxs")?;
        let origin = decode::read_vec3_finite(data, off + 24, "model.origin")?;
        let headnode = [
            decode::read_i32_le(data, off + 36)?,
            decode::read_i32_le(data, off + 40)?,
            decode::read_i32_le(data, off + 44)?,
            decode::read_i32_le(data, off + 48)?,
        ];
        let visleafs = decode::read_i32_le(data, off + 52)?;
        let face_id = read_non_negative_i32_as_u32(data, off + 56, "model.face_id")?;
        let face_num = read_non_negative_i32_as_u32(data, off + 60, "model.face_num")?;

        models.push(Model {
            mins,
            maxs,
            origin,
            headnode,
            visleafs,
            face_id,
            face_num,
        });
    }
    Ok(models)
}

/// Parse texinfo lump (lump 6): 40 bytes per texinfo.
pub fn parse_texinfo(data: &[u8], lump: &LumpRange) -> Result<Vec<Texinfo>, BspReport> {
    const STRIDE: u32 = 40;
    let count = lump.size / STRIDE;
    if lump.size % STRIDE != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("texinfo lump size {} not divisible by 40", lump.size),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        STRIDE,
        u32::MAX / STRIDE,
        DiagnosticCode::StructuralCorruptLump,
        "texinfo",
    )?;

    let mut texinfos = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * 40;
        let vec_s = decode::read_vec3_finite(data, off, "texinfo.vec_s")?;
        let dist_s = decode::read_f32_finite(data, off + 12, "texinfo.dist_s")?;
        let vec_t = decode::read_vec3_finite(data, off + 16, "texinfo.vec_t")?;
        let dist_t = decode::read_f32_finite(data, off + 28, "texinfo.dist_t")?;
        let miptex = decode::read_u32_le(data, off + 32)?;
        let flags = decode::read_u32_le(data, off + 36)?;
        texinfos.push(Texinfo {
            vec_s,
            dist_s,
            vec_t,
            dist_t,
            miptex,
            flags,
        });
    }
    Ok(texinfos)
}

/// Parse edges lump (lump 12): 4 bytes per edge (u16×2) in BSP29, 8 bytes (u32×2) in BSP2.
pub fn parse_edges(
    data: &[u8],
    lump: &LumpRange,
    profile: BspProfile,
) -> Result<Vec<Edge>, BspReport> {
    let stride = if profile.uses_32bit_indices() {
        8u32
    } else {
        4u32
    };
    let count = lump.size / stride;
    if lump.size % stride != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("edges lump size {} not divisible by {}", lump.size, stride),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        stride,
        profile.max_edges(),
        DiagnosticCode::StructuralEdgeCount,
        "edges",
    )?;

    let mut edges = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * stride as usize;
        let (v0, v1) = if profile.uses_32bit_indices() {
            (
                decode::read_u32_le(data, off)?,
                decode::read_u32_le(data, off + 4)?,
            )
        } else {
            (
                decode::read_u16_le(data, off)? as u32,
                decode::read_u16_le(data, off + 2)? as u32,
            )
        };
        edges.push(Edge { v: [v0, v1] });
    }
    Ok(edges)
}

/// Parse surfedges lump (lump 13): array of i32 (signed edge index).
pub fn parse_surfedges(
    data: &[u8],
    lump: &LumpRange,
    profile: BspProfile,
) -> Result<Vec<i32>, BspReport> {
    const STRIDE: u32 = 4;
    let count = lump.size / STRIDE;
    if lump.size % STRIDE != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("surfedges lump size {} not divisible by 4", lump.size),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        STRIDE,
        profile.max_surfedges(),
        DiagnosticCode::StructuralSurfedgeCount,
        "surfedges",
    )?;

    let mut surfedges = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * 4;
        let se = decode::read_i32_le(data, off)?;
        surfedges.push(se);
    }
    Ok(surfedges)
}

/// Parse markfaces lump (lump 11): u16 in BSP29, u32 in BSP2.
pub fn parse_markfaces(
    data: &[u8],
    lump: &LumpRange,
    profile: BspProfile,
) -> Result<Vec<u32>, BspReport> {
    let stride = if profile.uses_32bit_indices() {
        4u32
    } else {
        2u32
    };
    let count = lump.size / stride;
    if lump.size % stride != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!(
                "markfaces lump size {} not divisible by {}",
                lump.size, stride
            ),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        stride,
        profile.max_markfaces(),
        DiagnosticCode::StructuralMarkfaceCount,
        "markfaces",
    )?;

    let mut markfaces = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * stride as usize;
        let mf = if profile.uses_32bit_indices() {
            decode::read_u32_le(data, off)?
        } else {
            decode::read_u16_le(data, off)? as u32
        };
        markfaces.push(mf);
    }
    Ok(markfaces)
}

/// Parse clipnodes lump (lump 9): 8 bytes in BSP29, 12 bytes in BSP2.
pub fn parse_clipnodes(
    data: &[u8],
    lump: &LumpRange,
    profile: BspProfile,
) -> Result<Vec<Clipnode>, BspReport> {
    let stride = if profile.uses_32bit_indices() {
        12u32
    } else {
        8u32
    };
    let count = lump.size / stride;
    if lump.size % stride != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!(
                "clipnodes lump size {} not divisible by {}",
                lump.size, stride
            ),
        ));
    }
    let offset = lump.offset as usize;
    let capacity = checked_vec_capacity(
        count,
        stride,
        profile.max_clipnodes(),
        DiagnosticCode::StructuralClipnodeCount,
        "clipnodes",
    )?;

    let mut clipnodes = Vec::with_capacity(capacity);
    for i in 0..count {
        let off = offset + (i as usize) * stride as usize;
        let plane = decode::read_u32_le(data, off)?;
        let (c0, c1) = if profile.uses_32bit_indices() {
            (
                decode::read_i32_le(data, off + 4)?,
                decode::read_i32_le(data, off + 8)?,
            )
        } else {
            (
                decode::read_i16_le(data, off + 4)? as i32,
                decode::read_i16_le(data, off + 6)? as i32,
            )
        };
        clipnodes.push(Clipnode {
            plane,
            children: [c0, c1],
        });
    }
    Ok(clipnodes)
}

// ── Cross-lump validation ──

/// Validate cross-lump indices: planes, vertices, edges, surfedges, faces,
/// markfaces, nodes, leaves, clipnodes, texinfos, and model references.
pub fn validate_cross_lump_indices(
    planes: &[Plane],
    vertices: &[Vec3],
    nodes: &[Node],
    leaves: &[Leaf],
    faces: &[Face],
    models: &[Model],
    edges: &[Edge],
    surfedges: &[i32],
    markfaces: &[u32],
    clipnodes: &[Clipnode],
    texinfos: &[Texinfo],
) -> Result<(), BspReport> {
    let num_planes = planes.len() as u32;
    let num_vertices = vertices.len() as u32;
    let num_nodes = nodes.len() as u32;
    let num_leaves = leaves.len() as u32;
    let num_faces = faces.len() as u32;
    let _num_models = models.len() as u32;
    let num_edges = edges.len() as u32;
    let num_surfedges = surfedges.len() as u32;
    let num_markfaces = markfaces.len() as u32;
    let num_clipnodes = clipnodes.len() as u32;
    let num_texinfos = texinfos.len() as u32;

    // Validate node plane references
    for (i, node) in nodes.iter().enumerate() {
        limits::check_index(node.plane_id, num_planes, &format!("node[{}].plane", i))?;
        // Validate node children
        limits::check_node_child(
            node.children[0],
            num_nodes,
            num_leaves,
            &format!("node[{}].child[0]", i),
        )?;
        limits::check_node_child(
            node.children[1],
            num_nodes,
            num_leaves,
            &format!("node[{}].child[1]", i),
        )?;
        // Validate node face ranges
        let face_end = node.face_id.checked_add(node.face_num).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptOverflow,
                format!("node[{}].face_id + face_num overflow", i),
            )
        })?;
        if face_end > num_faces {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "node[{}] face range [{}, {}) exceeds face count {}",
                    i, node.face_id, face_end, num_faces
                ),
            ));
        }
    }

    // Validate leaf markface ranges
    for (i, leaf) in leaves.iter().enumerate() {
        let mark_end = leaf.mark_id.checked_add(leaf.mark_num).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptOverflow,
                format!("leaf[{}].mark_id + mark_num overflow", i),
            )
        })?;
        if mark_end > num_markfaces {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "leaf[{}] markface range [{}, {}) exceeds markface count {}",
                    i, leaf.mark_id, mark_end, num_markfaces
                ),
            ));
        }
    }

    // Validate face references
    for (i, face) in faces.iter().enumerate() {
        limits::check_index(face.plane_id, num_planes, &format!("face[{}].plane", i))?;

        // Validate styles
        for &style in &face.styles {
            if style != limits::LIGHT_STYLE_SENTINEL && style > limits::MAX_STYLE_IDENTIFIER {
                return Err(BspReport::fatal(
                    DiagnosticCode::UnsupportedStyleSlot,
                    format!(
                        "face[{}] style {} exceeds max style {}",
                        i,
                        style,
                        limits::MAX_STYLE_IDENTIFIER
                    ),
                ));
            }
        }

        limits::check_index(
            face.texinfo_id,
            num_texinfos,
            &format!("face[{}].texinfo", i),
        )?;

        // Validate surfedge range
        let ledge_end = face.ledge_id.checked_add(face.ledge_num).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptOverflow,
                format!("face[{}].ledge_id + ledge_num overflow", i),
            )
        })?;
        if ledge_end > num_surfedges {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "face[{}] surfedge range [{}, {}) exceeds surfedge count {}",
                    i, face.ledge_id, ledge_end, num_surfedges
                ),
            ));
        }
    }

    // Validate model references, including model 0/worldspawn; corrupt ranges reject the asset.
    for (i, model) in models.iter().enumerate() {
        let face_end = model.face_id.checked_add(model.face_num).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::StructuralCorruptOverflow,
                format!("model[{}].face_id + face_num overflow", i),
            )
        })?;
        if face_end > num_faces {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "model[{}] face range [{}, {}) exceeds face count {}",
                    i, model.face_id, face_end, num_faces
                ),
            ));
        }
    }

    // Validate edge references
    for (i, edge) in edges.iter().enumerate() {
        for &v in &edge.v {
            limits::check_index(v, num_vertices, &format!("edge[{}].v", i))?;
        }
    }

    // Validate surfedge → edge references
    for (i, &se) in surfedges.iter().enumerate() {
        let edge_idx = if se >= 0 { se as u32 } else { (-se) as u32 };
        limits::check_index(edge_idx, num_edges, &format!("surfedge[{}] -> edge", i))?;
    }

    // Validate markface → face references
    for (i, &mf) in markfaces.iter().enumerate() {
        limits::check_index(mf, num_faces, &format!("markface[{}] -> face", i))?;
    }

    // Validate clipnode plane references
    for (i, cn) in clipnodes.iter().enumerate() {
        // Clipnodes reference a subset of render planes.
        limits::check_index(cn.plane, num_planes, &format!("clipnode[{}].plane", i))?;
        // Validate children
        for (j, &child) in cn.children.iter().enumerate() {
            let ctx = format!("clipnode[{}].children[{}]", i, j);
            if child >= 0 {
                limits::check_index(child as u32, num_clipnodes, &ctx)?;
            } else {
                // Negative children reference contents values (-1 = empty, -2 = solid, etc.)
                let contents = child;
                if contents > 0 || contents < -6 {
                    // Valid contents are typically -1 through -6
                }
            }
        }
    }

    // Check acyclicity of the node graph
    if num_nodes > 0 {
        limits::check_graph_acyclic(
            num_nodes,
            |n| {
                let node = &nodes[n as usize];
                let c0 = if node.children[0] >= 0 {
                    Some(node.children[0] as u32)
                } else {
                    None
                };
                let c1 = if node.children[1] >= 0 {
                    Some(node.children[1] as u32)
                } else {
                    None
                };
                (c0, c1)
            },
            limits::MAX_TREE_DEPTH,
            "node tree",
        )?;
    }

    // Check clipnode tree acyclicity
    if num_clipnodes > 0 {
        limits::check_graph_acyclic(
            num_clipnodes,
            |n| {
                let cn = &clipnodes[n as usize];
                let c0 = if cn.children[0] >= 0 {
                    Some(cn.children[0] as u32)
                } else {
                    None
                };
                let c1 = if cn.children[1] >= 0 {
                    Some(cn.children[1] as u32)
                } else {
                    None
                };
                (c0, c1)
            },
            limits::MAX_TREE_DEPTH,
            "clipnode tree",
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bsp29_header(lumps: &[(u32, u32); 15]) -> Vec<u8> {
        let mut data = vec![0u8; 124];
        data[0..4].copy_from_slice(&29u32.to_le_bytes());
        for (i, &(off, sz)) in lumps.iter().enumerate() {
            let base = 4 + i * 8;
            data[base..base + 4].copy_from_slice(&off.to_le_bytes());
            data[base + 4..base + 8].copy_from_slice(&sz.to_le_bytes());
        }
        data
    }

    #[test]
    fn parse_empty_lump_table() {
        let header = make_bsp29_header(&[(0, 0); 15]);
        let lumps = parse_lump_table(&header).unwrap();
        for l in &lumps {
            assert!(l.is_empty());
        }
    }

    #[test]
    fn validate_overlapping_lumps_detected() {
        // Non-overlapping lumps: entities[124..224), planes[224..324)
        let lumps_ok = [
            LumpRange {
                offset: 124,
                size: 100,
            }, // entities: 124..224
            LumpRange {
                offset: 224,
                size: 100,
            }, // planes: 224..324
            LumpRange { offset: 0, size: 0 }, // miptex (empty)
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
        ];
        // Non-overlapping: no error
        assert!(validate_lump_ranges(&lumps_ok, 400, None).is_ok());

        // Overlapping: planes start at 174, entities end at 224
        let bad = [
            LumpRange {
                offset: 124,
                size: 100,
            },
            LumpRange {
                offset: 174,
                size: 100,
            }, // overlaps entities
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
            LumpRange { offset: 0, size: 0 },
        ];
        assert!(validate_lump_ranges(&bad, 400, None).is_err());
    }

    #[test]
    fn parse_planes_valid() {
        let mut data = Vec::new();
        // One plane: normal (0,0,1), dist 100, type 0
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&100.0f32.to_le_bytes());
        data.extend_from_slice(&0i32.to_le_bytes());
        let lump = LumpRange {
            offset: 0,
            size: data.len() as u32,
        };
        let planes = parse_planes(&data, &lump).unwrap();
        assert_eq!(planes.len(), 1);
        assert!((planes[0].dist - 100.0).abs() < 0.001);
    }

    #[test]
    fn parse_vertices_valid() {
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        let lump = LumpRange {
            offset: 0,
            size: 12,
        };
        let verts = parse_vertices(&data, &lump, BspProfile::Bsp29).unwrap();
        assert_eq!(verts.len(), 1);
        assert!((verts[0].x - 1.0).abs() < 0.001);
    }

    #[test]
    fn cross_lump_validation_rejects_bad_face_plane() {
        let planes = vec![Plane {
            normal: Vec3::Z,
            dist: 0.0,
            plane_type: 0,
        }];
        let faces = vec![Face {
            plane_id: 99,
            side: 0,
            ledge_id: 0,
            ledge_num: 0,
            texinfo_id: 0,
            styles: [255, 255, 255, 255],
            lightofs: -1,
        }];
        let r = validate_cross_lump_indices(
            &planes,
            &[],
            &[],
            &[],
            &faces,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptIndex);
    }

    #[test]
    fn face_style_validation() {
        let faces = vec![Face {
            plane_id: 0,
            side: 0,
            ledge_id: 0,
            ledge_num: 0,
            texinfo_id: 0,
            styles: [0, 64, 255, 255],
            lightofs: -1,
        }];
        let r = validate_cross_lump_indices(
            &[Plane {
                normal: Vec3::Z,
                dist: 0.0,
                plane_type: 0,
            }],
            &[],
            &[],
            &[],
            &faces,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        // style 64 > MAX_STYLE_IDENTIFIER (63)
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::UnsupportedStyleSlot);
    }
}
