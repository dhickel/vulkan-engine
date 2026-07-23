//! Immutable owned `BspWorld` — the validated, read-only parsed BSP.
//!
//! Construction only after all structural/cross-reference validation succeeds.
//! No invalid cross-references, cyclic graphs, unchecked counts, or mutable records.

use glam::Vec3;

use crate::bspx::BspxDirectory;
use crate::companions::ColoredLightSource;
use crate::diagnostic::BspReport;
use crate::entities::Entity;
use crate::lumps;
use crate::profile::BspProfile;
use crate::resources::Palette;
use crate::wad::WadArchive;

/// The immutable, fully validated BSP world.
///
/// All data is in Quake space (no coordinate conversion).
/// Construction requires passing all structural and cross-reference validation.
#[derive(Debug, Clone)]
pub struct BspWorld {
    /// Recognized profile.
    pub profile: BspProfile,

    // ── Parsed lumps ──
    /// Raw entity string bytes.
    pub entity_raw: Vec<u8>,
    /// Parsed entities.
    pub entities: Vec<Entity>,

    /// Validated planes (lump 1).
    pub planes: Vec<lumps::Plane>,
    /// Validated vertices (lump 3).
    pub vertices: Vec<Vec3>,
    /// Validated nodes (lump 5).
    pub nodes: Vec<lumps::Node>,
    /// Validated leaves (lump 10).
    pub leaves: Vec<lumps::Leaf>,
    /// Validated faces (lump 7).
    pub faces: Vec<lumps::Face>,
    /// Validated models (lump 14).
    pub models: Vec<lumps::Model>,
    /// Validated texinfos (lump 6).
    pub texinfos: Vec<lumps::Texinfo>,
    /// Validated edges (lump 12).
    pub edges: Vec<lumps::Edge>,
    /// Validated surfedges (lump 13).
    pub surfedges: Vec<i32>,
    /// Validated markfaces (lump 11).
    pub markfaces: Vec<u32>,
    /// Validated clipnodes (lump 9).
    pub clipnodes: Vec<lumps::Clipnode>,

    // ── Data lumps (raw bytes) ──
    /// Raw miptex lump (lump 2).
    pub miptex_data: Vec<u8>,
    /// Raw lightmap lump (lump 8).
    pub lightmap_data: Vec<u8>,
    /// Raw VIS data (lump 4).
    pub vis_data: Vec<u8>,

    // ── Extensions ──
    /// BSPX directory and extension lumps (if present).
    pub bspx: Option<BspxDirectory>,
    /// Raw BSPX RGBLIGHTING data (if present).
    pub bspx_rgb_lighting: Option<Vec<u8>>,

    // ── Companions ──
    /// Loaded palette (if provided).
    pub palette: Option<Palette>,
    /// Colored light source actually used.
    pub colored_light_source: ColoredLightSource,
    /// Raw .lit data (if loaded).
    pub lit_data: Option<Vec<u8>>,
    /// Loaded WAD archives (by basename).
    pub wad_archives: Vec<(String, WadArchive)>,

    // ── Metadata ──
    /// Deterministic Phase 02 fingerprint of the original BSP bytes.
    pub content_hash: [u8; 32],
    /// Package-relative source identity.
    pub source_identity: String,
    /// All diagnostics accumulated during parsing.
    pub diagnostics: Vec<BspReport>,
}

impl BspWorld {
    /// Accessor for the number of models.
    pub fn num_models(&self) -> u32 {
        self.models.len() as u32
    }

    /// Accessor for the number of faces.
    pub fn num_faces(&self) -> u32 {
        self.faces.len() as u32
    }

    /// Accessor for the number of vertices.
    pub fn num_vertices(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Accessor for the number of leaves.
    pub fn num_leaves(&self) -> u32 {
        self.leaves.len() as u32
    }

    /// Accessor for the number of nodes.
    pub fn num_nodes(&self) -> u32 {
        self.nodes.len() as u32
    }

    /// Whether this world has any fatal errors.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error())
    }

    /// Get the worldspawn entity (entity 0, if it exists and is worldspawn).
    pub fn worldspawn(&self) -> Option<&Entity> {
        self.entities
            .first()
            .filter(|e| e.class == crate::entities::EntityClass::Worldspawn)
    }

    /// Total number of entities.
    pub fn num_entities(&self) -> u32 {
        self.entities.len() as u32
    }
}

/// Builder for constructing a validated `BspWorld`.
///
/// Collects parsed data, validates cross-references, and constructs
/// the immutable world only when all checks pass.
pub struct BspWorldBuilder {
    profile: BspProfile,
    entity_raw: Vec<u8>,
    entities: Vec<Entity>,
    planes: Vec<lumps::Plane>,
    vertices: Vec<Vec3>,
    nodes: Vec<lumps::Node>,
    leaves: Vec<lumps::Leaf>,
    faces: Vec<lumps::Face>,
    models: Vec<lumps::Model>,
    texinfos: Vec<lumps::Texinfo>,
    edges: Vec<lumps::Edge>,
    surfedges: Vec<i32>,
    markfaces: Vec<u32>,
    clipnodes: Vec<lumps::Clipnode>,
    miptex_data: Vec<u8>,
    lightmap_data: Vec<u8>,
    vis_data: Vec<u8>,
    bspx: Option<BspxDirectory>,
    bspx_rgb_lighting: Option<Vec<u8>>,
    palette: Option<Palette>,
    colored_light_source: ColoredLightSource,
    lit_data: Option<Vec<u8>>,
    wad_archives: Vec<(String, WadArchive)>,
    content_hash: [u8; 32],
    source_identity: String,
    diagnostics: Vec<BspReport>,
}

impl BspWorldBuilder {
    /// Create a new builder with the detected profile.
    pub fn new(profile: BspProfile) -> Self {
        BspWorldBuilder {
            profile,
            entity_raw: Vec::new(),
            entities: Vec::new(),
            planes: Vec::new(),
            vertices: Vec::new(),
            nodes: Vec::new(),
            leaves: Vec::new(),
            faces: Vec::new(),
            models: Vec::new(),
            texinfos: Vec::new(),
            edges: Vec::new(),
            surfedges: Vec::new(),
            markfaces: Vec::new(),
            clipnodes: Vec::new(),
            miptex_data: Vec::new(),
            lightmap_data: Vec::new(),
            vis_data: Vec::new(),
            bspx: None,
            bspx_rgb_lighting: None,
            palette: None,
            colored_light_source: ColoredLightSource::Monochrome,
            lit_data: None,
            wad_archives: Vec::new(),
            content_hash: [0u8; 32],
            source_identity: String::new(),
            diagnostics: Vec::new(),
        }
    }

    pub fn set_entity_raw(&mut self, data: Vec<u8>) {
        self.entity_raw = data;
    }
    pub fn set_entities(&mut self, entities: Vec<Entity>) {
        self.entities = entities;
    }
    pub fn set_planes(&mut self, planes: Vec<lumps::Plane>) {
        self.planes = planes;
    }
    pub fn set_vertices(&mut self, vertices: Vec<Vec3>) {
        self.vertices = vertices;
    }
    pub fn set_nodes(&mut self, nodes: Vec<lumps::Node>) {
        self.nodes = nodes;
    }
    pub fn set_leaves(&mut self, leaves: Vec<lumps::Leaf>) {
        self.leaves = leaves;
    }
    pub fn set_faces(&mut self, faces: Vec<lumps::Face>) {
        self.faces = faces;
    }
    pub fn set_models(&mut self, models: Vec<lumps::Model>) {
        self.models = models;
    }
    pub fn set_texinfos(&mut self, texinfos: Vec<lumps::Texinfo>) {
        self.texinfos = texinfos;
    }
    pub fn set_edges(&mut self, edges: Vec<lumps::Edge>) {
        self.edges = edges;
    }
    pub fn set_surfedges(&mut self, surfedges: Vec<i32>) {
        self.surfedges = surfedges;
    }
    pub fn set_markfaces(&mut self, markfaces: Vec<u32>) {
        self.markfaces = markfaces;
    }
    pub fn set_clipnodes(&mut self, clipnodes: Vec<lumps::Clipnode>) {
        self.clipnodes = clipnodes;
    }
    pub fn set_miptex_data(&mut self, data: Vec<u8>) {
        self.miptex_data = data;
    }
    pub fn set_lightmap_data(&mut self, data: Vec<u8>) {
        self.lightmap_data = data;
    }
    pub fn set_vis_data(&mut self, data: Vec<u8>) {
        self.vis_data = data;
    }
    pub fn set_bspx(&mut self, bspx: Option<BspxDirectory>) {
        self.bspx = bspx;
    }
    pub fn set_bspx_rgb_lighting(&mut self, data: Option<Vec<u8>>) {
        self.bspx_rgb_lighting = data;
    }
    pub fn set_palette(&mut self, palette: Option<Palette>) {
        self.palette = palette;
    }
    pub fn set_colored_light_source(&mut self, source: ColoredLightSource) {
        self.colored_light_source = source;
    }
    pub fn set_lit_data(&mut self, data: Option<Vec<u8>>) {
        self.lit_data = data;
    }
    pub fn set_wad_archives(&mut self, archives: Vec<(String, WadArchive)>) {
        self.wad_archives = archives;
    }
    pub fn set_content_hash(&mut self, hash: [u8; 32]) {
        self.content_hash = hash;
    }
    pub fn set_source_identity(&mut self, id: String) {
        self.source_identity = id;
    }
    pub fn add_diagnostics(&mut self, diags: Vec<BspReport>) {
        self.diagnostics.extend(diags);
    }
    pub fn add_diagnostic(&mut self, diag: BspReport) {
        self.diagnostics.push(diag);
    }

    /// Validate cross-lump indices and return errors if any.
    pub fn validate_cross_references(&self) -> Result<(), BspReport> {
        lumps::validate_cross_lump_indices(
            &self.planes,
            &self.vertices,
            &self.nodes,
            &self.leaves,
            &self.faces,
            &self.models,
            &self.edges,
            &self.surfedges,
            &self.markfaces,
            &self.clipnodes,
            &self.texinfos,
        )
    }

    /// Validate profile-specific element counts.
    pub fn validate_profile_limits(&self) -> Vec<BspReport> {
        let mut reports = Vec::new();
        let profile = self.profile;

        let checks: &[(usize, u32, crate::diagnostic::DiagnosticCode, &str)] = &[
            (
                self.vertices.len(),
                profile.max_vertices(),
                crate::diagnostic::DiagnosticCode::StructuralVertexCount,
                "vertices",
            ),
            (
                self.edges.len(),
                profile.max_edges(),
                crate::diagnostic::DiagnosticCode::StructuralEdgeCount,
                "edges",
            ),
            (
                self.surfedges.len(),
                profile.max_surfedges(),
                crate::diagnostic::DiagnosticCode::StructuralSurfedgeCount,
                "surfedges",
            ),
            (
                self.faces.len(),
                profile.max_faces(),
                crate::diagnostic::DiagnosticCode::StructuralFaceCount,
                "faces",
            ),
            (
                self.markfaces.len(),
                profile.max_markfaces(),
                crate::diagnostic::DiagnosticCode::StructuralMarkfaceCount,
                "markfaces",
            ),
            (
                self.nodes.len(),
                profile.max_nodes(),
                crate::diagnostic::DiagnosticCode::StructuralNodeCount,
                "nodes",
            ),
            (
                self.leaves.len(),
                profile.max_leaves(),
                crate::diagnostic::DiagnosticCode::StructuralLeafCount,
                "leaves",
            ),
            (
                self.clipnodes.len(),
                profile.max_clipnodes(),
                crate::diagnostic::DiagnosticCode::StructuralClipnodeCount,
                "clipnodes",
            ),
            (
                self.models.len(),
                profile.max_models(),
                crate::diagnostic::DiagnosticCode::StructuralModelCount,
                "models",
            ),
        ];

        for &(count, max, code, name) in checks {
            if count as u32 > max {
                reports.push(BspReport::fatal(
                    code,
                    format!("{} count {} exceeds profile limit {}", name, count, max),
                ));
            }
        }

        reports
    }

    /// Build the BspWorld, performing final validation first.
    pub fn build(mut self) -> Result<BspWorld, BspReport> {
        // 1. Fatal diagnostics collected by subsystem parsers reject the whole asset.
        if let Some(report) = self.diagnostics.iter().find(|r| r.is_error()) {
            return Err(report.clone());
        }

        // 2. Validate cross-lump references
        self.validate_cross_references()?;

        // 3. Validate profile limits
        let limit_reports = self.validate_profile_limits();
        for r in &limit_reports {
            if r.is_error() {
                return Err(r.clone());
            }
        }
        self.diagnostics.extend(limit_reports);

        Ok(BspWorld {
            profile: self.profile,
            entity_raw: self.entity_raw,
            entities: self.entities,
            planes: self.planes,
            vertices: self.vertices,
            nodes: self.nodes,
            leaves: self.leaves,
            faces: self.faces,
            models: self.models,
            texinfos: self.texinfos,
            edges: self.edges,
            surfedges: self.surfedges,
            markfaces: self.markfaces,
            clipnodes: self.clipnodes,
            miptex_data: self.miptex_data,
            lightmap_data: self.lightmap_data,
            vis_data: self.vis_data,
            bspx: self.bspx,
            bspx_rgb_lighting: self.bspx_rgb_lighting,
            palette: self.palette,
            colored_light_source: self.colored_light_source,
            lit_data: self.lit_data,
            wad_archives: self.wad_archives,
            content_hash: self.content_hash,
            source_identity: self.source_identity,
            diagnostics: self.diagnostics,
        })
    }
}
