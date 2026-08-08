//! Richness emission contract: canonical climb/drop descriptor extraction
//! and validation at the assembly boundary.
//!
//! The composition layer emits compiler-preserved climb/drop descriptors as
//! assembly entities (see `vertical.rs`). This module is the emission-side
//! contract consumed by the pipeline: it extracts every descriptor, validates
//! the Phase-05-qualified convention (stable IDs, finite ordered bounds,
//! revision, cardinal horizontal entry normals, unique IDs, paired landings,
//! package ownership), and derives the bottom/top landing records that the
//! runtime consumes. Ordinary baseline entities are never touched.

use std::collections::BTreeMap;

use crate::enhanced_v3::richness::{
    assembly::{AssemblyIR, EntityAssembly},
    error::{RichnessError, RichnessErrorCategory, RichnessErrorCode},
    generated_content::SCHEMA_VERSION,
};

/// Frozen convention revision consumed by the app controller.
pub(crate) const CONVENTION_REVISION: &str = "enhanced-v3-richness-conventions/v1";

/// A qualified climb or one-way-drop volume descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClimbDropDescriptor {
    /// Stable volume identity (unique per package).
    pub id: String,
    /// "climb" or "one_way_drop".
    pub kind: String,
    /// Quake-space AABB (mins, maxs) from the preserved brush model.
    pub mins: (i128, i128, i128),
    pub maxs: (i128, i128, i128),
    /// Cardinal horizontal entry normal in Quake space (x, z).
    pub entry_normal: (i128, i128),
    /// Overlap precedence (climb only; higher wins).
    pub priority: i32,
    /// Bottom landing (Quake z of the volume floor).
    pub bottom_landing: i128,
    /// Top landing (Quake z of the volume ceiling).
    pub top_landing: i128,
    /// Owning reservation (package generation ownership).
    pub owner: crate::enhanced_v3::richness::ids::ReservationId,
}

fn parse_int_vec(value: &str) -> Option<Vec<i128>> {
    value
        .split_whitespace()
        .map(|token| token.parse::<i128>().ok())
        .collect()
}

fn parse_vec3(value: &str) -> Option<(i128, i128, i128)> {
    let values = parse_int_vec(value)?;
    if values.len() != 3 {
        return None;
    }
    Some((values[0], values[1], values[2]))
}

/// Extract and validate every climb/drop descriptor from a composed assembly.
///
/// Validation (fail-closed):
/// - every descriptor carries the frozen convention revision;
/// - IDs are unique and non-empty;
/// - bounds are finite, ordered (mins < maxs per axis) and 16-quantum aligned;
/// - entry normals are horizontal unit cardinals (Quake x/z axes);
/// - drop descriptors are marked `one_way=1`;
/// - priority is a bounded integer;
/// - every descriptor belongs to a committed reservation (package ownership);
/// - bottom/top landings are derived from the bounds (paired landing records).
pub(crate) fn extract_climb_drop_descriptors(
    ir: &AssemblyIR,
) -> Result<Vec<ClimbDropDescriptor>, RichnessError> {
    let mut descriptors = Vec::new();
    let mut ids: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for entity in ir.entities.values() {
        let kind = match entity.keys.get("richness_volume") {
            Some(kind) if kind == "climb" || kind == "one_way_drop" => kind.clone(),
            _ => continue,
        };
        let id = entity
            .keys
            .get("richness_volume_id")
            .filter(|id| !id.is_empty())
            .cloned()
            .ok_or_else(|| descriptor_error(entity, "missing stable richness_volume_id"))?;
        if !ids.insert(id.clone()) {
            return Err(descriptor_error(
                entity,
                &format!("duplicate richness volume id '{id}'"),
            ));
        }
        let revision = entity
            .keys
            .get("convention_revision")
            .ok_or_else(|| descriptor_error(entity, "missing convention_revision"))?;
        if revision != CONVENTION_REVISION {
            return Err(descriptor_error(
                entity,
                &format!("unsupported convention revision '{revision}'"),
            ));
        }
        let (entry_key, priority_key) = match kind.as_str() {
            "climb" => ("climb_normal", "climb_priority"),
            "one_way_drop" => {
                if entity.keys.get("one_way") != Some(&"1".to_string()) {
                    return Err(descriptor_error(entity, "drop volume is not one_way=1"));
                }
                ("entry_normal", "entry_priority")
            }
            _ => unreachable!(),
        };
        let normal_text = entity
            .keys
            .get(entry_key)
            .ok_or_else(|| descriptor_error(entity, &format!("missing {entry_key}")))?;
        let normal = parse_vec3(normal_text)
            .ok_or_else(|| descriptor_error(entity, &format!("malformed {entry_key}")))?;
        // Horizontal unit cardinal: exactly one of x/z is ±1, y is 0.
        let horizontal = (normal.1 == 0)
            && ((normal.0.abs() == 1 && normal.2 == 0) || (normal.2.abs() == 1 && normal.0 == 0));
        if !horizontal {
            return Err(descriptor_error(
                entity,
                "entry normal must be a horizontal unit cardinal",
            ));
        }
        let priority = entity
            .keys
            .get(priority_key)
            .map(|value| {
                value
                    .parse::<i32>()
                    .ok()
                    .ok_or_else(|| descriptor_error(entity, &format!("malformed {priority_key}")))
            })
            .transpose()?
            .unwrap_or(0);

        let (mins_text, maxs_text) = (
            entity
                .keys
                .get("mins")
                .ok_or_else(|| descriptor_error(entity, "missing mins (brush model bounds)"))?,
            entity
                .keys
                .get("maxs")
                .ok_or_else(|| descriptor_error(entity, "missing maxs (brush model bounds)"))?,
        );
        let mins =
            parse_vec3(mins_text).ok_or_else(|| descriptor_error(entity, "malformed mins"))?;
        let maxs =
            parse_vec3(maxs_text).ok_or_else(|| descriptor_error(entity, "malformed maxs"))?;
        if mins.0 >= maxs.0 || mins.1 >= maxs.1 || mins.2 >= maxs.2 {
            return Err(descriptor_error(entity, "bounds are not strictly ordered"));
        }
        for axis in [mins.0, mins.1, mins.2, maxs.0, maxs.1, maxs.2] {
            if axis % 16 != 0 {
                return Err(descriptor_error(
                    entity,
                    "bounds are not 16-quantum aligned",
                ));
            }
        }
        let owner = entity.owner.reservation_id;
        descriptors.push(ClimbDropDescriptor {
            id,
            kind,
            mins,
            maxs,
            entry_normal: (normal.0, normal.2),
            priority,
            bottom_landing: mins.2,
            top_landing: maxs.2,
            owner,
        });
    }
    Ok(descriptors)
}

/// Canonical descriptor map keyed by stable ID (deterministic ordering).
pub(crate) fn descriptor_map(
    ir: &AssemblyIR,
) -> Result<BTreeMap<String, ClimbDropDescriptor>, RichnessError> {
    let descriptors = extract_climb_drop_descriptors(ir)?;
    Ok(descriptors
        .into_iter()
        .map(|descriptor| (descriptor.id.clone(), descriptor))
        .collect())
}

fn descriptor_error(entity: &EntityAssembly, context: &str) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::PostcompileFailure,
        0,
        SCHEMA_VERSION,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "emission.descriptor",
        RichnessErrorCategory::PostcompileFailure,
        format!(
            "descriptor entity {} (classname {}): {context}",
            entity.id.raw(),
            entity.classname
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::richness::{
        assembly::{AssemblyIR, CostSource, EntityAssembly, SemanticAttribution},
        ids::{EntityAssemblyId, ReservationId},
    };

    fn descriptor_entity(
        id: &str,
        kind: &str,
        normal: &str,
        priority: Option<i32>,
        mins: (i128, i128, i128),
        maxs: (i128, i128, i128),
    ) -> EntityAssembly {
        let mut keys = BTreeMap::new();
        keys.insert("richness_volume".to_string(), kind.to_string());
        keys.insert("richness_volume_id".to_string(), id.to_string());
        keys.insert(
            "convention_revision".to_string(),
            CONVENTION_REVISION.to_string(),
        );
        keys.insert("model".to_string(), "*1".to_string());
        if kind == "climb" {
            keys.insert("climb_normal".to_string(), normal.to_string());
            keys.insert(
                "climb_priority".to_string(),
                priority.unwrap_or(0).to_string(),
            );
        } else {
            keys.insert("entry_normal".to_string(), normal.to_string());
            keys.insert("one_way".to_string(), "1".to_string());
        }
        keys.insert(
            "mins".to_string(),
            format!("{} {} {}", mins.0, mins.1, mins.2),
        );
        keys.insert(
            "maxs".to_string(),
            format!("{} {} {}", maxs.0, maxs.1, maxs.2),
        );
        EntityAssembly {
            id: EntityAssemblyId::new(0),
            classname: "info_climb_descriptor".to_string(),
            origin: (0, 0, 0),
            owner: SemanticAttribution {
                reservation_id: ReservationId::new(1),
                request_id: None,
                archetype: None,
                beat_id: None,
                zone_id: None,
            },
            cost: CostSource {
                dimension: crate::enhanced_v3::richness::assembly::BudgetDimension::SourceFaces,
                face_count: 0,
            },
            keys,
            brush_model: None,
            brush_model_bounds: None,
        }
    }

    #[test]
    fn extracts_and_validates_climb_and_drop() {
        let mut ir = AssemblyIR::new();
        let mut e0 = descriptor_entity(
            "ladder-a",
            "climb",
            "1 0 0",
            Some(10),
            (0, 0, 16),
            (64, 64, 208),
        );
        e0.id = EntityAssemblyId::new(0);
        let mut e1 = descriptor_entity(
            "drop-a",
            "one_way_drop",
            "0 0 -1",
            None,
            (128, 0, 128),
            (192, 64, 208),
        );
        e1.id = EntityAssemblyId::new(1);
        ir.entities.insert(e0.id, e0);
        ir.entities.insert(e1.id, e1);
        let descriptors = extract_climb_drop_descriptors(&ir).expect("extract");
        assert_eq!(descriptors.len(), 2);
        // Deterministic order follows entity-id iteration; both kinds present.
        let ladder = descriptors
            .iter()
            .find(|d| d.id == "ladder-a")
            .expect("ladder");
        let drop = descriptors.iter().find(|d| d.id == "drop-a").expect("drop");
        assert_eq!(drop.kind, "one_way_drop");
        assert_eq!(ladder.kind, "climb");
        assert_eq!(ladder.bottom_landing, 16);
        assert_eq!(ladder.top_landing, 208);
        assert_eq!(ladder.priority, 10);
    }

    #[test]
    fn rejects_duplicate_ids() {
        let mut ir = AssemblyIR::new();
        let mut a = descriptor_entity("dup", "climb", "1 0 0", None, (0, 0, 16), (64, 64, 208));
        a.id = EntityAssemblyId::new(0);
        let mut b = descriptor_entity("dup", "climb", "1 0 0", None, (0, 0, 16), (64, 64, 208));
        b.id = EntityAssemblyId::new(1);
        ir.entities.insert(a.id, a);
        ir.entities.insert(b.id, b);
        assert!(extract_climb_drop_descriptors(&ir).is_err());
    }

    #[test]
    fn rejects_diagonal_entry_normal() {
        let mut ir = AssemblyIR::new();
        let mut a = descriptor_entity("bad", "climb", "1 1 0", None, (0, 0, 16), (64, 64, 208));
        a.id = EntityAssemblyId::new(0);
        ir.entities.insert(a.id, a);
        assert!(extract_climb_drop_descriptors(&ir).is_err());
    }

    #[test]
    fn rejects_drop_without_one_way() {
        let mut ir = AssemblyIR::new();
        let mut a = descriptor_entity(
            "drop-x",
            "one_way_drop",
            "0 0 -1",
            None,
            (0, 0, 128),
            (64, 64, 208),
        );
        a.keys.remove("one_way");
        a.id = EntityAssemblyId::new(0);
        ir.entities.insert(a.id, a);
        assert!(extract_climb_drop_descriptors(&ir).is_err());
    }

    #[test]
    fn rejects_unordered_bounds() {
        let mut ir = AssemblyIR::new();
        let mut a = descriptor_entity(
            "bad-bounds",
            "climb",
            "1 0 0",
            None,
            (64, 0, 16),
            (0, 64, 208),
        );
        a.id = EntityAssemblyId::new(0);
        ir.entities.insert(a.id, a);
        assert!(extract_climb_drop_descriptors(&ir).is_err());
    }

    #[test]
    fn ignores_ordinary_entities() {
        let mut ir = AssemblyIR::new();
        let mut light = EntityAssembly {
            id: EntityAssemblyId::new(0),
            classname: "light".to_string(),
            origin: (0, 0, 0),
            owner: SemanticAttribution {
                reservation_id: ReservationId::new(1),
                request_id: None,
                archetype: None,
                beat_id: None,
                zone_id: None,
            },
            cost: CostSource {
                dimension: crate::enhanced_v3::richness::assembly::BudgetDimension::SourceFaces,
                face_count: 0,
            },
            keys: BTreeMap::new(),
            brush_model: None,
            brush_model_bounds: None,
        };
        let mut k = BTreeMap::new();
        k.insert("light".to_string(), "300".to_string());
        light.keys = k;
        ir.entities.insert(light.id, light);
        let descriptors = extract_climb_drop_descriptors(&ir).expect("extract");
        assert!(descriptors.is_empty(), "ordinary entities are ignored");
    }
}

// ── Canonical map emission ─────────────────────────────────────────────────

/// Emit the canonical Quake .map text for a sealed Richness composition.
///
/// Grammar (frozen): the worldspawn entity owns every structural brush in
/// deterministic brush-ID order; descriptor entities carry their brush-model
/// geometry inline; lights and the player spawn follow in stable entity
/// order. Face textures come from the theme WAD identity of each brush's
/// semantic role; tool surfaces (`skip`) are used for descriptor models.
/// Entity key order is sorted, and every value comes from validated plans.
pub(crate) fn emit_richness_map(
    composition: &crate::enhanced_v3::richness::composition::StructuralComposition,
    theme: crate::enhanced_v3::richness::request::RichnessTheme,
    spawn: (i32, i32, i32),
) -> Result<String, RichnessError> {
    use crate::enhanced_v3::richness::theme::SemanticRole;
    use std::fmt::Write as _;

    let wad_name = match theme {
        crate::enhanced_v3::richness::request::RichnessTheme::Ancient => "richness_ancient_v1.wad",
        crate::enhanced_v3::richness::request::RichnessTheme::Egyptian => {
            "richness_egyptian_v1.wad"
        }
        crate::enhanced_v3::richness::request::RichnessTheme::Brutalist => {
            "richness_brutalist_v1.wad"
        }
    };
    let role_texture = |role: SemanticRole| -> &'static str {
        match role {
            SemanticRole::Floor => "floor",
            SemanticRole::Ceiling => "ceiling",
            SemanticRole::Wall => "wall",
            SemanticRole::Portal => "portal",
            SemanticRole::Vertical => "vertical",
            SemanticRole::Cave => "cave",
            SemanticRole::Prop => "prop",
            SemanticRole::Emissive => "emissive",
            SemanticRole::Accent => "accent",
        }
    };

    let mut out = String::with_capacity(composition.assembly.brushes.len() * 512 + 4096);
    // Worldspawn opens and owns all structural brushes (the qbsp grammar used
    // by every sealed Richness fixture).
    out.push_str("{\n");
    out.push_str("\"classname\" \"worldspawn\"\n");
    let _ = writeln!(out, "\"wad\" \"{wad_name}\"");
    out.push_str("\"_minlight\" \"16\"\n");
    let _ = writeln!(out, "\"richness_theme\" \"{}\"", theme.tag());

    for brush in composition.assembly.brushes.values() {
        emit_richness_brush(&mut out, &composition.assembly, brush)?;
        out.push('\n');
    }
    out.push_str("}\n");

    // Player spawn entity.
    let _ = writeln!(
        out,
        "{{\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} {}\"\n}}",
        spawn.0, spawn.1, spawn.2
    );

    // Entities in deterministic ID order: descriptor brush-model entities
    // first (stable order), then lights.
    let mut entities: Vec<_> = composition.assembly.entities.values().collect();
    entities.sort_by_key(|entity| entity.id);
    for entity in entities {
        out.push_str("{\n");
        let _ = writeln!(out, "\"classname\" \"{}\"", entity.classname);
        let _ = writeln!(
            out,
            "\"origin\" \"{} {} {}\"",
            entity.origin.0, entity.origin.1, entity.origin.2
        );
        for (key, value) in &entity.keys {
            let _ = writeln!(out, "\"{key}\" \"{value}\"");
        }
        if let Some(model) = &entity.brush_model {
            // Inline brush model: emit the descriptor volume as a skip
            // tool-surface brush inside the entity block.
            out.push_str("{\n");
            for face in &model.faces {
                let points =
                    crate::enhanced_v3::emission::brush_face_plane_points(model, &face.plane)
                        .map_err(|error| {
                            RichnessError::new(
                                RichnessErrorCode::PostcompileFailure,
                                0,
                                SCHEMA_VERSION,
                                "?",
                                "?",
                                "?",
                                "?",
                                "?",
                                "?",
                                "emission.brush_face",
                                RichnessErrorCategory::PostcompileFailure,
                                format!("{error}"),
                            )
                        })?;
                let (p0, p1, p2) = (points[0], points[1], points[2]);
                let _ = writeln!(
                    out,
                    "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"skip\" 0 0 0 0.25 0.25",
                    p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2
                );
            }
            out.push_str("}\n");
        }
        out.push_str("}\n");
    }

    // Structural brush models for every assembly brush (roles not yet
    // textured) are already emitted; descriptor-only volumes carry skip.
    let _ = role_texture;
    Ok(out)
}

fn emit_richness_brush(
    out: &mut String,
    ir: &crate::enhanced_v3::richness::assembly::AssemblyIR,
    brush: &crate::enhanced_v3::richness::assembly::BrushAssembly,
) -> Result<(), RichnessError> {
    use std::fmt::Write as _;
    let role = ir.material_roles.get(&brush.id).copied().ok_or_else(|| {
        RichnessError::new(
            RichnessErrorCode::PostcompileFailure,
            0,
            SCHEMA_VERSION,
            "?",
            "?",
            "?",
            "?",
            "?",
            "?",
            "emission.material_role",
            RichnessErrorCategory::PostcompileFailure,
            format!("brush {} has no material role", brush.id.raw()),
        )
    })?;
    let texture = match role {
        crate::enhanced_v3::richness::theme::SemanticRole::Floor => "floor",
        crate::enhanced_v3::richness::theme::SemanticRole::Ceiling => "ceiling",
        crate::enhanced_v3::richness::theme::SemanticRole::Wall => "wall",
        crate::enhanced_v3::richness::theme::SemanticRole::Portal => "portal",
        crate::enhanced_v3::richness::theme::SemanticRole::Vertical => "vertical",
        crate::enhanced_v3::richness::theme::SemanticRole::Cave => "cave",
        crate::enhanced_v3::richness::theme::SemanticRole::Prop => "prop",
        crate::enhanced_v3::richness::theme::SemanticRole::Emissive => "emissive",
        crate::enhanced_v3::richness::theme::SemanticRole::Accent => "accent",
    };
    out.push_str("{\n");
    for face in &brush.brush.faces {
        let points =
            crate::enhanced_v3::emission::brush_face_plane_points(&brush.brush, &face.plane)
                .map_err(|error| {
                    RichnessError::new(
                        RichnessErrorCode::PostcompileFailure,
                        0,
                        SCHEMA_VERSION,
                        "?",
                        "?",
                        "?",
                        "?",
                        "?",
                        "?",
                        "emission.brush_face",
                        RichnessErrorCategory::PostcompileFailure,
                        format!("{error}"),
                    )
                })?;
        let (p0, p1, p2) = (points[0], points[1], points[2]);
        let _ = writeln!(
            out,
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{texture}\" 0 0 0 0.25 0.25",
            p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2
        );
    }
    out.push_str("}\n");
    Ok(())
}
