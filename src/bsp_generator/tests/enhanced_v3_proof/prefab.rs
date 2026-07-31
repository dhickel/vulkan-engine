//! Compile grounded assemblies from typed dimensions and plan outcomes.
//!
//! Transforms plan FeatureInstances into validated AssemblyBrush trees
//! with Floor/Wall/Ceiling/SupportedBy support, positive-area contact,
//! and acyclic support graph. Uses only the Phase 04 geometry kernel.

use std::collections::{BTreeMap, BTreeSet};

use super::assembly::{Assembly, AssemblyBrush, BrushRole, Interface, Support};
use super::contract::{self, ContractError};
use super::geometry::{ConvexBrush, FaceRole};
use super::ir::{InstanceId, PlanOutcome, SupportRelation};

/// Compile plan outcome instances into a validated Assembly.
pub fn compile_assembly(outcome: &PlanOutcome) -> Result<Assembly, ContractError> {
    if !outcome.identity_satisfied {
        return Err(ContractError::MinimumIdentityFailure {
            preset: outcome.preset.to_string(),
            required: 1,
            actual: 0,
        });
    }

    let mut brushes: Vec<AssemblyBrush> = Vec::new();
    let mut interfaces: Vec<Interface> = Vec::new();
    let mut id_to_brush: BTreeMap<InstanceId, String> = BTreeMap::new();

    let _q = contract::CONSTRUCTION_QUANTUM as i128;

    for instance in &outcome.instances {
        let brush_id = format!("feature_{:04}", instance.id.raw());

        // Build a ConvexBrush from the quantum volume
        let volume = &instance.volume;
        let brush = ConvexBrush::make_box(
            (volume.x0 as i128, volume.x1 as i128),
            (volume.y0 as i128, volume.y1 as i128),
            (volume.z0 as i128, volume.z1 as i128),
        )
        .map_err(|e| ContractError::InvariantViolation {
            detail: format!("failed to build brush for instance {}: {e}", instance.id),
        })?;

        // Determine brush role
        let role = if instance.tags.contains("grounded-assembly") {
            BrushRole::Feature
        } else {
            BrushRole::Feature
        };

        // Determine support
        let support = match &instance.support {
            Some(SupportRelation::Floor(_)) => Support::World {
                surface: FaceRole::Floor,
            },
            Some(SupportRelation::Wall(_)) => Support::World {
                surface: FaceRole::NorthWall,
            },
            Some(SupportRelation::Ceiling(_)) => Support::World {
                surface: FaceRole::Ceiling,
            },
            Some(SupportRelation::SupportedBy(parent_id)) => {
                // Find parent brush ID
                let parent_brush_id = id_to_brush
                    .get(parent_id)
                    .cloned()
                    .unwrap_or_else(|| format!("feature_{:04}", parent_id.raw()));
                Support::SupportedBy {
                    brush_id: parent_brush_id.clone(),
                    interface_id: format!("if_{}_on_{}", instance.id.raw(), parent_id.raw()),
                }
            }
            None => Support::World {
                surface: FaceRole::Floor,
            },
        };

        // If SupportedBy, add interface
        if let Support::SupportedBy {
            brush_id: ref parent_id,
            ref interface_id,
        } = support
        {
            interfaces.push(Interface::new(
                interface_id.clone(),
                &brush_id,
                parent_id,
                FaceRole::Floor,
                FaceRole::Ceiling,
            ));
        }

        id_to_brush.insert(instance.id, brush_id.clone());

        brushes.push(AssemblyBrush::new(brush_id, role, brush, support));
    }

    // Sort brushes by ID for canonical ordering
    brushes.sort();

    // Build assembly
    Assembly::new(brushes, interfaces, vec![], vec![]).map_err(|e| {
        ContractError::InvariantViolation {
            detail: format!("assembly validation failed: {e}"),
        }
    })
}

/// Build a room shell assembly (floor, walls, ceiling) for a committed topology.
///
/// Returns brushes for the room with a floor slab, four wall shells, and
/// a ceiling slab. All brushes are world-supported.
pub fn build_room_shell(
    room_id: &str,
    x0: i128,
    y0: i128,
    z0: i128,
    x1: i128,
    y1: i128,
    z1: i128,
    wall_thickness: i128,
) -> Result<(Vec<AssemblyBrush>, Vec<Interface>), ContractError> {
    let mut brushes = Vec::new();
    let mut interfaces = Vec::new();

    // Floor slab
    let floor_id = format!("{room_id}_floor");
    let floor =
        ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z0 + wall_thickness)).map_err(|e| {
            ContractError::InvariantViolation {
                detail: format!("floor brush: {e}"),
            }
        })?;

    brushes.push(AssemblyBrush::new(
        &floor_id,
        BrushRole::FloorSlab,
        floor,
        Support::World {
            surface: FaceRole::Floor,
        },
    ));

    // Ceiling slab
    let ceil_id = format!("{room_id}_ceiling");
    let ceiling =
        ConvexBrush::make_box((x0, x1), (y0, y1), (z1 - wall_thickness, z1)).map_err(|e| {
            ContractError::InvariantViolation {
                detail: format!("ceiling brush: {e}"),
            }
        })?;

    brushes.push(AssemblyBrush::new(
        &ceil_id,
        BrushRole::CeilingSlab,
        ceiling,
        Support::World {
            surface: FaceRole::Ceiling,
        },
    ));

    // Walls: North, South, East, West
    let walls = [
        (
            "north",
            x0 + wall_thickness,
            y0,
            z0 + wall_thickness,
            x1 - wall_thickness,
            y0 + wall_thickness,
            z1 - wall_thickness,
        ),
        (
            "south",
            x0 + wall_thickness,
            y1 - wall_thickness,
            z0 + wall_thickness,
            x1 - wall_thickness,
            y1,
            z1 - wall_thickness,
        ),
        (
            "east",
            x1 - wall_thickness,
            y0 + wall_thickness,
            z0 + wall_thickness,
            x1,
            y1 - wall_thickness,
            z1 - wall_thickness,
        ),
        (
            "west",
            x0,
            y0 + wall_thickness,
            z0 + wall_thickness,
            x0 + wall_thickness,
            y1 - wall_thickness,
            z1 - wall_thickness,
        ),
    ];

    for (dir, wx0, wy0, wz0, wx1, wy1, wz1) in &walls {
        let wall_id = format!("{room_id}_wall_{dir}");
        let wall =
            ConvexBrush::make_box((*wx0, *wx1), (*wy0, *wy1), (*wz0, *wz1)).map_err(|e| {
                ContractError::InvariantViolation {
                    detail: format!("wall {dir} brush: {e}"),
                }
            })?;

        let support = Support::SupportedBy {
            brush_id: floor_id.clone(),
            interface_id: format!("if_{wall_id}_on_floor"),
        };

        brushes.push(AssemblyBrush::new(
            &wall_id,
            BrushRole::WallShell,
            wall,
            support,
        ));

        interfaces.push(Interface::new(
            format!("if_{wall_id}_on_floor"),
            &wall_id,
            &floor_id,
            FaceRole::Floor,
            FaceRole::Ceiling,
        ));
    }

    Ok((brushes, interfaces))
}

/// Build a corridor throat brush connecting two rooms through a portal.
pub fn build_corridor_throat(
    throat_id: &str,
    x0: i128,
    y0: i128,
    z0: i128,
    x1: i128,
    y1: i128,
    z1: i128,
    source_wall_id: &str,
    target_wall_id: &str,
) -> Result<(AssemblyBrush, Vec<Interface>), ContractError> {
    let brush = ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z1)).map_err(|e| {
        ContractError::InvariantViolation {
            detail: format!("corridor throat brush: {e}"),
        }
    })?;

    let ab = AssemblyBrush::new(
        throat_id,
        BrushRole::PortalThroat,
        brush,
        Support::World {
            surface: FaceRole::Floor,
        },
    );

    let interfaces = vec![
        Interface::new(
            format!("if_{throat_id}_src"),
            throat_id,
            source_wall_id,
            FaceRole::WestWall,
            FaceRole::EastWall,
        ),
        Interface::new(
            format!("if_{throat_id}_tgt"),
            throat_id,
            target_wall_id,
            FaceRole::EastWall,
            FaceRole::WestWall,
        ),
    ];

    Ok((ab, interfaces))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::ir::{
        CompositionId, FeatureId, FeatureInstance, InstanceId, QuantumVolume, RoomId,
    };
    use super::*;
    use std::collections::BTreeMap;

    fn make_test_outcome() -> PlanOutcome {
        PlanOutcome {
            composition_id: CompositionId(0),
            preset: "sparse",
            grammar_families: ["portal_chamber".into()].into(),
            instances: vec![FeatureInstance {
                id: InstanceId(0),
                feature_id: FeatureId(0),
                room_id: RoomId(0),
                volume: QuantumVolume::new(32, 32, 16, 96, 96, 80).unwrap(),
                support: Some(SupportRelation::Floor(super::super::ir::SurfaceId(0))),
                tags: {
                    let mut t = BTreeSet::new();
                    t.insert("grounded-assembly".into());
                    t
                },
                estimated_faces: 120,
            }],
            simplified: vec![],
            rejected: BTreeMap::new(),
            support_edges: vec![],
            identity_satisfied: true,
            estimated_total_faces: 120,
            estimated_total_entities: 1,
        }
    }

    #[test]
    fn compile_single_instance_assembly() {
        let outcome = make_test_outcome();
        let assembly = compile_assembly(&outcome).unwrap();
        assert!(assembly.validated);
        assert_eq!(assembly.brushes.len(), 1);
    }

    #[test]
    fn identity_not_satisfied_rejected() {
        let mut outcome = make_test_outcome();
        outcome.identity_satisfied = false;
        assert!(compile_assembly(&outcome).is_err());
    }

    #[test]
    fn room_shell_all_components() {
        let (brushes, interfaces) = build_room_shell("room_0", 0, 0, 0, 128, 128, 176, 16).unwrap();

        // 1 floor + 1 ceiling + 4 walls = 6 brushes
        assert_eq!(brushes.len(), 6);
        // 4 wall-floor interfaces
        assert_eq!(interfaces.len(), 4);

        let roles: BTreeSet<_> = brushes.iter().map(|b| b.role).collect();
        assert!(roles.contains(&BrushRole::FloorSlab));
        assert!(roles.contains(&BrushRole::CeilingSlab));
        assert!(roles.contains(&BrushRole::WallShell));
    }

    #[test]
    fn corridor_throat_builds() {
        let (throat, ifaces) = build_corridor_throat(
            "corr_01",
            112,
            48,
            16,
            128,
            64,
            96,
            "wall_east",
            "wall_west",
        )
        .unwrap();

        assert_eq!(throat.role, BrushRole::PortalThroat);
        assert_eq!(ifaces.len(), 2);
        assert!(throat.support.is_world());
    }
}
