//! Deterministic grammar materialization for Enhanced V3.
//!
//! The planner produces real feature instances only after their volumes, world
//! support surfaces, protected clearances, and support graph have been proven.

use std::collections::BTreeSet;

use super::config::{GrammarMode, V3Config, CONSTRUCTION_QUANTUM, HEADROOM};
use super::error::V3Error;
use super::ids::{
    CommittedRoom, CommittedTopology, FeatureId, FeatureInstance, InstanceId, PlanOutcome,
    QuantumVolume, RoomId, SupportRelation, SupportSurfaceKind, SurfaceId,
};
use super::rng::{tags, CandidateSelector, V3Seed};

#[derive(Debug, Clone, Copy)]
enum DraftSupport {
    Surface(SupportSurfaceKind),
    Parent(usize),
}

#[derive(Debug, Clone)]
struct FeatureDraft {
    volume: QuantumVolume,
    support: DraftSupport,
    tags: BTreeSet<String>,
}

fn tags(family: &str, role: &str, detail: &str) -> BTreeSet<String> {
    [family.to_string(), role.to_string(), detail.to_string()]
        .into_iter()
        .collect()
}

fn room_interior(room: &CommittedRoom) -> Option<(i32, i32, i32, i32, i32, i32)> {
    let q = CONSTRUCTION_QUANTUM;
    let bounds = (
        room.shell.0 + q,
        room.shell.1 + q,
        room.floor_z + q,
        room.shell.2 - q,
        room.shell.3 - q,
        room.floor_z + room.dims.2 as i32 - q,
    );
    (bounds.0 < bounds.3 && bounds.1 < bounds.4 && bounds.2 < bounds.5).then_some(bounds)
}

fn surface_id(
    topology: &CommittedTopology,
    room: &CommittedRoom,
    kind: SupportSurfaceKind,
    volume: &QuantumVolume,
) -> Result<SurfaceId, V3Error> {
    let direction = if kind == SupportSurfaceKind::Wall {
        if volume.y0 == room.shell.1 + CONSTRUCTION_QUANTUM {
            Some("north")
        } else if volume.y1 == room.shell.3 - CONSTRUCTION_QUANTUM {
            Some("south")
        } else if volume.x0 == room.shell.0 + CONSTRUCTION_QUANTUM {
            Some("west")
        } else if volume.x1 == room.shell.2 - CONSTRUCTION_QUANTUM {
            Some("east")
        } else {
            None
        }
    } else {
        None
    };
    topology
        .surfaces
        .iter()
        .find(|surface| {
            surface.room_id == room.id
                && surface.kind == kind
                && direction.is_none_or(|direction| surface.owner.direction == direction)
        })
        .map(|surface| surface.id)
        .ok_or_else(|| V3Error::CompositionInvariant {
            detail: format!(
                "room {} has no committed {} support surface",
                room.id,
                kind.face()
            ),
        })
}

fn spawn_room(topology: &CommittedTopology) -> Option<RoomId> {
    topology
        .rooms
        .iter()
        .filter(|room| room.layer == 0)
        .max_by_key(|room| {
            (room.shell.2 - room.shell.0) as i64 * (room.shell.3 - room.shell.1) as i64
        })
        .map(|room| room.id)
}

fn point_in_convex_footprint(room: &CommittedRoom, point: (i32, i32)) -> bool {
    let vertices = &room.footprint_vertices;
    if vertices.len() < 3 {
        return false;
    }
    let mut sign = 0i64;
    for index in 0..vertices.len() {
        let a = vertices[index];
        let b = vertices[(index + 1) % vertices.len()];
        let cross = (b.0 - a.0) as i64 * (point.1 - a.1) as i64
            - (b.1 - a.1) as i64 * (point.0 - a.0) as i64;
        if cross == 0 {
            continue;
        }
        let next = cross.signum();
        if sign != 0 && sign != next {
            return false;
        }
        sign = next;
    }
    true
}

fn volume_in_clear_interior(room: &CommittedRoom, volume: &QuantumVolume) -> bool {
    let Some((x0, y0, z0, x1, y1, z1)) = room_interior(room) else {
        return false;
    };
    if volume.x0 < x0
        || volume.x1 > x1
        || volume.y0 < y0
        || volume.y1 > y1
        || volume.z0 < z0
        || volume.z1 > z1
    {
        return false;
    }
    [
        (volume.x0, volume.y0),
        (volume.x0, volume.y1),
        (volume.x1, volume.y0),
        (volume.x1, volume.y1),
    ]
    .into_iter()
    .all(|point| point_in_convex_footprint(room, point))
}

fn intersects_any(volume: &QuantumVolume, protected: &[QuantumVolume]) -> bool {
    protected.iter().any(|other| volume.intersects(other))
}

fn portal_clearance_intersects(
    volume: &QuantumVolume,
    topology: &CommittedTopology,
    room: &CommittedRoom,
) -> bool {
    let q = CONSTRUCTION_QUANTUM;
    topology
        .portals
        .iter()
        .filter(|portal| portal.source_room == room.id || portal.target_room == Some(room.id))
        .filter_map(|portal| {
            let half = portal.width as i32 / 2;
            let wall = if portal.source_room == room.id {
                portal.wall.as_str()
            } else {
                match portal.wall.as_str() {
                    "north" => "south",
                    "south" => "north",
                    "west" => "east",
                    "east" => "west",
                    _ => return None,
                }
            };
            let (x0, y0, x1, y1) = match wall {
                "north" => (
                    portal.anchor.0 - half,
                    room.shell.1,
                    portal.anchor.0 + half,
                    room.shell.1 + 2 * q,
                ),
                "south" => (
                    portal.anchor.0 - half,
                    room.shell.3 - 2 * q,
                    portal.anchor.0 + half,
                    room.shell.3,
                ),
                "west" => (
                    room.shell.0,
                    portal.anchor.1 - half,
                    room.shell.0 + 2 * q,
                    portal.anchor.1 + half,
                ),
                "east" => (
                    room.shell.2 - 2 * q,
                    portal.anchor.1 - half,
                    room.shell.2,
                    portal.anchor.1 + half,
                ),
                _ => return None,
            };
            QuantumVolume::new(
                x0,
                y0,
                room.floor_z + q,
                x1,
                y1,
                room.floor_z + q + HEADROOM,
            )
        })
        .any(|clearance| volume.intersects(&clearance))
}

fn transition_clearance_intersects(
    volume: &QuantumVolume,
    topology: &CommittedTopology,
    room_id: RoomId,
) -> bool {
    topology.transitions.iter().any(|transition| {
        (transition.lower_room == room_id || transition.upper_room == room_id)
            && transition
                .headroom_volumes
                .iter()
                .any(|&(x0, y0, z0, x1, y1, z1)| {
                    QuantumVolume::new(x0, y0, z0, x1, y1, z1)
                        .is_some_and(|clearance| volume.intersects(&clearance))
                })
    })
}

fn feature_is_clear(
    draft: &FeatureDraft,
    room: &CommittedRoom,
    topology: &CommittedTopology,
    protected: &[QuantumVolume],
    accepted: &[FeatureInstance],
) -> bool {
    volume_in_clear_interior(room, &draft.volume)
        && !intersects_any(&draft.volume, protected)
        && !portal_clearance_intersects(&draft.volume, topology, room)
        && !transition_clearance_intersects(&draft.volume, topology, room.id)
        // Portal approach reservations are the actual traversable room route;
        // do not substitute an arbitrary room-centre cross for that topology.
        && !accepted.iter().any(|feature| draft.volume.intersects(&feature.volume))
}

fn portal_chamber(room: &CommittedRoom) -> Option<Vec<FeatureDraft>> {
    let (x0, y0, z0, x1, _y1, z1) = room_interior(room)?;
    let q = CONSTRUCTION_QUANTUM;
    // Room-scaled blades: in rooms ≥ 192 short axis, span full clear
    // floor-to-ceiling height; smaller rooms use headroom-limited pilasters.
    let short_axis = (x1 - x0).min(z1 - z0);
    let large_room = short_axis >= 12 * q;
    let blade_top = if large_room {
        z1
    } else {
        (z0 + HEADROOM).min(z1)
    };
    let blade_thickness = if large_room { 2 * q } else { q };
    let centre = ((x0 + x1) / 2 / q) * q;
    let left = QuantumVolume::new(
        centre - 2 * q,
        y0,
        z0,
        centre - 2 * q + blade_thickness,
        y0 + q,
        blade_top,
    )?;
    let right = QuantumVolume::new(
        centre + 2 * q - blade_thickness,
        y0,
        z0,
        centre + 2 * q,
        y0 + q,
        blade_top,
    )?;
    let role_tag = if large_room { "blade" } else { "pilaster" };
    Some(vec![
        FeatureDraft {
            volume: left,
            support: DraftSupport::Surface(SupportSurfaceKind::Wall),
            tags: tags("portal-chamber", role_tag, "left-frame"),
        },
        FeatureDraft {
            volume: right,
            support: DraftSupport::Surface(SupportSurfaceKind::Wall),
            tags: tags("portal-chamber", role_tag, "right-frame"),
        },
    ])
}

fn buttressed_hall(room: &CommittedRoom) -> Option<Vec<FeatureDraft>> {
    if room.is_chamfered {
        return None;
    }
    let (x0, y0, z0, _x1, y1, z1) = room_interior(room)?;
    let q = CONSTRUCTION_QUANTUM;
    // Room-scaled buttresses: 2-quantum thickness when the shorter axis
    // is ≥ 192, otherwise keep the 1-quantum minimum.
    let short_axis = ((_x1 - x0).min(y1 - y0)).min(z1 - z0);
    let thickness = if short_axis >= 12 * q { 2 * q } else { q };
    let a = QuantumVolume::new(x0, y0, z0, x0 + thickness, y0 + thickness, z1)?;
    let b = QuantumVolume::new(x0, y1 - thickness, z0, x0 + thickness, y1, z1)?;
    Some(vec![
        FeatureDraft {
            volume: a,
            support: DraftSupport::Surface(SupportSurfaceKind::Wall),
            tags: tags("buttressed-hall", "buttress", "north"),
        },
        FeatureDraft {
            volume: b,
            support: DraftSupport::Surface(SupportSurfaceKind::Wall),
            tags: tags("buttressed-hall", "buttress", "south"),
        },
    ])
}

fn column_grove(room: &CommittedRoom, rich: bool, detail_rank: u64) -> Option<Vec<FeatureDraft>> {
    let (x0, y0, z0, x1, y1, z1) = room_interior(room)?;
    let q = CONSTRUCTION_QUANTUM;
    if x1 - x0 < 48 || y1 - y0 < 48 {
        return None;
    }
    if rich {
        if z1 - z0 < 3 * q || x1 - x0 < 64 || y1 - y0 < 48 {
            return None;
        }
        let y = if detail_rank & 1 == 0 { y0 } else { y1 - 32 };
        let a = QuantumVolume::new(x0, y, z0, x0 + 32, y + 32, z0 + 3 * q)?;
        let b = QuantumVolume::new(x0 + q, y, z0 + 3 * q, x0 + 3 * q, y + 32, z0 + 6 * q)?;
        let c = QuantumVolume::new(x0, y, z0 + 6 * q, x0 + 32, y + 32, z0 + 9 * q)?;
        return Some(vec![
            FeatureDraft {
                volume: a,
                support: DraftSupport::Surface(SupportSurfaceKind::Floor),
                tags: tags("column-grove", "twisted", "segment-0"),
            },
            FeatureDraft {
                volume: b,
                support: DraftSupport::Parent(0),
                tags: tags("column-grove", "twisted", "segment-1"),
            },
            FeatureDraft {
                volume: c,
                support: DraftSupport::Parent(1),
                tags: tags("column-grove", "twisted", "segment-2"),
            },
        ]);
    }
    // Room-scaled pillars: 32×32 (2q) when the shorter axis is ≥ 192,
    // otherwise keep the compact 16×16 (1q) form.
    let short_axis = (x1 - x0).min(y1 - y0);
    let pillar_w = if short_axis >= 12 * q { 2 * q } else { q };
    let a = QuantumVolume::new(x0, y0, z0, x0 + pillar_w, y0 + pillar_w, z1)?;
    let b = QuantumVolume::new(x1 - pillar_w, y1 - pillar_w, z0, x1, y1, z1)?;
    Some(vec![
        FeatureDraft {
            volume: a,
            support: DraftSupport::Surface(SupportSurfaceKind::Floor),
            tags: tags("column-grove", "pillar", "northwest"),
        },
        FeatureDraft {
            volume: b,
            support: DraftSupport::Surface(SupportSurfaceKind::Floor),
            tags: tags("column-grove", "pillar", "southeast"),
        },
    ])
}

fn fractured_vault(room: &CommittedRoom) -> Option<Vec<FeatureDraft>> {
    let (x0, y0, z0, x1, y1, z1) = room_interior(room)?;
    let q = CONSTRUCTION_QUANTUM;
    if z1 - z0 < HEADROOM + q || x1 - x0 < 48 || y1 - y0 < 48 {
        return None;
    }
    Some(vec![
        FeatureDraft {
            volume: QuantumVolume::new(x0, y0, z1 - q, x0 + 2 * q, y0 + q, z1)?,
            support: DraftSupport::Surface(SupportSurfaceKind::Ceiling),
            tags: tags("fractured-vault", "fracture", "hanging-0"),
        },
        FeatureDraft {
            volume: QuantumVolume::new(x1 - 2 * q, y1 - q, z1 - 2 * q, x1, y1, z1)?,
            support: DraftSupport::Surface(SupportSurfaceKind::Ceiling),
            tags: tags("fractured-vault", "fracture", "hanging-1"),
        },
        FeatureDraft {
            volume: QuantumVolume::new(x0, y1 - q, z1 - 3 * q, x0 + q, y1, z1)?,
            support: DraftSupport::Surface(SupportSurfaceKind::Ceiling),
            tags: tags("fractured-vault", "fracture", "hanging-2"),
        },
    ])
}

fn terraced_shrine(room: &CommittedRoom) -> Option<Vec<FeatureDraft>> {
    let (x0, y0, z0, x1, y1, z1) = room_interior(room)?;
    let q = CONSTRUCTION_QUANTUM;
    if x1 - x0 < 48 || y1 - y0 < 48 || z1 - z0 < 3 * q {
        return None;
    }
    let base = QuantumVolume::new(x0, y0, z0, x0 + 3 * q, y0 + 3 * q, z0 + q)?;
    let mid = QuantumVolume::new(x0 + q, y0 + q, z0 + q, x0 + 3 * q, y0 + 3 * q, z0 + 2 * q)?;
    let top = QuantumVolume::new(
        x0 + 2 * q,
        y0 + 2 * q,
        z0 + 2 * q,
        x0 + 3 * q,
        y0 + 3 * q,
        z0 + 3 * q,
    )?;
    Some(vec![
        FeatureDraft {
            volume: base,
            support: DraftSupport::Surface(SupportSurfaceKind::Floor),
            tags: tags("terraced-shrine", "terrace", "level-0"),
        },
        FeatureDraft {
            volume: mid,
            support: DraftSupport::Parent(0),
            tags: tags("terraced-shrine", "terrace", "level-1"),
        },
        FeatureDraft {
            volume: top,
            support: DraftSupport::Parent(1),
            tags: tags("terraced-shrine", "terrace", "level-2"),
        },
    ])
}

fn monolithic_chamber(room: &CommittedRoom, detail_rank: u64) -> Option<Vec<Vec<FeatureDraft>>> {
    let (x0, y0, z0, x1, y1, z1) = room_interior(room)?;
    let q = CONSTRUCTION_QUANTUM;
    if x1 - x0 < 64 || y1 - y0 < 48 || z1 - z0 < 4 * q {
        return None;
    }
    let height = 4 * q;
    // One candidate group per corner; the placement loop falls through to
    // the next corner when the ranked choice is blocked by portal or
    // protected-volume clearance.
    let corners = [
        (x0, y0, "northwest"),
        (x1 - 2 * q, y0, "northeast"),
        (x0, y1 - 2 * q, "southwest"),
        (x1 - 2 * q, y1 - 2 * q, "southeast"),
    ];
    let rotated = (0..4).map(|offset| corners[(detail_rank as usize + offset) % corners.len()]);
    Some(
        rotated
            .filter_map(|(mx, my, corner)| {
                QuantumVolume::new(mx, my, z0, mx + 2 * q, my + 2 * q, z0 + height).map(|volume| {
                    vec![FeatureDraft {
                        volume,
                        support: DraftSupport::Surface(SupportSurfaceKind::Floor),
                        tags: tags("monolithic-chamber", "monolith", corner),
                    }]
                })
            })
            .collect(),
    )
}

/// Candidate draft groups for a room. The placement loop tries each group
/// in order and commits the first whose drafts are all clear of protected
/// volumes, portals, transitions, and accepted features. Families that have
/// a single shape return one group; families with deterministic alternatives
/// (e.g. monolith corner choice) return one group per alternative so a
/// clearance-blocked variant can fall through to the next.
fn drafts_for(
    family: &str,
    room: &CommittedRoom,
    detail_rank: u64,
    rich: bool,
) -> Option<Vec<Vec<FeatureDraft>>> {
    match family {
        "portal-chamber" => portal_chamber(room).map(|drafts| vec![drafts]),
        "buttressed-hall" => buttressed_hall(room).map(|drafts| vec![drafts]),
        "column-grove" => column_grove(room, rich, detail_rank).map(|drafts| vec![drafts]),
        "fractured-vault" => fractured_vault(room).map(|drafts| vec![drafts]),
        "terraced-shrine" => terraced_shrine(room).map(|drafts| vec![drafts]),
        "monolithic-chamber" => monolithic_chamber(room, detail_rank),
        _ => None,
    }
}

fn rank_rooms(
    seed: V3Seed,
    family: &str,
    topology: &CommittedTopology,
    spawn: Option<RoomId>,
) -> Vec<(RoomId, u64)> {
    let feature_selector = CandidateSelector::new(seed, tags::COMPOSITION, true);
    let detail_selector = CandidateSelector::new(seed, tags::EMISSION, true);
    let mut rooms: Vec<_> = topology
        .rooms
        .iter()
        .filter(|room| Some(room.id) != spawn)
        .map(|room| {
            let key = format!("{family}/{}", room.id.stable_key());
            let feature_rank = feature_selector.rank_for(key.as_bytes());
            let detail_rank = detail_selector.rank_for(key.as_bytes());
            (room.id, feature_rank, detail_rank)
        })
        .collect();
    rooms.sort_by(|left, right| {
        left.1
            .cmp(&right.1)
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.0.cmp(&right.0))
    });
    rooms
        .into_iter()
        .map(|(id, _, detail)| (id, detail))
        .collect()
}

/// Materialize deterministic grammar instances from committed topology.
pub fn plan_composition(
    seed: V3Seed,
    config: &V3Config,
    topology: &CommittedTopology,
    spawn_volume: &QuantumVolume,
    light_volumes: &[QuantumVolume],
) -> Result<PlanOutcome, V3Error> {
    config.validate()?;
    let preset = config.preset.tag();
    let eligible = config.enabled_grammar_families();
    let available_rooms = topology.rooms.len().saturating_sub(1);
    let baseline = (config.preset.minimum_assemblies() as usize).min(available_rooms);
    let target_assemblies = if config.uses_default_composition() {
        baseline
    } else if config.feature_density <= 0.5 {
        ((baseline as f32 * config.feature_density / 0.5).round() as usize).min(available_rooms)
    } else {
        (baseline
            + ((available_rooms.saturating_sub(baseline)) as f32 * (config.feature_density - 0.5)
                / 0.5)
                .round() as usize)
            .min(available_rooms)
    };

    let family_schedule: Vec<&str> = if config.uses_default_composition() {
        config.preset.required_families().to_vec()
    } else if target_assemblies == 0 {
        Vec::new()
    } else {
        match config.grammar_mode {
            GrammarMode::Single => {
                let family = eligible
                    .iter()
                    .copied()
                    .min_by_key(|family| {
                        let key = format!("single-family/{family}");
                        (
                            seed.candidate_seed(tags::COMPOSITION, key.as_bytes())
                                .u64_at(0),
                            *family,
                        )
                    })
                    .ok_or_else(|| V3Error::CompositionInvariant {
                        detail: "single grammar mode has no eligible family".into(),
                    })?;
                vec![family; target_assemblies]
            }
            GrammarMode::Mixed => (0..target_assemblies)
                .map(|index| eligible[index % eligible.len()])
                .collect(),
        }
    };

    let mut instances = Vec::new();
    let mut families = BTreeSet::new();
    let mut rejected = Vec::new();
    let mut used_rooms = BTreeSet::new();
    let protected: Vec<_> = std::iter::once(*spawn_volume)
        .chain(light_volumes.iter().copied())
        .collect();

    for family in &family_schedule {
        let family = *family;
        let mut placed = false;
        for (room_id, detail_rank) in rank_rooms(seed, family, topology, spawn_room(topology)) {
            if used_rooms.contains(&room_id) {
                continue;
            }
            let room = topology
                .room(room_id)
                .ok_or_else(|| V3Error::TopologyInvariant {
                    detail: format!("candidate room {room_id} disappeared"),
                })?;
            let Some(candidate_groups) = drafts_for(family, room, detail_rank, preset == "rich")
            else {
                continue;
            };
            // Try each deterministic candidate group in order; commit the
            // first whose drafts are all clear.
            let mut committed: Option<Vec<FeatureDraft>> = None;
            for drafts in &candidate_groups {
                if drafts
                    .iter()
                    .all(|draft| feature_is_clear(draft, room, topology, &protected, &instances))
                {
                    committed = Some(drafts.clone());
                    break;
                }
            }
            let Some(drafts) = committed else {
                continue;
            };

            let first_instance = instances.len();
            for draft in &drafts {
                let support = match draft.support {
                    DraftSupport::Surface(kind) => match kind {
                        SupportSurfaceKind::Floor => {
                            SupportRelation::Floor(surface_id(topology, room, kind, &draft.volume)?)
                        }
                        SupportSurfaceKind::Wall => {
                            SupportRelation::Wall(surface_id(topology, room, kind, &draft.volume)?)
                        }
                        SupportSurfaceKind::Ceiling => SupportRelation::Ceiling(surface_id(
                            topology,
                            room,
                            kind,
                            &draft.volume,
                        )?),
                    },
                    DraftSupport::Parent(local_parent) => {
                        let parent =
                            instances
                                .get(first_instance + local_parent)
                                .ok_or_else(|| V3Error::CompositionInvariant {
                                    detail: format!(
                                        "{family} references missing local support {local_parent}"
                                    ),
                                })?;
                        SupportRelation::SupportedBy(parent.id)
                    }
                };
                let index = instances.len() as u32;
                instances.push(FeatureInstance {
                    id: InstanceId(index),
                    feature_id: FeatureId(index),
                    room_id,
                    volume: draft.volume,
                    support: Some(support),
                    tags: draft.tags.clone(),
                    // A box has six authored faces; this deliberately leaves a
                    // large deterministic allowance for compiler splitting.
                    estimated_faces: 24,
                });
            }
            used_rooms.insert(room_id);
            families.insert(family.to_string());
            placed = true;
            break;
        }
        if !placed {
            rejected.push((
                FeatureId(rejected.len() as u32),
                format!("{family}: no legal room"),
            ));
        }
    }

    let minimum_ok = if config.uses_default_composition() {
        families.len() as u32 >= config.preset.minimum_families()
            && used_rooms.len() as u32 >= config.preset.minimum_assemblies()
            && instances.len() as u32 >= config.preset.minimum_feature_brushes()
    } else {
        used_rooms.len() == family_schedule.len()
    };
    if !minimum_ok {
        return Err(V3Error::MinimumIdentityFailure {
            preset: preset.to_string(),
            required: family_schedule.len() as u32,
            actual: used_rooms.len() as u32,
        });
    }

    let support_edges: Vec<_> = instances
        .iter()
        .map(|instance| {
            let support =
                instance
                    .support
                    .clone()
                    .ok_or_else(|| V3Error::CompositionInvariant {
                        detail: format!("feature {} has no support", instance.id),
                    })?;
            if let SupportRelation::SupportedBy(parent) = support {
                if !instances.iter().any(|instance| instance.id == parent) {
                    return Err(V3Error::CompositionInvariant {
                        detail: format!(
                            "feature {} references missing parent {parent}",
                            instance.id
                        ),
                    });
                }
                return Ok((instance.id, SupportRelation::SupportedBy(parent)));
            }
            Ok((instance.id, support))
        })
        .collect::<Result<_, V3Error>>()?;

    // The preset face ceiling is a conservative source estimate.  The pipeline
    // verifies it against the actual assembly count before it can become metadata.
    Ok(PlanOutcome {
        composition_id: super::ids::CompositionId(0),
        preset: preset.to_string(),
        grammar_families: families,
        instances,
        simplified: Vec::new(),
        rejected,
        support_edges,
        identity_satisfied: true,
        estimated_total_faces: if config.has_overrides() {
            super::config::FACE_BUDGET - 1
        } else {
            config.preset.face_budget()
        },
        estimated_total_entities: light_volumes.len() as u32 + 2,
    })
}

#[cfg(test)]
mod tests {
    use super::super::footprint::build_footprints;
    use super::super::ids::V3IdAllocator;
    use super::super::topology::{build_topology, compute_reservations};
    use super::*;

    fn plan(
        preset: super::super::config::V3Preset,
        seed: u64,
        extent: u32,
    ) -> Result<PlanOutcome, V3Error> {
        let config = V3Config::new(seed, preset, extent)?;
        let mut allocator = V3IdAllocator::new();
        let (footprints, layout) = build_footprints(&config, V3Seed::new(seed), &mut allocator)?;
        let topology = build_topology(
            &config,
            &footprints,
            &layout,
            V3Seed::new(seed),
            &mut allocator,
        )?;
        let (spawn, lights) = compute_reservations(&topology)?;
        plan_composition(V3Seed::new(seed), &config, &topology, &spawn, &lights)
    }

    #[test]
    fn exact_preset_family_and_feature_contracts() {
        let sparse = plan(super::super::config::V3Preset::Sparse, 42, 2048).unwrap();
        assert_eq!(
            sparse.grammar_families,
            ["portal-chamber".to_string()].into_iter().collect()
        );
        assert!(sparse.instances.len() >= 2);
        let moderate = plan(super::super::config::V3Preset::Moderate, 42, 2048).unwrap();
        assert_eq!(moderate.grammar_families.len(), 3);
        assert!(moderate.instances.len() >= 6);
        let rich = plan(super::super::config::V3Preset::Rich, 42, 2048).unwrap();
        assert_eq!(rich.grammar_families.len(), 6);
        assert!(rich.instances.len() >= 12);
    }

    #[test]
    fn identifiers_are_unique_and_supports_are_real() {
        let outcome = plan(super::super::config::V3Preset::Rich, 42, 2048).unwrap();
        let features: BTreeSet<_> = outcome
            .instances
            .iter()
            .map(|instance| instance.feature_id)
            .collect();
        let instances: BTreeSet<_> = outcome
            .instances
            .iter()
            .map(|instance| instance.id)
            .collect();
        assert_eq!(features.len(), outcome.instances.len());
        assert_eq!(instances.len(), outcome.instances.len());
        for (child, support) in &outcome.support_edges {
            if let SupportRelation::SupportedBy(parent) = support {
                assert!(instances.contains(child));
                assert!(instances.contains(parent));
                assert!(parent < child);
            }
        }
    }

    #[test]
    fn candidate_keyed_selection_replays_and_varies() {
        let a = plan(super::super::config::V3Preset::Rich, 42, 2048).unwrap();
        let b = plan(super::super::config::V3Preset::Rich, 42, 2048).unwrap();
        let c = plan(super::super::config::V3Preset::Rich, 99, 2048).unwrap();
        assert_eq!(a, b);
        assert_ne!(a.instances, c.instances);
    }

    #[test]
    fn rich_has_ceiling_fractures_and_contacting_twist() {
        let outcome = plan(super::super::config::V3Preset::Rich, 42, 2048).unwrap();
        assert!(outcome
            .instances
            .iter()
            .any(|instance| instance.tags.contains("fractured-vault")));
        let twist: Vec<_> = outcome
            .instances
            .iter()
            .filter(|instance| instance.tags.contains("twisted"))
            .collect();
        assert_eq!(twist.len(), 3);
        assert_eq!(twist[0].volume.z1, twist[1].volume.z0);
        assert_eq!(twist[1].volume.z1, twist[2].volume.z0);
        assert_ne!(twist[0].volume.x0, twist[1].volume.x0);
    }

    #[test]
    fn empty_topology_is_a_typed_minimum_identity_failure() {
        let config = V3Config::nominal_sparse();
        let volume = QuantumVolume::new(0, 0, 0, 16, 16, 16).unwrap();
        let result = plan_composition(
            V3Seed::new(0),
            &config,
            &CommittedTopology {
                rooms: vec![],
                surfaces: vec![],
                portals: vec![],
                routes: vec![],
                transitions: vec![],
            },
            &volume,
            &[],
        );
        assert!(matches!(
            result,
            Err(V3Error::MinimumIdentityFailure { .. })
        ));
    }

    #[test]
    fn all_36_preset_seed_extent_plans_succeed() {
        for preset in [
            super::super::config::V3Preset::Sparse,
            super::super::config::V3Preset::Moderate,
            super::super::config::V3Preset::Rich,
        ] {
            for seed in [0, 42, 99, 255] {
                let extents: &[u32] = match preset {
                    super::super::config::V3Preset::Sparse => &[1024, 2048, 3072],
                    super::super::config::V3Preset::Moderate => &[2048, 3072],
                    super::super::config::V3Preset::Rich => &[3072],
                };
                for &extent in extents {
                    plan(preset, seed, extent)
                        .unwrap_or_else(|error| panic!("{preset:?}/{seed}/{extent}: {error}"));
                }
            }
        }
    }
}
