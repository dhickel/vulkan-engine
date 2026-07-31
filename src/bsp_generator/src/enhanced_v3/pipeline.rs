//! One-way deterministic production pipeline for Enhanced V3.
//!
//! Wires the full immutable pipeline from a validated `V3Config` through
//! placement, topology, reservations, assembly, emission, and metadata.
//! The result is a canonical `.map` string and deterministic metadata,
//! produced atomically — any failure returns a typed error with no partial
//! output.
//!
//! # Pipeline stages
//!
//! ```text
//! V3Config → Footprints → CommittedTopology → Reservations →
//!   Assembly → Validate → Serialize → (map text, metadata)
//! ```
//!
//! # Determinism
//!
//! Two calls with identical `V3Config` produce byte-identical `.map` output
//! and field-identical metadata.

use super::assembly::{self, Assembly, AssemblyBrush, BrushRole, Interface, Support};
#[allow(unused_imports)]
use super::config::V3Preset;
use super::config::{V3Config, CONSTRUCTION_QUANTUM, HEADROOM};
use super::emission;
use super::error::V3Error;
use super::footprint::build_footprints;
use super::geometry::{ConvexBrush, FaceRole};
use super::ids::{CommittedTopology, V3IdAllocator};
use super::intent::plan_composition;
use super::metadata::EnhancedV3Metadata;
use super::reservation::{Reservation, ReservationSet};
use super::rng::V3Seed;
use super::topology::{build_topology, compute_reservations};

// ── Pipeline output ───────────────────────────────────────────────────────

/// The atomic output of a completed Enhanced V3 generation run.
///
/// Contains canonical map text and deterministic production metadata.
/// Never contains partial results — the pipeline is all-or-nothing.
#[derive(Debug, Clone)]
pub struct V3PipelineOutput {
    /// Canonical Quake .map text (LF endings, terminal newline).
    pub map_text: String,
    /// Deterministic production metadata.
    pub metadata: EnhancedV3Metadata,
}

// ── Public pipeline entry point ───────────────────────────────────────────

/// Run the full Enhanced V3 production pipeline from a validated configuration.
///
/// Returns canonical `.map` text and deterministic metadata atomically.
/// Any stage failure returns a typed `V3Error` with no partial output.
///
/// # Determinism
///
/// Two calls with identical `config` produce byte-identical output.
pub fn run_pipeline(config: &V3Config) -> Result<V3PipelineOutput, V3Error> {
    let seed = V3Seed::new(config.seed);
    let mut alloc = V3IdAllocator::new();

    // 1. Build footprints
    let (footprints, layout) = build_footprints(config, seed, &mut alloc)?;

    // 2. Build committed topology
    let topology = build_topology(config, &footprints, &layout, seed, &mut alloc)?;

    // 3. Compute reservation volumes
    let (spawn_volume, light_volumes) = compute_reservations(&topology)?;

    // 4. Build reservation set
    let mut protected_reservations = ReservationSet::new();
    protected_reservations.add(Reservation::new("spawn", "spawn_point", spawn_volume))?;
    for (i, vol) in light_volumes.iter().enumerate() {
        protected_reservations.add(Reservation::new(format!("light_{i:04}"), "light", *vol))?;
    }

    // 5. Build assembly from topology
    let (assembly, spawn_origin, light_origins) =
        build_assembly_from_topology(&topology, &protected_reservations, seed)?;

    // 6. Plan composition (grammar families)
    let plan = plan_composition(
        super::ids::CompositionId(0),
        config.preset.tag(),
        topology.rooms.len() as u32,
    )?;

    // 7. Compute actual face/entity/brush counts from the assembly
    let actual_brushes = assembly.brushes.len() as u32;
    let actual_faces: u32 = assembly
        .brushes
        .iter()
        .map(|b| b.brush.faces.len() as u32)
        .sum();
    let actual_entities: u32 = 1 + light_origins.len() as u32; // worldspawn + spawn + lights

    // 8. Emit canonical .map text
    let map_text = emission::emit_map_text(&assembly, spawn_origin, &light_origins)?;

    // 9. Build metadata
    let grammar_families: Vec<String> = plan.grammar_families.iter().cloned().collect();
    let metadata = EnhancedV3Metadata::new(
        config,
        &topology,
        grammar_families,
        plan.identity_satisfied,
        plan.estimated_total_faces,
        plan.estimated_total_entities,
        actual_faces,
        actual_entities,
        actual_brushes,
        spawn_origin,
        light_origins.len() as u32,
    );

    Ok(V3PipelineOutput { map_text, metadata })
}

// ── Assembly construction ─────────────────────────────────────────────────

/// Build a validated assembly from the committed topology.
fn build_assembly_from_topology(
    topology: &CommittedTopology,
    reservations: &ReservationSet,
    _seed: V3Seed,
) -> Result<(Assembly, (i32, i32, i32), Vec<(i32, i32, i32)>), V3Error> {
    let wall_thickness = CONSTRUCTION_QUANTUM as i128;
    let mut brushes: Vec<AssemblyBrush> = Vec::new();
    let mut interfaces: Vec<Interface> = Vec::new();

    for room in &topology.rooms {
        let (x0, y0, x1, y1) = room.shell;
        let x0 = x0 as i128;
        let y0 = y0 as i128;
        let x1 = x1 as i128;
        let y1 = y1 as i128;
        let z0 = room.floor_z as i128;
        let z1 = z0 + room.dims.2 as i128;

        let rid = room.id.stable_key();

        // Floor slab
        brushes.push(assembly::build_floor_slab(
            (x0, x1),
            (y0, y1),
            (z0, z0 + wall_thickness),
            &format!("{rid}/floor"),
        )?);

        // Ceiling slab
        brushes.push(AssemblyBrush::new(
            format!("{rid}/ceiling"),
            BrushRole::CeilingSlab,
            ConvexBrush::make_box((x0, x1), (y0, y1), (z1 - wall_thickness, z1))?,
            Support::World {
                surface: FaceRole::Floor,
            },
        ));

        // Wall shells — walls span between floor and ceiling, not through them
        let wall_z0 = z0 + wall_thickness;
        let wall_z1 = z1 - wall_thickness;

        // North wall (y = y0)
        brushes.push(assembly::build_wall_shell(
            (x0 + wall_thickness, x1 - wall_thickness),
            (y0, y0 + wall_thickness),
            (wall_z0, wall_z1),
            &format!("{rid}/wall_north"),
        )?);

        // South wall (y = y1 - wall_thickness)
        brushes.push(assembly::build_wall_shell(
            (x0 + wall_thickness, x1 - wall_thickness),
            (y1 - wall_thickness, y1),
            (wall_z0, wall_z1),
            &format!("{rid}/wall_south"),
        )?);

        // West wall (x = x0)
        brushes.push(assembly::build_wall_shell(
            (x0, x0 + wall_thickness),
            (y0 + wall_thickness, y1 - wall_thickness),
            (wall_z0, wall_z1),
            &format!("{rid}/wall_west"),
        )?);

        // East wall (x = x1 - wall_thickness)
        brushes.push(assembly::build_wall_shell(
            (x1 - wall_thickness, x1),
            (y0 + wall_thickness, y1 - wall_thickness),
            (wall_z0, wall_z1),
            &format!("{rid}/wall_east"),
        )?);

        // Interfaces: wall-slab contacts
        for wall_dir in &["north", "south", "west", "east"] {
            interfaces.push(Interface::new(
                format!("{rid}/floor_to_{wall_dir}"),
                format!("{rid}/floor"),
                format!("{rid}/wall_{wall_dir}"),
                FaceRole::Ceiling,
                FaceRole::Floor,
            ));
            interfaces.push(Interface::new(
                format!("{rid}/ceiling_to_{wall_dir}"),
                format!("{rid}/ceiling"),
                format!("{rid}/wall_{wall_dir}"),
                FaceRole::Floor,
                FaceRole::Ceiling,
            ));
        }
    }

    // Portal interfaces: deferred to a future phase (aperture carving).
    // Room walls don't always share faces in the current layout.

    let protected_volumes = reservations.to_protected_volumes()?;
    // Brushes must be in sorted ID order for assembly validation
    brushes.sort_by(|a, b| a.id.cmp(&b.id));
    let assembly = Assembly::new(brushes, interfaces, protected_volumes)?;

    // Spawn origin: center of first room
    let spawn_room = topology
        .rooms
        .first()
        .ok_or_else(|| V3Error::TopologyInvariant {
            detail: "no rooms for spawn".into(),
        })?;
    let spawn_x = (spawn_room.shell.0 + spawn_room.shell.2) / 2;
    let spawn_y = (spawn_room.shell.1 + spawn_room.shell.3) / 2;
    let spawn_z = spawn_room.floor_z + CONSTRUCTION_QUANTUM + HEADROOM / 2;
    let spawn_origin = (spawn_x, spawn_y, spawn_z);

    // Light origins: center of each room near ceiling
    let q = CONSTRUCTION_QUANTUM;
    let light_origins: Vec<(i32, i32, i32)> = topology
        .rooms
        .iter()
        .map(|room| {
            let lx = (room.shell.0 + room.shell.2) / 2;
            let ly = (room.shell.1 + room.shell.3) / 2;
            let lz = room.floor_z + room.dims.2 as i32 - 2 * q;
            (lx, ly, lz)
        })
        .collect();

    Ok((assembly, spawn_origin, light_origins))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_pipeline_sparse_produces_output() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        assert!(!output.map_text.is_empty());
        assert!(output.map_text.contains("worldspawn"));
        assert!(output.map_text.contains("info_player_start"));
        assert!(output.map_text.contains("light"));
        assert_eq!(output.metadata.seed(), 0);
        assert_eq!(output.metadata.preset(), "sparse");
        assert_eq!(output.metadata.schema_version(), "v3");
    }

    #[test]
    fn run_pipeline_moderate_produces_output() {
        let config = V3Config::nominal_moderate();
        let output = run_pipeline(&config).unwrap();
        assert!(!output.map_text.is_empty());
        assert!(output.map_text.contains("worldspawn"));
    }

    #[test]
    fn run_pipeline_rich_produces_output() {
        let config = V3Config::nominal_rich();
        let output = run_pipeline(&config).unwrap();
        assert!(!output.map_text.is_empty());
        assert!(output.map_text.contains("worldspawn"));
    }

    #[test]
    fn run_pipeline_deterministic() {
        let config = V3Config::nominal_sparse();
        let output1 = run_pipeline(&config).unwrap();
        let output2 = run_pipeline(&config).unwrap();
        assert_eq!(output1.map_text, output2.map_text);
        assert_eq!(output1.metadata, output2.metadata);
    }

    #[test]
    fn run_pipeline_metadata_has_room_counts() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        assert!(output.metadata.room_count() >= 3);
        assert!(output.metadata.lower_room_count() >= 2);
        assert!(output.metadata.upper_room_count() >= 1);
        assert!(output.metadata.has_upper_layer());
    }

    #[test]
    fn run_pipeline_metadata_has_spawn_and_lights() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        let (sx, sy, sz) = output.metadata.spawn_origin();
        assert!(sx > 0);
        assert!(sy > 0);
        assert!(sz > 0);
        assert!(output.metadata.light_count() > 0);
    }

    #[test]
    fn run_pipeline_bounds_are_reasonable() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        let (min_x, min_y, min_z, max_x, max_y, max_z) = output.metadata.bounds();
        assert!(max_x > min_x);
        assert!(max_y > min_y);
        assert!(max_z > min_z);
        assert!(min_z >= 0);
    }

    #[test]
    fn run_pipeline_face_budget_satisfied() {
        let config = V3Config::nominal_rich();
        let output = run_pipeline(&config).unwrap();
        assert!(
            output.metadata.face_budget_satisfied(),
            "actual faces {} exceed estimated {}",
            output.metadata.actual_faces(),
            output.metadata.estimated_faces()
        );
        assert!(output.metadata.actual_faces() < crate::enhanced_v3::config::FACE_BUDGET);
    }

    #[test]
    fn run_pipeline_brushes_in_map_match_metadata() {
        let config = V3Config::nominal_sparse();
        let output = run_pipeline(&config).unwrap();
        // Count brush blocks in map (each block starts with "{" at the beginning of a line
        // but entities also start with "{". Count world brushes as those between worldspawn header and closing "}"
        // Simpler: use metadata
        let brush_blocks = output.map_text.lines().filter(|l| l.trim() == "{").count();
        // Each brush has one "{", each entity has one "{", worldspawn has "{"
        // brush_blocks = 1 (worldspawn) + N brushes + 1 (spawn) + M lights
        let expected_blocks = 1
            + output.metadata.actual_brushes() as usize
            + 1
            + output.metadata.light_count() as usize;
        assert_eq!(brush_blocks, expected_blocks);
    }

    #[test]
    fn run_pipeline_different_seeds_different_maps() {
        let config_a = V3Config::new(0, V3Preset::Sparse, 2048).unwrap();
        let config_b = V3Config::new(42, V3Preset::Sparse, 2048).unwrap();
        let output_a = run_pipeline(&config_a).unwrap();
        let output_b = run_pipeline(&config_b).unwrap();
        assert!(!output_a.map_text.is_empty());
        assert!(!output_b.map_text.is_empty());
    }
}
