//! Voxel Cave Demo — Interactive Application (Phase 04)
//!
//! Windowed mode: winit event loop, FPS camera with WASD+mouse, noclip toggle
//! (F), manual capture (C), cave generation + MC33 mesh + PBR scene.
//!
//! Headless mode (`--headless`): generates cave, renders at 5 landmark
//! viewpoints, writes PNG captures with enriched JSON sidecars.

mod cave_gen;
mod cli;
mod config;
mod editor;
mod materials;
mod meshers;
mod regeneration;
mod scene_package;
mod telemetry;
mod validate;
mod validation_campaign;

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use glam::{Vec3, Vec4};
use renderer::prelude::{
    AssetManifestMode, AssetPolicyConfig, CaptureTarget, EnvironmentSource, FrameCaptureRequest,
    FrameCaptureSource, FrameCaptureStatus, FrameRenderOutcome, PbrMaterialDesc, PointLight,
    ProceduralMeshData, ProceduralVertex, Renderer, RendererConfig, Scene, VisualTuning,
};
use renderer::{Camera, FPSController};
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::WindowBuilder;

use cave_gen::generators::topology_first::TopologyFirst;
use cave_gen::generators::{AttemptContext, Generator};
use cave_gen::lattice::VoxelWorld;
use cave_gen::rng::PhaseTaggedRng;
use config::{NormalizedConfig, PresentationConfig, ResolvedAssetRef};
use meshers::mc33::Mc33;
use meshers::FieldMesher;
use validate::validate_normalized;

const APP_TITLE: &str = "Voxel Cave Demo";
const NOCLIP_TOGGLE_ACTION: &str = "noclip.toggle";
const CAPTURE_SCREENSHOT_ACTION: &str = "capture.screenshot";
const MAX_POINT_LIGHTS: usize = 16;

// ─── Main entrypoint ──────────────────────────────────────────────────────

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = cli::CliArgs::parse();

    if args.is_v2 {
        // v2 route: validate config and print identities (generation stub)
        if let Err(e) = run_v2(&args) {
            log::error!("{e}");
            std::process::exit(1);
        }
    } else {
        // v1 legacy route: unchanged one-shot path
        let normalized = args.to_v1_normalized();
        let presentation = args.to_v1_presentation();

        if let Err(e) = validate_normalized(&normalized) {
            log::error!("Configuration validation failed: {e}");
            std::process::exit(1);
        }

        log::info!(
            "NormalizedConfig: seed={} resolution={}³ shell={} light_budget={}",
            normalized.seed,
            normalized.resolution,
            normalized.shell_thickness,
            normalized.light_budget
        );

        if let Err(e) = run(normalized, presentation) {
            log::error!("{e}");
            std::process::exit(1);
        }
    }
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// ─── v2 Entrypoint ─────────────────────────────────────────────────────────

fn run_v2(args: &cli::CliArgs) -> Result<(), AppError> {
    use config::{
        compute_geometry_identity, compute_scene_config_identity, get_embedded_preset,
        known_catalog_ids, load_preset, normalize_document, resolve_asset_ref,
        resolve_config_document_path, DocumentSource, LoadedDocument, ResolvedAppConfig,
        RuntimeOptions,
    };
    use validate::{validate_preset_document, validate_runtime_light_budget};

    // 1. Select exactly one complete base and attach source context.
    let (source_name, loaded) = if let Some(ref preset_name) = args.preset {
        let (name, document) = get_embedded_preset(preset_name)
            .ok_or_else(|| AppError::Validation(format!("unknown preset: '{preset_name}'")))?;
        (
            name.to_string(),
            LoadedDocument {
                document,
                source: DocumentSource::Preset {
                    name: name.to_string(),
                },
                source_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            },
        )
    } else if let Some(ref config_path) = args.config {
        let absolute_path = resolve_config_document_path(config_path)
            .map_err(|e| AppError::Validation(format!("failed to resolve config: {e}")))?;
        let document = load_preset(&absolute_path)
            .map_err(|e| AppError::Validation(format!("failed to load config: {e}")))?;
        let source_dir = absolute_path
            .parent()
            .ok_or_else(|| AppError::Validation("config path has no parent directory".into()))?
            .to_path_buf();
        (
            absolute_path.display().to_string(),
            LoadedDocument {
                document,
                source: DocumentSource::ConfigFile {
                    path: absolute_path,
                },
                source_dir,
            },
        )
    } else {
        let (name, document) = get_embedded_preset("default")
            .ok_or_else(|| AppError::Validation("embedded default preset not found".to_string()))?;
        (
            name.to_string(),
            LoadedDocument {
                document,
                source: DocumentSource::Embedded {
                    name: name.to_string(),
                },
                source_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            },
        )
    };
    let LoadedDocument {
        document: mut source_doc,
        source,
        source_dir,
    } = loaded;

    // 2. Normalize and resolve the selected base before merging overrides.
    normalize_document(&mut source_doc)
        .map_err(|e| AppError::Validation(format!("base normalization failed: {e}")))?;
    for reference in [
        &source_doc.materials.wall.albedo,
        &source_doc.materials.wall.normal,
        &source_doc.materials.wall.roughness,
        &source_doc.materials.wall.ao,
        &source_doc.materials.floor.albedo,
        &source_doc.materials.floor.normal,
        &source_doc.materials.floor.roughness,
        &source_doc.materials.floor.ao,
    ] {
        resolve_asset_ref(reference, &source_dir, known_catalog_ids())
            .map_err(|e| AppError::Validation(format!("base asset reference: {e}")))?;
    }

    // 3. Apply only explicitly present CLI overrides.
    if let Some(seed) = args.seed {
        source_doc.generator.seed = seed;
    }
    if let Some(resolution) = args.resolution {
        source_doc.generator.resolution = resolution;
    }
    if let Some(shell_thickness) = args.shell_thickness {
        source_doc.generator.shell_thickness = shell_thickness;
    }
    if let Some(cavern_count) = args.cavern_count {
        source_doc.generator.cavern_count = cavern_count;
    }
    if let Some(tunnel_count) = args.tunnel_count {
        source_doc.generator.tunnel_count = tunnel_count;
    }
    if let Some(tunnel_radius_min) = args.tunnel_radius_min {
        source_doc.generator.tunnel_radius_min = tunnel_radius_min;
    }
    if let Some(tunnel_radius_max) = args.tunnel_radius_max {
        source_doc.generator.tunnel_radius_max = tunnel_radius_max;
    }
    if let Some(cavern_radius_min) = args.cavern_radius_min {
        source_doc.generator.cavern_radius_min = cavern_radius_min;
    }
    if let Some(cavern_radius_max) = args.cavern_radius_max {
        source_doc.generator.cavern_radius_max = cavern_radius_max;
    }
    if let Some(spline_tension) = args.spline_tension {
        source_doc.generator.spline_tension = spline_tension;
    }
    if let Some(roughness) = args.roughness {
        source_doc.generator.roughness = roughness;
    }
    if let Some(maze_density) = args.maze_density {
        source_doc.generator.maze_density = maze_density;
    }
    if let Some(maze_twistiness) = args.maze_twistiness {
        source_doc.generator.maze_twistiness = maze_twistiness;
    }
    if let Some(maze_radius) = args.maze_radius {
        source_doc.generator.maze_radius = maze_radius;
    }
    if let Some(maze_retries) = args.maze_retries {
        source_doc.generator.maze_retries = maze_retries;
    }
    if let Some(maze_search_budget) = args.maze_search_budget {
        source_doc.generator.maze_search_budget = maze_search_budget;
    }
    if let Some(floor_threshold) = args.floor_threshold {
        source_doc.generator.floor_threshold = floor_threshold;
    }
    if let Some(wall_uv_scale) = args.wall_uv_scale {
        source_doc.generator.wall_uv_scale = wall_uv_scale;
    }
    if let Some(floor_uv_scale) = args.floor_uv_scale {
        source_doc.generator.floor_uv_scale = floor_uv_scale;
    }

    // 4. Normalize the merged document again before validation or identity.
    normalize_document(&mut source_doc)
        .map_err(|e| AppError::Validation(format!("float normalization failed: {e}")))?;

    // 5. Resolve the canonical merged asset references.

    let catalog_ids = known_catalog_ids();
    let resolve = |asset_ref: &config::AssetRef| -> Result<config::ResolvedAssetRef, AppError> {
        resolve_asset_ref(asset_ref, &source_dir, catalog_ids)
            .map_err(|e| AppError::Validation(format!("asset reference: {e}")))
    };

    let resolved_wall_albedo = resolve(&source_doc.materials.wall.albedo)?;
    let resolved_wall_normal = resolve(&source_doc.materials.wall.normal)?;
    let resolved_wall_roughness = resolve(&source_doc.materials.wall.roughness)?;
    let resolved_wall_ao = resolve(&source_doc.materials.wall.ao)?;
    let resolved_floor_albedo = resolve(&source_doc.materials.floor.albedo)?;
    let resolved_floor_normal = resolve(&source_doc.materials.floor.normal)?;
    let resolved_floor_roughness = resolve(&source_doc.materials.floor.roughness)?;
    let resolved_floor_ao = resolve(&source_doc.materials.floor.ao)?;

    // 6. Validate canonical document and runtime values.
    let errors = validate_preset_document(&source_doc);
    if !errors.is_empty() {
        for error in &errors {
            log::error!("Validation error: {error}");
        }
        return Err(AppError::Validation(format!(
            "{} validation error(s)",
            errors.len()
        )));
    }
    let light_budget = args.light_budget.unwrap_or(9);
    validate_runtime_light_budget(light_budget)
        .map_err(|e| AppError::Validation(format!("runtime validation: {e}")))?;

    // 7. Construct identities only from validated canonical typed values.
    let geometry_identity = compute_geometry_identity(
        source_doc.generator_version,
        source_doc.rng_version,
        &source_doc.generator,
    );
    let scene_config_identity = compute_scene_config_identity(
        &geometry_identity,
        &source_doc.generator,
        &source_doc.materials.wall,
        &source_doc.materials.floor,
        &resolved_wall_albedo,
        &resolved_wall_normal,
        &resolved_wall_roughness,
        &resolved_wall_ao,
        &resolved_floor_albedo,
        &resolved_floor_normal,
        &resolved_floor_roughness,
        &resolved_floor_ao,
    );

    let runtime = RuntimeOptions {
        light_budget,
        headless: args.headless,
        capture_dir: args.capture_dir.clone(),
        env_path: args.env_path.clone(),
    };

    // 8. Assemble resolved config and dispatch by validated versions.
    let resolved = ResolvedAppConfig {
        document: source_doc.clone(),
        runtime,
        source,
        resolved_wall_albedo,
        resolved_wall_normal,
        resolved_wall_roughness,
        resolved_wall_ao,
        resolved_floor_albedo,
        resolved_floor_normal,
        resolved_floor_roughness,
        resolved_floor_ao,
        geometry_identity: geometry_identity.clone(),
        scene_config_identity: scene_config_identity.clone(),
        asset_digests: Vec::new(),
    };

    if resolved.document.generator_version == config::V1_GENERATOR_VERSION
        && resolved.document.rng_version == config::V1_RNG_VERSION
    {
        let normalized = NormalizedConfig {
            seed: resolved.document.generator.seed,
            resolution: resolved.document.generator.resolution,
            shell_thickness: resolved.document.generator.shell_thickness,
            light_budget: resolved.runtime.light_budget,
        };
        validate_normalized(&normalized).map_err(|error| {
            AppError::Validation(format!("legacy document validation failed: {error}"))
        })?;
        let presentation = PresentationConfig {
            headless: resolved.runtime.headless,
            capture_dir: resolved.runtime.capture_dir.clone(),
            env_path: resolved.runtime.env_path.clone(),
        };
        return run(normalized, presentation);
    }

    // v2 generation
    log::info!("=== v2 Configuration Validated ===");
    log::info!("Source: {source_name}");
    log::info!("Schema version: {}", resolved.document.schema_version);
    log::info!("Generator version: {}", resolved.document.generator_version);
    log::info!("RNG version: {}", resolved.document.rng_version);
    log::info!(
        "GeometryIdentity: {}",
        bytes_to_hex(&resolved.geometry_identity.0)
    );
    log::info!(
        "SceneConfigIdentity: {}",
        bytes_to_hex(&resolved.scene_config_identity.0)
    );
    log::info!("Seed: {}", resolved.document.generator.seed);
    log::info!(
        "Resolution: {}cubed",
        resolved.document.generator.resolution
    );
    log::info!("Cavern count: {}", resolved.document.generator.cavern_count);
    log::info!("Tunnel count: {}", resolved.document.generator.tunnel_count);
    log::info!("Maze density: {}", resolved.document.generator.maze_density);
    log::info!("Light budget: {light_budget} (runtime-only)");

    // ── v2 generation + scene package ───────────────────────────────
    log::info!("Building CPU scene package...");
    let package = scene_package::build_scene_package(&resolved)
        .map_err(|e| AppError::Validation(format!("scene package build failed: {e}")))?;

    log::info!(
        "CPU package built: wall_tris={} floor_tris={} lights={} viewpoints={} total_voxels={}",
        package.wall_triangles,
        package.floor_triangles,
        package.lights.len(),
        package.viewpoints.len(),
        package.total_voxels
    );
    log::info!(
        "Timings: gen={}ms mesh={}ms partition={}ms",
        package.generation_time_ms,
        package.mesh_time_ms,
        package.partition_time_ms
    );

    if resolved.runtime.headless {
        run_headless_v2(&resolved, package)?;
    } else {
        let regen_after = args.regen_after_frames;
        run_windowed_v2(&resolved, package, regen_after)?;
    }

    Ok(())
}

#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("renderer init failed: {0}")]
    RendererInit(#[from] renderer::RendererError),
    #[error("scene error: {0}")]
    Scene(#[from] renderer::SceneError),
    #[error("asset error: {0}")]
    Asset(#[from] renderer::AssetError),
    #[error("capture config error: {0}")]
    CaptureConfig(String),
    #[error("event loop error: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error("window error: {0}")]
    Window(#[from] winit::error::OsError),
    #[error("mesher error: {0}")]
    Mesher(String),
    #[error("validation error: {0}")]
    Validation(String),
}

// ─── Shared types ──────────────────────────────────────────────────────────

/// Generated scene data held across frames.
struct CaveScene {
    /// Uploaded mesh handle.
    mesh_handle: renderer::MeshHandle,
    /// Our stone material handle.
    material: renderer::MaterialHandle,
    /// Point light IDs for validation.
    light_ids: Vec<renderer::PointLightId>,
    /// Generator result for viewpoint and metadata access.
    sites: Vec<cave_gen::metrics::Site>,
    /// Density lattice for in-air checks.
    world: VoxelWorld,
}

// ─── Run ───────────────────────────────────────────────────────────────────

fn run(normalized: NormalizedConfig, presentation: PresentationConfig) -> Result<(), AppError> {
    // ── 1. Generate cave ────────────────────────────────────────────────
    let res = normalized.resolution;
    let t_gen = std::time::Instant::now();
    let mut world = VoxelWorld::new(res, res, res);
    world.fill_solid();

    let mut rng = PhaseTaggedRng::new(normalized.seed);
    let mut ctx = AttemptContext::new();
    let gen = TopologyFirst;
    let gen_result = gen.generate(&mut world, &mut rng, &mut ctx);

    let gen_time = t_gen.elapsed();
    log::info!(
        "Generator: {} sites, {} edges, {}ms",
        gen_result.sites.len(),
        gen_result.edges.len(),
        gen_time.as_millis()
    );
    for site in &gen_result.sites {
        log::info!(
            "  Site '{}': ({}, {}, {})",
            site.label,
            site.x,
            site.y,
            site.z
        );
    }

    // ── 2. Mesh extraction ──────────────────────────────────────────────
    let t_mesh = std::time::Instant::now();
    let mesher = Mc33::new();
    let mesh_result = mesher
        .mesh(world.density())
        .map_err(|e| AppError::Mesher(e.to_string()))?;
    let mesh_time = t_mesh.elapsed();

    let tri_count = mesh_result.indices.len() / 3;
    log::info!(
        "Mesher: {} vertices, {} triangles, {}ms",
        mesh_result.vertices.len(),
        tri_count,
        mesh_time.as_millis()
    );

    // Validate mesh bounds are finite
    for &v in &mesh_result.vertices {
        if !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite() {
            return Err(AppError::Validation(
                "mesh contains non-finite vertex positions".into(),
            ));
        }
    }

    // ── 3. Convert to ProceduralMeshData ────────────────────────────────
    let proc_verts: Vec<ProceduralVertex> = mesh_result
        .vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let n = mesh_result.normals[i];
            let t = mesh_result.tangents[i];
            let uv = mesh_result.uvs[i];
            ProceduralVertex {
                position: Vec3::new(v[0], v[1], v[2]),
                normal: Vec3::new(n[0], n[1], n[2]),
                tangent: Vec4::new(t[0], t[1], t[2], t[3]),
                uv0: glam::Vec2::new(uv[0], uv[1]),
                uv1: glam::Vec2::ZERO,
                color: Vec4::ONE,
            }
        })
        .collect();

    // ── 4. Compute light positions (validated in-air) ───────────────────
    let light_positions = compute_light_positions(&gen_result.sites, &world)?;

    // ── 5. Compute camera viewpoints ────────────────────────────────────
    let viewpoints = compute_viewpoints(&gen_result.sites, &world);

    if presentation.headless {
        run_headless(
            &normalized,
            &presentation,
            proc_verts,
            mesh_result.indices,
            &light_positions,
            &viewpoints,
            &gen_result.sites,
            &world,
        )
    } else {
        run_windowed(
            &normalized,
            &presentation,
            proc_verts,
            mesh_result.indices,
            &light_positions,
            &viewpoints,
            &gen_result.sites,
            &world,
        )
    }
}

// ─── Light positions ───────────────────────────────────────────────────────

/// 9 point lights: 5 site lights + 4 edge lights at landmark positions.
fn compute_light_positions(
    sites: &[cave_gen::metrics::Site],
    world: &VoxelWorld,
) -> Result<Vec<(Vec3, Vec3, f32, f32)>, AppError> {
    let mut lights: Vec<(Vec3, Vec3, f32, f32)> = Vec::new();
    let max_idx = sites.len().saturating_sub(1);

    let light_colors: [Vec3; 5] = [
        Vec3::new(1.0, 0.85, 0.6), // spawn: warm orange
        Vec3::new(0.9, 0.7, 0.5),  // junction: amber
        Vec3::new(0.6, 0.75, 1.0), // grand_cavern: cool blue
        Vec3::new(0.8, 0.9, 0.7),  // shaft: pale green
        Vec3::new(1.0, 0.65, 0.4), // destination: warm orange
    ];

    let intensities: [f32; 5] = [25.0, 18.0, 40.0, 18.0, 25.0];

    // 5 site lights
    for (i, site) in sites.iter().enumerate().take(5) {
        let pos = Vec3::new(site.x as f32, site.y as f32 + 2.0, site.z as f32);
        validate_in_air(&pos, world, &format!("site light '{}'", site.label))?;
        let color = light_colors[i.min(light_colors.len() - 1)];
        let intensity = intensities[i.min(intensities.len() - 1)];
        lights.push((pos, color, intensity, 20.0));
        log::info!(
            "  Light {} at '{}': ({:.1}, {:.1}, {:.1})",
            i + 1,
            site.label,
            pos.x,
            pos.y,
            pos.z
        );
    }

    // 4 edge lights at midpoints
    let edge_pairs: [(usize, usize); 4] = [
        (0, 1),              // spawn→junction
        (1, 2),              // junction→grand_cavern
        (2, 4.min(max_idx)), // grand_cavern→destination
        (1, 3.min(max_idx)), // junction→shaft
    ];
    let edge_colors: [Vec3; 4] = [
        Vec3::new(1.0, 0.3, 0.15),
        Vec3::new(0.5, 0.5, 0.8),
        Vec3::new(0.8, 0.6, 0.3),
        Vec3::new(0.4, 0.7, 0.4),
    ];
    let edge_intensities: [f32; 4] = [12.0, 10.0, 10.0, 8.0];

    for (j, &(from, to)) in edge_pairs.iter().enumerate() {
        if from >= sites.len() || to >= sites.len() {
            continue;
        }
        let a = &sites[from];
        let b = &sites[to];
        let pos = Vec3::new(
            (a.x as f32 + b.x as f32) * 0.5,
            (a.y as f32 + b.y as f32) * 0.5 + 1.5,
            (a.z as f32 + b.z as f32) * 0.5,
        );
        validate_in_air(&pos, world, &format!("edge light {j}"))?;
        let color = edge_colors[j];
        let intensity = edge_intensities[j];
        lights.push((pos, color, intensity, 15.0));
        log::info!(
            "  Light {} (edge): ({:.1}, {:.1}, {:.1})",
            j + 6,
            pos.x,
            pos.y,
            pos.z
        );
    }

    if lights.len() > MAX_POINT_LIGHTS {
        return Err(AppError::Validation(format!(
            "light count {} exceeds maximum {}",
            lights.len(),
            MAX_POINT_LIGHTS
        )));
    }

    Ok(lights)
}

// ─── Viewpoints ────────────────────────────────────────────────────────────

/// 5 camera viewpoints derived from site positions.
fn compute_viewpoints(
    sites: &[cave_gen::metrics::Site],
    world: &VoxelWorld,
) -> Vec<(String, Vec3, Vec3)> {
    sites
        .iter()
        .take(5)
        .map(|site| {
            let target = Vec3::new(site.x as f32, site.y as f32, site.z as f32);
            // Move camera outward from the target for a good view
            let eye = Vec3::new(target.x + 8.0, target.y + 3.0, target.z + 8.0);
            // If eye is in solid, pull back further
            let eye = if !is_in_air(&eye, world) {
                Vec3::new(target.x + 12.0, target.y + 6.0, target.z + 12.0)
            } else {
                eye
            };
            (site.label.to_string(), eye, target)
        })
        .collect()
}

// ─── In-air checks ─────────────────────────────────────────────────────────

fn is_in_air(pos: &Vec3, world: &VoxelWorld) -> bool {
    let x = pos.x.round() as i32;
    let y = pos.y.round() as i32;
    let z = pos.z.round() as i32;
    let (w, h, d) = world.dims();
    if x < 0 || y < 0 || z < 0 || x >= w as i32 || y >= h as i32 || z >= d as i32 {
        // Outside bounds → considered air (free space beyond lattice)
        return true;
    }
    *world.density().read(x as u32, y as u32, z as u32) >= 0
}

fn validate_in_air(pos: &Vec3, world: &VoxelWorld, label: &str) -> Result<(), AppError> {
    if !is_in_air(pos, world) {
        return Err(AppError::Validation(format!(
            "{label} at ({:.1}, {:.1}, {:.1}) is inside solid rock",
            pos.x, pos.y, pos.z
        )));
    }
    Ok(())
}

// ─── Scene seeding ─────────────────────────────────────────────────────────

fn seed_scene(
    renderer: &mut Renderer,
    scene: &mut Scene,
    proc_verts: Vec<ProceduralVertex>,
    proc_indices: Vec<u32>,
    light_positions: &[(Vec3, Vec3, f32, f32)],
    env_path: Option<&PathBuf>,
    seed: u64,
    resolution: u32,
) -> Result<CaveScene, AppError> {
    let mut assets = renderer.assets();

    // Create stone PBR material
    let stone_mat = assets.create_material_pbr(PbrMaterialDesc {
        base_color: Vec4::new(0.52, 0.47, 0.42, 1.0), // warm gray stone
        metallic: 0.0,
        roughness: 0.75, // high roughness = cave walls
        ..Default::default()
    })?;

    // Upload mesh
    let mesh_data = ProceduralMeshData {
        name: format!("cave_{}x{}_{seed}", resolution, resolution),
        vertices: proc_verts,
        indices: proc_indices,
        material: Some(stone_mat),
    };
    let mesh_handle = assets.upload_procedural_mesh(mesh_data)?;

    // Build scene graph
    let root = scene.create_node(None, glam::Mat4::IDENTITY)?;
    let cave_node = scene.create_node(Some(root), glam::Mat4::IDENTITY)?;
    scene.add_mesh(cave_node, mesh_handle)?;

    // Place point lights
    let mut light_ids = Vec::new();
    for &(pos, color, intensity, range) in light_positions {
        let id = scene.create_point_light(PointLight {
            position: pos,
            color,
            intensity,
            range,
        })?;
        light_ids.push(id);
    }

    // Load IBL environment
    let fallback_env = PathBuf::from("apps/dungeon_dogfood/assets/sky_maps/indoor_4k.exr");
    let env_path_resolved = env_path.unwrap_or(&fallback_env);
    if env_path_resolved.exists() {
        match assets.load_environment(EnvironmentSource::Auto(env_path_resolved.clone())) {
            Ok(env_handle) => {
                scene.set_skybox(env_handle);
                log::info!("IBL environment loaded: {}", env_path_resolved.display());
            }
            Err(e) => {
                log::warn!(
                    "Failed to load environment {}: {e}",
                    env_path_resolved.display()
                );
                scene.set_skybox(assets.default_environment());
            }
        }
    } else {
        log::warn!(
            "Environment file not found: {}",
            env_path_resolved.display()
        );
        scene.set_skybox(assets.default_environment());
    }

    log::info!(
        "Scene seeded: mesh={} lights={}",
        mesh_handle.slot,
        light_ids.len()
    );

    Ok(CaveScene {
        mesh_handle,
        material: stone_mat,
        light_ids,
        sites: Vec::new(),               // populated by caller
        world: VoxelWorld::new(1, 1, 1), // placeholder, caller sets
    })
}

// ─── Windowed mode ─────────────────────────────────────────────────────────

fn run_windowed(
    normalized: &NormalizedConfig,
    presentation: &PresentationConfig,
    proc_verts: Vec<ProceduralVertex>,
    proc_indices: Vec<u32>,
    light_positions: &[(Vec3, Vec3, f32, f32)],
    viewpoints: &[(String, Vec3, Vec3)],
    _sites: &[cave_gen::metrics::Site],
    _world: &VoxelWorld,
) -> Result<(), AppError> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title(format!(
            "{} — seed={} res={}³",
            APP_TITLE, normalized.seed, normalized.resolution
        ))
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let config = RendererConfig {
        app_name: "voxel_demo".to_string(),
        window_width: 1280,
        window_height: 720,
        validation_layer: false,
        compile_shaders: false,
        shader_debug_mode: renderer::DebugRuntimeMode::Default,
        preload_startup_scene: false,
        visual_tuning: VisualTuning {
            exposure: 4.0,
            gamma: 2.2,
            ibl_ambient_scale: 0.35, // dim IBL
        },
        headless: false,
        asset_policy: AssetPolicyConfig {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
            compression: renderer::api::config::CompressionConfig::default(),
        },
    };

    let mut renderer = Renderer::new(config, &window)?;
    let mut scene = Scene::new();

    let _cave_scene = seed_scene(
        &mut renderer,
        &mut scene,
        proc_verts,
        proc_indices,
        light_positions,
        presentation.env_path.as_ref(),
        normalized.seed,
        normalized.resolution,
    )?;

    // Set up manual capture directory
    let manual_capture_dir = manual_capture_run_dir();
    std::fs::create_dir_all(&manual_capture_dir)
        .map_err(|e| AppError::CaptureConfig(format!("create capture dir: {e}")))?;
    renderer.configure_manual_frame_capture_dir(Some(manual_capture_dir.clone()))?;
    log::info!(
        "Manual captures will be saved under {}",
        manual_capture_dir.display()
    );

    // Initial camera at first viewpoint
    let (initial_eye, _initial_target) = if let Some((_, eye, target)) = viewpoints.first() {
        (*eye, *target)
    } else {
        let center = Vec3::new(
            normalized.resolution as f32 / 2.0,
            normalized.resolution as f32 / 2.0,
            normalized.resolution as f32 / 2.0,
        );
        (center + Vec3::new(10.0, 5.0, 10.0), center)
    };

    let mut camera = Camera::new(initial_eye);

    let mut fps_controller = FPSController::new(0.002, 1.0);
    let mut app_input = engine::input::InputSystem::new();
    install_app_fps_input(&mut app_input);
    let mut noclip = true; // noclip-only mode
    let mut reported_manual_captures: HashSet<PathBuf> = HashSet::new();

    log::info!("Voxel demo initialized, starting event loop");

    window.request_redraw();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        let _routing = match engine::input::route_platform_input_to_app(
            &mut renderer,
            &window,
            &mut app_input,
            &event,
        ) {
            Ok(routing) => routing,
            Err(e) => {
                log::error!("Platform input routing failed: {e}");
                elwt.exit();
                return;
            }
        };

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        log::info!("Close requested, exiting");
                        elwt.exit();
                    }
                    WindowEvent::Resized(new_size) => {
                        if service_window_resize(
                            &mut renderer,
                            new_size.width,
                            new_size.height,
                            "window resize event",
                        ) == WindowResizeOutcome::Terminal
                        {
                            elwt.exit();
                        } else {
                            window.request_redraw();
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let current_size = window.inner_size();
                        match service_window_resize(
                            &mut renderer,
                            current_size.width,
                            current_size.height,
                            "redraw resize retry",
                        ) {
                            WindowResizeOutcome::Ready => {}
                            WindowResizeOutcome::Deferred => {
                                if current_size.width > 0 && current_size.height > 0 {
                                    window.request_redraw();
                                }
                                return;
                            }
                            WindowResizeOutcome::Terminal => {
                                elwt.exit();
                                return;
                            }
                        }

                        // App-owned input has exactly one dispatch boundary per frame.
                        app_input.dispatch_frame();
                        let snapshot = app_input.snapshot();

                        let noclip_toggle = snapshot.action_just_pressed(
                            &engine::input::ActionId::from(NOCLIP_TOGGLE_ACTION),
                        );
                        let capture_screenshot = snapshot.action_just_pressed(
                            &engine::input::ActionId::from(CAPTURE_SCREENSHOT_ACTION),
                        );

                        if noclip_toggle {
                            noclip = !noclip;
                            log::info!("Noclip {}", if noclip { "enabled" } else { "disabled" });
                        }

                        if capture_screenshot {
                            if let Err(e) = renderer.queue_manual_frame_capture(CaptureTarget::Draw)
                            {
                                log::error!("Manual capture failed: {e}");
                            } else {
                                log::info!("Manual draw capture triggered");
                            }
                        }

                        // Update FPS controller
                        fps_controller.update_from_snapshot(
                            snapshot,
                            1.0 / 60.0, // approximate dt
                            &mut camera,
                        );

                        // Validate camera is in air
                        let cam_pos = camera.get_position();
                        if !is_in_air(&cam_pos, _world) {
                            log::warn!(
                                "Camera at ({:.1}, {:.1}, {:.1}) is inside solid; noclip allows it",
                                cam_pos.x,
                                cam_pos.y,
                                cam_pos.z
                            );
                        }

                        // Render
                        renderer.pump_asset_tasks(32).unwrap_or_default();

                        let view = engine::render::camera_view_for_size(
                            &camera,
                            current_size.width,
                            current_size.height,
                        );

                        match renderer.render_scene_with_view(&mut scene, view) {
                            Ok(renderer::FrameRenderOutcome::Rendered)
                            | Ok(renderer::FrameRenderOutcome::SkippedAcquireUnavailable)
                            | Ok(renderer::FrameRenderOutcome::SubmittedNotPresented)
                            | Ok(renderer::FrameRenderOutcome::PresentedSuboptimal) => {}
                            Ok(renderer::FrameRenderOutcome::SkippedResizePending) => {}
                            Err(e) => {
                                log::error!("Render failed: {e}");
                                elwt.exit();
                                return;
                            }
                        }

                        // Log capture status
                        log_manual_capture_status(&renderer, &mut reported_manual_captures);

                        window.request_redraw();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum WindowResizeOutcome {
    Ready,
    Deferred,
    Terminal,
}

fn service_window_resize(
    renderer: &mut Renderer,
    width: u32,
    height: u32,
    context: &str,
) -> WindowResizeOutcome {
    match renderer.resize(width, height) {
        Ok(()) if width == 0 || height == 0 => WindowResizeOutcome::Deferred,
        Ok(()) => WindowResizeOutcome::Ready,
        Err(renderer::RendererError::Frame(
            renderer::api::RendererFrameError::Resize(message),
        )) => {
            log::warn!(
                "{context} deferred for {width}x{height}; swapchain preflight will retry: {message}"
            );
            WindowResizeOutcome::Deferred
        }
        Err(error) => {
            log::error!("{context} failed for {width}x{height}: {error}");
            WindowResizeOutcome::Terminal
        }
    }
}

fn install_app_fps_input(input: &mut engine::input::InputSystem) {
    let mut map = engine::input::ActionMap::new();
    map.bind_key("move.forward", KeyCode::KeyW);
    map.bind_key("move.backward", KeyCode::KeyS);
    map.bind_key("move.left", KeyCode::KeyA);
    map.bind_key("move.right", KeyCode::KeyD);
    map.bind_key("move.up", KeyCode::Space);
    map.bind_key("move.down", KeyCode::ShiftLeft);
    map.bind_key(NOCLIP_TOGGLE_ACTION, KeyCode::KeyF);
    map.bind_key(CAPTURE_SCREENSHOT_ACTION, KeyCode::KeyC);

    input.add_layer(
        engine::input::LayerDescriptor::new(
            "voxel-demo-fps-actions",
            engine::input::LayerPriority(10),
        ),
        map.into_layer(),
    );
}

fn manual_capture_run_dir() -> PathBuf {
    let timestamp_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    PathBuf::from("captures").join(format!(
        "voxel-demo-{timestamp_millis}-pid{}",
        std::process::id()
    ))
}

fn log_manual_capture_status(renderer: &Renderer, reported_paths: &mut HashSet<PathBuf>) {
    match renderer.last_frame_capture_status() {
        Some(FrameCaptureStatus::Succeeded {
            ref output_path,
            source: FrameCaptureSource::Manual,
            ..
        }) if reported_paths.insert(output_path.clone()) => {
            log::info!("Manual draw capture completed: {}", output_path.display());
        }
        Some(FrameCaptureStatus::Failed {
            ref output_path,
            ref message,
            source: FrameCaptureSource::Manual,
            ..
        }) if reported_paths.insert(output_path.clone()) => {
            log::error!(
                "Manual draw capture failed for {}: {}",
                output_path.display(),
                message
            );
        }
        _ => {}
    }
}

// ─── Headless mode ─────────────────────────────────────────────────────────

fn run_headless(
    normalized: &NormalizedConfig,
    presentation: &PresentationConfig,
    proc_verts: Vec<ProceduralVertex>,
    proc_indices: Vec<u32>,
    light_positions: &[(Vec3, Vec3, f32, f32)],
    viewpoints: &[(String, Vec3, Vec3)],
    sites: &[cave_gen::metrics::Site],
    _world: &VoxelWorld,
) -> Result<(), AppError> {
    log::info!("Starting voxel demo headless capture run");

    let config = RendererConfig {
        app_name: "voxel_demo".to_string(),
        window_width: 1920,
        window_height: 1080,
        validation_layer: false,
        compile_shaders: false,
        shader_debug_mode: renderer::DebugRuntimeMode::Default,
        preload_startup_scene: false,
        visual_tuning: VisualTuning {
            exposure: 4.0,
            gamma: 2.2,
            ibl_ambient_scale: 0.35, // dim IBL
        },
        headless: true,
        asset_policy: AssetPolicyConfig {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
            compression: renderer::api::config::CompressionConfig::default(),
        },
    };

    let mut renderer = Renderer::new_headless(config)?;
    let mut scene = Scene::new();

    let _cave_scene = seed_scene(
        &mut renderer,
        &mut scene,
        proc_verts,
        proc_indices,
        light_positions,
        presentation.env_path.as_ref(),
        normalized.seed,
        normalized.resolution,
    )?;

    // Determine capture directory
    let capture_root = presentation.capture_dir.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            ".internal-dev/captures/voxel-demo/s{}_r{}_pid{}",
            normalized.seed,
            normalized.resolution,
            std::process::id()
        ))
    });
    std::fs::create_dir_all(&capture_root)
        .map_err(|e| AppError::CaptureConfig(format!("create capture dir: {e}")))?;
    log::info!("Capture directory: {}", capture_root.display());

    // Warmup frames
    for _ in 0..10 {
        renderer
            .render_scene_headless(&mut scene)
            .map_err(|e| AppError::RendererInit(e))?;
    }
    log::info!("Warmup frames complete");

    // Capture at each viewpoint
    for (label, eye, target) in viewpoints {
        log::info!(
            "Capturing viewpoint '{}': eye=({:.1},{:.1},{:.1}) target=({:.1},{:.1},{:.1})",
            label,
            eye.x,
            eye.y,
            eye.z,
            target.x,
            target.y,
            target.z
        );

        renderer
            .set_camera_look_at(*eye, *target, Vec3::Y)
            .map_err(|e| AppError::RendererInit(e))?;

        let png_path = capture_root.join(format!(
            "cave_s{}_r{}_{}.png",
            normalized.seed, normalized.resolution, label
        ));
        let renderer_sidecar = capture_root.join(format!(
            "cave_s{}_r{}_{}_sidecar.json",
            normalized.seed, normalized.resolution, label
        ));

        let req = FrameCaptureRequest {
            target: CaptureTarget::Draw,
            output_path: png_path.clone(),
            sidecar_path: Some(renderer_sidecar),
        };
        renderer
            .request_frame_capture(req)
            .map_err(|e| AppError::CaptureConfig(e.to_string()))?;

        // Render until capture completes
        let mut captured = false;
        for _attempt in 0..20 {
            renderer
                .render_scene_headless(&mut scene)
                .map_err(|e| AppError::RendererInit(e))?;

            match renderer.last_frame_capture_status() {
                Some(FrameCaptureStatus::Succeeded {
                    ref output_path,
                    width,
                    height,
                    ..
                }) if output_path == &png_path => {
                    let w = *width;
                    let h = *height;
                    log::info!(
                        "  ✓ Captured '{}': {} ({}×{})",
                        label,
                        output_path.display(),
                        w,
                        h
                    );

                    // Write enriched metadata sidecar (separate from renderer's sidecar)
                    let enriched_path = capture_root.join(format!(
                        "cave_s{}_r{}_{}_enriched.json",
                        normalized.seed, normalized.resolution, label
                    ));
                    if let Err(e) = write_enriched_sidecar(
                        &enriched_path,
                        normalized,
                        label,
                        sites,
                        eye,
                        target,
                        w,
                        h,
                    ) {
                        log::warn!("Failed to write enriched sidecar: {e}");
                    }

                    captured = true;
                    break;
                }
                Some(FrameCaptureStatus::Failed {
                    ref output_path,
                    ref message,
                    ..
                }) if output_path == &png_path => {
                    log::error!("  ✗ Capture failed for '{}': {message}", label);
                    return Err(AppError::CaptureConfig(format!(
                        "capture failed for {label}: {message}"
                    )));
                }
                _ => {}
            }
        }
        if !captured {
            return Err(AppError::CaptureConfig(format!(
                "capture for '{label}' did not complete within frame budget"
            )));
        }
    }

    log::info!(
        "All headless captures complete → {}",
        capture_root.display()
    );
    Ok(())
}

/// Write a JSON sidecar with custom metadata (seed, resolution, site names).
fn write_enriched_sidecar(
    path: &PathBuf,
    normalized: &NormalizedConfig,
    viewpoint_label: &str,
    sites: &[cave_gen::metrics::Site],
    eye: &Vec3,
    target: &Vec3,
    width: u32,
    height: u32,
) -> Result<(), AppError> {
    use serde::Serialize;
    #[derive(Serialize)]
    #[serde(deny_unknown_fields)]
    struct EnrichedSidecar<'a> {
        seed: u64,
        resolution: u32,
        viewpoint: &'a str,
        eye: [f32; 3],
        look_at: [f32; 3],
        image_width: u32,
        image_height: u32,
        site_count: usize,
        sites: Vec<SiteSummary<'a>>,
    }
    #[derive(Serialize)]
    #[serde(deny_unknown_fields)]
    struct SiteSummary<'a> {
        label: &'a str,
        x: u32,
        y: u32,
        z: u32,
    }

    let sidecar = EnrichedSidecar {
        seed: normalized.seed,
        resolution: normalized.resolution,
        viewpoint: viewpoint_label,
        eye: [eye.x, eye.y, eye.z],
        look_at: [target.x, target.y, target.z],
        image_width: width,
        image_height: height,
        site_count: sites.len(),
        sites: sites
            .iter()
            .map(|s| SiteSummary {
                label: s.label,
                x: s.x,
                y: s.y,
                z: s.z,
            })
            .collect(),
    };

    let json = serde_json::to_string_pretty(&sidecar)
        .map_err(|e| AppError::CaptureConfig(format!("JSON serialize: {e}")))?;
    std::fs::write(path, json)
        .map_err(|e| AppError::CaptureConfig(format!("write sidecar: {e}")))?;

    Ok(())
}

// ─── v2 Windowed mode ─────────────────────────────────────────────────────

fn is_editor_toggle_key(
    physical_key: PhysicalKey,
    state: ElementState,
    repeat: bool,
) -> bool {
    !repeat
        && state == ElementState::Pressed
        && matches!(physical_key, PhysicalKey::Code(KeyCode::F1 | KeyCode::F2))
}

fn is_editor_toggle_event(event: &Event<()>, expected_window: winit::window::WindowId) -> bool {
    matches!(
        event,
        Event::WindowEvent {
            window_id,
            event: WindowEvent::KeyboardInput { event, .. },
        } if *window_id == expected_window
            && is_editor_toggle_key(event.physical_key, event.state, event.repeat)
    )
}

#[cfg(test)]
mod editor_hotkey_tests {
    use super::*;

    #[test]
    fn f1_and_f2_toggle_only_on_initial_press() {
        for key in [KeyCode::F1, KeyCode::F2] {
            assert!(is_editor_toggle_key(
                PhysicalKey::Code(key),
                ElementState::Pressed,
                false,
            ));
            assert!(!is_editor_toggle_key(
                PhysicalKey::Code(key),
                ElementState::Pressed,
                true,
            ));
            assert!(!is_editor_toggle_key(
                PhysicalKey::Code(key),
                ElementState::Released,
                false,
            ));
        }
        assert!(!is_editor_toggle_key(
            PhysicalKey::Code(KeyCode::F3),
            ElementState::Pressed,
            false,
        ));
    }
}

fn queue_editor_capture_releases(input: &mut engine::input::InputSystem) {
    for code in [
        KeyCode::KeyW,
        KeyCode::KeyS,
        KeyCode::KeyA,
        KeyCode::KeyD,
        KeyCode::Space,
        KeyCode::ShiftLeft,
        KeyCode::KeyF,
        KeyCode::KeyC,
    ] {
        input.queue_event(engine::input::InputEvent::Key {
            code,
            state: ElementState::Released,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });
    }
}

fn set_editor_visible(
    renderer: &mut Renderer,
    window: &winit::window::Window,
    model: &std::rc::Rc<std::cell::RefCell<editor::EditorModel>>,
    visible: bool,
) {
    if visible {
        if renderer.has_app_ui() {
            model.borrow_mut().visible = true;
            if let Err(error) = renderer.refresh_cursor_capture(window) {
                log::error!("Failed to refresh editor cursor capture: {error}");
            }
            return;
        }
        let callback_model = model.clone();
        let callback: renderer::api::AppUiCallback = Box::new(move |ui, context| {
            editor::render_editor_ui(ui, context, &callback_model);
        });
        match renderer.register_app_ui(editor::EDITOR_VIEW_ID, callback) {
            Ok(_) => {
                let mut model = model.borrow_mut();
                model.visible = true;
                model.status_message = Some("Editor shown; camera input is suppressed".into());
                log::info!("Editor shown");
            }
            Err(error) => {
                let mut model = model.borrow_mut();
                model.visible = false;
                model.status_message = Some(format!("Failed to show editor: {error}"));
                log::error!("Failed to register editor UI: {error}");
            }
        }
    } else {
        renderer.unregister_app_ui(&renderer::api::DebugViewId::new(editor::EDITOR_VIEW_ID));
        let mut model = model.borrow_mut();
        model.visible = false;
        model.status_message = Some("Editor hidden; camera input restored".into());
        log::info!("Editor hidden");
    }

    if let Err(error) = renderer.refresh_cursor_capture(window) {
        log::error!("Failed to refresh editor cursor capture: {error}");
    }
}

fn run_windowed_v2(
    resolved: &crate::config::ResolvedAppConfig,
    package: scene_package::CpuScenePackage,
    regen_after_frames: Option<u64>,
) -> Result<(), AppError> {
    let gen_config = &resolved.document.generator;

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title(format!(
            "{} v2 — preset={} seed={} res={}³",
            APP_TITLE,
            match &resolved.source {
                crate::config::DocumentSource::Embedded { name } => name.clone(),
                crate::config::DocumentSource::Preset { name } => name.clone(),
                crate::config::DocumentSource::ConfigFile { path } => {
                    path.display().to_string()
                }
            },
            gen_config.seed,
            gen_config.resolution
        ))
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let config = RendererConfig {
        app_name: "voxel_demo".to_string(),
        window_width: 1280,
        window_height: 720,
        validation_layer: false,
        compile_shaders: false,
        shader_debug_mode: renderer::DebugRuntimeMode::Default,
        preload_startup_scene: false,
        visual_tuning: VisualTuning {
            exposure: 4.0,
            gamma: 2.2,
            ibl_ambient_scale: 0.35,
        },
        headless: false,
        asset_policy: AssetPolicyConfig {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
            compression: renderer::api::config::CompressionConfig::default(),
        },
    };

    let mut renderer = Renderer::new(config, &window)?;
    let mut scene = Scene::new();

    // Load initial materials
    let (wall_mat, floor_mat) = materials::create_wall_floor_materials(
        &mut renderer,
        &resolved.resolved_wall_albedo,
        &resolved.resolved_floor_albedo,
        resolved.document.materials.wall.roughness_factor,
        resolved.document.materials.wall.metallic_factor,
        resolved.document.materials.floor.roughness_factor,
        resolved.document.materials.floor.metallic_factor,
    )
    .map_err(|e| {
        AppError::Asset(renderer::AssetError::Internal(format!(
            "material creation failed: {e}"
        )))
    })?;

    log::info!(
        "Materials loaded: wall={:?} floor={:?}",
        wall_mat.material,
        floor_mat.material
    );

    // Stage initial scene using the regeneration infrastructure
    let presented = regeneration::stage_initial_scene(
        &mut renderer,
        &mut scene,
        &package,
        &wall_mat,
        &floor_mat,
        resolved.runtime.env_path.as_ref(),
    )
    .map_err(|e| AppError::Validation(format!("initial scene staging failed: {e}")))?;

    // Initialize regeneration state with the active package
    let mut regen_state = regeneration::RegenerationState::new();
    regen_state.active = Some(presented);

    // ── Editor model ────────────────────────────────────────────────
    let editor_source_dir: std::path::PathBuf = match &resolved.source {
        crate::config::DocumentSource::Embedded { .. }
        | crate::config::DocumentSource::Preset { .. } => {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        }
        crate::config::DocumentSource::ConfigFile { path } => path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from(".")),
    };
    let editor_model = std::rc::Rc::new(std::cell::RefCell::new(editor::EditorModel::new(
        resolved.document.clone(),
        &resolved.source,
        editor_source_dir,
        resolved.runtime.clone(),
        package.scene_config_identity.clone(),
        editor::ActiveStats::from_package(&package),
    )));

    // Register imgui only on this windowed branch. The headless branch never
    // constructs an editor model, callback, regeneration controller, or RNG action.
    set_editor_visible(&mut renderer, &window, &editor_model, true);

    // Set up manual capture directory
    let manual_capture_dir = manual_capture_run_dir();
    std::fs::create_dir_all(&manual_capture_dir)
        .map_err(|e| AppError::CaptureConfig(format!("create capture dir: {e}")))?;
    renderer.configure_manual_frame_capture_dir(Some(manual_capture_dir.clone()))?;

    // Initial camera at spawn viewpoint
    let initial_eye = if let Some(spawn_vp) = package.viewpoints.first() {
        Vec3::new(
            spawn_vp.position[0],
            spawn_vp.position[1],
            spawn_vp.position[2],
        )
    } else {
        let center = Vec3::new(
            gen_config.resolution as f32 / 2.0,
            gen_config.resolution as f32 / 2.0,
            gen_config.resolution as f32 / 2.0,
        );
        center + Vec3::new(10.0, 5.0, 10.0)
    };

    let mut camera = Camera::new(initial_eye);
    let mut fps_controller = FPSController::new(0.002, 1.0);
    let mut app_input = engine::input::InputSystem::new();
    install_app_fps_input(&mut app_input);
    let mut noclip = true;
    let mut reported_manual_captures: HashSet<PathBuf> = HashSet::new();

    // For auto-regen: alternate seed (seed + 1000)
    let regen_seed_alt = resolved.document.generator.seed.wrapping_add(1000);
    let resolved_for_regen = resolved.clone();

    log::info!(
        "Voxel demo v2 initialized (regen_after_frames={:?}), starting event loop",
        regen_after_frames
    );
    window.request_redraw();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        // A registered app UI suppresses app keyboard routing, so process the
        // editor's global visibility shortcuts before renderer routing. These
        // two presses are consumed here and do not toggle built-in debug panels.
        if is_editor_toggle_event(&event, window.id()) {
            let show = !editor_model.borrow().visible;
            if show {
                queue_editor_capture_releases(&mut app_input);
            }
            set_editor_visible(&mut renderer, &window, &editor_model, show);
        } else if let Err(e) = engine::input::route_platform_input_to_app(
            &mut renderer,
            &window,
            &mut app_input,
            &event,
        ) {
            log::error!("Platform input routing failed: {e}");
            elwt.exit();
            return;
        }

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    log::info!("Close requested, exiting");
                    elwt.exit();
                }
                WindowEvent::Resized(new_size) => {
                    if service_window_resize(
                        &mut renderer,
                        new_size.width,
                        new_size.height,
                        "window resize event",
                    ) == WindowResizeOutcome::Terminal
                    {
                        elwt.exit();
                    } else {
                        window.request_redraw();
                    }
                }
                WindowEvent::RedrawRequested => {
                    let current_size = window.inner_size();
                    match service_window_resize(
                        &mut renderer,
                        current_size.width,
                        current_size.height,
                        "redraw resize retry",
                    ) {
                        WindowResizeOutcome::Ready => {}
                        WindowResizeOutcome::Deferred => {
                            if current_size.width > 0 && current_size.height > 0 {
                                window.request_redraw();
                            }
                            return;
                        }
                        WindowResizeOutcome::Terminal => {
                            elwt.exit();
                            return;
                        }
                    }

                    // App-owned input has exactly one dispatch boundary per frame.
                    app_input.dispatch_frame();

                    // ── Regeneration lifecycle ──────────────────────────
                    regen_state.advance_frame();

                    // Auto-regen trigger
                    if let Some(n) = regen_after_frames {
                        if regen_state.frame_index == n && !regen_state.has_pending_work() {
                            log::info!(
                                "Auto-triggering regeneration at frame {} (seed={})",
                                n,
                                regen_seed_alt
                            );
                            let mut alt_config = resolved_for_regen.clone();
                            alt_config.document.generator.seed = regen_seed_alt;
                            regen_state.submit_request(alt_config);
                        }
                    }

                    // Poll worker completion
                    if let Some(result) = regen_state.poll_worker() {
                        let result_id = result.request_id;
                        let result_err = result.error.clone();
                        // Capture accepted presentation data before commit consumes the result.
                        let editor_success = result.package.as_ref().map(|package| {
                            (
                                package.scene_config_identity.clone(),
                                editor::ActiveStats::from_package(package),
                            )
                        });

                        if let Some(ref err_msg) = result_err {
                            log::error!("Regeneration failed (request_id={result_id}): {err_msg}");
                            if let Ok(mut editor) = editor_model.try_borrow_mut() {
                                editor.record_failure(result_id, err_msg.clone());
                            }
                        } else {
                            let frame = regen_state.frame_index;
                            let expected_request_id = regen_state.latest_request_id;
                            match regeneration::commit_replacement(
                                &mut renderer,
                                &mut scene,
                                &mut regen_state,
                                result,
                                expected_request_id,
                                frame,
                            ) {
                                Ok(()) => {
                                    log::info!("Regeneration committed at frame {frame}");
                                    if let (Ok(mut editor), Some((identity, stats))) =
                                        (editor_model.try_borrow_mut(), editor_success)
                                    {
                                        editor.record_success(result_id, identity, stats);
                                    }
                                }
                                Err(e) => {
                                    log::error!("Regeneration commit failed: {e}");
                                    if let Ok(mut editor) = editor_model.try_borrow_mut() {
                                        editor.record_failure(result_id, e.to_string());
                                    }
                                }
                            }
                        }
                    }

                    // Reap retired materials
                    regeneration::reap_retired_materials(&mut renderer, &mut regen_state);

                    // ── Input ───────────────────────────────────────────
                    let snapshot = app_input.snapshot();

                    let noclip_toggle = snapshot
                        .action_just_pressed(&engine::input::ActionId::from(NOCLIP_TOGGLE_ACTION));
                    let capture_screenshot = snapshot.action_just_pressed(
                        &engine::input::ActionId::from(CAPTURE_SCREENSHOT_ACTION),
                    );

                    // Registered app UI owns keyboard/mouse capture. Gameplay
                    // shortcuts and camera updates resume only after unregistration.
                    if !editor_model.borrow().visible {
                        if noclip_toggle {
                            noclip = !noclip;
                            log::info!("Noclip {}", if noclip { "enabled" } else { "disabled" });
                        }

                        if capture_screenshot {
                            if let Err(e) = renderer.queue_manual_frame_capture(CaptureTarget::Draw)
                            {
                                log::error!("Manual capture failed: {e}");
                            } else {
                                log::info!("Manual draw capture triggered");
                            }
                        }

                        fps_controller.update_from_snapshot(snapshot, 1.0 / 60.0, &mut camera);
                    }

                    renderer.pump_asset_tasks(32).unwrap_or_default();

                    // Drain editor commands
                    {
                        let commands = if let Ok(mut model) = editor_model.try_borrow_mut() {
                            model.sync_from_regen_state(&regen_state);
                            model.drain_commands()
                        } else {
                            Vec::new()
                        };
                        for command in commands {
                            if editor::handle_command(command, &editor_model, &mut regen_state) {
                                set_editor_visible(&mut renderer, &window, &editor_model, false);
                            }
                        }
                    }

                    let view = engine::render::camera_view_for_size(
                        &camera,
                        current_size.width,
                        current_size.height,
                    );

                    match renderer.render_scene_with_view(&mut scene, view) {
                        Ok(FrameRenderOutcome::Rendered)
                        | Ok(FrameRenderOutcome::SkippedAcquireUnavailable)
                        | Ok(FrameRenderOutcome::SubmittedNotPresented)
                        | Ok(FrameRenderOutcome::PresentedSuboptimal) => {}
                        Ok(FrameRenderOutcome::SkippedResizePending) => {}
                        Err(e) => {
                            log::error!("Render failed: {e}");
                            elwt.exit();
                            return;
                        }
                    }

                    log_manual_capture_status(&renderer, &mut reported_manual_captures);
                    window.request_redraw();
                }
                _ => {}
            },
            _ => {}
        }
    })?;

    Ok(())
}

// ─── v2 Headless mode ──────────────────────────────────────────────────────

fn run_headless_v2(
    resolved: &crate::config::ResolvedAppConfig,
    package: scene_package::CpuScenePackage,
) -> Result<(), AppError> {
    log::info!("Starting voxel demo v2 headless capture run");

    let config = RendererConfig {
        app_name: "voxel_demo".to_string(),
        window_width: 1920,
        window_height: 1080,
        validation_layer: false,
        compile_shaders: false,
        shader_debug_mode: renderer::DebugRuntimeMode::Default,
        preload_startup_scene: false,
        visual_tuning: VisualTuning {
            exposure: 4.0,
            gamma: 2.2,
            ibl_ambient_scale: 0.35,
        },
        headless: true,
        asset_policy: AssetPolicyConfig {
            manifest_mode: AssetManifestMode::BestEffort,
            allow_filename_heuristics: true,
            compression: renderer::api::config::CompressionConfig::default(),
        },
    };

    let mut renderer = Renderer::new_headless(config)?;
    let mut scene = Scene::new();

    // Load materials
    let (wall_mat, floor_mat) = materials::create_wall_floor_materials(
        &mut renderer,
        &resolved.resolved_wall_albedo,
        &resolved.resolved_floor_albedo,
        resolved.document.materials.wall.roughness_factor,
        resolved.document.materials.wall.metallic_factor,
        resolved.document.materials.floor.roughness_factor,
        resolved.document.materials.floor.metallic_factor,
    )
    .map_err(|e| {
        AppError::Asset(renderer::AssetError::Internal(format!(
            "material creation failed: {e}"
        )))
    })?;

    // Upload meshes
    let mut assets = renderer.assets();
    let root = scene.create_node(None, glam::Mat4::IDENTITY)?;
    let cave_node = scene.create_node(Some(root), glam::Mat4::IDENTITY)?;

    if let Some(ref cpu_mesh) = package.wall_mesh {
        let proc_verts: Vec<ProceduralVertex> = cpu_mesh
            .positions
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let n = cpu_mesh.normals[i];
                let t = cpu_mesh.tangents[i];
                let uv = cpu_mesh.uvs[i];
                let c = cpu_mesh.colors[i];
                ProceduralVertex {
                    position: Vec3::new(p[0], p[1], p[2]),
                    normal: Vec3::new(n[0], n[1], n[2]),
                    tangent: Vec4::new(t[0], t[1], t[2], t[3]),
                    uv0: glam::Vec2::new(uv[0], uv[1]),
                    uv1: glam::Vec2::ZERO,
                    color: Vec4::new(c[0], c[1], c[2], c[3]),
                }
            })
            .collect();
        let mesh_data = ProceduralMeshData {
            name: "cave_wall_v2".to_string(),
            vertices: proc_verts,
            indices: cpu_mesh.indices.clone(),
            material: Some(wall_mat.material),
        };
        let handle = assets.upload_procedural_mesh(mesh_data)?;
        let wall_node = scene.create_node(Some(cave_node), glam::Mat4::IDENTITY)?;
        scene.add_mesh(wall_node, handle)?;
    }

    if let Some(ref cpu_mesh) = package.floor_mesh {
        let proc_verts: Vec<ProceduralVertex> = cpu_mesh
            .positions
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let n = cpu_mesh.normals[i];
                let t = cpu_mesh.tangents[i];
                let uv = cpu_mesh.uvs[i];
                let c = cpu_mesh.colors[i];
                ProceduralVertex {
                    position: Vec3::new(p[0], p[1], p[2]),
                    normal: Vec3::new(n[0], n[1], n[2]),
                    tangent: Vec4::new(t[0], t[1], t[2], t[3]),
                    uv0: glam::Vec2::new(uv[0], uv[1]),
                    uv1: glam::Vec2::ZERO,
                    color: Vec4::new(c[0], c[1], c[2], c[3]),
                }
            })
            .collect();
        let mesh_data = ProceduralMeshData {
            name: "cave_floor_v2".to_string(),
            vertices: proc_verts,
            indices: cpu_mesh.indices.clone(),
            material: Some(floor_mat.material),
        };
        let handle = assets.upload_procedural_mesh(mesh_data)?;
        let floor_node = scene.create_node(Some(cave_node), glam::Mat4::IDENTITY)?;
        scene.add_mesh(floor_node, handle)?;
    }

    // Place point lights
    for light in &package.lights {
        scene.create_point_light(PointLight {
            position: Vec3::new(light.position[0], light.position[1], light.position[2]),
            color: Vec3::new(light.color[0], light.color[1], light.color[2]),
            intensity: light.intensity,
            range: light.range,
        })?;
    }

    // Environment
    let env_path = resolved
        .runtime
        .env_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("apps/dungeon_dogfood/assets/sky_maps/indoor_4k.exr"));
    if env_path.exists() {
        match assets.load_environment(EnvironmentSource::Auto(env_path.clone())) {
            Ok(env_handle) => {
                scene.set_skybox(env_handle);
                log::info!("IBL environment loaded: {}", env_path.display());
            }
            Err(e) => {
                log::warn!("Failed to load environment {}: {e}", env_path.display());
                scene.set_skybox(assets.default_environment());
            }
        }
    } else {
        log::warn!("Environment file not found: {}", env_path.display());
        scene.set_skybox(assets.default_environment());
    }

    // Determine capture directory
    let gen_config = &resolved.document.generator;
    let capture_root = resolved.runtime.capture_dir.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            ".internal-dev/captures/voxel-demo-v2/s{}_r{}_pid{}",
            gen_config.seed,
            gen_config.resolution,
            std::process::id()
        ))
    });
    std::fs::create_dir_all(&capture_root)
        .map_err(|e| AppError::CaptureConfig(format!("create capture dir: {e}")))?;
    log::info!("Capture directory: {}", capture_root.display());

    // Warmup frames
    for _ in 0..10 {
        renderer
            .render_scene_headless(&mut scene)
            .map_err(|e| AppError::RendererInit(e))?;
    }
    log::info!("Warmup frames complete");

    // Capture at each viewpoint
    for vp in &package.viewpoints {
        let eye = Vec3::new(vp.position[0], vp.position[1], vp.position[2]);
        let target = Vec3::new(vp.target[0], vp.target[1], vp.target[2]);

        log::info!(
            "Capturing viewpoint '{}': eye=({:.1},{:.1},{:.1}) target=({:.1},{:.1},{:.1})",
            vp.role,
            eye.x,
            eye.y,
            eye.z,
            target.x,
            target.y,
            target.z
        );

        renderer
            .set_camera_look_at(eye, target, Vec3::Y)
            .map_err(|e| AppError::RendererInit(e))?;

        let png_path = capture_root.join(format!(
            "cave_s{}_r{}_{}.png",
            gen_config.seed, gen_config.resolution, vp.role
        ));
        let renderer_sidecar = capture_root.join(format!(
            "cave_s{}_r{}_{}_sidecar.json",
            gen_config.seed, gen_config.resolution, vp.role
        ));

        let req = FrameCaptureRequest {
            target: CaptureTarget::Draw,
            output_path: png_path.clone(),
            sidecar_path: Some(renderer_sidecar),
        };
        renderer
            .request_frame_capture(req)
            .map_err(|e| AppError::CaptureConfig(e.to_string()))?;

        let mut captured = false;
        for _attempt in 0..20 {
            renderer
                .render_scene_headless(&mut scene)
                .map_err(|e| AppError::RendererInit(e))?;

            match renderer.last_frame_capture_status() {
                Some(FrameCaptureStatus::Succeeded {
                    ref output_path,
                    width,
                    height,
                    ..
                }) if output_path == &png_path => {
                    let w = *width;
                    let h = *height;
                    log::info!(
                        "  ✓ Captured '{}': {} ({}×{})",
                        vp.role,
                        output_path.display(),
                        w,
                        h
                    );

                    // Write enriched metadata sidecar
                    let enriched_path = capture_root.join(format!(
                        "cave_s{}_r{}_{}_enriched.json",
                        gen_config.seed, gen_config.resolution, vp.role
                    ));
                    if let Err(e) = write_enriched_sidecar_v2(
                        &enriched_path,
                        resolved,
                        &vp.role,
                        &package,
                        w,
                        h,
                    ) {
                        log::warn!("Failed to write enriched sidecar: {e}");
                    }

                    captured = true;
                    break;
                }
                Some(FrameCaptureStatus::Failed {
                    ref output_path,
                    ref message,
                    ..
                }) if output_path == &png_path => {
                    log::error!("  ✗ Capture failed for '{}': {message}", vp.role);
                    return Err(AppError::CaptureConfig(format!(
                        "capture failed for {}: {message}",
                        vp.role
                    )));
                }
                _ => {}
            }
        }
        if !captured {
            return Err(AppError::CaptureConfig(format!(
                "capture for '{}' did not complete within frame budget",
                vp.role
            )));
        }
    }

    log::info!(
        "All headless captures complete → {}",
        capture_root.display()
    );
    Ok(())
}

/// Write a JSON sidecar for v2 captures with config, identities, digests, topology, timings.
fn write_enriched_sidecar_v2(
    path: &PathBuf,
    resolved: &crate::config::ResolvedAppConfig,
    viewpoint_label: &str,
    package: &scene_package::CpuScenePackage,
    width: u32,
    height: u32,
) -> Result<(), AppError> {
    use serde::Serialize;

    #[derive(Serialize)]
    #[serde(deny_unknown_fields)]
    struct EnrichedSidecarV2<'a> {
        schema_version: u32,
        generator_version: u32,
        rng_version: u32,
        seed: u64,
        resolution: u32,
        viewpoint: &'a str,
        eye: [f32; 3],
        look_at: [f32; 3],
        image_width: u32,
        image_height: u32,
        geometry_identity: String,
        scene_config_identity: String,
        wall_triangles: usize,
        floor_triangles: usize,
        total_voxels: u64,
        light_count: usize,
        generation_time_ms: u64,
        mesh_time_ms: u64,
        partition_time_ms: u64,
        cavern_count: u32,
        tunnel_count: u32,
        wall_albedo: String,
        floor_albedo: String,
    }

    fn bytes_to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    fn resolved_ref_to_string(r: &ResolvedAssetRef) -> String {
        match r {
            ResolvedAssetRef::Catalog(id) => format!("catalog:{id}"),
            ResolvedAssetRef::Filesystem(p) => format!("filesystem:{}", p.display()),
        }
    }

    // Find the eye/target for this viewpoint from the package
    let vp = package
        .viewpoints
        .iter()
        .find(|v| v.role == viewpoint_label)
        .or_else(|| package.viewpoints.first());

    let (eye, look_at) = if let Some(v) = vp {
        (v.position, v.target)
    } else {
        ([0.0; 3], [0.0; 3])
    };

    let doc = &resolved.document;

    let sidecar = EnrichedSidecarV2 {
        schema_version: doc.schema_version,
        generator_version: doc.generator_version,
        rng_version: doc.rng_version,
        seed: doc.generator.seed,
        resolution: doc.generator.resolution,
        viewpoint: viewpoint_label,
        eye,
        look_at,
        image_width: width,
        image_height: height,
        geometry_identity: bytes_to_hex(&package.geometry_identity.0),
        scene_config_identity: bytes_to_hex(&package.scene_config_identity.0),
        wall_triangles: package.wall_triangles,
        floor_triangles: package.floor_triangles,
        total_voxels: package.total_voxels,
        light_count: package.lights.len(),
        generation_time_ms: package.generation_time_ms,
        mesh_time_ms: package.mesh_time_ms,
        partition_time_ms: package.partition_time_ms,
        cavern_count: doc.generator.cavern_count,
        tunnel_count: doc.generator.tunnel_count,
        wall_albedo: resolved_ref_to_string(&resolved.resolved_wall_albedo),
        floor_albedo: resolved_ref_to_string(&resolved.resolved_floor_albedo),
    };

    let json = serde_json::to_string_pretty(&sidecar)
        .map_err(|e| AppError::CaptureConfig(format!("JSON serialize: {e}")))?;
    std::fs::write(path, json)
        .map_err(|e| AppError::CaptureConfig(format!("write sidecar: {e}")))?;

    Ok(())
}

// ─── Parity test infrastructure ────────────────────────────────────────────

/// Compare output bytes against golden bytes.
fn assert_byte_equality(label: &str, actual: &[u8], expected: &[u8]) -> Result<(), String> {
    if actual == expected {
        return Ok(());
    }
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: length mismatch (actual {} vs expected {})",
            actual.len(),
            expected.len()
        ));
    }
    let first_diff = actual.iter().zip(expected.iter()).position(|(a, e)| a != e);
    match first_diff {
        Some(pos) => Err(format!(
            "{label}: first byte difference at offset {pos} (actual 0x{:02x} vs expected 0x{:02x})",
            actual[pos], expected[pos]
        )),
        None => Ok(()),
    }
}

// ─── Generator parity tests ────────────────────────────────────────────────

#[cfg(test)]
mod generator_parity_tests {
    use super::assert_byte_equality;
    use crate::cave_gen::generators::topology_first::TopologyFirst;
    use crate::cave_gen::generators::{AttemptContext, Generator};
    use crate::cave_gen::lattice::VoxelWorld;
    use crate::cave_gen::rng::PhaseTaggedRng;
    use std::path::Path;

    const RESOLUTION: u32 = 64;
    const SEED_COUNT: u64 = 12;

    fn golden_dir() -> std::path::PathBuf {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/goldens")).to_path_buf()
    }

    fn generate_port(
        seed: u64,
    ) -> (
        Vec<u8>,
        Vec<u8>,
        Vec<crate::cave_gen::metrics::Site>,
        Vec<crate::cave_gen::metrics::RouteEdge>,
    ) {
        let mut world = VoxelWorld::new(RESOLUTION, RESOLUTION, RESOLUTION);
        world.fill_solid();
        let mut rng = PhaseTaggedRng::new(seed);
        let mut ctx = AttemptContext::new();
        let gen = TopologyFirst;
        let result = gen.generate(&mut world, &mut rng, &mut ctx);

        let density: Vec<u8> = world.density().iter().map(|&d| d as u8).collect();
        let material: Vec<u8> = world.material().iter().copied().collect();
        (density, material, result.sites, result.edges)
    }

    #[test]
    fn parity_density_all_seeds() {
        let dir = golden_dir();
        if !dir.is_dir() {
            eprintln!("Golden directory not found — skipping parity test");
            return;
        }

        for seed in 1..=SEED_COUNT {
            let golden_path = dir.join(format!("seed_{:02}_density.bin", seed));
            if !golden_path.exists() {
                eprintln!("SKIP seed {seed}: golden file not found");
                continue;
            }
            let expected = std::fs::read(&golden_path).expect("read golden");
            let (actual, _, _, _) = generate_port(seed);
            match assert_byte_equality(&format!("seed_{seed:02}_density"), &actual, &expected) {
                Ok(()) => {}
                Err(msg) => panic!("{msg}"),
            }
        }
    }

    #[test]
    fn parity_material_all_seeds() {
        let dir = golden_dir();
        if !dir.is_dir() {
            eprintln!("Golden directory not found — skipping parity test");
            return;
        }

        for seed in 1..=SEED_COUNT {
            let golden_path = dir.join(format!("seed_{:02}_material.bin", seed));
            if !golden_path.exists() {
                eprintln!("SKIP seed {seed}: material golden file not found");
                continue;
            }
            let expected = std::fs::read(&golden_path).expect("read golden");
            let (_, actual, _, _) = generate_port(seed);
            match assert_byte_equality(&format!("seed_{seed:02}_material"), &actual, &expected) {
                Ok(()) => {}
                Err(msg) => panic!("{msg}"),
            }
        }
    }

    #[test]
    fn parity_sites_all_seeds() {
        let dir = golden_dir();
        if !dir.is_dir() {
            eprintln!("Golden directory not found — skipping parity test");
            return;
        }

        for seed in 1..=SEED_COUNT {
            let golden_path = dir.join(format!("seed_{:02}_sites.json", seed));
            if !golden_path.exists() {
                eprintln!("SKIP seed {seed}: sites golden file not found");
                continue;
            }
            let expected_json: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&golden_path).expect("read golden"))
                    .expect("parse golden JSON");

            let (_, _, sites, _) = generate_port(seed);

            let expected_spawn = expected_json["spawn_index"].as_u64().unwrap_or(0) as usize;
            assert_eq!(expected_spawn, 0, "seed {seed}: spawn index mismatch");

            let expected_sites = expected_json["sites"].as_array().expect("sites array");
            assert_eq!(
                sites.len(),
                expected_sites.len(),
                "seed {seed}: site count mismatch"
            );
            for (i, site) in sites.iter().enumerate() {
                let es = &expected_sites[i];
                assert_eq!(
                    site.x,
                    es["x"].as_u64().unwrap() as u32,
                    "seed {seed} site {i} x mismatch"
                );
                assert_eq!(
                    site.y,
                    es["y"].as_u64().unwrap() as u32,
                    "seed {seed} site {i} y mismatch"
                );
                assert_eq!(
                    site.z,
                    es["z"].as_u64().unwrap() as u32,
                    "seed {seed} site {i} z mismatch"
                );
                assert_eq!(
                    site.label,
                    es["label"].as_str().unwrap(),
                    "seed {seed} site {i} label mismatch"
                );
            }
        }
    }

    #[test]
    fn parity_edges_all_seeds() {
        let dir = golden_dir();
        if !dir.is_dir() {
            eprintln!("Golden directory not found — skipping parity test");
            return;
        }

        for seed in 1..=SEED_COUNT {
            let golden_path = dir.join(format!("seed_{:02}_edges.json", seed));
            if !golden_path.exists() {
                eprintln!("SKIP seed {seed}: edges golden file not found");
                continue;
            }
            let expected_json: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&golden_path).expect("read golden"))
                    .expect("parse golden JSON");

            let (_, _, _, edges) = generate_port(seed);

            let expected_edges = expected_json["edges"].as_array().expect("edges array");
            assert_eq!(
                edges.len(),
                expected_edges.len(),
                "seed {seed}: edge count mismatch"
            );
            for (i, edge) in edges.iter().enumerate() {
                let ee = &expected_edges[i];
                assert_eq!(
                    edge.from,
                    ee["from"].as_u64().unwrap() as usize,
                    "seed {seed} edge {i} from mismatch"
                );
                assert_eq!(
                    edge.to,
                    ee["to"].as_u64().unwrap() as usize,
                    "seed {seed} edge {i} to mismatch"
                );
                let expected_clearance: f32 = ee["clearance"].as_f64().unwrap() as f32;
                assert_eq!(
                    edge.clearance.to_bits(),
                    expected_clearance.to_bits(),
                    "seed {seed} edge {i} clearance mismatch"
                );
            }
        }
    }
}
