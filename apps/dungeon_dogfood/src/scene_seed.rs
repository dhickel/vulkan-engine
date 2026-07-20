use std::fs;
use std::path::{Path, PathBuf};

use glam::Mat4;
use renderer::prelude::{
    EnvironmentSource, MeshHandle, PbrMaterialDesc, ProceduralMeshData, ProceduralVertex,
    SceneFragmentNodeId, TextureHandle, TextureLoadOptions,
};
use renderer::{AssetManager, DirectionalLight, PointLight, PointLightId, Scene, SceneNodeId};
use thiserror::Error;

use crate::collision::WALL_HEIGHT;
use crate::content::{
    light_preset_for_marker_index, prop_for_marker_index, resolve_content_path, ContentPack,
    EnvironmentMode, EnvironmentSpec, MaterialFamily, MaterialSpec,
};
use crate::geometry::build_level_chunks;
use crate::layout::{tile_to_world, ParsedLevel, TileCoord};
use crate::mesh_collider_bridge::ColliderPolicy;

const WALL_MATERIAL_ID: &str = "stone_wall";
const FLOOR_MATERIAL_ID: &str = "stone_floor";
const FAST_STARTUP_ENV: &str = "DUNGEON_DOGFOOD_FAST_STARTUP";
const LOAD_PROPS_ENV: &str = "DUNGEON_DOGFOOD_LOAD_PROPS";
const LOAD_CUSTOM_ENV_ENV: &str = "DUNGEON_DOGFOOD_LOAD_CUSTOM_ENV";
const LOCKED_EXPOSURE: f32 = 2.8;
const LOCKED_GAMMA: f32 = 2.2;
const LOCKED_IBL_AMBIENT_SCALE: f32 = 0.45;
const LOCKED_POINT_LIGHT_INTENSITY: f32 = 30.0;
const LOCKED_POINT_LIGHT_RANGE: f32 = 6.0;

const MAX_POINT_LIGHTS: usize = 16;

#[derive(Debug, Error)]
pub enum SceneSeedError {
    #[error("asset error: {0}")]
    Asset(#[from] renderer::AssetError),
    #[error("scene error: {0}")]
    Scene(#[from] renderer::SceneError),
    #[error("point light budget exceeded: {count} markers, limit is {limit}")]
    PointLightBudgetExceeded { count: usize, limit: usize },
}

/// A mesh handle paired with its assigned collider policy.
#[derive(Debug, Clone)]
pub struct MeshColliderPolicyAssignment {
    pub mesh: MeshHandle,
    pub policy: ColliderPolicy,
}

/// Scene resources created from level data
pub struct LevelScene {
    pub wall_material: renderer::MaterialHandle,
    pub floor_material: renderer::MaterialHandle,
    pub chunk_meshes: Vec<MeshHandle>,
    pub chunk_nodes: Vec<SceneNodeId>,
    pub chunk_transforms: Vec<Mat4>,
    pub light_ids: Vec<PointLightId>,
    pub directional_light_id: Option<renderer::DirectionalLightId>,
    pub prop_roots: Vec<SceneNodeId>,
    /// Explicit collider policy assignments for meshes that need recipes.
    pub collider_policies: Vec<MeshColliderPolicyAssignment>,
    /// Handle for the small dynamic convex-hull proof mesh.
    pub dynamic_proof_mesh: Option<(MeshHandle, SceneNodeId)>,
}

#[derive(Debug, Copy, Clone)]
pub struct VisualLock {
    pub exposure: f32,
    pub gamma: f32,
    pub ibl_ambient_scale: f32,
    pub point_light_intensity: f32,
    pub point_light_range: f32,
}

pub const VISUAL_LOCK: VisualLock = VisualLock {
    exposure: LOCKED_EXPOSURE,
    gamma: LOCKED_GAMMA,
    ibl_ambient_scale: LOCKED_IBL_AMBIENT_SCALE,
    point_light_intensity: LOCKED_POINT_LIGHT_INTENSITY,
    point_light_range: LOCKED_POINT_LIGHT_RANGE,
};

pub fn renderer_visual_tuning() -> renderer::VisualTuning {
    renderer::VisualTuning {
        exposure: VISUAL_LOCK.exposure,
        gamma: VISUAL_LOCK.gamma,
        ibl_ambient_scale: VISUAL_LOCK.ibl_ambient_scale,
    }
}

impl LevelScene {
    /// Seed scene from parsed level data.
    pub fn from_level(
        level: &ParsedLevel,
        content_pack: &ContentPack,
        scene: &mut Scene,
        assets: &mut AssetManager,
    ) -> Result<Self, SceneSeedError> {
        // Preflight before allocating assets or mutating the scene so rejection
        // leaves the caller's scene and asset manager unchanged.
        if level.light_markers.len() > MAX_POINT_LIGHTS {
            return Err(SceneSeedError::PointLightBudgetExceeded {
                count: level.light_markers.len(),
                limit: MAX_POINT_LIGHTS,
            });
        }

        log_locked_visual_baseline(content_pack);

        // Build wall/floor materials from manifest texture sets with safe fallback.
        let wall_material = build_manifest_material(
            content_pack,
            assets,
            WALL_MATERIAL_ID,
            PbrMaterialDesc {
                base_color: glam::Vec4::new(0.42, 0.44, 0.46, 1.0),
                metallic: 0.0,
                roughness: 0.86,
                ..Default::default()
            },
        )?;

        let floor_material = build_manifest_material(
            content_pack,
            assets,
            FLOOR_MATERIAL_ID,
            PbrMaterialDesc {
                base_color: glam::Vec4::new(0.34, 0.35, 0.36, 1.0),
                metallic: 0.0,
                roughness: 0.78,
                ..Default::default()
            },
        )?;

        // Upload chunked dungeon mesh geometry and attach with authoritative bounds.
        let chunks = build_level_chunks(level, floor_material, wall_material);
        let mut chunk_meshes = Vec::with_capacity(chunks.len());
        let mut chunk_nodes = Vec::with_capacity(chunks.len());
        let mut chunk_transforms = Vec::with_capacity(chunks.len());
        let mut collider_policies = Vec::with_capacity(chunks.len() + 1);
        let level_root = scene.create_node(None, Mat4::IDENTITY)?;

        for chunk in chunks {
            let world_origin = chunk.world_origin;
            let mesh = assets.upload_procedural_mesh(chunk.mesh)?;
            let bounds = assets.mesh_scene_bounds(mesh).unwrap_or_else(|err| {
                log::warn!("Chunk mesh bounds lookup failed: {err}; using conservative-visible fallback.");
                renderer::SceneBounds::ConservativeVisible(renderer::BoundsUnknownReason::StaleHandle)
            });
            let model_to_instance = Mat4::from_translation(world_origin);
            let node = scene.create_node(Some(level_root), model_to_instance)?;
            scene.add_mesh_with_bounds(node, mesh, bounds)?;
            chunk_meshes.push(mesh);
            chunk_nodes.push(node);
            chunk_transforms.push(model_to_instance);
            // Static dungeon chunk geometry → StaticTrimesh.
            collider_policies.push(MeshColliderPolicyAssignment {
                mesh,
                policy: ColliderPolicy::StaticTrimesh,
            });
        }

        // Create a small dynamic convex-hull proof mesh at the spawn location.
        let dynamic_proof_mesh = {
            let spawn_world = tile_to_world(level.spawn.x, level.spawn.y);
            // A small tetrahedron positioned above the floor.
            let proof_verts: Vec<ProceduralVertex> = vec![
                // Base triangle
                make_proof_vertex([-0.3, 0.0, 0.3], [0.0, 1.0, 0.0]),
                make_proof_vertex([0.3, 0.0, 0.3], [0.0, 1.0, 0.0]),
                make_proof_vertex([0.0, 0.0, -0.3], [0.0, 1.0, 0.0]),
                // Apex
                make_proof_vertex([0.0, 0.6, 0.0], [0.0, 1.0, 0.0]),
            ];
            let proof_indices: Vec<u32> = vec![
                0, 2, 1, // base
                0, 1, 3, // face 1
                1, 2, 3, // face 2
                2, 0, 3, // face 3
            ];
            let proof_mesh_data = ProceduralMeshData {
                name: "dynamic_proof".to_string(),
                vertices: proof_verts,
                indices: proof_indices,
                material: Some(floor_material),
            };
            let proof_mesh = assets.upload_procedural_mesh(proof_mesh_data)?;
            let proof_node = scene.create_node(
                Some(level_root),
                Mat4::from_translation(glam::Vec3::new(
                    spawn_world.x + 1.5,
                    2.5, // elevated above floor
                    spawn_world.z,
                )),
            )?;
            let proof_bounds = assets.mesh_scene_bounds(proof_mesh).unwrap_or_else(|err| {
                log::warn!("Dynamic proof mesh bounds lookup failed: {err}; using conservative-visible fallback.");
                renderer::SceneBounds::ConservativeVisible(renderer::BoundsUnknownReason::StaleHandle)
            });
            scene.add_mesh_with_bounds(proof_node, proof_mesh, proof_bounds)?;
            collider_policies.push(MeshColliderPolicyAssignment {
                mesh: proof_mesh,
                policy: ColliderPolicy::ConvexHull,
            });
            Some((proof_mesh, proof_node))
        };

        // Spawn point lights from markers using deterministic 7/2/1 preset mapping.
        let mut light_ids = Vec::with_capacity(level.light_markers.len());
        for (marker_idx, &TileCoord { layer, x, y }) in level.light_markers.iter().enumerate() {
            let preset_id = light_preset_for_marker_index(marker_idx);
            let Some(light_preset) = content_pack.light_preset(preset_id) else {
                log::warn!(
                    "Skipping light marker {} at ({}, {}, {}): missing preset {:?}",
                    marker_idx,
                    layer,
                    x,
                    y,
                    preset_id
                );
                continue;
            };

            let y_offset = layer as f32 * WALL_HEIGHT;
            let world_pos = tile_to_world(x, y) + glam::Vec3::new(0.5, y_offset + 1.7, -0.5);
            let light_id = scene.create_point_light(PointLight {
                position: world_pos,
                color: glam::Vec3::new(
                    light_preset.color[0],
                    light_preset.color[1],
                    light_preset.color[2],
                ),
                intensity: light_preset.intensity * 0.95,
                range: light_preset.range * 1.55,
            })?;
            light_ids.push(light_id);
        }

        // Spawn deterministic props from model markers.
        let enabled_props = content_pack.enabled_props();
        let mut prop_roots = Vec::new();
        if env_flag_or(LOAD_PROPS_ENV, false) {
            let mut warned_torch_unlit_fallback = false;
            for (marker_idx, &TileCoord { layer, x, y }) in level.model_markers.iter().enumerate() {
                let prop_idx = prop_for_marker_index(marker_idx, enabled_props.len());
                let prop = enabled_props[prop_idx];
                let placement = prop.placement_policy();

                if placement.prefer_unlit_fallback && !warned_torch_unlit_fallback {
                    warned_torch_unlit_fallback = true;
                    log::warn!(
                        "Content pack prop '{}' prefers unlit fallback; keeping this as temporary non-PBR behavior",
                        prop.id
                    );
                }

                let prop_path = resolve_content_path(prop.path.as_path());
                let fragment = match assets.load_model(&prop_path) {
                    Ok(fragment) => fragment,
                    Err(err) => {
                        log::warn!(
                            "Skipping prop '{}' at marker {} ({}, {}, {}): model load failed: {}",
                            prop.id,
                            marker_idx,
                            layer,
                            x,
                            y,
                            err
                        );
                        continue;
                    }
                };

                let visual_only_meshes: Vec<MeshHandle> = (0..fragment.node_count())
                    .filter_map(|index| fragment.node(SceneFragmentNodeId::new(index as u32)))
                    .flat_map(|node| node.meshes.iter().copied())
                    .collect();
                let mount = match scene.merge_fragment(Some(level_root), fragment) {
                    Ok(mount) => mount,
                    Err(err) => {
                        log::warn!(
                            "Skipping prop '{}' at marker {} ({}, {}, {}): merge failed: {}",
                            prop.id,
                            marker_idx,
                            layer,
                            x,
                            y,
                            err
                        );
                        continue;
                    }
                };

                let y_offset = layer as f32 * WALL_HEIGHT;
                let world_pos =
                    tile_to_world(x, y) + glam::Vec3::new(0.5, y_offset + placement.y_offset, -0.5);
                let transform = Mat4::from_scale_rotation_translation(
                    placement.scale,
                    glam::Quat::from_rotation_y(placement.yaw_degrees.to_radians()),
                    world_pos,
                );

                if let Err(err) = scene.set_transform(mount.mounted_root, transform) {
                    log::warn!(
                        "Skipping prop '{}' at marker {} ({}, {}, {}): transform failed: {}",
                        prop.id,
                        marker_idx,
                        layer,
                        x,
                        y,
                        err
                    );
                    continue;
                }

                collider_policies.extend(visual_only_meshes.into_iter().map(|mesh| {
                    MeshColliderPolicyAssignment {
                        mesh,
                        policy: ColliderPolicy::None,
                    }
                }));
                prop_roots.push(mount.mounted_root);
            }
        } else {
            log::info!(
                "Skipping prop import for faster startup. Set {}=1 to enable manifest props.",
                LOAD_PROPS_ENV
            );
        }

        // Load primary environment from content pack (warn-only on failure).
        if env_flag_or(LOAD_CUSTOM_ENV_ENV, false) {
            let primary_env = content_pack.primary_environment();
            if let Err(err) = load_environment(scene, assets, primary_env) {
                log::warn!(
                    "Failed to load environment '{}': {}. Continuing with renderer default environment.",
                    primary_env.id,
                    err
                );
            }
        } else {
            log::info!(
                "Skipping custom environment import for faster startup. Set {}=1 to enable content-pack environment.",
                LOAD_CUSTOM_ENV_ENV
            );
        }

        // Create a directional light (sunlight-like) for shadow casting.
        let directional_light_id = scene
            .create_directional_light(DirectionalLight {
                direction: glam::Vec3::new(0.3, 0.8, 0.4),
                color: glam::Vec3::new(1.0, 0.95, 0.85),
                intensity: 2.5,
            })
            .ok();

        Ok(Self {
            wall_material,
            floor_material,
            chunk_meshes,
            chunk_nodes,
            chunk_transforms,
            light_ids,
            directional_light_id,
            prop_roots,
            collider_policies,
            dynamic_proof_mesh,
        })
    }
}

fn load_environment(
    scene: &mut Scene,
    assets: &mut AssetManager,
    environment: &EnvironmentSpec,
) -> Result<(), renderer::AssetError> {
    let resolved_path = resolve_content_path(environment.path.as_path());
    let source = match environment.mode {
        EnvironmentMode::Auto => EnvironmentSource::Auto(resolved_path),
    };
    let handle = assets.load_environment(source)?;
    scene.set_skybox(handle);
    Ok(())
}

fn build_manifest_material(
    content_pack: &ContentPack,
    assets: &mut AssetManager,
    material_id: &str,
    fallback_desc: PbrMaterialDesc,
) -> Result<renderer::MaterialHandle, SceneSeedError> {
    let Some(material_spec) = content_pack.material_by_id(material_id) else {
        log::warn!(
            "Missing material '{}' in content pack; using factor-only fallback material.",
            material_id
        );
        return Ok(create_factor_material_or_reserved_default(
            assets,
            fallback_desc,
            material_id,
        ));
    };

    match material_spec.family {
        MaterialFamily::Pbr => {}
    }

    let mut desc = fallback_desc.clone();
    let material_base_path = resolve_content_path(material_spec.base_path.as_path());

    let fast_startup = env_flag_or(FAST_STARTUP_ENV, false);
    if fast_startup {
        log::info!(
            "Fast startup enabled ({}=1): loading base-color maps only for material '{}'.",
            FAST_STARTUP_ENV,
            material_spec.id
        );
    }

    let map_paths = MaterialMapPaths {
        base_color: find_texture_map(
            &material_base_path,
            &[
                "_diff_",
                "_basecolor_",
                "_basecolor",
                "_base_color_",
                "_albedo_",
                "_color_",
            ],
        ),
        normal: if fast_startup {
            None
        } else {
            find_texture_map(
                &material_base_path,
                &["_nor_gl_", "_normal_", "_normal", "_nor_"],
            )
        },
        arm: if fast_startup {
            None
        } else {
            find_texture_map(
                &material_base_path,
                &[
                    "_arm_",
                    "_arm",
                    "_occlusionroughnessmetallic_",
                    "_orm_",
                ],
            )
        },
    };

    let loaded_maps =
        load_optional_texture_set(assets, material_spec, &map_paths, &material_base_path);
    desc.base_color_tex = loaded_maps.base_color;
    desc.normal_tex = loaded_maps.normal;
    desc.metallic_roughness_tex = loaded_maps.arm;
    desc.ao_tex = loaded_maps.arm;

    match assets.create_material_pbr(desc) {
        Ok(material) => Ok(material),
        Err(err) => {
            log::warn!(
                "Material '{}' failed during textured PBR allocation: {}. Falling back to factor-only material.",
                material_spec.id,
                err
            );
            Ok(create_factor_material_or_reserved_default(
                assets,
                fallback_desc,
                material_spec.id.as_str(),
            ))
        }
    }
}

fn create_factor_material_or_reserved_default(
    assets: &mut AssetManager,
    fallback_desc: PbrMaterialDesc,
    material_id: &str,
) -> renderer::MaterialHandle {
    match assets.create_material_pbr(fallback_desc) {
        Ok(material) => material,
        Err(err) => {
            log::warn!(
                "Material '{}' also failed factor-only allocation: {}. Using reserved default material handle (slot 0).",
                material_id,
                err
            );
            renderer::MaterialHandle::new(0, 0)
        }
    }
}

#[derive(Default)]
struct MaterialMapPaths {
    base_color: Option<PathBuf>,
    normal: Option<PathBuf>,
    arm: Option<PathBuf>,
}

#[derive(Default)]
struct LoadedMaterialMaps {
    base_color: Option<TextureHandle>,
    normal: Option<TextureHandle>,
    arm: Option<TextureHandle>,
}

#[derive(Copy, Clone)]
enum MaterialMapKind {
    BaseColor,
    Normal,
    Arm,
}

fn load_optional_texture_set(
    assets: &mut AssetManager,
    material_spec: &MaterialSpec,
    map_paths: &MaterialMapPaths,
    material_base_path: &Path,
) -> LoadedMaterialMaps {
    let mut requests = Vec::<(PathBuf, TextureLoadOptions)>::new();
    let mut kinds = Vec::<MaterialMapKind>::new();
    let mut loaded = LoadedMaterialMaps::default();

    let mut push_request = |kind: MaterialMapKind,
                            map_name: &str,
                            path: Option<&PathBuf>,
                            options: TextureLoadOptions| {
        let Some(path) = path else {
            log::warn!(
                "Material '{}' missing optional {} map under '{}'; using factor/default fallback.",
                material_spec.id,
                map_name,
                material_base_path.display()
            );
            return;
        };
        requests.push((path.clone(), options));
        kinds.push(kind);
    };

    push_request(
        MaterialMapKind::BaseColor,
        "base_color",
        map_paths.base_color.as_ref(),
        TextureLoadOptions {
            force_srgb: Some(true),
            generate_mips: Some(false),
            ..Default::default()
        },
    );
    push_request(
        MaterialMapKind::Normal,
        "normal",
        map_paths.normal.as_ref(),
        TextureLoadOptions {
            force_srgb: Some(false),
            generate_mips: Some(false),
            ..Default::default()
        },
    );
    push_request(
        MaterialMapKind::Arm,
        "arm",
        map_paths.arm.as_ref(),
        TextureLoadOptions {
            force_srgb: Some(false),
            generate_mips: Some(false),
            ..Default::default()
        },
    );

    if requests.is_empty() {
        return loaded;
    }

    match assets.load_textures_with_options(requests.clone()) {
        Ok(handles) => {
            for (kind, handle) in kinds.into_iter().zip(handles.into_iter()) {
                match kind {
                    MaterialMapKind::BaseColor => loaded.base_color = Some(handle),
                    MaterialMapKind::Normal => loaded.normal = Some(handle),
                    MaterialMapKind::Arm => loaded.arm = Some(handle),
                }
            }
            loaded
        }
        Err(err) => {
            log::warn!(
                "Material '{}' batched texture load failed: {}. Falling back to individual texture loads.",
                material_spec.id,
                err
            );

            loaded.base_color = load_optional_texture(
                assets,
                material_spec,
                "base_color",
                map_paths.base_color.as_deref(),
                TextureLoadOptions {
                    force_srgb: Some(true),
                    generate_mips: Some(false),
                    ..Default::default()
                },
            );
            loaded.normal = load_optional_texture(
                assets,
                material_spec,
                "normal",
                map_paths.normal.as_deref(),
                TextureLoadOptions {
                    force_srgb: Some(false),
                    generate_mips: Some(false),
                    ..Default::default()
                },
            );
            loaded.arm = load_optional_texture(
                assets,
                material_spec,
                "arm",
                map_paths.arm.as_deref(),
                TextureLoadOptions {
                    force_srgb: Some(false),
                    generate_mips: Some(false),
                    ..Default::default()
                },
            );
            loaded
        }
    }
}

fn load_optional_texture(
    assets: &mut AssetManager,
    material_spec: &MaterialSpec,
    map_name: &str,
    path: Option<&Path>,
    options: TextureLoadOptions,
) -> Option<TextureHandle> {
    let Some(path) = path else {
        log::warn!(
            "Material '{}' missing optional {} map under '{}'; using factor/default fallback.",
            material_spec.id,
            map_name,
            material_spec.base_path.display()
        );
        return None;
    };

    match assets.load_texture_with_options(path, options) {
        Ok(texture) => Some(texture),
        Err(err) => {
            log::warn!(
                "Material '{}' failed loading {} map '{}': {}. Using factor/default fallback.",
                material_spec.id,
                map_name,
                path.display(),
                err
            );
            None
        }
    }
}

fn make_proof_vertex(pos: [f32; 3], normal: [f32; 3]) -> ProceduralVertex {
    ProceduralVertex {
        position: glam::Vec3::from_array(pos),
        normal: glam::Vec3::from_array(normal),
        tangent: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
        uv0: glam::Vec2::ZERO,
        uv1: glam::Vec2::ZERO,
        color: glam::Vec4::ONE,
    }
}

fn env_flag_or(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn log_locked_visual_baseline(content_pack: &ContentPack) {
    log::info!(
        "Locked visual baseline: exposure {:.1}, gamma {:.1}, ibl ambient {:.1}, point light intensity {:.1}, range {:.1}",
        VISUAL_LOCK.exposure,
        VISUAL_LOCK.gamma,
        VISUAL_LOCK.ibl_ambient_scale,
        VISUAL_LOCK.point_light_intensity,
        VISUAL_LOCK.point_light_range
    );

    if let Some(warm) = content_pack.light_preset(crate::content::LightPresetId::Warm) {
        if !approx_eq(warm.intensity, VISUAL_LOCK.point_light_intensity)
            || !approx_eq(warm.range, VISUAL_LOCK.point_light_range)
        {
            log::warn!(
                "Warm light preset drifted from locked baseline (expected intensity {:.1} / range {:.1}, got {:.1} / {:.1})",
                VISUAL_LOCK.point_light_intensity,
                VISUAL_LOCK.point_light_range,
                warm.intensity,
                warm.range
            );
        }
    }
}

fn approx_eq(left: f32, right: f32) -> bool {
    (left - right).abs() <= 0.01
}

fn find_texture_map(base_path: &Path, needles: &[&str]) -> Option<PathBuf> {
    for search_dir in [base_path.join("textures"), base_path.to_path_buf()] {
        let Ok(read_dir) = fs::read_dir(&search_dir) else {
            continue;
        };

        let mut entries = read_dir
            .filter_map(|entry| entry.ok().map(|value| value.path()))
            .filter(|path| is_texture_ext(path))
            .collect::<Vec<_>>();
        entries.sort();

        for needle in needles {
            if let Some(path) = entries.iter().find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.to_ascii_lowercase().contains(needle))
                    .unwrap_or(false)
            }) {
                return Some(path.clone());
            }
        }
    }

    None
}

fn is_texture_ext(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };

    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "tga" | "dds" | "ktx2" | "hdr" | "exr"
    )
}
