use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use image::ImageError;

use crate::api::scene::{SceneFragment, SceneFragmentNodeId};
use crate::data::assimp_util::{self, ModelMeta};
use crate::data::data_cache::{LoadResult, MeshCache, TextureCache, VkDataCache};
use crate::data::gpu_data::TextureMeta;
use crate::data::handles::{CacheError, MaterialHandle, MeshHandle, TextureHandle};
use crate::scene::scene_world::{SceneNodeId, SceneWorld};
use crate::vulkan::vk_render::VkRenderCore;
use crate::vulkan::vk_storage::BufferPlacement;

use super::errors::AssetError;

pub struct AssetManager<'a> {
    core: &'a mut VkRenderCore,
}

impl<'a> AssetManager<'a> {
    pub(crate) fn new(core: &'a mut VkRenderCore) -> Self {
        Self { core }
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn load_mesh(&mut self, path: impl AsRef<Path>) -> Result<MeshHandle, AssetError> {
        let path = path.as_ref().to_path_buf();
        let path_for_err = path.clone();
        let loaded_model =
            self.run_sync_upload_task(move |data_cache| load_model_gpu_ready(path, data_cache))?;

        loaded_model
            .mesh_ids
            .first()
            .copied()
            .ok_or_else(|| AssetError::Load {
                path: Some(path_for_err),
                message: "model import did not contain any meshes".to_string(),
            })
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn load_texture(&mut self, path: impl AsRef<Path>) -> Result<TextureHandle, AssetError> {
        let path = path.as_ref().to_path_buf();
        self.run_sync_upload_task(move |data_cache| load_texture_gpu_ready(path, data_cache))
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn load_model(&mut self, path: impl AsRef<Path>) -> Result<SceneFragment, AssetError> {
        let path = path.as_ref().to_path_buf();
        let loaded_model =
            self.run_sync_upload_task(move |data_cache| load_model_gpu_ready(path, data_cache))?;

        scene_world_to_fragment(loaded_model.scene_world)
    }

    /// Thread: Main
    /// May Stall: No
    pub fn unload_mesh(&mut self, mesh: MeshHandle) -> Result<(), AssetError> {
        let mut mesh_cache = self
            .core
            .data_cache
            .mesh_cache
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_cache"))?;

        mesh_cache
            .get_id(mesh)
            .map_err(|err| map_cache_err("mesh", mesh.slot, mesh.generation, err))?;

        if mesh.slot == MeshCache::SKYBOX_MESH.slot {
            return Err(AssetError::ReservedHandle {
                resource: "mesh",
                slot: mesh.slot,
                generation: mesh.generation,
            });
        }

        mesh_cache.deallocate_id(mesh);
        Ok(())
    }

    /// Thread: Main
    /// May Stall: No
    pub fn unload_material(&mut self, material: MaterialHandle) -> Result<(), AssetError> {
        let mut texture_cache = self
            .core
            .data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;

        texture_cache
            .get_material(material)
            .map_err(|err| map_cache_err("material", material.slot, material.generation, err))?;

        if (material.slot as usize) < TextureCache::DEFAULT_MAT_ITER_START {
            return Err(AssetError::ReservedHandle {
                resource: "material",
                slot: material.slot,
                generation: material.generation,
            });
        }

        texture_cache.deallocate_materials(vec![material]);
        Ok(())
    }

    /// Thread: Main
    /// May Stall: No
    pub fn unload_texture(&mut self, texture: TextureHandle) -> Result<(), AssetError> {
        let mut texture_cache = self
            .core
            .data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;

        texture_cache
            .get_texture(texture)
            .map_err(|err| map_cache_err("texture", texture.slot, texture.generation, err))?;

        if (texture.slot as usize) < TextureCache::DEFAULT_TEX_ITER_START {
            return Err(AssetError::ReservedHandle {
                resource: "texture",
                slot: texture.slot,
                generation: texture.generation,
            });
        }

        texture_cache.deallocate_texture(texture);
        Ok(())
    }

    fn run_sync_upload_task<T, F>(&mut self, task: F) -> Result<T, AssetError>
    where
        T: Send + 'static,
        F: FnOnce(Arc<VkDataCache>) -> Result<T, AssetError> + Send + 'static,
    {
        let data_cache = self.core.data_cache.clone();
        let worker = std::thread::spawn(move || task(data_cache));

        while !worker.is_finished() {
            self.pump_transfer_submissions();
            std::thread::sleep(Duration::from_millis(1));
        }

        self.pump_transfer_submissions();
        worker
            .join()
            .map_err(|_| AssetError::Sync("asset upload worker thread panicked".to_string()))?
    }

    fn pump_transfer_submissions(&mut self) {
        let core = &mut self.core;
        core.fence_await_queue.check_fences(&core.device);
        while let Some(cmd) = core.transfer.query_channel() {
            cmd.submit(
                &core.device,
                &core.vulkan_cache.queues,
                &mut core.fence_await_queue,
            );
        }
    }
}

fn load_model_gpu_ready(
    path: PathBuf,
    data_cache: Arc<VkDataCache>,
) -> Result<ModelMeta, AssetError> {
    let path_str = path.to_str().ok_or_else(|| {
        AssetError::Unsupported(format!(
            "asset path '{}' is not valid UTF-8",
            path.display()
        ))
    })?;

    let loaded_model =
        assimp_util::load_model(path_str, data_cache.clone(), false).map_err(|err| {
            AssetError::Load {
                path: Some(path.clone()),
                message: err,
            }
        })?;

    let upload_result = promote_model_gpu_allocations(&path, &data_cache, &loaded_model);
    if upload_result.is_err() {
        let _ = rollback_model_allocations(
            &data_cache,
            &loaded_model.mesh_ids,
            &loaded_model.material_ids,
        );
    }
    upload_result?;

    Ok(loaded_model)
}

fn promote_model_gpu_allocations(
    path: &Path,
    data_cache: &Arc<VkDataCache>,
    loaded_model: &ModelMeta,
) -> Result<(), AssetError> {
    {
        let mut mesh_cache = data_cache
            .mesh_cache
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_cache"))?;

        match mesh_cache.allocate_ids(
            loaded_model.mesh_ids.as_slice(),
            BufferPlacement::ContiguousPreferred,
            false,
        ) {
            LoadResult::Success(_) => {}
            LoadResult::Failed(_) => {
                return Err(AssetError::Load {
                    path: Some(path.to_path_buf()),
                    message: "mesh GPU allocation failed".to_string(),
                });
            }
        }
    }

    {
        let mut texture_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;

        match texture_cache.allocate_ids(
            loaded_model.material_ids.as_slice(),
            BufferPlacement::ContiguousPreferred,
            false,
        ) {
            LoadResult::Success(_) => {}
            LoadResult::Failed(_) => {
                return Err(AssetError::Load {
                    path: Some(path.to_path_buf()),
                    message: "material/texture GPU allocation failed".to_string(),
                });
            }
        }
    }

    Ok(())
}

fn rollback_model_allocations(
    data_cache: &Arc<VkDataCache>,
    mesh_ids: &[MeshHandle],
    material_ids: &[MaterialHandle],
) -> Result<(), AssetError> {
    with_mesh_texture_cache_locks(data_cache, |mesh_cache, texture_cache| {
        texture_cache.deallocate_materials(material_ids.to_vec());
        mesh_cache.deallocate_ids(mesh_ids);
    })
}

fn load_texture_gpu_ready(
    path: PathBuf,
    data_cache: Arc<VkDataCache>,
) -> Result<TextureHandle, AssetError> {
    let image = image::open(&path).map_err(|err| map_texture_path_err(&path, err))?;
    let format = assimp_util::to_vk_format(&image);
    let texture_meta = TextureMeta {
        bytes: image.as_bytes().to_vec(),
        width: image.width(),
        height: image.height(),
        format,
        mips_levels: 1,
        uv_index: 0,
    };

    let texture_id = {
        let mut texture_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;
        texture_cache.add_texture(texture_meta)
    };

    if texture_id == TextureCache::DEFAULT_ERROR_TEX {
        return Err(AssetError::Load {
            path: Some(path),
            message: "texture conversion failed during cache import".to_string(),
        });
    }

    let did_allocate = {
        let mut texture_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;
        texture_cache.allocate_textures(vec![texture_id])
    };

    if !did_allocate {
        let _ = rollback_texture_allocation(&data_cache, texture_id);
        return Err(AssetError::Load {
            path: Some(path),
            message: "texture GPU allocation failed".to_string(),
        });
    }

    Ok(texture_id)
}

fn rollback_texture_allocation(
    data_cache: &Arc<VkDataCache>,
    texture: TextureHandle,
) -> Result<(), AssetError> {
    let mut texture_cache = data_cache
        .texture_cache
        .lock()
        .map_err(|_| poisoned_lock_err("texture_cache"))?;
    texture_cache.deallocate_texture(texture);
    Ok(())
}

fn scene_world_to_fragment(scene_world: SceneWorld) -> Result<SceneFragment, AssetError> {
    let root = scene_world.root_id().ok_or_else(|| AssetError::Load {
        path: None,
        message: "imported model does not contain a scene root".to_string(),
    })?;

    let mut fragment = SceneFragment::new();
    let mut stack: Vec<(SceneNodeId, Option<SceneFragmentNodeId>)> = vec![(root, None)];

    while let Some((source_node_id, fragment_parent_id)) = stack.pop() {
        let source_node = scene_world.get_node(source_node_id).ok_or_else(|| {
            AssetError::Internal("model scene graph node became invalid".to_string())
        })?;

        let fragment_node_id = fragment
            .add_node(
                fragment_parent_id,
                source_node.local_transform,
                source_node.meshes.clone(),
            )
            .map_err(|err| {
                AssetError::Internal(format!("failed to build scene fragment: {err}"))
            })?;

        for child in source_node.children.iter().rev().copied() {
            stack.push((child, Some(fragment_node_id)));
        }
    }

    Ok(fragment)
}

fn with_mesh_texture_cache_locks<T>(
    data_cache: &Arc<VkDataCache>,
    f: impl FnOnce(&mut MeshCache, &mut TextureCache) -> T,
) -> Result<T, AssetError> {
    let mut mesh_cache = data_cache
        .mesh_cache
        .lock()
        .map_err(|_| poisoned_lock_err("mesh_cache"))?;
    let mut texture_cache = data_cache
        .texture_cache
        .lock()
        .map_err(|_| poisoned_lock_err("texture_cache"))?;
    Ok(f(&mut mesh_cache, &mut texture_cache))
}

fn map_texture_path_err(path: &Path, err: ImageError) -> AssetError {
    match err {
        ImageError::IoError(io_err) => AssetError::Io {
            path: path.to_path_buf(),
            message: io_err.to_string(),
        },
        other => AssetError::Decode {
            path: path.to_path_buf(),
            message: other.to_string(),
        },
    }
}

fn map_cache_err(
    resource: &'static str,
    slot: u32,
    generation: u32,
    err: CacheError,
) -> AssetError {
    match err {
        CacheError::InvalidHandle => AssetError::InvalidHandle {
            resource,
            slot,
            generation,
        },
        CacheError::StaleHandle => AssetError::StaleHandle {
            resource,
            slot,
            generation,
        },
        CacheError::NotLoaded => AssetError::NotLoaded {
            resource,
            slot,
            generation,
        },
        CacheError::OutOfBounds => AssetError::OutOfBounds {
            resource,
            slot,
            generation,
        },
    }
}

fn poisoned_lock_err(lock_name: &str) -> AssetError {
    AssetError::Sync(format!("{lock_name} lock poisoned"))
}
