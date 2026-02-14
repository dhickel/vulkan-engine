use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{collections::HashMap, collections::VecDeque};

use image::ImageError;

use crate::api::loading::{LoadStatus, LoadTicket};
use crate::api::scene::{SceneFragment, SceneFragmentNodeId};
use crate::data::assimp_util::{self, ModelMeta};
use crate::data::data_cache::{
    CachedEnvironment, LoadResult, MeshCache, TextureCache, VkDataCache,
};
use crate::data::gpu_data::{MaterialMeta, MeshMeta, TextureMeta, Vertex};
use crate::data::handles::{
    CacheError, EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle,
};
use crate::scene::scene_world::{SceneNodeId, SceneWorld};
use crate::vulkan::vk_render::VkRenderCore;
use crate::vulkan::vk_storage::BufferPlacement;

use super::errors::AssetError;

const TERMINAL_TICKET_RETAIN_LIMIT: usize = 2_048;

pub use crate::data::environment_import::{EnvironmentSource, FacePattern};

#[derive(Debug)]
pub enum EnvironmentState {
    Unloaded,
    Loading,
    Ready,
    Failed(AssetError),
}

enum DeferredModelState {
    Queued {
        path: PathBuf,
    },
    InFlight {
        receiver: Receiver<Result<SceneFragment, AssetError>>,
    },
    Uploaded {
        value: SceneFragment,
    },
    Failed {
        error: AssetError,
    },
    Cancelled,
}

enum DeferredTextureState {
    Queued {
        path: PathBuf,
    },
    InFlight {
        receiver: Receiver<Result<TextureHandle, AssetError>>,
    },
    Uploaded {
        value: TextureHandle,
    },
    Failed {
        error: AssetError,
    },
    Cancelled,
}

enum DeferredLoadTask {
    Model {
        queued_at: Instant,
        state: DeferredModelState,
    },
    Texture {
        queued_at: Instant,
        state: DeferredTextureState,
    },
}

impl DeferredLoadTask {
    fn is_terminal(&self) -> bool {
        match self {
            Self::Model { state, .. } => matches!(
                state,
                DeferredModelState::Uploaded { .. }
                    | DeferredModelState::Failed { .. }
                    | DeferredModelState::Cancelled
            ),
            Self::Texture { state, .. } => matches!(
                state,
                DeferredTextureState::Uploaded { .. }
                    | DeferredTextureState::Failed { .. }
                    | DeferredTextureState::Cancelled
            ),
        }
    }
}

pub(crate) struct AssetLoadTracker {
    next_ticket: u64,
    queued_tickets: VecDeque<LoadTicket>,
    terminal_tickets: VecDeque<LoadTicket>,
    tasks: HashMap<LoadTicket, DeferredLoadTask>,
}

impl AssetLoadTracker {
    pub(crate) fn new() -> Self {
        Self {
            next_ticket: 1,
            queued_tickets: VecDeque::new(),
            terminal_tickets: VecDeque::new(),
            tasks: HashMap::new(),
        }
    }

    pub(crate) fn request_model_load(&mut self, path: PathBuf) -> LoadTicket {
        let ticket = self.next_ticket();
        self.tasks.insert(
            ticket,
            DeferredLoadTask::Model {
                queued_at: Instant::now(),
                state: DeferredModelState::Queued { path },
            },
        );
        self.queued_tickets.push_back(ticket);
        ticket
    }

    pub(crate) fn request_texture_load(&mut self, path: PathBuf) -> LoadTicket {
        let ticket = self.next_ticket();
        self.tasks.insert(
            ticket,
            DeferredLoadTask::Texture {
                queued_at: Instant::now(),
                state: DeferredTextureState::Queued { path },
            },
        );
        self.queued_tickets.push_back(ticket);
        ticket
    }

    pub(crate) fn poll_model_load(&self, ticket: LoadTicket) -> LoadStatus<SceneFragment> {
        let Some(task) = self.tasks.get(&ticket) else {
            return LoadStatus::Failed {
                error: AssetError::UnknownTicket {
                    ticket: ticket.raw(),
                },
            };
        };

        match task {
            DeferredLoadTask::Model { queued_at, state } => match state {
                DeferredModelState::Queued { .. } | DeferredModelState::InFlight { .. } => {
                    LoadStatus::Pending {
                        queued_at: *queued_at,
                    }
                }
                DeferredModelState::Uploaded { value } => LoadStatus::Uploaded {
                    value: value.clone(),
                },
                DeferredModelState::Failed { error } => LoadStatus::Failed {
                    error: error.clone(),
                },
                DeferredModelState::Cancelled => LoadStatus::Cancelled,
            },
            DeferredLoadTask::Texture { .. } => LoadStatus::Failed {
                error: AssetError::UnknownTicket {
                    ticket: ticket.raw(),
                },
            },
        }
    }

    pub(crate) fn poll_texture_load(&self, ticket: LoadTicket) -> LoadStatus<TextureHandle> {
        let Some(task) = self.tasks.get(&ticket) else {
            return LoadStatus::Failed {
                error: AssetError::UnknownTicket {
                    ticket: ticket.raw(),
                },
            };
        };

        match task {
            DeferredLoadTask::Texture { queued_at, state } => match state {
                DeferredTextureState::Queued { .. } | DeferredTextureState::InFlight { .. } => {
                    LoadStatus::Pending {
                        queued_at: *queued_at,
                    }
                }
                DeferredTextureState::Uploaded { value } => LoadStatus::Uploaded { value: *value },
                DeferredTextureState::Failed { error } => LoadStatus::Failed {
                    error: error.clone(),
                },
                DeferredTextureState::Cancelled => LoadStatus::Cancelled,
            },
            DeferredLoadTask::Model { .. } => LoadStatus::Failed {
                error: AssetError::UnknownTicket {
                    ticket: ticket.raw(),
                },
            },
        }
    }

    pub(crate) fn cancel_load(&mut self, ticket: LoadTicket) -> Result<(), AssetError> {
        let Some(task) = self.tasks.get_mut(&ticket) else {
            return Err(AssetError::UnknownTicket {
                ticket: ticket.raw(),
            });
        };

        let mut mark_terminal = false;
        let result = match task {
            DeferredLoadTask::Model { state, .. } => match state {
                DeferredModelState::Queued { .. } => {
                    *state = DeferredModelState::Cancelled;
                    mark_terminal = true;
                    Ok(())
                }
                DeferredModelState::InFlight { .. } => Err(AssetError::CancelRejected {
                    ticket: ticket.raw(),
                    reason: "load is already running".to_string(),
                }),
                DeferredModelState::Uploaded { .. } => Err(AssetError::CancelRejected {
                    ticket: ticket.raw(),
                    reason: "load already completed".to_string(),
                }),
                DeferredModelState::Failed { .. } | DeferredModelState::Cancelled => Ok(()),
            },
            DeferredLoadTask::Texture { state, .. } => match state {
                DeferredTextureState::Queued { .. } => {
                    *state = DeferredTextureState::Cancelled;
                    mark_terminal = true;
                    Ok(())
                }
                DeferredTextureState::InFlight { .. } => Err(AssetError::CancelRejected {
                    ticket: ticket.raw(),
                    reason: "load is already running".to_string(),
                }),
                DeferredTextureState::Uploaded { .. } => Err(AssetError::CancelRejected {
                    ticket: ticket.raw(),
                    reason: "load already completed".to_string(),
                }),
                DeferredTextureState::Failed { .. } | DeferredTextureState::Cancelled => Ok(()),
            },
        };

        if mark_terminal {
            self.record_terminal(ticket);
        }
        result
    }

    pub(crate) fn pending_load_count(&self) -> usize {
        self.tasks
            .values()
            .filter(|task| match task {
                DeferredLoadTask::Model { state, .. } => matches!(
                    state,
                    DeferredModelState::Queued { .. } | DeferredModelState::InFlight { .. }
                ),
                DeferredLoadTask::Texture { state, .. } => matches!(
                    state,
                    DeferredTextureState::Queued { .. } | DeferredTextureState::InFlight { .. }
                ),
            })
            .count()
    }

    pub(crate) fn has_pending_loads(&self) -> bool {
        self.pending_load_count() > 0
    }

    pub(crate) fn pump(&mut self, core: &mut VkRenderCore, max_steps: usize) -> usize {
        if max_steps == 0 {
            return 0;
        }

        let mut progressed = 0usize;
        progressed += core.pump_transfer_submissions(max_steps - progressed);

        if progressed < max_steps {
            progressed += self.collect_finished(max_steps - progressed);
        }

        if progressed < max_steps {
            progressed += self.start_queued(core.data_cache.clone(), max_steps - progressed);
        }

        self.cleanup_terminal_tickets();
        progressed
    }

    fn next_ticket(&mut self) -> LoadTicket {
        loop {
            let ticket = LoadTicket::new(self.next_ticket);
            self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
            if !self.tasks.contains_key(&ticket) {
                return ticket;
            }
        }
    }

    fn collect_finished(&mut self, max_steps: usize) -> usize {
        if max_steps == 0 {
            return 0;
        }

        let mut finished = 0usize;
        let tickets: Vec<LoadTicket> = self.tasks.keys().copied().collect();

        for ticket in tickets {
            if finished >= max_steps {
                break;
            }

            let mut transitioned_terminal = false;

            if let Some(task) = self.tasks.get_mut(&ticket) {
                match task {
                    DeferredLoadTask::Model { state, .. } => {
                        if let DeferredModelState::InFlight { receiver } = state {
                            match receiver.try_recv() {
                                Ok(Ok(value)) => {
                                    *state = DeferredModelState::Uploaded { value };
                                    transitioned_terminal = true;
                                    finished += 1;
                                }
                                Ok(Err(error)) => {
                                    *state = DeferredModelState::Failed { error };
                                    transitioned_terminal = true;
                                    finished += 1;
                                }
                                Err(TryRecvError::Disconnected) => {
                                    *state = DeferredModelState::Failed {
                                        error: AssetError::Sync(
                                            "deferred model worker disconnected".to_string(),
                                        ),
                                    };
                                    transitioned_terminal = true;
                                    finished += 1;
                                }
                                Err(TryRecvError::Empty) => {}
                            }
                        }
                    }
                    DeferredLoadTask::Texture { state, .. } => {
                        if let DeferredTextureState::InFlight { receiver } = state {
                            match receiver.try_recv() {
                                Ok(Ok(value)) => {
                                    *state = DeferredTextureState::Uploaded { value };
                                    transitioned_terminal = true;
                                    finished += 1;
                                }
                                Ok(Err(error)) => {
                                    *state = DeferredTextureState::Failed { error };
                                    transitioned_terminal = true;
                                    finished += 1;
                                }
                                Err(TryRecvError::Disconnected) => {
                                    *state = DeferredTextureState::Failed {
                                        error: AssetError::Sync(
                                            "deferred texture worker disconnected".to_string(),
                                        ),
                                    };
                                    transitioned_terminal = true;
                                    finished += 1;
                                }
                                Err(TryRecvError::Empty) => {}
                            }
                        }
                    }
                }
            }

            if transitioned_terminal {
                self.record_terminal(ticket);
            }
        }

        finished
    }

    fn start_queued(&mut self, data_cache: Arc<VkDataCache>, max_steps: usize) -> usize {
        if max_steps == 0 {
            return 0;
        }

        if self.has_in_flight_load() {
            return 0;
        }

        let mut started = 0usize;

        while started < max_steps {
            let Some(ticket) = self.queued_tickets.pop_front() else {
                break;
            };

            let Some(task) = self.tasks.get_mut(&ticket) else {
                continue;
            };

            match task {
                DeferredLoadTask::Model { state, .. } => {
                    let path = match state {
                        DeferredModelState::Queued { path } => path.clone(),
                        DeferredModelState::Cancelled => continue,
                        _ => continue,
                    };

                    let (sender, receiver) = mpsc::channel();
                    let task_cache = data_cache.clone();
                    std::thread::spawn(move || {
                        let result = load_model_gpu_ready(path, task_cache)
                            .and_then(|loaded| scene_world_to_fragment(loaded.scene_world));
                        let _ = sender.send(result);
                    });

                    *state = DeferredModelState::InFlight { receiver };
                    started += 1;
                }
                DeferredLoadTask::Texture { state, .. } => {
                    let path = match state {
                        DeferredTextureState::Queued { path } => path.clone(),
                        DeferredTextureState::Cancelled => continue,
                        _ => continue,
                    };

                    let (sender, receiver) = mpsc::channel();
                    let task_cache = data_cache.clone();
                    std::thread::spawn(move || {
                        let result = load_texture_gpu_ready(path, task_cache);
                        let _ = sender.send(result);
                    });

                    *state = DeferredTextureState::InFlight { receiver };
                    started += 1;
                }
            }
        }

        started
    }

    fn has_in_flight_load(&self) -> bool {
        self.tasks.values().any(|task| match task {
            DeferredLoadTask::Model { state, .. } => {
                matches!(state, DeferredModelState::InFlight { .. })
            }
            DeferredLoadTask::Texture { state, .. } => {
                matches!(state, DeferredTextureState::InFlight { .. })
            }
        })
    }

    fn cleanup_terminal_tickets(&mut self) {
        while self.terminal_tickets.len() > TERMINAL_TICKET_RETAIN_LIMIT {
            let Some(ticket) = self.terminal_tickets.pop_front() else {
                break;
            };

            if self
                .tasks
                .get(&ticket)
                .map(|task| task.is_terminal())
                .unwrap_or(false)
            {
                self.tasks.remove(&ticket);
            }
        }
    }

    fn record_terminal(&mut self, ticket: LoadTicket) {
        self.terminal_tickets.push_back(ticket);
    }
}

pub struct AssetManager<'a> {
    core: &'a mut VkRenderCore,
    load_tracker: &'a mut AssetLoadTracker,
}

impl<'a> AssetManager<'a> {
    pub(crate) fn new(core: &'a mut VkRenderCore, load_tracker: &'a mut AssetLoadTracker) -> Self {
        Self { core, load_tracker }
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
    /// May Stall: No (load records environment source data, first activation can still stall)
    pub fn load_environment(
        &mut self,
        source: EnvironmentSource,
    ) -> Result<EnvironmentHandle, AssetError> {
        let env_id = {
            let mut env_cache = self
                .core
                .data_cache
                .environment_cache
                .lock()
                .map_err(|_| poisoned_lock_err("environment_cache"))?;

            env_cache
                .import_environment(source)
                .map_err(|message| AssetError::Load {
                    path: None,
                    message,
                })?
        };

        self.core.clear_environment_failure(env_id);
        Ok(env_id)
    }

    /// Thread: Any
    /// May Stall: No
    pub fn environment_state(
        &self,
        env: EnvironmentHandle,
    ) -> Result<EnvironmentState, AssetError> {
        if let Some(message) = self.core.environment_failure(env) {
            return Ok(EnvironmentState::Failed(AssetError::Internal(message)));
        }

        let env_cache = self
            .core
            .data_cache
            .environment_cache
            .lock()
            .map_err(|_| poisoned_lock_err("environment_cache"))?;

        let skybox = env_cache
            .get_skybox(env)
            .map_err(|err| map_cache_err("environment", env.slot, env.generation, err))?;
        let has_env_maps = env_cache
            .get_env_map(env)
            .map_err(|err| map_cache_err("environment", env.slot, env.generation, err))?
            .is_some();

        match skybox {
            CachedEnvironment::Unloaded(_) => Ok(EnvironmentState::Unloaded),
            CachedEnvironment::Loaded(_) => {
                if has_env_maps {
                    Ok(EnvironmentState::Ready)
                } else {
                    Ok(EnvironmentState::Loading)
                }
            }
        }
    }

    /// Thread: Any
    /// May Stall: No
    pub fn default_environment(&self) -> EnvironmentHandle {
        self.core.default_env_id
    }

    /// Thread: Main
    /// May Stall: No
    pub fn request_model_load(&mut self, path: impl AsRef<Path>) -> Result<LoadTicket, AssetError> {
        Ok(self
            .load_tracker
            .request_model_load(path.as_ref().to_path_buf()))
    }

    /// Thread: Main
    /// May Stall: No
    pub fn request_texture_load(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<LoadTicket, AssetError> {
        Ok(self
            .load_tracker
            .request_texture_load(path.as_ref().to_path_buf()))
    }

    /// Thread: Main
    /// May Stall: No
    pub fn poll_model_load(&mut self, ticket: LoadTicket) -> LoadStatus<SceneFragment> {
        self.load_tracker.poll_model_load(ticket)
    }

    /// Thread: Main
    /// May Stall: No
    pub fn poll_texture_load(&mut self, ticket: LoadTicket) -> LoadStatus<TextureHandle> {
        self.load_tracker.poll_texture_load(ticket)
    }

    /// Thread: Main
    /// May Stall: No
    pub fn cancel_load(&mut self, ticket: LoadTicket) -> Result<(), AssetError> {
        self.load_tracker.cancel_load(ticket)
    }

    /// Thread: Main
    /// May Stall: No
    pub fn pending_load_count(&self) -> usize {
        self.load_tracker.pending_load_count()
    }

    /// Thread: Main
    /// May Stall: No
    pub fn has_pending_loads(&self) -> bool {
        self.load_tracker.has_pending_loads()
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

    /// Create a PBR material from runtime-generated parameters.
    ///
    /// Thread: Main
    /// May Stall: No
    ///
    /// Validates material parameters and creates a GPU-resident material. Material parameter
    /// values are clamped to safe ranges (metallic [0,1], roughness [0.02,1], etc).
    ///
    /// Returns a stable MaterialHandle that can be used with mesh creation and scene nodes.
    pub fn create_material_pbr(
        &mut self,
        desc: PbrMaterialDesc,
    ) -> Result<MaterialHandle, AssetError> {
        // Validate descriptor
        validate_material_desc(&desc)?;

        // Convert to internal MaterialMeta
        let material_meta = material_desc_to_meta(&desc);

        // Validate texture handles if provided
        {
            let texture_cache = self
                .core
                .data_cache
                .texture_cache
                .lock()
                .map_err(|_| poisoned_lock_err("texture_cache"))?;

            for tex_handle in material_meta.texture_ids.to_vec() {
                if tex_handle.slot >= TextureCache::DEFAULT_TEX_ITER_START as u32 {
                    texture_cache
                        .get_texture(tex_handle)
                        .map_err(|err| map_cache_err("texture", tex_handle.slot, tex_handle.generation, err))?;
                }
            }
        }

        // Add material to cache
        let material_id = {
            let mut texture_cache = self
                .core
                .data_cache
                .texture_cache
                .lock()
                .map_err(|_| poisoned_lock_err("texture_cache"))?;

            texture_cache.add_material(material_meta)
        };

        // Allocate GPU resources
        let allocation_result = {
            let mut texture_cache = self
                .core
                .data_cache
                .texture_cache
                .lock()
                .map_err(|_| poisoned_lock_err("texture_cache"))?;

            texture_cache.allocate_id(material_id, BufferPlacement::ContiguousPreferred, false)
        };

        // Handle allocation failure
        match allocation_result {
            LoadResult::Success(_) => Ok(material_id),
            LoadResult::Failed(_) => {
                // Rollback: deallocate the material
                let mut texture_cache = self
                    .core
                    .data_cache
                    .texture_cache
                    .lock()
                    .map_err(|_| poisoned_lock_err("texture_cache"))?;

                texture_cache.deallocate_materials(vec![material_id]);

                Err(AssetError::Internal(
                    "material GPU allocation failed".to_string(),
                ))
            }
        }
    }

    /// Upload a procedural mesh to the GPU.
    ///
    /// Thread: Main
    /// May Stall: No
    ///
    /// Validates mesh data (non-empty, valid indices, finite values) and uploads to GPU.
    /// If a material handle is provided, it must be valid and loaded.
    ///
    /// Returns a stable MeshHandle that can be attached to scene nodes.
    pub fn upload_procedural_mesh(
        &mut self,
        mesh: ProceduralMeshData,
    ) -> Result<MeshHandle, AssetError> {
        // Validate mesh data
        validate_procedural_mesh(&mesh)?;

        // Validate material handle if provided
        if let Some(material_handle) = mesh.material {
            let texture_cache = self
                .core
                .data_cache
                .texture_cache
                .lock()
                .map_err(|_| poisoned_lock_err("texture_cache"))?;

            // Verify material exists
            texture_cache
                .get_material(material_handle)
                .map_err(|err| {
                    map_cache_err("material", material_handle.slot, material_handle.generation, err)
                })?;

            // Verify material is loaded (not just cached)
            texture_cache
                .get_loaded_material(material_handle)
                .map_err(|err| {
                    map_cache_err("material", material_handle.slot, material_handle.generation, err)
                })?;
        }

        // Convert vertices to internal format
        let vertices: Vec<Vertex> = mesh
            .vertices
            .iter()
            .map(procedural_vertex_to_gpu)
            .collect();

        // Create internal MeshMeta
        let mesh_meta = MeshMeta {
            name: mesh.name,
            indices: mesh.indices,
            vertices,
            material_index: mesh.material,
        };

        // Add mesh to cache
        let mesh_id = {
            let mut mesh_cache = self
                .core
                .data_cache
                .mesh_cache
                .lock()
                .map_err(|_| poisoned_lock_err("mesh_cache"))?;

            mesh_cache.add(mesh_meta)
        };

        // Allocate GPU resources
        let allocation_result = {
            let mut mesh_cache = self
                .core
                .data_cache
                .mesh_cache
                .lock()
                .map_err(|_| poisoned_lock_err("mesh_cache"))?;

            mesh_cache.allocate_id(mesh_id, BufferPlacement::ContiguousPreferred, false)
        };

        // Handle allocation failure
        match allocation_result {
            LoadResult::Success(_) => Ok(mesh_id),
            LoadResult::Failed(_) => {
                // Rollback: deallocate the mesh
                let mut mesh_cache = self
                    .core
                    .data_cache
                    .mesh_cache
                    .lock()
                    .map_err(|_| poisoned_lock_err("mesh_cache"))?;

                mesh_cache.deallocate_id(mesh_id);

                Err(AssetError::Internal("mesh GPU allocation failed".to_string()))
            }
        }
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
        self.core.pump_transfer_submissions(usize::MAX);
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

fn map_environment_load_err(path: &Path, message: String) -> AssetError {
    AssetError::Load {
        path: Some(path.to_path_buf()),
        message,
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

//////////////////////////////////
// PROCEDURAL ASSET PUBLIC API //
//////////////////////////////////

/// Public vertex format for procedural mesh creation.
///
/// This is the facade type exposed to library consumers. It converts to the internal
/// `Vertex` type which has additional fields for advanced features like skinning.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProceduralVertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub tangent: glam::Vec4,
    pub uv0: glam::Vec2,
    pub color: glam::Vec4,
}

/// PBR material descriptor for runtime material creation.
///
/// All texture handles are optional. When `None`, the renderer will use default textures.
/// Factor values will be clamped to safe ranges during material creation.
#[derive(Clone, Debug)]
pub struct PbrMaterialDesc {
    pub base_color: glam::Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub base_color_tex: Option<TextureHandle>,
    pub normal_tex: Option<TextureHandle>,
    pub metallic_roughness_tex: Option<TextureHandle>,
    pub ao_tex: Option<TextureHandle>,
    pub emissive_tex: Option<TextureHandle>,
    pub emissive_factor: glam::Vec3,
    pub emissive_strength: f32,
}

impl Default for PbrMaterialDesc {
    fn default() -> Self {
        Self {
            base_color: glam::Vec4::ONE,
            metallic: 0.0,
            roughness: 0.5,
            base_color_tex: None,
            normal_tex: None,
            metallic_roughness_tex: None,
            ao_tex: None,
            emissive_tex: None,
            emissive_factor: glam::Vec3::ZERO,
            emissive_strength: 0.0,
        }
    }
}

/// Procedural mesh data container for runtime mesh creation.
///
/// Vertices and indices must form valid triangles. The renderer will validate:
/// - Non-empty vertices and indices
/// - Index count divisible by 3 (complete triangles)
/// - All indices within bounds
/// - All vertex components are finite (no NaN/Inf)
/// - Tangent.w is either +1.0 or -1.0
#[derive(Clone, Debug, Default)]
pub struct ProceduralMeshData {
    pub name: String,
    pub vertices: Vec<ProceduralVertex>,
    pub indices: Vec<u32>,
    pub material: Option<MaterialHandle>,
}

/// Validate a PBR material descriptor.
fn validate_material_desc(desc: &PbrMaterialDesc) -> Result<(), AssetError> {
    // Check for non-finite values
    if !desc.metallic.is_finite() {
        return Err(AssetError::Internal(
            "material metallic factor is not finite".to_string(),
        ));
    }
    if !desc.roughness.is_finite() {
        return Err(AssetError::Internal(
            "material roughness factor is not finite".to_string(),
        ));
    }
    if !desc.emissive_strength.is_finite() || desc.emissive_strength < 0.0 {
        return Err(AssetError::Internal(
            "material emissive strength is not finite or negative".to_string(),
        ));
    }

    // Check color vector for NaN/Inf
    if !desc.base_color.is_finite() {
        return Err(AssetError::Internal(
            "material base color contains non-finite values".to_string(),
        ));
    }
    if !desc.emissive_factor.is_finite() {
        return Err(AssetError::Internal(
            "material emissive factor contains non-finite values".to_string(),
        ));
    }

    Ok(())
}

/// Validate procedural mesh data.
fn validate_procedural_mesh(mesh: &ProceduralMeshData) -> Result<(), AssetError> {
    let name = if mesh.name.is_empty() {
        "unnamed mesh"
    } else {
        mesh.name.as_str()
    };

    // Check for empty buffers
    if mesh.vertices.is_empty() {
        return Err(AssetError::Internal(format!(
            "procedural mesh '{}' has empty vertex buffer",
            name
        )));
    }

    if mesh.indices.is_empty() {
        return Err(AssetError::Internal(format!(
            "procedural mesh '{}' has empty index buffer",
            name
        )));
    }

    // Check index count divisible by 3
    if mesh.indices.len() % 3 != 0 {
        return Err(AssetError::Internal(format!(
            "procedural mesh '{}' index count ({}) is not divisible by 3 (incomplete triangles)",
            name,
            mesh.indices.len()
        )));
    }

    // Check for out-of-bounds indices
    let vertex_count = mesh.vertices.len();
    for (i, &index) in mesh.indices.iter().enumerate() {
        if (index as usize) >= vertex_count {
            return Err(AssetError::Internal(format!(
                "procedural mesh '{}' has out-of-bounds index at position {}: index {} exceeds vertex count {}",
                name, i, index, vertex_count
            )));
        }
    }

    // Validate vertex data
    for (i, vertex) in mesh.vertices.iter().enumerate() {
        if !vertex.position.is_finite() {
            return Err(AssetError::Internal(format!(
                "procedural mesh '{}' vertex {} has non-finite position",
                name, i
            )));
        }
        if !vertex.normal.is_finite() {
            return Err(AssetError::Internal(format!(
                "procedural mesh '{}' vertex {} has non-finite normal",
                name, i
            )));
        }
        if !vertex.tangent.is_finite() {
            return Err(AssetError::Internal(format!(
                "procedural mesh '{}' vertex {} has non-finite tangent",
                name, i
            )));
        }
        if !vertex.uv0.is_finite() {
            return Err(AssetError::Internal(format!(
                "procedural mesh '{}' vertex {} has non-finite UV coordinates",
                name, i
            )));
        }
        if !vertex.color.is_finite() {
            return Err(AssetError::Internal(format!(
                "procedural mesh '{}' vertex {} has non-finite color",
                name, i
            )));
        }

        // Validate tangent handedness (w component must be +1.0 or -1.0)
        let tangent_w = vertex.tangent.w;
        if (tangent_w - 1.0).abs() > 0.01 && (tangent_w + 1.0).abs() > 0.01 {
            return Err(AssetError::Internal(format!(
                "procedural mesh '{}' vertex {} has invalid tangent.w ({}), must be +1.0 or -1.0",
                name, i, tangent_w
            )));
        }
    }

    Ok(())
}

/// Convert PbrMaterialDesc to internal MaterialMeta with clamping.
fn material_desc_to_meta(desc: &PbrMaterialDesc) -> MaterialMeta {
    let mut meta = MaterialMeta::default();

    // Clamp base color channels to [0.0, 1.0]
    meta.material_values.base_color_factor = desc.base_color.clamp(
        glam::Vec4::ZERO,
        glam::Vec4::ONE,
    );

    // Clamp metallic to [0.0, 1.0]
    meta.material_values.metallic_factor = desc.metallic.clamp(0.0, 1.0);

    // Clamp roughness to [0.02, 1.0] (practical minimum to avoid specular aliasing)
    meta.material_values.roughness_factor = desc.roughness.clamp(0.02, 1.0);

    // Emissive factor (no upper clamp, but ensure non-negative)
    meta.material_values.emissive_factor = desc.emissive_factor.max(glam::Vec3::ZERO).extend(0.0);
    meta.material_values.emissive_strength = desc.emissive_strength.max(0.0);

    // Set texture handles if provided
    if let Some(tex) = desc.base_color_tex {
        meta.texture_ids.base_color = tex;
        meta.material_values.base_color_uv_set = 0;
    }

    if let Some(tex) = desc.metallic_roughness_tex {
        meta.texture_ids.metallic_roughness = tex;
        meta.material_values.met_rough_uv_set = 0;
    }

    if let Some(tex) = desc.normal_tex {
        meta.texture_ids.normal_map = tex;
        meta.material_values.normal_uv_set = 0;
    }

    if let Some(tex) = desc.ao_tex {
        meta.texture_ids.occlusion_map = tex;
        meta.material_values.occlusion_uv_set = 0;
    }

    if let Some(tex) = desc.emissive_tex {
        meta.texture_ids.emissive_map = tex;
        meta.material_values.emissive_uv_set = 0;
    }

    meta
}

/// Convert ProceduralVertex to internal Vertex format.
fn procedural_vertex_to_gpu(v: &ProceduralVertex) -> Vertex {
    Vertex {
        position: v.position,
        uv0_x: v.uv0.x,
        normal: v.normal,
        uv0_y: v.uv0.y,
        color: v.color,
        tangent: v.tangent,
        joints: glam::UVec4::ZERO,
        weights: glam::Vec4::ZERO,
        uv1_x: 0.0,
        uv1_y: 0.0,
        _pad: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_desc_clamps_metallic_and_roughness() {
        let desc = PbrMaterialDesc {
            metallic: 1.5,      // Should clamp to 1.0
            roughness: -0.5,    // Should clamp to 0.02
            ..Default::default()
        };

        let meta = material_desc_to_meta(&desc);
        assert_eq!(meta.material_values.metallic_factor, 1.0);
        assert_eq!(meta.material_values.roughness_factor, 0.02);
    }

    #[test]
    fn material_desc_rejects_non_finite() {
        let desc_nan_metallic = PbrMaterialDesc {
            metallic: f32::NAN,
            ..Default::default()
        };
        assert!(validate_material_desc(&desc_nan_metallic).is_err());

        let desc_inf_roughness = PbrMaterialDesc {
            roughness: f32::INFINITY,
            ..Default::default()
        };
        assert!(validate_material_desc(&desc_inf_roughness).is_err());

        let desc_neg_emissive = PbrMaterialDesc {
            emissive_strength: -1.0,
            ..Default::default()
        };
        assert!(validate_material_desc(&desc_neg_emissive).is_err());

        let desc_nan_color = PbrMaterialDesc {
            base_color: glam::Vec4::new(f32::NAN, 1.0, 1.0, 1.0),
            ..Default::default()
        };
        assert!(validate_material_desc(&desc_nan_color).is_err());
    }

    #[test]
    fn procedural_mesh_rejects_empty_buffers() {
        let empty_vertices = ProceduralMeshData {
            name: "test".to_string(),
            vertices: vec![],
            indices: vec![0, 1, 2],
            material: None,
        };
        assert!(validate_procedural_mesh(&empty_vertices).is_err());

        let empty_indices = ProceduralMeshData {
            name: "test".to_string(),
            vertices: vec![ProceduralVertex::default()],
            indices: vec![],
            material: None,
        };
        assert!(validate_procedural_mesh(&empty_indices).is_err());
    }

    #[test]
    fn procedural_mesh_rejects_non_triangle_index_count() {
        let mesh = ProceduralMeshData {
            name: "test".to_string(),
            vertices: vec![
                ProceduralVertex::default(),
                ProceduralVertex::default(),
            ],
            indices: vec![0, 1], // Only 2 indices, not divisible by 3
            material: None,
        };
        assert!(validate_procedural_mesh(&mesh).is_err());
    }

    #[test]
    fn procedural_mesh_rejects_out_of_bounds_index() {
        let mesh = ProceduralMeshData {
            name: "test".to_string(),
            vertices: vec![
                ProceduralVertex::default(),
                ProceduralVertex::default(),
            ],
            indices: vec![0, 1, 5], // Index 5 is out of bounds (only 2 vertices)
            material: None,
        };
        assert!(validate_procedural_mesh(&mesh).is_err());
    }

    #[test]
    fn procedural_mesh_rejects_non_finite_vertex() {
        let mut mesh = ProceduralMeshData {
            name: "test".to_string(),
            vertices: vec![ProceduralVertex {
                position: glam::Vec3::new(f32::NAN, 0.0, 0.0),
                normal: glam::Vec3::Y,
                tangent: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
                uv0: glam::Vec2::ZERO,
                color: glam::Vec4::ONE,
            }],
            indices: vec![0, 0, 0],
            material: None,
        };
        assert!(validate_procedural_mesh(&mesh).is_err());

        mesh.vertices[0].position = glam::Vec3::ZERO;
        mesh.vertices[0].normal = glam::Vec3::new(f32::INFINITY, 0.0, 0.0);
        assert!(validate_procedural_mesh(&mesh).is_err());
    }

    #[test]
    fn procedural_mesh_rejects_invalid_tangent_w() {
        let mesh = ProceduralMeshData {
            name: "test".to_string(),
            vertices: vec![ProceduralVertex {
                position: glam::Vec3::ZERO,
                normal: glam::Vec3::Y,
                tangent: glam::Vec4::new(1.0, 0.0, 0.0, 0.5), // Invalid w component
                uv0: glam::Vec2::ZERO,
                color: glam::Vec4::ONE,
            }],
            indices: vec![0, 0, 0],
            material: None,
        };
        assert!(validate_procedural_mesh(&mesh).is_err());
    }

    #[test]
    fn procedural_mesh_accepts_valid_data() {
        let mesh = ProceduralMeshData {
            name: "test".to_string(),
            vertices: vec![
                ProceduralVertex {
                    position: glam::Vec3::new(0.0, 0.0, 0.0),
                    normal: glam::Vec3::Y,
                    tangent: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
                    uv0: glam::Vec2::ZERO,
                    color: glam::Vec4::ONE,
                },
                ProceduralVertex {
                    position: glam::Vec3::new(1.0, 0.0, 0.0),
                    normal: glam::Vec3::Y,
                    tangent: glam::Vec4::new(1.0, 0.0, 0.0, -1.0),
                    uv0: glam::Vec2::ZERO,
                    color: glam::Vec4::ONE,
                },
                ProceduralVertex {
                    position: glam::Vec3::new(0.0, 0.0, 1.0),
                    normal: glam::Vec3::Y,
                    tangent: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
                    uv0: glam::Vec2::ZERO,
                    color: glam::Vec4::ONE,
                },
            ],
            indices: vec![0, 1, 2],
            material: None,
        };
        assert!(validate_procedural_mesh(&mesh).is_ok());
    }

    #[test]
    fn material_desc_accepts_valid_data() {
        let desc = PbrMaterialDesc {
            base_color: glam::Vec4::ONE,
            metallic: 0.5,
            roughness: 0.5,
            emissive_factor: glam::Vec3::new(1.0, 0.5, 0.0),
            emissive_strength: 2.0,
            ..Default::default()
        };
        assert!(validate_material_desc(&desc).is_ok());
    }
}
