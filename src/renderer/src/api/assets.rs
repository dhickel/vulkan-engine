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
use crate::data::gpu_data::TextureMeta;
use crate::data::handles::{
    CacheError, EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle,
};
use crate::scene::scene_world::{SceneNodeId, SceneWorld};
use crate::vulkan::vk_render::VkRenderCore;
use crate::vulkan::vk_storage::BufferPlacement;

use super::errors::AssetError;

const TERMINAL_TICKET_RETAIN_LIMIT: usize = 2_048;

#[derive(Debug)]
pub enum EnvironmentSource {
    EquirectFile(PathBuf),
    CubemapDir(PathBuf),
}

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
        let env_id =
            {
                let mut env_cache = self
                    .core
                    .data_cache
                    .environment_cache
                    .lock()
                    .map_err(|_| poisoned_lock_err("environment_cache"))?;

                match source {
                    EnvironmentSource::EquirectFile(path) => env_cache
                        .load_cubemap_file(path.as_path())
                        .map_err(|message| map_environment_load_err(&path, message))?,
                    EnvironmentSource::CubemapDir(path) => env_cache
                        .load_cubemap_dir(path.as_path())
                        .map_err(|message| map_environment_load_err(&path, message))?,
                }
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
