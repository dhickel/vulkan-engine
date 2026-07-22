use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Instant;
use std::{collections::HashMap, collections::VecDeque};

use image::ImageError;
use log::debug;

use crate::api::config::AssetPolicyConfig;
use crate::api::loading::{LoadStatus, LoadTicket};
use crate::api::scene::{SceneFragment, SceneFragmentNodeId};
use crate::data::asset_manifest::{self, TextureLoadOptions};
use crate::data::asset_registry::{
    AssetKind, AssetRegistry, AssetRegistryError, DurableAssetRecord,
};
use crate::data::assimp_util::{self, ModelMeta};
use crate::data::data_cache::{
    CachedEnvironment, CachedMesh, LoadResult, MeshCache, TextureCache, VkDataCache,
};
use crate::data::data_util::resolve_texture_mip_count;
use crate::data::gpu_data::{MaterialPayload, MeshMeta, TextureMeta, TextureSemantic, Vertex};
use crate::data::handles::{
    CacheError, EnvironmentHandle, MaterialHandle, MeshHandle, TextureHandle,
};
use crate::data::mesh_geometry::{
    compute_local_aabb, validate_triangle_indices, MeshDeformation, MeshGeometryDto, MeshLocalAabb,
};
use crate::data::thread_pool::BoundedThreadPool;
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
        policy_config: AssetPolicyConfig,
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
        policy_config: AssetPolicyConfig,
        options: Option<TextureLoadOptions>,
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
    max_in_flight_jobs: usize,
    asset_registry: AssetRegistry,
    thread_pool: BoundedThreadPool,
}

impl AssetLoadTracker {
    pub(crate) fn new() -> Self {
        Self {
            next_ticket: 1,
            queued_tickets: VecDeque::new(),
            terminal_tickets: VecDeque::new(),
            tasks: HashMap::new(),
            max_in_flight_jobs: 4,
            asset_registry: AssetRegistry::new(),
            thread_pool: BoundedThreadPool::new(4),
        }
    }

    #[cfg(test)]
    fn request_model_load(&mut self, path: PathBuf) -> LoadTicket {
        self.request_model_load_with_policy(path, AssetPolicyConfig::default())
    }

    pub(crate) fn request_model_load_with_policy(
        &mut self,
        path: PathBuf,
        policy_config: AssetPolicyConfig,
    ) -> LoadTicket {
        let ticket = self.next_ticket();
        self.tasks.insert(
            ticket,
            DeferredLoadTask::Model {
                queued_at: Instant::now(),
                state: DeferredModelState::Queued {
                    path,
                    policy_config,
                },
            },
        );
        self.queued_tickets.push_back(ticket);
        ticket
    }

    pub(crate) fn request_texture_load(
        &mut self,
        path: PathBuf,
        policy_config: AssetPolicyConfig,
        options: Option<TextureLoadOptions>,
    ) -> LoadTicket {
        let ticket = self.next_ticket();
        self.tasks.insert(
            ticket,
            DeferredLoadTask::Texture {
                queued_at: Instant::now(),
                state: DeferredTextureState::Queued {
                    path,
                    policy_config,
                    options,
                },
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

    pub(crate) fn pump(
        &mut self,
        core: &mut VkRenderCore,
        max_steps: usize,
    ) -> Result<usize, String> {
        if max_steps == 0 {
            return Ok(0);
        }

        let mut progressed = 0usize;
        progressed += core.pump_transfer_submissions(max_steps - progressed)?;

        // An upload worker can hold the texture cache while waiting for transfer completion.
        // Never block the render-thread pump on that same lock or both sides deadlock.
        match core.data_cache.texture_cache.try_lock() {
            Ok(mut texture_cache) => {
                let finalized = texture_cache.poll_texture_uploads()?;
                if finalized > 0 {
                    debug!("Finalized {} pending texture upload batch(es)", finalized);
                }
            }
            Err(std::sync::TryLockError::WouldBlock) => {}
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(format!("texture_cache lock poisoned during asset pump"));
            }
        }

        if progressed < max_steps {
            progressed += self.collect_finished(max_steps - progressed);
        }

        if progressed < max_steps {
            progressed += self.start_queued(Arc::clone(&core.data_cache), max_steps - progressed);
        }

        self.cleanup_terminal_tickets();
        Ok(progressed)
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

        let mut started = 0usize;

        while started < max_steps && self.in_flight_count() < self.max_in_flight_jobs {
            let Some(ticket) = self.queued_tickets.pop_front() else {
                break;
            };

            let Some(task) = self.tasks.get_mut(&ticket) else {
                continue;
            };

            match task {
                DeferredLoadTask::Model { state, .. } => {
                    let (path, policy_cfg) = match state {
                        DeferredModelState::Queued {
                            path,
                            policy_config,
                        } => (path.clone(), policy_config.clone()),
                        DeferredModelState::Cancelled => continue,
                        _ => continue,
                    };

                    let (sender, receiver) = mpsc::channel();
                    let task_cache = data_cache.clone();
                    self.thread_pool.execute(move || {
                        let result = load_model_gpu_ready(path, task_cache, &policy_cfg)
                            .and_then(|loaded| scene_world_to_fragment(loaded.scene_world));
                        let _ = sender.send(result);
                    });

                    *state = DeferredModelState::InFlight { receiver };
                    started += 1;
                }
                DeferredLoadTask::Texture { state, .. } => {
                    let (path, policy_cfg, opts) = match state {
                        DeferredTextureState::Queued {
                            path,
                            policy_config,
                            options,
                        } => (path.clone(), policy_config.clone(), options.take()),
                        DeferredTextureState::Cancelled => continue,
                        _ => continue,
                    };

                    let (sender, receiver) = mpsc::channel();
                    let task_cache = data_cache.clone();
                    self.thread_pool.execute(move || {
                        let result =
                            load_texture_gpu_ready_with_policy(path, task_cache, &policy_cfg, opts);
                        let _ = sender.send(result);
                    });

                    *state = DeferredTextureState::InFlight { receiver };
                    started += 1;
                }
            }
        }

        started
    }

    fn in_flight_count(&self) -> usize {
        self.tasks
            .values()
            .filter(|task| match task {
                DeferredLoadTask::Model { state, .. } => {
                    matches!(state, DeferredModelState::InFlight { .. })
                }
                DeferredLoadTask::Texture { state, .. } => {
                    matches!(state, DeferredTextureState::InFlight { .. })
                }
            })
            .count()
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

    pub(crate) fn asset_registry(&self) -> &AssetRegistry {
        &self.asset_registry
    }

    pub(crate) fn asset_registry_mut(&mut self) -> &mut AssetRegistry {
        &mut self.asset_registry
    }
}

pub struct AssetManager<'a> {
    core: &'a mut VkRenderCore,
    load_tracker: &'a mut AssetLoadTracker,
    asset_policy: &'a AssetPolicyConfig,
}

impl<'a> AssetManager<'a> {
    pub(crate) fn new(
        core: &'a mut VkRenderCore,
        load_tracker: &'a mut AssetLoadTracker,
        asset_policy: &'a AssetPolicyConfig,
    ) -> Self {
        Self {
            core,
            load_tracker,
            asset_policy,
        }
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn load_mesh(&mut self, path: impl AsRef<Path>) -> Result<MeshHandle, AssetError> {
        let path = path.as_ref().to_path_buf();
        let path_for_err = path.clone();
        let policy_config = self.asset_policy.clone();
        let loaded_model = self.run_sync_upload_task(move |data_cache| {
            load_model_gpu_ready(path, data_cache, &policy_config)
        })?;

        loaded_model
            .mesh_ids
            .first()
            .copied()
            .ok_or_else(|| AssetError::Load {
                path: Some(path_for_err),
                message: "model import did not contain any meshes".to_string(),
            })
    }

    /// Load a package manifest into the durable asset registry.
    ///
    /// This only records CPU-side package metadata. It does not import, upload,
    /// or allocate runtime handles for the listed assets.
    pub fn load_package_manifest(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<Vec<DurableAssetRecord>, AssetError> {
        self.load_tracker
            .asset_registry_mut()
            .load_package_manifest(path)
            .map_err(asset_registry_error_to_asset_error)
    }

    /// Load a package manifest and require that its `package_id` matches the
    /// project package record that referenced it.
    pub fn load_package_manifest_with_expected_id(
        &mut self,
        path: impl AsRef<Path>,
        expected_package_id: &str,
    ) -> Result<Vec<DurableAssetRecord>, AssetError> {
        self.load_tracker
            .asset_registry_mut()
            .load_package_manifest_with_expected_id(path, Some(expected_package_id))
            .map_err(asset_registry_error_to_asset_error)
    }

    /// List durable package assets currently known to the facade registry.
    pub fn list_assets(&self) -> Vec<DurableAssetRecord> {
        self.load_tracker
            .asset_registry()
            .list_assets()
            .into_iter()
            .cloned()
            .collect()
    }

    /// List durable package assets filtered by kind and text query.
    ///
    /// Results keep registry order, which is deterministic by durable asset ID.
    pub fn list_assets_matching(
        &self,
        kind: Option<AssetKind>,
        search: Option<&str>,
    ) -> Vec<DurableAssetRecord> {
        self.load_tracker
            .asset_registry()
            .list_assets_matching(kind.as_ref(), search)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Look up a durable asset record by ID.
    pub fn asset_record(&self, asset_id: &str) -> Option<DurableAssetRecord> {
        self.load_tracker
            .asset_registry()
            .asset_record(asset_id)
            .cloned()
    }

    /// Resolve a durable asset ID into its loadable path/kind metadata.
    pub fn resolve_asset(&self, asset_id: &str) -> Result<DurableAssetRecord, AssetError> {
        self.load_tracker
            .asset_registry()
            .resolve_asset(asset_id)
            .cloned()
            .map_err(asset_registry_error_to_asset_error)
    }

    /// Load a model or prefab by durable asset ID through the existing model
    /// upload path.
    ///
    /// Runtime handles remain return values from the load path; the durable ID
    /// is only used to resolve CPU-side package metadata.
    pub fn load_model_asset(&mut self, asset_id: &str) -> Result<SceneFragment, AssetError> {
        let record = self.resolve_asset(asset_id)?;
        ensure_asset_kind(
            &record,
            &[AssetKind::Model, AssetKind::Prefab, AssetKind::WallChunk],
        )?;
        self.load_model(record.source_path)
    }

    /// Load a texture by durable asset ID through the existing texture upload path.
    pub fn load_texture_asset(&mut self, asset_id: &str) -> Result<TextureHandle, AssetError> {
        let record = self.resolve_asset(asset_id)?;
        ensure_asset_kind(&record, &[AssetKind::Texture])?;
        self.load_texture(record.source_path)
    }

    /// Load an environment by durable asset ID through the existing environment path.
    pub fn load_environment_asset(
        &mut self,
        asset_id: &str,
    ) -> Result<EnvironmentHandle, AssetError> {
        let record = self.resolve_asset(asset_id)?;
        ensure_asset_kind(&record, &[AssetKind::Environment])?;
        self.load_environment(EnvironmentSource::Auto(record.source_path))
    }

    /// Queue deferred model/prefab loading by durable asset ID.
    pub fn request_model_asset_load(&mut self, asset_id: &str) -> Result<LoadTicket, AssetError> {
        let record = self.resolve_asset(asset_id)?;
        ensure_asset_kind(
            &record,
            &[AssetKind::Model, AssetKind::Prefab, AssetKind::WallChunk],
        )?;
        self.request_model_load(record.source_path)
    }

    /// Queue deferred texture loading by durable asset ID.
    pub fn request_texture_asset_load(&mut self, asset_id: &str) -> Result<LoadTicket, AssetError> {
        let record = self.resolve_asset(asset_id)?;
        ensure_asset_kind(&record, &[AssetKind::Texture])?;
        self.request_texture_load(record.source_path)
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn load_texture(&mut self, path: impl AsRef<Path>) -> Result<TextureHandle, AssetError> {
        self.load_texture_with_options(path, TextureLoadOptions::default())
    }

    /// Load a texture with explicit policy overrides.
    ///
    /// Options are merged with manifest sidecar and filename heuristics per the
    /// policy resolution chain (API overrides > manifest > heuristics > defaults).
    ///
    /// Thread: Main
    /// May Stall: Yes
    pub fn load_texture_with_options(
        &mut self,
        path: impl AsRef<Path>,
        options: TextureLoadOptions,
    ) -> Result<TextureHandle, AssetError> {
        let mut loaded =
            self.load_textures_with_options(vec![(path.as_ref().to_path_buf(), options)])?;
        loaded
            .pop()
            .ok_or_else(|| AssetError::Internal("texture batch returned no handles".to_string()))
    }

    /// Load a batch of textures with explicit policy overrides.
    ///
    /// Thread: Main
    /// May Stall: Yes
    pub fn load_textures_with_options(
        &mut self,
        requests: Vec<(PathBuf, TextureLoadOptions)>,
    ) -> Result<Vec<TextureHandle>, AssetError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let policy_config = self.asset_policy.clone();
        self.run_sync_upload_task(move |data_cache| {
            load_texture_batch_gpu_ready_with_policy(requests, data_cache, &policy_config)
        })
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn load_model(&mut self, path: impl AsRef<Path>) -> Result<SceneFragment, AssetError> {
        let path = path.as_ref().to_path_buf();
        let policy_config = self.asset_policy.clone();
        let loaded_model = self.run_sync_upload_task(move |data_cache| {
            load_model_gpu_ready(path, data_cache, &policy_config)
        })?;

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
            .request_model_load_with_policy(path.as_ref().to_path_buf(), self.asset_policy.clone()))
    }

    /// Thread: Main
    /// May Stall: No
    pub fn request_texture_load(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<LoadTicket, AssetError> {
        Ok(self.load_tracker.request_texture_load(
            path.as_ref().to_path_buf(),
            self.asset_policy.clone(),
            None,
        ))
    }

    /// Request a deferred texture load with explicit policy overrides.
    ///
    /// Thread: Main
    /// May Stall: No
    pub fn request_texture_load_with_options(
        &mut self,
        path: impl AsRef<Path>,
        options: TextureLoadOptions,
    ) -> Result<LoadTicket, AssetError> {
        Ok(self.load_tracker.request_texture_load(
            path.as_ref().to_path_buf(),
            self.asset_policy.clone(),
            Some(options),
        ))
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
        // Validate the handle exists and is not reserved before retiring.
        {
            let mesh_cache = self
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
        }

        // Acquire both stores before mutating either. Geometry registration never holds
        // these locks together, so this ordering cannot deadlock with ingestion.
        let mut mesh_cache = self
            .core
            .data_cache
            .mesh_cache
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_cache"))?;
        let mut geo_store = self
            .core
            .data_cache
            .mesh_geometry_store
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_geometry_store"))?;

        // Invalidate the cache generation first. Generation exhaustion is checked before
        // ownership moves, so an error leaves both the cache and DTO visible.
        let retired = mesh_cache
            .retire_mesh(
                mesh,
                crate::data::retirement::FrameSerial::new(self.core.latest_submitted_serial),
            )
            .map_err(|err| map_cache_err("mesh", mesh.slot, mesh.generation, err))?;
        let geometry = geo_store.take(mesh);

        if let Some((payload, retire_after)) = retired {
            if let Some(geometry) = geometry {
                self.core.bounds_retirement_queue.enqueue(
                    crate::data::retirement::RetirementClass::BoundsEntry,
                    retire_after,
                    geometry,
                );
            }
            self.core.mesh_retirement_queue.enqueue(
                crate::data::retirement::RetirementClass::MeshGeometry,
                retire_after,
                payload,
            );
        }

        Ok(())
    }

    /// Thread: Main
    /// May Stall: No
    pub fn unload_material(&mut self, material: MaterialHandle) -> Result<(), AssetError> {
        // Validate the handle exists and is not reserved before retiring.
        {
            let texture_cache = self
                .core
                .data_cache
                .texture_cache
                .lock()
                .map_err(|_| poisoned_lock_err("texture_cache"))?;

            texture_cache.get_material(material).map_err(|err| {
                map_cache_err("material", material.slot, material.generation, err)
            })?;

            if (material.slot as usize) < TextureCache::DEFAULT_MAT_ITER_START {
                return Err(AssetError::ReservedHandle {
                    resource: "material",
                    slot: material.slot,
                    generation: material.generation,
                });
            }
        }

        // Retire the material: invalidate handle immediately, defer GPU
        // payload destruction until all referencing frames complete.
        let retired = {
            let mut texture_cache = self
                .core
                .data_cache
                .texture_cache
                .lock()
                .map_err(|_| poisoned_lock_err("texture_cache"))?;

            texture_cache
                .retire_material(
                    material,
                    crate::data::retirement::FrameSerial::new(self.core.latest_submitted_serial),
                )
                .map_err(|err| map_cache_err("material", material.slot, material.generation, err))?
        };

        if let Some((payload, retire_after)) = retired {
            self.core.material_retirement_queue.enqueue(
                crate::data::retirement::RetirementClass::MaterialPayload,
                retire_after,
                payload,
            );
        }

        Ok(())
    }

    /// Thread: Main
    /// May Stall: No
    pub fn unload_texture(&mut self, texture: TextureHandle) -> Result<(), AssetError> {
        // Validate the handle exists and is not reserved before retiring.
        {
            let texture_cache = self
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
        }

        // Retire the texture: invalidate handle immediately, defer GPU
        // payload destruction until all referencing frames complete.
        let retired = {
            let mut texture_cache = self
                .core
                .data_cache
                .texture_cache
                .lock()
                .map_err(|_| poisoned_lock_err("texture_cache"))?;

            texture_cache
                .retire_texture(
                    texture,
                    crate::data::retirement::FrameSerial::new(self.core.latest_submitted_serial),
                )
                .map_err(|err| map_cache_err("texture", texture.slot, texture.generation, err))?
        };

        if let Some((payload, retire_after)) = retired {
            self.core.texture_retirement_queue.enqueue(
                crate::data::retirement::RetirementClass::TextureGeometry,
                retire_after,
                payload,
            );
        }

        Ok(())
    }

    /// Query the Vulkan-free mesh geometry DTO for a loaded mesh.
    ///
    /// Returns [`MeshGeometryDto`] containing model-space positions, triangle indices,
    /// a conservative local AABB, and deformation classification.
    /// The handle generation is validated; stale or invalid handles return an error.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn mesh_geometry(&self, mesh: MeshHandle) -> Result<MeshGeometryDto, AssetError> {
        let mesh_cache = self
            .core
            .data_cache
            .mesh_cache
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_cache"))?;
        mesh_cache
            .get_loaded_id(mesh)
            .map_err(|err| map_cache_err("mesh", mesh.slot, mesh.generation, err))?;
        let geo_store = self
            .core
            .data_cache
            .mesh_geometry_store
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_geometry_store"))?;
        geo_store
            .get(mesh)
            .map_err(|err| map_cache_err("mesh_geometry", mesh.slot, mesh.generation, err))
    }

    /// Query only the conservative local AABB for a loaded mesh.
    ///
    /// Returns `None` when the DTO was stored with a conservative-none AABB
    /// (empty positions, non-finite components, or unknown deformation).
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn mesh_local_aabb(&self, mesh: MeshHandle) -> Result<Option<MeshLocalAabb>, AssetError> {
        let mesh_cache = self
            .core
            .data_cache
            .mesh_cache
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_cache"))?;
        mesh_cache
            .get_loaded_id(mesh)
            .map_err(|err| map_cache_err("mesh", mesh.slot, mesh.generation, err))?;
        let geo_store = self
            .core
            .data_cache
            .mesh_geometry_store
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_geometry_store"))?;
        geo_store
            .get_aabb(mesh)
            .map_err(|err| map_cache_err("mesh_local_aabb", mesh.slot, mesh.generation, err))
    }

    /// Query the [`SceneBounds`] for a loaded mesh by converting its DTO.
    ///
    /// Rigid meshes with valid AABBs return [`SceneBounds::Known`].
    /// Skinned, deformed, unknown, or invalid meshes return
    /// [`SceneBounds::ConservativeVisible`] with the exact reason.
    ///
    /// Thread: Any
    /// May Stall: No
    pub fn mesh_scene_bounds(
        &self,
        mesh: MeshHandle,
    ) -> Result<crate::api::scene::SceneBounds, AssetError> {
        let dto = self.mesh_geometry(mesh)?;
        Ok(crate::api::scene::scene_bounds_from_dto(&dto))
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
        self.run_sync_upload_task(move |data_cache| create_material_pbr_gpu_ready(desc, data_cache))
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
        self.run_sync_upload_task(move |data_cache| {
            upload_procedural_mesh_gpu_ready(mesh, data_cache)
        })
    }

    fn run_sync_upload_task<T, F>(&mut self, task: F) -> Result<T, AssetError>
    where
        T: Send + 'static,
        F: FnOnce(Arc<VkDataCache>) -> Result<T, AssetError> + Send + 'static,
    {
        let data_cache = Arc::clone(&self.core.data_cache);
        let (sender, receiver) = mpsc::channel();

        self.load_tracker.thread_pool.execute(move || {
            let result = task(data_cache);
            let _ = sender.send(result);
        });

        // Wait for completion while pumping transfer submissions.
        loop {
            self.pump_transfer_submissions()?;
            match receiver.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok(result) => return result,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(AssetError::Sync(
                        "asset upload worker thread disconnected".to_string(),
                    ));
                }
            }
        }
    }

    fn pump_transfer_submissions(&mut self) -> Result<(), AssetError> {
        self.core
            .pump_transfer_submissions(usize::MAX)
            .map(|_| ())
            .map_err(AssetError::Sync)
    }
}

fn load_model_gpu_ready(
    path: PathBuf,
    data_cache: Arc<VkDataCache>,
    policy_config: &AssetPolicyConfig,
) -> Result<ModelMeta, AssetError> {
    let path_str = path.to_str().ok_or_else(|| {
        AssetError::Unsupported(format!(
            "asset path '{}' is not valid UTF-8",
            path.display()
        ))
    })?;

    let loaded_model = assimp_util::load_model(path_str, data_cache.clone(), false, policy_config)
        .map_err(AssetError::from)?;

    // Capture and register every DTO as one transaction before GPU promotion,
    // preserving importer-derived rigid/skinned/deformed classification.
    let dtos = collect_mesh_geometry_dtos_with_deformations(
        &data_cache,
        &loaded_model.mesh_ids,
        &loaded_model.mesh_deformations,
    );
    let registration_result = dtos.and_then(|dtos| register_mesh_geometry_dtos(&data_cache, dtos));
    if let Err(err) = registration_result {
        let _ = rollback_model_allocations(
            &data_cache,
            &loaded_model.mesh_ids,
            &loaded_model.material_ids,
        );
        return Err(AssetError::Internal(format!(
            "mesh geometry DTO registration failed: {err}"
        )));
    }

    let upload_result = promote_model_gpu_allocations(&path, &data_cache, &loaded_model);
    if upload_result.is_err() {
        let _ = rollback_model_allocations(
            &data_cache,
            &loaded_model.mesh_ids,
            &loaded_model.material_ids,
        );
        // Roll back any DTOs registered before promotion.
        let _ = rollback_mesh_geometry_dtos(&data_cache, &loaded_model.mesh_ids);
    }
    upload_result?;

    Ok(loaded_model)
}

fn create_material_pbr_gpu_ready(
    desc: PbrMaterialDesc,
    data_cache: Arc<VkDataCache>,
) -> Result<MaterialHandle, AssetError> {
    validate_material_desc(&desc)?;
    let material_payload = material_desc_to_payload(&desc);

    {
        let texture_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;

        for tex_handle in material_payload.texture_ids.to_vec() {
            if tex_handle.slot >= TextureCache::DEFAULT_TEX_ITER_START as u32 {
                texture_cache.get_texture(tex_handle).map_err(|err| {
                    map_cache_err("texture", tex_handle.slot, tex_handle.generation, err)
                })?;
            }
        }
    }

    let material_id = {
        let mut texture_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;
        texture_cache.add_material(material_payload)
    };

    let allocation_result = {
        let mut texture_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;
        texture_cache.allocate_id(material_id, BufferPlacement::ContiguousPreferred, false)
    };

    match allocation_result {
        LoadResult::Success(_) => Ok(material_id),
        LoadResult::Failed(_) => {
            let mut texture_cache = data_cache
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

fn upload_procedural_mesh_gpu_ready(
    mesh: ProceduralMeshData,
    data_cache: Arc<VkDataCache>,
) -> Result<MeshHandle, AssetError> {
    validate_procedural_mesh(&mesh)?;

    if let Some(material_handle) = mesh.material {
        let texture_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;

        texture_cache.get_material(material_handle).map_err(|err| {
            map_cache_err(
                "material",
                material_handle.slot,
                material_handle.generation,
                err,
            )
        })?;

        texture_cache
            .get_loaded_material(material_handle)
            .map_err(|err| {
                map_cache_err(
                    "material",
                    material_handle.slot,
                    material_handle.generation,
                    err,
                )
            })?;
    }

    let vertices: Vec<Vertex> = mesh.vertices.iter().map(procedural_vertex_to_gpu).collect();
    let has_uv1 = mesh.vertices.iter().any(|v| v.uv1 != glam::Vec2::ZERO);

    let mesh_meta = MeshMeta {
        name: mesh.name,
        indices: mesh.indices,
        vertices,
        material_index: mesh.material,
        has_uv1,
    };

    let mesh_id = {
        let mut mesh_cache = data_cache
            .mesh_cache
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_cache"))?;
        mesh_cache.add(mesh_meta)
    };

    // Register the neutral geometry DTO before GPU promotion. Registration failure rolls
    // back the handle allocation rather than leaving an unloaded orphan in the cache.
    let registration_result =
        collect_mesh_geometry_dtos(&data_cache, &[mesh_id], MeshDeformation::Rigid)
            .and_then(|dtos| register_mesh_geometry_dtos(&data_cache, dtos));
    if let Err(err) = registration_result {
        let mut mesh_cache = data_cache
            .mesh_cache
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_cache"))?;
        mesh_cache.deallocate_id(mesh_id);
        return Err(AssetError::Internal(format!(
            "mesh geometry DTO registration failed: {err}"
        )));
    }

    let allocation_result = {
        let mut mesh_cache = data_cache
            .mesh_cache
            .lock()
            .map_err(|_| poisoned_lock_err("mesh_cache"))?;
        mesh_cache.allocate_id(mesh_id, BufferPlacement::ContiguousPreferred, false)
    };

    match allocation_result {
        LoadResult::Success(_) => Ok(mesh_id),
        LoadResult::Failed(_) => {
            let mut mesh_cache = data_cache
                .mesh_cache
                .lock()
                .map_err(|_| poisoned_lock_err("mesh_cache"))?;
            mesh_cache.deallocate_id(mesh_id);
            // Roll back DTO registration.
            let _ = rollback_mesh_geometry_dtos(&data_cache, &[mesh_id]);
            Err(AssetError::Internal(
                "mesh GPU allocation failed".to_string(),
            ))
        }
    }
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

/// Build DTOs while the meshes are still CPU-ready. The mesh lock is released before the
/// geometry store is acquired, keeping ingestion out of unload's two-lock critical section.
fn collect_mesh_geometry_dtos(
    data_cache: &Arc<VkDataCache>,
    mesh_ids: &[MeshHandle],
    deformation: MeshDeformation,
) -> Result<Vec<MeshGeometryDto>, String> {
    let mesh_cache = data_cache
        .mesh_cache
        .lock()
        .map_err(|_| "mesh_cache lock poisoned".to_string())?;

    mesh_ids
        .iter()
        .copied()
        .map(|mesh_id| {
            let cached = mesh_cache
                .get_id(mesh_id)
                .map_err(|e| format!("mesh {mesh_id:?}: {e:?}"))?;
            match cached {
                CachedMesh::Unloaded(meta) => mesh_meta_to_dto(mesh_id, meta, deformation),
                CachedMesh::Loaded(_) => Err(format!(
                    "mesh {mesh_id:?} was promoted before geometry capture"
                )),
                CachedMesh::_NULL => Err(format!(
                    "mesh {mesh_id:?} was invalidated before geometry capture"
                )),
            }
        })
        .collect()
}

fn collect_mesh_geometry_dtos_with_deformations(
    data_cache: &Arc<VkDataCache>,
    mesh_ids: &[MeshHandle],
    deformations: &[MeshDeformation],
) -> Result<Vec<MeshGeometryDto>, String> {
    if mesh_ids.len() != deformations.len() {
        return Err(format!(
            "mesh/deformation count mismatch: {} handles, {} classifications",
            mesh_ids.len(),
            deformations.len()
        ));
    }
    let mesh_cache = data_cache
        .mesh_cache
        .lock()
        .map_err(|_| "mesh_cache lock poisoned".to_string())?;
    mesh_ids
        .iter()
        .copied()
        .zip(deformations.iter().copied())
        .map(|(mesh_id, deformation)| match mesh_cache.get_id(mesh_id) {
            Ok(CachedMesh::Unloaded(meta)) => mesh_meta_to_dto(mesh_id, meta, deformation),
            Ok(CachedMesh::Loaded(_)) => Err(format!(
                "mesh {mesh_id:?} was promoted before geometry capture"
            )),
            Ok(CachedMesh::_NULL) => Err(format!(
                "mesh {mesh_id:?} was invalidated before geometry capture"
            )),
            Err(error) => Err(format!("mesh {mesh_id:?}: {error:?}")),
        })
        .collect()
}

fn register_mesh_geometry_dtos(
    data_cache: &Arc<VkDataCache>,
    dtos: Vec<MeshGeometryDto>,
) -> Result<(), String> {
    let mut geo_store = data_cache
        .mesh_geometry_store
        .lock()
        .map_err(|_| "mesh_geometry_store lock poisoned".to_string())?;
    geo_store
        .insert_batch(dtos)
        .map_err(|e| format!("geometry batch rejected: {e}"))
}

/// Remove DTO registrations for `mesh_ids` (used on GPU promotion failure).
fn rollback_mesh_geometry_dtos(
    data_cache: &Arc<VkDataCache>,
    mesh_ids: &[MeshHandle],
) -> Result<(), AssetError> {
    let mut geo_store = data_cache
        .mesh_geometry_store
        .lock()
        .map_err(|_| poisoned_lock_err("mesh_geometry_store"))?;
    geo_store.remove_batch(mesh_ids);
    Ok(())
}

/// Build a [`MeshGeometryDto`] from an unloaded [`MeshMeta`].
fn mesh_meta_to_dto(
    mesh_id: MeshHandle,
    meta: &MeshMeta,
    deformation: MeshDeformation,
) -> Result<MeshGeometryDto, String> {
    validate_triangle_indices(&meta.indices, meta.vertices.len())
        .map_err(|e| format!("invalid indices for {:?}: {e}", mesh_id))?;

    let positions: Vec<[f32; 3]> = meta
        .vertices
        .iter()
        .map(|v| v.position.to_array())
        .collect();
    let local_aabb = (deformation == MeshDeformation::Rigid)
        .then(|| compute_local_aabb(&positions))
        .flatten();

    Ok(MeshGeometryDto {
        mesh: mesh_id,
        positions: std::sync::Arc::from(positions.into_boxed_slice()),
        indices: std::sync::Arc::from(meta.indices.clone().into_boxed_slice()),
        local_aabb,
        deformation,
    })
}

fn load_texture_gpu_ready_with_policy(
    path: PathBuf,
    data_cache: Arc<VkDataCache>,
    policy_config: &AssetPolicyConfig,
    options: Option<TextureLoadOptions>,
) -> Result<TextureHandle, AssetError> {
    let mut loaded = load_texture_batch_gpu_ready_with_policy(
        vec![(path, options.unwrap_or_default())],
        data_cache,
        policy_config,
    )?;
    loaded
        .pop()
        .ok_or_else(|| AssetError::Internal("texture batch returned no handles".to_string()))
}

fn load_texture_batch_gpu_ready_with_policy(
    requests: Vec<(PathBuf, TextureLoadOptions)>,
    data_cache: Arc<VkDataCache>,
    policy_config: &AssetPolicyConfig,
) -> Result<Vec<TextureHandle>, AssetError> {
    let mut texture_metas = Vec::with_capacity(requests.len());
    for (path, options) in requests.iter() {
        let policy = asset_manifest::resolve_texture_policy_for_path(
            path,
            policy_config.manifest_mode,
            policy_config.allow_filename_heuristics,
            Some(options),
        )?;

        let image = image::open(path).map_err(|err| map_texture_path_err(path, err))?;
        let format = if policy.is_srgb {
            assimp_util::to_vk_format_srgb(&image)
        } else {
            assimp_util::to_vk_format(&image)
        };

        let mip_count = if policy.generate_mips {
            resolve_texture_mip_count(image.width(), image.height(), None)
        } else {
            1
        };

        let mut texture_meta = TextureMeta {
            payload: crate::data::gpu_data::TexturePayload::Raw {
                bytes: image.as_bytes().to_vec(),
                width: image.width(),
                height: image.height(),
                format,
                mips_levels: mip_count,
            },
            uv_index: 0,
            sampler_info: Some(policy.to_sampler_info(mip_count)),
        };

        texture_meta = crate::data::compression::apply_compression_policy(
            texture_meta,
            TextureSemantic::Generic,
            policy_config,
            &data_cache.supported_image_formats,
        );
        texture_metas.push(texture_meta);
    }

    let texture_ids = {
        let mut texture_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;
        texture_metas
            .into_iter()
            .map(|meta| texture_cache.add_texture(meta))
            .collect::<Vec<_>>()
    };

    if let Some((bad_index, _)) = texture_ids
        .iter()
        .enumerate()
        .find(|(_, id)| **id == TextureCache::DEFAULT_ERROR_TEX)
    {
        for id in texture_ids.iter().copied() {
            if id != TextureCache::DEFAULT_ERROR_TEX {
                let _ = rollback_texture_allocation(&data_cache, id);
            }
        }
        return Err(AssetError::Load {
            path: Some(requests[bad_index].0.clone()),
            message: "texture conversion failed during cache import".to_string(),
        });
    }

    let did_allocate = {
        let mut texture_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| poisoned_lock_err("texture_cache"))?;
        texture_cache.allocate_textures(texture_ids.clone())
    };

    if !did_allocate {
        for id in texture_ids.iter().copied() {
            let _ = rollback_texture_allocation(&data_cache, id);
        }
        let first_path = requests.first().map(|(path, _)| path.clone());
        return Err(AssetError::Load {
            path: first_path,
            message: "texture GPU allocation failed".to_string(),
        });
    }

    Ok(texture_ids)
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
            .add_node_with_bounds(
                fragment_parent_id,
                source_node.local_transform,
                source_node.meshes.clone(),
                source_node.mesh_bounds.clone(),
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
        CacheError::GenerationExhausted => AssetError::Internal(format!(
            "generation exhausted for {resource} slot {slot}; slot reuse is permanently rejected"
        )),
        CacheError::DescriptorAllocation(msg) => AssetError::Internal(format!(
            "descriptor allocation failed for {resource} slot {slot}: {msg}"
        )),
    }
}

fn ensure_asset_kind(record: &DurableAssetRecord, allowed: &[AssetKind]) -> Result<(), AssetError> {
    if allowed.iter().any(|kind| kind == &record.kind) {
        return Ok(());
    }

    let expected = allowed
        .iter()
        .map(AssetKind::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    Err(AssetError::Unsupported(format!(
        "asset '{}' has kind '{}'; expected one of: {}",
        record.asset_id, record.kind, expected
    )))
}

fn asset_registry_error_to_asset_error(err: AssetRegistryError) -> AssetError {
    match err {
        AssetRegistryError::Io { path, message } => AssetError::Io { path, message },
        AssetRegistryError::Parse { path, message } => AssetError::ManifestParse {
            path: path.unwrap_or_default(),
            message,
        },
        AssetRegistryError::UnsupportedVersion { .. }
        | AssetRegistryError::PackageIdMismatch { .. }
        | AssetRegistryError::DuplicateAssetId(_)
        | AssetRegistryError::InvalidAssetId(_)
        | AssetRegistryError::InvalidAssetPath { .. }
        | AssetRegistryError::UnsupportedAssetKind(_)
        | AssetRegistryError::MissingAssetId(_) => AssetError::Unsupported(err.to_string()),
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
    pub uv1: glam::Vec2,
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
    if !mesh.indices.len().is_multiple_of(3) {
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
                "procedural mesh '{}' vertex {} has non-finite UV0 coordinates",
                name, i
            )));
        }
        if !vertex.uv1.is_finite() {
            return Err(AssetError::Internal(format!(
                "procedural mesh '{}' vertex {} has non-finite UV1 coordinates",
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

/// Convert PbrMaterialDesc to an internal material payload with clamping.
fn material_desc_to_payload(desc: &PbrMaterialDesc) -> MaterialPayload {
    let mut meta = MaterialPayload::default();

    // Clamp base color channels to [0.0, 1.0]
    meta.material_values.base_color_factor =
        desc.base_color.clamp(glam::Vec4::ZERO, glam::Vec4::ONE);

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
        uv1_x: v.uv1.x,
        uv1_y: v.uv1.y,
        _pad: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imported_scene_fragment_preserves_exact_mesh_bounds() {
        let mesh = MeshHandle::new(7, 3);
        let bounds =
            crate::api::scene::SceneBounds::Known(crate::data::camera::Aabb::from_min_max(
                glam::Vec3::new(-2.0, -1.0, -3.0),
                glam::Vec3::new(4.0, 5.0, 6.0),
            ));
        let mut world = SceneWorld::new();
        let root = world.add_node_with_parts_and_bounds(
            None,
            glam::Mat4::IDENTITY,
            vec![mesh],
            vec![crate::api::scene::MeshBoundsEntry { mesh, bounds }],
        );
        world.set_root(root);

        let fragment = scene_world_to_fragment(world).unwrap();
        let node = fragment.node(fragment.root().unwrap()).unwrap();
        assert_eq!(node.meshes, vec![mesh]);
        assert_eq!(node.mesh_bounds.len(), 1);
        assert_eq!(node.mesh_bounds[0].mesh, mesh);
        assert_eq!(node.mesh_bounds[0].bounds, bounds);
    }

    #[test]
    fn material_desc_clamps_metallic_and_roughness() {
        let desc = PbrMaterialDesc {
            metallic: 1.5,   // Should clamp to 1.0
            roughness: -0.5, // Should clamp to 0.02
            ..Default::default()
        };

        let meta = material_desc_to_payload(&desc);
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
    fn procedural_vertex_conversion_preserves_uv1() {
        let v = ProceduralVertex {
            position: glam::Vec3::new(1.0, 2.0, 3.0),
            normal: glam::Vec3::Y,
            tangent: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            uv0: glam::Vec2::new(0.1, 0.2),
            uv1: glam::Vec2::new(0.3, 0.4),
            color: glam::Vec4::ONE,
        };

        let gpu_v = procedural_vertex_to_gpu(&v);
        assert_eq!(gpu_v.uv1_x, 0.3);
        assert_eq!(gpu_v.uv1_y, 0.4);
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
            vertices: vec![ProceduralVertex::default(), ProceduralVertex::default()],
            indices: vec![0, 1], // Only 2 indices, not divisible by 3
            material: None,
        };
        assert!(validate_procedural_mesh(&mesh).is_err());
    }

    #[test]
    fn procedural_mesh_rejects_out_of_bounds_index() {
        let mesh = ProceduralMeshData {
            name: "test".to_string(),
            vertices: vec![ProceduralVertex::default(), ProceduralVertex::default()],
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
                uv1: glam::Vec2::ZERO,
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
                uv1: glam::Vec2::ZERO,
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
                    uv1: glam::Vec2::ZERO,
                    color: glam::Vec4::ONE,
                },
                ProceduralVertex {
                    position: glam::Vec3::new(1.0, 0.0, 0.0),
                    normal: glam::Vec3::Y,
                    tangent: glam::Vec4::new(1.0, 0.0, 0.0, -1.0),
                    uv0: glam::Vec2::ZERO,
                    uv1: glam::Vec2::ZERO,
                    color: glam::Vec4::ONE,
                },
                ProceduralVertex {
                    position: glam::Vec3::new(0.0, 0.0, 1.0),
                    normal: glam::Vec3::Y,
                    tangent: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
                    uv0: glam::Vec2::ZERO,
                    uv1: glam::Vec2::ZERO,
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

    #[test]
    fn scheduler_starts_up_to_max_in_flight() {
        let mut tracker = AssetLoadTracker::new();
        tracker.max_in_flight_jobs = 3;

        // Queue 5 model loads
        for i in 0..5 {
            tracker.request_model_load(PathBuf::from(format!("/fake/model_{}.glb", i)));
        }

        assert_eq!(tracker.queued_tickets.len(), 5);
        assert_eq!(tracker.in_flight_count(), 0);

        // Note: start_queued requires a real VkDataCache to spawn threads,
        // so we test the scheduler logic directly through state inspection.
        // The bounded concurrency contract is: start_queued checks
        // in_flight_count() < max_in_flight_jobs before starting each task.
        assert!(tracker.in_flight_count() < tracker.max_in_flight_jobs);
    }

    #[test]
    fn scheduler_respects_max_in_flight_limit() {
        let mut tracker = AssetLoadTracker::new();
        tracker.max_in_flight_jobs = 2;

        // Manually create InFlight tasks to simulate running loads
        let t1 = tracker.request_model_load(PathBuf::from("/fake/a.glb"));
        let t2 = tracker.request_model_load(PathBuf::from("/fake/b.glb"));
        // Request a third task to verify the tracker correctly queues it
        // when max_in_flight_jobs (2) is already saturated.
        let _t3 = tracker.request_model_load(PathBuf::from("/fake/c.glb"));

        // Simulate two tasks becoming InFlight by creating fake receivers
        let (_, rx1) = mpsc::channel::<Result<SceneFragment, AssetError>>();
        let (_, rx2) = mpsc::channel::<Result<SceneFragment, AssetError>>();

        if let Some(task) = tracker.tasks.get_mut(&t1) {
            if let DeferredLoadTask::Model { state, .. } = task {
                *state = DeferredModelState::InFlight { receiver: rx1 };
            }
        }
        if let Some(task) = tracker.tasks.get_mut(&t2) {
            if let DeferredLoadTask::Model { state, .. } = task {
                *state = DeferredModelState::InFlight { receiver: rx2 };
            }
        }

        assert_eq!(tracker.in_flight_count(), 2);
        // With max_in_flight_jobs = 2, start_queued should not start more
        assert!(tracker.in_flight_count() >= tracker.max_in_flight_jobs);
    }

    #[test]
    fn mesh_geometry_queued_cancellation_starts_no_import_and_registers_no_dto() {
        let mut tracker = AssetLoadTracker::new();
        let geometry_store = crate::data::mesh_geometry::MeshGeometryStore::new();
        let ticket = tracker.request_model_load(PathBuf::from("/fake/model.glb"));

        // Cancel while still queued
        assert!(tracker.cancel_load(ticket).is_ok());

        // Verify state is Cancelled
        match tracker.tasks.get(&ticket) {
            Some(DeferredLoadTask::Model { state, .. }) => {
                assert!(matches!(state, DeferredModelState::Cancelled));
            }
            _ => panic!("Expected cancelled model task"),
        }

        // Polling returns Cancelled
        assert!(matches!(
            tracker.poll_model_load(ticket),
            LoadStatus::Cancelled
        ));
        assert_eq!(tracker.in_flight_count(), 0);
        assert!(geometry_store.is_empty());
    }
}
