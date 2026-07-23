//! BSP transaction coordinator: two-step prepare/validate/commit with
//! generation-token guards, idempotent rollback, unload/reload/reimport,
//! and app bridge orchestration.
//!
//! The coordinator owns the integration transaction, the active mount state,
//! and the source-link lifecycle. It coordinates resources owned by the
//! renderer and app bridges, never creating GPU or physics objects directly.
//!
//! The caller is responsible for building a [`PreparedBspMount`] from the
//! staged extraction DTOs (using renderer APIs), then passing it to
//! [`commit_with_mount`].

use crate::bridge::{
    AppBridge, BehaviorEntityRecipe, BridgeAggregator, EntityCollisionRecipe, LightEntityRecipe,
    WorldCollisionRecipe,
};
use crate::cache::CacheIdentity;
use crate::error::{BridgePhase, BspRuntimeError};
use crate::generation::{BspGenerationCounter, BspGenerationToken};
use crate::source_link::{
    reconcile_overrides, BspOverrideLayer, BspSourceLink, BspSourceReference,
    OverrideReconciliation,
};

use bsp::extract::{EntityDescriptor, ExtractedBsp};
use renderer::api::bsp::PreparedBspMount;
use renderer::api::{PointLight, PointLightId, Scene};

/// Result of a prepare operation.
#[derive(Debug)]
pub struct PrepareResult {
    /// Generation token for this preparation.
    pub token: BspGenerationToken,
    /// Human-readable map identity for display.
    pub source_identity: String,
    /// Number of faces in the BSP.
    pub face_count: usize,
    /// Number of entities in the BSP.
    pub entity_count: usize,
    /// Number of light entities extracted.
    pub light_count: usize,
    /// Number of render batches produced.
    pub batch_count: usize,
    /// Has PVS data.
    pub has_pvs: bool,
    /// Whether the target was already occupied when prepare started.
    pub was_occupied: bool,
}

/// Result of a commit operation.
#[derive(Debug)]
pub struct CommitResult {
    /// Number of scene nodes created.
    pub node_count: usize,
    /// Number of point lights created.
    pub light_count: usize,
    /// Number of bridge participants committed.
    pub bridge_count: usize,
    /// Active cache identity after commit.
    pub cache_identity: CacheIdentity,
}

/// Result of a reload operation.
#[derive(Debug)]
pub struct ReloadResult {
    /// Prepare result for the new BSP.
    pub prepare: PrepareResult,
    /// Commit result for the new BSP.
    pub commit: CommitResult,
    /// Override reconciliation report (if previous overrides exist).
    pub reconciliation: Option<OverrideReconciliation>,
}

/// BSP transaction coordinator.
///
/// The coordinator implements the two-step prepare → validate → commit
/// transaction with generation-token guards, idempotent rollback, and
/// unload/reload/reimport semantics. It owns the active mount state and
/// orchestrates renderer and app bridge participants.
///
/// # Usage
///
/// ```ignore
/// let mut coordinator = BspCoordinator::new();
/// let prepare = coordinator.prepare(&bsp_bytes, None, "maps/e1m1")?;
/// coordinator.validate(prepare.token)?;
///
/// // Build PreparedBspMount from extraction using renderer APIs
/// let mount = build_mount_from_extraction(coordinator.staged_extraction());
///
/// let commit = coordinator.commit_with_mount(prepare.token, &mut scene, mount)?;
/// ```
pub struct BspCoordinator {
    /// Monotonic generation counter for serialize-and-stale detection.
    generation: BspGenerationCounter,

    /// Active extracted BSP (if a mount is active).
    active_extracted: Option<ExtractedBsp>,
    /// Active source link metadata.
    active_source_link: Option<BspSourceLink>,
    /// Active mount identity for cache separation.
    active_cache_identity: Option<CacheIdentity>,

    /// Staged DTOs from the current prepare (hidden until commit).
    staged_extracted: Option<ExtractedBsp>,
    /// Staged source link from prepare.
    staged_source_link: Option<BspSourceLink>,
    /// Staged cache identity from prepare.
    staged_cache_identity: Option<CacheIdentity>,
    /// Active point-light IDs created for the published BSP mount.
    active_lights: Vec<PointLightId>,

    /// Generation that has completed validation and is allowed to commit.
    validated_generation: Option<u64>,

    /// App bridge aggregator.
    bridges: BridgeAggregator,

    /// Whether the coordinator has been poisoned.
    poisoned: bool,
}

impl BspCoordinator {
    /// Create a new BSP coordinator with no active mount.
    pub fn new() -> Self {
        Self {
            generation: BspGenerationCounter::new(),
            active_extracted: None,
            active_source_link: None,
            active_cache_identity: None,
            staged_extracted: None,
            staged_source_link: None,
            staged_cache_identity: None,
            active_lights: Vec::new(),
            validated_generation: None,
            bridges: BridgeAggregator::new(),
            poisoned: false,
        }
    }

    // ── Query ──────────────────────────────────────────────────────────

    /// Returns true if a BSP mount is currently active.
    pub fn is_active(&self) -> bool {
        self.active_extracted.is_some()
    }

    /// Returns true if the coordinator is poisoned.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Returns the current generation value.
    pub fn current_generation(&self) -> u64 {
        self.generation.current()
    }

    /// Returns a reference to the active source link, if any.
    pub fn source_link(&self) -> Option<&BspSourceLink> {
        self.active_source_link.as_ref()
    }

    /// Returns a reference to the active cache identity, if any.
    pub fn cache_identity(&self) -> Option<&CacheIdentity> {
        self.active_cache_identity.as_ref()
    }

    /// Returns a reference to the staged extraction, if any.
    ///
    /// The caller uses this to build a [`PreparedBspMount`] before commit.
    pub fn staged_extraction(&self) -> Option<&ExtractedBsp> {
        self.staged_extracted.as_ref()
    }

    /// Returns staged entity descriptors.
    pub fn staged_entity_descriptors(&self) -> Option<&[EntityDescriptor]> {
        self.staged_extracted
            .as_ref()
            .map(|e| e.entity_descriptors.as_slice())
    }

    // ── Bridge Registration ───────────────────────────────────────────

    /// Register an app bridge for transaction participation.
    ///
    /// Bridges are called during prepare/validate/commit/rollback with
    /// no engine lock held. Registering a bridge does not activate it;
    /// it participates in the next prepare cycle.
    pub fn register_bridge(&mut self, name: impl Into<String>, bridge: Box<dyn AppBridge>) {
        self.bridges.register(name, bridge);
    }

    // ── Prepare ────────────────────────────────────────────────────────

    /// Prepare a BSP from raw bytes for subsequent commit.
    ///
    /// This is the first step of the two-step transaction. It:
    /// 1. Increments the generation counter (cancelling any in-flight prepare)
    /// 2. Parses and validates the BSP bytes
    /// 3. Extracts neutral DTOs (geometry, entities, lights)
    /// 4. Calls app bridge prepare hooks
    /// 5. Stages all extracted data for subsequent commit
    ///
    /// The prepared state is hidden; nothing is visible in the scene yet.
    /// After prepare, call [`validate`](BspCoordinator::validate) then
    /// [`commit_with_mount`](BspCoordinator::commit_with_mount).
    pub fn prepare(
        &mut self,
        bsp_bytes: &[u8],
        scale: Option<f32>,
        source_identity: impl Into<String>,
    ) -> Result<PrepareResult, BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        // Increment generation (cancels any previous in-flight prepare)
        let _gen = self
            .generation
            .increment()
            .ok_or(BspRuntimeError::GenerationExhausted)?;
        self.validated_generation = None;
        self.rollback_staged()?;
        log::info!(
            "BSP coordinator: generation {} — starting prepare",
            self.generation.current()
        );
        let token = self.generation.token();

        let source_identity = source_identity.into();

        // Parse BSP
        let load_options = bsp::LoadOptions {
            strict: false,
            palette: None,
            lit_data: None,
            wad_archives: Vec::new(),
            texture_overrides: Vec::new(),
            source_identity: source_identity.clone(),
        };

        let world = bsp::BspLoader::load(bsp_bytes, &load_options).map_err(|e| {
            BspRuntimeError::SourceUnavailable {
                reason: format!("BSP parse failed: {} (code {:?})", e.message, e.code),
            }
        })?;

        let was_occupied = self.is_active();

        // Extract DTOs
        let extracted = bsp::extract::extract(bsp::BspExtractionRequest {
            world,
            scale: scale.unwrap_or(0.0254),
            ..Default::default()
        }).map_err(|e| {
            BspRuntimeError::SourceUnavailable {
                reason: format!("BSP extraction failed: {} (code {:?})", e.message, e.code),
            }
        })?;

        let face_count = extracted.face_geometries.len();
        let entity_count = extracted.entity_descriptors.len();
        let light_count = extracted.light_descriptors.len();
        let batch_count = extracted.render_batches.len();
        let has_pvs = extracted.has_pvs;

        // Build bridge DTOs
        let world_collision = WorldCollisionRecipe {
            planes: extracted.world_collision_planes.clone(),
        };

        let collision_by_entity: std::collections::HashMap<u32, Vec<bsp::collision::CollisionRecipe>> =
            extracted
                .collision_recipes
                .iter()
                .cloned()
                .fold(std::collections::HashMap::new(), |mut map, recipe| {
                    map.entry(recipe.entity_index).or_default().push(recipe);
                    map
                });
        let entity_colliders: Vec<EntityCollisionRecipe> = extracted
            .inline_models
            .iter()
            .map(|im| EntityCollisionRecipe {
                entity_index: im.entity_index,
                classname: im.classname.clone(),
                origin: im.origin,
                is_trigger: im.classname.starts_with("trigger_"),
                recipes: collision_by_entity
                    .get(&im.entity_index)
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect();

        let light_recipes: Vec<LightEntityRecipe> = extracted
            .light_descriptors
            .iter()
            .map(|l| LightEntityRecipe {
                entity_index: l.entity_index,
                origin: l.origin,
                intensity: l.intensity,
                color: l.color,
                radius: l.radius,
                style: l.style.clone(),
            })
            .collect();

        let behavior_recipes: Vec<BehaviorEntityRecipe> = extracted
            .entity_descriptors
            .iter()
            .filter(|ed| ed.classname.starts_with("func_") || ed.classname.starts_with("trigger_"))
            .map(|ed| BehaviorEntityRecipe {
                entity_index: ed.entity_index,
                classname: ed.classname.clone(),
                origin: ed.origin.unwrap_or(glam::Vec3::ZERO),
                targetname: ed.targetname.clone(),
                target: ed.target.clone(),
            })
            .collect();

        // Call app bridge prepare
        if self.bridges.has_bridges() {
            if let Err(err) = self.bridges.prepare(
                &world_collision,
                &entity_colliders,
                &light_recipes,
                &behavior_recipes,
            ) {
                if matches!(err, BspRuntimeError::CoordinatorPoisoned) {
                    self.poisoned = true;
                }
                self.clear_staged_state();
                return Err(err);
            }
        }

        // Build source link from extraction metadata
        let content_hash_hex: String = extracted
            .content_hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        let source_ref = BspSourceReference {
            asset_id: source_identity.clone(),
            content_hash: format!("sha256:{}", content_hash_hex),
            compiler_provenance: None,
            import_settings: Some(crate::source_link::BspImportSettings {
                scale: scale.unwrap_or(0.0254),
                palette_hash: String::new(),
                texture_roots: vec![],
                light_calibration: Default::default(),
            }),
            entity_identity_map: extracted
                .entity_identities
                .iter()
                .filter_map(|id| {
                    if let bsp::identity::IdentitySource::TrenchbroomUuid(ref uuid) = id.source {
                        let ed = extracted.entity_descriptors.get(id.entity_index as usize);
                        Some(crate::source_link::EntityIdentityEntry {
                            uuid: uuid.clone(),
                            stable_handle: uuid.clone(),
                            classname: ed.map(|d| d.classname.clone()).unwrap_or_default(),
                            origin: ed
                                .and_then(|d| d.origin)
                                .map(|v| [v.x, v.y, v.z])
                                .unwrap_or([0.0; 3]),
                        })
                    } else {
                        None
                    }
                })
                .collect(),
        };

        let source_link = BspSourceLink::new(source_ref);

        let cache_identity = self.build_cache_identity(&extracted);

        // Stage everything
        self.staged_extracted = Some(extracted);
        self.staged_source_link = Some(source_link);
        self.staged_cache_identity = Some(cache_identity);

        Ok(PrepareResult {
            token,
            source_identity,
            face_count,
            entity_count,
            light_count,
            batch_count,
            has_pvs,
            was_occupied,
        })
    }

    // ── Validate ───────────────────────────────────────────────────────

    /// Validate the current staged preparation.
    ///
    /// Checks:
    /// 1. Generation token matches
    /// 2. Staged extraction is present
    /// 3. App bridges confirm readiness
    ///
    /// All-or-nothing: any failure → caller must call `rollback`.
    pub fn validate(&mut self, token: BspGenerationToken) -> Result<(), BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        // Check generation
        self.generation.validate(token)?;

        // Check staged extraction exists
        if self.staged_extracted.is_none() {
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Validate,
                message: "no staged preparation to validate".to_string(),
            });
        }

        // Validate app bridges
        if self.bridges.has_bridges() {
            self.bridges.validate()?;
        }

        self.validated_generation = Some(token.generation);
        Ok(())
    }

    // ── Commit ─────────────────────────────────────────────────────────

    /// Commit the staged preparation with a caller-built [`PreparedBspMount`].
    ///
    /// The caller is responsible for building the mount from the staged
    /// extraction (accessible via [`staged_extraction`]). This separates
    /// GPU upload concerns from transaction coordination.
    ///
    /// On success, the mount is published to the scene, lights are created,
    /// and app bridges are committed. On failure, the coordinator may be
    /// poisoned.
    pub fn commit_with_mount(
        &mut self,
        token: BspGenerationToken,
        scene: &mut Scene,
        mount: PreparedBspMount,
    ) -> Result<CommitResult, BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        // Check generation and enforce the explicit prepare → validate → commit sequence.
        self.generation.validate(token)?;
        if self.validated_generation != Some(token.generation) {
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Commit,
                message: "generation has not completed validation".to_string(),
            });
        }

        let extracted_ref = self.staged_extracted.as_ref().ok_or_else(|| {
            self.poisoned = true;
            BspRuntimeError::CoordinatorPoisoned
        })?;
        let source_link_ref = self.staged_source_link.as_ref().ok_or_else(|| {
            self.poisoned = true;
            BspRuntimeError::CoordinatorPoisoned
        })?;

        self.preflight_light_publication(extracted_ref, scene)?;
        let source_link_json = serde_json::to_value(source_link_ref).map_err(|e| {
            BspRuntimeError::SourceUnavailable {
                reason: format!("BSP source-link serialization failed: {e}"),
            }
        })?;

        let bridge_count = if self.bridges.has_bridges() {
            match self.bridges.commit() {
                Ok(()) => self.bridges.len(),
                Err(BspRuntimeError::CoordinatorPoisoned) => {
                    self.poisoned = true;
                    return Err(BspRuntimeError::CoordinatorPoisoned);
                }
                Err(err) => {
                    self.poisoned = true;
                    return Err(err);
                }
            }
        } else {
            0
        };

        let extracted = self.staged_extracted.take().ok_or_else(|| {
            self.poisoned = true;
            BspRuntimeError::CoordinatorPoisoned
        })?;
        let source_link = self.staged_source_link.take().ok_or_else(|| {
            self.staged_extracted = Some(extracted.clone());
            self.poisoned = true;
            BspRuntimeError::CoordinatorPoisoned
        })?;
        let cache_identity = self
            .staged_cache_identity
            .take()
            .unwrap_or_else(|| self.build_cache_identity(&extracted));

        if self.is_active() {
            self.unload_active_from_scene(scene);
        }

        scene.set_bsp_mount(mount);
        scene.set_bsp_source_link(source_link_json);

        let mut new_light_ids = Vec::with_capacity(extracted.light_descriptors.len());
        for light_desc in &extracted.light_descriptors {
            let light = PointLight {
                position: light_desc.origin,
                color: glam::Vec3::from_array(light_desc.color),
                intensity: light_desc.intensity,
                range: light_desc.radius.max(1.0),
            };
            match scene.create_point_light(light) {
                Ok(id) => new_light_ids.push(id),
                Err(e) => {
                    for id in new_light_ids.drain(..) {
                        let _ = scene.remove_point_light(id);
                    }
                    scene.clear_bsp_mount();
                    scene.clear_bsp_source_link();
                    self.poisoned = true;
                    return Err(BspRuntimeError::SourceUnavailable {
                        reason: format!(
                            "BSP light publication failed after preflight for entity {}: {:?}",
                            light_desc.entity_index, e
                        ),
                    });
                }
            }
        }

        let node_count = 0;
        let light_count = new_light_ids.len();
        self.active_lights = new_light_ids;
        self.active_extracted = Some(extracted);
        self.active_source_link = Some(source_link);
        self.active_cache_identity = Some(cache_identity.clone());
        self.validated_generation = None;

        Ok(CommitResult {
            node_count,
            light_count,
            bridge_count,
            cache_identity,
        })
    }

    // ── Rollback ───────────────────────────────────────────────────────

    /// Roll back the current staged or active preparation.
    ///
    /// Idempotent: can be called multiple times. Removes staged resources,
    /// calls app bridge rollback hooks, and returns the coordinator to a
    /// clean pre-prepare state.
    pub fn rollback(&mut self) -> Result<(), BspRuntimeError> {
        self.rollback_staged()
    }

    // ── Unload ─────────────────────────────────────────────────────────

    /// Unload the active BSP mount, removing all associated resources from
    /// the scene.
    ///
    /// 1. Increments generation (cancels any in-flight prepare)
    /// 2. Removes BSP scene nodes, lights, materials
    /// 3. Calls app bridge rollback hooks
    /// 4. Clears coordinator state
    pub fn unload(&mut self, scene: &mut Scene) -> Result<(), BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        // Cancel any in-flight prepare
        self.generation
            .increment()
            .ok_or(BspRuntimeError::GenerationExhausted)?;
        self.validated_generation = None;

        // Roll back any staged state
        self.rollback_staged()?;

        // Remove active mount from scene
        self.unload_active_from_scene(scene);

        // Clear active state
        self.active_extracted = None;
        self.active_source_link = None;
        self.active_cache_identity = None;

        Ok(())
    }

    // ── Reload ─────────────────────────────────────────────────────────

    /// Reload the BSP from the same source bytes.
    ///
    /// Equivalent to `unload` followed by `prepare` → `validate` → `commit`
    /// with override reconciliation. The caller must provide a function to
    /// build the PreparedBspMount from the staged extraction.
    pub fn reload(
        &mut self,
        bsp_bytes: &[u8],
        scale: Option<f32>,
        source_identity: impl Into<String>,
        scene: &mut Scene,
        build_mount: impl FnOnce(&ExtractedBsp) -> PreparedBspMount,
    ) -> Result<ReloadResult, BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        // Capture previous overrides for reconciliation
        let previous_overrides = self
            .active_source_link
            .as_ref()
            .map(|link| link.bsp_overrides.clone())
            .unwrap_or_default();

        // Prepare new
        let prepare = self.prepare(bsp_bytes, scale, source_identity)?;

        // Validate
        self.validate(prepare.token)?;

        // Build mount from extraction
        let extracted =
            self.staged_extracted
                .as_ref()
                .ok_or_else(|| BspRuntimeError::BridgeFailure {
                    bridge_name: "coordinator".to_string(),
                    phase: BridgePhase::Commit,
                    message: "staged extraction missing after validate".to_string(),
                })?;
        let mount = build_mount(extracted);

        // Commit (this unloads the old mount)
        let commit = self.commit_with_mount(prepare.token, scene, mount)?;

        // Reconcile overrides if we had previous overrides
        let reconciliation = if !previous_overrides.entity_overrides.is_empty()
            || !previous_overrides.light_overrides.is_empty()
        {
            if let Some(ref active_extracted) = self.active_extracted {
                let (report, reconciled) = reconcile_overrides(
                    &previous_overrides,
                    &active_extracted.entity_identities,
                    &active_extracted.entity_descriptors,
                );

                if let Some(ref mut link) = self.active_source_link {
                    link.bsp_overrides = reconciled;
                    let source_link_json = serde_json::to_value(link).map_err(|e| {
                        BspRuntimeError::SourceUnavailable {
                            reason: format!("BSP source-link serialization failed: {e}"),
                        }
                    })?;
                    scene.set_bsp_source_link(source_link_json);
                }

                Some(report)
            } else {
                None
            }
        } else {
            None
        };

        Ok(ReloadResult {
            prepare,
            commit,
            reconciliation,
        })
    }

    // ── Reimport ───────────────────────────────────────────────────────

    /// Reimport a BSP from different source bytes (same logical map,
    /// different compilation).
    ///
    /// Prepares a new mount (hidden), computes source-link reconciliation,
    /// then atomically swaps old → new on commit.
    pub fn reimport(
        &mut self,
        bsp_bytes: &[u8],
        scale: Option<f32>,
        source_identity: impl Into<String>,
        scene: &mut Scene,
        build_mount: impl FnOnce(&ExtractedBsp) -> PreparedBspMount,
    ) -> Result<(ReloadResult, OverrideReconciliation), BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        let source_identity = source_identity.into();

        // Prepare new
        let prepare = self.prepare(bsp_bytes, scale, source_identity.clone())?;

        // Validate
        self.validate(prepare.token)?;

        // Capture previous overrides for reconciliation
        let previous_overrides = self
            .active_source_link
            .as_ref()
            .map(|link| link.bsp_overrides.clone())
            .unwrap_or_default();

        // Reconcile overrides against staged extraction
        let (reconciliation, reconciled) = if let Some(ref extracted) = self.staged_extracted {
            reconcile_overrides(
                &previous_overrides,
                &extracted.entity_identities,
                &extracted.entity_descriptors,
            )
        } else {
            (OverrideReconciliation::new(), BspOverrideLayer::default())
        };

        // Update staged source link with reconciled overrides
        if let Some(ref mut link) = self.staged_source_link {
            link.bsp_overrides = reconciled;
        }

        // Build mount from extraction
        let extracted =
            self.staged_extracted
                .as_ref()
                .ok_or_else(|| BspRuntimeError::BridgeFailure {
                    bridge_name: "coordinator".to_string(),
                    phase: BridgePhase::Commit,
                    message: "staged extraction missing after validate".to_string(),
                })?;
        let mount = build_mount(extracted);

        // Commit (atomic swap)
        let commit = self.commit_with_mount(prepare.token, scene, mount)?;

        Ok((
            ReloadResult {
                prepare,
                commit,
                reconciliation: Some(reconciliation.clone()),
            },
            reconciliation,
        ))
    }

    // ── Internal Helpers ───────────────────────────────────────────────

    /// Build a cache identity from extracted DTOs.
    fn build_cache_identity(&self, extracted: &ExtractedBsp) -> CacheIdentity {
        CacheIdentity::compute(
            extracted.content_hash,
            extracted.profile_tag,
            0.0254,  // default scale
            [0; 32], // palette not tracked yet
            vec![],
            vec![],
            vec![],
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }

    fn preflight_light_publication(
        &self,
        extracted: &ExtractedBsp,
        scene: &Scene,
    ) -> Result<(), BspRuntimeError> {
        let replacing_active_lights = if self.is_active() {
            self.active_lights.len()
        } else {
            0
        };
        let available_slots = scene
            .available_point_light_slots()
            .saturating_add(replacing_active_lights);
        if extracted.light_descriptors.len() > available_slots {
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Validate,
                message: format!(
                    "BSP point-light publication would exceed capacity: need {}, available {}",
                    extracted.light_descriptors.len(),
                    available_slots
                ),
            });
        }

        for light in &extracted.light_descriptors {
            let color = glam::Vec3::from_array(light.color);
            if !light.origin.is_finite()
                || !color.is_finite()
                || !light.intensity.is_finite()
                || light.intensity < 0.0
                || !light.radius.is_finite()
            {
                return Err(BspRuntimeError::BridgeFailure {
                    bridge_name: "coordinator".to_string(),
                    phase: BridgePhase::Validate,
                    message: format!(
                        "BSP light descriptor for entity {} is not publishable",
                        light.entity_index
                    ),
                });
            }
        }
        Ok(())
    }

    fn rollback_staged(&mut self) -> Result<(), BspRuntimeError> {
        let failures = self.bridges.rollback();
        self.clear_staged_state();
        if !failures.is_empty() {
            self.poisoned = true;
            return Err(BspRuntimeError::RollbackFailure { failures });
        }
        Ok(())
    }

    fn clear_staged_state(&mut self) {
        self.staged_extracted = None;
        self.staged_source_link = None;
        self.staged_cache_identity = None;
        self.validated_generation = None;
    }

    /// Remove the active BSP mount from a scene.
    fn unload_active_from_scene(&mut self, scene: &mut Scene) {
        if self.is_active() {
            scene.clear_bsp_mount();
            scene.clear_bsp_source_link();

            for light_id in self.active_lights.drain(..) {
                if let Err(e) = scene.remove_point_light(light_id) {
                    log::warn!("failed to remove BSP point light during unload: {:?}", e);
                }
            }
        }
    }
}

impl Default for BspCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
