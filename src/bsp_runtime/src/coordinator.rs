//! BSP transaction coordinator: two-step prepare/validate/commit with
//! generation-token guards, idempotent rollback, unload/reload/reimport,
//! and app bridge orchestration.
//!
//! The coordinator owns the integration transaction, the active mount state,
//! the source-link lifecycle, and the current preparation candidate. It
//! coordinates resources owned by the renderer and app bridges, never
//! creating GPU or physics objects directly.
//!
//! # Architecture (Phase 05)
//!
//! A [`BspCandidate`] holds all staged state for one generation. The
//! coordinator holds at most one candidate at a time. A new prepare
//! atomically replaces the previous candidate (cancellation).
//!
//! The commit step is a pure publish: it moves the candidate's ready
//! resources into the active world without performing any new parsing,
//! package resolution, external asset loading, allocation, upload,
//! lookup, serialization, bridge validation, restored-state validation,
//! or app-world capacity reservation. All of those are done before commit.

use crate::bridge::{
    AppBridge, BehaviorEntityRecipe, BridgeAggregator, EntityCollisionRecipe, LightEntityRecipe,
    WorldCollisionRecipe,
};
use crate::cache::{compute_identity_hash, CacheIdentity, CompanionId};
use crate::candidate::{BspCandidate, CandidatePointLight};
use crate::error::{BridgePhase, BspRuntimeError};
use crate::generation::{BspGenerationCounter, BspGenerationToken};
use crate::source_link::{reconcile_overrides, BspSourceLink, OverrideReconciliation};

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
/// unload/reload/reimport semantics. It owns the active mount state,
/// the current preparation candidate, and the source-link lifecycle.
///
/// # Usage (Phase 05)
///
/// ```ignore
/// let mut coordinator = BspCoordinator::new();
///
/// // Step 1: Prepare from raw bytes (or use prepare_from_package)
/// let prepare = coordinator.prepare(&bsp_bytes, None, "maps/e1m1")?;
///
/// // Step 2: Upload renderer resources
/// let mount = renderer.prepare_bsp_mount(coordinator.staged_extraction().unwrap())?;
/// coordinator.set_renderer_mount_ready(prepare.token, mount)?;
///
/// // Step 3: Validate
/// coordinator.validate(prepare.token)?;
///
/// // Step 4: Commit (pure publish — no new work)
/// let commit = coordinator.commit(prepare.token, &mut scene)?;
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
    /// Active point-light IDs created for the published BSP mount.
    active_lights: Vec<PointLightId>,

    /// Current preparation candidate (hidden until commit).
    candidate: Option<BspCandidate>,

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
            active_lights: Vec::new(),
            candidate: None,
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

    /// Returns a reference to the staged extraction from the current
    /// candidate, if any.
    ///
    /// The caller uses this to build a [`PreparedBspMount`] before commit.
    pub fn staged_extraction(&self) -> Option<&ExtractedBsp> {
        self.candidate.as_ref().map(|c| &c.extracted)
    }

    /// Returns staged entity descriptors from the current candidate.
    pub fn staged_entity_descriptors(&self) -> Option<&[EntityDescriptor]> {
        self.candidate
            .as_ref()
            .map(|c| c.extracted.entity_descriptors.as_slice())
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
    /// 1. Increments the generation counter (cancelling any previous candidate)
    /// 2. Parses and validates the BSP bytes
    /// 3. Extracts neutral DTOs (geometry, entities, lights)
    /// 4. Calls app bridge prepare hooks
    /// 5. Creates a [`BspCandidate`] holding all staged state
    ///
    /// The candidate state is hidden; nothing is visible in the scene yet.
    /// After prepare, call [`set_renderer_mount_ready`](BspCoordinator::set_renderer_mount_ready)
    /// then [`validate`](BspCoordinator::validate) then
    /// [`commit`](BspCoordinator::commit).
    pub fn prepare(
        &mut self,
        bsp_bytes: &[u8],
        scale: Option<f32>,
        source_identity: impl Into<String>,
    ) -> Result<PrepareResult, BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        // Increment generation (cancels any previous in-flight candidate)
        let _gen = self
            .generation
            .increment()
            .ok_or(BspRuntimeError::GenerationExhausted)?;
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

        self.build_candidate(world, Vec::new(), Vec::new(), scale, source_identity, token)
    }

    /// Prepare a package-loaded BSP and its auto-discovered PBR companions.
    pub fn prepare_from_loaded_package(
        &mut self,
        package: crate::package::LoadedBspPackage,
        scale: Option<f32>,
    ) -> Result<PrepareResult, BspRuntimeError> {
        let crate::package::LoadedBspPackage {
            world,
            bsp_resource,
            pbr_texture_resources,
            ..
        } = package;
        let source_identity = bsp_resource.id.as_str().to_string();
        drop(bsp_resource);
        let texture_companions = pbr_texture_resources
            .into_iter()
            .map(|resource| {
                let logical_path = resource.id.as_str().to_string();
                bsp::resources::TextureCompanion::new(logical_path, resource.bytes.into_bytes())
            })
            .collect();
        self.prepare_from_world_with_texture_companions(
            world,
            texture_companions,
            Vec::new(),
            scale,
            source_identity,
        )
    }

    /// Prepare a pre-parsed BSP world.
    ///
    /// Use [`prepare_from_world_with_texture_companions`](Self::prepare_from_world_with_texture_companions)
    /// when the caller has authorized external PBR texture bytes.
    pub fn prepare_from_world(
        &mut self,
        world: bsp::world::BspWorld,
        scale: Option<f32>,
        source_identity: impl Into<String>,
    ) -> Result<PrepareResult, BspRuntimeError> {
        self.prepare_from_world_with_texture_companions(
            world,
            Vec::new(),
            Vec::new(),
            scale,
            source_identity,
        )
    }

    /// Prepare a pre-parsed world with authorized external PBR texture companions.
    ///
    /// Companion bytes are matched during neutral extraction by the exact
    /// `<texture>_norm.png` / `<texture>_gloss.png` filename convention.
    /// `wad_archives` provide raw WAD file bytes for BSP texture resolution.
    pub fn prepare_from_world_with_texture_companions(
        &mut self,
        world: bsp::world::BspWorld,
        texture_companions: Vec<bsp::resources::TextureCompanion>,
        wad_archives: Vec<(String, Vec<u8>)>,
        scale: Option<f32>,
        source_identity: impl Into<String>,
    ) -> Result<PrepareResult, BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        let _gen = self
            .generation
            .increment()
            .ok_or(BspRuntimeError::GenerationExhausted)?;
        self.rollback_staged()?;
        log::info!(
            "BSP coordinator: generation {} — starting prepare from world",
            self.generation.current()
        );
        let token = self.generation.token();

        self.build_candidate(
            world,
            texture_companions,
            wad_archives,
            scale,
            source_identity.into(),
            token,
        )
    }

    /// Common candidate construction from a parsed BspWorld.
    fn build_candidate(
        &mut self,
        world: bsp::world::BspWorld,
        texture_companions: Vec<bsp::resources::TextureCompanion>,
        wad_archives: Vec<(String, Vec<u8>)>,
        scale: Option<f32>,
        source_identity: String,
        token: BspGenerationToken,
    ) -> Result<PrepareResult, BspRuntimeError> {
        let was_occupied = self.is_active();

        let resolved_scale = scale.unwrap_or(0.0254);

        // Extract DTOs
        let extracted = bsp::extract::extract(bsp::BspExtractionRequest {
            world,
            texture_companions,
            wad_archives,
            scale: resolved_scale,
            ..Default::default()
        })
        .map_err(|e| BspRuntimeError::SourceUnavailable {
            reason: format!("BSP extraction failed: {} (code {:?})", e.message, e.code),
        })?;

        let face_count = extracted.face_geometries.len();
        let entity_count = extracted.entity_descriptors.len();
        let light_count = extracted.light_descriptors.len();
        let batch_count = extracted.render_batches.len();
        let has_pvs = extracted.has_pvs;

        // Build all fallible coordinator-owned candidate payloads before
        // bridge prepare so later local validation cannot leak bridge resources.
        let source_link = self.build_source_link(&extracted, &source_identity, resolved_scale);
        let envelope = crate::source_link::BspPersistenceEnvelope::new(source_link.clone());
        let source_link_json =
            serde_json::to_value(&envelope).map_err(|e| BspRuntimeError::SourceUnavailable {
                reason: format!("BSP source-link serialization failed: {e}"),
            })?;
        let cache_identity = self.build_cache_identity(&extracted);
        let point_lights = Self::build_candidate_point_lights(&extracted)?;

        // Build bridge DTOs
        let bridge_dtos = self.build_bridge_dtos(&extracted);

        // Call app bridge prepare
        let bridge_tokens = if self.bridges.has_bridges() {
            self.bridges.prepare_with_tokens(
                &bridge_dtos.world_collision,
                &bridge_dtos.entity_colliders,
                &bridge_dtos.lights,
                &bridge_dtos.behaviors,
            )?
        } else {
            Vec::new()
        };

        // Create candidate
        let candidate = BspCandidate::new(
            token.generation,
            source_identity.clone(),
            extracted,
            cache_identity,
            source_link,
            source_link_json,
            point_lights,
            bridge_tokens,
            was_occupied,
        );

        self.candidate = Some(candidate);

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

    // ── Renderer Mount Integration ─────────────────────────────────────

    /// Signal that the renderer upload has started for the current candidate.
    ///
    /// This transitions the candidate's renderer lease from `NotStarted` to
    /// `Pending`. Call after issuing an async upload.
    pub fn start_renderer_upload(
        &mut self,
        token: BspGenerationToken,
    ) -> Result<(), BspRuntimeError> {
        self.generation.validate(token)?;
        self.require_candidate()?;
        let candidate = self.candidate.as_mut().unwrap();
        candidate.start_renderer_upload()
    }

    /// Transition the candidate's renderer lease to [`Ready`](RendererLease::Ready).
    ///
    /// The caller provides the completed [`PreparedBspMount`] from the renderer.
    /// After this, the candidate is eligible for commit (once validated).
    pub fn set_renderer_mount_ready(
        &mut self,
        token: BspGenerationToken,
        mount: PreparedBspMount,
    ) -> Result<(), BspRuntimeError> {
        self.generation.validate(token)?;
        self.require_candidate()?;
        let candidate = self.candidate.as_mut().unwrap();
        candidate.set_renderer_ready(mount)
    }

    /// Mark the renderer upload as failed for the current candidate.
    pub fn fail_renderer_upload(
        &mut self,
        token: BspGenerationToken,
        reason: String,
    ) -> Result<(), BspRuntimeError> {
        self.generation.validate(token)?;
        self.require_candidate()?;
        self.candidate
            .as_mut()
            .unwrap()
            .fail_renderer_upload(reason);
        Ok(())
    }

    // ── Validate ───────────────────────────────────────────────────────

    /// Validate the current staged candidate.
    ///
    /// Checks:
    /// 1. Generation token matches
    /// 2. Candidate is present
    /// 3. App bridges confirm readiness
    ///
    /// Candidates with no scene point lights are fully publication-ready after
    /// this call. Candidates that publish lights must use
    /// [`validate_for_scene`](BspCoordinator::validate_for_scene) so scene
    /// capacity is checked before commit.
    pub fn validate(&mut self, token: BspGenerationToken) -> Result<(), BspRuntimeError> {
        self.validate_candidate(token, None)
    }

    /// Validate the current staged candidate against a target scene.
    ///
    /// This performs all fallible scene-publication checks (currently BSP
    /// point-light capacity) before commit. Use this path for any production
    /// commit; commit is then publication-only.
    pub fn validate_for_scene(
        &mut self,
        token: BspGenerationToken,
        scene: &mut Scene,
    ) -> Result<(), BspRuntimeError> {
        self.validate_candidate(token, Some(scene))
    }

    // ── Commit (Pure Publish) ──────────────────────────────────────────

    /// Commit the prepared candidate, atomically publishing it to the scene.
    ///
    /// **This is a pure publish operation.** It performs no parsing, package
    /// resolution, external asset loading, allocation, upload, lookup,
    /// serialization, bridge validation, restored-state validation, or
    /// app-world capacity reservation. All preparation work must be complete
    /// before calling commit.
    ///
    /// Requirements:
    /// - Candidate must be validated ([`validate`](BspCoordinator::validate))
    /// - Renderer mount must be ready ([`set_renderer_mount_ready`](BspCoordinator::set_renderer_mount_ready))
    /// - Generation token must match
    ///
    /// On success, the previous active mount is unloaded and its resources
    /// retired. The new mount, lights, bridge state, and source link become
    /// the active generation.
    ///
    /// # Panic Safety
    ///
    /// Bridge commit panics poison the coordinator. Renderer publication is
    /// non-fallible after validation.
    pub fn commit(
        &mut self,
        token: BspGenerationToken,
        scene: &mut Scene,
    ) -> Result<CommitResult, BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        // Check generation
        self.generation.validate(token)?;

        // Candidate must be fully validated before commit.
        let candidate = self.require_candidate_mut()?;
        if !candidate.is_validated() {
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Commit,
                message: "candidate has not completed validation".to_string(),
            });
        }
        if !candidate.is_publication_validated() {
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Commit,
                message: "candidate publication has not completed validation".to_string(),
            });
        }

        // Renderer mount must be ready
        if !candidate.is_renderer_ready() {
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Commit,
                message: "renderer mount not ready".to_string(),
            });
        }

        // Commit bridges (non-fallible activation)
        let bridge_count = if self.bridges.has_bridges() {
            match self
                .bridges
                .commit_candidate(self.candidate.as_mut().unwrap())
            {
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

        // Take ownership from the candidate
        let mut candidate = self.candidate.take().unwrap();
        let mount = match candidate.take_ready_mount() {
            Ok(m) => m,
            Err(e) => {
                self.poisoned = true;
                return Err(e);
            }
        };

        // Unload previous active mount if present
        if self.is_active() {
            self.unload_active_from_scene(scene);
        }

        // Publish to scene
        scene.set_bsp_mount(mount);
        scene.set_bsp_source_link(candidate.source_link_json);

        // Publish lights from prevalidated payloads.
        let point_lights = std::mem::take(&mut candidate.point_lights);
        let mut new_light_ids = std::mem::take(&mut self.active_lights);
        new_light_ids.clear();
        debug_assert!(new_light_ids.capacity() >= point_lights.len());
        for candidate_light in &point_lights {
            match scene.create_point_light(candidate_light.light) {
                Ok(id) => new_light_ids.push(id),
                Err(e) => {
                    // This is an invariant breach after validate_for_scene: leave no
                    // partially published new BSP state and poison the coordinator.
                    for id in new_light_ids.drain(..) {
                        let _ = scene.remove_point_light(id);
                    }
                    scene.clear_bsp_mount();
                    scene.clear_bsp_source_link();
                    self.poisoned = true;
                    return Err(BspRuntimeError::SourceUnavailable {
                        reason: format!(
                            "prevalidated BSP light publication failed for entity {}: {:?}",
                            candidate_light.entity_index, e
                        ),
                    });
                }
            }
        }

        let light_count = new_light_ids.len();
        self.active_lights = new_light_ids;
        self.active_extracted = Some(candidate.extracted);
        self.active_source_link = Some(candidate.source_link);
        self.active_cache_identity = Some(candidate.cache_identity);

        // Depleted the candidate; the bridge tokens were consumed during commit_candidate.
        Ok(CommitResult {
            node_count: 0,
            light_count,
            bridge_count,
            cache_identity: self.active_cache_identity.clone().unwrap(),
        })
    }

    /// Legacy compatibility wrapper. Equivalent to setting the mount ready
    /// then calling commit.
    #[doc(hidden)]
    pub fn commit_with_mount(
        &mut self,
        token: BspGenerationToken,
        scene: &mut Scene,
        mount: PreparedBspMount,
    ) -> Result<CommitResult, BspRuntimeError> {
        self.set_renderer_mount_ready(token, mount)?;
        self.commit(token, scene)
    }

    // ── Rollback ───────────────────────────────────────────────────────

    /// Roll back the current staged candidate or active preparation.
    ///
    /// Idempotent: can be called multiple times. Removes staged resources,
    /// calls app bridge rollback hooks, and returns the coordinator to a
    /// clean pre-prepare state. Does not affect the active published mount.
    ///
    /// Returns `CoordinatorPoisoned` if the coordinator is poisoned.
    pub fn rollback(&mut self) -> Result<(), BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }
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

        // Cancel any in-flight candidate
        self.generation
            .increment()
            .ok_or(BspRuntimeError::GenerationExhausted)?;

        // Roll back any staged candidate
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
    /// Prepares a new candidate beside the active world. The old world remains
    /// visible until the new candidate is fully prepared, validated, and
    /// committed. On failure, the old world is unchanged.
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

        let source_identity = source_identity.into();
        let previous_overrides = self
            .active_source_link
            .as_ref()
            .map(|link| link.overrides.clone())
            .unwrap_or_default();

        // Prepare new candidate (hidden, beside active world)
        let prepare = self.prepare(bsp_bytes, scale, source_identity.clone())?;

        // Build mount from extraction
        let extracted = self
            .candidate
            .as_ref()
            .map(|c| &c.extracted)
            .ok_or_else(|| BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Commit,
                message: "candidate missing after prepare".to_string(),
            })?;
        let mount = build_mount(extracted);

        // Set mount ready
        self.set_renderer_mount_ready(prepare.token, mount)?;

        // Validate all fallible publication checks before commit.
        self.validate_for_scene(prepare.token, scene)?;

        // Reconcile overrides against staged extraction
        let reconciliation = if !previous_overrides.entity_overrides.is_empty()
            || !previous_overrides.light_overrides.is_empty()
        {
            let candidate = self.candidate.as_ref().unwrap();
            let (report, reconciled) = reconcile_overrides(
                &previous_overrides,
                &candidate.extracted.entity_identities,
                &candidate.extracted.entity_descriptors,
            );
            // Update candidate's source link and pre-serialized scene payload.
            if let Some(ref mut c) = self.candidate {
                c.source_link.overrides = reconciled;
                let envelope =
                    crate::source_link::BspPersistenceEnvelope::new(c.source_link.clone());
                c.source_link_json = serde_json::to_value(&envelope).map_err(|e| {
                    BspRuntimeError::SourceUnavailable {
                        reason: format!("BSP source-link serialization failed: {e}"),
                    }
                })?;
            }
            Some(report)
        } else {
            None
        };

        // Commit (publishes new, retires old)
        let commit = self.commit(prepare.token, scene)?;

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
    /// Prepares a new candidate (hidden), computes source-link reconciliation,
    /// then atomically swaps old → new on commit. The old world is unchanged
    /// until the new candidate is committed.
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

        // Capture previous overrides for reconciliation
        let previous_overrides = self
            .active_source_link
            .as_ref()
            .map(|link| link.overrides.clone())
            .unwrap_or_default();

        // Reconcile overrides against candidate extraction
        let (reconciliation, reconciled) = {
            let candidate =
                self.candidate
                    .as_ref()
                    .ok_or_else(|| BspRuntimeError::BridgeFailure {
                        bridge_name: "coordinator".to_string(),
                        phase: BridgePhase::Commit,
                        message: "candidate missing after prepare".to_string(),
                    })?;
            reconcile_overrides(
                &previous_overrides,
                &candidate.extracted.entity_identities,
                &candidate.extracted.entity_descriptors,
            )
        };

        // Update candidate's source link with reconciled overrides and refresh
        // the pre-serialized scene payload before commit.
        if let Some(ref mut c) = self.candidate {
            c.source_link.overrides = reconciled;
            let envelope = crate::source_link::BspPersistenceEnvelope::new(c.source_link.clone());
            c.source_link_json = serde_json::to_value(&envelope).map_err(|e| {
                BspRuntimeError::SourceUnavailable {
                    reason: format!("BSP source-link serialization failed: {e}"),
                }
            })?;
        }

        // Build mount from extraction
        let extracted = self
            .candidate
            .as_ref()
            .map(|c| &c.extracted)
            .ok_or_else(|| BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Commit,
                message: "candidate missing after prepare".to_string(),
            })?;
        let mount = build_mount(extracted);

        // Set mount ready
        self.set_renderer_mount_ready(prepare.token, mount)?;

        // Validate all fallible publication checks before commit.
        self.validate_for_scene(prepare.token, scene)?;

        // Commit (atomic swap)
        let commit = self.commit(prepare.token, scene)?;

        Ok((
            ReloadResult {
                prepare,
                commit,
                reconciliation: Some(reconciliation.clone()),
            },
            reconciliation,
        ))
    }

    // ── Terminal Shutdown ──────────────────────────────────────────────

    /// Tear down all BSP resources for renderer shutdown or device loss.
    ///
    /// Releases the active mount and any staged candidate. After calling this,
    /// the coordinator is clean but still usable for future prepares.
    /// Callers must ensure the scene has already been cleared externally.
    pub fn teardown(&mut self, scene: &mut Scene) {
        // Release candidate first
        self.rollback_staged().ok();
        // Release active mount
        self.unload_active_from_scene(scene);
        self.active_extracted = None;
        self.active_source_link = None;
        self.active_cache_identity = None;
    }

    // ── Persistence: Save ───────────────────────────────────────────

    /// Capture the current source-link payload for persistence.
    ///
    /// Only source-linked reconstruction data is stored: identity records,
    /// overrides, mutable behavior state. GPU handles, descriptors,
    /// allocations, cache slots, transient generation handles, and
    /// generated geometry are NEVER included.
    pub fn capture_source_link(
        &self,
        mutable_behavior: crate::source_link::MutableBehaviorState,
    ) -> Option<crate::source_link::BspPersistenceEnvelope> {
        let source_link = self.active_source_link.as_ref()?;
        let mut link = source_link.clone();
        link.mutable_behavior = mutable_behavior;
        Some(crate::source_link::BspPersistenceEnvelope::new(link))
    }

    /// Capture mutable behavior state from the current active mount.
    ///
    /// This reads one immutable snapshot of the active behavior state.
    /// Only reconstruction data (door/button/platform pose+state, trigger
    /// activation, light-style table, timers/counters) is returned.
    pub fn capture_mutable_behavior(&self) -> crate::source_link::MutableBehaviorState {
        // Default: empty mutable behavior state.
        // The app bridge populates this from its live state machines.
        crate::source_link::MutableBehaviorState::default()
    }

    // ── Persistence: Restore ────────────────────────────────────────

    /// Restore a BSP mount from a persistence payload.
    ///
    /// Builds a hidden candidate from the same identity model, applies
    /// all validation checks, then commits. On failure, the candidate is
    /// cancelled and the active generation is proven unchanged.
    ///
    /// Restore order: resolve→parse→extract→upload→identity reconcile→
    /// mapping validation→mutable behavior validation→generation commit.
    pub fn restore_from_persistence(
        &mut self,
        envelope: &crate::source_link::BspPersistenceEnvelope,
        bsp_bytes: &[u8],
        scale: Option<f32>,
        scene: &mut Scene,
        build_mount: impl FnOnce(&ExtractedBsp) -> PreparedBspMount,
    ) -> Result<ReloadResult, BspRuntimeError> {
        // 1. Validate schema version
        envelope
            .validate_schema()
            .map_err(|_| BspRuntimeError::UnsupportedSchema {
                version: envelope.schema_version as u32,
                current: crate::source_link::SchemaVersion::CURRENT as u32,
            })?;

        let stored_link = &envelope.bsp_source;

        // 2. Validate no runtime handles in the stored payload
        stored_link.validate_no_runtime_handles().map_err(|e| {
            BspRuntimeError::InvalidMutableBehavior {
                detail: e.to_string(),
            }
        })?;

        // 3. Resolve/parse/extract hidden candidate from raw bytes.
        let source_identity = stored_link.asset_id.clone();
        let prepare = self.prepare(bsp_bytes, scale, source_identity.clone())?;

        // 4. Verify content hash matches the restored source before any publication.
        {
            let candidate =
                self.candidate
                    .as_ref()
                    .ok_or_else(|| BspRuntimeError::BridgeFailure {
                        bridge_name: "coordinator".into(),
                        phase: BridgePhase::Validate,
                        message: "candidate missing after prepare".into(),
                    })?;

            if stored_link.content_hash != candidate.source_link.content_hash {
                let err = BspRuntimeError::SourceMismatch {
                    expected: stored_link.content_hash.clone(),
                    actual: candidate.source_link.content_hash.clone(),
                };
                return self.cancel_restore_candidate(err);
            }
        }

        // 5. Upload/build renderer mount readiness while still hidden.
        let mount = {
            let candidate = self.candidate.as_ref().unwrap();
            build_mount(&candidate.extracted)
        };
        if let Err(err) = self.set_renderer_mount_ready(prepare.token, mount) {
            return self.cancel_restore_candidate(err);
        }

        // 6. Reconcile entity identities and fail ambiguous restores before commit.
        let (reconciliation, reconciled_overrides) = {
            let candidate = self.candidate.as_ref().unwrap();
            let previous_overrides = stored_link.overrides.clone();
            reconcile_overrides(
                &previous_overrides,
                &candidate.extracted.entity_identities,
                &candidate.extracted.entity_descriptors,
            )
        };
        if reconciliation.ambiguous > 0 {
            return self.cancel_restore_candidate(BspRuntimeError::IdentityAmbiguous {
                entity_count: reconciliation.ambiguous,
                context: "restore source-link overrides".to_string(),
            });
        }

        // 7. Validate companion/model mapping identities against the candidate.
        {
            let candidate = self.candidate.as_ref().unwrap();
            if let Err(err) =
                Self::validate_restore_source_link(stored_link, &candidate.source_link)
            {
                return self.cancel_restore_candidate(err);
            }
        }

        // 8. Validate mutable behavior state after readiness but before publication.
        if let Err(e) = stored_link.mutable_behavior.validate() {
            return self.cancel_restore_candidate(BspRuntimeError::InvalidMutableBehavior {
                detail: e.to_string(),
            });
        }

        // 9. Validate scene publication preflight.
        if let Err(err) = self.validate_for_scene(prepare.token, scene) {
            return Err(err);
        }

        // 10. Update candidate's source link with reconciled overrides + stored mutable behavior.
        if let Some(ref mut c) = self.candidate {
            c.source_link.overrides = reconciled_overrides;
            c.source_link.mutable_behavior = stored_link.mutable_behavior.clone();
            match serde_json::to_value(crate::source_link::BspPersistenceEnvelope::new(
                c.source_link.clone(),
            )) {
                Ok(json) => c.source_link_json = json,
                Err(e) => {
                    return self.cancel_restore_candidate(BspRuntimeError::SourceUnavailable {
                        reason: format!("source-link serialization failed: {e}"),
                    });
                }
            }
        }

        // 11. Commit (pure publish)
        let commit = self.commit(prepare.token, scene)?;

        Ok(ReloadResult {
            prepare,
            commit,
            reconciliation: Some(reconciliation),
        })
    }

    // ── Internal Helpers ───────────────────────────────────────────────

    fn require_candidate(&self) -> Result<(), BspRuntimeError> {
        if self.candidate.is_none() {
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Validate,
                message: "no prepared candidate".to_string(),
            });
        }
        Ok(())
    }

    fn require_candidate_mut(&mut self) -> Result<&mut BspCandidate, BspRuntimeError> {
        self.candidate
            .as_mut()
            .ok_or_else(|| BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Validate,
                message: "no prepared candidate".to_string(),
            })
    }

    fn validate_candidate(
        &mut self,
        token: BspGenerationToken,
        scene: Option<&mut Scene>,
    ) -> Result<(), BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        self.generation.validate(token)?;
        let mut scene = scene;

        let result = (|| {
            let candidate_ref =
                self.candidate
                    .as_ref()
                    .ok_or_else(|| BspRuntimeError::BridgeFailure {
                        bridge_name: "coordinator".to_string(),
                        phase: BridgePhase::Validate,
                        message: "no prepared candidate".to_string(),
                    })?;

            if self.bridges.has_bridges() && !candidate_ref.is_validated() {
                self.bridges.validate_candidate(candidate_ref)?;
            }

            if let Some(scene) = scene.as_deref() {
                self.preflight_light_publication(candidate_ref, scene)?;
            } else if !candidate_ref.point_lights.is_empty() {
                return Err(BspRuntimeError::BridgeFailure {
                    bridge_name: "coordinator".to_string(),
                    phase: BridgePhase::Validate,
                    message: "BSP point-light publication requires validate_for_scene".to_string(),
                });
            }

            Ok(())
        })();

        if let Err(err) = result {
            match self.rollback_staged() {
                Ok(()) => return Err(err),
                Err(rollback_err) => return Err(rollback_err),
            }
        }

        let point_light_count = self
            .candidate
            .as_ref()
            .map(|c| c.point_lights.len())
            .unwrap_or(0);
        let reserve_additional = point_light_count.saturating_sub(self.active_lights.len());
        self.active_lights.reserve(reserve_additional);
        if let Some(scene) = scene.as_deref_mut() {
            let replacing_active_lights = if self.is_active() {
                self.active_lights.len()
            } else {
                0
            };
            let total_slots = scene
                .available_point_light_slots()
                .saturating_add(replacing_active_lights);
            scene.reserve_point_light_storage(total_slots);
        }

        if let Some(ref mut c) = self.candidate {
            c.mark_validated();
            c.mark_publication_validated();
        }
        Ok(())
    }

    /// Maximum BSP point lights published to the scene (GPU limit is 16).
    const MAX_BSP_POINT_LIGHTS: usize = 16;

    fn build_candidate_point_lights(
        extracted: &ExtractedBsp,
    ) -> Result<Vec<CandidatePointLight>, BspRuntimeError> {
        let mut point_lights = Vec::with_capacity(
            extracted
                .light_descriptors
                .len()
                .min(Self::MAX_BSP_POINT_LIGHTS),
        );
        for light in &extracted.light_descriptors {
            if point_lights.len() >= Self::MAX_BSP_POINT_LIGHTS {
                log::warn!(
                    "BSP light cap ({}) reached; {} remaining light entities not published",
                    Self::MAX_BSP_POINT_LIGHTS,
                    extracted.light_descriptors.len() - point_lights.len()
                );
                break;
            }
            let color = glam::Vec3::from_array(light.color);
            if !light.origin.is_finite()
                || !color.is_finite()
                || !light.intensity.is_finite()
                || light.intensity < 0.0
                || !light.radius.is_finite()
            {
                log::warn!(
                    "BSP light descriptor for entity {} is not publishable (non-finite values); skipping",
                    light.entity_index
                );
                continue;
            }

            point_lights.push(CandidatePointLight {
                entity_index: light.entity_index,
                light: PointLight {
                    position: light.origin,
                    color: color.max(glam::Vec3::ZERO),
                    intensity: light.intensity,
                    range: light.radius.max(1.0),
                },
            });
        }
        Ok(point_lights)
    }

    /// Build bridge DTOs from an extracted BSP.
    fn build_bridge_dtos(&self, extracted: &ExtractedBsp) -> BridgeDtos {
        let world_collision = WorldCollisionRecipe {
            planes: extracted.world_collision_planes.clone(),
        };

        let collision_by_entity: std::collections::HashMap<
            u32,
            Vec<bsp::collision::CollisionRecipe>,
        > = extracted.collision_recipes.iter().cloned().fold(
            std::collections::HashMap::new(),
            |mut map, recipe| {
                map.entry(recipe.entity_index).or_default().push(recipe);
                map
            },
        );
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
            .map(|ed| {
                // Extract door/button/platform-specific properties from key_values.
                let kv = &ed.key_values;
                let movedir = parse_vec3_opt(&get_kv(kv, "movedir"))
                    .map(|v| [v.x, v.y, v.z]);
                let speed = get_kv(kv, "speed").and_then(|s| s.parse::<f32>().ok());
                let wait = get_kv(kv, "wait").and_then(|s| s.parse::<f32>().ok());
                let lip = get_kv(kv, "lip").and_then(|s| s.parse::<f32>().ok());
                let height = get_kv(kv, "height").and_then(|s| s.parse::<f32>().ok());
                let killtarget = get_kv(kv, "killtarget").map(|s| s.to_string());
                let light_style = get_kv(kv, "style").map(|s| s.to_string());
                BehaviorEntityRecipe {
                    entity_index: ed.entity_index,
                    classname: ed.classname.clone(),
                    origin: ed.origin.unwrap_or(glam::Vec3::ZERO),
                    targetname: ed.targetname.clone(),
                    target: ed.target.clone(),
                    killtarget,
                    movedir,
                    speed,
                    wait,
                    lip,
                    height,
                    light_style,
                }
            })
            .collect();

        BridgeDtos {
            world_collision,
            entity_colliders,
            lights: light_recipes,
            behaviors: behavior_recipes,
        }
    }

    /// Build a source link from extracted DTOs using the versioned schema.
    fn build_source_link(
        &self,
        extracted: &ExtractedBsp,
        source_identity: &str,
        scale: f32,
    ) -> BspSourceLink {
        let content_hash_hex: String = extracted
            .content_hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        let content_hash = format!("sha256:{}", content_hash_hex);

        let entity_identity_records = crate::source_link::build_identity_records(
            &extracted.entity_identities,
            &extracted.entity_descriptors,
        );

        let mut link = BspSourceLink::new(source_identity.to_string(), content_hash);
        link.import_policy = crate::source_link::ImportPolicy {
            scale: crate::source_link::CanonicalFloat(scale),
            ..Default::default()
        };
        link.entity_identity_records = entity_identity_records;

        link
    }

    /// Build a cache identity from extracted DTOs.
    fn build_cache_identity(&self, extracted: &ExtractedBsp) -> CacheIdentity {
        let mut companion_identities = extracted
            .textures
            .iter()
            .flat_map(|texture| {
                [
                    texture
                        .pbr_companions
                        .normal
                        .as_ref()
                        .map(|companion| CompanionId {
                            kind: format!("pbr-normal:{}", texture.identity),
                            content_hash: compute_identity_hash(&companion.bytes),
                        }),
                    texture
                        .pbr_companions
                        .gloss
                        .as_ref()
                        .map(|companion| CompanionId {
                            kind: format!("pbr-gloss:{}", texture.identity),
                            content_hash: compute_identity_hash(&companion.bytes),
                        }),
                ]
                .into_iter()
                .flatten()
            })
            .collect::<Vec<_>>();
        companion_identities.sort();

        CacheIdentity::compute(
            extracted.content_hash,
            extracted.profile_tag,
            0.0254,
            [0; 32],
            companion_identities,
            vec![],
            vec![],
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }

    fn validate_restore_source_link(
        stored: &BspSourceLink,
        current: &BspSourceLink,
    ) -> Result<(), BspRuntimeError> {
        if stored.companion_hashes != current.companion_hashes {
            if stored.companion_hashes.palette != current.companion_hashes.palette {
                return Err(BspRuntimeError::CompanionMismatch {
                    kind: "palette".into(),
                    expected: stored
                        .companion_hashes
                        .palette
                        .clone()
                        .unwrap_or_else(|| "<none>".into()),
                    actual: current
                        .companion_hashes
                        .palette
                        .clone()
                        .unwrap_or_else(|| "<none>".into()),
                });
            }
            if stored.companion_hashes.lit != current.companion_hashes.lit {
                return Err(BspRuntimeError::CompanionMismatch {
                    kind: "lit".into(),
                    expected: stored
                        .companion_hashes
                        .lit
                        .clone()
                        .unwrap_or_else(|| "<none>".into()),
                    actual: current
                        .companion_hashes
                        .lit
                        .clone()
                        .unwrap_or_else(|| "<none>".into()),
                });
            }
            return Err(BspRuntimeError::CompanionMismatch {
                kind: "wad".into(),
                expected: format!("{:?}", stored.companion_hashes.wads),
                actual: format!("{:?}", current.companion_hashes.wads),
            });
        }

        if stored.model_mapping_identity != current.model_mapping_identity {
            return Err(BspRuntimeError::MappingMismatch {
                reason: "stored model-mapping identity does not match current import".into(),
            });
        }

        Ok(())
    }

    fn cancel_restore_candidate<T>(&mut self, err: BspRuntimeError) -> Result<T, BspRuntimeError> {
        match self.rollback_staged() {
            Ok(()) => Err(err),
            Err(rollback_err) => Err(rollback_err),
        }
    }

    fn preflight_light_publication(
        &self,
        candidate: &BspCandidate,
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
        if candidate.point_lights.len() > available_slots {
            return Err(BspRuntimeError::BridgeFailure {
                bridge_name: "coordinator".to_string(),
                phase: BridgePhase::Validate,
                message: format!(
                    "BSP point-light publication would exceed capacity: need {}, available {}",
                    candidate.point_lights.len(),
                    available_slots
                ),
            });
        }
        Ok(())
    }

    fn rollback_staged(&mut self) -> Result<(), BspRuntimeError> {
        // Roll back bridge tokens from candidate
        if let Some(mut candidate) = self.candidate.take() {
            let tokens = std::mem::take(&mut candidate.bridge_tokens);
            let failures = self.bridges.rollback_tokens(tokens);
            if !failures.is_empty() {
                self.poisoned = true;
                return Err(BspRuntimeError::RollbackFailure { failures });
            }
        }
        Ok(())
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

/// Container for bridge DTOs built from extraction.
struct BridgeDtos {
    world_collision: WorldCollisionRecipe,
    entity_colliders: Vec<EntityCollisionRecipe>,
    lights: Vec<LightEntityRecipe>,
    behaviors: Vec<BehaviorEntityRecipe>,
}

// ── Key/Value helpers ────────────────────────────────────────────────

/// Get a singleton key value from entity key/value pairs.
fn get_kv<'a>(kvs: &'a [bsp::entities::KeyValue], key: &str) -> Option<&'a str> {
    for kv in kvs.iter().rev() {
        if kv.key == key {
            return Some(kv.value.as_str());
        }
    }
    None
}

/// Parse an optional `"x y z"` string into a glam::Vec3.
fn parse_vec3_opt(s: &Option<&str>) -> Option<glam::Vec3> {
    let s = s.as_ref()?;
    let parts: Vec<f32> = s
        .split_whitespace()
        .filter_map(|p| p.parse::<f32>().ok())
        .collect();
    if parts.len() == 3 {
        Some(glam::Vec3::new(parts[0], parts[1], parts[2]))
    } else {
        None
    }
}
