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
use crate::cache::{
    canonical_f32_bytes, compute_identity_hash, CacheIdentity, CompanionId, PbrClosureEntry,
    WadCacheEntry,
};
use crate::candidate::{
    ActiveBspMount, BspCandidate, CandidatePointLight, ImportProvenanceRecord,
};
use crate::error::{BridgePhase, BspRuntimeError};
use crate::generation::{BspGenerationCounter, BspGenerationToken};
use crate::package::{AuthorizedBspImport, ImportMode, PbrCompanionKind};
use crate::source_link::{reconcile_overrides, BspSourceLink, OverrideReconciliation};

use bsp::extract::{EntityDescriptor, ExtractedBsp};
use renderer::api::bsp::PreparedBspMount;
use renderer::api::{PointLight, Scene};

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
/// // Step 1: Pass the complete AuthorizedBspImport returned by
/// // authorize_package_import or authorize_direct_import.
/// let prepare = coordinator.prepare_authorized_import(import)?;
///
/// // Step 2: Upload renderer resources
/// let mount = renderer.prepare_bsp_mount(coordinator.staged_extraction().unwrap())?;
/// coordinator.set_renderer_mount_ready(prepare.token, mount)?;
///
/// // Step 3: Validate publication
/// coordinator.validate_for_scene(prepare.token, &mut scene)?;
///
/// // Step 4: Commit (pure publish — no new work)
/// let commit = coordinator.commit(prepare.token, &mut scene)?;
/// ```
pub struct BspCoordinator {
    /// Monotonic generation counter for serialize-and-stale detection.
    generation: BspGenerationCounter,

    /// Active published BSP mount (if a mount is active).
    ///
    /// Separated from the candidate: the candidate is hidden preparation;
    /// the active mount is the published state visible in the scene.
    active_mount: Option<ActiveBspMount>,

    /// Current preparation candidate (hidden until commit).
    candidate: Option<BspCandidate>,

    /// App bridge aggregator.
    bridges: BridgeAggregator,

    /// Whether the coordinator has been poisoned.
    poisoned: bool,

    /// Diagnostic counter: number of opaque retirement handoffs recorded.
    /// Incremented each time a replaced/unloaded active mount's lease is
    /// sent to renderer retirement.
    retired_mount_count: u64,
}

impl BspCoordinator {
    /// Create a new BSP coordinator with no active mount.
    pub fn new() -> Self {
        Self {
            generation: BspGenerationCounter::new(),
            active_mount: None,
            candidate: None,
            bridges: BridgeAggregator::new(),
            poisoned: false,
            retired_mount_count: 0,
        }
    }

    // ── Query ──────────────────────────────────────────────────────────

    /// Returns true if a BSP mount is currently active.
    pub fn is_active(&self) -> bool {
        self.active_mount.is_some()
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
        self.active_mount.as_ref().map(|m| &m.source_link)
    }

    /// Returns a reference to the active cache identity, if any.
    pub fn cache_identity(&self) -> Option<&CacheIdentity> {
        self.active_mount.as_ref().map(|m| &m.cache_identity)
    }

    /// Returns the number of retired mounts (diagnostics).
    pub fn retired_mount_count(&self) -> u64 {
        self.retired_mount_count
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

    /// Prepare raw bytes through the explicit development/test compatibility path.
    ///
    /// Package and direct startup must use [`prepare_authorized_import`](Self::prepare_authorized_import).
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
    #[doc(hidden)]
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
            strict: ImportMode::Development.is_strict(),
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

    /// Prepare a legacy package wrapper through the one authorized-import path.
    ///
    /// The wrapper owns a complete [`AuthorizedBspImport`], so this compatibility
    /// entry point cannot discard policy or companion resources.
    #[doc(hidden)]
    pub fn prepare_from_loaded_package(
        &mut self,
        package: crate::package::LoadedBspPackage,
    ) -> Result<PrepareResult, BspRuntimeError> {
        self.prepare_authorized_import(package.into_authorized_import())
    }

    /// Prepare an authorized BSP import — the single entry point for package
    /// and direct launch paths.
    ///
    /// Derives [`BspExtractionRequest`] from the authorized import record,
    /// carries all authorized bytes and settings into extraction, and builds
    /// cache identity and source-link provenance from the import closure.
    /// Required resource failures (palette, WAD, .lit in strict mode) stop
    /// before candidate/GPU work.
    pub fn prepare_authorized_import(
        &mut self,
        import: AuthorizedBspImport,
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
            "BSP coordinator: generation {} — starting authorized import (route={}, policy={:?})",
            self.generation.current(),
            import.provenance.route,
            import.policy,
        );
        let token = self.generation.token();

        let was_occupied = self.is_active();
        let source_identity = import.bsp.logical_id.clone();
        let resolved_scale = import.scale;

        // Derive extraction request from the authorized import.
        let extraction_request = import.to_extraction_request();
        let world_profile_tag = extraction_request.world.profile.tag().to_string();

        let extracted = bsp::extract::extract(extraction_request).map_err(|e| {
            BspRuntimeError::SourceUnavailable {
                reason: format!("BSP extraction failed: {} (code {:?})", e.message, e.code),
            }
        })?;

        let face_count = extracted.face_geometries.len();
        let entity_count = extracted.entity_descriptors.len();
        let light_count = extracted.light_descriptors.len();
        let batch_count = extracted.render_batches.len();
        let has_pvs = extracted.has_pvs;

        // Build source link from authorized import.
        let source_link = self.build_source_link_from_import(
            &extracted,
            &import,
            &source_identity,
            resolved_scale,
        );
        let envelope = crate::source_link::BspPersistenceEnvelope::new(source_link.clone());
        let source_link_json =
            serde_json::to_value(&envelope).map_err(|e| BspRuntimeError::SourceUnavailable {
                reason: format!("BSP source-link serialization failed: {e}"),
            })?;

        // Build cache identity from authorized import.
        let cache_identity =
            self.build_cache_identity_from_import(&extracted, &import, &world_profile_tag);
        let point_lights = Self::build_candidate_point_lights(&extracted)?;

        // Build bridge DTOs and call app bridge prepare.
        let bridge_dtos = self.build_bridge_dtos(&extracted);
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

        let mut candidate = BspCandidate::new(
            token.generation,
            source_identity.clone(),
            extracted,
            cache_identity.clone(),
            source_link,
            source_link_json,
            point_lights,
            bridge_tokens,
            was_occupied,
        );

        // Attach Phase 03 import provenance for diagnostics.
        candidate.import_provenance = Some(ImportProvenanceRecord {
            route: import.provenance.route.clone(),
            strict: import.policy.is_strict(),
            asset_id: import.bsp.logical_id.clone(),
        });

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
            palette: None,
            texture_companions,
            wad_archives,
            strict: ImportMode::Development.is_strict(),
            scale: resolved_scale,
            fullbright_start: 224,
            fullbright_end: 255,
            max_atlas_pages: bsp::lightmaps::MAX_ATLAS_PAGES,
            overbright: 2.0,
            light_scale: 1.0,
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
    /// This transitions the candidate from `CpuPrepared` to `RendererPending`.
    /// Call after issuing an async upload.
    pub fn start_renderer_upload(
        &mut self,
        token: BspGenerationToken,
    ) -> Result<(), BspRuntimeError> {
        self.generation.validate(token)?;
        self.require_candidate()?;
        let current_gen = self.generation.current();
        let candidate = self.candidate.as_mut().unwrap();
        candidate.transition_to_renderer_pending(current_gen)
    }

    /// Complete a renderer upload, transitioning the candidate to `RendererReady`.
    ///
    /// The caller provides the completed [`PreparedBspMount`] from the renderer.
    /// After this, the candidate is eligible for validation.
    ///
    /// # Phase 05: Stale Completion Handling
    ///
    /// If the candidate generation no longer matches the coordinator's current
    /// generation, the mount is NOT accepted. Instead, the coordinator records
    /// a stale completion event and returns `StaleRendererCompletion`. The caller
    /// is responsible for retiring the stale mount.
    pub fn complete_renderer_upload(
        &mut self,
        token: BspGenerationToken,
        mount: PreparedBspMount,
    ) -> Result<(), BspRuntimeError> {
        self.generation.validate(token)?;

        // If the candidate is missing, this is a stale completion.
        let candidate = match self.candidate.as_mut() {
            Some(c) => c,
            None => {
                // No candidate — the mount belongs to a cancelled generation.
                // Return the stale error; caller retires the mount.
                return Err(BspRuntimeError::StaleRendererCompletion {
                    candidate_generation: token.generation,
                    current_generation: self.generation.current(),
                });
            }
        };

        let current_gen = self.generation.current();
        match candidate.transition_to_renderer_ready(current_gen, mount) {
            Ok(()) => Ok(()),
            // Stale or duplicate — the mount was not consumed. Caller retires it.
            Err(e @ BspRuntimeError::StaleRendererCompletion { .. }) => Err(e),
            Err(e @ BspRuntimeError::DuplicateReadyLease { .. }) => Err(e),
            Err(other) => Err(other),
        }
    }

    /// Legacy compatibility: set mount ready + validate.
    #[doc(hidden)]
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
        let current_gen = self.generation.current();
        let candidate = match self.candidate.as_mut() {
            Some(c) => c,
            None => {
                return Err(BspRuntimeError::StaleRendererCompletion {
                    candidate_generation: token.generation,
                    current_generation: current_gen,
                });
            }
        };
        candidate.transition_to_renderer_failed(current_gen, reason)
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

        // Candidate must be ValidatedForScene.
        {
            let candidate = self.require_candidate_mut()?;
            if !candidate.is_commit_ready() {
                return Err(BspRuntimeError::InvalidCandidateTransition {
                    current: crate::error::CandidatePhase::from(candidate.state),
                    attempted: crate::error::CandidatePhase::Consumed,
                    detail: "candidate must be ValidatedForScene before commit".to_string(),
                });
            }
        }

        // Commit bridges (non-fallible activation after validation).
        let bridge_count = if self.bridges.has_bridges() {
            match self.bridges.commit_candidate(self.candidate.as_mut().unwrap()) {
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

        // Take candidate and consume into active mount.
        let candidate = self.candidate.take().ok_or_else(|| {
            self.poisoned = true;
            BspRuntimeError::CommitContractViolated {
                detail: "candidate missing after bridge commit".to_string(),
            }
        })?;

        // Retire the old active mount (opaque handoff to renderer).
        if let Some(old_mount) = self.active_mount.take() {
            self.retire_active_mount_into_scene(old_mount, scene);
            self.retired_mount_count = self.retired_mount_count.saturating_add(1);
        }

        // Publish scene lights from prevalidated payloads.
        let point_lights: Vec<_> = candidate.point_lights.iter().cloned().collect();
        let mut new_light_ids = Vec::with_capacity(point_lights.len());
        for candidate_light in &point_lights {
            match scene.create_point_light(candidate_light.light) {
                Ok(id) => new_light_ids.push(id),
                Err(e) => {
                    // Contract violation after validate_for_scene: poison.
                    for id in new_light_ids.drain(..) {
                        let _ = scene.remove_point_light(id);
                    }
                    self.poisoned = true;
                    return Err(BspRuntimeError::CommitContractViolated {
                        detail: format!(
                            "prevalidated BSP light publication failed for entity {}: {:?}",
                            candidate_light.entity_index, e
                        ),
                    });
                }
            }
        }

        let light_count = new_light_ids.len();

        // Consume candidate into ActiveBspMount.
        let active_mount = match candidate.consume_into_active(
            self.generation.current(),
            new_light_ids,
        ) {
            Ok(m) => m,
            Err(e) => {
                // The candidate was not consumed; lights and mount are rolled back.
                // The error variant should propagate.
                self.poisoned = true;
                return Err(e);
            }
        };

        // Publish to scene.
        scene.set_bsp_mount(active_mount.renderer_lease.clone());
        scene.set_bsp_source_link(active_mount.source_link_json.clone());

        let cache_identity = active_mount.cache_identity.clone();
        self.active_mount = Some(active_mount);

        Ok(CommitResult {
            node_count: 0,
            light_count,
            bridge_count,
            cache_identity,
        })
    }

    /// Legacy compatibility wrapper. Sets the mount ready, auto-validates
    /// if not already validated (for candidates with no scene lights),
    /// then commits.
    #[doc(hidden)]
    pub fn commit_with_mount(
        &mut self,
        token: BspGenerationToken,
        scene: &mut Scene,
        mount: PreparedBspMount,
    ) -> Result<CommitResult, BspRuntimeError> {
        self.set_renderer_mount_ready(token, mount)?;

        // Auto-validate if the candidate has no scene lights (backward compat).
        // If there are lights, validate_for_scene must have been called already.
        let needs_validate = self
            .candidate
            .as_ref()
            .map(|c| c.state == crate::candidate::CandidateState::RendererReady)
            .unwrap_or(false);
        if needs_validate {
            let point_lights_empty = self
                .candidate
                .as_ref()
                .map(|c| c.point_lights.is_empty())
                .unwrap_or(true);
            if point_lights_empty {
                // Auto-validate: no scene lights means no scene preflight needed.
                if let Some(ref mut c) = self.candidate {
                    c.transition_to_validated_for_scene(self.generation.current())?;
                }
            }
        }

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
        self.rollback_candidate_with_retirement()
    }

    /// The number of opaque retirement handoffs recorded (diagnostics).
    pub fn retirement_diagnostics(&self) -> u64 {
        self.retired_mount_count
    }

    // ── Unload ─────────────────────────────────────────────────────────

    /// Unload the active BSP mount, removing all associated resources from
    /// the scene.
    ///
    /// 1. Increments generation (cancels any in-flight prepare)
    /// 2. Removes BSP scene nodes, lights, materials
    /// 3. Calls app bridge rollback hooks
    /// 4. Clears coordinator state and retires GPU resources
    pub fn unload(&mut self, scene: &mut Scene) -> Result<(), BspRuntimeError> {
        if self.poisoned {
            return Err(BspRuntimeError::CoordinatorPoisoned);
        }

        // Cancel any in-flight candidate
        self.generation
            .increment()
            .ok_or(BspRuntimeError::GenerationExhausted)?;

        // Roll back any staged candidate
        self.rollback_candidate_with_retirement()?;

        // Remove active mount from scene (opaque retirement)
        if let Some(old_mount) = self.active_mount.take() {
            self.retire_active_mount_into_scene(old_mount, scene);
            self.retired_mount_count = self.retired_mount_count.saturating_add(1);
        }

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
            .active_mount
            .as_ref()
            .map(|m| m.source_link.overrides.clone())
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
            .active_mount
            .as_ref()
            .map(|m| m.source_link.overrides.clone())
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
        self.rollback_candidate_with_retirement().ok();
        // Release active mount
        if let Some(old_mount) = self.active_mount.take() {
            self.retire_active_mount_into_scene(old_mount, scene);
            self.retired_mount_count = self.retired_mount_count.saturating_add(1);
        }
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
        let source_link = &self.active_mount.as_ref()?.source_link;
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
        let current_gen = self.generation.current();
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

            // Phase 05: Bridge validation (non-fallible after this point)
            if self.bridges.has_bridges() {
                self.bridges.validate_candidate(candidate_ref)?;
            }

            // Phase 05: Scene light preflight
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
            // Roll back only this candidate; leave active mount unchanged.
            self.rollback_candidate_with_retirement()?;
            return Err(err);
        }

        // Reserve point-light capacity before commit (no growing during commit).
        let active_light_count = self
            .active_mount
            .as_ref()
            .map(|m| m.light_ids.len())
            .unwrap_or(0);

        if let Some(scene) = scene.as_deref_mut() {
            let total_slots = scene
                .available_point_light_slots()
                .saturating_add(active_light_count);
            scene.reserve_point_light_storage(total_slots);
        }

        // Transition candidate to ValidatedForScene.
        if let Some(ref mut c) = self.candidate {
            c.transition_to_validated_for_scene(current_gen)?;
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
                let movedir = parse_vec3_opt(&get_kv(kv, "movedir")).map(|v| [v.x, v.y, v.z]);
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

    /// Build a source link from extracted DTOs and an authorized import record.
    fn build_source_link_from_import(
        &self,
        extracted: &ExtractedBsp,
        import: &AuthorizedBspImport,
        source_identity: &str,
        scale: f32,
    ) -> BspSourceLink {
        // The package resolver issued the authoritative SHA-256. Do not
        // persist the BSP crate's internal extraction fingerprint as though
        // it were a SHA-256 resource identity.
        let content_hash = format!("sha256:{}", import.bsp.identity.hex());

        let entity_identity_records = crate::source_link::build_identity_records(
            &extracted.entity_identities,
            &extracted.entity_descriptors,
        );

        let mut link = BspSourceLink::new(source_identity.to_string(), content_hash);

        // Populate import policy from authorized import.
        link.import_policy = crate::source_link::ImportPolicy {
            scale: crate::source_link::CanonicalFloat(scale),
            light_calibration: crate::source_link::ImportLightCalibration {
                intensity_scale: crate::source_link::CanonicalFloat(import.light_scale),
                overbright: crate::source_link::CanonicalFloat(import.overbright),
            },
            atlas_policy: crate::source_link::AtlasPolicy::default(),
            texture_roots: import.provenance.logical_root.iter().cloned().collect(),
            strict: import.policy.is_strict(),
        };

        // Populate companion hashes from authorized import.
        link.companion_hashes.palette = import
            .palette
            .as_ref()
            .map(|r| format!("sha256:{}", r.identity.hex()));
        link.companion_hashes.lit = import
            .lit
            .as_ref()
            .map(|r| format!("sha256:{}", r.identity.hex()));
        link.companion_hashes.wads = import
            .wads
            .iter()
            .map(|w| {
                (
                    w.basename.clone(),
                    format!("sha256:{}", w.resource.identity.hex()),
                )
            })
            .collect();

        link.entity_identity_records = entity_identity_records;

        link
    }

    /// Build a cache identity from extracted DTOs and an authorized import record.
    fn build_cache_identity_from_import(
        &self,
        _extracted: &ExtractedBsp,
        import: &AuthorizedBspImport,
        profile_tag: &str,
    ) -> CacheIdentity {
        let bsp_content_hash = import.bsp.identity.as_bytes();
        let palette_present = import.palette.is_some();
        let palette_hash = import
            .palette
            .as_ref()
            .map(|r| *r.identity.as_bytes())
            .unwrap_or([0u8; 32]);

        // Build companion identities: .lit and WAD entries.
        let mut companion_identities: Vec<CompanionId> = Vec::new();
        if let Some(ref lit) = import.lit {
            companion_identities.push(CompanionId {
                kind: "lit".to_string(),
                logical_id: Some(lit.logical_id.clone()),
                content_hash: *lit.identity.as_bytes(),
            });
        }
        for wad in &import.wads {
            companion_identities.push(CompanionId {
                kind: format!("wad:{}", wad.basename),
                logical_id: Some(wad.resource.logical_id.clone()),
                content_hash: *wad.resource.identity.as_bytes(),
            });
        }
        companion_identities.sort();

        // Build WAD cache entries (preserve declaration order).
        let wad_entries: Vec<WadCacheEntry> = import
            .wads
            .iter()
            .map(|w| WadCacheEntry {
                ordinal: w.ordinal,
                basename: w.basename.clone(),
                logical_id: w.resource.logical_id.clone(),
                content_hash: *w.resource.identity.as_bytes(),
            })
            .collect();

        // Build PBR closure entries.
        let mut pbr_closure: Vec<PbrClosureEntry> = import
            .pbr
            .iter()
            .map(|c| {
                let kind_str = match c.kind {
                    PbrCompanionKind::Normal => "normal",
                    PbrCompanionKind::Gloss => "gloss",
                };
                PbrClosureEntry {
                    source_slot: c.source_slot,
                    texture_identity: c.texture_identity.clone(),
                    kind: kind_str.to_string(),
                    match_mode: c.match_mode.tag().to_string(),
                    present: c.resource.is_some(),
                    logical_id: c.resource.as_ref().map(|r| r.logical_id.clone()),
                    content_hash: c
                        .resource
                        .as_ref()
                        .map(|r| *r.identity.as_bytes())
                        .unwrap_or([0u8; 32]),
                }
            })
            .collect();
        pbr_closure.sort_by(|a, b| a.source_slot.cmp(&b.source_slot).then(a.kind.cmp(&b.kind)));

        CacheIdentity::compute(
            *bsp_content_hash,
            profile_tag,
            import.scale,
            palette_present,
            palette_hash,
            import.policy.is_strict(),
            companion_identities,
            wad_entries,
            pbr_closure,
            vec![],
            vec![],
            crate::cache::LightCalibration {
                intensity_scale: canonical_f32_bytes(import.light_scale),
                overbright: canonical_f32_bytes(import.overbright),
            },
            crate::cache::AtlasPolicy {
                page_size: 2048,
                padding: 2,
                style_count: 4,
                max_pages: u64::try_from(import.max_atlas_pages).unwrap_or(u64::MAX),
            },
            crate::cache::CollisionPolicy::default(),
            [import.fullbright_start, import.fullbright_end],
            import.overbright,
        )
    }

    /// Build a source link from extracted DTOs (raw-byte/development path).
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

    /// Build a cache identity from extracted DTOs (raw-byte/development path).
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
                            logical_id: None,
                            content_hash: compute_identity_hash(&companion.bytes),
                        }),
                    texture
                        .pbr_companions
                        .gloss
                        .as_ref()
                        .map(|companion| CompanionId {
                            kind: format!("pbr-gloss:{}", texture.identity),
                            logical_id: None,
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
            false,
            [0; 32],
            false,
            companion_identities,
            vec![],
            vec![],
            vec![],
            vec![],
            Default::default(),
            Default::default(),
            Default::default(),
            [224, 255],
            2.0,
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
        match self.rollback_candidate_with_retirement() {
            Ok(()) => Err(err),
            Err(rollback_err) => Err(rollback_err),
        }
    }

    fn preflight_light_publication(
        &self,
        candidate: &BspCandidate,
        scene: &Scene,
    ) -> Result<(), BspRuntimeError> {
        let replacing_active_lights = self
            .active_mount
            .as_ref()
            .map(|m| m.light_ids.len())
            .unwrap_or(0);
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

    /// Roll back a staged candidate, including any ready renderer lease
    /// sent to retirement, and bridge token rollback.
    ///
    /// After this call, the candidate is cleared (taken).
    fn rollback_candidate_with_retirement(&mut self) -> Result<(), BspRuntimeError> {
        let mut candidate = match self.candidate.take() {
            Some(c) => c,
            None => return Ok(()),
        };

        let (tokens, ready_mount) = candidate.rollback();

        // Roll back bridge tokens.
        let failures = self.bridges.rollback_tokens(tokens);
        if !failures.is_empty() {
            self.poisoned = true;
            return Err(BspRuntimeError::RollbackFailure { failures });
        }

        // If there was a ready renderer mount, it needs opaque retirement.
        // We discard it here since we don't have a Scene; the caller should
        // have retired it already, or this is a terminal path like cancel.
        if ready_mount.is_some() {
            log::debug!(
                "BSP candidate rollback: discarding ready renderer mount (retirement not enqueued)"
            );
            self.retired_mount_count = self.retired_mount_count.saturating_add(1);
        }

        // Candidate is dropped here; its state is already RolledBack.
        Ok(())
    }

    /// Retire an active mount: clear scene references and hand off the opaque
    /// renderer lease to the scene for fence-based retirement.
    ///
    /// `bsp_runtime` records the handoff but does not read raw GPU handles,
    /// compute serials, or reap slots.
    fn retire_active_mount_into_scene(
        &mut self,
        old_mount: ActiveBspMount,
        scene: &mut Scene,
    ) {
        // Remove the old mount's scene resources.
        scene.clear_bsp_mount();
        scene.clear_bsp_source_link();

        for light_id in &old_mount.light_ids {
            if let Err(e) = scene.remove_point_light(*light_id) {
                log::warn!(
                    "failed to remove BSP point light slot={} during retirement: {:?}",
                    light_id.slot,
                    e
                );
            }
        }

        // The old renderer lease is dropped here. The scene has cleared its
        // BSP mount, which queues GPU resources for fence-observed retirement.
        log::debug!(
            "BSP coordinator: retired active mount '{}' (generation {})",
            old_mount.source_identity,
            old_mount.committed_generation
        );
    }

    /// Legacy rollback helper — use rollback_candidate_with_retirement instead.
    fn rollback_staged(&mut self) -> Result<(), BspRuntimeError> {
        self.rollback_candidate_with_retirement()
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
