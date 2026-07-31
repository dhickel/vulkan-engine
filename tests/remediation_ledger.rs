//! Authoritative Finding Ledger — Engine Safety Refactor Stabilization
//!
//! This file is the clean-checkout canonical 21-row finding ledger produced by
//! Phase 10 closeout. Each row records a finding ID, the fix commit(s),
//! focused test evidence, and the final verified status.
//!
//! ## Reading this ledger
//!
//! - **Status `Resolved`**: code, tests, and docs align; no residual risk.
//! - **Status `PartiallyResolved`**: core invariants are fixed; one or more
//!   sub-scopes remain deferred with explicit justification.
//! - **Status `Deferred`**: acknowledged but intentionally left for a future
//!   milestone (see `deferred_reason`).
//!
//! ## Companion artifacts
//!
//! - `.internal-dev/plans/engine-safety-refactor-stabilization/finding-matrix.md`
//! - `.internal-dev/specifications/decisions.md`
//! - `.internal-dev/changelogs/2026-07-22-safety-refactor-red-team-remediation.md`
//! - `docs/internal/17-safety-refactor-remediation-ledger.md`

use std::collections::HashMap;

// ── Ledger entry ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FindingStatus {
    Resolved,
    PartiallyResolved,
    Deferred,
}

impl FindingStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Resolved => "Resolved",
            Self::PartiallyResolved => "PartiallyResolved",
            Self::Deferred => "Deferred",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FindingEntry {
    /// Canonical finding ID (e.g. "H-A1", "M-A7").
    pub id: &'static str,
    /// Human-readable one-line summary.
    pub summary: &'static str,
    /// Owning phase in the safety-refactor sprint.
    pub phase: &'static str,
    /// Commit(s) that resolved or advanced this finding.
    pub fix_commits: &'static [&'static str],
    /// Focused test evidence (function names or test module paths).
    pub test_evidence: &'static [&'static str],
    /// Final validated status.
    pub status: FindingStatus,
    /// Rationale when not fully Resolved.
    pub deferred_reason: Option<&'static str>,
}

// ── Ledger ────────────────────────────────────────────────────────────────

pub struct FindingLedger {
    entries: Vec<FindingEntry>,
}

impl FindingLedger {
    pub fn canonical() -> Self {
        Self {
            entries: vec![
                // ── High-severity addressed findings (H-A1 – H-A10) ──────

                FindingEntry {
                    id: "H-A1",
                    summary: "Animation atomicity: fallible try_set_clip/try_set_speed/try_update with typed AnimationError; validation before mutation; cubic Hermite spline interpolation with normalized quaternion slerp",
                    phase: "08",
                    fix_commits: &["398ea3c3"],
                    test_evidence: &[
                        "renderer::tests::integration::animation_try_set_clip_rejects_invalid_duration",
                        "renderer::tests::integration::animation_try_update_rejects_non_finite_input",
                        "renderer::tests::compatibility::set_clip_no_result (legacy compat)",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "H-A2",
                    summary: "Input multi-binding: per-instance BindingInstanceId tracking; aggregate action value as max across active instances; correct just_pressed/just_released edges when only last binding releases",
                    phase: "09",
                    fix_commits: &["398ea3c3"],
                    test_evidence: &[
                        "input::tests::action_remains_active_when_one_of_two_bindings_is_released",
                        "input::tests::binding_instance_ids_are_unique",
                        "input::tests::release_removes_only_the_released_binding",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "H-A3",
                    summary: "Scripting isolation: per-evaluation state replaces shared 'current_script' field; thread-safe identity via ScriptingContext-local evaluation records",
                    phase: "09",
                    fix_commits: &["398ea3c3"],
                    test_evidence: &[
                        "scripting::tests::concurrent_evaluations_have_independent_identities",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "H-A4",
                    summary: "Retirement atomicity: texture/material retirement through GpuRetirementQueue keyed by RetirementClass; fence-observed completion before destruction; reference tracking during scene flattening",
                    phase: "04",
                    fix_commits: &["ea238e9b", "398ea3c3"],
                    test_evidence: &[
                        "renderer::tests::integration::texture_retirement_waits_for_gpu_completion",
                        "renderer::tests::integration::material_retirement_preserves_shared_textures",
                        "renderer::tests::integration::retirement_rejects_reserved_default_slots",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "H-A5",
                    summary: "Scene v2 format: canonical DTO preserves directional/spot lights, collision, audio, visibility, prefab metadata; all three light kinds round-trip; save is failure-atomic with staged file publication",
                    phase: "08",
                    fix_commits: &["06baf00f", "6bc17a37"],
                    test_evidence: &[
                        "renderer::tests::integration::scene_round_trip_preserves_directional_lights",
                        "renderer::tests::integration::scene_round_trip_preserves_spot_lights",
                        "renderer::tests::integration::scene_round_trip_preserves_collision_metadata",
                        "renderer::tests::integration::scene_v2_rejects_stale_handles_in_payload",
                        "renderer::tests::integration::scene_save_is_failure_atomic",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "H-A6",
                    summary: "Shadow persistence: directional shadow owner survives scene save/load round-trip; CSM cascade config serialized in v2 scene format; legacy single-light path preserved for v1 compatibility",
                    phase: "08",
                    fix_commits: &["06baf00f"],
                    test_evidence: &[
                        "renderer::tests::integration::shadow_owner_survives_round_trip",
                        "renderer::tests::fixtures::scenes::v2_shadow_owner_parses_correctly",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "H-A7",
                    summary: "Transactional Assimp: RAII scene ownership with aiReleaseImport; null-pointer guards on aiMesh/aiMaterial dereference; property-store lifecycle tied to import scope; typed AssimpImportError replaces string errors",
                    phase: "05",
                    fix_commits: &["06baf00f"],
                    test_evidence: &[
                        "renderer::tests::integration::assimp_null_mesh_returns_error_not_panic",
                        "renderer::tests::integration::assimp_resource_cleanup_on_import_failure",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "H-A8",
                    summary: "Image lifetime: VkImageAlloc destroys image view before image/VMA allocation; sampler cache owns sampler destruction; descriptor-set images returned to pool before image teardown; prefilter cubemap mip batching prevents GPU watchdog timeout",
                    phase: "06",
                    fix_commits: &["06baf00f", "6bc17a37"],
                    test_evidence: &[
                        "renderer::tests::gpu_smoke::image_view_destroyed_before_image",
                        "renderer::tests::gpu_smoke::prefilter_mip_batches_isolated",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "H-A9",
                    summary: "Pipeline ownership: staged construction with failure-atomic rollback; deduplicated Vulkan handle destruction for shared pipeline layouts; depth/attachment invariants validated per pipeline class; transactional pipeline cache with staging directory and atomic rename",
                    phase: "07",
                    fix_commits: &["f9726280", "6bc17a37"],
                    test_evidence: &[
                        "renderer::tests::integration::pipeline_cache_is_transactional",
                        "renderer::tests::integration::pipeline_construction_rollback_cleans_all_handles",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "H-A10",
                    summary: "Capture truthfulness: serde_json serialization replaces manual JSON emission; NaN/±Inf floats serialized as JSON null; recording initialization validates output path and writes start record before activation; due-capture exact-match semantics with missed-frame status reporting",
                    phase: "09",
                    fix_commits: &["398ea3c3", "6bc17a37"],
                    test_evidence: &[
                        "renderer::tests::integration::debug_recording_rejects_invalid_output_path",
                        "renderer::tests::integration::capture_nan_floats_serialized_as_null",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },

                // ── Medium-severity addressed findings (M-A1 – M-A10) ─────

                FindingEntry {
                    id: "M-A1",
                    summary: "CSM fallbacks: feature-gated compilation (#[cfg(feature = \"csm\")]) eliminates dead-code warnings; legacy shadow path preserved for default builds; CSM symbols gated not suppressed; legacy/CSM dual-path documented as intentional",
                    phase: "08",
                    fix_commits: &["06baf00f"],
                    test_evidence: &[
                        "cargo check -p renderer (0 warnings default)",
                        "cargo check -p renderer --features csm (6 expected legacy-warning)",
                        "renderer::examples::capture_csm (headless validation)",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "M-A2",
                    summary: "Race-free scene save: StagedSceneFile reserves temp sibling, writes and fsyncs bytes, then atomically renames over target; partial writes never published; cleanup on failure removes staged file",
                    phase: "10",
                    fix_commits: &["6bc17a37"],
                    test_evidence: &[
                        "renderer::tests::integration::scene_save_is_failure_atomic",
                        "renderer::tests::integration::scene_save_cleanup_on_write_failure",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "M-A3",
                    summary: "O_NOFOLLOW staging: engine_pack fs_tx uses symlink_metadata (not metadata) for scan traversal; per-entry canonicalization with root containment; visited-device+inode set rejects filesystem cycles; symlinks silently skipped, not followed",
                    phase: "10",
                    fix_commits: &["6bc17a37"],
                    test_evidence: &[
                        "engine_pack::tests::symlink_scan_skips_external_targets",
                        "engine_pack::tests::scan_cycle_detection_rejects_loops",
                        "engine_pack::tests::root_containment_rejects_escape_candidates",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "M-A4",
                    summary: "UTF-8 rejection: asset registry uses serializer-backed TOML/JSON output; ad-hoc sanitize_id_component and toml_escape replaced with validated OsStr→str conversion that rejects invalid UTF-8 at the boundary",
                    phase: "10",
                    fix_commits: &["6bc17a37"],
                    test_evidence: &[
                        "engine_pack::tests::non_utf8_path_rejected_at_scan",
                        "engine_pack::tests::serializer_output_never_emits_invalid_utf8",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "M-A5",
                    summary: "CLI schema unification: single declarative ArgParser handles spaced and equals forms; duplicate singleton options explicitly rejected; usage text generated from schema, not hand-maintained",
                    phase: "10",
                    fix_commits: &["3609d1d0"],
                    test_evidence: &[
                        "engine_pack::tests::duplicate_option_rejected",
                        "engine_pack::tests::equals_form_parsed_identically_to_spaced",
                    ],
                    status: FindingStatus::PartiallyResolved,
                    deferred_reason: Some("root launcher and voxel_demo maintain independent parsers; full schema unification deferred past Phase 10"),
                },
                FindingEntry {
                    id: "M-A6",
                    summary: "Asset identity normalization: canonical project-relative asset keys derived from durable package IDs; registry deduplication by (package_id, asset_id) tuple; version fields consolidated; unsupported compression policy rejected at load",
                    phase: "10",
                    fix_commits: &["3609d1d0", "6bc17a37"],
                    test_evidence: &[
                        "engine_pack::tests::duplicate_asset_id_across_packages_rejected",
                        "engine_pack::tests::canonical_key_is_project_relative",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "M-A7",
                    summary: "Fail-closed pack publication: staging directory built as sibling; validated before atomic rename; ExistingTarget error when output exists; rollback journal on partial failure; stale PACK_REPORT.json removed before validation",
                    phase: "10",
                    fix_commits: &["6bc17a37"],
                    test_evidence: &[
                        "engine_pack::tests::pack_refuses_existing_output_directory",
                        "engine_pack::tests::pack_rollback_on_copy_failure",
                        "engine_pack::tests::stale_report_removed_before_validation",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "M-A8",
                    summary: "Hook failure observation: HookReport per-frame aggregation with per-hook status and error payload; register_hook returns typed RendererError on registration failure; frame render outcome distinguishes successful submit from retryable/rejected frames",
                    phase: "09",
                    fix_commits: &["398ea3c3"],
                    test_evidence: &[
                        "renderer::tests::integration::hook_report_aggregates_per_hook_status",
                        "renderer::tests::integration::hook_registration_failure_is_typed",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "M-A9",
                    summary: "Descriptor reset safety: CompletedFrameSlot single-use token created by fence-wait path; slot identity, exact frame serial, and epoch monotonicity validated before Vulkan call; partial reset failure quarantines all tracked pools as exhausted",
                    phase: "02",
                    fix_commits: &["d1ed19ae"],
                    test_evidence: &[
                        "renderer::tests::integration::descriptor_reset_rejects_consumed_token",
                        "renderer::tests::integration::descriptor_reset_rejects_mismatched_serial",
                        "renderer::tests::integration::partial_reset_quarantines_all_pools",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },
                FindingEntry {
                    id: "M-A10",
                    summary: "Frame indexing safety: VkPresent::get_next_frame/get_curr_frame_mut use checked_add for frame counters; NoActiveReservation sentinel when curr_frame_count == 0; empty frame_data rejected at construction; safe indexing replaces get_unchecked; rewind_frame preserves fence-signaled invariant",
                    phase: "02",
                    fix_commits: &["3609d1d0"],
                    test_evidence: &[
                        "renderer::tests::integration::frame_ring_rejects_zero_length",
                        "renderer::tests::integration::pre_first_frame_access_returns_no_active_reservation",
                        "renderer::tests::integration::frame_counter_exhaustion_is_checked",
                    ],
                    status: FindingStatus::Resolved,
                    deferred_reason: None,
                },

                // ── Ledger tail: one fully-deferred finding recorded for
                //     completeness ──────────────────────────────────────────

                FindingEntry {
                    id: "D-01",
                    summary: "God-module split (M2): scene.rs (4219→reduced lines), data_cache.rs, vk_render.rs still mix responsibilities; public SceneWorld exposure through api/scene.rs remains broad. Deferred past safety-refactor sprint.",
                    phase: "Deferred",
                    fix_commits: &[],
                    test_evidence: &[],
                    status: FindingStatus::Deferred,
                    deferred_reason: Some("requires stable facade extraction plan and cross-crate migration; no safety invariants depend on it"),
                },
            ],
        }
    }

    pub fn entries(&self) -> &[FindingEntry] {
        &self.entries
    }

    pub fn count_by_status(&self) -> HashMap<FindingStatus, usize> {
        let mut counts = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.status.clone()).or_insert(0) += 1;
        }
        counts
    }

    pub fn validate_expected_counts(&self) -> Result<(), String> {
        let counts = self.count_by_status();
        let resolved = counts.get(&FindingStatus::Resolved).copied().unwrap_or(0);
        let partial = counts
            .get(&FindingStatus::PartiallyResolved)
            .copied()
            .unwrap_or(0);
        let deferred = counts.get(&FindingStatus::Deferred).copied().unwrap_or(0);
        let total = resolved + partial + deferred;

        if total != 21 {
            return Err(format!(
                "ledger has {} entries; expected exactly 21 (resolved={}, partial={}, deferred={})",
                total, resolved, partial, deferred
            ));
        }
        if deferred > 1 {
            return Err(format!("more than one deferred entry ({deferred})"));
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_has_exactly_21_entries() {
        let ledger = FindingLedger::canonical();
        assert_eq!(
            ledger.entries().len(),
            21,
            "canonical ledger must have exactly 21 entries"
        );
    }

    #[test]
    fn ledger_validates_expected_counts() {
        let ledger = FindingLedger::canonical();
        ledger
            .validate_expected_counts()
            .expect("ledger counts must pass");
    }

    #[test]
    fn every_entry_has_non_empty_id_and_summary() {
        let ledger = FindingLedger::canonical();
        for entry in ledger.entries() {
            assert!(!entry.id.is_empty(), "entry must have non-empty id");
            assert!(
                !entry.summary.is_empty(),
                "entry {id} must have non-empty summary",
                id = entry.id
            );
        }
    }

    #[test]
    fn resolved_entries_have_commits_and_test_evidence() {
        let ledger = FindingLedger::canonical();
        for entry in ledger.entries() {
            if entry.status == FindingStatus::Resolved {
                assert!(
                    !entry.fix_commits.is_empty(),
                    "resolved entry {id} must have at least one fix commit",
                    id = entry.id
                );
                assert!(
                    !entry.test_evidence.is_empty(),
                    "resolved entry {id} must have at least one test evidence item",
                    id = entry.id
                );
            }
        }
    }

    #[test]
    fn deferred_entries_have_reason() {
        let ledger = FindingLedger::canonical();
        for entry in ledger.entries() {
            if entry.status == FindingStatus::Deferred {
                assert!(
                    entry.deferred_reason.is_some(),
                    "deferred entry {id} must have a deferred_reason",
                    id = entry.id
                );
            }
        }
    }

    #[test]
    fn partial_entries_have_reason() {
        let ledger = FindingLedger::canonical();
        for entry in ledger.entries() {
            if entry.status == FindingStatus::PartiallyResolved {
                assert!(
                    entry.deferred_reason.is_some(),
                    "partially-resolved entry {id} must have a deferred_reason",
                    id = entry.id
                );
            }
        }
    }

    #[test]
    fn no_duplicate_ids() {
        let ledger = FindingLedger::canonical();
        let mut seen = std::collections::HashSet::new();
        for entry in ledger.entries() {
            assert!(
                seen.insert(entry.id),
                "duplicate finding ID: {id}",
                id = entry.id
            );
        }
    }

    #[test]
    fn all_ids_follow_convention() {
        let ledger = FindingLedger::canonical();
        for entry in ledger.entries() {
            let valid = entry.id.starts_with("H-A")
                || entry.id.starts_with("M-A")
                || entry.id.starts_with("D-");
            assert!(
                valid,
                "finding ID {id} must follow H-A*, M-A*, or D-* convention",
                id = entry.id
            );
        }
    }
}
