//! Deterministic semantic pacing blueprint construction.
//!
//! Derives entrance, descent, landmark peak, and release beats from the
//! resolved request and seed using Richness field tags. Theme NEVER
//! participates in candidate keys or selection — the blueprint bytes must
//! be IDENTICAL across all three themes for the same seed/preset.
//!
//! # Architecture
//!
//! 1. Derive beats from preset + seed.
//! 2. Select required landmarks from compatible archetypes BEFORE natural rarity.
//!    Track forced selections separately.
//! 3. Apply normalized integer rarity at weights 70/25/5/1 for natural selections.
//! 4. Generate side-branch intents with exactly one observable payoff per leaf.
//! 5. Generate shortcut intents for Moderate/Rich.
//! 6. Validate all invariants.
//!
//! # Contract
//!
//! - Sparse/Moderate/Rich = exactly 1/2/3 distinct critical-path landmarks.
//! - Rare and legendary: no-repeat + one-per-map cap.
//! - Common archetype repeats only after all compatible unused common candidates
//!   are exhausted.
//! - Every leaf branch terminates in exactly one observable payoff.
//! - Label-only payoffs and invalid shortcuts are rejected.
//! - Moderate and Rich reserve at least one backward shortcut intent.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::fields::FieldTag;
use super::generated_content;
use super::ids::*;
use super::request::{ResolvedRichnessRequestV1, RichnessPreset, RichnessPresetRevision};
use super::zones::build_zone_blueprint;

// ── Pacing domain ──────────────────────────────────────────────────────────

/// Pacing selection domain tag for hash frames.
const PACING_DOMAIN: &[u8] = b"dungeon-gen/v3-richness/v1/pacing";

// ── Rarity weight constants ────────────────────────────────────────────────

/// Total weight for normalized rarity selection.
const RARITY_TOTAL_WEIGHT: u32 = 101;

/// Cumulative weight thresholds.
const COMMON_THRESHOLD: u32 = 70;
const UNCOMMON_THRESHOLD: u32 = 95; // 70 + 25
const RARE_THRESHOLD: u32 = 100; // 70 + 25 + 5
/// Legendary occupies weight 1 (index 100).

// ── Hash helper ────────────────────────────────────────────────────────────

/// Build a deterministic hash for pacing decisions.
///
/// Hash frame (frozen):
/// - u32 LE: len(domain)
/// - domain bytes
/// - u32 LE: len(tag)
/// - tag bytes
/// - u64 LE: seed
/// - u64 LE: preset_code (0=Sparse, 1=Moderate, 2=Rich)
/// - u32 LE: candidate_key
/// - u32 LE: reserved (0)
///
/// Theme MUST NOT appear in this frame.
fn pacing_hash(seed: u64, preset: RichnessPreset, tag: FieldTag, candidate_key: u32) -> [u8; 32] {
    let preset_code: u64 = match preset {
        RichnessPreset::Sparse => 0,
        RichnessPreset::Moderate => 1,
        RichnessPreset::Rich => 2,
    };

    let mut hasher = Sha256::new();
    hasher.update(&(PACING_DOMAIN.len() as u32).to_le_bytes());
    hasher.update(PACING_DOMAIN);
    let tag_bytes = tag.as_bytes();
    hasher.update(&(tag_bytes.len() as u32).to_le_bytes());
    hasher.update(tag_bytes);
    hasher.update(&seed.to_le_bytes());
    hasher.update(&preset_code.to_le_bytes());
    hasher.update(&candidate_key.to_le_bytes());
    hasher.update(&0u32.to_le_bytes()); // reserved
    hasher.finalize().into()
}

/// Build a canonical candidate key. Every archetype choice is framed by the
/// frozen preset revision, its semantic slot, and the stable archetype ID;
/// theme is intentionally absent.
fn pacing_candidate_key(
    preset: RichnessPreset,
    semantic_slot: u32,
    archetype: Option<ArchetypeIndex>,
) -> u32 {
    let mut hasher = Sha256::new();
    let revision = RichnessPresetRevision::V1.tag().as_bytes();
    let preset_tag = preset.tag().as_bytes();
    hasher.update(&(revision.len() as u32).to_le_bytes());
    hasher.update(revision);
    hasher.update(&(preset_tag.len() as u32).to_le_bytes());
    hasher.update(preset_tag);
    hasher.update(&semantic_slot.to_le_bytes());
    if let Some(archetype) = archetype {
        let id = archetype.id_str().as_bytes();
        hasher.update(&(id.len() as u32).to_le_bytes());
        hasher.update(id);
    } else {
        hasher.update(&0u32.to_le_bytes());
    }
    u32::from_le_bytes(hasher.finalize()[0..4].try_into().unwrap())
}

/// Extract a u32 from the first 4 bytes of a hash.
fn u32_from_hash(hash: &[u8; 32]) -> u32 {
    let bytes: [u8; 4] = hash[0..4].try_into().unwrap();
    u32::from_le_bytes(bytes)
}

/// Extract a u64 from the first 8 bytes of a hash.
fn u64_from_hash(hash: &[u8; 32]) -> u64 {
    let bytes: [u8; 8] = hash[0..8].try_into().unwrap();
    u64::from_le_bytes(bytes)
}

// ── Archetype compatibility ────────────────────────────────────────────────

/// Check if an archetype is compatible with a beat type.
fn archetype_compatible_with_beat_type(idx: ArchetypeIndex, beat_type: BeatType) -> bool {
    // All archetypes are potentially compatible; specific filtering is
    // done by zone families and beat type.
    let _zone_compat = generated_content::ARCHETYPE_ZONE_COMPAT[idx.raw() as usize];
    match beat_type {
        BeatType::Entrance => {
            // Entrance-appropriate archetypes: entrance_hall, vestibule, antechamber
            let id = idx.id_str();
            id == "entrance_hall" || id == "vestibule" || id == "antechamber"
        }
        BeatType::Descent => {
            // Descent-appropriate: grand_stair_hall, spiral_tower, ladder_hub, pit_room
            let id = idx.id_str();
            id == "grand_stair_hall"
                || id == "spiral_tower"
                || id == "ladder_hub"
                || id == "pit_room"
        }
        BeatType::LandmarkPeak => {
            // Landmark-appropriate: large, visually distinctive rooms
            let id = idx.id_str();
            id == "shrine"
                || id == "throne_hall"
                || id == "grand_arena"
                || id == "hypostyle_hall"
                || id == "observatory"
                || id == "treasury"
                || id == "reliquary"
                || id == "vault"
                || id == "ossuary"
                || id == "foundry"
                || id == "gallery"
                || id == "arena"
                || id == "guard_hall"
                || id == "crossroads"
                || id == "ambush_cross"
                || id == "kill_court"
        }
        BeatType::Release => {
            // Release-appropriate: rooms that provide satisfying conclusions
            let id = idx.id_str();
            id == "throne_hall"
                || id == "treasury"
                || id == "vault"
                || id == "reliquary"
                || id == "shrine"
                || id == "grand_arena"
                || id == "observatory"
        }
        BeatType::SideBranchLeaf => false,
    }
}

/// Get compatible archetypes for a beat type.
fn compatible_archetypes(beat_type: BeatType) -> Vec<ArchetypeIndex> {
    ArchetypeIndex::all()
        .filter(|&idx| archetype_compatible_with_beat_type(idx, beat_type))
        .collect()
}

// ── Beat derivation ────────────────────────────────────────────────────────

/// Derive beat properties (density class, degree intent) from seed + preset.
///
/// Ensures dense/quiet alternation on the critical path: Dense never follows
/// Dense, Quiet never follows Quiet. Transition can follow anything.
fn derive_beat_properties(
    seed: u64,
    preset: RichnessPreset,
    beat_index: u32,
    beat_type: BeatType,
    previous_density: Option<DensityClass>,
) -> (DensityClass, DegreeIntent) {
    let hash = pacing_hash(
        seed,
        preset,
        FieldTag::ValueNoise,
        pacing_candidate_key(preset, beat_index, None),
    );
    let val = u64_from_hash(&hash);

    let density = match beat_type {
        BeatType::Entrance => DensityClass::Transition,
        BeatType::Descent => DensityClass::Transition,
        BeatType::LandmarkPeak | BeatType::Release => {
            // Choose a density that alternates with the previous beat
            match previous_density {
                Some(DensityClass::Dense) => {
                    if val % 2 == 0 {
                        DensityClass::Quiet
                    } else {
                        DensityClass::Transition
                    }
                }
                Some(DensityClass::Quiet) => {
                    if val % 2 == 0 {
                        DensityClass::Dense
                    } else {
                        DensityClass::Transition
                    }
                }
                Some(DensityClass::Transition) | None => {
                    if val % 3 == 0 {
                        DensityClass::Dense
                    } else if val % 3 == 1 {
                        DensityClass::Quiet
                    } else {
                        DensityClass::Transition
                    }
                }
            }
        }
        BeatType::SideBranchLeaf => DensityClass::Transition,
    };

    let degree = match beat_type {
        BeatType::Entrance => DegreeIntent::new(2, 3),
        BeatType::Descent => DegreeIntent::new(2, 4),
        BeatType::LandmarkPeak => {
            let min = 1 + (val % 3) as u32;
            let max = min + 1 + (val % 2) as u32;
            DegreeIntent::new(min, max)
        }
        BeatType::Release => DegreeIntent::new(1, 2),
        BeatType::SideBranchLeaf => DegreeIntent::new(1, 1),
    };

    (density, degree)
}

// ── Landmark selection (forced, before natural rarity) ─────────────────────

/// Select required critical-path landmarks from compatible archetypes.
///
/// These are forced selections tracked separately from natural rarity evidence.
/// Returns a list of (archetype_index, beat_id) pairs.
fn select_forced_landmarks(
    seed: u64,
    preset: RichnessPreset,
    landmark_beats: &[(BeatId, u32)],
) -> Vec<ForcedLandmark> {
    let mut forced = Vec::new();
    let mut used = BTreeSet::new();
    let compat = compatible_archetypes(BeatType::LandmarkPeak);

    if compat.is_empty() {
        return forced;
    }

    for (i, &(beat_id, _beat_index)) in landmark_beats.iter().enumerate() {
        let Some(archetype) = compat
            .iter()
            .copied()
            .filter(|archetype| !used.contains(archetype))
            .min_by_key(|archetype| {
                u32_from_hash(&pacing_hash(
                    seed,
                    preset,
                    FieldTag::Fbm,
                    pacing_candidate_key(preset, beat_id.raw(), Some(*archetype)),
                ))
            })
        else {
            continue;
        };
        let val = u32_from_hash(&pacing_hash(
            seed,
            preset,
            FieldTag::Fbm,
            pacing_candidate_key(preset, beat_id.raw(), Some(archetype)),
        )) as usize;
        used.insert(archetype);

        // Determine the reason
        let reason = if i == 0 {
            LandmarkForceReason::PresetRequirement
        } else if i == 1 {
            // Alternate reasons for variety
            if val % 2 == 0 {
                LandmarkForceReason::GrandVolume
            } else {
                LandmarkForceReason::DenseSetpiece
            }
        } else {
            LandmarkForceReason::QuietNegativeSpace
        };

        forced.push(ForcedLandmark {
            archetype,
            reason,
            beat_id,
        });
    }

    forced
}

// ── Natural rarity selection ───────────────────────────────────────────────

/// Select an archetype using normalized integer rarity weights.
///
/// The weights are 70/25/5/1 for Common/Uncommon/Rare/Legendary.
/// Returns the selected archetype index.
fn select_by_rarity(
    seed: u64,
    preset: RichnessPreset,
    semantic_slot: u32,
    beat_type: BeatType,
) -> Option<(ArchetypeIndex, RarityClass)> {
    // First, select the rarity class with exact integer normalization.
    let hash = pacing_hash(
        seed,
        preset,
        FieldTag::WorleyF1,
        pacing_candidate_key(preset, semantic_slot, None),
    );
    let val = u32_from_hash(&hash);
    let bucket = val % RARITY_TOTAL_WEIGHT;

    let rarity = if bucket < COMMON_THRESHOLD {
        RarityClass::Common
    } else if bucket < UNCOMMON_THRESHOLD {
        RarityClass::Uncommon
    } else if bucket < RARE_THRESHOLD {
        RarityClass::Rare
    } else {
        RarityClass::Legendary
    };

    // Select only beat-compatible archetypes. A type absent from the slot's
    // compatible set falls back to that set's common candidates; the returned
    // class always matches the concrete selected archetype.
    let compatible = compatible_archetypes(beat_type);
    let matching: Vec<_> = compatible
        .iter()
        .copied()
        .filter(|idx| {
            RarityClass::from_tier(generated_content::ARCHETYPE_RARITY[idx.raw() as usize])
                == rarity
        })
        .collect();
    let candidates = if matching.is_empty() {
        compatible
            .iter()
            .copied()
            .filter(|idx| {
                generated_content::ARCHETYPE_RARITY[idx.raw() as usize]
                    == super::content_types::RarityTier::Common
            })
            .collect()
    } else {
        matching
    };
    let selected = candidates.into_iter().min_by_key(|archetype| {
        u32_from_hash(&pacing_hash(
            seed,
            preset,
            FieldTag::WorleyF1,
            pacing_candidate_key(preset, semantic_slot, Some(*archetype)),
        ))
    })?;
    let actual_rarity =
        RarityClass::from_tier(generated_content::ARCHETYPE_RARITY[selected.raw() as usize]);

    Some((selected, actual_rarity))
}

/// Select natural (non-forced) archetypes for side-branches and filler beats.
///
/// Respects: rare/legendary no-repeat, one-per-map cap, common repeats only
/// after compatible unused common candidates exhausted.
fn select_natural_archetypes(
    seed: u64,
    preset: RichnessPreset,
    slots: &[(BeatId, BeatType)],
    forced_archetypes: &BTreeSet<ArchetypeIndex>,
) -> (Vec<(ArchetypeIndex, RarityClass)>, RarityEvidence) {
    let mut selections = Vec::new();
    let mut evidence = RarityEvidence::new();
    let mut used_rare: BTreeSet<ArchetypeIndex> = BTreeSet::new();
    let mut used_legendary: BTreeSet<ArchetypeIndex> = BTreeSet::new();
    let mut used_common: BTreeSet<ArchetypeIndex> = BTreeSet::new();

    for &(beat_id, beat_type) in slots {
        let Some((archetype, rarity)) = select_by_rarity(seed, preset, beat_id.raw(), beat_type)
        else {
            continue;
        };
        let all_common: Vec<ArchetypeIndex> = compatible_archetypes(beat_type)
            .into_iter()
            .filter(|idx| {
                generated_content::ARCHETYPE_RARITY[idx.raw() as usize]
                    == super::content_types::RarityTier::Common
            })
            .collect();

        // Check caps
        let can_select = match rarity {
            RarityClass::Rare => {
                if used_rare.contains(&archetype)
                    || !used_rare.is_empty()
                    || forced_archetypes.contains(&archetype)
                {
                    false
                } else {
                    true
                }
            }
            RarityClass::Legendary => {
                if used_legendary.contains(&archetype)
                    || !used_legendary.is_empty()
                    || forced_archetypes.contains(&archetype)
                {
                    false
                } else {
                    true
                }
            }
            RarityClass::Common => {
                if forced_archetypes.contains(&archetype) {
                    false
                } else if used_common.contains(&archetype) {
                    // Only repeat after all compatible unused commons exhausted
                    let unused: Vec<_> = all_common
                        .iter()
                        .filter(|&&c| !used_common.contains(&c) && !forced_archetypes.contains(&c))
                        .collect();
                    unused.is_empty()
                } else {
                    true
                }
            }
            RarityClass::Uncommon => !forced_archetypes.contains(&archetype),
        };

        if can_select {
            match rarity {
                RarityClass::Rare => {
                    used_rare.insert(archetype);
                }
                RarityClass::Legendary => {
                    used_legendary.insert(archetype);
                }
                RarityClass::Common => {
                    used_common.insert(archetype);
                }
                _ => {}
            }
            evidence.record(rarity);
            selections.push((archetype, rarity));
        } else {
            // Fallback: pick an unused common archetype
            let unused_common: Vec<ArchetypeIndex> = all_common
                .iter()
                .filter(|&&c| !used_common.contains(&c) && !forced_archetypes.contains(&c))
                .copied()
                .collect();

            let fallback_candidates = if unused_common.is_empty() {
                // All compatible commons used — a repeat is now permitted.
                all_common
            } else {
                unused_common
            };
            let Some(fallback) = fallback_candidates.into_iter().min_by_key(|archetype| {
                u32_from_hash(&pacing_hash(
                    seed,
                    preset,
                    FieldTag::WorleyF1,
                    pacing_candidate_key(preset, beat_id.raw(), Some(*archetype)),
                ))
            }) else {
                continue;
            };

            used_common.insert(fallback);
            evidence.record(RarityClass::Common);
            selections.push((fallback, RarityClass::Common));
        }
    }

    (selections, evidence)
}

// ── Branch payoff generation ───────────────────────────────────────────────

/// Generate branch intents with exactly one observable payoff per leaf.
fn generate_branch_payoffs(
    seed: u64,
    preset: RichnessPreset,
    parent_beats: &[BeatId],
    leaf_beats: &[BeatId],
) -> BTreeMap<BranchId, BranchPayoff> {
    let mut payoffs = BTreeMap::new();

    for (i, &leaf_beat) in leaf_beats.iter().enumerate() {
        let i = i as u32;
        let hash = pacing_hash(
            seed,
            preset,
            FieldTag::WorleyF2,
            pacing_candidate_key(preset, i, None),
        );
        let val = u32_from_hash(&hash);

        let branch_id = BranchId::new(i);
        let payoff_type = match val % 4 {
            0 => PayoffType::Shortcut,
            1 => PayoffType::DiscoveryLandmark,
            2 => PayoffType::LoreMarker,
            _ => PayoffType::AuthoredTreasureTableau,
        };

        let parent_idx = (val as usize / 4) % parent_beats.len().max(1);
        let from_beat = parent_beats[parent_idx];

        // A shortcut payoff carries concrete later-to-earlier endpoints; all
        // other payoff forms still identify their branch origin as a
        // downstream-realizable observable record.
        let shortcut_endpoints = if payoff_type == PayoffType::Shortcut && parent_beats.len() >= 2 {
            let later_index = 1 + (val as usize % (parent_beats.len() - 1));
            let earlier_index = (val as usize / 4) % later_index;
            Some((parent_beats[later_index], parent_beats[earlier_index]))
        } else {
            None
        };

        payoffs.insert(
            branch_id,
            BranchPayoff {
                branch_id,
                payoff_type,
                from_beat,
                to_beat: Some(leaf_beat),
                shortcut_endpoints,
                observable: true,
            },
        );
    }

    payoffs
}

// ── Shortcut intent generation (Moderate/Rich) ─────────────────────────────

/// Generate backward shortcut intents for Moderate and Rich presets.
///
/// Returns at least one backward shortcut that connects later-to-earlier
/// progression without bypassing any unvisited mandatory beat.
fn generate_shortcut_intents(
    seed: u64,
    preset: RichnessPreset,
    beat_order: &[BeatId],
    beats: &BTreeMap<BeatId, Beat>,
) -> Vec<ShortcutIntent> {
    // Shortcuts only for Moderate and Rich
    match preset {
        RichnessPreset::Sparse => return Vec::new(),
        _ => {}
    }

    let min_shortcuts = match preset {
        RichnessPreset::Moderate => 1,
        RichnessPreset::Rich => 2,
        _ => 0,
    };

    let mut shortcuts = Vec::new();

    // Need at least 3 beats for a meaningful shortcut (later → earlier)
    if beat_order.len() < 3 {
        return shortcuts;
    }

    for i in 0..min_shortcuts {
        let hash = pacing_hash(
            seed,
            preset,
            FieldTag::FbmDomainWarpX,
            pacing_candidate_key(preset, i, None),
        );
        let val = u32_from_hash(&hash);

        // Pick a later beat and an earlier beat
        let n = beat_order.len() as u32;
        if n < 2 {
            continue;
        }

        let from_idx = 1 + (val % (n - 1)) as usize; // at least beat 1
        let to_idx = (val as usize / 4) % from_idx; // strictly earlier

        let from_beat = beat_order[from_idx];
        let to_beat = beat_order[to_idx];

        // Validate: from must come strictly after to in progression
        let from_beat_rec = match beats.get(&from_beat) {
            Some(b) => b,
            None => continue,
        };
        let to_beat_rec = match beats.get(&to_beat) {
            Some(b) => b,
            None => continue,
        };

        if from_beat_rec.progression > to_beat_rec.progression {
            shortcuts.push(ShortcutIntent {
                id: ShortcutId::new(i),
                from_beat,
                to_beat,
            });
        }
    }

    shortcuts
}

// ── Blueprint construction ─────────────────────────────────────────────────

/// Build the complete pacing blueprint from a resolved request.
///
/// Theme is never consulted during construction — the blueprint is identical
/// for all three themes given the same seed and preset.
pub(crate) fn build_pacing_blueprint(
    resolved: &ResolvedRichnessRequestV1,
) -> Result<PacingBlueprint, RichnessError> {
    let seed = resolved.seed();
    let preset = resolved.preset();
    let _theme = resolved.theme(); // Intentionally unused — theme independence
    let landmark_count = resolved.critical_path_landmarks().value();

    // ── 1. Derive beats ────────────────────────────────────────────────
    let mut beats = BTreeMap::new();
    let mut beat_order = Vec::new();
    let mut next_id = 0u32;
    let mut next_prog = ProgressionOrder::FIRST;
    let mut archetype_requests = BTreeMap::new();
    let mut next_req_id = 0u32;

    // Helper to create a beat
    let mut create_beat = |beat_type: BeatType,
                           on_critical: bool,
                           density: DensityClass,
                           degree: DegreeIntent|
     -> BeatId {
        let id = BeatId::new(next_id);
        next_id += 1;
        let prog = next_prog;
        if on_critical {
            next_prog = next_prog.next();
        }

        let beat = Beat {
            id,
            beat_type,
            requests: Vec::new(),
            density,
            degree,
            progression: prog,
            on_critical_path: on_critical,
            is_grand_volume: false,
            is_quiet_negative_space: false,
            is_dense_setpiece: false,
        };
        beats.insert(id, beat);
        if on_critical {
            beat_order.push(id);
        }
        id
    };

    // Plan all beat densities first to ensure alternation
    let mut prev_density: Option<DensityClass> = None;

    // Entrance beat
    let (entrance_density, entrance_degree) =
        derive_beat_properties(seed, preset, 0, BeatType::Entrance, prev_density);
    prev_density = Some(entrance_density);

    // Optionally a Descent beat for Moderate/Rich
    let has_descent = matches!(preset, RichnessPreset::Moderate | RichnessPreset::Rich);
    let (desc_density, desc_degree) = if has_descent {
        let d = derive_beat_properties(seed, preset, 100, BeatType::Descent, prev_density);
        prev_density = Some(d.0);
        d
    } else {
        (DensityClass::Transition, DegreeIntent::new(0, 0))
    };

    // Landmark beats
    let mut landmark_densities: Vec<(DensityClass, DegreeIntent, u32)> = Vec::new();
    for i in 0..landmark_count {
        let d = derive_beat_properties(seed, preset, 1 + i, BeatType::LandmarkPeak, prev_density);
        prev_density = Some(d.0);
        landmark_densities.push((d.0, d.1, i));
    }

    // Release beat
    let (release_density, release_degree) = derive_beat_properties(
        seed,
        preset,
        1 + landmark_count,
        BeatType::Release,
        prev_density,
    );

    // Now create all beats
    let _entrance_id = create_beat(BeatType::Entrance, true, entrance_density, entrance_degree);

    if has_descent {
        let _desc_id = create_beat(BeatType::Descent, true, desc_density, desc_degree);
    }

    let mut landmark_beats = Vec::new();
    for (lm_density, lm_degree, i) in landmark_densities {
        let lm_id = create_beat(BeatType::LandmarkPeak, true, lm_density, lm_degree);
        landmark_beats.push((lm_id, i));
    }

    let _release_id = create_beat(BeatType::Release, true, release_density, release_degree);

    // ── 2. Select forced landmarks ─────────────────────────────────────
    let forced_landmarks = select_forced_landmarks(seed, preset, &landmark_beats);
    let mut forced_archetypes: BTreeSet<ArchetypeIndex> = BTreeSet::new();
    let mut critical_path_landmarks = Vec::new();

    for fl in &forced_landmarks {
        forced_archetypes.insert(fl.archetype);
        let lm_id = LandmarkId::new(fl.beat_id.raw());
        critical_path_landmarks.push(lm_id);

        // Create archetype request for this forced landmark
        let req_id = ArchetypeRequestId::new(next_req_id);
        next_req_id += 1;

        if let Some(beat) = beats.get_mut(&fl.beat_id) {
            beat.requests.push(req_id);
            // Mark beat invariants based on reason
            match fl.reason {
                LandmarkForceReason::GrandVolume => beat.is_grand_volume = true,
                LandmarkForceReason::DenseSetpiece => beat.is_dense_setpiece = true,
                LandmarkForceReason::QuietNegativeSpace => beat.is_quiet_negative_space = true,
                LandmarkForceReason::PresetRequirement => {
                    // First landmark is grand volume by default
                    if critical_path_landmarks.len() == 1 {
                        beat.is_grand_volume = true;
                    }
                }
            }
        }

        let req = ArchetypeRequest {
            id: req_id,
            archetype: fl.archetype,
            beat_id: fl.beat_id,
            zone_id: ZoneId::new(0), // assigned later
            degree: DegreeIntent::new(1, 3),
            forced: true,
            rarity_class: RarityClass::from_tier(
                generated_content::ARCHETYPE_RARITY[fl.archetype.raw() as usize],
            ),
            progression: ProgressionOrder::new(
                beats
                    .get(&fl.beat_id)
                    .map(|b| b.progression.raw())
                    .unwrap_or(0),
            ),
            density: beats
                .get(&fl.beat_id)
                .map(|b| b.density)
                .unwrap_or(DensityClass::Transition),
        };
        archetype_requests.insert(req_id, req);
    }

    // ── 3. Natural rarity selections for remaining beats ───────────────
    // Count how many non-landmark beats need filling
    let non_landmark_beats: Vec<BeatId> = beat_order
        .iter()
        .filter(|bid| !landmark_beats.iter().any(|(lbid, _)| lbid == *bid))
        .copied()
        .collect();

    let natural_slots: Vec<_> = non_landmark_beats
        .iter()
        .filter_map(|beat_id| beats.get(beat_id).map(|beat| (*beat_id, beat.beat_type)))
        .collect();

    let (natural_selections, natural_rarity_evidence) =
        select_natural_archetypes(seed, preset, &natural_slots, &forced_archetypes);

    for (i, bid) in non_landmark_beats.iter().enumerate() {
        if i < natural_selections.len() {
            let (archetype, rarity) = natural_selections[i];
            let req_id = ArchetypeRequestId::new(next_req_id);
            next_req_id += 1;

            if let Some(beat) = beats.get_mut(bid) {
                beat.requests.push(req_id);
            }

            let req = ArchetypeRequest {
                id: req_id,
                archetype,
                beat_id: *bid,
                zone_id: ZoneId::new(0), // assigned later
                degree: DegreeIntent::new(1, 2),
                forced: false,
                rarity_class: rarity,
                progression: beats
                    .get(bid)
                    .map(|b| b.progression)
                    .unwrap_or(ProgressionOrder::FIRST),
                density: beats
                    .get(bid)
                    .map(|b| b.density)
                    .unwrap_or(DensityClass::Transition),
            };
            archetype_requests.insert(req_id, req);
        }
    }

    // ── 4. Side-branch payoffs ─────────────────────────────────────────
    let branch_count = match preset {
        RichnessPreset::Sparse => 1,
        RichnessPreset::Moderate => 2,
        RichnessPreset::Rich => 4,
    };

    let parent_beats: Vec<BeatId> = beat_order.clone();
    let mut leaf_beats = Vec::new();
    for _ in 0..branch_count {
        let leaf_id = BeatId::new(next_id);
        next_id += 1;
        beats.insert(
            leaf_id,
            Beat {
                id: leaf_id,
                beat_type: BeatType::SideBranchLeaf,
                requests: Vec::new(),
                density: DensityClass::Transition,
                degree: DegreeIntent::new(1, 1),
                progression: next_prog,
                on_critical_path: false,
                is_grand_volume: false,
                is_quiet_negative_space: false,
                is_dense_setpiece: false,
            },
        );
        leaf_beats.push(leaf_id);
    }
    let branch_payoffs = generate_branch_payoffs(seed, preset, &parent_beats, &leaf_beats);

    // ── 5. Mandatory edges ─────────────────────────────────────────────
    let mut mandatory_edges = Vec::new();
    for w in beat_order.windows(2) {
        mandatory_edges.push(MandatoryEdge {
            from_beat: w[0],
            to_beat: w[1],
        });
    }

    // ── 6. Shortcut intents ────────────────────────────────────────────
    let shortcut_intents = generate_shortcut_intents(seed, preset, &beat_order, &beats);

    // ── 8. Apply invariants ────────────────────────────────────────────
    // Ensure grand-volume, quiet-negative-space, and dense-setpiece flags
    let mut grand_volume = false;
    let mut quiet_negative = false;
    let mut dense_setpiece = false;

    for beat in beats.values() {
        if beat.is_grand_volume {
            grand_volume = true;
        }
        if beat.is_quiet_negative_space {
            quiet_negative = true;
        }
        if beat.is_dense_setpiece {
            dense_setpiece = true;
        }
    }

    // If any invariant is missing, assign it to appropriate beats
    if !grand_volume {
        // Assign to the first landmark beat
        for (lbid, _) in &landmark_beats {
            if let Some(beat) = beats.get_mut(lbid) {
                if !beat.is_grand_volume && !beat.is_dense_setpiece {
                    beat.is_grand_volume = true;
                    grand_volume = true;
                    break;
                }
            }
        }
    }

    if !quiet_negative {
        // Assign to a non-landmark beat on the critical path
        for bid in &beat_order {
            if let Some(beat) = beats.get_mut(bid) {
                if !beat.is_grand_volume && !beat.is_quiet_negative_space && !beat.is_dense_setpiece
                {
                    beat.is_quiet_negative_space = true;
                    quiet_negative = true;
                    break;
                }
            }
        }
    }

    if !dense_setpiece {
        // Assign to the last landmark or release beat
        for (lbid, _) in landmark_beats.iter().rev() {
            if let Some(beat) = beats.get_mut(lbid) {
                if !beat.is_grand_volume && !beat.is_dense_setpiece {
                    beat.is_dense_setpiece = true;
                    dense_setpiece = true;
                    break;
                }
            }
        }
        if !dense_setpiece {
            if let Some(last) = beat_order.last() {
                if let Some(beat) = beats.get_mut(last) {
                    if !beat.is_dense_setpiece {
                        beat.is_dense_setpiece = true;
                        dense_setpiece = true;
                    }
                }
            }
        }
    }

    // ── 8. Zone assignment ─────────────────────────────────────────────
    // The complete ZoneBlueprint is the sole zone path. Build it after beat
    // flags are final so its concrete realization requirements agree exactly.
    let mut zone_blueprint =
        build_zone_blueprint(preset, resolved.zone_count().value(), &beat_order, &beats);
    // A branch leaf inherits its origin's zone, preserving local correlation
    // while making every semantic node visible to downstream composition.
    for payoff in branch_payoffs.values() {
        if let Some((leaf, zone_id)) = payoff
            .to_beat
            .zip(zone_blueprint.beat_zone_map.get(&payoff.from_beat).copied())
        {
            zone_blueprint.beat_zone_map.insert(leaf, zone_id);
            if let Some(zone) = zone_blueprint.zones.get_mut(&zone_id) {
                zone.semantic_slots.push(leaf);
                zone.semantic_slots.sort_unstable();
            }
        }
    }

    // Update archetype requests with their canonical zone assignments.
    let mut reqs_with_zones = BTreeMap::new();
    for (rid, req) in &archetype_requests {
        let mut req = req.clone();
        if let Some(zone_id) = zone_blueprint.beat_zone_map.get(&req.beat_id) {
            req.zone_id = *zone_id;
        }
        reqs_with_zones.insert(*rid, req);
    }

    // ── 9. Build blueprint ─────────────────────────────────────────────
    let blueprint = PacingBlueprint {
        preset,
        seed,
        beats,
        beat_order,
        zone_blueprint,
        archetype_requests: reqs_with_zones,
        critical_path_landmarks,
        forced_landmarks,
        natural_rarity_evidence,
        mandatory_edges,
        branch_payoffs,
        shortcut_intents,
        grand_volume_landmark_present: grand_volume,
        quiet_negative_space_present: quiet_negative,
        dense_setpiece_present: dense_setpiece,
    };

    // ── 10. Validate ───────────────────────────────────────────────────
    let errors = blueprint.validate();
    if !errors.is_empty() {
        return Err(RichnessError::new(
            RichnessErrorCode::SemanticInfeasible,
            seed,
            resolved.provenance().request_schema_revision.tag(),
            resolved.provenance().algorithm_revision.tag(),
            resolved.provenance().content_revision.tag(),
            resolved.provenance().preset_revision.tag(),
            resolved.provenance().theme_revision.tag(),
            resolved.provenance().asset_revision.tag(),
            resolved.provenance().convention_revision.tag(),
            "pacing_blueprint",
            RichnessErrorCategory::SemanticInfeasibility,
            format!("pacing blueprint validation failed: {:?}", errors),
        ));
    }

    Ok(blueprint)
}

// ── Theme-independence verification ────────────────────────────────────────

/// Verify that theme doesn't affect the blueprint.
///
/// Builds blueprints for all three themes with the same seed/preset and
/// checks that the archetype requests, beats, zones, landmarks, and payoffs
/// are identical.
pub(crate) fn verify_theme_independence(
    seed: u64,
    preset: RichnessPreset,
) -> Result<bool, RichnessError> {
    use super::request::{RichnessDocumentV1, RichnessTheme};

    let themes = [
        RichnessTheme::Ancient,
        RichnessTheme::Egyptian,
        RichnessTheme::Brutalist,
    ];
    let mut blueprints = Vec::new();

    for &theme in &themes {
        let doc = RichnessDocumentV1::new(seed, 2048, preset, theme).map_err(|e| {
            RichnessError::new(
                RichnessErrorCode::SemanticInfeasible,
                seed,
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "theme_independence",
                RichnessErrorCategory::SemanticInfeasibility,
                format!("failed to create document: {}", e),
            )
        })?;
        let resolved = ResolvedRichnessRequestV1::resolve(doc)?;
        let bp = build_pacing_blueprint(&resolved)?;
        blueprints.push(bp);
    }

    // All blueprints must be identical
    if blueprints.len() < 2 {
        return Ok(true);
    }

    for i in 1..blueprints.len() {
        // Compare key fields
        if blueprints[0].beats != blueprints[i].beats {
            return Ok(false);
        }
        if blueprints[0].beat_order != blueprints[i].beat_order {
            return Ok(false);
        }
        if blueprints[0].archetype_requests != blueprints[i].archetype_requests {
            return Ok(false);
        }
        if blueprints[0].critical_path_landmarks != blueprints[i].critical_path_landmarks {
            return Ok(false);
        }
        if blueprints[0].mandatory_edges != blueprints[i].mandatory_edges {
            return Ok(false);
        }
        if blueprints[0].branch_payoffs != blueprints[i].branch_payoffs {
            return Ok(false);
        }
        if blueprints[0].shortcut_intents != blueprints[i].shortcut_intents {
            return Ok(false);
        }
        if blueprints[0].zone_blueprint != blueprints[i].zone_blueprint {
            return Ok(false);
        }
        if blueprints[0] != blueprints[i] {
            return Ok(false);
        }
    }

    Ok(true)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::request::{RichnessDocumentV1, RichnessTheme};
    use super::*;

    // ── Helper ─────────────────────────────────────────────────────────

    fn make_resolved(
        seed: u64,
        preset: RichnessPreset,
        theme: RichnessTheme,
    ) -> ResolvedRichnessRequestV1 {
        let doc = RichnessDocumentV1::new(seed, 2048, preset, theme).unwrap();
        ResolvedRichnessRequestV1::resolve(doc).unwrap()
    }

    // ── Beat derivation tests ──────────────────────────────────────────

    #[test]
    fn build_blueprint_sparse_has_one_landmark() {
        let resolved = make_resolved(0, RichnessPreset::Sparse, RichnessTheme::Ancient);
        let bp = build_pacing_blueprint(&resolved).unwrap();
        assert_eq!(bp.critical_path_landmarks.len(), 1);
        assert!(bp.is_valid());
    }

    #[test]
    fn build_blueprint_moderate_has_two_landmarks() {
        let resolved = make_resolved(42, RichnessPreset::Moderate, RichnessTheme::Ancient);
        let bp = build_pacing_blueprint(&resolved).unwrap();
        assert_eq!(bp.critical_path_landmarks.len(), 2);
        assert!(bp.is_valid());
    }

    #[test]
    fn build_blueprint_rich_has_three_landmarks() {
        let resolved = make_resolved(99, RichnessPreset::Rich, RichnessTheme::Ancient);
        let bp = build_pacing_blueprint(&resolved).unwrap();
        assert_eq!(bp.critical_path_landmarks.len(), 3);
        assert!(bp.is_valid());
    }

    #[test]
    fn build_blueprint_deterministic() {
        let resolved = make_resolved(42, RichnessPreset::Moderate, RichnessTheme::Ancient);
        let bp1 = build_pacing_blueprint(&resolved).unwrap();
        let bp2 = build_pacing_blueprint(&resolved).unwrap();
        assert_eq!(bp1, bp2);
    }

    #[test]
    fn build_blueprint_different_seed_different_output() {
        let r1 = make_resolved(0, RichnessPreset::Sparse, RichnessTheme::Ancient);
        let r2 = make_resolved(1, RichnessPreset::Sparse, RichnessTheme::Ancient);
        let bp1 = build_pacing_blueprint(&r1).unwrap();
        let bp2 = build_pacing_blueprint(&r2).unwrap();
        assert_ne!(bp1, bp2);
    }

    // ── Theme independence test ────────────────────────────────────────

    #[test]
    fn theme_independence_same_seed_preset() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            for seed in &[0u64, 42, 99, 255] {
                let result = verify_theme_independence(*seed, *preset);
                assert!(
                    result.unwrap(),
                    "theme independence failed for seed={}, preset={:?}",
                    seed,
                    preset
                );
            }
        }
    }

    // ── Landmark count tests ───────────────────────────────────────────

    #[test]
    fn sparse_exactly_one_landmark() {
        for seed in &[0u64, 1, 42, 99, 255, 1024, u64::MAX] {
            let resolved = make_resolved(*seed, RichnessPreset::Sparse, RichnessTheme::Ancient);
            let bp = build_pacing_blueprint(&resolved).unwrap();
            assert_eq!(
                bp.critical_path_landmarks.len(),
                1,
                "seed {} produced {} landmarks",
                seed,
                bp.critical_path_landmarks.len()
            );
        }
    }

    #[test]
    fn moderate_exactly_two_landmarks() {
        for seed in &[0u64, 1, 42, 99, 255, 1024] {
            let resolved = make_resolved(*seed, RichnessPreset::Moderate, RichnessTheme::Ancient);
            let bp = build_pacing_blueprint(&resolved).unwrap();
            assert_eq!(
                bp.critical_path_landmarks.len(),
                2,
                "seed {} produced {} landmarks",
                seed,
                bp.critical_path_landmarks.len()
            );
        }
    }

    #[test]
    fn rich_exactly_three_landmarks() {
        for seed in &[0u64, 1, 42, 99, 255, 1024] {
            let resolved = make_resolved(*seed, RichnessPreset::Rich, RichnessTheme::Ancient);
            let bp = build_pacing_blueprint(&resolved).unwrap();
            assert_eq!(
                bp.critical_path_landmarks.len(),
                3,
                "seed {} produced {} landmarks",
                seed,
                bp.critical_path_landmarks.len()
            );
        }
    }

    // ── Landmark distinctness ──────────────────────────────────────────

    #[test]
    fn landmarks_are_distinct() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            for seed in &[0u64, 42, 99] {
                let resolved = make_resolved(*seed, *preset, RichnessTheme::Ancient);
                let bp = build_pacing_blueprint(&resolved).unwrap();
                let mut seen = BTreeSet::new();
                for lid in &bp.critical_path_landmarks {
                    assert!(
                        seen.insert(*lid),
                        "duplicate landmark {:?} in seed {}",
                        lid,
                        seed
                    );
                }
            }
        }
    }

    // ── Rarity cap tests ───────────────────────────────────────────────

    #[test]
    fn rare_cap_at_most_one() {
        for seed in &[0u64, 42, 99, 255] {
            let resolved = make_resolved(*seed, RichnessPreset::Rich, RichnessTheme::Ancient);
            let bp = build_pacing_blueprint(&resolved).unwrap();
            let natural_rare_count: u32 = bp
                .archetype_requests
                .values()
                .filter(|r| !r.forced && r.rarity_class == RarityClass::Rare)
                .count() as u32;
            assert!(
                natural_rare_count <= 1,
                "seed {} has {} natural rare selections (cap 1)",
                seed,
                natural_rare_count
            );
        }
    }

    #[test]
    fn legendary_cap_at_most_one() {
        for seed in &[0u64, 42, 99, 255] {
            let resolved = make_resolved(*seed, RichnessPreset::Rich, RichnessTheme::Ancient);
            let bp = build_pacing_blueprint(&resolved).unwrap();
            let natural_legendary_count: u32 = bp
                .archetype_requests
                .values()
                .filter(|r| !r.forced && r.rarity_class == RarityClass::Legendary)
                .count() as u32;
            assert!(
                natural_legendary_count <= 1,
                "seed {} has {} natural legendary selections (cap 1)",
                seed,
                natural_legendary_count
            );
        }
    }

    // ── Shortcut intent tests ──────────────────────────────────────────

    #[test]
    fn sparse_has_no_shortcuts() {
        let resolved = make_resolved(0, RichnessPreset::Sparse, RichnessTheme::Ancient);
        let bp = build_pacing_blueprint(&resolved).unwrap();
        assert!(bp.shortcut_intents.is_empty());
    }

    #[test]
    fn moderate_has_at_least_one_shortcut() {
        for seed in &[0u64, 42, 99, 255] {
            let resolved = make_resolved(*seed, RichnessPreset::Moderate, RichnessTheme::Ancient);
            let bp = build_pacing_blueprint(&resolved).unwrap();
            assert!(
                !bp.shortcut_intents.is_empty(),
                "seed {} moderate has no shortcuts",
                seed
            );
        }
    }

    #[test]
    fn rich_has_at_least_one_shortcut() {
        for seed in &[0u64, 42, 99, 255] {
            let resolved = make_resolved(*seed, RichnessPreset::Rich, RichnessTheme::Ancient);
            let bp = build_pacing_blueprint(&resolved).unwrap();
            assert!(
                !bp.shortcut_intents.is_empty(),
                "seed {} rich has no shortcuts",
                seed
            );
        }
    }

    #[test]
    fn shortcuts_connect_later_to_earlier() {
        for preset in &[RichnessPreset::Moderate, RichnessPreset::Rich] {
            for seed in &[0u64, 42, 99] {
                let resolved = make_resolved(*seed, *preset, RichnessTheme::Ancient);
                let bp = build_pacing_blueprint(&resolved).unwrap();
                for shortcut in &bp.shortcut_intents {
                    let from_beat = bp.beats.get(&shortcut.from_beat).unwrap();
                    let to_beat = bp.beats.get(&shortcut.to_beat).unwrap();
                    assert!(
                        from_beat.progression > to_beat.progression,
                        "shortcut {:?} does not go later->earlier (seed={})",
                        shortcut.id,
                        seed
                    );
                }
            }
        }
    }

    // ── Invariant flag tests ───────────────────────────────────────────

    #[test]
    fn blueprint_has_grand_volume_landmark() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            for seed in &[0u64, 42, 255] {
                let resolved = make_resolved(*seed, *preset, RichnessTheme::Ancient);
                let bp = build_pacing_blueprint(&resolved).unwrap();
                assert!(
                    bp.grand_volume_landmark_present,
                    "seed {} {:?} missing grand volume",
                    seed, preset
                );
            }
        }
    }

    #[test]
    fn blueprint_has_quiet_negative_space() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            for seed in &[0u64, 42, 255] {
                let resolved = make_resolved(*seed, *preset, RichnessTheme::Ancient);
                let bp = build_pacing_blueprint(&resolved).unwrap();
                assert!(
                    bp.quiet_negative_space_present,
                    "seed {} {:?} missing quiet negative space",
                    seed, preset
                );
            }
        }
    }

    #[test]
    fn blueprint_has_dense_setpiece() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            for seed in &[0u64, 42, 255] {
                let resolved = make_resolved(*seed, *preset, RichnessTheme::Ancient);
                let bp = build_pacing_blueprint(&resolved).unwrap();
                assert!(
                    bp.dense_setpiece_present,
                    "seed {} {:?} missing dense setpiece",
                    seed, preset
                );
            }
        }
    }

    // ── Beat reachability ──────────────────────────────────────────────

    #[test]
    fn mandatory_edges_cover_critical_path() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            for seed in &[0u64, 42, 255] {
                let resolved = make_resolved(*seed, *preset, RichnessTheme::Ancient);
                let bp = build_pacing_blueprint(&resolved).unwrap();
                // Every adjacent pair in beat_order must have a mandatory edge
                for w in bp.beat_order.windows(2) {
                    let has_edge = bp
                        .mandatory_edges
                        .iter()
                        .any(|e| e.from_beat == w[0] && e.to_beat == w[1]);
                    assert!(
                        has_edge,
                        "seed {} missing mandatory edge {:?} -> {:?}",
                        seed, w[0], w[1]
                    );
                }
            }
        }
    }

    // ── Dense/quiet alternation ────────────────────────────────────────

    #[test]
    fn dense_quiet_alternates_on_critical_path() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            for seed in &[0u64, 42, 255] {
                let resolved = make_resolved(*seed, *preset, RichnessTheme::Ancient);
                let bp = build_pacing_blueprint(&resolved).unwrap();
                let errors = bp.validate();
                // Allow transition between same types
                let alternation_errors: Vec<_> = errors
                    .iter()
                    .filter(|e| e.contains("non-alternation"))
                    .collect();
                assert!(
                    alternation_errors.is_empty(),
                    "seed {} has alternation errors: {:?}",
                    seed,
                    alternation_errors
                );
            }
        }
    }

    // ── Branch payoff completeness ─────────────────────────────────────

    #[test]
    fn branch_payoffs_are_observable() {
        for preset in &[
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ] {
            for seed in &[0u64, 42, 255] {
                let resolved = make_resolved(*seed, *preset, RichnessTheme::Ancient);
                let bp = build_pacing_blueprint(&resolved).unwrap();
                for (bid, payoff) in &bp.branch_payoffs {
                    assert!(
                        payoff.observable,
                        "seed {} branch {:?} has unobservable payoff",
                        seed, bid
                    );
                    assert!(
                        !payoff.payoff_type.is_label_only(),
                        "seed {} branch {:?} has label-only payoff",
                        seed,
                        bid
                    );
                }
            }
        }
    }

    // ── Zone and branch integration ───────────────────────────────────

    #[test]
    fn pacing_emits_complete_zone_blueprint_and_concrete_branch_leaves() {
        for &preset in RichnessPreset::ALL {
            let resolved = make_resolved(42, preset, RichnessTheme::Ancient);
            let blueprint = build_pacing_blueprint(&resolved).unwrap();
            assert!(blueprint.zone_blueprint.is_valid());
            assert_eq!(
                blueprint.zone_blueprint.transitions.len(),
                blueprint.zone_blueprint.zones.len().saturating_sub(1)
            );
            for beat_id in &blueprint.beat_order {
                assert!(blueprint.zone_blueprint.beat_zone_map.contains_key(beat_id));
            }
            for payoff in blueprint.branch_payoffs.values() {
                let leaf = payoff.to_beat.unwrap();
                assert_eq!(
                    blueprint.beats.get(&leaf).unwrap().beat_type,
                    BeatType::SideBranchLeaf
                );
            }
        }
    }

    // ── Exhaustive sweep test ──────────────────────────────────────────

    #[test]
    fn deterministic_sweep_broad_seeds() {
        let seeds: Vec<u64> = (0..32).collect();
        let presets = [
            RichnessPreset::Sparse,
            RichnessPreset::Moderate,
            RichnessPreset::Rich,
        ];

        for &preset in &presets {
            for &seed in &seeds {
                let resolved = make_resolved(seed, preset, RichnessTheme::Ancient);
                let bp = build_pacing_blueprint(&resolved);
                match bp {
                    Ok(bp) => {
                        assert!(
                            bp.is_valid(),
                            "seed {} {:?} produced invalid blueprint",
                            seed,
                            preset
                        );
                    }
                    Err(e) => {
                        panic!(
                            "seed {} {:?} failed to build blueprint: {}",
                            seed, preset, e
                        );
                    }
                }
            }
        }
    }

    // ── Forced rarity test ─────────────────────────────────────────────

    #[test]
    fn forced_landmarks_tracked_separately_from_natural() {
        for seed in &[0u64, 42, 99] {
            let resolved = make_resolved(*seed, RichnessPreset::Rich, RichnessTheme::Ancient);
            let bp = build_pacing_blueprint(&resolved).unwrap();

            // Forced landmarks should not be counted in natural rarity
            let forced_count = bp.forced_landmarks.len() as u32;
            let natural_total = bp.natural_rarity_evidence.total_natural;

            // Natural + forced should not exceed total archetype requests
            let total_requests = bp.archetype_requests.len() as u32;
            assert!(
                natural_total + forced_count <= total_requests,
                "seed {}: natural({}) + forced({}) > total({})",
                seed,
                natural_total,
                forced_count,
                total_requests
            );

            // Verify forced landmarks have `forced: true` in their requests
            for fl in &bp.forced_landmarks {
                let reqs: Vec<_> = bp
                    .archetype_requests
                    .values()
                    .filter(|r| r.beat_id == fl.beat_id && r.forced)
                    .collect();
                assert!(
                    !reqs.is_empty(),
                    "seed {} forced landmark at beat {:?} missing forced request",
                    seed,
                    fl.beat_id
                );
            }
        }
    }
}
