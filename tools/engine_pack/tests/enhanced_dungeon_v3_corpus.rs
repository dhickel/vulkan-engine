//! Phase 09 — Enhanced V3 Corpus Package Validation
//!
//! Validates every frozen corpus entry through the full publication pipeline:
//! generation → compilation → publication → strict-reload → budget measurement.
//! Uses the engine_pack `build_v3_package` transaction system.
//!
//! # Run
//!
//! ```bash
//! cargo test -p engine_pack --test enhanced_dungeon_v3_corpus -- --nocapture
//! ```
//!
//! # Requirements
//!
//! - ericw-tools 2.0.0-alpha3 at ~/.local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin/

use bsp::BspLoader;
use bsp_generator::enhanced_v3::{self, V3Config, V3Preset};
use engine_pack::enhanced_dungeon_v3::build_v3_package;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// ── Corpus entries ────────────────────────────────────────────────────────

fn corpus_entries() -> Vec<(u64, V3Preset, u32)> {
    // Authoritative EnhancedV3 corpus: Sparse/Moderate/Rich × seeds
    // 0/42/99/255; Sparse+Moderate at 2048, Rich at 3072 (specification
    // bsp-dungeon-generation.md, settled decision 7).
    let seeds = [0u64, 42, 99, 255];
    let mut entries = Vec::new();
    for &seed in &seeds {
        entries.push((seed, V3Preset::Sparse, 2048));
        entries.push((seed, V3Preset::Moderate, 2048));
        entries.push((seed, V3Preset::Rich, 3072));
    }
    entries
}

// ── Paths ─────────────────────────────────────────────────────────────────

fn ericw_tools_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn debug_reports_dir() -> PathBuf {
    repo_root().join(".internal-dev/debug_reports/enhanced-v3-production")
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

// ── Evidence types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusPackageEvidence {
    timestamp: String,
    compiler_available: bool,
    entries: BTreeMap<String, PackageEntryEvidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackageEntryEvidence {
    id: String,
    seed: u64,
    preset: String,
    extent: u32,
    published: bool,
    bsp_sha256: Option<String>,
    lit_sha256: Option<String>,
    map_sha256: Option<String>,
    metadata_sha256: Option<String>,
    bsp_faces: Option<u32>,
    bsp_entities: Option<u32>,
    #[serde(default)]
    static_batches: Option<u32>,
    budget_pass: bool,
    errors: Vec<String>,
}

/// Strictly load and extract a compiled package. BSP face/entity counts and
/// static batches all come from the same validated representation.
fn strict_compiled_measurement(
    bsp_bytes: &[u8],
    lit_bytes: &[u8],
    palette_bytes: &[u8],
    wad_name: String,
    wad_bytes: Vec<u8>,
) -> Result<(u32, u32, u32), String> {
    let world = BspLoader::load(
        bsp_bytes,
        &bsp::LoadOptions {
            strict: true,
            palette: Some(palette_bytes.to_vec()),
            lit_data: Some(lit_bytes.to_vec()),
            wad_archives: vec![(wad_name.clone(), wad_bytes.clone())],
            source_identity: "enhanced-v3-package-corpus".to_string(),
            ..Default::default()
        },
    )
    .map_err(|report| format!("strict load for batch measurement: {report}"))?;
    if !world.diagnostics.is_empty() {
        return Err(format!("strict load diagnostics: {:?}", world.diagnostics));
    }
    let faces = u32::try_from(world.faces.len()).map_err(|_| "face count exceeds u32")?;
    let entities = u32::try_from(world.entities.len()).map_err(|_| "entity count exceeds u32")?;
    let extracted = bsp::extract::extract(bsp::BspExtractionRequest {
        world,
        palette: Some(bsp::resources::decode_palette(palette_bytes)),
        wad_archives: vec![(wad_name, wad_bytes)],
        strict: true,
        ..Default::default()
    })
    .map_err(|report| format!("strict extraction for batch measurement: {report}"))?;
    let static_batches = u32::try_from(extracted.render_batches.len())
        .map_err(|_| "static batch count exceeds u32".to_string())?;
    Ok((faces, entities, static_batches))
}

// ── Test: build and validate all corpus packages ─────────────────────────

#[test]
fn build_all_corpus_packages() {
    std::fs::create_dir_all(debug_reports_dir()).unwrap();

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string();

    let compiler_available = tools_available(&ericw_tools_dir());
    let mut evidence = CorpusPackageEvidence {
        timestamp,
        compiler_available,
        entries: BTreeMap::new(),
    };

    for (seed, preset, extent) in corpus_entries() {
        let id = format!("v3-{}-seed-{}", preset.tag(), seed);
        let mut errors: Vec<String> = Vec::new();

        // Generate for source metadata
        let config = V3Config::new(seed, preset, extent).expect("valid config");
        let output = enhanced_v3::run_pipeline(&config).expect("generation");
        let map_sha256 = sha256_hex(output.map_text.as_bytes());
        let meta_json = serde_json::to_string(&output.metadata).unwrap();
        let meta_sha256 = sha256_hex(meta_json.as_bytes());

        let source_faces = output.metadata.actual_faces();
        let source_entities = output.metadata.actual_entities();

        // Source budgets are faces and entities. Source brush count is not a
        // renderer batch count; static batches are measured below from strict
        // extraction of the compiled publication.
        assert!(
            source_faces < 10000,
            "{id}: source faces {source_faces} exceeds 10000"
        );
        assert!(
            source_entities < 300,
            "{id}: source entities {source_entities} exceeds 300"
        );

        // If compiler available, do full publication
        let (published, bsp_hash, lit_hash, bsp_faces, bsp_entities, static_batches) =
            if compiler_available {
                let out_dir = tempfile::TempDir::new().expect("tempdir");
                let tool_path = ericw_tools_dir();
                let name = &id;

                match build_v3_package(
                    seed,
                    preset,
                    extent,
                    &out_dir.path().join("package"),
                    Some(&tool_path),
                    name,
                    None,
                ) {
                    Ok(result) => {
                        let target = match &result {
                            engine_pack::BuildV3Result::Published { target, .. } => target.clone(),
                            engine_pack::BuildV3Result::Unchanged { target, .. } => target.clone(),
                        };

                        // Strict-validate the published closure
                        let bsp_path = target.join(format!("{name}.bsp"));
                        let lit_path = target.join(format!("{name}.lit"));
                        let map_path = target.join(format!("{name}.map"));
                        let metadata_path = target.join("metadata.json");
                        let manifest_path = target.join(format!("{name}.manifest.toml"));
                        let wad_file = target.join("cc0_dungeon_v2.wad");
                        let palette_file = target.join("palette.lmp");

                        // All required files present
                        let mut missing = Vec::new();
                        for (path, label) in &[
                            (&bsp_path, ".bsp"),
                            (&lit_path, ".lit"),
                            (&map_path, ".map"),
                            (&metadata_path, "metadata.json"),
                            (&manifest_path, ".manifest.toml"),
                            (&wad_file, "WAD"),
                            (&palette_file, "palette.lmp"),
                        ] {
                            if !path.exists() {
                                missing.push(*label);
                            }
                        }
                        assert!(
                            missing.is_empty(),
                            "{id}: missing published files: {missing:?}"
                        );

                        // BSP2 magic
                        let bsp_bytes = std::fs::read(&bsp_path).expect("read bsp");
                        assert!(
                            bsp_bytes.len() >= 4 && &bsp_bytes[0..4] == b"BSP2",
                            "{id}: BSP2 magic invalid"
                        );

                        // QLIT magic
                        let lit_bytes = std::fs::read(&lit_path).expect("read lit");
                        assert!(
                            lit_bytes.len() >= 4 && &lit_bytes[0..4] == b"QLIT",
                            "{id}: QLIT magic invalid"
                        );

                        let bsp_h = sha256_hex(&bsp_bytes);
                        let lit_h = sha256_hex(&lit_bytes);

                        let palette_bytes = std::fs::read(&palette_file).expect("read palette");
                        let wad_bytes = std::fs::read(&wad_file).expect("read WAD");
                        let wad_name = wad_file
                            .file_name()
                            .and_then(|name| name.to_str())
                            .expect("WAD basename")
                            .to_string();
                        let (faces, entities, static_batches) = strict_compiled_measurement(
                            &bsp_bytes,
                            &lit_bytes,
                            &palette_bytes,
                            wad_name,
                            wad_bytes,
                        )
                        .unwrap_or_else(|error| panic!("{id}: {error}"));

                        // Budget checks
                        assert!(faces < 10000, "{id}: compiled faces {faces} exceeds 10000");
                        assert!(
                            entities < 300,
                            "{id}: compiled entities {entities} exceeds 300"
                        );
                        assert!(
                            static_batches < 500,
                            "{id}: strict extracted static batches {static_batches} exceeds 500"
                        );

                        (
                            true,
                            Some(bsp_h),
                            Some(lit_h),
                            Some(faces),
                            Some(entities),
                            Some(static_batches),
                        )
                    }
                    Err(e) => {
                        errors.push(format!("publication failed: {e:?}"));
                        (false, None, None, None, None, None)
                    }
                }
            } else {
                (false, None, None, None, None, None)
            };

        let budget_pass = source_faces < 10000
            && source_entities < 300
            && static_batches.is_some_and(|count| count < 500);

        evidence.entries.insert(
            id.clone(),
            PackageEntryEvidence {
                id,
                seed,
                preset: preset.tag().to_string(),
                extent,
                published,
                bsp_sha256: bsp_hash,
                lit_sha256: lit_hash,
                map_sha256: Some(map_sha256),
                metadata_sha256: Some(meta_sha256),
                bsp_faces,
                bsp_entities,
                static_batches,
                budget_pass,
                errors,
            },
        );
    }

    // Write evidence
    let evidence_path = debug_reports_dir().join("corpus-package-evidence.json");
    let evidence_json = serde_json::to_string_pretty(&evidence).unwrap();
    std::fs::write(&evidence_path, &evidence_json).unwrap();

    // Summary
    let published_count = evidence.entries.values().filter(|e| e.published).count();
    println!(
        "Corpus package validation: {}/{} entries published",
        published_count,
        evidence.entries.len()
    );

    for entry in evidence.entries.values() {
        let status = if entry.published {
            "PUBLISHED"
        } else if evidence.compiler_available {
            "FAILED"
        } else {
            "COMPILER_UNAVAILABLE"
        };
        println!(
            "  {}: {} (faces={:?}, entities={:?}, static_batches={:?})",
            entry.id, status, entry.bsp_faces, entry.bsp_entities, entry.static_batches
        );
    }

    if compiler_available {
        let failures = evidence
            .entries
            .values()
            .filter(|entry| !entry.published || !entry.budget_pass || !entry.errors.is_empty())
            .map(|entry| format!("{}: {:?}", entry.id, entry.errors))
            .collect::<Vec<_>>();
        assert!(
            failures.is_empty(),
            "corpus publication, strict extraction, or budget validation failed: {failures:?}"
        );
    }
}

// ── Source-level budget test (no compiler required) ──────────────────────

#[test]
fn source_budget_validation() {
    for (seed, preset, extent) in corpus_entries() {
        let config = V3Config::new(seed, preset, extent).expect("valid config");
        let output = enhanced_v3::run_pipeline(&config).expect("generation");

        let faces = output.metadata.actual_faces();
        let entities = output.metadata.actual_entities();
        let id = format!("v3-{}-seed-{}", preset.tag(), seed);

        assert!(faces < 10000, "{id}: faces {faces} >= 10000");
        assert!(entities < 300, "{id}: entities {entities} >= 300");
    }
}

// ── Determinism test ─────────────────────────────────────────────────────

#[test]
fn corpus_entries_are_deterministic() {
    for (seed, preset, extent) in corpus_entries() {
        let config = V3Config::new(seed, preset, extent).expect("valid config");
        let o1 = enhanced_v3::run_pipeline(&config).expect("run1");
        let o2 = enhanced_v3::run_pipeline(&config).expect("run2");

        assert_eq!(o1.map_text, o2.map_text);
        assert_eq!(o1.metadata, o2.metadata);
    }
}
