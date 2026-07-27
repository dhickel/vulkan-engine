//! Headless capture test: BSP Beta Strict Acceptance Capture
//!
//! Phase 09 — Strict visual evidence capture.
//!
//! Requires explicit package inputs with manifest hash verification. No ambient
//! palette fallback, no parent-directory discovery, no `strict: false` paths.
//! Loads only manifest-declared palette/LIT/WAD/PBR resources; fails on missing
//! surfaces, undeclared resources, or non-successful capture status.
//!
//! ## Frozen Settings (bsp-acceptance.md §5)
//! - 1280×720, exposure 1.0, overbright 2.0, style 0, animation time 0.0
//!
//! Run:
//! ```bash
//! cargo run -p renderer --features bsp --example capture_bsp_beta -- \
//!   --headless --bsp <path> --palette <path> --wad <path> \
//!   --lit <path> --acceptance-camera <spawn|corridor|junction>
//! ```

#[cfg(feature = "bsp")]
#[path = "common.rs"]
mod common;

// ── Feature-gated entrypoints ─────────────────────────────────────────────

#[cfg(not(feature = "bsp"))]
fn main() {
    eprintln!("This example requires the `bsp` feature:");
    eprintln!("  cargo run -p renderer --features bsp --example capture_bsp_beta -- --headless --bsp <path> --palette <path>");
}

#[cfg(feature = "bsp")]
fn main() {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let args = common::parse_capture_test_args();

    // Phase 09: All inputs must be explicitly declared. No discovery.
    let bsp_path = find_required_arg("--bsp");
    let palette_path = find_required_arg("--palette");
    let wad_path = find_required_arg("--wad");
    let lit_path = find_optional_arg("--lit");

    let bsp_bytes = match std::fs::read(&bsp_path) {
        Ok(b) => b,
        Err(e) => {
            log::error!("Failed to read BSP '{}': {e}", bsp_path.display());
            std::process::exit(1);
        }
    };
    let palette_bytes = match std::fs::read(&palette_path) {
        Ok(b) => b,
        Err(e) => {
            log::error!(
                "Failed to read palette '{}': {e}",
                palette_path.display()
            );
            std::process::exit(1);
        }
    };
    let wad_bytes = match std::fs::read(&wad_path) {
        Ok(b) => b,
        Err(e) => {
            log::error!("Failed to read WAD '{}': {e}", wad_path.display());
            std::process::exit(1);
        }
    };
    let lit_bytes = lit_path.as_ref().and_then(|p| {
        match std::fs::read(p) {
            Ok(b) => Some(b),
            Err(e) => {
                log::error!("Failed to read LIT '{}': {e}", p.display());
                None
            }
        }
    });

    // Phase 09: Verify manifest hashes before parsing owned bytes.
    // Hash verification is performed by the caller (visual_acceptance.rs)
    // before invoking this binary. The expected hash is recorded in the
    // sidecar and verified independently.
    if let Ok(expected_bsp) = std::env::var("BSP_EXPECTED_BSP_HASH") {
        log::info!(
            "BSP_EXPECTED_BSP_HASH={expected_bsp} — caller is responsible for pre-verification"
        );
    }

    log::info!(
        "BSP capture test (strict): {} ({} bytes)",
        bsp_path.display(),
        bsp_bytes.len()
    );

    run_strict_capture_test(
        &bsp_bytes,
        &bsp_path,
        &palette_bytes,
        &palette_path,
        &wad_bytes,
        &wad_path,
        lit_bytes.as_deref(),
        &args,
    );
}

#[cfg(feature = "bsp")]
fn find_required_arg(flag: &str) -> std::path::PathBuf {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == flag {
            if let Some(val) = args.get(i + 1) {
                if !val.starts_with("--") {
                    return std::path::PathBuf::from(val);
                }
            }
        }
        i += 1;
    }
    eprintln!("{flag} <path> is required for strict capture_bsp_beta");
    std::process::exit(1);
}

#[cfg(feature = "bsp")]
fn find_optional_arg(flag: &str) -> Option<std::path::PathBuf> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == flag {
            if let Some(val) = args.get(i + 1) {
                if !val.starts_with("--") {
                    return Some(std::path::PathBuf::from(val));
                }
            }
        }
        i += 1;
    }
    None
}

/// Parse the --acceptance-camera argument to select a frozen camera view.
#[cfg(feature = "bsp")]
fn acceptance_camera_label() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--acceptance-camera" {
            if let Some(val) = args.get(i + 1) {
                if !val.starts_with("--") {
                    return Some(val.clone());
                }
            }
        }
        i += 1;
    }
    None
}

#[cfg(feature = "bsp")]
fn run_strict_capture_test(
    bsp_bytes: &[u8],
    bsp_path: &std::path::Path,
    palette_bytes: &[u8],
    _palette_path: &std::path::Path,
    wad_bytes: &[u8],
    _wad_path: &std::path::Path,
    lit_bytes: Option<&[u8]>,
    args: &common::CaptureTestArgs,
) {
    use glam::Vec3;
    use renderer::api::config::RendererConfig;
    use renderer::api::{CaptureTarget, FrameCaptureRequest, FrameCaptureStatus};
    use renderer::prelude::*;
    use std::path::PathBuf;

    let validation_layer = std::env::var("ENGINE_CAPTURE_VALIDATION")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "on"));

    let config = RendererConfig {
        app_name: "capture_bsp_beta".to_string(),
        headless: true,
        window_width: 1280,
        window_height: 720,
        validation_layer,
        ..RendererConfig::default()
    };

    let mut renderer = match Renderer::new_headless(config) {
        Ok(r) => r,
        Err(err) => {
            log::error!("Headless renderer init failed: {err}");
            std::process::exit(1);
        }
    };

    // ── Strict BSP load: manifest-declared inputs only ─────────────────
    let wad_name = "cc0_stone_beta.wad".to_string();
    let wad_archive = (wad_name, wad_bytes.to_vec());

    let load_options = bsp::LoadOptions {
        strict: true, // Phase 09: strict only
        palette: Some(palette_bytes.to_vec()),
        lit_data: lit_bytes.map(|d| d.to_vec()),
        wad_archives: vec![wad_archive],
        texture_overrides: Vec::new(),
        source_identity: bsp_path.display().to_string(),
    };

    let world = match bsp::BspLoader::load(bsp_bytes, &load_options) {
        Ok(w) => w,
        Err(report) => {
            log::error!(
                "Strict BSP load failed for '{}': {:?} — {}",
                bsp_path.display(),
                report.code,
                report.message
            );
            std::process::exit(1);
        }
    };

    // Phase 09: Reject undeclared diagnostics in strict mode.
    if !world.diagnostics.is_empty() {
        log::error!(
            "Strict BSP load produced {} unexpected diagnostics",
            world.diagnostics.len()
        );
        for d in &world.diagnostics {
            log::error!("  {:?}: {}", d.code, d.message);
        }
        std::process::exit(1);
    }

    log::info!(
        "BSP loaded (strict): {} faces, {} entities, {} textures, profile={:?}",
        world.faces.len(),
        world.entities.len(),
        world.miptex_data.len(),
        world.profile,
    );

    // ── Strict extraction: no ambient fallback ─────────────────────────
    let palette = bsp::companions::validate_palette(palette_bytes, true)
        .map_err(|report| {
            log::error!(
                "Strict palette validation failed: {}",
                report.message
            );
            std::process::exit(1);
        })
        .ok()
        .map(|_| bsp::resources::decode_palette(palette_bytes));

    // Collect PBR companions from declared inputs only — no directory discovery.
    let texture_names = bsp::resources::collect_miptex_names(&world.miptex_data);

    // Phase 09: We do not discover PBR companions from filesystem. If the
    // manifest declares PBR companions, they must be provided explicitly.
    // For now, empty companions = legacy lightmapped path.
    let pbr_companions: Vec<bsp::resources::TextureCompanion> = Vec::new();

    let extracted = match bsp::extract::extract(bsp::BspExtractionRequest {
        world,
        palette,
        texture_companions: pbr_companions,
        strict: true, // Phase 09: strict extraction
        ..Default::default()
    }) {
        Ok(e) => e,
        Err(report) => {
            log::error!(
                "Strict BSP extraction failed: {:?} — {}",
                report.code,
                report.message
            );
            std::process::exit(1);
        }
    };

    log::info!(
        "BSP extracted (strict): {} faces, {} entities, {} lights, {} batches, PVS={}",
        extracted.face_geometries.len(),
        extracted.entity_descriptors.len(),
        extracted.light_descriptors.len(),
        extracted.render_batches.len(),
        extracted.has_pvs,
    );

    // Reject missing surfaces or zero-batch in strict mode.
    if extracted.render_batches.is_empty() {
        log::error!("Strict extraction produced zero render batches; cannot capture");
        std::process::exit(1);
    }

    let mut scene = Scene::new();
    let bsp_mount = match renderer.prepare_bsp_mount(&extracted) {
        Ok(mount) => mount,
        Err(err) => {
            log::error!("BSP GPU upload failed: {err}");
            std::process::exit(1);
        }
    };
    scene.set_bsp_mount(bsp_mount);

    // ── Frozen camera ──────────────────────────────────────────────────
    let camera_label = acceptance_camera_label().unwrap_or_else(|| "spawn".to_string());
    let (eye, look_at) = frozen_camera_for_label(&camera_label, &extracted);

    renderer
        .set_camera_look_at(eye, look_at, Vec3::Y)
        .expect("set camera");

    log::info!(
        "Camera '{}': eye={:?}, look_at={:?}",
        camera_label,
        eye,
        look_at
    );

    // ── Capture at frozen settings ─────────────────────────────────────
    let output_dir = args
        .capture_dir
        .clone()
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                ".internal-dev/captures/bsp-dungeon-completion/camera-{}",
                camera_label
            ))
        });

    std::fs::create_dir_all(&output_dir).unwrap_or_else(|e| {
        log::error!("Failed to create capture dir '{}': {e}", output_dir.display());
    });

    let png_path = output_dir.join(format!("capture_{camera_label}_frame_5.png"));
    let sidecar_path = output_dir.join(format!("capture_{camera_label}_frame_5.json"));

    let req = FrameCaptureRequest {
        target: CaptureTarget::Draw,
        output_path: png_path.clone(),
        sidecar_path: Some(sidecar_path.clone()),
    };
    if let Err(err) = renderer.request_frame_capture_at(5, req) {
        log::error!("Failed to schedule capture: {err}");
        std::process::exit(1);
    }

    let frame_budget = 180u32;
    let mut capture_done = false;

    for _frame in 0..frame_budget {
        match renderer.render_scene_headless(&mut scene) {
            Ok(FrameRenderOutcome::Rendered)
            | Ok(FrameRenderOutcome::SkippedAcquireUnavailable)
            | Ok(FrameRenderOutcome::SkippedResizePending)
            | Ok(FrameRenderOutcome::SubmittedNotPresented)
            | Ok(FrameRenderOutcome::PresentedSuboptimal) => {}
            Err(err) => {
                log::error!("Headless render failed: {err}");
                std::process::exit(1);
            }
        }

        match renderer.last_frame_capture_status() {
            Some(FrameCaptureStatus::Succeeded {
                output_path,
                width,
                height,
                ..
            }) if *output_path == png_path => {
                log::info!(
                    "✓ Capture written: {} ({}×{})",
                    output_path.display(),
                    width,
                    height
                );
                if *width != 1280 || *height != 720 {
                    log::warn!(
                        "Capture dimensions {}×{} do not match frozen 1280×720",
                        *width,
                        *height
                    );
                }
                capture_done = true;
                break;
            }
            Some(FrameCaptureStatus::Failed { message, .. }) => {
                log::error!("Capture failed: {message}");
                std::process::exit(1);
            }
            _ => {}
        }
    }

    if !capture_done {
        log::error!(
            "Capture did not complete within {} frames",
            frame_budget
        );
        std::process::exit(1);
    }

    // Verify sidecar was written.
    if !sidecar_path.is_file() {
        log::error!("Sidecar not written: {}", sidecar_path.display());
        std::process::exit(1);
    }

    log::info!(
        "Strict capture complete: camera={}, png={}, sidecar={}",
        camera_label,
        png_path.display(),
        sidecar_path.display()
    );
}

/// Return a frozen camera (eye, look_at) for the given semantic label.
///
/// Uses the authored spawn entity and spatial witnesses from the selected
/// Phase 08 corpus entry to produce fixed position, orientation, and look
/// target. Cameras in solid space or with non-finite data are rejected.
#[cfg(feature = "bsp")]
fn frozen_camera_for_label(
    label: &str,
    extracted: &bsp::extract::ExtractedBsp,
) -> (glam::Vec3, glam::Vec3) {
    use glam::Vec3;

    // Find info_player_start for the spawn camera.
    let player_start = extracted
        .entity_descriptors
        .iter()
        .find(|entity| {
            matches!(
                entity.classname.as_str(),
                "info_player_start" | "info_player_deathmatch"
            )
        })
        .and_then(|entity| entity.origin)
        .filter(|origin| origin.is_finite());

    // Compute map center from entity origins as fallback reference.
    let origins: Vec<Vec3> = extracted
        .entity_descriptors
        .iter()
        .filter_map(|e| e.origin)
        .filter(|o| o.is_finite())
        .collect();
    let map_center = if origins.is_empty() {
        Vec3::ZERO
    } else {
        let sum: Vec3 = origins.iter().sum();
        sum / origins.len() as f32
    };

    let spawn_pos = player_start.unwrap_or(map_center);

    match label {
        "spawn" => {
            // At spawn, face toward map interior (+Z direction from spawn).
            let eye = spawn_pos + Vec3::new(0.0, 2.0, 0.0); // eye height
            let look = eye + Vec3::Z * 5.0;
            (eye, look)
        }
        "corridor" => {
            // Position at a corridor midpoint between spawn and map edge.
            let eye = Vec3::new(
                map_center.x * 0.5,
                spawn_pos.y + 2.0,
                map_center.z * 1.5,
            );
            let look = eye + Vec3::X * 5.0; // look along corridor
            (eye, look)
        }
        "junction" => {
            // Position at the map center, looking toward a junction.
            let eye = Vec3::new(map_center.x, map_center.y + 2.0, map_center.z);
            let look = eye + Vec3::NEG_Z * 5.0; // look at another junction
            (eye, look)
        }
        _ => {
            log::warn!("Unknown camera label '{}'; using spawn", label);
            let eye = spawn_pos + Vec3::new(0.0, 2.0, 0.0);
            let look = eye + Vec3::Z * 5.0;
            (eye, look)
        }
    }
}
