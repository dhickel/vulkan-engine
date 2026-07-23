//! Headless capture test: BSP Beta Acceptance Proof
//!
//! Renders a loaded BSP scene at a fixed camera viewpoint and captures
//! deterministic draw-target frames for visual acceptance validation.
//!
//! Run: cargo run -p renderer --features bsp --example capture_bsp_beta -- --headless --bsp <path>
//!
//! This example requires the `bsp` feature gate. Without it, the example
//! exits with a message directing the user to enable the feature.
//!
//! This example is part of the BSP beta Phase 09 acceptance evidence matrix.
//! It demonstrates the full load→mount→render→capture pipeline for BSP data.

#[cfg(feature = "bsp")]
#[path = "common.rs"]
mod common;

// ── Feature-gated entrypoints ─────────────────────────────────────────────

#[cfg(not(feature = "bsp"))]
fn main() {
    eprintln!("This example requires the `bsp` feature:");
    eprintln!("  cargo run -p renderer --features bsp --example capture_bsp_beta -- --headless --bsp <path>");
}

#[cfg(feature = "bsp")]
fn main() {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();

    let args = common::parse_capture_test_args();

    // --bsp is required for this capture test
    let bsp_path = find_bsp_arg();
    let bsp_bytes = match std::fs::read(&bsp_path) {
        Ok(b) => b,
        Err(e) => {
            log::error!("Failed to read BSP '{}': {e}", bsp_path.display());
            return;
        }
    };

    log::info!(
        "BSP capture test: {} ({} bytes)",
        bsp_path.display(),
        bsp_bytes.len()
    );

    run_headless_capture_test(&bsp_bytes, &bsp_path, &args);
}

#[cfg(feature = "bsp")]
fn find_bsp_arg() -> std::path::PathBuf {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--bsp" {
            if let Some(val) = args.get(i + 1) {
                return std::path::PathBuf::from(val);
            }
        }
        i += 1;
    }
    eprintln!("--bsp <path> is required for capture_bsp_beta");
    std::process::exit(1);
}

#[cfg(feature = "bsp")]
fn run_headless_capture_test(
    bsp_bytes: &[u8],
    bsp_path: &std::path::PathBuf,
    args: &common::CaptureTestArgs,
) {
    use glam::Vec3;
    use renderer::api::bsp::PreparedBspMount;
    use renderer::api::config::RendererConfig;
    use renderer::api::{CaptureTarget, FrameCaptureRequest, FrameCaptureStatus};
    use renderer::prelude::*;
    use std::path::PathBuf;

    let validation_layer = std::env::var("ENGINE_CAPTURE_VALIDATION")
        .is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "on"));

    let config = RendererConfig {
        app_name: "capture_bsp_beta".to_string(),
        headless: true,
        validation_layer,
        ..RendererConfig::default()
    };

    let mut renderer = match Renderer::new_headless(config) {
        Ok(r) => r,
        Err(err) => {
            log::error!("Headless renderer init failed: {err}");
            return;
        }
    };

    // ── Prepare BSP ────────────────────────────────────────────────────
    let load_options = bsp::LoadOptions {
        strict: false,
        source_identity: bsp_path.display().to_string(),
        ..bsp::LoadOptions::default()
    };
    let world = match bsp::BspLoader::load(bsp_bytes, &load_options) {
        Ok(world) => world,
        Err(report) => {
            log::error!(
                "BSP load failed for '{}': {:?} — {}",
                bsp_path.display(),
                report.code,
                report.message
            );
            return;
        }
    };
    let extracted = bsp::extract::extract(&world, None);

    log::info!(
        "BSP prepared: {} faces, {} entities, {} lights, {} batches, PVS={}",
        extracted.face_geometries.len(),
        extracted.entity_descriptors.len(),
        extracted.light_descriptors.len(),
        extracted.render_batches.len(),
        extracted.has_pvs,
    );

    let mut scene = Scene::new();
    scene.set_bsp_mount(PreparedBspMount::from_extraction_stub(&extracted));

    // ── Camera ─────────────────────────────────────────────────────────
    renderer
        .set_camera_look_at(Vec3::new(0.0, 3.0, 10.0), Vec3::new(0.0, 1.0, 0.0), Vec3::Y)
        .expect("set camera");

    // ── Capture setup ──────────────────────────────────────────────────
    let output_dir = args
        .capture_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(".internal-dev/captures/bsp-beta"));

    // Default: single capture at frame 5
    let expected_captures = args.capture_frames.unwrap_or(1) as usize;
    let output_path = output_dir.join("capture_bsp_beta_frame_5.png");

    let req = FrameCaptureRequest::new(CaptureTarget::Draw, output_path.clone());
    if let Err(err) = renderer.request_frame_capture_at(5, req) {
        log::error!("Failed to schedule capture: {err}");
        return;
    }

    let mut succeeded_paths = std::collections::HashSet::new();
    let frame_budget = args
        .capture_frame_start
        .unwrap_or(5)
        .saturating_add(
            args.capture_frame_interval
                .unwrap_or(5)
                .saturating_mul(args.capture_frames.unwrap_or(1).saturating_add(2)),
        )
        .max(180);

    for _frame in 0..frame_budget {
        match renderer.render_scene_headless(&mut scene) {
            Ok(FrameRenderOutcome::Rendered)
            | Ok(FrameRenderOutcome::SkippedAcquireUnavailable)
            | Ok(FrameRenderOutcome::SkippedResizePending)
            | Ok(FrameRenderOutcome::SubmittedNotPresented)
            | Ok(FrameRenderOutcome::PresentedSuboptimal) => {}
            Err(err) => {
                log::error!("Headless render failed: {err}");
                return;
            }
        }

        match renderer.last_frame_capture_status() {
            Some(FrameCaptureStatus::Succeeded { output_path, .. }) => {
                succeeded_paths.insert(output_path.clone());
                if succeeded_paths.len() >= expected_captures {
                    log::info!(
                        "Headless capture completed: {} capture(s) written",
                        succeeded_paths.len()
                    );
                    return;
                }
            }
            Some(FrameCaptureStatus::Failed { message, .. }) => {
                log::error!("Headless capture failed: {message}");
                return;
            }
            _ => {}
        }
    }

    if expected_captures > 0 {
        log::error!(
            "Capture did not complete within {} frames ({} of {} written)",
            frame_budget,
            succeeded_paths.len(),
            expected_captures
        );
    }
}
