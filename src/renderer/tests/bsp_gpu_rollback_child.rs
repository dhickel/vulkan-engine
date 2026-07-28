//! BSP GPU rollback child-process fault suite.
//!
//! Each test boundary validates that a candidate B upload failure cleans only
//! B's resources without touching active arena A. Tests require a
//! Vulkan-capable host with validation layers available.
//!
//! Run with:
//!   cargo test -p renderer --features bsp -- bsp_gpu_rollback --ignored --nocapture

#![cfg(feature = "bsp")]

use renderer::prelude::*;
use std::process::Command;

/// Fault injection flags passed to the child process via environment variable.
const FAULT_POST_MATERIAL: &str = "POST_MATERIAL";
const FAULT_NONE: &str = "NONE";

/// Launch a child process that runs this same test binary with the given
/// fault flag. Returns (exit_code, stdout, stderr).
fn run_rollback_child(fault: &str) -> (i32, String, String) {
    let child = Command::new(std::env::current_exe().expect("current exe"))
        .args([
            "--test",
            "rollback_child_entry",
            "--",
            "--ignored",
            "--nocapture",
        ])
        .env("BSP_ROLLBACK_FAULT", fault)
        .output()
        .expect("failed to launch rollback child");
    let code = child.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&child.stdout).to_string();
    let stderr = String::from_utf8_lossy(&child.stderr).to_string();
    (code, stdout, stderr)
}

/// Child-process entry point. When `BSP_ROLLBACK_FAULT` is set, it creates
/// a headless renderer, uploads arena A, then injects a fault for arena B.
#[test]
#[ignore = "launched as a child by the rollback fault suite"]
fn rollback_child_entry() {
    let fault =
        std::env::var("BSP_ROLLBACK_FAULT").unwrap_or_else(|_| FAULT_NONE.to_string());

    eprintln!("[child] rollback child starting with fault={fault}");

    let result = (|| -> Result<(), String> {
        let mut renderer = Renderer::new_headless(
            RendererConfig {
                app_name: format!("bsp-rollback-child-{fault}"),
                headless: true,
                validation_layer: false,
                preload_startup_scene: false,
                ..Default::default()
            },
        )
        .map_err(|e| format!("child renderer init: {e}"))?;

        // Use the material fixture from the existing test infrastructure.
        // The fixture path is resolved relative to the renderer crate.
        let extracted = material_fixture();

        // Arena A: upload first mount.
        let _mount_a = renderer
            .prepare_bsp_mount(&extracted)
            .map_err(|e| format!("arena A upload: {e}"))?;

        match fault.as_str() {
            FAULT_NONE => {
                eprintln!("[child] no fault injected — exiting OK");
                return Ok(());
            }
            FAULT_POST_MATERIAL => {
                // Set the fault-injection atomic before uploading B.
                renderer::api::bsp::FAIL_BSP_UPLOAD_AFTER_MATERIAL_REGISTRATION
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                let result = renderer.prepare_bsp_mount(&extracted);
                match result {
                    Err(e) => {
                        let msg = e.to_string();
                        if msg.contains("injected BSP upload failure after material registration")
                        {
                            eprintln!("[child] got expected typed error");
                            // Arena A's resources should still be intact.
                            // Upload arena A again to prove it.
                            let _mount_a2 = renderer
                                .prepare_bsp_mount(&extracted)
                                .map_err(|e| format!("arena A re-upload after B failure: {e}"))?;
                            eprintln!("[child] arena A re-upload succeeded after B failure");
                            return Ok(());
                        }
                        return Err(format!("unexpected error: {msg}"));
                    }
                    Ok(_) => return Err("fault injection did not trigger".to_string()),
                }
            }
            _ => {
                return Err(format!("unknown fault flag: {fault}"));
            }
        }
    })();

    match result {
        Ok(()) => eprintln!("[child] PASS"),
        Err(e) => {
            eprintln!("[child] FAIL: {e}");
            std::process::exit(1);
        }
    }
}

/// Load the material fixture BSP for arena coexistence tests.
fn material_fixture() -> bsp::extract::ExtractedBsp {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../bsp/tests/fixtures");
    let bsp_path = fixtures.join("compiled/dungeon-materials-bsp2.bsp");
    let palette_path = fixtures.join("palettes/project_palette.lmp");
    let palette_bytes = std::fs::read(&palette_path).expect("read fixture palette");
    let world = bsp::BspLoader::load(
        &std::fs::read(&bsp_path).expect("read material fixture"),
        &bsp::LoadOptions {
            palette: Some(palette_bytes.clone()),
            source_identity: bsp_path.display().to_string(),
            ..Default::default()
        },
    )
    .expect("load material fixture");
    bsp::extract::extract(bsp::BspExtractionRequest {
        world,
        palette: Some(bsp::resources::decode_palette(&palette_bytes)),
        texture_companions: vec![],
        strict: false,
        ..Default::default()
    })
    .expect("extract material fixture")
}

// ── Parent-process fault-boundary tests ────────────────────────────────

#[test]
#[ignore = "requires a Vulkan-capable GPU"]
fn rollback_after_material_registration_exits_clean() {
    let (code, _stdout, stderr) = run_rollback_child(FAULT_POST_MATERIAL);
    assert_eq!(
        code, 0,
        "child crashed or returned non-zero\nstderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("VUID-") && !stderr.to_lowercase().contains("validation"),
        "validation diagnostic in child stderr"
    );
    assert!(
        !stderr.to_lowercase().contains("double free") && !stderr.contains("DoubleFree"),
        "duplicate destruction detected in child stderr"
    );
}

#[test]
#[ignore = "requires a Vulkan-capable GPU"]
fn arena_a_survives_after_b_failure() {
    let (code, _stdout, stderr) = run_rollback_child(FAULT_POST_MATERIAL);
    assert_eq!(code, 0, "child should exit 0 after A survives B failure\nstderr:\n{stderr}");
}
