use std::process::{Command, Output};

fn engine_pack() -> Command {
    Command::new(env!("CARGO_BIN_EXE_engine_pack"))
}

#[test]
fn validate_package_accepts_valid_fixture() {
    let output = engine_pack()
        .args([
            "validate-package",
            "fixtures/packages/valid.package.toml",
            "--expected-package-id",
            "fixture",
        ])
        .output()
        .expect("run engine_pack");

    assert_success_contains(output, "valid[package]");
}

#[test]
fn validation_options_can_precede_positional_paths() {
    let output = engine_pack()
        .args([
            "validate-package",
            "--expected-package-id",
            "editor_sample",
            "../../apps/editor/sample_project/assets/editor_sample.package.toml",
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "valid[package]");

    let output = engine_pack()
        .args([
            "validate-scene",
            "--project",
            "../../apps/editor/sample_project/engine.project.toml",
            "../../apps/editor/sample_project/scenes/start.engine.scene.json",
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "valid[scene]");
}

#[test]
fn validate_package_rejects_invalid_fixtures_with_stable_codes() {
    let cases = [
        (
            "fixtures/packages/missing-version.package.toml",
            "package.missing_format_version",
        ),
        (
            "fixtures/packages/duplicate-id.package.toml",
            "package.duplicate_asset_id",
        ),
        (
            "fixtures/packages/runtime-handle.package.toml",
            "asset.runtime_handle_identity",
        ),
    ];

    for (path, code) in cases {
        let output = engine_pack()
            .args(["validate-package", path])
            .output()
            .expect("run engine_pack");
        assert_failure_contains(output, code);
    }
}

#[test]
fn validate_project_accepts_valid_fixture_and_rejects_missing_scene() {
    let output = engine_pack()
        .args([
            "validate-project",
            "fixtures/projects/valid/engine.project.toml",
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "valid[project]");

    let output = engine_pack()
        .args([
            "validate-project",
            "fixtures/projects/invalid_missing_scene/engine.project.toml",
        ])
        .output()
        .expect("run engine_pack");
    assert_failure_contains(output, "project.missing_startup_scene");
}

#[test]
fn validate_scene_resolves_known_assets_from_project() {
    let output = engine_pack()
        .args([
            "validate-scene",
            "fixtures/projects/valid/scenes/start.engine.scene.json",
            "--project",
            "fixtures/projects/valid/engine.project.toml",
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "valid[scene]");

    let output = engine_pack()
        .args([
            "validate-scene",
            "fixtures/scenes/unknown-asset.engine.scene.json",
            "--project",
            "fixtures/projects/valid/engine.project.toml",
        ])
        .output()
        .expect("run engine_pack");
    assert_failure_contains(output, "scene.unknown_asset_id");
}

#[test]
fn usage_errors_exit_with_code_two() {
    let output = engine_pack()
        .args([
            "validate-scene",
            "fixtures/projects/valid/scenes/start.engine.scene.json",
        ])
        .output()
        .expect("run engine_pack");

    assert_eq!(
        output.status.code(),
        Some(2),
        "expected CLI usage exit code\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("error[cli.usage]"),
        "stderr missing usage code: {stderr}"
    );
}

fn assert_success_contains(output: Output, needle: &str) {
    assert!(
        output.status.success(),
        "expected success\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(needle), "stdout missing {needle}: {stdout}");
}

fn assert_failure_contains(output: Output, needle: &str) {
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected validation exit code\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "expected failure\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(needle), "stderr missing {needle}: {stderr}");
}
