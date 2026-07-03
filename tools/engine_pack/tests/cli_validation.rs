use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

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
fn validate_package_reports_collision_metadata_failures() {
    let temp = temp_dir("collision_metadata");
    let package_path = temp.join("bad_collision.package.toml");
    fs::write(
        &package_path,
        r#"format_version = 1
package_id = "bad_collision"
display_name = "Bad Collision"

[[assets]]
id = "bad_collision.prefab.wall"
kind = "prefab"
path = "prefabs/wall.prefab"

[assets.metadata.collision]
body_kind = "static"
shape = { kind = "box", half_extents = [0.5, 0.0, 0.5] }
"#,
    )
    .expect("write package");

    let output = engine_pack()
        .args(["validate-package", path_str(&package_path).as_str()])
        .output()
        .expect("run engine_pack");
    assert_failure_contains(output, "asset.collision_invalid_dimension");
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

#[test]
fn authoring_commands_generate_valid_project_package_and_asset_records() {
    let temp = temp_dir("authoring");
    let project_dir = temp.join("project");
    let output = engine_pack()
        .args([
            "new-project",
            path_str(&project_dir).as_str(),
            "--id",
            "project.cli_authoring",
            "--name",
            "CLI Authoring",
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "created[project]");

    let output = engine_pack()
        .args([
            "validate-project",
            path_str(&project_dir.join("engine.project.toml")).as_str(),
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "valid[project]");

    let package_path = temp.join("package/assets/fixture.package.toml");
    let output = engine_pack()
        .args([
            "new-package",
            path_str(&package_path).as_str(),
            "--id",
            "fixture",
            "--name",
            "Fixture",
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "created[package]");

    fs::create_dir_all(package_path.parent().unwrap().join("models")).expect("create model dir");
    fs::write(
        package_path.parent().unwrap().join("models/generated.obj"),
        b"# generated\n",
    )
    .expect("write model");
    let output = engine_pack()
        .args([
            "add-asset",
            path_str(&package_path).as_str(),
            "--id",
            "fixture.model.generated",
            "--kind",
            "model",
            "--path",
            "models/generated.obj",
            "--tag",
            "generated",
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "added[asset]");

    let output = engine_pack()
        .args(["validate-package", path_str(&package_path).as_str()])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "valid[package]");
}

#[test]
fn scan_assets_is_deterministic() {
    let output = engine_pack()
        .args([
            "scan-assets",
            "fixtures/projects/valid/assets",
            "--package-id",
            "fixture",
        ])
        .output()
        .expect("run engine_pack");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("id = \"fixture.model.models.crate\""));
    assert!(stdout.contains("kind = \"model\""));
    assert!(stdout.contains("path = \"models/crate.obj\""));
    assert_success_contains(output, "[[assets]]");
}

#[test]
fn pack_outputs_folder_package_and_report() {
    let temp = temp_dir("pack");
    let out = temp.join("packed");
    let output = engine_pack()
        .args([
            "pack",
            "../../apps/editor/sample_project/engine.project.toml",
            "--out",
            path_str(&out).as_str(),
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "packed[project]");

    assert!(out.join("engine.project.toml").exists());
    assert!(out.join("assets/editor_sample.package.toml").exists());
    assert!(out.join("assets/models/block_prop.obj").exists());
    assert!(out.join("scenes/start.engine.scene.json").exists());
    let report = fs::read_to_string(out.join("PACK_REPORT.json")).expect("read report");
    assert!(report.contains("\"validation_status\": \"passed\""));
    assert!(report.contains("assets/prefabs/wall_straight_2m.obj"));
}

#[test]
fn pack_rejects_missing_asset_files() {
    let temp = temp_dir("pack_missing");
    let project = temp.join("engine.project.toml");
    fs::create_dir_all(temp.join("assets")).expect("create assets");
    fs::create_dir_all(temp.join("scenes")).expect("create scenes");
    fs::write(
        &project,
        "format_version = 1\nproject_id = \"project.missing\"\nname = \"Missing\"\nproject_version = \"0.1.0\"\nasset_root = \"assets\"\nstartup_scene = \"scenes/start.engine.scene.json\"\n\n[[packages]]\npackage_id = \"missing\"\nmanifest = \"assets/missing.package.toml\"\nenabled = true\n\n[settings]\nwindow_width = 800\nwindow_height = 600\nfullscreen = false\nvsync = true\n",
    )
    .expect("write project");
    fs::write(
        temp.join("scenes/start.engine.scene.json"),
        "{\n  \"format_version\": 1,\n  \"scene_id\": \"scene.missing\",\n  \"root_nodes\": [],\n  \"nodes\": [],\n  \"lights\": [],\n  \"environment\": null,\n  \"editor\": {}\n}\n",
    )
    .expect("write scene");
    fs::write(
        temp.join("assets/missing.package.toml"),
        "format_version = 1\npackage_id = \"missing\"\ndisplay_name = \"Missing\"\n\n[[assets]]\nid = \"missing.model.crate\"\nkind = \"model\"\npath = \"models/missing.obj\"\n",
    )
    .expect("write package");

    let output = engine_pack()
        .args([
            "pack",
            path_str(&project).as_str(),
            "--out",
            path_str(&temp.join("out")).as_str(),
        ])
        .output()
        .expect("run engine_pack");
    assert_failure_contains(output, "asset.missing_source_path");
}

#[test]
fn pack_rejects_parent_traversing_paths() {
    let temp = temp_dir("pack_traversal");
    let project = temp.join("engine.project.toml");
    fs::create_dir_all(temp.join("assets")).expect("create assets");
    fs::create_dir_all(temp.join("scenes")).expect("create scenes");
    fs::write(
        &project,
        "format_version = 1\nproject_id = \"project.traversal\"\nname = \"Traversal\"\nproject_version = \"0.1.0\"\nasset_root = \"assets\"\nstartup_scene = \"scenes/start.engine.scene.json\"\n\n[[packages]]\npackage_id = \"traversal\"\nmanifest = \"assets/../traversal.package.toml\"\nenabled = true\n\n[settings]\nwindow_width = 800\nwindow_height = 600\nfullscreen = false\nvsync = true\n",
    )
    .expect("write project");
    fs::write(
        temp.join("scenes/start.engine.scene.json"),
        "{\n  \"format_version\": 1,\n  \"scene_id\": \"scene.traversal\",\n  \"root_nodes\": [],\n  \"nodes\": [],\n  \"lights\": [],\n  \"environment\": null,\n  \"editor\": {}\n}\n",
    )
    .expect("write scene");
    fs::write(
        temp.join("traversal.package.toml"),
        "format_version = 1\npackage_id = \"traversal\"\ndisplay_name = \"Traversal\"\n",
    )
    .expect("write package");

    let output = engine_pack()
        .args([
            "pack",
            path_str(&project).as_str(),
            "--out",
            path_str(&temp.join("out")).as_str(),
        ])
        .output()
        .expect("run engine_pack");
    assert_failure_contains(output, "pack.invalid_project_path");
}

#[test]
fn pack_rejects_parent_traversing_asset_paths() {
    let temp = temp_dir("pack_asset_traversal");
    let project = temp.join("engine.project.toml");
    fs::create_dir_all(temp.join("assets/models")).expect("create assets");
    fs::create_dir_all(temp.join("scenes")).expect("create scenes");
    fs::write(
        &project,
        "format_version = 1\nproject_id = \"project.asset_traversal\"\nname = \"Asset Traversal\"\nproject_version = \"0.1.0\"\nasset_root = \"assets\"\nstartup_scene = \"scenes/start.engine.scene.json\"\n\n[[packages]]\npackage_id = \"asset_traversal\"\nmanifest = \"assets/asset_traversal.package.toml\"\nenabled = true\n\n[settings]\nwindow_width = 800\nwindow_height = 600\nfullscreen = false\nvsync = true\n",
    )
    .expect("write project");
    fs::write(
        temp.join("scenes/start.engine.scene.json"),
        "{\n  \"format_version\": 1,\n  \"scene_id\": \"scene.asset_traversal\",\n  \"root_nodes\": [],\n  \"nodes\": [],\n  \"lights\": [],\n  \"environment\": null,\n  \"editor\": {}\n}\n",
    )
    .expect("write scene");
    fs::write(temp.join("assets/outside.obj"), b"# outside\n").expect("write asset");
    fs::write(
        temp.join("assets/asset_traversal.package.toml"),
        "format_version = 1\npackage_id = \"asset_traversal\"\ndisplay_name = \"Asset Traversal\"\n\n[[assets]]\nid = \"asset_traversal.model.outside\"\nkind = \"model\"\npath = \"models/../outside.obj\"\n",
    )
    .expect("write package");

    let output = engine_pack()
        .args([
            "pack",
            path_str(&project).as_str(),
            "--out",
            path_str(&temp.join("out")).as_str(),
        ])
        .output()
        .expect("run engine_pack");
    assert_failure_contains(output, "pack.invalid_asset_path");
}

#[test]
fn failed_repack_removes_stale_success_report() {
    let temp = temp_dir("pack_stale_report");
    let project = temp.join("engine.project.toml");
    let out = temp.join("out");
    fs::create_dir_all(temp.join("assets/models")).expect("create assets");
    fs::create_dir_all(temp.join("scenes")).expect("create scenes");
    fs::write(
        &project,
        "format_version = 1\nproject_id = \"project.stale_report\"\nname = \"Stale Report\"\nproject_version = \"0.1.0\"\nasset_root = \"assets\"\nstartup_scene = \"scenes/start.engine.scene.json\"\n\n[[packages]]\npackage_id = \"stale_report\"\nmanifest = \"assets/stale_report.package.toml\"\nenabled = true\n\n[settings]\nwindow_width = 800\nwindow_height = 600\nfullscreen = false\nvsync = true\n",
    )
    .expect("write project");
    fs::write(
        temp.join("scenes/start.engine.scene.json"),
        "{\n  \"format_version\": 1,\n  \"scene_id\": \"scene.stale_report\",\n  \"root_nodes\": [],\n  \"nodes\": [],\n  \"lights\": [],\n  \"environment\": null,\n  \"editor\": {}\n}\n",
    )
    .expect("write scene");
    fs::write(temp.join("assets/models/model.obj"), b"# model\n").expect("write asset");
    fs::write(
        temp.join("assets/stale_report.package.toml"),
        "format_version = 1\npackage_id = \"stale_report\"\ndisplay_name = \"Stale Report\"\n\n[[assets]]\nid = \"stale_report.model.model\"\nkind = \"model\"\npath = \"models/model.obj\"\n",
    )
    .expect("write package");

    let output = engine_pack()
        .args([
            "pack",
            path_str(&project).as_str(),
            "--out",
            path_str(&out).as_str(),
        ])
        .output()
        .expect("run engine_pack");
    assert_success_contains(output, "packed[project]");
    assert!(out.join("PACK_REPORT.json").exists());

    fs::remove_file(temp.join("assets/models/model.obj")).expect("remove asset");
    let output = engine_pack()
        .args([
            "pack",
            path_str(&project).as_str(),
            "--out",
            path_str(&out).as_str(),
        ])
        .output()
        .expect("run engine_pack");
    assert_failure_contains(output, "asset.missing_source_path");
    assert!(!out.join("PACK_REPORT.json").exists());
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

fn temp_dir(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "engine_pack_{label}_{}_{}",
        std::process::id(),
        unique
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn path_str(path: &std::path::Path) -> String {
    path.to_string_lossy().to_string()
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
