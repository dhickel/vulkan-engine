//! CLI validation tests — freeze current command compatibility.
//!
//! Table-driven tests for every `engine_pack` command and root launcher.
//! Each test asserts option singleton/repeatable, value/flag, allowed
//! spaced/equals forms, aliases, required options, duplicate singleton
//! rejection, and positional cardinality.

use launch_shared::{parse_cli_args, CliOption, OptionValuePolicy};

/// Convenience: create Vec<String> from &str slices.
macro_rules! svec {
    ($($x:expr),* $(,)?) => {
        vec![$($x.to_string()),*]
    };
}

// ────────────────────────────────────────────────────
// Shared schema for root launcher
// ────────────────────────────────────────────────────

fn root_launcher_schema() -> Vec<CliOption> {
    vec![
        CliOption {
            name: "--help",
            short: Some("-h"),
            value_policy: OptionValuePolicy::Flag,
            allow_spaced: true,
            allow_equals: false,
            repeatable: false,
            help: "Print help",
            value_placeholder: None,
        },
        CliOption {
            name: "--project",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Project manifest to launch",
            value_placeholder: Some("<path>"),
        },
        CliOption {
            name: "--scene",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Startup scene override",
            value_placeholder: Some("<path>"),
        },
        CliOption {
            name: "--headless",
            short: None,
            value_policy: OptionValuePolicy::Flag,
            allow_spaced: true,
            allow_equals: false,
            repeatable: false,
            help: "Headless runtime",
            value_placeholder: None,
        },
        CliOption {
            name: "--capture_target",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Capture target",
            value_placeholder: Some("<present|draw>"),
        },
        CliOption {
            name: "--capture_frame",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Single frame capture",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--capture_frames",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Sequence capture frame count",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--capture_frame_start",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Sequence capture start frame",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--capture_frame_interval",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Sequence capture interval",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--capture_dir",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Capture output dir",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--capture_frame_path",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Single frame output path",
            value_placeholder: Some("<path>"),
        },
        CliOption {
            name: "--manual_capture_dir",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Manual capture dir",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--record_debug",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Debug record seconds",
            value_placeholder: Some("<seconds>"),
        },
        CliOption {
            name: "--record_debug_interval",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Debug record interval ms",
            value_placeholder: Some("<ms>"),
        },
        CliOption {
            name: "--record_debug_path",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Debug record output path",
            value_placeholder: Some("<path>"),
        },
    ]
}

// ────────────────────────────────────────────────────
// Table-driven option property tests
// ────────────────────────────────────────────────────

#[test]
fn root_launcher_all_options_declared() {
    let schema = root_launcher_schema();
    // Every option has a name starting with `--` or `-`
    for opt in &schema {
        assert!(
            opt.name.starts_with("--") || opt.name.starts_with('-'),
            "option name must start with -- or -: {}",
            opt.name
        );
    }
}

#[test]
fn engine_pack_all_options_declared() {
    use engine_pack::cli;
    for (cmd, schema) in [
        ("validate-package", cli::validate_package_schema()),
        ("validate-project", cli::validate_project_schema()),
        ("validate-scene", cli::validate_scene_schema()),
        ("new-app", cli::new_app_schema()),
        ("new-project", cli::new_project_schema()),
        ("new-package", cli::new_package_schema()),
        ("scan-assets", cli::scan_assets_schema()),
        ("add-asset", cli::add_asset_schema()),
        ("pack", cli::pack_schema()),
    ] {
        for opt in schema {
            assert!(
                opt.name.starts_with("--"),
                "{cmd} option name must start with --: {}",
                opt.name
            );
        }
    }
}

// ────────────────────────────────────────────────────
// Root launcher: spaced and equals forms
// ────────────────────────────────────────────────────

#[test]
fn root_launcher_accepts_spaced_project() {
    let args = svec!["--project", "engine.project.toml"];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(
        result.singleton_value("--project"),
        Some("engine.project.toml")
    );
}

#[test]
fn root_launcher_accepts_equals_project() {
    let args = svec!["--project=engine.project.toml"];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(
        result.singleton_value("--project"),
        Some("engine.project.toml")
    );
}

#[test]
fn root_launcher_accepts_spaced_scene_debug_capture() {
    let args = svec![
        "--project",
        "engine.project.toml",
        "--scene",
        "scenes/start.engine.scene.json",
        "--headless",
        "--capture_frames",
        "3",
        "--capture_frame_start",
        "5",
        "--capture_frame_interval",
        "5",
        "--capture_dir",
        "captures",
        "--capture_target",
        "draw",
        "--record_debug",
        "10",
        "--record_debug_interval",
        "50",
        "--record_debug_path",
        "timing.jsonl",
    ];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(
        result.singleton_value("--project"),
        Some("engine.project.toml")
    );
    assert_eq!(
        result.singleton_value("--scene"),
        Some("scenes/start.engine.scene.json")
    );
    assert!(result.flag_present("--headless"));
    assert_eq!(result.singleton_value("--capture_frames"), Some("3"));
    assert_eq!(result.singleton_value("--capture_target"), Some("draw"));
    assert_eq!(result.singleton_value("--record_debug"), Some("10"));
}

#[test]
fn root_launcher_accepts_mixed_spaced_and_equals() {
    let args = svec![
        "--project=engine.project.toml",
        "--scene",
        "scenes/start.engine.scene.json",
        "--headless",
        "--capture_frames=3",
        "--capture_frame_start",
        "5",
    ];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(
        result.singleton_value("--project"),
        Some("engine.project.toml")
    );
    assert_eq!(
        result.singleton_value("--scene"),
        Some("scenes/start.engine.scene.json")
    );
    assert_eq!(result.singleton_value("--capture_frames"), Some("3"));
    assert_eq!(result.singleton_value("--capture_frame_start"), Some("5"));
}

// ────────────────────────────────────────────────────
// Root launcher: help flag
// ────────────────────────────────────────────────────

#[test]
fn root_launcher_help_flag() {
    let args = svec!["--help"];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(result.is_ok());
    assert!(result.flag_present("--help"));

    let args = svec!["-h"];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(result.is_ok());
    assert!(result.flag_present("--help"));
}

// ────────────────────────────────────────────────────
// Duplicate singleton rejection
// ────────────────────────────────────────────────────

#[test]
fn root_launcher_rejects_duplicate_project() {
    let args = svec!["--project", "a.toml", "--project", "b.toml"];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(!result.is_ok());
    assert!(result.errors.iter().any(|e| e.contains("duplicate")));
}

#[test]
fn root_launcher_rejects_duplicate_scene() {
    let args = svec![
        "--project",
        "p.toml",
        "--scene",
        "a.json",
        "--scene",
        "b.json"
    ];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(!result.is_ok());
    assert!(result.errors.iter().any(|e| e.contains("duplicate")));
}

#[test]
fn root_launcher_rejects_duplicate_headless() {
    let args = svec!["--project", "p.toml", "--headless", "--headless"];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(!result.is_ok());
    assert!(result.errors.iter().any(|e| e.contains("duplicate")));
}

#[test]
fn root_launcher_rejects_duplicate_help() {
    let args = svec!["--project", "p.toml", "--help", "--help"];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(!result.is_ok());
    assert!(result.errors.iter().any(|e| e.contains("duplicate")));
}

#[test]
fn engine_pack_rejects_duplicate_singleton_option() {
    use engine_pack::cli;
    let args = svec!["--id", "a", "--id", "b"];
    let result = cli::parse_command("new-app", cli::new_app_schema(), &args);
    assert!(!result.is_ok());
    assert!(result.errors.iter().any(|e| e.contains("duplicate")));
}

// ────────────────────────────────────────────────────
// engine_pack: spaced and equals forms
// ────────────────────────────────────────────────────

#[test]
fn engine_pack_add_asset_spaced_and_equals() {
    use engine_pack::cli;
    // spaced
    let args = svec![
        "manifest.toml",
        "--id",
        "asset.1",
        "--kind",
        "model",
        "--path",
        "m.glb",
    ];
    let result = cli::parse_command("add-asset", cli::add_asset_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(result.singleton_value("--id"), Some("asset.1"));
    assert_eq!(result.singleton_value("--kind"), Some("model"));
    assert_eq!(result.singleton_value("--path"), Some("m.glb"));
    assert_eq!(result.positionals, vec!["manifest.toml"]);

    // equals
    let args = svec![
        "manifest.toml",
        "--id=asset.2",
        "--kind=texture",
        "--path=tex.png",
    ];
    let result = cli::parse_command("add-asset", cli::add_asset_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(result.singleton_value("--id"), Some("asset.2"));
    assert_eq!(result.singleton_value("--kind"), Some("texture"));
    assert_eq!(result.singleton_value("--path"), Some("tex.png"));
}

// ────────────────────────────────────────────────────
// engine_pack: repeatable --tag
// ────────────────────────────────────────────────────

#[test]
fn engine_pack_repeatable_tag() {
    use engine_pack::cli;
    let args = svec![
        "manifest.toml",
        "--id",
        "asset.1",
        "--kind",
        "model",
        "--path",
        "m.glb",
        "--tag",
        "wall",
        "--tag",
        "stone",
        "--tag=concrete",
    ];
    let result = cli::parse_command("add-asset", cli::add_asset_schema(), &args);
    assert!(result.is_ok());
    let tags = result.repeated_values("--tag");
    assert_eq!(tags, vec!["wall", "stone", "concrete"]);
}

// ────────────────────────────────────────────────────
// Missing required options
// ────────────────────────────────────────────────────

#[test]
fn root_launcher_missing_project_produces_no_value() {
    let args: Vec<String> = vec![];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(result.is_ok()); // parser doesn't enforce required — caller does
    assert!(result.singleton_value("--project").is_none());
}

#[test]
fn engine_pack_missing_required_fails_in_caller() {
    use engine_pack::cli;
    let args: Vec<String> = vec![];
    let result = cli::parse_command("new-app", cli::new_app_schema(), &args);
    assert!(result.is_ok()); // parser success — but caller should enforce --id presence
    assert!(result.singleton_value("--id").is_none());
}

// ────────────────────────────────────────────────────
// Root launcher: unknown flags and positionals
// ────────────────────────────────────────────────────

#[test]
fn root_launcher_rejects_unknown_flags() {
    let args = svec!["--bogus"];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(!result.is_ok());
    assert!(result.errors.iter().any(|e| e.contains("unknown option")));
}

#[test]
fn root_launcher_treats_non_flag_as_positional() {
    let args = svec!["engine.project.toml"];
    let result = parse_cli_args(&root_launcher_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(result.positionals, vec!["engine.project.toml"]);
}

// ────────────────────────────────────────────────────
// engine_pack: positional cardinality
// ────────────────────────────────────────────────────

#[test]
fn engine_pack_pack_missing_out_and_project() {
    use engine_pack::cli;
    let args: Vec<String> = vec![];
    let result = cli::parse_command("pack", cli::pack_schema(), &args);
    assert!(result.is_ok()); // parser success — caller enforces
    assert!(result.singleton_value("--out").is_none());
    assert!(result.positionals.is_empty());
}

#[test]
fn engine_pack_validate_package_accepts_options_and_positional() {
    use engine_pack::cli;
    let args = svec![
        "package.toml",
        "--expected-package-id",
        "core",
        "--project-root=some/dir",
    ];
    let result = cli::parse_command("validate-package", cli::validate_package_schema(), &args);
    assert!(result.is_ok());
    assert_eq!(
        result.singleton_value("--expected-package-id"),
        Some("core")
    );
    assert_eq!(result.singleton_value("--project-root"), Some("some/dir"));
    assert_eq!(result.positionals, vec!["package.toml"]);
}

// ────────────────────────────────────────────────────
// engine_pack: unknown flags in subcommands
// ────────────────────────────────────────────────────

#[test]
fn engine_pack_rejects_unknown_flags() {
    use engine_pack::cli;
    let args = svec!["--bogus"];
    let result = cli::parse_command("validate-project", cli::validate_project_schema(), &args);
    assert!(!result.is_ok());
    assert!(result.errors.iter().any(|e| e.contains("unknown option")));
}

// ────────────────────────────────────────────────────
// Help generation
// ────────────────────────────────────────────────────

#[test]
fn engine_pack_help_contains_all_commands() {
    let help = engine_pack::cli::global_help();
    let cmds = [
        "validate-package",
        "validate-project",
        "validate-scene",
        "new-app",
        "new-project",
        "new-package",
        "scan-assets",
        "add-asset",
        "pack",
    ];
    for cmd in cmds {
        assert!(help.contains(cmd), "help missing command: {cmd}");
    }
}

#[test]
fn render_help_contains_option_descriptions() {
    let rendered = launch_shared::render_help(
        &root_launcher_schema(),
        "engine --project <path> [options]",
        "Root runtime launcher.",
    );
    assert!(rendered.contains("--project"));
    assert!(rendered.contains("--headless"));
    assert!(rendered.contains("Root runtime launcher"));
}
