//! Declarative CLI schema for `engine_pack`.
//!
//! Uses the shared `launch_shared` parser to define per-command option schemas,
//! reject duplicate singleton options, accept repeatable `--tag`, and
//! generate help from declarations.

use launch_shared::{parse_cli_args, render_help, CliOption, CliParseResult, OptionValuePolicy};

/// Schema for `validate-package <package.toml> [--expected-package-id <id>] [--project-root <path>]`
pub fn validate_package_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--expected-package-id",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Assert package_id matches this value",
            value_placeholder: Some("<id>"),
        },
        CliOption {
            name: "--project-root",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Resolve relative asset paths against this directory",
            value_placeholder: Some("<path>"),
        },
    ]
}

/// Schema for `validate-project <engine.project.toml>`
pub fn validate_project_schema() -> &'static [CliOption] {
    &[]
}

/// Schema for `validate-scene <scene.engine.scene.json> --project <engine.project.toml>`
pub fn validate_scene_schema() -> &'static [CliOption] {
    &[CliOption {
        name: "--project",
        short: None,
        value_policy: OptionValuePolicy::Value,
        allow_spaced: true,
        allow_equals: true,
        repeatable: false,
        help: "Project manifest with asset IDs for reference validation",
        value_placeholder: Some("<engine.project.toml>"),
    }]
}

/// Schema for `new-app <dir> --id <app_id> --name <display_name>`
pub fn new_app_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--id",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Stable app identity",
            value_placeholder: Some("<app_id>"),
        },
        CliOption {
            name: "--name",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Display name",
            value_placeholder: Some("<display_name>"),
        },
    ]
}

/// Schema for `new-project <dir> --id <project_id> --name <name>`
pub fn new_project_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--id",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Stable project identity",
            value_placeholder: Some("<project_id>"),
        },
        CliOption {
            name: "--name",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Display name",
            value_placeholder: Some("<name>"),
        },
    ]
}

/// Schema for `new-package <path> --id <package_id> --name <display_name>`
pub fn new_package_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--id",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Stable package identity",
            value_placeholder: Some("<package_id>"),
        },
        CliOption {
            name: "--name",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Display name",
            value_placeholder: Some("<display_name>"),
        },
    ]
}

/// Schema for `scan-assets <asset-root> [--package-id <id>]`
pub fn scan_assets_schema() -> &'static [CliOption] {
    &[CliOption {
        name: "--package-id",
        short: None,
        value_policy: OptionValuePolicy::Value,
        allow_spaced: true,
        allow_equals: true,
        repeatable: false,
        help: "Package ID used in generated asset ID suggestions",
        value_placeholder: Some("<id>"),
    }]
}

/// Schema for `add-asset <package.toml> --id <asset_id> --kind <kind> --path <path> [--tag <tag>]`
pub fn add_asset_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--id",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Durable asset ID",
            value_placeholder: Some("<asset_id>"),
        },
        CliOption {
            name: "--kind",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Asset kind (model, texture, material, environment, prefab, wall_chunk, scene_fragment, audio)",
            value_placeholder: Some("<kind>"),
        },
        CliOption {
            name: "--path",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Package-relative asset source path",
            value_placeholder: Some("<path>"),
        },
        CliOption {
            name: "--tag",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: true,
            help: "Search/filter tag (repeatable)",
            value_placeholder: Some("<tag>"),
        },
    ]
}

/// Schema for `enhanced-dungeon --seed <u64> --out <dir> [--tool-path <dir>] [--rooms <n>] [--loops <n>] [--vertical-edges <n>]`
pub fn enhanced_dungeon_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--seed",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Master seed for deterministic generation",
            value_placeholder: Some("<u64>"),
        },
        CliOption {
            name: "--out",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Output directory for published artifact set",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--tool-path",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Directory containing qbsp, vis, light executables",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--rooms",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Number of rooms (M2 range: 17-40, default: 28)",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--loops",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Number of horizontal loops (default: 3)",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--vertical-edges",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Number of vertical transitions (1-3, default: 1)",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--name",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Base name for output files (default: 'enhanced_dungeon')",
            value_placeholder: Some("<name>"),
        },
        CliOption {
            name: "--profile",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Compiler profile TOML (default: bundled ericw-q1-bsp2-generated)",
            value_placeholder: Some("<profile.toml>"),
        },
    ]
}

/// Schema for `enhanced-dungeon-v3 --seed <u64> --preset <sparse|moderate|rich> --out <dir> [--extent <n>] [--tool-path <dir>] [--name <name>]`
pub fn enhanced_dungeon_v3_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--seed",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Master seed for deterministic generation",
            value_placeholder: Some("<u64>"),
        },
        CliOption {
            name: "--preset",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Density preset: sparse, moderate, or rich (default: moderate)",
            value_placeholder: Some("<preset>"),
        },
        CliOption {
            name: "--extent",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "XY extent per axis (1024, 2048, 3072; default: 2048)",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--out",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Output directory for published artifact set",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--tool-path",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Directory containing qbsp, vis, light executables",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--name",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Base name for output files (default: 'enhanced_v3_dungeon')",
            value_placeholder: Some("<name>"),
        },
        CliOption {
            name: "--profile",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Compiler profile TOML (default: bundled ericw-q1-bsp2-generated)",
            value_placeholder: Some("<profile.toml>"),
        },
    ]
}

/// Schema for `enhanced-dungeon-v3-richness-v1 --seed <u64> --preset <sparse|moderate|rich> --theme <ancient|egyptian|brutalist> --out <dir> [--extent <n>] [--landmarks <n>] [--zones <n>] [--cave-mode <mode>] [--vertical-openings <n>] [--budget <n>] [--tool-path <dir>] [--name <name>]`
pub fn enhanced_dungeon_v3_richness_v1_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--seed",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Master seed for deterministic generation",
            value_placeholder: Some("<u64>"),
        },
        CliOption {
            name: "--preset",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Richness density preset: sparse, moderate, or rich",
            value_placeholder: Some("<preset>"),
        },
        CliOption {
            name: "--theme",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Visual theme: ancient, egyptian, or brutalist",
            value_placeholder: Some("<theme>"),
        },
        CliOption {
            name: "--extent",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "XY extent per axis (1024, 2048, 3072; default: 2048)",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--landmarks",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Critical-path landmark count (1-5; inherited from preset)",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--zones",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Semantic zone count (1-6; inherited from preset)",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--cave-mode",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Cave eligibility: required, preferred, or omitted (inherited from preset)",
            value_placeholder: Some("<mode>"),
        },
        CliOption {
            name: "--vertical-openings",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Vertical feature count (0-12; inherited from preset)",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--budget",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Budget ceiling in source faces (1000-8000; inherited from preset)",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--out",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Output directory for published artifact set",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--tool-path",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Directory containing qbsp, vis, light executables",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--name",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Base name for output files (default: 'enhanced_v3_richness')",
            value_placeholder: Some("<name>"),
        },
        CliOption {
            name: "--profile",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Compiler profile TOML (default: bundled ericw-q1-bsp2-generated)",
            value_placeholder: Some("<profile.toml>"),
        },
    ]
}

/// Schema for `pack <engine.project.toml> --out <dir>`
pub fn pack_schema() -> &'static [CliOption] {
    &[CliOption {
        name: "--out",
        short: None,
        value_policy: OptionValuePolicy::Value,
        allow_spaced: true,
        allow_equals: true,
        repeatable: false,
        help: "Output directory",
        value_placeholder: Some("<dir>"),
    }]
}

/// Schema for `validate-bsp <file.bsp> [--palette <file.lmp>] [--strict]`
pub fn validate_bsp_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--palette",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Palette file (768 bytes) for texture/lightmap validation",
            value_placeholder: Some("<file.lmp>"),
        },
        CliOption {
            name: "--strict",
            short: None,
            value_policy: OptionValuePolicy::Flag,
            allow_spaced: true,
            allow_equals: false,
            repeatable: false,
            help: "Enforce strict/release validation policy",
            value_placeholder: None,
        },
    ]
}

/// Schema for `compile-bsp <source.map> --profile <profile.toml> --out <dir> [--palette <file.lmp>] [--tool-path <dir>] [--wad <file.wad>]`
pub fn compile_bsp_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--profile",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Compiler profile TOML file",
            value_placeholder: Some("<profile.toml>"),
        },
        CliOption {
            name: "--out",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Output directory for compiled .bsp and companions",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--palette",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Palette file (768 bytes)",
            value_placeholder: Some("<file.lmp>"),
        },
        CliOption {
            name: "--tool-path",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Directory containing qbsp, vis, light executables",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--wad",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: true,
            help: "WAD2 texture archive to stage alongside source (repeatable)",
            value_placeholder: Some("<file.wad>"),
        },
    ]
}

/// Parse the command, args, and positional args for a given engine_pack subcommand.
pub fn parse_command(_command: &str, schema: &[CliOption], args: &[String]) -> CliParseResult {
    parse_cli_args(schema, args)
}

#[derive(Clone, Copy, Debug)]
pub struct CommandSchema {
    pub name: &'static str,
    pub usage: &'static str,
    pub description: &'static str,
    pub options: &'static [CliOption],
}

pub fn command_schemas() -> [CommandSchema; 14] {
    [
        CommandSchema {
            name: "validate-package",
            usage: "engine_pack validate-package <package.toml> [--expected-package-id <id>] [--project-root <path>]",
            description: "Validate a package manifest.",
            options: validate_package_schema(),
        },
        CommandSchema {
            name: "validate-project",
            usage: "engine_pack validate-project <engine.project.toml>",
            description: "Validate a project manifest.",
            options: validate_project_schema(),
        },
        CommandSchema {
            name: "validate-scene",
            usage: "engine_pack validate-scene <scene.engine.scene.json> --project <engine.project.toml>",
            description: "Validate a scene against a project asset set.",
            options: validate_scene_schema(),
        },
        CommandSchema {
            name: "validate-bsp",
            usage: "engine_pack validate-bsp <file.bsp> [--palette <file.lmp>] [--strict]",
            description: "Validate a compiled BSP file through the parser.",
            options: validate_bsp_schema(),
        },
        CommandSchema {
            name: "compile-bsp",
            usage: "engine_pack compile-bsp <source.map> --profile <profile.toml> --out <dir> [--palette <file.lmp>] [--tool-path <dir>] [--wad <file.wad>]",
            description: "Compile a .map source to .bsp using a trusted external compiler.",
            options: compile_bsp_schema(),
        },
        CommandSchema {
            name: "new-app",
            usage: "engine_pack new-app <dir> --id <app_id> --name <display_name>",
            description: "Create a standalone support-crate app scaffold.",
            options: new_app_schema(),
        },
        CommandSchema {
            name: "new-project",
            usage: "engine_pack new-project <dir> --id <project_id> --name <name>",
            description: "Create a project manifest and starter scene.",
            options: new_project_schema(),
        },
        CommandSchema {
            name: "new-package",
            usage: "engine_pack new-package <path> --id <package_id> --name <display_name>",
            description: "Create a package manifest.",
            options: new_package_schema(),
        },
        CommandSchema {
            name: "scan-assets",
            usage: "engine_pack scan-assets <asset-root> [--package-id <id>]",
            description: "Scan an asset tree without following symlinks.",
            options: scan_assets_schema(),
        },
        CommandSchema {
            name: "add-asset",
            usage: "engine_pack add-asset <package.toml> --id <asset_id> --kind <kind> --path <path> [--tag <tag>]",
            description: "Append an asset record to a package manifest.",
            options: add_asset_schema(),
        },
        CommandSchema {
            name: "pack",
            usage: "engine_pack pack <engine.project.toml> --out <dir>",
            description: "Validate and publish a project package into a new output directory.",
            options: pack_schema(),
        },
        CommandSchema {
            name: "enhanced-dungeon-v3",
            usage: "engine_pack enhanced-dungeon-v3 --seed <u64> --preset <preset> --out <dir> [--extent <n>] [--tool-path <dir>] [--name <name>]",
            description: "Generate, compile, and publish an Enhanced V3 dungeon.",
            options: enhanced_dungeon_v3_schema(),
        },
        CommandSchema {
            name: "enhanced-dungeon",
            usage: "engine_pack enhanced-dungeon --seed <u64> --out <dir> [--tool-path <dir>] [--rooms <n>] [--loops <n>] [--vertical-edges <n>] [--name <name>]",
            description: "Generate, compile, and publish an Enhanced v2 dungeon.",
            options: enhanced_dungeon_schema(),
        },
        CommandSchema {
            name: "enhanced-dungeon-v3-richness-v1",
            usage: "engine_pack enhanced-dungeon-v3-richness-v1 --seed <u64> --preset <preset> --theme <theme> --out <dir> [--extent <n>] [--landmarks <n>] [--zones <n>] [--cave-mode <mode>] [--vertical-openings <n>] [--budget <n>] [--tool-path <dir>] [--name <name>]",
            description: "Generate, compile, and publish an Enhanced V3 Richness V1 dungeon.",
            options: enhanced_dungeon_v3_richness_v1_schema(),
        },
    ]
}

/// Produce help for `engine_pack` global usage from command declarations.
pub fn global_help() -> String {
    let mut lines = vec!["engine_pack commands:".to_string()];
    for command in command_schemas() {
        lines.push(format!(
            "  {}",
            command.usage.trim_start_matches("engine_pack ")
        ));
    }
    lines.push(String::new());
    lines.push("Use `engine_pack <command> --help` for command options.".to_string());
    lines.join("\n")
}

pub fn command_help(command: &str) -> Option<String> {
    command_schemas()
        .into_iter()
        .find(|schema| schema.name == command)
        .map(|schema| render_help(schema.options, schema.usage, schema.description))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_singleton_option() {
        let args: Vec<String> = ["--id", "a", "--id", "b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let result = parse_command("new-app", new_app_schema(), &args);
        assert!(!result.is_ok());
        assert!(result.errors.iter().any(|e| e.contains("duplicate")));
    }

    #[test]
    fn accepts_repeatable_tag() {
        let args: Vec<String> = [
            "--id", "asset.1", "--kind", "model", "--path", "m.glb", "--tag", "wall", "--tag",
            "stone",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let result = parse_command("add-asset", add_asset_schema(), &args);
        assert!(result.is_ok());
        let tags = result.repeated_values("--tag");
        assert_eq!(tags, vec!["wall", "stone"]);
    }

    #[test]
    fn accepts_spaced_and_equals_forms() {
        // spaced
        let args: Vec<String> = ["--id", "app.x", "--name", "App X"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let result = parse_command("new-app", new_app_schema(), &args);
        assert!(result.is_ok());
        assert_eq!(result.singleton_value("--id"), Some("app.x"));
        assert_eq!(result.singleton_value("--name"), Some("App X"));

        // equals
        let args: Vec<String> = ["--id=app.y", "--name=App Y"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let result = parse_command("new-app", new_app_schema(), &args);
        assert!(result.is_ok());
        assert_eq!(result.singleton_value("--id"), Some("app.y"));
        assert_eq!(result.singleton_value("--name"), Some("App Y"));
    }

    #[test]
    fn rejects_unknown_flags() {
        let args: Vec<String> = ["--bogus"].iter().map(|s| s.to_string()).collect();
        let result = parse_command("validate-project", validate_project_schema(), &args);
        assert!(!result.is_ok());
        assert!(result.errors.iter().any(|e| e.contains("unknown option")));
    }

    #[test]
    fn captures_positionals() {
        let args: Vec<String> = ["pos1", "pos2"].iter().map(|s| s.to_string()).collect();
        let result = parse_command("validate-project", validate_project_schema(), &args);
        assert_eq!(result.positionals, vec!["pos1", "pos2"]);
    }

    #[test]
    fn help_contains_all_commands() {
        let help = global_help();
        for cmd in [
            "validate-package",
            "validate-project",
            "validate-scene",
            "validate-bsp",
            "compile-bsp",
            "new-app",
            "new-project",
            "new-package",
            "scan-assets",
            "add-asset",
            "pack",
            "enhanced-dungeon",
            "enhanced-dungeon-v3",
            "enhanced-dungeon-v3-richness-v1",
        ] {
            assert!(help.contains(cmd), "help missing command: {cmd}");
        }
    }
}
