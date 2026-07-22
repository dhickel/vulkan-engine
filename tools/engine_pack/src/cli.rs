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

pub fn command_schemas() -> [CommandSchema; 9] {
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
            "new-app",
            "new-project",
            "new-package",
            "scan-assets",
            "add-asset",
            "pack",
        ] {
            assert!(help.contains(cmd), "help missing command: {cmd}");
        }
    }
}
