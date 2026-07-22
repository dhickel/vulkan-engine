//! Shared launch argument parsing utilities used by both the root `engine`
//! runtime launcher and the `editor` application.
//!
//! Extracted during gate review remediation (AGR-008) to eliminate ~150 lines
//! of duplicated parsing logic between `src/launch.rs` and `apps/editor/src/launch.rs`.
//!
//! Phase 10 adds a declarative CLI schema (table-driven per-option form policy,
//! occurrence tracking, duplicate singleton rejection, help generation) that
//! serves both the root launcher and `engine_pack` CLI surfaces.

use renderer::prelude::CaptureTarget;

/// Parse a flag value as a positive u64 (>= 1).
pub fn parse_positive_u64(flag: &str, value: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("{flag} expects a positive integer, got '{value}'"))?;
    if parsed == 0 {
        return Err(format!("{flag} expects a value >= 1, got '{value}'"));
    }
    Ok(parsed)
}

/// Parse a flag value as a positive u32 (>= 1).
pub fn parse_positive_u32(flag: &str, value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("{flag} expects a positive integer, got '{value}'"))?;
    if parsed == 0 {
        return Err(format!("{flag} expects a value >= 1, got '{value}'"));
    }
    Ok(parsed)
}

/// Parse a flag value as a u32 (any non-negative value).
pub fn parse_u32(flag: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("{flag} expects an integer, got '{value}'"))
}

/// Parse `--capture_target` flag value into a CaptureTarget.
pub fn parse_capture_target(value: &str) -> Result<CaptureTarget, String> {
    CaptureTarget::parse(value)
        .ok_or_else(|| format!("--capture_target expects present or draw, got '{value}'"))
}

/// Validate that capture-related flags are used correctly together.
///
/// Pass the individual option fields rather than the whole options struct,
/// so this function can be used by both the root launcher and editor
/// (which have slightly different struct layouts).
pub fn validate_capture_options(
    capture_frame: Option<u32>,
    capture_frame_path: Option<&std::path::Path>,
    capture_frames: Option<u32>,
    capture_frame_start: Option<u32>,
    capture_frame_interval: Option<u32>,
    capture_dir: Option<&std::path::Path>,
) -> Result<(), String> {
    if capture_frame.is_some() && capture_frames.is_some() {
        return Err("--capture_frame and --capture_frames cannot be used together".to_string());
    }
    if capture_frame_path.is_some() && capture_frame.is_none() {
        return Err("--capture_frame_path requires --capture_frame".to_string());
    }
    if capture_dir.is_some() && capture_frames.is_none() {
        return Err("--capture_dir requires --capture_frames".to_string());
    }
    if capture_frame_start.is_some() && capture_frames.is_none() {
        return Err("--capture_frame_start requires --capture_frames".to_string());
    }
    if capture_frame_interval.is_some() && capture_frames.is_none() {
        return Err("--capture_frame_interval requires --capture_frames".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Declarative CLI schema (Phase 10)
// ---------------------------------------------------------------------------

/// Describes how an option accepts its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionValuePolicy {
    /// The option is a boolean flag (no value).
    Flag,
    /// The option requires a value (spaced or equals form).
    Value,
}

/// The recognized form of an option occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionForm {
    /// `--flag` (flag) or `--flag value` (value)
    Spaced,
    /// `--flag=value` (value only)
    Equals,
    /// `--flag` for a flag option
    BareFlag,
}

/// Declaration of one CLI option (flag or value option).
#[derive(Clone, Debug)]
pub struct CliOption {
    /// Canonical long name, e.g. `"--project"`.
    pub name: &'static str,
    /// Short alias if any, e.g. `Some("-h")`.
    pub short: Option<&'static str>,
    /// Whether this option takes a value or is a flag.
    pub value_policy: OptionValuePolicy,
    /// Whether spaced form is allowed.
    pub allow_spaced: bool,
    /// Whether equals form is allowed.
    pub allow_equals: bool,
    /// If true the option may appear multiple times; if false duplicate is rejected.
    pub repeatable: bool,
    /// Human description for help.
    pub help: &'static str,
    /// If set, a value placeholder shown in help (e.g. `"<path>"`).
    pub value_placeholder: Option<&'static str>,
}

/// One occurrence of an option, carrying its form and (optional) value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliOccurrence {
    pub name: &'static str,
    pub form: OptionForm,
    pub value: Option<String>,
}

/// Result of parsing a set of options against a schema.
#[derive(Clone, Debug, Default)]
pub struct CliParseResult {
    pub occurrences: Vec<CliOccurrence>,
    pub positionals: Vec<String>,
    pub errors: Vec<String>,
}

impl CliParseResult {
    /// Return the first value for a singleton option, if present.
    pub fn singleton_value(&self, name: &str) -> Option<&str> {
        self.occurrences
            .iter()
            .find(|occ| occ.name == name)
            .and_then(|occ| occ.value.as_deref())
    }

    /// Return all values for a repeatable option.
    pub fn repeated_values(&self, name: &str) -> Vec<&str> {
        self.occurrences
            .iter()
            .filter(|occ| occ.name == name)
            .filter_map(|occ| occ.value.as_deref())
            .collect()
    }

    /// Return true if a flag option is present.
    pub fn flag_present(&self, name: &str) -> bool {
        self.occurrences.iter().any(|occ| occ.name == name)
    }

    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn into_result(self) -> Result<Self, String> {
        if self.errors.is_empty() {
            Ok(self)
        } else {
            Err(self.errors.join("\n"))
        }
    }
}

/// Parse raw CLI arguments against a declarative schema.
///
/// Returns `CliParseResult` containing matched occurrences, unconsumed
/// positionals, and any errors. This is the core parser shared by the
/// root launcher and `engine_pack`.
pub fn parse_cli_args(schema: &[CliOption], args: &[String]) -> CliParseResult {
    let mut result = CliParseResult::default();
    let mut index = 0;

    while index < args.len() {
        let arg = args[index].as_str();

        // Spaced form: `--flag value` or `--flag` (flag-only)
        if let Some(opt) = schema
            .iter()
            .find(|o| o.name == arg || o.short == Some(arg))
        {
            if !opt.allow_spaced {
                result.errors.push(format!(
                    "spaced form not allowed for '{}'; use '{}=' form",
                    arg, opt.name
                ));
                index += 1;
                continue;
            }
            match opt.value_policy {
                OptionValuePolicy::Flag => {
                    check_duplicate_singleton(&mut result, opt);
                    result.occurrences.push(CliOccurrence {
                        name: opt.name,
                        form: OptionForm::BareFlag,
                        value: None,
                    });
                    index += 1;
                }
                OptionValuePolicy::Value => {
                    if index + 1 >= args.len() || args[index + 1].starts_with('-') {
                        result.errors.push(format!("'{}' requires a value", arg));
                        index += 1;
                        continue;
                    }
                    check_duplicate_singleton(&mut result, opt);
                    let value = args[index + 1].clone();
                    result.occurrences.push(CliOccurrence {
                        name: opt.name,
                        form: OptionForm::Spaced,
                        value: Some(value),
                    });
                    index += 2;
                }
            }
            continue;
        }

        // Equals form: `--flag=value`
        if let Some(eq_pos) = arg.find('=') {
            let flag = &arg[..eq_pos];
            if let Some(opt) = schema.iter().find(|o| o.name == flag) {
                if !opt.allow_equals {
                    result.errors.push(format!(
                        "equals form not allowed for '{}'; use spaced form",
                        flag
                    ));
                    index += 1;
                    continue;
                }
                if opt.value_policy == OptionValuePolicy::Flag {
                    result
                        .errors
                        .push(format!("'{}' is a flag and does not accept a value", flag));
                    index += 1;
                    continue;
                }
                let value = arg[eq_pos + 1..].to_string();
                if value.is_empty() {
                    result.errors.push(format!("'{}' requires a value", flag));
                    index += 1;
                    continue;
                }
                check_duplicate_singleton(&mut result, opt);
                result.occurrences.push(CliOccurrence {
                    name: opt.name,
                    form: OptionForm::Equals,
                    value: Some(value),
                });
                index += 1;
                continue;
            }
        }

        // Not a recognized flag → positional (or error if starts with '-')
        if arg.starts_with('-') {
            result.errors.push(format!("unknown option '{}'", arg));
        } else {
            result.positionals.push(arg.to_string());
        }
        index += 1;
    }

    result
}

fn check_duplicate_singleton(result: &mut CliParseResult, opt: &CliOption) {
    if !opt.repeatable && result.occurrences.iter().any(|o| o.name == opt.name) {
        result.errors.push(format!(
            "duplicate option '{}' (only one allowed)",
            opt.name
        ));
    }
}

/// Shared root launcher schema.
pub fn root_launcher_schema() -> &'static [CliOption] {
    &[
        CliOption {
            name: "--help",
            short: Some("-h"),
            value_policy: OptionValuePolicy::Flag,
            allow_spaced: true,
            allow_equals: false,
            repeatable: false,
            help: "Print this help text.",
            value_placeholder: None,
        },
        CliOption {
            name: "--project",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Project manifest to launch. Required.",
            value_placeholder: Some("<path>"),
        },
        CliOption {
            name: "--scene",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Optional startup scene override.",
            value_placeholder: Some("<path>"),
        },
        CliOption {
            name: "--headless",
            short: None,
            value_policy: OptionValuePolicy::Flag,
            allow_spaced: true,
            allow_equals: false,
            repeatable: false,
            help: "Use the headless runtime path.",
            value_placeholder: None,
        },
        CliOption {
            name: "--capture_target",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Capture target (present or draw).",
            value_placeholder: Some("<present|draw>"),
        },
        CliOption {
            name: "--capture_frame",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Capture one positive frame index.",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--capture_frame_path",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Output path for --capture_frame.",
            value_placeholder: Some("<path>"),
        },
        CliOption {
            name: "--capture_frames",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Capture a positive number of frames.",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--capture_frame_start",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "First frame for --capture_frames.",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--capture_frame_interval",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Frame interval for --capture_frames.",
            value_placeholder: Some("<n>"),
        },
        CliOption {
            name: "--capture_dir",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Directory for --capture_frames.",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--manual_capture_dir",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Directory used for manual capture requests.",
            value_placeholder: Some("<dir>"),
        },
        CliOption {
            name: "--record_debug",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Write debug timing records for this many seconds.",
            value_placeholder: Some("<seconds>"),
        },
        CliOption {
            name: "--record_debug_interval",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Debug timing sample interval in milliseconds.",
            value_placeholder: Some("<ms>"),
        },
        CliOption {
            name: "--record_debug_path",
            short: None,
            value_policy: OptionValuePolicy::Value,
            allow_spaced: true,
            allow_equals: true,
            repeatable: false,
            help: "Debug timing JSONL output path.",
            value_placeholder: Some("<path>"),
        },
    ]
}

/// Render root launcher help from the shared schema.
pub fn render_root_launcher_help() -> String {
    render_help(
        root_launcher_schema(),
        "engine --project <path> [options]",
        "Root runtime launcher options.",
    )
}

/// Render a help string from a schema and optional usage line.
pub fn render_help(schema: &[CliOption], usage: &str, description: &str) -> String {
    let mut lines = Vec::new();
    if !description.is_empty() {
        lines.push(description.to_string());
        lines.push(String::new());
    }
    lines.push(format!("Usage: {usage}"));
    lines.push(String::new());
    lines.push("Options:".to_string());

    for opt in schema {
        let mut flag = String::new();
        if let Some(short) = opt.short {
            flag.push_str(short);
            flag.push_str(", ");
        }
        flag.push_str(opt.name);
        if opt.value_policy == OptionValuePolicy::Value {
            if let Some(placeholder) = opt.value_placeholder {
                flag.push(' ');
                flag.push_str(placeholder);
            } else {
                flag.push_str(" <value>");
            }
            if opt.allow_equals {
                flag.push_str(&format!(
                    " (or {}={})",
                    opt.name,
                    opt.value_placeholder.unwrap_or("value")
                ));
            }
        }
        lines.push(format!("  {:<44} {}", flag, opt.help));
    }

    lines.join("\n")
}
