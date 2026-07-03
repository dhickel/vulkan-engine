use std::env;
use std::path::PathBuf;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaunchOptions {
    pub project_path: Option<PathBuf>,
    pub scene_path: Option<PathBuf>,
    pub record_debug_secs: Option<u64>,
    pub record_debug_interval_ms: Option<u64>,
    pub record_debug_path: Option<String>,
}

impl LaunchOptions {
    pub fn parse_env() -> Result<Self, String> {
        Self::parse(env::args().skip(1))
    }

    pub fn parse(args: impl IntoIterator<Item = impl Into<String>>) -> Result<Self, String> {
        let args: Vec<String> = args.into_iter().map(Into::into).collect();
        let mut options = LaunchOptions::default();
        let mut index = 0;

        while index < args.len() {
            let arg = args[index].as_str();
            match arg {
                "--project" => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| "--project requires a path argument".to_string())?;
                    options.project_path = Some(PathBuf::from(value));
                    index += 2;
                }
                "--scene" => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| "--scene requires a path argument".to_string())?;
                    options.scene_path = Some(PathBuf::from(value));
                    index += 2;
                }
                "--record_debug" => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| "--record_debug requires seconds".to_string())?;
                    options.record_debug_secs = Some(parse_positive_u64("--record_debug", value)?);
                    index += 2;
                }
                "--record_debug_interval" => {
                    let value = args.get(index + 1).ok_or_else(|| {
                        "--record_debug_interval requires milliseconds".to_string()
                    })?;
                    options.record_debug_interval_ms =
                        Some(parse_positive_u64("--record_debug_interval", value)?);
                    index += 2;
                }
                "--record_debug_path" => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| "--record_debug_path requires a file path".to_string())?;
                    options.record_debug_path = Some(value.to_string());
                    index += 2;
                }
                _ if arg.starts_with("--project=") => {
                    options.project_path = Some(PathBuf::from(&arg["--project=".len()..]));
                    index += 1;
                }
                _ if arg.starts_with("--scene=") => {
                    options.scene_path = Some(PathBuf::from(&arg["--scene=".len()..]));
                    index += 1;
                }
                _ if arg.starts_with("--record_debug=") => {
                    options.record_debug_secs = Some(parse_positive_u64(
                        "--record_debug",
                        &arg["--record_debug=".len()..],
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--record_debug_interval=") => {
                    options.record_debug_interval_ms = Some(parse_positive_u64(
                        "--record_debug_interval",
                        &arg["--record_debug_interval=".len()..],
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--record_debug_path=") => {
                    options.record_debug_path =
                        Some(arg["--record_debug_path=".len()..].to_string());
                    index += 1;
                }
                _ => {
                    index += 1;
                }
            }
        }

        Ok(options)
    }
}

fn parse_positive_u64(flag: &str, value: &str) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("{flag} expects a positive integer, got '{value}'"))?;
    if parsed == 0 {
        return Err(format!("{flag} expects a value >= 1, got '{value}'"));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_project_scene_and_debug_flags() {
        let options = LaunchOptions::parse([
            "--project",
            "engine.project.toml",
            "--scene=scenes/start.engine.scene.json",
            "--record_debug=10",
            "--record_debug_interval",
            "50",
            "--record_debug_path=.internal-dev/debug_reports/editor.jsonl",
        ])
        .expect("launch options should parse");

        assert_eq!(
            options.project_path,
            Some(PathBuf::from("engine.project.toml"))
        );
        assert_eq!(
            options.scene_path,
            Some(PathBuf::from("scenes/start.engine.scene.json"))
        );
        assert_eq!(options.record_debug_secs, Some(10));
        assert_eq!(options.record_debug_interval_ms, Some(50));
        assert_eq!(
            options.record_debug_path.as_deref(),
            Some(".internal-dev/debug_reports/editor.jsonl")
        );
    }

    #[test]
    fn reject_zero_debug_duration() {
        let err = LaunchOptions::parse(["--record_debug=0"]).expect_err("zero should fail");
        assert!(err.contains("value >= 1"));
    }
}
