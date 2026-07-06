use std::fmt;
use std::path::PathBuf;

pub use renderer::prelude::CaptureTarget;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchOptions {
    pub project_path: PathBuf,
    pub scene_path: Option<PathBuf>,
    pub record_debug_secs: Option<u64>,
    pub record_debug_interval_ms: Option<u64>,
    pub record_debug_path: Option<PathBuf>,
    pub capture_frame: Option<u32>,
    pub capture_frame_path: Option<PathBuf>,
    pub capture_frames: Option<u32>,
    pub capture_frame_start: Option<u32>,
    pub capture_frame_interval: Option<u32>,
    pub capture_dir: Option<PathBuf>,
    pub capture_target: CaptureTarget,
    pub headless: bool,
    pub manual_capture_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PartialLaunchOptions {
    project_path: Option<PathBuf>,
    scene_path: Option<PathBuf>,
    record_debug_secs: Option<u64>,
    record_debug_interval_ms: Option<u64>,
    record_debug_path: Option<PathBuf>,
    capture_frame: Option<u32>,
    capture_frame_path: Option<PathBuf>,
    capture_frames: Option<u32>,
    capture_frame_start: Option<u32>,
    capture_frame_interval: Option<u32>,
    capture_dir: Option<PathBuf>,
    capture_target: CaptureTarget,
    headless: bool,
    manual_capture_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaunchCommand {
    Help,
    Run(LaunchOptions),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaunchError {
    Usage(String),
    Runtime(String),
}

impl LaunchError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::Runtime(_) => 1,
        }
    }

    pub fn is_usage(&self) -> bool {
        matches!(self, Self::Usage(_))
    }
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) => write!(f, "usage error: {message}"),
            Self::Runtime(message) => write!(f, "runtime error: {message}"),
        }
    }
}

pub fn parse_command(
    args: impl IntoIterator<Item = impl Into<String>>,
) -> Result<LaunchCommand, LaunchError> {
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        if args.len() == 1 {
            return Ok(LaunchCommand::Help);
        }
        return Err(LaunchError::Usage(
            "--help cannot be combined with runtime options".to_string(),
        ));
    }

    LaunchOptions::parse(args).map(LaunchCommand::Run)
}

impl LaunchOptions {
    pub fn parse(args: impl IntoIterator<Item = impl Into<String>>) -> Result<Self, LaunchError> {
        let args: Vec<String> = args.into_iter().map(Into::into).collect();
        let mut options = PartialLaunchOptions::default();
        let mut index = 0;

        while index < args.len() {
            let arg = args[index].as_str();
            match arg {
                "--project" => {
                    options.project_path =
                        Some(PathBuf::from(next_value(&args, index, "--project")?));
                    index += 2;
                }
                "--scene" => {
                    options.scene_path = Some(PathBuf::from(next_value(&args, index, "--scene")?));
                    index += 2;
                }
                "--record_debug" => {
                    let value = next_value(&args, index, "--record_debug")?;
                    options.record_debug_secs = Some(parse_positive_u64("--record_debug", value)?);
                    index += 2;
                }
                "--record_debug_interval" => {
                    let value = next_value(&args, index, "--record_debug_interval")?;
                    options.record_debug_interval_ms =
                        Some(parse_positive_u64("--record_debug_interval", value)?);
                    index += 2;
                }
                "--record_debug_path" => {
                    options.record_debug_path = Some(PathBuf::from(next_value(
                        &args,
                        index,
                        "--record_debug_path",
                    )?));
                    index += 2;
                }
                "--capture_frame" => {
                    let value = next_value(&args, index, "--capture_frame")?;
                    options.capture_frame = Some(parse_positive_u32("--capture_frame", value)?);
                    index += 2;
                }
                "--capture_frame_path" => {
                    options.capture_frame_path = Some(PathBuf::from(next_value(
                        &args,
                        index,
                        "--capture_frame_path",
                    )?));
                    index += 2;
                }
                "--capture_frames" => {
                    let value = next_value(&args, index, "--capture_frames")?;
                    options.capture_frames = Some(parse_positive_u32("--capture_frames", value)?);
                    index += 2;
                }
                "--capture_frame_start" => {
                    let value = next_value(&args, index, "--capture_frame_start")?;
                    options.capture_frame_start =
                        Some(parse_positive_u32("--capture_frame_start", value)?);
                    index += 2;
                }
                "--capture_frame_interval" => {
                    let value = next_value(&args, index, "--capture_frame_interval")?;
                    options.capture_frame_interval =
                        Some(parse_positive_u32("--capture_frame_interval", value)?);
                    index += 2;
                }
                "--capture_dir" => {
                    options.capture_dir =
                        Some(PathBuf::from(next_value(&args, index, "--capture_dir")?));
                    index += 2;
                }
                "--capture_target" => {
                    let value = next_value(&args, index, "--capture_target")?;
                    options.capture_target = parse_capture_target(value)?;
                    index += 2;
                }
                "--headless" => {
                    options.headless = true;
                    index += 1;
                }
                "--manual_capture_dir" => {
                    options.manual_capture_dir = Some(PathBuf::from(next_value(
                        &args,
                        index,
                        "--manual_capture_dir",
                    )?));
                    index += 2;
                }
                _ if arg.starts_with("--project=") => {
                    options.project_path = Some(PathBuf::from(inline_value(arg, "--project")?));
                    index += 1;
                }
                _ if arg.starts_with("--scene=") => {
                    options.scene_path = Some(PathBuf::from(inline_value(arg, "--scene")?));
                    index += 1;
                }
                _ if arg.starts_with("--record_debug=") => {
                    options.record_debug_secs = Some(parse_positive_u64(
                        "--record_debug",
                        inline_value(arg, "--record_debug")?,
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--record_debug_interval=") => {
                    options.record_debug_interval_ms = Some(parse_positive_u64(
                        "--record_debug_interval",
                        inline_value(arg, "--record_debug_interval")?,
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--record_debug_path=") => {
                    options.record_debug_path =
                        Some(PathBuf::from(inline_value(arg, "--record_debug_path")?));
                    index += 1;
                }
                _ if arg.starts_with("--capture_frame=") => {
                    options.capture_frame = Some(parse_positive_u32(
                        "--capture_frame",
                        inline_value(arg, "--capture_frame")?,
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--capture_frame_path=") => {
                    options.capture_frame_path =
                        Some(PathBuf::from(inline_value(arg, "--capture_frame_path")?));
                    index += 1;
                }
                _ if arg.starts_with("--capture_frames=") => {
                    options.capture_frames = Some(parse_positive_u32(
                        "--capture_frames",
                        inline_value(arg, "--capture_frames")?,
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--capture_frame_start=") => {
                    options.capture_frame_start = Some(parse_positive_u32(
                        "--capture_frame_start",
                        inline_value(arg, "--capture_frame_start")?,
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--capture_frame_interval=") => {
                    options.capture_frame_interval = Some(parse_positive_u32(
                        "--capture_frame_interval",
                        inline_value(arg, "--capture_frame_interval")?,
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--capture_dir=") => {
                    options.capture_dir = Some(PathBuf::from(inline_value(arg, "--capture_dir")?));
                    index += 1;
                }
                _ if arg.starts_with("--capture_target=") => {
                    options.capture_target =
                        parse_capture_target(inline_value(arg, "--capture_target")?)?;
                    index += 1;
                }
                _ if arg.starts_with("--manual_capture_dir=") => {
                    options.manual_capture_dir =
                        Some(PathBuf::from(inline_value(arg, "--manual_capture_dir")?));
                    index += 1;
                }
                _ if arg.starts_with('-') => {
                    return Err(LaunchError::Usage(format!("unknown flag '{arg}'")));
                }
                _ => {
                    return Err(LaunchError::Usage(format!(
                        "unexpected positional argument '{arg}'"
                    )));
                }
            }
        }

        options.finish()
    }
}

impl PartialLaunchOptions {
    fn finish(self) -> Result<LaunchOptions, LaunchError> {
        validate_capture_options(&self)?;
        let project_path = self.project_path.ok_or_else(|| {
            LaunchError::Usage("--project is required for the root runtime launcher".to_string())
        })?;

        Ok(LaunchOptions {
            project_path,
            scene_path: self.scene_path,
            record_debug_secs: self.record_debug_secs,
            record_debug_interval_ms: self.record_debug_interval_ms,
            record_debug_path: self.record_debug_path,
            capture_frame: self.capture_frame,
            capture_frame_path: self.capture_frame_path,
            capture_frames: self.capture_frames,
            capture_frame_start: self.capture_frame_start,
            capture_frame_interval: self.capture_frame_interval,
            capture_dir: self.capture_dir,
            capture_target: self.capture_target,
            headless: self.headless,
            manual_capture_dir: self.manual_capture_dir,
        })
    }
}

fn next_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str, LaunchError> {
    let value = args
        .get(index + 1)
        .ok_or_else(|| LaunchError::Usage(format!("{flag} requires a value")))?;
    if value.is_empty() || value.starts_with('-') {
        return Err(LaunchError::Usage(format!("{flag} requires a value")));
    }
    Ok(value)
}

fn inline_value<'a>(arg: &'a str, flag: &str) -> Result<&'a str, LaunchError> {
    let prefix = format!("{flag}=");
    let value = arg
        .strip_prefix(&prefix)
        .expect("caller checked inline flag prefix");
    if value.is_empty() {
        return Err(LaunchError::Usage(format!("{flag} requires a value")));
    }
    Ok(value)
}

fn parse_positive_u64(flag: &str, value: &str) -> Result<u64, LaunchError> {
    launch_shared::parse_positive_u64(flag, value).map_err(LaunchError::Usage)
}

fn parse_positive_u32(flag: &str, value: &str) -> Result<u32, LaunchError> {
    launch_shared::parse_positive_u32(flag, value).map_err(LaunchError::Usage)
}

fn parse_capture_target(value: &str) -> Result<CaptureTarget, LaunchError> {
    launch_shared::parse_capture_target(value).map_err(LaunchError::Usage)
}

fn validate_capture_options(options: &PartialLaunchOptions) -> Result<(), LaunchError> {
    launch_shared::validate_capture_options(
        options.capture_frame,
        options.capture_frame_path.as_deref(),
        options.capture_frames,
        options.capture_frame_start,
        options.capture_frame_interval,
        options.capture_dir.as_deref(),
    )
    .map_err(LaunchError::Usage)
}

pub fn usage() -> &'static str {
    "Usage: engine --project <path> [options]\n\
\n\
Root runtime launcher options:\n\
  -h, --help                         Print this help text.\n\
      --project <path>               Project manifest to launch. Required.\n\
      --project=<path>               Project manifest to launch. Required.\n\
      --scene <path>                 Optional startup scene override.\n\
      --scene=<path>                 Optional startup scene override.\n\
      --headless                     Use the headless runtime path.\n\
\n\
Capture options:\n\
      --capture_target <present|draw>\n\
      --capture_target=<present|draw>\n\
      --capture_frame <n>\n\
      --capture_frame=<n>\n\
      --capture_frame_path <path>\n\
      --capture_frame_path=<path>\n\
      --capture_frames <n>\n\
      --capture_frames=<n>\n\
      --capture_frame_start <n>\n\
      --capture_frame_start=<n>\n\
      --capture_frame_interval <n>\n\
      --capture_frame_interval=<n>\n\
      --capture_dir <dir>\n\
      --capture_dir=<dir>\n\
      --manual_capture_dir <dir>\n\
      --manual_capture_dir=<dir>\n\
\n\
Debug timing options:\n\
      --record_debug <seconds>\n\
      --record_debug=<seconds>\n\
      --record_debug_interval <ms>\n\
      --record_debug_interval=<ms>\n\
      --record_debug_path <path>\n\
      --record_debug_path=<path>\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<LaunchOptions, LaunchError> {
        LaunchOptions::parse(args.iter().copied())
    }

    #[test]
    fn help_is_a_command() {
        assert_eq!(parse_command(["--help"]), Ok(LaunchCommand::Help));
        assert!(usage().contains("--project"));
        assert!(usage().contains("--headless"));
        assert!(usage().contains("--capture_target <present|draw>"));
        assert!(usage().contains("--record_debug <seconds>"));
    }

    #[test]
    fn accepts_project_space_and_equals_forms() {
        assert_eq!(
            parse(&[
                "--project",
                "apps/editor/sample_project/engine.project.toml"
            ])
            .expect("space project should parse")
            .project_path,
            PathBuf::from("apps/editor/sample_project/engine.project.toml")
        );
        assert_eq!(
            parse(&["--project=apps/editor/sample_project/engine.project.toml"])
                .expect("equals project should parse")
                .project_path,
            PathBuf::from("apps/editor/sample_project/engine.project.toml")
        );
    }

    #[test]
    fn accepts_scene_debug_and_capture_forms() {
        let options = parse(&[
            "--project=engine.project.toml",
            "--scene",
            "scenes/start.engine.scene.json",
            "--headless",
            "--capture_frames=3",
            "--capture_frame_start",
            "5",
            "--capture_frame_interval=5",
            "--capture_dir",
            ".internal-dev/captures/sprint-04-runtime-launcher/headless-draw",
            "--capture_target=draw",
            "--manual_capture_dir=.internal-dev/captures/manual",
            "--record_debug",
            "10",
            "--record_debug_interval=50",
            "--record_debug_path",
            ".internal-dev/debug_reports/sprint-04-runtime-launcher/root-runtime-timing.jsonl",
        ])
        .expect("full launcher options should parse");

        assert_eq!(
            options.scene_path,
            Some(PathBuf::from("scenes/start.engine.scene.json"))
        );
        assert!(options.headless);
        assert_eq!(options.capture_frames, Some(3));
        assert_eq!(options.capture_frame_start, Some(5));
        assert_eq!(options.capture_frame_interval, Some(5));
        assert_eq!(options.capture_target, CaptureTarget::Draw);
        assert_eq!(options.record_debug_secs, Some(10));
        assert_eq!(options.record_debug_interval_ms, Some(50));
    }

    #[test]
    fn rejects_missing_project() {
        let err = parse(&["--headless"]).expect_err("project is required");
        assert_eq!(err.exit_code(), 2);
        assert!(err.to_string().contains("--project is required"));
    }

    #[test]
    fn rejects_unknown_flags_and_positionals() {
        assert!(parse(&["--project=engine.project.toml", "--bogus"])
            .unwrap_err()
            .to_string()
            .contains("unknown flag"));
        assert!(parse(&["engine.project.toml"])
            .unwrap_err()
            .to_string()
            .contains("unexpected positional"));
    }

    #[test]
    fn rejects_missing_values() {
        assert!(parse(&["--project"])
            .unwrap_err()
            .to_string()
            .contains("requires a value"));
        assert!(parse(&["--project="])
            .unwrap_err()
            .to_string()
            .contains("requires a value"));
        assert!(parse(&["--project", "--headless"])
            .unwrap_err()
            .to_string()
            .contains("requires a value"));
    }

    #[test]
    fn rejects_invalid_capture_target() {
        let err = parse(&[
            "--project=apps/editor/sample_project/engine.project.toml",
            "--capture_target",
            "swapchain",
        ])
        .expect_err("swapchain is not accepted");
        let message = err.to_string();
        assert!(message.contains("present"));
        assert!(message.contains("draw"));
        assert!(message.contains("swapchain"));
    }

    #[test]
    fn rejects_zero_count_interval_and_debug_values() {
        assert!(parse(&["--project=p", "--capture_frames=0"])
            .unwrap_err()
            .to_string()
            .contains("value >= 1"));
        assert!(parse(&[
            "--project=p",
            "--capture_frames=2",
            "--capture_frame_interval=0",
        ])
        .unwrap_err()
        .to_string()
        .contains("value >= 1"));
        assert!(parse(&["--project=p", "--record_debug=0"])
            .unwrap_err()
            .to_string()
            .contains("value >= 1"));
    }

    #[test]
    fn rejects_capture_dependency_errors() {
        assert!(parse(&["--project=p", "--capture_frame_path=frame.png"])
            .unwrap_err()
            .to_string()
            .contains("requires --capture_frame"));
        assert!(parse(&["--project=p", "--capture_dir=frames"])
            .unwrap_err()
            .to_string()
            .contains("requires --capture_frames"));
        assert!(parse(&["--project=p", "--capture_frame_start=2"])
            .unwrap_err()
            .to_string()
            .contains("requires --capture_frames"));
        assert!(parse(&["--project=p", "--capture_frame_interval=2"])
            .unwrap_err()
            .to_string()
            .contains("requires --capture_frames"));
    }

    #[test]
    fn rejects_single_and_sequence_capture_together() {
        assert!(
            parse(&["--project=p", "--capture_frame=1", "--capture_frames=2"])
                .unwrap_err()
                .to_string()
                .contains("cannot be used together")
        );
    }
}
