use std::env;
use std::path::PathBuf;

use renderer::CaptureTarget;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaunchOptions {
    pub project_path: Option<PathBuf>,
    pub scene_path: Option<PathBuf>,
    pub record_debug_secs: Option<u64>,
    pub record_debug_interval_ms: Option<u64>,
    pub record_debug_path: Option<String>,
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
                "--capture_frame" => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| "--capture_frame requires a frame number".to_string())?;
                    options.capture_frame = Some(parse_u32("--capture_frame", value)?);
                    index += 2;
                }
                "--capture_frame_path" => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| "--capture_frame_path requires a file path".to_string())?;
                    options.capture_frame_path = Some(PathBuf::from(value));
                    index += 2;
                }
                "--capture_frames" => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| "--capture_frames requires a frame count".to_string())?;
                    options.capture_frames = Some(parse_positive_u32("--capture_frames", value)?);
                    index += 2;
                }
                "--capture_frame_start" => {
                    let value = args.get(index + 1).ok_or_else(|| {
                        "--capture_frame_start requires a frame number".to_string()
                    })?;
                    options.capture_frame_start = Some(parse_u32("--capture_frame_start", value)?);
                    index += 2;
                }
                "--capture_frame_interval" => {
                    let value = args.get(index + 1).ok_or_else(|| {
                        "--capture_frame_interval requires a frame interval".to_string()
                    })?;
                    options.capture_frame_interval =
                        Some(parse_positive_u32("--capture_frame_interval", value)?);
                    index += 2;
                }
                "--capture_dir" => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| "--capture_dir requires a directory path".to_string())?;
                    options.capture_dir = Some(PathBuf::from(value));
                    index += 2;
                }
                "--capture_target" => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| "--capture_target requires present or draw".to_string())?;
                    options.capture_target = parse_capture_target(value)?;
                    index += 2;
                }
                "--headless" => {
                    options.headless = true;
                    index += 1;
                }
                "--manual_capture_dir" => {
                    let value = args.get(index + 1).ok_or_else(|| {
                        "--manual_capture_dir requires a directory path".to_string()
                    })?;
                    options.manual_capture_dir = Some(PathBuf::from(value));
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
                _ if arg.starts_with("--capture_frame=") => {
                    options.capture_frame = Some(parse_u32(
                        "--capture_frame",
                        &arg["--capture_frame=".len()..],
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--capture_frame_path=") => {
                    options.capture_frame_path =
                        Some(PathBuf::from(&arg["--capture_frame_path=".len()..]));
                    index += 1;
                }
                _ if arg.starts_with("--capture_frames=") => {
                    options.capture_frames = Some(parse_positive_u32(
                        "--capture_frames",
                        &arg["--capture_frames=".len()..],
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--capture_frame_start=") => {
                    options.capture_frame_start = Some(parse_u32(
                        "--capture_frame_start",
                        &arg["--capture_frame_start=".len()..],
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--capture_frame_interval=") => {
                    options.capture_frame_interval = Some(parse_positive_u32(
                        "--capture_frame_interval",
                        &arg["--capture_frame_interval=".len()..],
                    )?);
                    index += 1;
                }
                _ if arg.starts_with("--capture_dir=") => {
                    options.capture_dir = Some(PathBuf::from(&arg["--capture_dir=".len()..]));
                    index += 1;
                }
                _ if arg.starts_with("--capture_target=") => {
                    options.capture_target =
                        parse_capture_target(&arg["--capture_target=".len()..])?;
                    index += 1;
                }
                _ if arg.starts_with("--manual_capture_dir=") => {
                    options.manual_capture_dir =
                        Some(PathBuf::from(&arg["--manual_capture_dir=".len()..]));
                    index += 1;
                }
                _ => {
                    index += 1;
                }
            }
        }

        validate_capture_options(&options)?;
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

fn parse_u32(flag: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("{flag} expects an integer, got '{value}'"))
}

fn parse_positive_u32(flag: &str, value: &str) -> Result<u32, String> {
    let parsed = parse_u32(flag, value)?;
    if parsed == 0 {
        return Err(format!("{flag} expects a value >= 1, got '{value}'"));
    }
    Ok(parsed)
}

fn parse_capture_target(value: &str) -> Result<CaptureTarget, String> {
    CaptureTarget::parse(value)
        .ok_or_else(|| format!("--capture_target expects present or draw, got '{value}'"))
}

fn validate_capture_options(options: &LaunchOptions) -> Result<(), String> {
    if options.capture_frame.is_some() && options.capture_frames.is_some() {
        return Err(
            "--capture_frame and --capture_frames cannot be used in the same launch".to_string(),
        );
    }
    if options.capture_frame_path.is_some() && options.capture_frame.is_none() {
        return Err("--capture_frame_path requires --capture_frame".to_string());
    }
    if options.capture_dir.is_some() && options.capture_frames.is_none() {
        return Err("--capture_dir requires --capture_frames".to_string());
    }
    if options.capture_frame_start.is_some() && options.capture_frames.is_none() {
        return Err("--capture_frame_start requires --capture_frames".to_string());
    }
    if options.capture_frame_interval.is_some() && options.capture_frames.is_none() {
        return Err("--capture_frame_interval requires --capture_frames".to_string());
    }
    Ok(())
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
        assert_eq!(options.capture_target, CaptureTarget::Present);
    }

    #[test]
    fn reject_zero_debug_duration() {
        let err = LaunchOptions::parse(["--record_debug=0"]).expect_err("zero should fail");
        assert!(err.contains("value >= 1"));
    }

    #[test]
    fn parse_capture_flags() {
        let options = LaunchOptions::parse([
            "--capture_frame=12",
            "--capture_frame_path",
            ".internal-dev/debug_reports/editor-frame.png",
            "--capture_target=draw",
            "--headless",
            "--manual_capture_dir=.internal-dev/debug_reports/manual-editor",
        ])
        .expect("capture flags should parse");

        assert_eq!(options.capture_frame, Some(12));
        assert_eq!(
            options.capture_frame_path,
            Some(PathBuf::from(
                ".internal-dev/debug_reports/editor-frame.png"
            ))
        );
        assert_eq!(options.capture_target, CaptureTarget::Draw);
        assert!(options.headless);
        assert_eq!(
            options.manual_capture_dir,
            Some(PathBuf::from(".internal-dev/debug_reports/manual-editor"))
        );
    }

    #[test]
    fn parse_capture_sequence_flags() {
        let options = LaunchOptions::parse([
            "--capture_frames",
            "4",
            "--capture_frame_start=20",
            "--capture_frame_interval",
            "5",
            "--capture_dir=.internal-dev/debug_reports/editor-captures",
        ])
        .expect("capture sequence flags should parse");

        assert_eq!(options.capture_frames, Some(4));
        assert_eq!(options.capture_frame_start, Some(20));
        assert_eq!(options.capture_frame_interval, Some(5));
        assert_eq!(
            options.capture_dir,
            Some(PathBuf::from(".internal-dev/debug_reports/editor-captures"))
        );
    }

    #[test]
    fn reject_invalid_capture_values() {
        assert!(LaunchOptions::parse(["--capture_frames=0"])
            .unwrap_err()
            .contains("value >= 1"));
        assert!(
            LaunchOptions::parse(["--capture_frames=2", "--capture_frame_interval=0"])
                .unwrap_err()
                .contains("value >= 1")
        );
        assert!(LaunchOptions::parse(["--capture_target=swapchain"])
            .unwrap_err()
            .contains("present or draw"));
        assert!(
            LaunchOptions::parse(["--capture_frame=1", "--capture_frames=2"])
                .unwrap_err()
                .contains("cannot be used")
        );
    }
}
