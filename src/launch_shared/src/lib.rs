//! Shared launch argument parsing utilities used by both the root `engine`
//! runtime launcher and the `editor` application.
//!
//! Extracted during gate review remediation (AGR-008) to eliminate ~150 lines
//! of duplicated parsing logic between `src/launch.rs` and `apps/editor/src/launch.rs`.

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
        return Err(
            "--capture_frame and --capture_frames cannot be used together".to_string(),
        );
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
