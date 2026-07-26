//! Trusted external BSP compiler invocation.
//!
//! Provides shell-free subprocess execution of ericw-tools (qbsp, vis, light)
//! with controlled arguments, working directory, environment, output-size
//! limits, and post-compile validation through the `bsp` parser.
//!
//! This is a direct process boundary — NOT a security sandbox. The compiler
//! executable is outside the parser trust boundary.

use bsp::{
    BspLightingCalibration, BspLoader, BspPackageManifest, BspReport, CompanionBinding,
    CompanionKind, CompileResult, CompilerHashes, CompilerProfile, CompilerProvenance, LoadOptions,
    PackageContentHash,
};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, Instant};

const MAX_VERSION_STREAM_BYTES: u64 = 1024 * 1024;

/// Parse a `CompilerProfile` from a TOML string.
pub fn parse_compiler_profile(content: &str) -> Result<CompilerProfile, String> {
    let root: toml::Value = toml::from_str(content).map_err(|e| format!("invalid TOML: {e}"))?;
    let table = root
        .as_table()
        .ok_or_else(|| "profile root is not a table".to_string())?;

    let hashes = table
        .get("expected_hashes")
        .map(|value| {
            let h = value
                .as_table()
                .ok_or_else(|| "expected_hashes must be a table".to_string())?;
            let hashes = CompilerHashes {
                qbsp_sha256: required_sha256(h, "qbsp_sha256")?,
                vis_sha256: required_sha256(h, "vis_sha256")?,
                light_sha256: required_sha256(h, "light_sha256")?,
            };
            Ok::<CompilerHashes, String>(hashes)
        })
        .transpose()?;

    Ok(CompilerProfile {
        name: table_str(table, "name")?,
        compiler_identity: table_str(table, "compiler_identity")?,
        required_version: table_str(table, "required_version")?,
        qbsp_executable: table_str(table, "qbsp_executable")?,
        vis_executable: table_str(table, "vis_executable")?,
        light_executable: table_str(table, "light_executable")?,
        default_qbsp_args: table_str_vec(table, "default_qbsp_args"),
        default_vis_args: table_str_vec(table, "default_vis_args"),
        default_light_args: table_str_vec(table, "default_light_args"),
        timeout_seconds: table_u64_with_default(table, "timeout_seconds", 120)?,
        max_output_size: table_u64_with_default(table, "max_output_size", 128 * 1024 * 1024)?,
        expected_hashes: hashes,
    })
}

/// Serialize a `BspPackageManifest` to a TOML string.
pub fn manifest_to_toml(manifest: &BspPackageManifest) -> Result<String, String> {
    use toml::Value;

    let mut root = toml::Table::new();

    root.insert(
        "format_version".into(),
        Value::Integer(manifest.format_version as i64),
    );
    root.insert("asset_id".into(), Value::String(manifest.asset_id.clone()));
    root.insert(
        "display_name".into(),
        Value::String(manifest.display_name.clone()),
    );
    root.insert(
        "bsp_path".into(),
        Value::String(path_to_string(&manifest.bsp_path)),
    );
    root.insert(
        "palette_path".into(),
        Value::String(path_to_string(&manifest.palette_path)),
    );
    root.insert("strict".into(), Value::Boolean(manifest.strict));

    if !manifest.wad_roots.is_empty() {
        root.insert(
            "wad_roots".into(),
            Value::Array(
                manifest
                    .wad_roots
                    .iter()
                    .map(|p| Value::String(path_to_string(p)))
                    .collect(),
            ),
        );
    }
    if !manifest.texture_roots.is_empty() {
        root.insert(
            "texture_roots".into(),
            Value::Array(
                manifest
                    .texture_roots
                    .iter()
                    .map(|p| Value::String(path_to_string(p)))
                    .collect(),
            ),
        );
    }
    if !manifest.model_mappings.is_empty() {
        let mappings: Vec<Value> = manifest
            .model_mappings
            .iter()
            .map(|(classname, model_id)| {
                let mut table = toml::Table::new();
                table.insert("classname".into(), Value::String(classname.clone()));
                table.insert("model_id".into(), Value::String(model_id.clone()));
                Value::Table(table)
            })
            .collect();
        root.insert("model_mappings".into(), Value::Array(mappings));
    }

    if let Some(scale) = manifest.scale_override {
        root.insert("scale_override".into(), Value::Float(scale as f64));
    }

    // Lighting calibration
    {
        let mut cal = toml::Table::new();
        cal.insert(
            "overbright".into(),
            Value::Float(manifest.lighting_calibration.overbright as f64),
        );
        cal.insert(
            "light_scale".into(),
            Value::Float(manifest.lighting_calibration.light_scale as f64),
        );
        cal.insert(
            "saturation".into(),
            Value::Float(manifest.lighting_calibration.saturation as f64),
        );
        root.insert("lighting_calibration".into(), Value::Table(cal));
    }

    // Compiler provenance
    if let Some(ref prov) = manifest.compiler_provenance {
        let mut p = toml::Table::new();
        p.insert(
            "compiler_identity".into(),
            Value::String(prov.compiler_identity.clone()),
        );
        p.insert(
            "compiler_version".into(),
            Value::String(prov.compiler_version.clone()),
        );
        if !prov.qbsp_args.is_empty() {
            p.insert(
                "qbsp_args".into(),
                Value::Array(
                    prov.qbsp_args
                        .iter()
                        .map(|a| Value::String(a.clone()))
                        .collect(),
                ),
            );
        }
        if !prov.vis_args.is_empty() {
            p.insert(
                "vis_args".into(),
                Value::Array(
                    prov.vis_args
                        .iter()
                        .map(|a| Value::String(a.clone()))
                        .collect(),
                ),
            );
        }
        if !prov.light_args.is_empty() {
            p.insert(
                "light_args".into(),
                Value::Array(
                    prov.light_args
                        .iter()
                        .map(|a| Value::String(a.clone()))
                        .collect(),
                ),
            );
        }
        if !prov.source_hashes.is_empty() {
            p.insert(
                "source_hashes".into(),
                Value::Array(content_hashes_to_toml_values(&prov.source_hashes)),
            );
        }
        if !prov.output_hashes.is_empty() {
            p.insert(
                "output_hashes".into(),
                Value::Array(content_hashes_to_toml_values(&prov.output_hashes)),
            );
        }
        if let Some(ref hashes) = prov.compiler_hashes {
            let mut h = toml::Table::new();
            h.insert(
                "qbsp_sha256".into(),
                Value::String(hashes.qbsp_sha256.clone()),
            );
            h.insert(
                "vis_sha256".into(),
                Value::String(hashes.vis_sha256.clone()),
            );
            h.insert(
                "light_sha256".into(),
                Value::String(hashes.light_sha256.clone()),
            );
            p.insert("compiler_hashes".into(), Value::Table(h));
        }
        root.insert("compiler_provenance".into(), Value::Table(p));
    }

    // Companion bindings
    if !manifest.companion_bindings.is_empty() {
        let bindings: Vec<Value> = manifest
            .companion_bindings
            .iter()
            .map(|cb| {
                let mut t = toml::Table::new();
                t.insert("kind".into(), Value::String(cb.kind.as_str().into()));
                t.insert("path".into(), Value::String(path_to_string(&cb.path)));
                if let Some(ref hash) = cb.content_hash {
                    t.insert("content_hash".into(), Value::String(hash.clone()));
                }
                Value::Table(t)
            })
            .collect();
        root.insert("companion_bindings".into(), Value::Array(bindings));
    }

    toml::to_string_pretty(&Value::Table(root))
        .map_err(|e| format!("failed to serialize BSP package manifest: {e}"))
}

/// Deserialize a `BspPackageManifest` from a TOML string.
pub fn manifest_from_toml(content: &str) -> Result<BspPackageManifest, String> {
    let root: toml::Value = toml::from_str(content).map_err(|e| format!("invalid TOML: {e}"))?;
    let table = root
        .as_table()
        .ok_or_else(|| "manifest root is not a table".to_string())?;

    let format_version = table
        .get("format_version")
        .and_then(toml::Value::as_integer)
        .ok_or_else(|| "missing format_version".to_string())? as u32;
    let asset_id = table_str(table, "asset_id")?;
    let display_name = table_str(table, "display_name")?;
    let bsp_path = PathBuf::from(table_str(table, "bsp_path")?);
    let palette_path = PathBuf::from(table_str(table, "palette_path")?);
    let strict = table
        .get("strict")
        .and_then(toml::Value::as_bool)
        .unwrap_or(false);

    let wad_roots = table
        .get("wad_roots")
        .and_then(toml::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(PathBuf::from))
                .collect()
        })
        .unwrap_or_default();

    let texture_roots = table
        .get("texture_roots")
        .and_then(toml::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(PathBuf::from))
                .collect()
        })
        .unwrap_or_default();

    let model_mappings = table
        .get("model_mappings")
        .and_then(toml::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let t = v.as_table()?;
                    Some((
                        t.get("classname")?.as_str()?.to_string(),
                        t.get("model_id")?.as_str()?.to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();

    let scale_override = table
        .get("scale_override")
        .and_then(toml::Value::as_float)
        .map(|f| f as f32);

    let lighting_calibration = {
        let default = BspLightingCalibration::default();
        let cal = table
            .get("lighting_calibration")
            .and_then(toml::Value::as_table);
        BspLightingCalibration {
            overbright: cal
                .and_then(|c| c.get("overbright"))
                .and_then(toml::Value::as_float)
                .map(|f| f as f32)
                .unwrap_or(default.overbright),
            light_scale: cal
                .and_then(|c| c.get("light_scale"))
                .and_then(toml::Value::as_float)
                .map(|f| f as f32)
                .unwrap_or(default.light_scale),
            saturation: cal
                .and_then(|c| c.get("saturation"))
                .and_then(toml::Value::as_float)
                .map(|f| f as f32)
                .unwrap_or(default.saturation),
        }
    };

    let compiler_provenance = table
        .get("compiler_provenance")
        .and_then(toml::Value::as_table)
        .map(|p| {
            let hashes = p
                .get("compiler_hashes")
                .and_then(toml::Value::as_table)
                .map(|h| CompilerHashes {
                    qbsp_sha256: table_str(h, "qbsp_sha256").unwrap_or_default(),
                    vis_sha256: table_str(h, "vis_sha256").unwrap_or_default(),
                    light_sha256: table_str(h, "light_sha256").unwrap_or_default(),
                });
            Ok::<CompilerProvenance, String>(CompilerProvenance {
                compiler_identity: table_str(p, "compiler_identity").unwrap_or_default(),
                compiler_version: table_str(p, "compiler_version").unwrap_or_default(),
                qbsp_args: table_str_vec(p, "qbsp_args"),
                vis_args: table_str_vec(p, "vis_args"),
                light_args: table_str_vec(p, "light_args"),
                source_hashes: content_hashes_from_toml(p, "source_hashes"),
                output_hashes: content_hashes_from_toml(p, "output_hashes"),
                compiler_hashes: hashes,
            })
        })
        .transpose()?;

    let companion_bindings = table
        .get("companion_bindings")
        .and_then(toml::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let t = v.as_table()?;
                    let kind_str = t.get("kind")?.as_str()?;
                    let kind = CompanionKind::from_str(kind_str)?;
                    let path = PathBuf::from(t.get("path")?.as_str()?);
                    let content_hash = t
                        .get("content_hash")
                        .and_then(toml::Value::as_str)
                        .map(String::from);
                    Some(CompanionBinding {
                        kind,
                        path,
                        content_hash,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(BspPackageManifest {
        format_version,
        asset_id,
        display_name,
        bsp_path,
        palette_path,
        wad_roots,
        texture_roots,
        model_mappings,
        scale_override,
        lighting_calibration,
        compiler_provenance,
        strict,
        companion_bindings,
    })
}

fn path_to_string(path: &PathBuf) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn content_hashes_to_toml_values(hashes: &[PackageContentHash]) -> Vec<toml::Value> {
    hashes
        .iter()
        .map(|hash| {
            let mut table = toml::Table::new();
            table.insert(
                "path".into(),
                toml::Value::String(path_to_string(&hash.path)),
            );
            table.insert("sha256".into(), toml::Value::String(hash.sha256.clone()));
            toml::Value::Table(table)
        })
        .collect()
}

fn content_hashes_from_toml(table: &toml::Table, key: &str) -> Vec<PackageContentHash> {
    table
        .get(key)
        .and_then(toml::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|value| {
                    let table = value.as_table()?;
                    Some(PackageContentHash {
                        path: PathBuf::from(table.get("path")?.as_str()?),
                        sha256: table.get("sha256")?.as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn table_str(table: &toml::Table, key: &str) -> Result<String, String> {
    table
        .get(key)
        .and_then(toml::Value::as_str)
        .map(String::from)
        .ok_or_else(|| format!("missing or invalid field: {key}"))
}

fn table_str_vec(table: &toml::Table, key: &str) -> Vec<String> {
    table
        .get(key)
        .and_then(toml::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn table_u64_with_default(table: &toml::Table, key: &str, default: u64) -> Result<u64, String> {
    let Some(value) = table.get(key) else {
        return Ok(default);
    };
    let Some(integer) = value.as_integer() else {
        return Err(format!("{key} must be an integer"));
    };
    if integer <= 0 {
        return Err(format!("{key} must be positive"));
    }
    Ok(integer as u64)
}

fn required_sha256(table: &toml::Table, key: &str) -> Result<String, String> {
    let value = table_str(table, key)?;
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "expected_hashes.{key} must be a 64-character SHA-256 hex digest"
        ));
    }
    Ok(value.to_ascii_lowercase())
}

/// Error returned by compiler operations.
#[derive(Debug)]
pub enum CompilerError {
    /// I/O error during execution.
    Io {
        message: String,
        source: std::io::Error,
    },
    /// Compiler executable not found.
    NotFound {
        executable: String,
        searched_paths: Vec<PathBuf>,
    },
    /// Compiler version check failed.
    VersionMismatch {
        executable: String,
        expected: String,
        found: String,
    },
    /// Compiler executable hash did not match the pinned profile.
    HashMismatch {
        executable: String,
        expected: String,
        found: String,
    },
    /// Compiler returned non-zero exit code.
    CompilationFailed {
        stage: String,
        exit_code: i32,
        stdout: String,
        stderr: String,
    },
    /// Compiler returned success but reported an authoring warning. Generated
    /// BSP publication is warning-free and fails closed on these diagnostics.
    CompilerWarning {
        stage: String,
        stdout: String,
        stderr: String,
    },
    /// Compiler timed out.
    Timeout { stage: String, timeout_seconds: u64 },
    /// Output file not produced.
    MissingOutput {
        expected_path: PathBuf,
        stage: String,
    },
    /// Output exceeds size limit.
    OutputTooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    /// Post-compile BSP validation failed.
    ValidationFailed(BspReport),
    /// Invalid profile configuration.
    InvalidProfile(String),
    /// Source file missing or unreadable.
    SourceError { path: PathBuf, message: String },
    /// Source input is a symlink or non-regular file.
    NonRegularInput { path: PathBuf, kind: String },
    /// Compiler emitted a missing-texture diagnostic.
    MissingTexture {
        stage: String,
        stdout: String,
        stderr: String,
    },
    /// Compiler leaked a pointfile (.pts or .prt).
    PointfileLeaked { path: PathBuf, stage: String },
    /// Compiler output stream exceeded size bound.
    StreamBoundExceeded {
        stage: String,
        stream: String,
        limit: u64,
    },
    /// BSP magic mismatch with profile expectation.
    BspMagicMismatch {
        expected: String,
        found: String,
        path: PathBuf,
    },
    /// Lit file is invalid (wrong header, version, or size mismatch).
    InvalidLit { path: PathBuf, reason: String },
}

impl std::fmt::Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilerError::Io { message, source } => {
                write!(f, "{message}: {source}")
            }
            CompilerError::NotFound {
                executable,
                searched_paths,
            } => {
                write!(
                    f,
                    "executable '{executable}' not found; searched: {:?}",
                    searched_paths
                )
            }
            CompilerError::VersionMismatch {
                executable,
                expected,
                found,
            } => {
                write!(
                    f,
                    "version mismatch for '{executable}': expected '{expected}', found '{found}'"
                )
            }
            CompilerError::HashMismatch {
                executable,
                expected,
                found,
            } => {
                write!(
                    f,
                    "hash mismatch for '{executable}': expected '{expected}', found '{found}'"
                )
            }
            CompilerError::CompilationFailed {
                stage,
                exit_code,
                stdout,
                stderr,
            } => {
                write!(
                    f,
                    "{stage} failed (exit {exit_code}):\nstdout:\n{stdout}\nstderr:\n{stderr}"
                )
            }
            CompilerError::CompilerWarning {
                stage,
                stdout,
                stderr,
            } => {
                write!(
                    f,
                    "{stage} reported a compiler warning:\nstdout:\n{stdout}\nstderr:\n{stderr}"
                )
            }
            CompilerError::Timeout {
                stage,
                timeout_seconds,
            } => {
                write!(f, "{stage} timed out after {timeout_seconds}s")
            }
            CompilerError::MissingOutput {
                expected_path,
                stage,
            } => {
                write!(
                    f,
                    "{stage} did not produce expected output '{}'",
                    expected_path.display()
                )
            }
            CompilerError::OutputTooLarge { path, size, limit } => {
                write!(
                    f,
                    "output '{}' is too large: {size} bytes > {limit} bytes limit",
                    path.display()
                )
            }
            CompilerError::ValidationFailed(report) => {
                write!(f, "BSP validation failed: {report}")
            }
            CompilerError::InvalidProfile(msg) => {
                write!(f, "invalid compiler profile: {msg}")
            }
            CompilerError::SourceError { path, message } => {
                write!(f, "source error '{}': {message}", path.display())
            }
            CompilerError::NonRegularInput { path, kind } => {
                write!(f, "non-regular input '{}' is a {kind}", path.display())
            }
            CompilerError::MissingTexture {
                stage,
                stdout,
                stderr,
            } => {
                write!(
                    f,
                    "{stage} reported missing texture:\nstdout:\n{stdout}\nstderr:\n{stderr}"
                )
            }
            CompilerError::PointfileLeaked { path, stage } => {
                write!(f, "{stage} leaked pointfile '{}'", path.display())
            }
            CompilerError::StreamBoundExceeded {
                stage,
                stream,
                limit,
            } => {
                write!(f, "{stage} {stream} stream exceeded {limit} byte bound")
            }
            CompilerError::BspMagicMismatch {
                expected,
                found,
                path,
            } => {
                write!(
                    f,
                    "BSP magic mismatch in '{}': expected {expected}, found {found}",
                    path.display()
                )
            }
            CompilerError::InvalidLit { path, reason } => {
                write!(f, "invalid .lit file '{}': {reason}", path.display())
            }
        }
    }
}

impl std::error::Error for CompilerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CompilerError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CompilerError {
    fn from(e: std::io::Error) -> Self {
        CompilerError::Io {
            message: "I/O error".into(),
            source: e,
        }
    }
}

/// Minimal allowed environment for compiler subprocesses.
fn minimal_env() -> HashMap<String, String> {
    let mut env = HashMap::new();
    // PATH is required for dynamic linking of compiler executables
    if let Some(path) = std::env::var_os("PATH") {
        env.insert("PATH".into(), path.to_string_lossy().into());
    }
    // HOME / TMPDIR for temporary file creation by compilers
    if let Some(home) = std::env::var_os("HOME") {
        env.insert("HOME".into(), home.to_string_lossy().into());
    }
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        env.insert("TMPDIR".into(), tmp.to_string_lossy().into());
    }
    if let Some(tmp) = std::env::var_os("TEMP") {
        env.insert("TEMP".into(), tmp.to_string_lossy().into());
    }
    // USER for any logging
    if let Some(user) = std::env::var_os("USER") {
        env.insert("USER".into(), user.to_string_lossy().into());
    }
    env
}

/// Find an executable on PATH or at an explicit path.
fn resolve_executable(
    name_or_path: &str,
    tool_path: Option<&Path>,
) -> Result<PathBuf, CompilerError> {
    // If tool_path is provided, check there first
    if let Some(tool_dir) = tool_path {
        let candidate = tool_dir.join(name_or_path);
        if candidate.is_file() {
            return Ok(candidate);
        }
        // On Windows, try .exe suffix
        #[cfg(target_os = "windows")]
        {
            let exe_candidate = tool_dir.join(format!("{name_or_path}.exe"));
            if exe_candidate.is_file() {
                return Ok(exe_candidate);
            }
        }
    }

    // Try PATH
    if let Some(path_env) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_env) {
            let candidate = dir.join(name_or_path);
            if candidate.is_file() {
                return Ok(candidate);
            }
            #[cfg(target_os = "windows")]
            {
                let exe_candidate = dir.join(format!("{name_or_path}.exe"));
                if exe_candidate.is_file() {
                    return Ok(exe_candidate);
                }
            }
        }
    }

    Err(CompilerError::NotFound {
        executable: name_or_path.into(),
        searched_paths: tool_path.into_iter().map(PathBuf::from).collect(),
    })
}

/// Verify compiler version by running it with `--version` and checking output.
fn verify_compiler_version(
    executable: &Path,
    expected_version: &str,
    env: &HashMap<String, String>,
    timeout_seconds: u64,
) -> Result<(), CompilerError> {
    let output = run_direct_process(
        executable,
        &["--version".to_string()],
        None,
        env,
        timeout_seconds,
        MAX_VERSION_STREAM_BYTES,
        "version check",
    )?;

    let mut combined = process_output_text(&output);

    // Some tools only report version-like metadata in help output.
    if combined.trim().is_empty() || !combined.contains(expected_version) {
        let fallback = run_direct_process(
            executable,
            &["-help".to_string()],
            None,
            env,
            timeout_seconds,
            MAX_VERSION_STREAM_BYTES,
            "version help check",
        )?;
        let fallback_text = process_output_text(&fallback);
        if !fallback_text.trim().is_empty() {
            combined = if combined.trim().is_empty() {
                fallback_text
            } else {
                format!("{combined}\n{fallback_text}")
            };
        }
    }

    if combined.contains(expected_version) {
        return Ok(());
    }

    Err(CompilerError::VersionMismatch {
        executable: executable.display().to_string(),
        expected: expected_version.into(),
        found: combined.lines().next().unwrap_or("<no output>").to_string(),
    })
}

/// Compute SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String, CompilerError> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = std::io::Read::read(&mut file, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hash_bytes = hasher.finalize();
    Ok(hash_bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn verify_expected_hash(
    executable: &Path,
    tool_name: &str,
    actual: &str,
    expected: Option<&str>,
) -> Result<(), CompilerError> {
    let Some(expected) = expected else {
        return Ok(());
    };
    if actual.eq_ignore_ascii_case(expected) {
        return Ok(());
    }
    Err(CompilerError::HashMismatch {
        executable: format!("{} ({})", tool_name, executable.display()),
        expected: expected.to_ascii_lowercase(),
        found: actual.to_string(),
    })
}

// Simple SHA-256 implementation to avoid adding a dependency.
pub mod sha2 {
    pub struct Sha256 {
        state: [u32; 8],
        buf: [u8; 64],
        buf_len: usize,
        total_len: u64,
    }

    impl Sha256 {
        pub fn new() -> Self {
            Sha256 {
                state: [
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                    0x1f83d9ab, 0x5be0cd19,
                ],
                buf: [0u8; 64],
                buf_len: 0,
                total_len: 0,
            }
        }

        pub fn update(&mut self, data: &[u8]) {
            self.total_len += data.len() as u64;
            let mut offset = 0usize;
            while offset < data.len() {
                let space = 64 - self.buf_len;
                let copy = (data.len() - offset).min(space);
                self.buf[self.buf_len..self.buf_len + copy]
                    .copy_from_slice(&data[offset..offset + copy]);
                self.buf_len += copy;
                offset += copy;
                if self.buf_len == 64 {
                    self.process_block();
                    self.buf_len = 0;
                }
            }
        }

        pub fn finalize(mut self) -> [u8; 32] {
            let total_bits = self.total_len * 8;
            // Padding
            self.buf[self.buf_len] = 0x80;
            self.buf_len += 1;
            if self.buf_len > 56 {
                // Fill remaining and process
                for i in self.buf_len..64 {
                    self.buf[i] = 0;
                }
                self.process_block();
                self.buf_len = 0;
            }
            // Pad to 56
            for i in self.buf_len..56 {
                self.buf[i] = 0;
            }
            self.buf[56..64].copy_from_slice(&total_bits.to_be_bytes());
            self.process_block();

            let mut result = [0u8; 32];
            for (i, &word) in self.state.iter().enumerate() {
                result[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
            }
            result
        }

        fn process_block(&mut self) {
            const K: [u32; 64] = [
                0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
                0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
                0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
                0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
                0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
                0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
                0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
                0xc67178f2,
            ];

            let mut w = [0u32; 64];
            for i in 0..16 {
                let base = i * 4;
                w[i] = u32::from_be_bytes([
                    self.buf[base],
                    self.buf[base + 1],
                    self.buf[base + 2],
                    self.buf[base + 3],
                ]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ (!e & g);
                let temp1 = h
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            self.state[0] = self.state[0].wrapping_add(a);
            self.state[1] = self.state[1].wrapping_add(b);
            self.state[2] = self.state[2].wrapping_add(c);
            self.state[3] = self.state[3].wrapping_add(d);
            self.state[4] = self.state[4].wrapping_add(e);
            self.state[5] = self.state[5].wrapping_add(f);
            self.state[6] = self.state[6].wrapping_add(g);
            self.state[7] = self.state[7].wrapping_add(h);
        }
    }
}

fn process_output_text(output: &Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    match (stdout.is_empty(), stderr.is_empty()) {
        (false, false) => format!("{stdout}\n{stderr}"),
        (false, true) => stdout,
        (true, false) => stderr,
        (true, true) => String::new(),
    }
}

/// Run a direct subprocess with a cleared environment, captured output, and timeout.
fn run_direct_process(
    executable: &Path,
    args: &[String],
    working_dir: Option<&Path>,
    env: &HashMap<String, String>,
    timeout_seconds: u64,
    stream_limit_bytes: u64,
    stage_name: &str,
) -> Result<Output, CompilerError> {
    let mut cmd = Command::new(executable);
    cmd.args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(working_dir) = working_dir {
        cmd.current_dir(working_dir);
    }

    cmd.env_clear();
    for (key, value) in env {
        cmd.env(key, value);
    }

    let mut child = cmd.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            CompilerError::NotFound {
                executable: executable.display().to_string(),
                searched_paths: vec![],
            }
        } else {
            CompilerError::Io {
                message: format!("failed to spawn {stage_name}"),
                source: e,
            }
        }
    })?;

    let mut stdout = child.stdout.take().ok_or_else(|| {
        CompilerError::InvalidProfile(format!("failed to capture stdout for {stage_name}"))
    })?;
    let mut stderr = child.stderr.take().ok_or_else(|| {
        CompilerError::InvalidProfile(format!("failed to capture stderr for {stage_name}"))
    })?;

    let stream_limit = usize::try_from(stream_limit_bytes).unwrap_or(usize::MAX);
    let stdout_reader = thread::spawn(move || read_stream_bounded(&mut stdout, stream_limit));
    let stderr_reader = thread::spawn(move || read_stream_bounded(&mut stderr, stream_limit));

    let started = Instant::now();
    let timeout = Duration::from_secs(timeout_seconds.max(1));
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| CompilerError::Io {
            message: format!("failed to poll {stage_name}"),
            source: e,
        })? {
            break status;
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(CompilerError::Timeout {
                stage: stage_name.to_string(),
                timeout_seconds,
            });
        }
        thread::sleep(Duration::from_millis(10));
    };

    let stdout = join_output_reader(stdout_reader, stage_name, "stdout")?;
    let stderr = join_output_reader(stderr_reader, stage_name, "stderr")?;
    if stdout.exceeded {
        return Err(CompilerError::StreamBoundExceeded {
            stage: stage_name.to_string(),
            stream: "stdout".to_string(),
            limit: stream_limit_bytes,
        });
    }
    if stderr.exceeded {
        return Err(CompilerError::StreamBoundExceeded {
            stage: stage_name.to_string(),
            stream: "stderr".to_string(),
            limit: stream_limit_bytes,
        });
    }

    Ok(Output {
        status,
        stdout: stdout.bytes,
        stderr: stderr.bytes,
    })
}

struct CapturedStream {
    bytes: Vec<u8>,
    exceeded: bool,
}

fn read_stream_bounded(stream: &mut impl Read, limit: usize) -> std::io::Result<CapturedStream> {
    let mut bytes = Vec::new();
    let mut exceeded = false;
    let mut buffer = [0u8; 8192];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        if !exceeded {
            if bytes
                .len()
                .checked_add(count)
                .map_or(true, |length| length > limit)
            {
                // Continue draining so the child cannot block on a full pipe,
                // but do not retain unbounded diagnostic output in memory.
                bytes.clear();
                exceeded = true;
            } else {
                bytes.extend_from_slice(&buffer[..count]);
            }
        }
    }
    Ok(CapturedStream { bytes, exceeded })
}

fn join_output_reader(
    handle: thread::JoinHandle<std::io::Result<CapturedStream>>,
    stage_name: &str,
    stream: &str,
) -> Result<CapturedStream, CompilerError> {
    handle
        .join()
        .map_err(|_| {
            CompilerError::InvalidProfile(format!("{stage_name} {stream} reader panicked"))
        })?
        .map_err(|e| CompilerError::Io {
            message: format!("failed to read {stream} from {stage_name}"),
            source: e,
        })
}

/// Run a compiler stage (qbsp, vis, or light) and return its output.
fn run_compiler_stage(
    executable: &Path,
    args: &[String],
    working_dir: &Path,
    env: &HashMap<String, String>,
    timeout_seconds: u64,
    stream_limit_bytes: u64,
    stage_name: &str,
) -> Result<Output, CompilerError> {
    run_direct_process(
        executable,
        args,
        Some(working_dir),
        env,
        timeout_seconds,
        stream_limit_bytes,
        stage_name,
    )
}

fn profile_uses_bsp2(profile: &CompilerProfile) -> bool {
    profile.default_qbsp_args.iter().any(|a| a == "-bsp2")
        || profile.default_light_args.iter().any(|a| a == "-bsp2")
}

fn contains_compiler_warning(stdout: &str, stderr: &str) -> bool {
    [stdout, stderr].into_iter().any(|stream| {
        let stream = stream.to_ascii_lowercase();
        stream.contains("warning:")
            || stream.contains("no entities in empty space")
            || stream.contains("no filling performed")
    })
}

/// Check whether compiler output indicates missing textures — a hard failure.
fn contains_missing_texture(stdout: &str, stderr: &str) -> bool {
    [stdout, stderr].into_iter().any(|stream| {
        let lower = stream.to_ascii_lowercase();
        lower.contains("unable to find texture")
            || lower.contains("could not load texture")
            || lower.contains("missing texture")
    })
}

/// Detect and reject missing-texture diagnostics.
fn reject_missing_textures(stage: &str, output: &Output) -> Result<(), CompilerError> {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if contains_missing_texture(&stdout, &stderr) {
        return Err(CompilerError::MissingTexture {
            stage: stage.to_string(),
            stdout,
            stderr,
        });
    }
    Ok(())
}

/// Check for leaked pointfiles after compilation.
/// Only `.pts` files indicate leaks; `.prt` is the normal portal file
/// produced by qbsp and consumed by vis.
fn check_pointfile_leaks(
    work_dir: &Path,
    bsp_stem: &str,
    stage: &str,
) -> Result<(), CompilerError> {
    let pts_path = work_dir.join(format!("{bsp_stem}.pts"));
    if pts_path.exists() {
        return Err(CompilerError::PointfileLeaked {
            path: pts_path,
            stage: stage.to_string(),
        });
    }
    Ok(())
}

/// Build a deterministic identity for the exact minimized compiler environment.
///
/// The identity hashes values rather than serializing host-specific paths,
/// usernames, or temporary directories into a reproducible package manifest.
pub fn controlled_environment_identity() -> String {
    let env = minimal_env();
    let mut entries: Vec<_> = env.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));

    let mut hasher = sha2::Sha256::new();
    for (key, value) in entries {
        hasher.update(key.as_bytes());
        hasher.update(&[0]);
        hasher.update(value.as_bytes());
        hasher.update(&[0]);
    }
    let digest = hasher.finalize();
    let digest = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("sha256:{digest}")
}

/// Validate source input is a regular non-symlink file.
pub fn validate_input_regular(path: &Path) -> Result<(), CompilerError> {
    let meta = path.symlink_metadata().map_err(|e| CompilerError::Io {
        message: format!("cannot stat input '{}'", path.display()),
        source: e,
    })?;
    if meta.file_type().is_symlink() {
        return Err(CompilerError::NonRegularInput {
            path: path.to_path_buf(),
            kind: "symlink".to_string(),
        });
    }
    if !meta.is_file() {
        return Err(CompilerError::NonRegularInput {
            path: path.to_path_buf(),
            kind: if meta.is_dir() {
                "directory".to_string()
            } else {
                "special file".to_string()
            },
        });
    }
    Ok(())
}

/// Validate a .lit file: QLIT magic, version 1, payload size multiple of 3.
pub fn validate_lit_data(lit_data: &[u8], bsp_lightdata_size: usize) -> Result<(), CompilerError> {
    if lit_data.len() < 8 {
        return Err(CompilerError::InvalidLit {
            path: PathBuf::from("<lit>"),
            reason: format!("too short: {} bytes (minimum 8)", lit_data.len()),
        });
    }
    if &lit_data[0..4] != b"QLIT" {
        return Err(CompilerError::InvalidLit {
            path: PathBuf::from("<lit>"),
            reason: "invalid magic (expected QLIT)".to_string(),
        });
    }
    let version = u32::from_le_bytes([lit_data[4], lit_data[5], lit_data[6], lit_data[7]]);
    if version != 1 {
        return Err(CompilerError::InvalidLit {
            path: PathBuf::from("<lit>"),
            reason: format!("unsupported version {version} (expected 1)"),
        });
    }
    let payload_size = lit_data.len() - 8;
    if payload_size % 3 != 0 {
        return Err(CompilerError::InvalidLit {
            path: PathBuf::from("<lit>"),
            reason: format!("payload size {payload_size} not a multiple of 3"),
        });
    }
    let expected_payload = bsp_lightdata_size.saturating_mul(3);
    if payload_size != expected_payload {
        return Err(CompilerError::InvalidLit {
            path: PathBuf::from("<lit>"),
            reason: format!(
                "payload size {payload_size} does not match BSP lightdata {bsp_lightdata_size} × 3 = {expected_payload}"
            ),
        });
    }
    Ok(())
}

fn reject_compiler_warnings(stage: &str, output: &Output) -> Result<(), CompilerError> {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if contains_compiler_warning(&stdout, &stderr) {
        return Err(CompilerError::CompilerWarning {
            stage: stage.to_string(),
            stdout,
            stderr,
        });
    }
    Ok(())
}

/// Compile a .map source file using a compiler profile.
///
/// Steps:
/// 1. Locate and verify compiler executables (version check).
/// 2. Run qbsp → .bsp (BSP29 or BSP2 based on args).
/// 3. Run vis → visibility data in .bsp.
/// 4. Run light → lightmaps in .bsp (optionally .lit).
/// 5. Validate output size limits.
/// 6. Re-parse output .bsp through `bsp` crate for validation.
/// 7. Return `CompileResult` with provenance.
pub fn compile_map(
    source_map: &Path,
    profile: &CompilerProfile,
    work_dir: &Path,
    palette_path: &Path,
    tool_path: Option<&Path>,
    wad_paths: &[PathBuf],
) -> Result<CompileResult, CompilerError> {
    // Validate inputs — reject non-regular and symlinked files
    validate_input_regular(source_map)?;
    validate_input_regular(palette_path)?;

    // Validate and sanitize WAD paths
    let mut wad_staging_names: Vec<(PathBuf, String)> = Vec::new();
    let mut seen_basenames: std::collections::HashSet<String> = std::collections::HashSet::new();
    for wad_path in wad_paths {
        validate_input_regular(wad_path)?;
        // Sanitize basename
        let basename = wad_path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
            .unwrap_or_default();
        if basename.is_empty()
            || basename.contains("..")
            || basename.contains('/')
            || basename.contains('\\')
        {
            return Err(CompilerError::SourceError {
                path: wad_path.clone(),
                message: format!("WAD basename '{}' is unsafe", basename),
            });
        }
        if !seen_basenames.insert(basename.clone()) {
            return Err(CompilerError::SourceError {
                path: wad_path.clone(),
                message: format!("duplicate WAD basename '{}'", basename),
            });
        }
        wad_staging_names.push((wad_path.clone(), basename));
    }

    let env = minimal_env();

    // Locate executables and verify pinned hashes before executing them.
    let qbsp_exe = resolve_executable(&profile.qbsp_executable, tool_path)?;
    let vis_exe = resolve_executable(&profile.vis_executable, tool_path)?;
    let light_exe = resolve_executable(&profile.light_executable, tool_path)?;

    let actual_hashes = CompilerHashes {
        qbsp_sha256: sha256_file(&qbsp_exe)?,
        vis_sha256: sha256_file(&vis_exe)?,
        light_sha256: sha256_file(&light_exe)?,
    };
    verify_expected_hash(
        &qbsp_exe,
        "qbsp",
        &actual_hashes.qbsp_sha256,
        profile
            .expected_hashes
            .as_ref()
            .map(|hashes| hashes.qbsp_sha256.as_str()),
    )?;
    verify_expected_hash(
        &vis_exe,
        "vis",
        &actual_hashes.vis_sha256,
        profile
            .expected_hashes
            .as_ref()
            .map(|hashes| hashes.vis_sha256.as_str()),
    )?;
    verify_expected_hash(
        &light_exe,
        "light",
        &actual_hashes.light_sha256,
        profile
            .expected_hashes
            .as_ref()
            .map(|hashes| hashes.light_sha256.as_str()),
    )?;

    verify_compiler_version(
        &qbsp_exe,
        &profile.required_version,
        &env,
        profile.timeout_seconds,
    )?;
    verify_compiler_version(
        &vis_exe,
        &profile.required_version,
        &env,
        profile.timeout_seconds,
    )?;
    verify_compiler_version(
        &light_exe,
        &profile.required_version,
        &env,
        profile.timeout_seconds,
    )?;

    // Copy source and palette to work dir
    let work_source =
        work_dir.join(
            source_map
                .file_name()
                .ok_or_else(|| CompilerError::SourceError {
                    path: source_map.to_path_buf(),
                    message: "source map has no filename".into(),
                })?,
        );
    std::fs::copy(source_map, &work_source)?;

    let work_palette = work_dir.join("palette.lmp");
    std::fs::copy(palette_path, &work_palette)?;

    // Stage WAD files to work dir by basename
    for (wad_path, basename) in &wad_staging_names {
        let dest = work_dir.join(basename);
        std::fs::copy(wad_path, &dest)?;
    }

    let bsp_filename = work_source.with_extension("bsp");
    let bsp_filename_str = bsp_filename
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| CompilerError::SourceError {
            path: bsp_filename.clone(),
            message: "invalid bsp filename".into(),
        })?;

    // Stage 1: qbsp
    let qbsp_args: Vec<String> = {
        let mut args = profile.default_qbsp_args.clone();
        args.push(
            work_source
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("source.map")
                .to_string(),
        );
        args
    };
    let qbsp_output = run_compiler_stage(
        &qbsp_exe,
        &qbsp_args,
        work_dir,
        &env,
        profile.timeout_seconds,
        profile.max_output_size,
        "qbsp",
    )?;
    if !qbsp_output.status.success() {
        return Err(CompilerError::CompilationFailed {
            stage: "qbsp".into(),
            exit_code: qbsp_output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&qbsp_output.stdout).into(),
            stderr: String::from_utf8_lossy(&qbsp_output.stderr).into(),
        });
    }
    reject_compiler_warnings("qbsp", &qbsp_output)?;
    reject_missing_textures("qbsp", &qbsp_output)?;
    if !bsp_filename.exists() {
        return Err(CompilerError::MissingOutput {
            expected_path: bsp_filename.clone(),
            stage: "qbsp".into(),
        });
    }
    // Extract stem for pointfile checks
    let bsp_stem = bsp_filename
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    // Stage 2: vis
    let vis_args: Vec<String> = {
        let mut args = profile.default_vis_args.clone();
        args.push(bsp_filename_str.to_string());
        args
    };
    let vis_output = run_compiler_stage(
        &vis_exe,
        &vis_args,
        work_dir,
        &env,
        profile.timeout_seconds,
        profile.max_output_size,
        "vis",
    )?;
    if !vis_output.status.success() {
        return Err(CompilerError::CompilationFailed {
            stage: "vis".into(),
            exit_code: vis_output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&vis_output.stdout).into(),
            stderr: String::from_utf8_lossy(&vis_output.stderr).into(),
        });
    }
    reject_compiler_warnings("vis", &vis_output)?;
    reject_missing_textures("vis", &vis_output)?;

    // Stage 3: light
    let light_args: Vec<String> = {
        let mut args = profile.default_light_args.clone();
        args.push(bsp_filename_str.to_string());
        args
    };
    let light_output = run_compiler_stage(
        &light_exe,
        &light_args,
        work_dir,
        &env,
        profile.timeout_seconds,
        profile.max_output_size,
        "light",
    )?;
    if !light_output.status.success() {
        return Err(CompilerError::CompilationFailed {
            stage: "light".into(),
            exit_code: light_output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&light_output.stdout).into(),
            stderr: String::from_utf8_lossy(&light_output.stderr).into(),
        });
    }
    reject_compiler_warnings("light", &light_output)?;
    reject_missing_textures("light", &light_output)?;
    check_pointfile_leaks(work_dir, bsp_stem, "light")?;

    // Check output size
    let bsp_size = bsp_filename.metadata()?.len();
    if bsp_size > profile.max_output_size {
        return Err(CompilerError::OutputTooLarge {
            path: bsp_filename.clone(),
            size: bsp_size,
            limit: profile.max_output_size,
        });
    }

    // Read compiled BSP
    let bsp_data = std::fs::read(&bsp_filename)?;

    // Check for .lit companion
    let lit_path = bsp_filename.with_extension("lit");
    let uses_bsp2 = profile_uses_bsp2(profile);
    let lit_data = if lit_path.exists() {
        let lit_size = lit_path.metadata()?.len();
        if lit_size > profile.max_output_size {
            return Err(CompilerError::OutputTooLarge {
                path: lit_path.clone(),
                size: lit_size,
                limit: profile.max_output_size,
            });
        }
        let lit_bytes = std::fs::read(&lit_path)?;
        // Validate .lit structure before accepting it
        // We'll validate properly after BSP parse, but do structural check now
        if lit_bytes.len() >= 8 && &lit_bytes[0..4] == b"QLIT" {
            let version =
                u32::from_le_bytes([lit_bytes[4], lit_bytes[5], lit_bytes[6], lit_bytes[7]]);
            if version != 1 {
                return Err(CompilerError::InvalidLit {
                    path: lit_path.clone(),
                    reason: format!("unsupported QLIT version {version} (expected 1)"),
                });
            }
        }
        Some(lit_bytes)
    } else {
        None
    };

    // Check BSP length before inspecting magic
    if bsp_data.len() < 4 {
        return Err(CompilerError::SourceError {
            path: bsp_filename.clone(),
            message: format!("BSP too short to inspect magic: {} bytes", bsp_data.len()),
        });
    }

    // Validate BSP magic matches profile expectation
    let bsp_magic = &bsp_data[..4];
    if uses_bsp2 {
        if bsp_magic != b"BSP2" {
            return Err(CompilerError::BspMagicMismatch {
                expected: "BSP2".to_string(),
                found: format!("{bsp_magic:02x?}"),
                path: bsp_filename.clone(),
            });
        }
    } else {
        let magic_int =
            i32::from_le_bytes([bsp_magic[0], bsp_magic[1], bsp_magic[2], bsp_magic[3]]);
        if magic_int != 29 {
            return Err(CompilerError::BspMagicMismatch {
                expected: "29 (BSP29)".to_string(),
                found: format!("{bsp_magic:02x?} (version {magic_int})"),
                path: bsp_filename.clone(),
            });
        }
    }

    let compiler_hashes = Some(actual_hashes);

    // Read palette for re-validation
    let palette_data = std::fs::read(&work_palette)?;

    // Collect staged WAD bytes for re-validation
    let mut wad_archives: Vec<(String, Vec<u8>)> = Vec::new();
    for (_, basename) in &wad_staging_names {
        let wad_path = work_dir.join(basename);
        if wad_path.exists() {
            let wad_bytes = std::fs::read(&wad_path)?;
            wad_archives.push((basename.clone(), wad_bytes));
        }
    }

    let mut source_hashes = vec![
        PackageContentHash {
            path: PathBuf::from(
                work_source
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("source.map"),
            ),
            sha256: sha256_file(&work_source)?,
        },
        PackageContentHash {
            path: PathBuf::from("palette.lmp"),
            sha256: sha256_file(&work_palette)?,
        },
    ];
    // Include WAD hashes in source provenance
    for (_, basename) in &wad_staging_names {
        let wad_path = work_dir.join(basename);
        if wad_path.exists() {
            source_hashes.push(PackageContentHash {
                path: PathBuf::from(basename),
                sha256: sha256_file(&wad_path)?,
            });
        }
    }
    source_hashes.sort_by(|left, right| left.path.cmp(&right.path));

    let mut output_hashes = vec![PackageContentHash {
        path: PathBuf::from(bsp_filename_str),
        sha256: sha256_file(&bsp_filename)?,
    }];
    if lit_data.is_some() {
        output_hashes.push(PackageContentHash {
            path: PathBuf::from(
                lit_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("output.lit"),
            ),
            sha256: sha256_file(&lit_path)?,
        });
    }
    output_hashes.sort_by(|left, right| left.path.cmp(&right.path));

    // Re-validate compiled BSP through the bsp parser
    let load_options = LoadOptions {
        strict: true,
        palette: Some(palette_data),
        lit_data: lit_data.clone(),
        wad_archives,
        texture_overrides: Vec::new(),
        source_identity: source_map.display().to_string(),
    };
    BspLoader::load(&bsp_data, &load_options).map_err(CompilerError::ValidationFailed)?;

    // Build provenance
    let provenance = CompilerProvenance {
        compiler_identity: profile.compiler_identity.clone(),
        compiler_version: profile.required_version.clone(),
        qbsp_args: qbsp_args.clone(),
        vis_args: vis_args.clone(),
        light_args: light_args.clone(),
        source_hashes,
        output_hashes,
        compiler_hashes,
    };

    let stdout = format!(
        "[qbsp]\n{}\n[vis]\n{}\n[light]\n{}",
        String::from_utf8_lossy(&qbsp_output.stdout).trim(),
        String::from_utf8_lossy(&vis_output.stdout).trim(),
        String::from_utf8_lossy(&light_output.stdout).trim(),
    );

    let stderr = format!(
        "[qbsp]\n{}\n[vis]\n{}\n[light]\n{}",
        String::from_utf8_lossy(&qbsp_output.stderr).trim(),
        String::from_utf8_lossy(&vis_output.stderr).trim(),
        String::from_utf8_lossy(&light_output.stderr).trim(),
    );

    Ok(CompileResult {
        bsp_data,
        lit_data,
        provenance,
        stdout,
        stderr,
    })
}

/// Validate a compiled .bsp file against the bsp parser.
///
/// Returns `Ok(())` if the BSP is structurally valid and recognized.
pub fn validate_bsp(bsp_data: &[u8], palette_data: Option<&[u8]>) -> Result<(), BspReport> {
    let options = LoadOptions {
        strict: true,
        palette: palette_data.map(|d| d.to_vec()),
        lit_data: None,
        wad_archives: Vec::new(),
        texture_overrides: Vec::new(),
        source_identity: String::new(),
    };
    BspLoader::load(bsp_data, &options)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_empty() {
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"");
        let result = hasher.finalize();
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let hex: String = result.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, expected);
    }

    #[test]
    fn sha256_abc() {
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"abc");
        let result = hasher.finalize();
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let hex: String = result.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, expected);
    }

    #[test]
    fn compiler_warning_detection_is_case_insensitive_and_checks_both_streams() {
        assert!(contains_compiler_warning(
            "WARNING: unable to find texture stone_missing",
            ""
        ));
        assert!(contains_compiler_warning(
            "",
            "Warning: No entities in empty space -- no filling performed"
        ));
        assert!(contains_compiler_warning(
            "No filling performed for hull 0",
            ""
        ));
        assert!(!contains_compiler_warning(
            "Lighting Completed; 0 warnings.",
            ""
        ));
    }

    #[test]
    fn compiler_warning_is_a_typed_publication_failure() {
        let output = Output {
            status: Default::default(),
            stdout: b"WARNING: No entities in empty space -- no filling performed".to_vec(),
            stderr: Vec::new(),
        };
        let error = reject_compiler_warnings("qbsp", &output).unwrap_err();
        assert!(matches!(
            error,
            CompilerError::CompilerWarning { ref stage, .. } if stage == "qbsp"
        ));
    }

    #[test]
    fn zero_warning_summary_is_not_a_publication_failure() {
        let output = Output {
            status: Default::default(),
            stdout: b"Completed with 0 warnings.".to_vec(),
            stderr: Vec::new(),
        };
        reject_compiler_warnings("light", &output).unwrap();
    }

    #[test]
    fn validate_bsp_rejects_empty() {
        let result = validate_bsp(b"", None);
        assert!(result.is_err());
    }

    #[test]
    fn minimal_env_contains_path() {
        let env = minimal_env();
        assert!(env.contains_key("PATH"));
    }

    #[test]
    fn profile_rejects_non_positive_limits() {
        let content = r#"
name = "bad"
compiler_identity = "ericw-tools"
required_version = "2.0.0-alpha3"
qbsp_executable = "qbsp"
vis_executable = "vis"
light_executable = "light"
timeout_seconds = 0
"#;
        let err = parse_compiler_profile(content).unwrap_err();
        assert!(err.contains("timeout_seconds"));
    }

    #[test]
    fn profile_rejects_malformed_expected_hashes() {
        let content = r#"
name = "bad-hash"
compiler_identity = "ericw-tools"
required_version = "2.0.0-alpha3"
qbsp_executable = "qbsp"
vis_executable = "vis"
light_executable = "light"

[expected_hashes]
qbsp_sha256 = "not-a-sha"
vis_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
light_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
"#;
        let err = parse_compiler_profile(content).unwrap_err();
        assert!(err.contains("qbsp_sha256"));
    }

    #[test]
    fn compiler_error_display() {
        let err = CompilerError::NotFound {
            executable: "qbsp".into(),
            searched_paths: vec![],
        };
        let msg = err.to_string();
        assert!(msg.contains("qbsp"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn output_too_large_error() {
        let err = CompilerError::OutputTooLarge {
            path: PathBuf::from("test.bsp"),
            size: 999,
            limit: 100,
        };
        let msg = err.to_string();
        assert!(msg.contains("test.bsp"));
        assert!(msg.contains("999"));
        assert!(msg.contains("100"));
    }
}
