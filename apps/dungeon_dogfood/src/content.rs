use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use audio::AudioClipId;
use serde::Deserialize;
use thiserror::Error;

const CONTENT_PACK_VERSION: u32 = 1;
const REQUIRED_MATERIAL_IDS: [&str; 2] = ["stone_wall", "stone_floor"];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentPack {
    pub version: u32,
    pub deterministic: DeterministicSpec,
    pub props: Vec<PropSpec>,
    pub materials: Vec<MaterialSpec>,
    pub environments: Vec<EnvironmentSpec>,
    #[serde(default)]
    pub audio_clips: Vec<AudioClipSpec>,
    pub light_presets: Vec<LightPresetSpec>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeterministicSpec {
    pub prop_selector: PropSelector,
    pub light_selector: LightSelector,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize)]
pub enum PropSelector {
    #[serde(rename = "marker_modulo")]
    MarkerModulo,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize)]
pub enum LightSelector {
    #[serde(rename = "marker_modulo_7_2_1")]
    MarkerModulo721,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropSpec {
    pub id: String,
    pub enabled: bool,
    pub path: PathBuf,
    pub prefer_unlit_fallback: bool,
    pub scale: [f32; 3],
    pub yaw_degrees: f32,
    pub y_offset: f32,
    pub placement_half_extents: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialSpec {
    pub id: String,
    pub family: MaterialFamily,
    pub base_path: PathBuf,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize)]
pub enum MaterialFamily {
    #[serde(rename = "pbr")]
    Pbr,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentSpec {
    pub id: String,
    pub path: PathBuf,
    pub mode: EnvironmentMode,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize)]
pub enum EnvironmentMode {
    #[serde(rename = "auto")]
    Auto,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioClipSpec {
    pub id: String,
    pub path: PathBuf,
    pub format: AudioClipFormat,
    pub usage: AudioClipUsage,
    #[serde(default)]
    pub default_gain: Option<f32>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize)]
pub enum AudioClipFormat {
    #[serde(rename = "wav")]
    Wav,
    #[serde(rename = "ogg")]
    Ogg,
    #[serde(rename = "flac")]
    Flac,
    #[serde(rename = "mp3")]
    Mp3,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize)]
pub enum AudioClipUsage {
    #[serde(rename = "effect")]
    Effect,
    #[serde(rename = "music")]
    Music,
    #[serde(rename = "ambient")]
    Ambient,
    #[serde(rename = "voice")]
    Voice,
    #[serde(rename = "ui")]
    Ui,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightPresetSpec {
    pub id: String,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LightPresetId {
    Warm,
    Cool,
    Accent,
}

#[derive(Debug, Copy, Clone)]
pub struct PropPlacementPolicy {
    pub scale: glam::Vec3,
    pub yaw_degrees: f32,
    pub y_offset: f32,
    pub prefer_unlit_fallback: bool,
}

impl PropSpec {
    pub fn placement_policy(&self) -> PropPlacementPolicy {
        PropPlacementPolicy {
            scale: glam::Vec3::new(self.scale[0], self.scale[1], self.scale[2]),
            yaw_degrees: self.yaw_degrees,
            y_offset: self.y_offset,
            prefer_unlit_fallback: self.prefer_unlit_fallback,
        }
    }

    /// Build a validated placement envelope from spec data.
    /// Returns `None` when half-extents are non-positive, non-finite, or overflow.
    pub fn placement_envelope(&self) -> Option<PropPlacementEnvelope> {
        let local = self.placement_half_extents;
        if local[0] <= 0.0 || local[1] <= 0.0 || local[2] <= 0.0 {
            return None;
        }
        if !local.iter().all(|v| v.is_finite()) {
            return None;
        }
        if !self.scale.iter().all(|v| v.is_finite() && *v > 0.0) {
            return None;
        }
        if !self.yaw_degrees.is_finite() {
            return None;
        }
        let envelope = PropPlacementEnvelope {
            half_extents_local: local,
            scale: self.scale,
            yaw_degrees: self.yaw_degrees,
        };
        envelope
            .world_half_extents()
            .iter()
            .all(|value| value.is_finite())
            .then_some(envelope)
    }
}

/// App-local normalized prop placement envelope.
/// Combines declared half-extents with the existing scale and yaw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropPlacementEnvelope {
    pub half_extents_local: [f32; 3],
    pub scale: [f32; 3],
    pub yaw_degrees: f32,
}

impl PropPlacementEnvelope {
    /// Conservative horizontal half-extents in world space.
    /// For yaw θ, using the formula:
    /// hx = |cos θ| * |sx| * lx + |sin θ| * |sz| * lz
    /// hz = |sin θ| * |sx| * lx + |cos θ| * |sz| * lz
    /// hy = |sy| * ly
    pub fn world_half_extents(&self) -> [f32; 3] {
        let theta = self.yaw_degrees.to_radians();
        let cos = theta.cos().abs();
        let sin = theta.sin().abs();
        let sx = self.scale[0].abs();
        let sy = self.scale[1].abs();
        let sz = self.scale[2].abs();
        let lx = self.half_extents_local[0];
        let ly = self.half_extents_local[1];
        let lz = self.half_extents_local[2];
        [
            cos * sx * lx + sin * sz * lz,
            sy * ly,
            sin * sx * lx + cos * sz * lz,
        ]
    }
}

impl ContentPack {
    pub fn enabled_props(&self) -> Vec<&PropSpec> {
        self.props.iter().filter(|prop| prop.enabled).collect()
    }

    pub fn material_by_id(&self, id: &str) -> Option<&MaterialSpec> {
        self.materials.iter().find(|material| material.id == id)
    }

    pub fn primary_environment(&self) -> &EnvironmentSpec {
        // Validation guarantees at least one environment exists.
        &self.environments[0]
    }

    pub fn startup_audio_clip(&self) -> Option<&AudioClipSpec> {
        self.audio_clips.first()
    }

    pub fn light_preset(&self, id: LightPresetId) -> Option<&LightPresetSpec> {
        let needle = match id {
            LightPresetId::Warm => "warm",
            LightPresetId::Cool => "cool",
            LightPresetId::Accent => "accent",
        };

        self.light_presets.iter().find(|preset| preset.id == needle)
    }
}

#[derive(Debug, Error)]
pub enum ContentError {
    #[error("failed to read content pack '{path}': {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse content pack '{path}': {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("invalid content pack '{path}' at '{key}': {message}")]
    Validation {
        path: PathBuf,
        key: String,
        message: String,
    },
}

pub fn load_content_pack(path: impl AsRef<Path>) -> Result<ContentPack, ContentError> {
    let requested_path = path.as_ref().to_path_buf();
    let resolved_path = resolve_content_path(&requested_path);

    let raw = fs::read_to_string(&resolved_path).map_err(|source| ContentError::Read {
        path: requested_path.clone(),
        source,
    })?;

    let pack: ContentPack = toml::from_str(&raw).map_err(|source| ContentError::Parse {
        path: requested_path.clone(),
        source,
    })?;

    validate_content_pack(&pack, &requested_path)?;
    validate_required_runtime_content(&pack, &requested_path)?;
    Ok(pack)
}

fn validate_required_runtime_content(
    pack: &ContentPack,
    pack_path: &Path,
) -> Result<(), ContentError> {
    for required_id in REQUIRED_MATERIAL_IDS {
        if pack.material_by_id(required_id).is_none() {
            return Err(validation_err(
                pack_path,
                "materials",
                format!("required material id '{required_id}' is missing"),
            ));
        }
    }

    Ok(())
}

pub fn prop_for_marker_index(marker_idx: usize, props_len: usize) -> usize {
    assert!(
        props_len > 0,
        "props_len must be > 0 for deterministic selection"
    );
    marker_idx % props_len
}

pub fn light_preset_for_marker_index(marker_idx: usize) -> LightPresetId {
    match marker_idx % 10 {
        0..=6 => LightPresetId::Warm,
        7 | 8 => LightPresetId::Cool,
        _ => LightPresetId::Accent,
    }
}

fn validate_content_pack(pack: &ContentPack, pack_path: &Path) -> Result<(), ContentError> {
    if pack.version != CONTENT_PACK_VERSION {
        return Err(validation_err(
            pack_path,
            "version",
            format!(
                "unsupported content pack version {}, expected {}",
                pack.version, CONTENT_PACK_VERSION
            ),
        ));
    }

    match pack.deterministic.prop_selector {
        PropSelector::MarkerModulo => {}
    }
    match pack.deterministic.light_selector {
        LightSelector::MarkerModulo721 => {}
    }

    if pack.props.is_empty() {
        return Err(validation_err(
            pack_path,
            "props",
            "at least one prop is required",
        ));
    }

    let mut prop_ids = HashSet::new();
    let mut enabled_prop_count = 0usize;
    for (idx, prop) in pack.props.iter().enumerate() {
        let key = format!("props[{idx}]");
        validate_id(&prop.id, &format!("{key}.id"), pack_path)?;

        if !prop_ids.insert(prop.id.as_str()) {
            return Err(validation_err(
                pack_path,
                format!("{key}.id"),
                format!("duplicate id '{}'", prop.id),
            ));
        }

        if prop.enabled {
            enabled_prop_count += 1;
            require_existing_path(&prop.path, &format!("{key}.path"), &prop.id, pack_path)?;
        }

        for (axis, value) in ["x", "y", "z"].into_iter().zip(prop.scale) {
            if !value.is_finite() || value <= 0.0 {
                return Err(validation_err(
                    pack_path,
                    format!("{key}.scale.{axis}"),
                    "scale values must be finite and > 0",
                ));
            }
        }

        for (axis, value) in ["x", "y", "z"].into_iter().zip(prop.placement_half_extents) {
            if !value.is_finite() || value <= 0.0 {
                return Err(validation_err(
                    pack_path,
                    format!("{key}.placement_half_extents.{axis}"),
                    "placement_half_extents must be finite and > 0",
                ));
            }
        }

        if !prop.yaw_degrees.is_finite() {
            return Err(validation_err(
                pack_path,
                format!("{key}.yaw_degrees"),
                "yaw_degrees must be finite",
            ));
        }

        if prop.placement_envelope().is_none() {
            return Err(validation_err(
                pack_path,
                format!("{key}.placement_half_extents"),
                "transformed placement envelope must remain finite",
            ));
        }

        if !prop.y_offset.is_finite() {
            return Err(validation_err(
                pack_path,
                format!("{key}.y_offset"),
                "y_offset must be finite",
            ));
        }
    }

    if enabled_prop_count == 0 {
        return Err(validation_err(
            pack_path,
            "props",
            "at least one prop must have enabled=true",
        ));
    }

    if pack.materials.is_empty() {
        return Err(validation_err(
            pack_path,
            "materials",
            "at least one material is required",
        ));
    }

    let mut material_ids = HashSet::new();
    for (idx, material) in pack.materials.iter().enumerate() {
        let key = format!("materials[{idx}]");
        validate_id(&material.id, &format!("{key}.id"), pack_path)?;

        if !material_ids.insert(material.id.as_str()) {
            return Err(validation_err(
                pack_path,
                format!("{key}.id"),
                format!("duplicate id '{}'", material.id),
            ));
        }

        require_existing_path(
            &material.base_path,
            &format!("{key}.base_path"),
            &material.id,
            pack_path,
        )?;
    }

    if pack.environments.is_empty() {
        return Err(validation_err(
            pack_path,
            "environments",
            "at least one environment is required",
        ));
    }

    let mut environment_ids = HashSet::new();
    for (idx, env) in pack.environments.iter().enumerate() {
        let key = format!("environments[{idx}]");
        validate_id(&env.id, &format!("{key}.id"), pack_path)?;

        if !environment_ids.insert(env.id.as_str()) {
            return Err(validation_err(
                pack_path,
                format!("{key}.id"),
                format!("duplicate id '{}'", env.id),
            ));
        }

        require_existing_path(&env.path, &format!("{key}.path"), &env.id, pack_path)?;
    }

    let mut audio_ids = HashSet::new();
    for (idx, clip) in pack.audio_clips.iter().enumerate() {
        let key = format!("audio_clips[{idx}]");
        validate_audio_clip_id(&clip.id, &format!("{key}.id"), pack_path)?;

        if !audio_ids.insert(clip.id.as_str()) {
            return Err(validation_err(
                pack_path,
                format!("{key}.id"),
                format!("duplicate id '{}'", clip.id),
            ));
        }

        require_existing_path(&clip.path, &format!("{key}.path"), &clip.id, pack_path)?;

        if let Some(default_gain) = clip.default_gain {
            if !default_gain.is_finite() || default_gain <= 0.0 {
                return Err(validation_err(
                    pack_path,
                    format!("{key}.default_gain"),
                    "default_gain must be finite and > 0",
                ));
            }
        }
    }

    if pack.light_presets.is_empty() {
        return Err(validation_err(
            pack_path,
            "light_presets",
            "at least one light preset is required",
        ));
    }

    let mut light_ids = HashSet::new();
    let mut has_warm = false;
    let mut has_cool = false;
    let mut has_accent = false;
    for (idx, preset) in pack.light_presets.iter().enumerate() {
        let key = format!("light_presets[{idx}]");
        validate_id(&preset.id, &format!("{key}.id"), pack_path)?;

        if !light_ids.insert(preset.id.as_str()) {
            return Err(validation_err(
                pack_path,
                format!("{key}.id"),
                format!("duplicate id '{}'", preset.id),
            ));
        }

        has_warm |= preset.id == "warm";
        has_cool |= preset.id == "cool";
        has_accent |= preset.id == "accent";

        for (component, value) in ["r", "g", "b"].into_iter().zip(preset.color) {
            if !value.is_finite() {
                return Err(validation_err(
                    pack_path,
                    format!("{key}.color.{component}"),
                    "light color components must be finite",
                ));
            }
        }

        if !preset.intensity.is_finite() {
            return Err(validation_err(
                pack_path,
                format!("{key}.intensity"),
                "intensity must be finite",
            ));
        }

        if !preset.range.is_finite() {
            return Err(validation_err(
                pack_path,
                format!("{key}.range"),
                "range must be finite",
            ));
        }
    }

    if !has_warm || !has_cool || !has_accent {
        return Err(validation_err(
            pack_path,
            "light_presets",
            "required preset ids are: warm, cool, accent",
        ));
    }

    Ok(())
}

fn validate_audio_clip_id(id: &str, key: &str, pack_path: &Path) -> Result<(), ContentError> {
    AudioClipId::new(id.to_string()).map_err(|err| {
        validation_err(
            pack_path,
            key,
            format!("invalid durable audio clip id '{id}': {err}"),
        )
    })?;

    let lowered = id.to_ascii_lowercase();
    if lowered.contains("slot") && lowered.contains("generation") {
        return Err(validation_err(
            pack_path,
            key,
            format!("audio clip id '{id}' looks like a runtime handle"),
        ));
    }

    Ok(())
}

fn validate_id(id: &str, key: &str, pack_path: &Path) -> Result<(), ContentError> {
    let valid = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');

    if !valid {
        return Err(validation_err(
            pack_path,
            key,
            format!("invalid id '{}'; expected ^[a-z0-9_]+$", id),
        ));
    }

    Ok(())
}

fn require_existing_path(
    path: &Path,
    key: &str,
    id: &str,
    pack_path: &Path,
) -> Result<(), ContentError> {
    if resolve_content_path(path).exists() {
        return Ok(());
    }

    Err(validation_err(
        pack_path,
        key,
        format!("entry '{id}' references missing path '{}'", path.display()),
    ))
}

pub fn resolve_content_path(path: &Path) -> PathBuf {
    if path.is_absolute() || path.exists() {
        return path.to_path_buf();
    }

    // Support running from crate-local cwd (tests) and workspace cwd (runtime).
    let workspace_candidate = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path);

    if workspace_candidate.exists() {
        workspace_candidate
    } else {
        path.to_path_buf()
    }
}

fn validation_err(
    pack_path: &Path,
    key: impl Into<String>,
    message: impl Into<String>,
) -> ContentError {
    ContentError::Validation {
        path: pack_path.to_path_buf(),
        key: key.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn write_temp_toml(contents: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();

        path.push(format!("dungeon_dogfood_content_pack_{nonce}.toml"));
        fs::write(&path, contents).expect("failed to write temp content pack");
        path
    }

    #[test]
    fn parse_success_canonical_content_pack() {
        let canonical = include_str!("../assets/content_pack.toml");
        let temp = write_temp_toml(canonical);
        let pack = load_content_pack(&temp).expect("canonical content pack should parse");

        assert_eq!(pack.version, 1);
        assert!(!pack.enabled_props().is_empty());
        assert_eq!(
            pack.startup_audio_clip().map(|clip| clip.id.as_str()),
            Some("dogfood.audio.startup_ping")
        );

        let _ = fs::remove_file(temp);
    }

    #[test]
    fn parse_failure_includes_file_path() {
        let temp = write_temp_toml("version = 1\nthis = [");

        let err = load_content_pack(&temp).expect_err("invalid toml should fail");
        match err {
            ContentError::Parse { path, .. } => assert_eq!(path, temp),
            other => panic!("expected parse error, got {other:?}"),
        }

        let _ = fs::remove_file(temp);
    }

    #[test]
    fn missing_required_key_is_fatal() {
        let temp = write_temp_toml(
            r#"
version = 1

[[props]]
id = "prop_wall_torch"
enabled = true
path = "apps/dungeon_dogfood/assets/models/props/torch_sconce/scene.gltf"
prefer_unlit_fallback = true
scale = [1.0, 1.0, 1.0]
yaw_degrees = 0.0
y_offset = 0.0
"#,
        );

        let err = load_content_pack(&temp).expect_err("missing key should fail");
        match err {
            ContentError::Parse { source, .. } => {
                let msg = source.to_string();
                assert!(msg.contains("missing field"));
            }
            other => panic!("expected parse error for missing key, got {other:?}"),
        }

        let _ = fs::remove_file(temp);
    }

    #[test]
    fn deterministic_prop_selector_repeatable() {
        let markers = [0usize, 1, 2, 3, 4, 5, 6, 15, 21, 22, 57];
        let first: Vec<usize> = markers
            .iter()
            .map(|idx| prop_for_marker_index(*idx, 3))
            .collect();
        let second: Vec<usize> = markers
            .iter()
            .map(|idx| prop_for_marker_index(*idx, 3))
            .collect();

        assert_eq!(first, second);
        assert_eq!(first, vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn deterministic_light_selector_has_721_distribution() {
        let mut warm = 0usize;
        let mut cool = 0usize;
        let mut accent = 0usize;

        for idx in 0..10 {
            match light_preset_for_marker_index(idx) {
                LightPresetId::Warm => warm += 1,
                LightPresetId::Cool => cool += 1,
                LightPresetId::Accent => accent += 1,
            }
        }

        assert_eq!(warm, 7);
        assert_eq!(cool, 2);
        assert_eq!(accent, 1);
    }

    #[test]
    fn deterministic_light_selector_sequence_is_stable() {
        let sequence: Vec<LightPresetId> = (0..12).map(light_preset_for_marker_index).collect();
        let expected = vec![
            LightPresetId::Warm,
            LightPresetId::Warm,
            LightPresetId::Warm,
            LightPresetId::Warm,
            LightPresetId::Warm,
            LightPresetId::Warm,
            LightPresetId::Warm,
            LightPresetId::Cool,
            LightPresetId::Cool,
            LightPresetId::Accent,
            LightPresetId::Warm,
            LightPresetId::Warm,
        ];
        assert_eq!(sequence, expected);
    }

    #[test]
    fn transformed_prop_envelope_overflow_is_rejected() {
        let canonical = include_str!("../assets/content_pack.toml");
        let broken = canonical
            .replacen(
                "scale = [1.0, 1.0, 1.0]",
                "scale = [3.0e38, 1.0, 3.0e38]",
                1,
            )
            .replacen(
                "placement_half_extents = [0.3, 0.5, 0.3]",
                "placement_half_extents = [3.0e38, 0.5, 3.0e38]",
                1,
            );
        let temp = write_temp_toml(&broken);

        let err = load_content_pack(&temp).expect_err("overflowing envelope should fail");
        match err {
            ContentError::Validation { key, message, .. } => {
                assert_eq!(key, "props[0].placement_half_extents");
                assert!(message.contains("remain finite"));
            }
            other => panic!("expected validation error, got {other:?}"),
        }

        let _ = fs::remove_file(temp);
    }

    #[test]
    fn missing_enabled_prop_path_is_fatal_with_key_context() {
        let canonical = include_str!("../assets/content_pack.toml");
        let broken = canonical.replacen(
            "apps/dungeon_dogfood/assets/models/props/torch_sconce/scene.gltf",
            "apps/dungeon_dogfood/assets/models/props/torch_sconce/DOES_NOT_EXIST.gltf",
            1,
        );
        let temp = write_temp_toml(&broken);

        let err = load_content_pack(&temp).expect_err("missing enabled prop path should fail");
        match err {
            ContentError::Validation { key, message, .. } => {
                assert_eq!(key, "props[0].path");
                assert!(message.contains("missing path"));
            }
            other => panic!("expected validation error, got {other:?}"),
        }

        let _ = fs::remove_file(temp);
    }

    #[test]
    fn missing_required_material_id_is_fatal() {
        let canonical = include_str!("../assets/content_pack.toml");
        let broken = canonical.replace(
            "[[materials]]\nid = \"stone_floor\"\nfamily = \"pbr\"\nbase_path = \"apps/dungeon_dogfood/assets/textures/pbr/stone_floor\"\n",
            "",
        );
        let temp = write_temp_toml(&broken);

        let err = load_content_pack(&temp).expect_err("missing required material should fail");
        match err {
            ContentError::Validation { key, message, .. } => {
                assert_eq!(key, "materials");
                assert!(message.contains("stone_floor"));
            }
            other => panic!("expected validation error, got {other:?}"),
        }

        let _ = fs::remove_file(temp);
    }

    #[test]
    fn audio_clip_path_is_validated() {
        let canonical = include_str!("../assets/content_pack.toml");
        let broken = canonical.replace(
            "apps/dungeon_dogfood/assets/audio/startup_ping.wav",
            "apps/dungeon_dogfood/assets/audio/DOES_NOT_EXIST.wav",
        );
        let temp = write_temp_toml(&broken);

        let err = load_content_pack(&temp).expect_err("missing audio clip path should fail");
        match err {
            ContentError::Validation { key, message, .. } => {
                assert_eq!(key, "audio_clips[0].path");
                assert!(message.contains("missing path"));
            }
            other => panic!("expected validation error, got {other:?}"),
        }

        let _ = fs::remove_file(temp);
    }

    #[test]
    fn duplicate_audio_clip_ids_are_rejected() {
        let canonical = include_str!("../assets/content_pack.toml");
        let duplicate = canonical.replace(
            "[[light_presets]]",
            "[[audio_clips]]\nid = \"dogfood.audio.startup_ping\"\npath = \"apps/dungeon_dogfood/assets/audio/startup_ping.wav\"\nformat = \"wav\"\nusage = \"ui\"\n\n[[light_presets]]",
        );
        let temp = write_temp_toml(&duplicate);

        let err = load_content_pack(&temp).expect_err("duplicate audio clip id should fail");
        match err {
            ContentError::Validation { key, message, .. } => {
                assert_eq!(key, "audio_clips[1].id");
                assert!(message.contains("duplicate id"));
            }
            other => panic!("expected validation error, got {other:?}"),
        }

        let _ = fs::remove_file(temp);
    }

    #[test]
    fn invalid_audio_gain_is_rejected() {
        let canonical = include_str!("../assets/content_pack.toml");
        let broken = canonical.replace("default_gain = 0.15", "default_gain = 0.0");
        let temp = write_temp_toml(&broken);

        let err = load_content_pack(&temp).expect_err("invalid audio gain should fail");
        match err {
            ContentError::Validation { key, message, .. } => {
                assert_eq!(key, "audio_clips[0].default_gain");
                assert!(message.contains("finite and > 0"));
            }
            other => panic!("expected validation error, got {other:?}"),
        }

        let _ = fs::remove_file(temp);
    }
}
