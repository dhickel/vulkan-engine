//! Canonical serialization and deterministic identity hashing for
//! Richness V1 request documents and resolved requests.
//!
//! # Canonical format rules
//!
//! - Fixed field order (deterministic, not alphabetically sorted).
//! - Lowercase exact tags for all enum values.
//! - Decimal integer formatting (no hex, no scientific notation).
//! - LF line endings (`\n`).
//! - Exactly one trailing newline.
//! - No unordered maps — all fields are written in a fixed order.
//! - Parse/validate/save round trip preserves exact bytes.
//!
//! # Identity hashing
//!
//! Deterministic SHA-256 hashes under the non-generation domain
//! `dungeon-gen/v3-richness/v1/request`. Uses length-framed UTF-8 tags,
//! little-endian integer widths, and frozen field order.

use sha2::{Digest, Sha256};

use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::request::{
    InheritedOr, ResolvedRichnessRequestV1, RichnessCaveMode, RichnessDocumentV1,
    RichnessGateIdentity, RichnessPreset, RichnessTheme,
};

// ── Hash domain ────────────────────────────────────────────────────────────

/// Domain separator for Richness V1 request identity hashes.
/// This is a non-generation domain — hashes identify the request, not
/// the generated output.
pub const RICHNESS_REQUEST_DOMAIN: &[u8] = b"dungeon-gen/v3-richness/v1/request";

// ── Canonical serialization: RichnessDocumentV1 ────────────────────────────

impl RichnessDocumentV1 {
    /// Serialize to canonical byte representation.
    ///
    /// Returns bytes with fixed field order, lowercase tags, LF endings,
    /// and one terminal newline.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);

        // seed: decimal u64
        push_field(&mut buf, "seed", &self.seed.to_string());

        // extent: decimal u32
        push_field(&mut buf, "extent", &self.extent.to_string());

        // preset: lowercase tag
        push_field(&mut buf, "preset", self.preset.tag());

        // theme and gate: lowercase exact tags
        push_field(&mut buf, "theme", self.theme.tag());
        push_field(&mut buf, "gate", RichnessGateIdentity::V1.tag());

        // revision envelope — all lowercase exact tags
        push_field(
            &mut buf,
            "request_schema",
            self.request_schema_revision.tag(),
        );
        push_field(&mut buf, "algorithm", self.algorithm_revision.tag());
        push_field(&mut buf, "content", self.content_revision.tag());
        push_field(&mut buf, "preset_revision", self.preset_revision.tag());
        push_field(&mut buf, "theme_revision", self.theme_revision.tag());
        push_field(&mut buf, "asset", self.asset_revision.tag());
        push_field(&mut buf, "convention", self.convention_revision.tag());

        // Controls — encode InheritedOr with explicit-state markers
        push_inherited_or_u32(&mut buf, "landmarks", self.critical_path_landmarks);
        push_inherited_or_u32(&mut buf, "zones", self.zone_count);
        push_inherited_or_cave(&mut buf, "cave_mode", self.cave_mode);
        push_inherited_or_u32(&mut buf, "vertical_openings", self.vertical_openings);
        push_inherited_or_u32(&mut buf, "budget", self.budget_ceiling);

        buf
    }

    /// Parse from canonical bytes.
    ///
    /// Returns an error if the format is invalid, unknown fields are present,
    /// or any value is unrecognized.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RichnessError> {
        let text = std::str::from_utf8(bytes).map_err(|_| {
            RichnessError::new(
                RichnessErrorCode::UnknownRequestSchemaRevision,
                0,
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "canonical",
                RichnessErrorCategory::SchemaRevision,
                "canonical bytes are not valid UTF-8",
            )
        })?;

        let mut seed: Option<u64> = None;
        let mut extent: Option<u32> = None;
        let mut preset: Option<RichnessPreset> = None;
        let mut theme: Option<RichnessTheme> = None;
        let mut gate: Option<RichnessGateIdentity> = None;
        let mut request_schema_revision: Option<&str> = None;
        let mut algorithm_revision: Option<&str> = None;
        let mut content_revision: Option<&str> = None;
        let mut preset_revision: Option<&str> = None;
        let mut theme_revision: Option<&str> = None;
        let mut asset_revision: Option<&str> = None;
        let mut convention_revision: Option<&str> = None;
        let mut landmarks: Option<InheritedOr<u32>> = None;
        let mut zones: Option<InheritedOr<u32>> = None;
        let mut cave_mode: Option<InheritedOr<RichnessCaveMode>> = None;
        let mut vertical_openings: Option<InheritedOr<u32>> = None;
        let mut budget: Option<InheritedOr<u32>> = None;

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let (key, value) = split_field(line).ok_or_else(|| {
                RichnessError::new(
                    RichnessErrorCode::UnknownRequestSchemaRevision,
                    0,
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    "canonical",
                    RichnessErrorCategory::SchemaRevision,
                    format!("malformed canonical line: '{line}'"),
                )
            })?;

            match key {
                "seed" => {
                    seed = Some(parse_u64(value).map_err(|e| canonical_error("seed", &e))?);
                }
                "extent" => {
                    extent = Some(parse_u32(value).map_err(|e| canonical_error("extent", &e))?);
                }
                "preset" => {
                    preset = Some(RichnessPreset::from_tag(value).ok_or_else(|| {
                        canonical_error("preset", &format!("unknown preset tag '{value}'"))
                    })?);
                }
                "theme" => {
                    theme = Some(RichnessTheme::from_tag(value).ok_or_else(|| {
                        canonical_error("theme", &format!("unknown theme tag '{value}'"))
                    })?);
                }
                "gate" => {
                    gate = Some(RichnessGateIdentity::from_tag(value).ok_or_else(|| {
                        canonical_error_with_code(
                            RichnessErrorCode::UnsupportedRichnessGate,
                            "gate",
                            &format!("unsupported Richness gate '{value}'"),
                        )
                    })?);
                }
                "request_schema" => {
                    request_schema_revision = Some(value);
                }
                "algorithm" => {
                    algorithm_revision = Some(value);
                }
                "content" => {
                    content_revision = Some(value);
                }
                "preset_revision" => {
                    preset_revision = Some(value);
                }
                "theme_revision" => {
                    theme_revision = Some(value);
                }
                "asset" => {
                    asset_revision = Some(value);
                }
                "convention" => {
                    convention_revision = Some(value);
                }
                "landmarks" => {
                    landmarks = Some(
                        parse_inherited_or_u32(value)
                            .map_err(|e| canonical_error("landmarks", &e))?,
                    );
                }
                "zones" => {
                    zones = Some(
                        parse_inherited_or_u32(value).map_err(|e| canonical_error("zones", &e))?,
                    );
                }
                "cave_mode" => {
                    cave_mode = Some(
                        parse_inherited_or_cave(value)
                            .map_err(|e| canonical_error("cave_mode", &e))?,
                    );
                }
                "vertical_openings" => {
                    vertical_openings = Some(
                        parse_inherited_or_u32(value)
                            .map_err(|e| canonical_error("vertical_openings", &e))?,
                    );
                }
                "budget" => {
                    budget = Some(
                        parse_inherited_or_u32(value).map_err(|e| canonical_error("budget", &e))?,
                    );
                }
                _ => {
                    return Err(canonical_error(
                        "canonical",
                        &format!("unknown field '{key}'"),
                    ));
                }
            }
        }

        // Require all fields present
        let seed = seed.ok_or_else(|| canonical_error("seed", "missing field"))?;
        let extent = extent.ok_or_else(|| canonical_error("extent", "missing field"))?;
        let preset = preset.ok_or_else(|| canonical_error("preset", "missing field"))?;
        let theme = theme.ok_or_else(|| canonical_error("theme", "missing field"))?;
        let gate = gate.ok_or_else(|| canonical_error("gate", "missing field"))?;
        if gate != RichnessGateIdentity::V1 {
            return Err(canonical_error("gate", "unsupported Richness gate"));
        }

        // Parse revisions
        let request_schema_revision_str = request_schema_revision
            .ok_or_else(|| canonical_error("request_schema", "missing field"))?;
        let algorithm_revision_str =
            algorithm_revision.ok_or_else(|| canonical_error("algorithm", "missing field"))?;
        let content_revision_str =
            content_revision.ok_or_else(|| canonical_error("content", "missing field"))?;
        let preset_revision_str =
            preset_revision.ok_or_else(|| canonical_error("preset_revision", "missing field"))?;
        let theme_revision_str =
            theme_revision.ok_or_else(|| canonical_error("theme_revision", "missing field"))?;
        let asset_revision_str =
            asset_revision.ok_or_else(|| canonical_error("asset", "missing field"))?;
        let convention_revision_str =
            convention_revision.ok_or_else(|| canonical_error("convention", "missing field"))?;

        // Validate revision tags — unknown must fail closed
        let request_schema_revision =
            super::request::RichnessRequestSchemaRevision::from_tag(request_schema_revision_str)
                .ok_or_else(|| {
                    canonical_error_with_code(
                        RichnessErrorCode::UnknownRequestSchemaRevision,
                        "request_schema",
                        &format!("unknown revision '{request_schema_revision_str}'"),
                    )
                })?;
        let algorithm_revision =
            super::request::RichnessAlgorithmRevision::from_tag(algorithm_revision_str)
                .ok_or_else(|| {
                    canonical_error_with_code(
                        RichnessErrorCode::UnknownAlgorithmRevision,
                        "algorithm",
                        &format!("unknown revision '{algorithm_revision_str}'"),
                    )
                })?;
        let content_revision = super::request::RichnessContentRevision::from_tag(
            content_revision_str,
        )
        .ok_or_else(|| {
            canonical_error_with_code(
                RichnessErrorCode::UnknownContentRevision,
                "content",
                &format!("unknown revision '{content_revision_str}'"),
            )
        })?;
        let preset_revision = super::request::RichnessPresetRevision::from_tag(preset_revision_str)
            .ok_or_else(|| {
                canonical_error_with_code(
                    RichnessErrorCode::UnknownPresetRevision,
                    "preset_revision",
                    &format!("unknown revision '{preset_revision_str}'"),
                )
            })?;
        let theme_revision = super::request::RichnessThemeRevision::from_tag(theme_revision_str)
            .ok_or_else(|| {
                canonical_error_with_code(
                    RichnessErrorCode::UnknownThemeRevision,
                    "theme_revision",
                    &format!("unknown revision '{theme_revision_str}'"),
                )
            })?;
        let asset_revision = super::request::RichnessAssetRevision::from_tag(asset_revision_str)
            .ok_or_else(|| {
                canonical_error_with_code(
                    RichnessErrorCode::UnknownAssetRevision,
                    "asset",
                    &format!("unknown revision '{asset_revision_str}'"),
                )
            })?;
        let convention_revision =
            super::request::RichnessConventionRevision::from_tag(convention_revision_str)
                .ok_or_else(|| {
                    canonical_error_with_code(
                        RichnessErrorCode::UnknownConventionRevision,
                        "convention",
                        &format!("unknown revision '{convention_revision_str}'"),
                    )
                })?;

        let landmarks = landmarks.ok_or_else(|| canonical_error("landmarks", "missing field"))?;
        let zones = zones.ok_or_else(|| canonical_error("zones", "missing field"))?;
        let cave_mode = cave_mode.ok_or_else(|| canonical_error("cave_mode", "missing field"))?;
        let vertical_openings = vertical_openings
            .ok_or_else(|| canonical_error("vertical_openings", "missing field"))?;
        let budget = budget.ok_or_else(|| canonical_error("budget", "missing field"))?;

        let document = RichnessDocumentV1::with_all_explicit(
            seed,
            extent,
            preset,
            theme,
            request_schema_revision,
            algorithm_revision,
            content_revision,
            preset_revision,
            theme_revision,
            asset_revision,
            convention_revision,
            landmarks,
            zones,
            cave_mode,
            vertical_openings,
            budget,
        )?;

        if document.to_canonical_bytes() != bytes {
            return Err(canonical_error(
                "canonical",
                "input is not the fixed-order canonical representation",
            ));
        }

        Ok(document)
    }
}

// ── Canonical serialization: ResolvedRichnessRequestV1 ─────────────────────

impl ResolvedRichnessRequestV1 {
    /// Serialize to canonical byte representation with resolved values
    /// and provenance markers.
    ///
    /// The canonical form includes explicit-state markers AND resolved
    /// values so both provenance and reproduction are auditable.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(768);

        // Re-emit the provenance document first
        buf.extend_from_slice(&self.provenance.to_canonical_bytes());

        // Then append resolved values section
        buf.extend_from_slice(b"---resolved\n");

        push_resolved_field_u32(&mut buf, "landmarks", &self.critical_path_landmarks);
        push_resolved_field_u32(&mut buf, "zones", &self.zone_count);
        push_resolved_field_cave(&mut buf, "cave_mode", &self.cave_mode);
        push_resolved_field_u32(&mut buf, "vertical_openings", &self.vertical_openings);
        push_resolved_field_u32(&mut buf, "budget", &self.budget_ceiling);

        buf
    }

    /// Parse from canonical bytes produced by `to_canonical_bytes`.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RichnessError> {
        let text = std::str::from_utf8(bytes).map_err(|_| {
            RichnessError::new(
                RichnessErrorCode::UnknownRequestSchemaRevision,
                0,
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "?",
                "canonical",
                RichnessErrorCategory::SchemaRevision,
                "canonical bytes are not valid UTF-8",
            )
        })?;

        // Split at the resolved section marker
        let (doc_text, resolved_text) = match text.split_once("---resolved\n") {
            Some((doc, res)) => (doc, res),
            None => {
                return Err(canonical_error(
                    "canonical",
                    "missing ---resolved section marker",
                ));
            }
        };

        let doc = RichnessDocumentV1::from_canonical_bytes(doc_text.as_bytes())?;
        let resolved = ResolvedRichnessRequestV1::resolve(doc)?;

        // Validate that the resolved section matches the resolved values
        // (we parse it but don't override — the resolved request is the authority)
        for line in resolved_text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (key, _value) = split_field(line).ok_or_else(|| {
                canonical_error("canonical", &format!("malformed resolved line: '{line}'"))
            })?;

            // Validate known keys
            match key {
                "landmarks" | "zones" | "cave_mode" | "vertical_openings" | "budget" => {}
                _ => {
                    return Err(canonical_error(
                        "canonical",
                        &format!("unknown resolved field '{key}'"),
                    ));
                }
            }
        }

        if resolved.to_canonical_bytes() != bytes {
            return Err(canonical_error(
                "canonical",
                "resolved values or field order do not match the canonical request",
            ));
        }

        Ok(resolved)
    }
}

// ── Identity hashing ───────────────────────────────────────────────────────

impl RichnessDocumentV1 {
    /// Compute the deterministic identity hash for this authored document.
    ///
    /// Hash framing is binary and frozen: a length-framed UTF-8 domain and
    /// field tags, followed by little-endian integer values and explicit
    /// provenance markers. Canonical text is deliberately not used as an
    /// implicit binary framing layer.
    pub fn identity_hash(&self) -> [u8; 32] {
        Sha256::digest(self.identity_hash_iter()).into()
    }

    /// Identity hash as a lowercase hex string.
    pub fn identity_hash_hex(&self) -> String {
        hex_encode(&self.identity_hash())
    }

    /// Return the frozen hash input for byte-vector tests.
    fn identity_hash_iter(&self) -> Vec<u8> {
        let mut buf = identity_prefix("authored-request");
        write_field_u64(&mut buf, "seed", self.seed);
        write_field_u32(&mut buf, "extent", self.extent);
        write_field_tag(&mut buf, "preset", self.preset.tag());
        write_field_tag(&mut buf, "theme", self.theme.tag());
        write_field_tag(&mut buf, "gate", RichnessGateIdentity::V1.tag());
        write_field_tag(
            &mut buf,
            "request_schema",
            self.request_schema_revision.tag(),
        );
        write_field_tag(&mut buf, "algorithm", self.algorithm_revision.tag());
        write_field_tag(&mut buf, "content", self.content_revision.tag());
        write_field_tag(&mut buf, "preset_revision", self.preset_revision.tag());
        write_field_tag(&mut buf, "theme_revision", self.theme_revision.tag());
        write_field_tag(&mut buf, "asset", self.asset_revision.tag());
        write_field_tag(&mut buf, "convention", self.convention_revision.tag());
        write_inherited_u32(&mut buf, "landmarks", self.critical_path_landmarks);
        write_inherited_u32(&mut buf, "zones", self.zone_count);
        write_inherited_cave(&mut buf, "cave_mode", self.cave_mode);
        write_inherited_u32(&mut buf, "vertical_openings", self.vertical_openings);
        write_inherited_u32(&mut buf, "budget", self.budget_ceiling);
        buf
    }
}

impl ResolvedRichnessRequestV1 {
    /// Compute the deterministic identity hash for this resolved request.
    pub fn identity_hash(&self) -> [u8; 32] {
        Sha256::digest(self.identity_hash_iter()).into()
    }

    /// Identity hash as a lowercase hex string.
    pub fn identity_hash_hex(&self) -> String {
        hex_encode(&self.identity_hash())
    }

    fn identity_hash_iter(&self) -> Vec<u8> {
        let mut buf = identity_prefix("resolved-request");
        buf.extend_from_slice(&self.provenance.identity_hash_iter());
        write_resolved_u32(&mut buf, "landmarks", self.critical_path_landmarks);
        write_resolved_u32(&mut buf, "zones", self.zone_count);
        write_resolved_cave(&mut buf, "cave_mode", self.cave_mode);
        write_resolved_u32(&mut buf, "vertical_openings", self.vertical_openings);
        write_resolved_u32(&mut buf, "budget", self.budget_ceiling);
        buf
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Start a frozen binary identity frame with the request domain and form tag.
fn identity_prefix(form: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512);
    write_tag(&mut buf, RICHNESS_REQUEST_DOMAIN);
    write_tag(&mut buf, form.as_bytes());
    buf
}

/// Write a length-framed UTF-8 tag using a little-endian u32 length.
fn write_tag(buf: &mut Vec<u8>, tag: &[u8]) {
    buf.extend_from_slice(&(tag.len() as u32).to_le_bytes());
    buf.extend_from_slice(tag);
}

fn write_field_u64(buf: &mut Vec<u8>, field: &str, value: u64) {
    write_tag(buf, field.as_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_field_u32(buf: &mut Vec<u8>, field: &str, value: u32) {
    write_tag(buf, field.as_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_field_tag(buf: &mut Vec<u8>, field: &str, value: &str) {
    write_tag(buf, field.as_bytes());
    write_tag(buf, value.as_bytes());
}

fn write_inherited_u32(buf: &mut Vec<u8>, field: &str, value: InheritedOr<u32>) {
    write_tag(buf, field.as_bytes());
    match value {
        InheritedOr::Inherited => buf.push(0),
        InheritedOr::Explicit(value) => {
            buf.push(1);
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn write_inherited_cave(buf: &mut Vec<u8>, field: &str, value: InheritedOr<RichnessCaveMode>) {
    write_tag(buf, field.as_bytes());
    match value {
        InheritedOr::Inherited => buf.push(0),
        InheritedOr::Explicit(value) => {
            buf.push(1);
            write_tag(buf, value.tag().as_bytes());
        }
    }
}

fn write_resolved_u32(buf: &mut Vec<u8>, field: &str, value: super::request::ResolvedField<u32>) {
    write_tag(buf, field.as_bytes());
    buf.extend_from_slice(&value.value.to_le_bytes());
    write_tag(buf, value.source.tag().as_bytes());
}

fn write_resolved_cave(
    buf: &mut Vec<u8>,
    field: &str,
    value: super::request::ResolvedField<RichnessCaveMode>,
) {
    write_tag(buf, field.as_bytes());
    write_tag(buf, value.value.tag().as_bytes());
    write_tag(buf, value.source.tag().as_bytes());
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Push a `key:value\n` line.
fn push_field(buf: &mut Vec<u8>, key: &str, value: &str) {
    buf.extend_from_slice(key.as_bytes());
    buf.push(b':');
    buf.extend_from_slice(value.as_bytes());
    buf.push(b'\n');
}

/// Push an InheritedOr<u32> field.
fn push_inherited_or_u32(buf: &mut Vec<u8>, key: &str, value: InheritedOr<u32>) {
    match value {
        InheritedOr::Inherited => push_field(buf, key, "inherited"),
        InheritedOr::Explicit(v) => push_field(buf, key, &format!("explicit:{v}")),
    }
}

/// Push an InheritedOr<RichnessCaveMode> field.
fn push_inherited_or_cave(buf: &mut Vec<u8>, key: &str, value: InheritedOr<RichnessCaveMode>) {
    match value {
        InheritedOr::Inherited => push_field(buf, key, "inherited"),
        InheritedOr::Explicit(v) => push_field(buf, key, &format!("explicit:{}", v.tag())),
    }
}

/// Push a resolved field.
fn push_resolved_field_u32(
    buf: &mut Vec<u8>,
    key: &str,
    field: &super::request::ResolvedField<u32>,
) {
    push_field(
        buf,
        key,
        &format!("{} source={}", field.value, field.source.tag()),
    );
}

/// Push a resolved cave mode field.
fn push_resolved_field_cave(
    buf: &mut Vec<u8>,
    key: &str,
    field: &super::request::ResolvedField<RichnessCaveMode>,
) {
    push_field(
        buf,
        key,
        &format!("{} source={}", field.value.tag(), field.source.tag()),
    );
}

/// Split a `key:value` line. Returns `None` if no colon.
fn split_field(line: &str) -> Option<(&str, &str)> {
    let colon = line.find(':')?;
    Some((&line[..colon], &line[colon + 1..]))
}

/// Parse a decimal u64.
fn parse_u64(s: &str) -> Result<u64, String> {
    s.parse::<u64>()
        .map_err(|e| format!("invalid u64 '{s}': {e}"))
}

/// Parse a decimal u32.
fn parse_u32(s: &str) -> Result<u32, String> {
    s.parse::<u32>()
        .map_err(|e| format!("invalid u32 '{s}': {e}"))
}

/// Parse an InheritedOr<u32> from a canonical value string.
fn parse_inherited_or_u32(s: &str) -> Result<InheritedOr<u32>, String> {
    if s == "inherited" {
        return Ok(InheritedOr::Inherited);
    }
    if let Some(rest) = s.strip_prefix("explicit:") {
        let v = rest
            .parse::<u32>()
            .map_err(|e| format!("invalid explicit value '{rest}': {e}"))?;
        return Ok(InheritedOr::Explicit(v));
    }
    Err(format!("invalid InheritedOr<u32> value: '{s}'"))
}

/// Parse an InheritedOr<RichnessCaveMode> from a canonical value string.
fn parse_inherited_or_cave(s: &str) -> Result<InheritedOr<RichnessCaveMode>, String> {
    if s == "inherited" {
        return Ok(InheritedOr::Inherited);
    }
    if let Some(rest) = s.strip_prefix("explicit:") {
        let mode = RichnessCaveMode::from_tag(rest)
            .ok_or_else(|| format!("unknown cave mode tag '{rest}'"))?;
        return Ok(InheritedOr::Explicit(mode));
    }
    Err(format!("invalid InheritedOr<cave> value: '{s}'"))
}

/// Create a canonical parse error.
fn canonical_error(path: &str, detail: &str) -> RichnessError {
    canonical_error_with_code(
        RichnessErrorCode::UnknownRequestSchemaRevision,
        path,
        detail,
    )
}

fn canonical_error_with_code(code: RichnessErrorCode, path: &str, detail: &str) -> RichnessError {
    RichnessError::new(
        code,
        0,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        path,
        RichnessErrorCategory::SchemaRevision,
        detail.to_string(),
    )
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::request::{
        InheritedOr, RichnessCaveMode, RichnessDocumentV1, RichnessPreset, RichnessTheme,
        ValueSource,
    };
    use super::*;

    // ── Canonical byte vector tests ────────────────────────────────────

    #[test]
    fn canonical_roundtrip_sparse_inherited() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let bytes = doc.to_canonical_bytes();
        let doc2 = RichnessDocumentV1::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(doc, doc2);
    }

    #[test]
    fn canonical_roundtrip_moderate_inherited() {
        let doc =
            RichnessDocumentV1::new(99, 2048, RichnessPreset::Moderate, RichnessTheme::Egyptian)
                .unwrap();
        let bytes = doc.to_canonical_bytes();
        let doc2 = RichnessDocumentV1::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(doc, doc2);
    }

    #[test]
    fn canonical_roundtrip_rich_explicit() {
        let doc = RichnessDocumentV1::with_all_explicit(
            255,
            3072,
            RichnessPreset::Rich,
            RichnessTheme::Brutalist,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(3),
            InheritedOr::Explicit(2),
            InheritedOr::Explicit(RichnessCaveMode::Required),
            InheritedOr::Explicit(6),
            InheritedOr::Explicit(8000),
        )
        .unwrap();
        let bytes = doc.to_canonical_bytes();
        let doc2 = RichnessDocumentV1::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(doc, doc2);
    }

    #[test]
    fn canonical_roundtrip_preserves_explicit_same_as_default() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(1), // same as Sparse default
            InheritedOr::Explicit(1), // same as Sparse default
            InheritedOr::Explicit(RichnessCaveMode::Preferred), // same as Sparse default
            InheritedOr::Explicit(0), // same as Sparse default
            InheritedOr::Explicit(3000), // same as Sparse default
        )
        .unwrap();
        let bytes = doc.to_canonical_bytes();
        let doc2 = RichnessDocumentV1::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(doc, doc2);
        // Verify explicit state preserved
        assert!(doc2.critical_path_landmarks.is_explicit());
        assert_eq!(doc2.critical_path_landmarks, InheritedOr::Explicit(1));
    }

    #[test]
    fn canonical_bytes_are_valid_utf8() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let bytes = doc.to_canonical_bytes();
        assert!(std::str::from_utf8(&bytes).is_ok());
    }

    #[test]
    fn canonical_bytes_end_with_newline() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let bytes = doc.to_canonical_bytes();
        assert_eq!(bytes.last(), Some(&b'\n'));
    }

    #[test]
    fn canonical_bytes_use_lf_only() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let bytes = doc.to_canonical_bytes();
        assert!(!bytes.contains(&b'\r'));
    }

    #[test]
    fn canonical_has_frozen_field_order() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let bytes = doc.to_canonical_bytes();
        let text = std::str::from_utf8(&bytes).unwrap();

        // Verify fields appear in frozen order by checking relative positions
        let seed_pos = text.find("seed:").unwrap();
        let extent_pos = text.find("extent:").unwrap();
        let preset_pos = text.find("preset:").unwrap();
        let theme_pos = text.find("theme:").unwrap();
        let req_pos = text.find("request_schema:").unwrap();
        let budget_pos = text.find("budget:").unwrap();

        assert!(seed_pos < extent_pos);
        assert!(extent_pos < preset_pos);
        assert!(preset_pos < theme_pos);
        assert!(theme_pos < req_pos);
        // budget should be after all other fields
        assert!(budget_pos > req_pos);
        assert!(budget_pos > text.find("vertical_openings:").unwrap());
        // budget should be the last field (no field after it)
        let after_budget = &text[budget_pos + "budget:inherited".len()..];
        assert_eq!(after_budget, "\n");
    }

    #[test]
    fn canonical_unknown_field_rejected() {
        let input = b"seed:0\nextent:2048\npreset:sparse\ntheme:ancient\nrequest_schema:v1\nalgorithm:v1\ncontent:v1\npreset_revision:v1\ntheme_revision:v1\nasset:v1\nconvention:v1\nlandmarks:inherited\nzones:inherited\ncave_mode:inherited\nvertical_openings:inherited\nbudget:inherited\nunknown:value\n";
        let result = RichnessDocumentV1::from_canonical_bytes(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.context.contains("unknown field"));
    }

    #[test]
    fn canonical_rejects_noncanonical_order_duplicates_and_line_endings() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let canonical = String::from_utf8(doc.to_canonical_bytes()).unwrap();

        assert!(RichnessDocumentV1::from_canonical_bytes(
            canonical
                .replacen("seed:0\nextent:2048", "extent:2048\nseed:0", 1)
                .as_bytes()
        )
        .is_err());
        assert!(RichnessDocumentV1::from_canonical_bytes(
            format!("{canonical}seed:0\n").as_bytes()
        )
        .is_err());
        assert!(RichnessDocumentV1::from_canonical_bytes(canonical.trim_end().as_bytes()).is_err());
        assert!(RichnessDocumentV1::from_canonical_bytes(
            canonical.replace('\n', "\r\n").as_bytes()
        )
        .is_err());
    }

    #[test]
    fn canonical_rejects_unsupported_gate_with_stable_code() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let input = String::from_utf8(doc.to_canonical_bytes())
            .unwrap()
            .replace("gate:richness-v1", "gate:m3");
        let error = RichnessDocumentV1::from_canonical_bytes(input.as_bytes()).unwrap_err();
        assert_eq!(error.code, RichnessErrorCode::UnsupportedRichnessGate);
        assert_eq!(error.path, "gate");
    }

    #[test]
    fn canonical_unknown_revision_rejected() {
        let input = b"seed:0\nextent:2048\npreset:sparse\ntheme:ancient\nrequest_schema:v99\nalgorithm:v1\ncontent:v1\npreset_revision:v1\ntheme_revision:v1\nasset:v1\nconvention:v1\nlandmarks:inherited\nzones:inherited\ncave_mode:inherited\nvertical_openings:inherited\nbudget:inherited\n";
        let result = RichnessDocumentV1::from_canonical_bytes(input);
        assert!(result.is_err());
    }

    #[test]
    fn canonical_every_unknown_revision_has_its_stable_code() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let canonical = String::from_utf8(doc.to_canonical_bytes()).unwrap();
        let cases = [
            (
                "request_schema:enhanced-v3-richness-request/v1",
                RichnessErrorCode::UnknownRequestSchemaRevision,
            ),
            (
                "algorithm:enhanced-v3-richness-algorithm/v1",
                RichnessErrorCode::UnknownAlgorithmRevision,
            ),
            (
                "content:enhanced-v3-richness-content/v1",
                RichnessErrorCode::UnknownContentRevision,
            ),
            (
                "preset_revision:enhanced-v3-richness-presets/v1",
                RichnessErrorCode::UnknownPresetRevision,
            ),
            (
                "theme_revision:enhanced-v3-richness-themes/v1",
                RichnessErrorCode::UnknownThemeRevision,
            ),
            (
                "asset:enhanced-v3-richness-assets/v1",
                RichnessErrorCode::UnknownAssetRevision,
            ),
            (
                "convention:enhanced-v3-richness-conventions/v1",
                RichnessErrorCode::UnknownConventionRevision,
            ),
        ];

        for (field, expected_code) in cases {
            let input = canonical.replacen(field, &format!("{field}-unknown"), 1);
            let error = RichnessDocumentV1::from_canonical_bytes(input.as_bytes()).unwrap_err();
            assert_eq!(error.code, expected_code, "field {field}");
        }
    }

    #[test]
    fn canonical_unknown_preset_rejected() {
        let input = b"seed:0\nextent:2048\npreset:unknown\ntheme:ancient\nrequest_schema:v1\nalgorithm:v1\ncontent:v1\npreset_revision:v1\ntheme_revision:v1\nasset:v1\nconvention:v1\nlandmarks:inherited\nzones:inherited\ncave_mode:inherited\nvertical_openings:inherited\nbudget:inherited\n";
        let result = RichnessDocumentV1::from_canonical_bytes(input);
        assert!(result.is_err());
    }

    #[test]
    fn canonical_missing_field_rejected() {
        let input = b"seed:0\nextent:2048\npreset:sparse\ntheme:ancient\n";
        let result = RichnessDocumentV1::from_canonical_bytes(input);
        assert!(result.is_err());
    }

    #[test]
    fn canonical_non_utf8_rejected() {
        let input = [0xFF, 0xFE, 0x00, 0x00];
        let result = RichnessDocumentV1::from_canonical_bytes(&input);
        assert!(result.is_err());
    }

    // ── Frozen canonical byte vectors ──────────────────────────────────

    #[test]
    fn frozen_canonical_vector_sparse_inherited() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let bytes = doc.to_canonical_bytes();
        let expected = concat!(
            "seed:0\nextent:2048\npreset:sparse\ntheme:ancient\ngate:richness-v1\n",
            "request_schema:enhanced-v3-richness-request/v1\n",
            "algorithm:enhanced-v3-richness-algorithm/v1\n",
            "content:enhanced-v3-richness-content/v1\n",
            "preset_revision:enhanced-v3-richness-presets/v1\n",
            "theme_revision:enhanced-v3-richness-themes/v1\n",
            "asset:enhanced-v3-richness-assets/v1\n",
            "convention:enhanced-v3-richness-conventions/v1\n",
            "landmarks:inherited\nzones:inherited\ncave_mode:inherited\n",
            "vertical_openings:inherited\nbudget:inherited\n",
        );
        assert_eq!(bytes, expected.as_bytes());
    }

    #[test]
    fn frozen_canonical_vector_rich_explicit() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            3072,
            RichnessPreset::Rich,
            RichnessTheme::Egyptian,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(3),
            InheritedOr::Inherited,
            InheritedOr::Explicit(RichnessCaveMode::Required),
            InheritedOr::Inherited,
            InheritedOr::Explicit(8000),
        )
        .unwrap();
        let bytes = doc.to_canonical_bytes();
        let expected = concat!(
            "seed:42\nextent:3072\npreset:rich\ntheme:egyptian\ngate:richness-v1\n",
            "request_schema:enhanced-v3-richness-request/v1\n",
            "algorithm:enhanced-v3-richness-algorithm/v1\n",
            "content:enhanced-v3-richness-content/v1\n",
            "preset_revision:enhanced-v3-richness-presets/v1\n",
            "theme_revision:enhanced-v3-richness-themes/v1\n",
            "asset:enhanced-v3-richness-assets/v1\n",
            "convention:enhanced-v3-richness-conventions/v1\n",
            "landmarks:explicit:3\nzones:inherited\ncave_mode:explicit:required\n",
            "vertical_openings:inherited\nbudget:explicit:8000\n",
        );
        assert_eq!(bytes, expected.as_bytes());
    }

    // ── Determinism tests ──────────────────────────────────────────────

    #[test]
    fn canonical_bytes_deterministic() {
        let doc =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient)
                .unwrap();
        let a = doc.to_canonical_bytes();
        let b = doc.to_canonical_bytes();
        assert_eq!(a, b);

        // Different document = different bytes
        let doc2 =
            RichnessDocumentV1::new(43, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient)
                .unwrap();
        assert_ne!(a, doc2.to_canonical_bytes());
    }

    // ── Resolved canonical tests ───────────────────────────────────────

    #[test]
    fn resolved_canonical_roundtrip() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let bytes = resolved.to_canonical_bytes();
        let resolved2 = ResolvedRichnessRequestV1::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(resolved, resolved2);
    }

    #[test]
    fn resolved_canonical_includes_resolved_values() {
        let doc =
            RichnessDocumentV1::new(99, 2048, RichnessPreset::Moderate, RichnessTheme::Egyptian)
                .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let bytes = resolved.to_canonical_bytes();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("---resolved"));
        assert!(text.contains("source=inherited"));
    }

    #[test]
    fn resolved_canonical_rejects_tampered_resolved_values() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        let tampered = String::from_utf8(resolved.to_canonical_bytes())
            .unwrap()
            .replace(
                "landmarks:1 source=inherited",
                "landmarks:2 source=inherited",
            );
        assert!(ResolvedRichnessRequestV1::from_canonical_bytes(tampered.as_bytes()).is_err());
    }

    #[test]
    fn resolved_canonical_preserves_source() {
        let doc = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Inherited,   // inherited = 1 from Sparse
            InheritedOr::Explicit(2), // explicit = 2 (within Sparse max_zones=3)
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        )
        .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc).unwrap();
        assert_eq!(
            resolved.critical_path_landmarks.source,
            ValueSource::Inherited
        );
        assert_eq!(resolved.zone_count.source, ValueSource::Explicit);

        let bytes = resolved.to_canonical_bytes();
        let resolved2 = ResolvedRichnessRequestV1::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(
            resolved2.critical_path_landmarks.source,
            ValueSource::Inherited
        );
        assert_eq!(resolved2.zone_count.source, ValueSource::Explicit);
    }

    // ── Identity hash tests ────────────────────────────────────────────

    #[test]
    fn identity_hash_deterministic() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let h1 = doc.identity_hash();
        let h2 = doc.identity_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn identity_hash_different_documents() {
        let doc1 =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let doc2 =
            RichnessDocumentV1::new(43, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        assert_ne!(doc1.identity_hash(), doc2.identity_hash());
    }

    #[test]
    fn identity_hash_different_presets() {
        let doc1 =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let doc2 =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Moderate, RichnessTheme::Ancient)
                .unwrap();
        assert_ne!(doc1.identity_hash(), doc2.identity_hash());
    }

    #[test]
    fn identity_hash_different_themes() {
        let doc1 =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let doc2 =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Brutalist)
                .unwrap();
        assert_ne!(doc1.identity_hash(), doc2.identity_hash());
    }

    #[test]
    fn identity_hash_explicit_state_affects_hash() {
        // Two documents with identical resolved values but different source
        // MUST produce different hashes (explicit state is part of identity)
        let doc_inherited =
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap();
        let doc_explicit = RichnessDocumentV1::with_all_explicit(
            42,
            2048,
            RichnessPreset::Sparse,
            RichnessTheme::Ancient,
            super::super::request::RichnessRequestSchemaRevision::V1,
            super::super::request::RichnessAlgorithmRevision::V1,
            super::super::request::RichnessContentRevision::V1,
            super::super::request::RichnessPresetRevision::V1,
            super::super::request::RichnessThemeRevision::V1,
            super::super::request::RichnessAssetRevision::V1,
            super::super::request::RichnessConventionRevision::V1,
            InheritedOr::Explicit(1), // same value as Sparse default, but explicit
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
            InheritedOr::Inherited,
        )
        .unwrap();
        assert_ne!(doc_inherited.identity_hash(), doc_explicit.identity_hash());
    }

    #[test]
    fn identity_hash_hex_is_lowercase_64_chars() {
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let hex = doc.identity_hash_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()));
    }

    #[test]
    fn frozen_identity_hash_vector() {
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let hash = doc.identity_hash();
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        // This hash is frozen; changing it means the contract changed.
        assert_eq!(
            hex, "0703a20c9e5e5b5cddd0e60ed591fcb3ef5d4c40412eb68d3a5927245213e1a7",
            "frozen identity hash vector changed — contract may have drifted"
        );
    }

    #[test]
    fn resolved_identity_hash_equals_doc_hash_when_all_inherited() {
        // When everything is inherited, resolved hash should match doc hash
        // (because the resolved canonical form embeds the doc)
        let doc = RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let resolved = ResolvedRichnessRequestV1::resolve(doc.clone()).unwrap();
        // They differ because resolved includes the resolved section
        assert_ne!(doc.identity_hash(), resolved.identity_hash());
    }

    #[test]
    fn domain_is_frozen() {
        assert_eq!(
            RICHNESS_REQUEST_DOMAIN,
            b"dungeon-gen/v3-richness/v1/request"
        );
    }

    #[test]
    fn hash_uses_length_framed_domain() {
        // Verify the hash includes a length-prefixed domain, not raw domain
        let doc = RichnessDocumentV1::new(0, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
            .unwrap();
        let hash_input = doc.identity_hash_iter();
        // First 4 bytes = LE u32 length of domain
        let domain_len =
            u32::from_le_bytes([hash_input[0], hash_input[1], hash_input[2], hash_input[3]]);
        assert_eq!(domain_len as usize, RICHNESS_REQUEST_DOMAIN.len());
        // Next bytes = domain
        assert_eq!(
            &hash_input[4..4 + domain_len as usize],
            RICHNESS_REQUEST_DOMAIN
        );
    }
}
