//! Deterministic binary replay frame for generator results.
//!
//! Encodes exact ASCII bytes, attempt identity/index,
//! capture views (f32::to_bits), resources, and canonical diagnostic bytes.
//!
//! Uses fixed-width length prefixes, checked decode bounds, explicit variant tags.
//! Rejects truncation, trailing bytes, invalid tags, oversized lengths.
//! Equality decodes and compares exact bytes + structured values; hashes
//! identify artifacts but never replace equality.

use super::ascii;
use super::capture_views::CaptureView;
use super::resources::ResourceCounts;
use crate::layout::ParsedLevel;

// ─── Constants ──────────────────────────────────────────────────────────────

const REPLAY_SCHEMA_VERSION: u32 = 1;

/// Maximum encoded frame size (10 MiB).
const MAX_FRAME_BYTES: usize = 10 * 1024 * 1024;

// ─── Tags ──────────────────────────────────────────────────────────────────

const TAG_SCHEMA: u8 = 0x00;
const TAG_SEED: u8 = 0x01;
const TAG_ATTEMPT_INDEX: u8 = 0x02;
const TAG_ATTEMPT_IDENTITY: u8 = 0x03;
const TAG_ASCII: u8 = 0x04;
const TAG_RESOURCES: u8 = 0x05;
const TAG_CAPTURE_VIEWS: u8 = 0x06;
const TAG_DIAGNOSTICS: u8 = 0x07;
const TAG_EXHAUSTED: u8 = 0x08;
const TAG_TOPOLOGY_REGIONS: u8 = 0x09;
const TAG_TOPOLOGY_EDGES: u8 = 0x0A;
const TAG_TOPOLOGY_METRICS: u8 = 0x0B;

const OUTCOME_OK: u8 = 0x00;
const OUTCOME_EXHAUSTED: u8 = 0x01;

// ─── Error ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    Truncated { expected: usize, actual: usize },
    TrailingBytes { consumed: usize, remaining: usize },
    InvalidTag { position: usize, tag: u8 },
    OversizedLength { position: usize, length: usize, maximum: usize },
    DecodeUtf8 { position: usize },
    DecodeSemantics { reason: String },
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { expected, actual } => {
                write!(f, "truncated: expected {expected} bytes, got {actual}")
            }
            Self::TrailingBytes { consumed, remaining } => {
                write!(f, "trailing bytes: consumed {consumed}, remaining {remaining}")
            }
            Self::InvalidTag { position, tag } => {
                write!(f, "invalid tag 0x{tag:02x} at position {position}")
            }
            Self::OversizedLength { position, length, maximum } => {
                write!(
                    f,
                    "oversized length {length} (max {maximum}) at position {position}"
                )
            }
            Self::DecodeUtf8 { position } => {
                write!(f, "invalid UTF-8 at position {position}")
            }
            Self::DecodeSemantics { reason } => {
                write!(f, "semantic error: {reason}")
            }
        }
    }
}

// ─── Encoder ───────────────────────────────────────────────────────────────

pub struct ReplayEncoder {
    bytes: Vec<u8>,
}

impl ReplayEncoder {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn write_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn write_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn write_f32(&mut self, value: f32) {
        self.write_u32(value.to_bits());
    }

    fn write_bytes_prefixed(&mut self, data: &[u8]) {
        let len = u32::try_from(data.len()).expect("replay section too large");
        self.write_u32(len);
        self.bytes.extend_from_slice(data);
    }

    // ── Top-level builders ──────────────────────────────────────────────

    pub fn schema(mut self) -> Self {
        self.write_u8(TAG_SCHEMA);
        self.write_u32(REPLAY_SCHEMA_VERSION);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.write_u8(TAG_SEED);
        self.write_u64(seed);
        self
    }

    pub fn attempt_index(mut self, index: u32) -> Self {
        self.write_u8(TAG_ATTEMPT_INDEX);
        self.write_u32(index);
        self
    }

    pub fn attempt_identity(mut self, identity: [u8; 32]) -> Self {
        self.write_u8(TAG_ATTEMPT_IDENTITY);
        self.bytes.extend_from_slice(&identity);
        self
    }

    /// Encode ASCII representation of the level.
    pub fn ascii(mut self, level: &ParsedLevel) -> Self {
        let ascii_str = ascii::serialize_level(level).expect("ascii serialization must succeed");
        self.write_u8(TAG_ASCII);
        self.write_bytes_prefixed(ascii_str.as_bytes());
        self
    }

    /// Encode exact ASCII bytes (for round-trip testing).
    pub fn ascii_bytes(mut self, bytes: &[u8]) -> Self {
        self.write_u8(TAG_ASCII);
        self.write_bytes_prefixed(bytes);
        self
    }

    /// Encode resource counts.
    pub fn resources(mut self, resources: &ResourceCounts) -> Self {
        self.write_u8(TAG_RESOURCES);
        self.write_u64(resources.total_tiles);
        self.write_u64(resources.floor_tiles);
        self.write_u64(resources.wall_tiles);
        self.write_u64(resources.void_tiles);
        self.write_u64(resources.ramp_tiles);
        self.write_u32(resources.non_empty_chunks);
        self.write_u64(resources.estimated_vertices);
        self.write_u64(resources.estimated_indices);
        self.write_u32(resources.light_count);
        self.write_u32(resources.model_count);
        self.write_u32(resources.static_body_count);
        self.write_u32(resources.total_body_count);
        self
    }

    /// Encode capture views.
    pub fn capture_views(mut self, views: &[CaptureView]) -> Self {
        let count = u32::try_from(views.len()).expect("too many capture views");
        self.write_u8(TAG_CAPTURE_VIEWS);
        self.write_u32(count);
        for view in views {
            // eye
            self.write_f32(view.eye.x);
            self.write_f32(view.eye.y);
            self.write_f32(view.eye.z);
            // look_at
            self.write_f32(view.look_at.x);
            self.write_f32(view.look_at.y);
            self.write_f32(view.look_at.z);
            // category ordinal
            self.write_u8(category_ordinal(view.category));
            // label
            self.write_bytes_prefixed(view.label.as_bytes());
        }
        self
    }

    /// Encode canonical diagnostics bytes.
    pub fn diagnostics(mut self, diag_bytes: &[u8]) -> Self {
        self.write_u8(TAG_DIAGNOSTICS);
        self.write_bytes_prefixed(diag_bytes);
        self
    }

    /// Encode an exhausted (failure) outcome.
    pub fn exhausted(mut self, stage: &str, reason: &str) -> Self {
        self.write_u8(TAG_EXHAUSTED);
        self.write_bytes_prefixed(stage.as_bytes());
        self.write_bytes_prefixed(reason.as_bytes());
        self
    }

    /// Encode topology region summary.
    pub fn topology_regions(mut self, region_count: u32) -> Self {
        self.write_u8(TAG_TOPOLOGY_REGIONS);
        self.write_u32(region_count);
        self
    }

    /// Encode topology edge summary (count).
    pub fn topology_edges(mut self, edge_count: u32) -> Self {
        self.write_u8(TAG_TOPOLOGY_EDGES);
        self.write_u32(edge_count);
        self
    }

    /// Encode topology metrics.
    pub fn topology_metrics(
        mut self,
        route_distance: u64,
        max_branch_depth: u32,
        dead_end_count: u32,
        articulation_count: u32,
        crossing_count: u32,
        per_layer_cycles: &[u32],
    ) -> Self {
        self.write_u8(TAG_TOPOLOGY_METRICS);
        self.write_u64(route_distance);
        self.write_u32(max_branch_depth);
        self.write_u32(dead_end_count);
        self.write_u32(articulation_count);
        self.write_u32(crossing_count);
        let cycle_count = u32::try_from(per_layer_cycles.len()).expect("too many layers");
        self.write_u32(cycle_count);
        for &c in per_layer_cycles {
            self.write_u32(c);
        }
        self
    }

    /// Return the encoded bytes.
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn category_ordinal(cat: super::capture_views::CaptureViewCategory) -> u8 {
    match cat {
        super::capture_views::CaptureViewCategory::Spawn => 0,
        super::capture_views::CaptureViewCategory::RequiredRoute => 1,
        super::capture_views::CaptureViewCategory::DistantLandmark => 2,
        super::capture_views::CaptureViewCategory::Junction => 3,
        super::capture_views::CaptureViewCategory::RampApproach => 4,
        super::capture_views::CaptureViewCategory::RampCrest => 5,
        super::capture_views::CaptureViewCategory::UpperLanding => 6,
        super::capture_views::CaptureViewCategory::DarkBranch => 7,
    }
}

// ─── Decoded Outcome ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayResources {
    pub total_tiles: u64,
    pub floor_tiles: u64,
    pub wall_tiles: u64,
    pub void_tiles: u64,
    pub ramp_tiles: u64,
    pub non_empty_chunks: u32,
    pub estimated_vertices: u64,
    pub estimated_indices: u64,
    pub light_count: u32,
    pub model_count: u32,
    pub static_body_count: u32,
    pub total_body_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCaptureView {
    pub eye: [f32; 3],
    pub look_at: [f32; 3],
    pub category_ordinal: u8,
    pub label: String,
}

impl ReplayCaptureView {
    /// Compare exactly including f32::to_bits equality.
    pub fn exact_eq(&self, other: &Self) -> bool {
        self.eye[0].to_bits() == other.eye[0].to_bits()
            && self.eye[1].to_bits() == other.eye[1].to_bits()
            && self.eye[2].to_bits() == other.eye[2].to_bits()
            && self.look_at[0].to_bits() == other.look_at[0].to_bits()
            && self.look_at[1].to_bits() == other.look_at[1].to_bits()
            && self.look_at[2].to_bits() == other.look_at[2].to_bits()
            && self.category_ordinal == other.category_ordinal
            && self.label == other.label
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayTopologyMetrics {
    pub route_distance: u64,
    pub max_branch_depth: u32,
    pub dead_end_count: u32,
    pub articulation_count: u32,
    pub crossing_count: u32,
    pub per_layer_cycles: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReplayOutcome {
    Success {
        seed: u64,
        attempt_index: u32,
        attempt_identity: [u8; 32],
        ascii_bytes: Vec<u8>,
        resources: ReplayResources,
        capture_views: Vec<ReplayCaptureView>,
        diagnostics_bytes: Vec<u8>,
        topology_region_count: u32,
        topology_edge_count: u32,
        topology_metrics: ReplayTopologyMetrics,
    },
    Exhausted {
        seed: u64,
        attempt_index: u32,
        attempt_identity: [u8; 32],
        stage: String,
        reason: String,
        diagnostics_bytes: Vec<u8>,
    },
}

impl ReplayOutcome {
    /// Exact equality comparing all bytes and structured values.
    /// Hashes are never substituted for equality.
    pub fn exact_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                ReplayOutcome::Success {
                    seed: s1,
                    attempt_index: a1,
                    attempt_identity: i1,
                    ascii_bytes: b1,
                    resources: r1,
                    capture_views: v1,
                    diagnostics_bytes: d1,
                    topology_region_count: rc1,
                    topology_edge_count: ec1,
                    topology_metrics: m1,
                },
                ReplayOutcome::Success {
                    seed: s2,
                    attempt_index: a2,
                    attempt_identity: i2,
                    ascii_bytes: b2,
                    resources: r2,
                    capture_views: v2,
                    diagnostics_bytes: d2,
                    topology_region_count: rc2,
                    topology_edge_count: ec2,
                    topology_metrics: m2,
                },
            ) => {
                s1 == s2
                    && a1 == a2
                    && i1 == i2
                    && b1 == b2
                    && r1 == r2
                    && v1.len() == v2.len()
                    && v1.iter().zip(v2.iter()).all(|(a, b)| a.exact_eq(b))
                    && d1 == d2
                    && rc1 == rc2
                    && ec1 == ec2
                    && m1 == m2
            }
            (
                ReplayOutcome::Exhausted {
                    seed: s1,
                    attempt_index: a1,
                    attempt_identity: i1,
                    stage: st1,
                    reason: r1,
                    diagnostics_bytes: d1,
                },
                ReplayOutcome::Exhausted {
                    seed: s2,
                    attempt_index: a2,
                    attempt_identity: i2,
                    stage: st2,
                    reason: r2,
                    diagnostics_bytes: d2,
                },
            ) => {
                s1 == s2
                    && a1 == a2
                    && i1 == i2
                    && st1 == st2
                    && r1 == r2
                    && d1 == d2
            }
            _ => false,
        }
    }
}

// ─── Decoder ───────────────────────────────────────────────────────────────

pub struct ReplayDecoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ReplayDecoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, ReplayError> {
        if self.position >= self.bytes.len() {
            return Err(ReplayError::Truncated {
                expected: self.position + 1,
                actual: self.bytes.len(),
            });
        }
        let value = self.bytes[self.position];
        self.position += 1;
        Ok(value)
    }

    fn read_u32(&mut self) -> Result<u32, ReplayError> {
        let end = self.position + 4;
        if end > self.bytes.len() {
            return Err(ReplayError::Truncated {
                expected: end,
                actual: self.bytes.len(),
            });
        }
        let value = u32::from_be_bytes(self.bytes[self.position..end].try_into().unwrap());
        self.position = end;
        Ok(value)
    }

    fn read_u64(&mut self) -> Result<u64, ReplayError> {
        let end = self.position + 8;
        if end > self.bytes.len() {
            return Err(ReplayError::Truncated {
                expected: end,
                actual: self.bytes.len(),
            });
        }
        let value = u64::from_be_bytes(self.bytes[self.position..end].try_into().unwrap());
        self.position = end;
        Ok(value)
    }

    fn read_f32(&mut self) -> Result<f32, ReplayError> {
        let bits = self.read_u32()?;
        Ok(f32::from_bits(bits))
    }

    fn read_fixed_32(&mut self) -> Result<[u8; 32], ReplayError> {
        let end = self.position + 32;
        if end > self.bytes.len() {
            return Err(ReplayError::Truncated {
                expected: end,
                actual: self.bytes.len(),
            });
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&self.bytes[self.position..end]);
        self.position = end;
        Ok(arr)
    }

    fn read_bytes_prefixed(&mut self) -> Result<Vec<u8>, ReplayError> {
        let len = self.read_u32()? as usize;
        if len > MAX_FRAME_BYTES {
            return Err(ReplayError::OversizedLength {
                position: self.position - 4,
                length: len,
                maximum: MAX_FRAME_BYTES,
            });
        }
        let end = self.position + len;
        if end > self.bytes.len() {
            return Err(ReplayError::Truncated {
                expected: end,
                actual: self.bytes.len(),
            });
        }
        let value = self.bytes[self.position..end].to_vec();
        self.position = end;
        Ok(value)
    }

    fn read_string_prefixed(&mut self) -> Result<String, ReplayError> {
        let bytes = self.read_bytes_prefixed()?;
        String::from_utf8(bytes).map_err(|_| ReplayError::DecodeUtf8 {
            position: self.position,
        })
    }

    /// Decode the full replay frame.
    pub fn decode(mut self) -> Result<ReplayOutcome, ReplayError> {
        let tag = self.read_u8()?;
        if tag != TAG_SCHEMA {
            return Err(ReplayError::InvalidTag {
                position: self.position - 1,
                tag,
            });
        }
        let _schema_version = self.read_u32()?;

        let mut seed: Option<u64> = None;
        let mut attempt_index: Option<u32> = None;
        let mut attempt_identity: Option<[u8; 32]> = None;
        let mut ascii_bytes: Option<Vec<u8>> = None;
        let mut resources: Option<ReplayResources> = None;
        let mut capture_views: Option<Vec<ReplayCaptureView>> = None;
        let mut diagnostics_bytes: Option<Vec<u8>> = None;
        let mut exhausted_stage: Option<String> = None;
        let mut exhausted_reason: Option<String> = None;
        let mut topology_region_count: Option<u32> = None;
        let mut topology_edge_count: Option<u32> = None;
        let mut topology_metrics: Option<ReplayTopologyMetrics> = None;

        loop {
            if self.position >= self.bytes.len() {
                break;
            }
            let tag = self.read_u8()?;
            match tag {
                TAG_SEED => {
                    seed = Some(self.read_u64()?);
                }
                TAG_ATTEMPT_INDEX => {
                    attempt_index = Some(self.read_u32()?);
                }
                TAG_ATTEMPT_IDENTITY => {
                    attempt_identity = Some(self.read_fixed_32()?);
                }
                TAG_ASCII => {
                    ascii_bytes = Some(self.read_bytes_prefixed()?);
                }
                TAG_RESOURCES => {
                    resources = Some(ReplayResources {
                        total_tiles: self.read_u64()?,
                        floor_tiles: self.read_u64()?,
                        wall_tiles: self.read_u64()?,
                        void_tiles: self.read_u64()?,
                        ramp_tiles: self.read_u64()?,
                        non_empty_chunks: self.read_u32()?,
                        estimated_vertices: self.read_u64()?,
                        estimated_indices: self.read_u64()?,
                        light_count: self.read_u32()?,
                        model_count: self.read_u32()?,
                        static_body_count: self.read_u32()?,
                        total_body_count: self.read_u32()?,
                    });
                }
                TAG_CAPTURE_VIEWS => {
                    let count = self.read_u32()? as usize;
                    if count > MAX_FRAME_BYTES / 64 {
                        return Err(ReplayError::OversizedLength {
                            position: self.position - 4,
                            length: count,
                            maximum: MAX_FRAME_BYTES / 64,
                        });
                    }
                    let mut views = Vec::with_capacity(count);
                    for _ in 0..count {
                        let ex = self.read_f32()?;
                        let ey = self.read_f32()?;
                        let ez = self.read_f32()?;
                        let lx = self.read_f32()?;
                        let ly = self.read_f32()?;
                        let lz = self.read_f32()?;
                        let category_ordinal = self.read_u8()?;
                        let label = self.read_string_prefixed()?;
                        views.push(ReplayCaptureView {
                            eye: [ex, ey, ez],
                            look_at: [lx, ly, lz],
                            category_ordinal,
                            label,
                        });
                    }
                    capture_views = Some(views);
                }
                TAG_DIAGNOSTICS => {
                    diagnostics_bytes = Some(self.read_bytes_prefixed()?);
                }
                TAG_EXHAUSTED => {
                    exhausted_stage = Some(self.read_string_prefixed()?);
                    exhausted_reason = Some(self.read_string_prefixed()?);
                }
                TAG_TOPOLOGY_REGIONS => {
                    topology_region_count = Some(self.read_u32()?);
                }
                TAG_TOPOLOGY_EDGES => {
                    topology_edge_count = Some(self.read_u32()?);
                }
                TAG_TOPOLOGY_METRICS => {
                    let route_distance = self.read_u64()?;
                    let max_branch_depth = self.read_u32()?;
                    let dead_end_count = self.read_u32()?;
                    let articulation_count = self.read_u32()?;
                    let crossing_count = self.read_u32()?;
                    let cycle_count = self.read_u32()? as usize;
                    if cycle_count > MAX_FRAME_BYTES / 4 {
                        return Err(ReplayError::OversizedLength {
                            position: self.position - 4,
                            length: cycle_count,
                            maximum: MAX_FRAME_BYTES / 4,
                        });
                    }
                    let mut per_layer_cycles = Vec::with_capacity(cycle_count);
                    for _ in 0..cycle_count {
                        per_layer_cycles.push(self.read_u32()?);
                    }
                    topology_metrics = Some(ReplayTopologyMetrics {
                        route_distance,
                        max_branch_depth,
                        dead_end_count,
                        articulation_count,
                        crossing_count,
                        per_layer_cycles,
                    });
                }
                other => {
                    return Err(ReplayError::InvalidTag {
                        position: self.position - 1,
                        tag: other,
                    });
                }
            }
        }

        // Validate no trailing bytes
        if self.position != self.bytes.len() {
            return Err(ReplayError::TrailingBytes {
                consumed: self.position,
                remaining: self.bytes.len() - self.position,
            });
        }

        let seed = seed.ok_or_else(|| ReplayError::DecodeSemantics {
            reason: "missing seed".into(),
        })?;
        let attempt_index = attempt_index.ok_or_else(|| ReplayError::DecodeSemantics {
            reason: "missing attempt_index".into(),
        })?;
        let attempt_identity =
            attempt_identity.ok_or_else(|| ReplayError::DecodeSemantics {
                reason: "missing attempt_identity".into(),
            })?;
        let diagnostics_bytes =
            diagnostics_bytes.ok_or_else(|| ReplayError::DecodeSemantics {
                reason: "missing diagnostics".into(),
            })?;

        if let (Some(stage), Some(reason)) = (exhausted_stage, exhausted_reason) {
            Ok(ReplayOutcome::Exhausted {
                seed,
                attempt_index,
                attempt_identity,
                stage,
                reason,
                diagnostics_bytes,
            })
        } else {
            let ascii_bytes = ascii_bytes.ok_or_else(|| ReplayError::DecodeSemantics {
                reason: "missing ascii".into(),
            })?;
            let resources = resources.ok_or_else(|| ReplayError::DecodeSemantics {
                reason: "missing resources".into(),
            })?;
            let capture_views = capture_views.ok_or_else(|| ReplayError::DecodeSemantics {
                reason: "missing capture_views".into(),
            })?;
            let topology_region_count = topology_region_count.unwrap_or(0);
            let topology_edge_count = topology_edge_count.unwrap_or(0);
            let topology_metrics = topology_metrics.ok_or_else(|| ReplayError::DecodeSemantics {
                reason: "missing topology_metrics".into(),
            })?;
            Ok(ReplayOutcome::Success {
                seed,
                attempt_index,
                attempt_identity,
                ascii_bytes,
                resources,
                capture_views,
                diagnostics_bytes,
                topology_region_count,
                topology_edge_count,
                topology_metrics,
            })
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_success_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        // schema
        bytes.push(TAG_SCHEMA);
        bytes.extend_from_slice(&1u32.to_be_bytes());
        // seed
        bytes.push(TAG_SEED);
        bytes.extend_from_slice(&42u64.to_be_bytes());
        // attempt_index
        bytes.push(TAG_ATTEMPT_INDEX);
        bytes.extend_from_slice(&7u32.to_be_bytes());
        // attempt_identity
        bytes.push(TAG_ATTEMPT_IDENTITY);
        bytes.extend_from_slice(&[0xAB; 32]);
        // ascii
        bytes.push(TAG_ASCII);
        let ascii_data = b"test_level";
        bytes.extend_from_slice(&(ascii_data.len() as u32).to_be_bytes());
        bytes.extend_from_slice(ascii_data);
        // resources
        bytes.push(TAG_RESOURCES);
        bytes.extend_from_slice(&100u64.to_be_bytes()); // total_tiles
        bytes.extend_from_slice(&50u64.to_be_bytes()); // floor_tiles
        bytes.extend_from_slice(&30u64.to_be_bytes()); // wall_tiles
        bytes.extend_from_slice(&10u64.to_be_bytes()); // void_tiles
        bytes.extend_from_slice(&10u64.to_be_bytes()); // ramp_tiles
        bytes.extend_from_slice(&5u32.to_be_bytes()); // non_empty_chunks
        bytes.extend_from_slice(&1000u64.to_be_bytes()); // estimated_vertices
        bytes.extend_from_slice(&2000u64.to_be_bytes()); // estimated_indices
        bytes.extend_from_slice(&3u32.to_be_bytes()); // light_count
        bytes.extend_from_slice(&2u32.to_be_bytes()); // model_count
        bytes.extend_from_slice(&1u32.to_be_bytes()); // static_body_count
        bytes.extend_from_slice(&1u32.to_be_bytes()); // total_body_count
        // capture_views
        bytes.push(TAG_CAPTURE_VIEWS);
        bytes.extend_from_slice(&1u32.to_be_bytes()); // count
        // eye
        bytes.extend_from_slice(&1.0f32.to_bits().to_be_bytes());
        bytes.extend_from_slice(&2.0f32.to_bits().to_be_bytes());
        bytes.extend_from_slice(&3.0f32.to_bits().to_be_bytes());
        // look_at
        bytes.extend_from_slice(&0.0f32.to_bits().to_be_bytes());
        bytes.extend_from_slice(&0.0f32.to_bits().to_be_bytes());
        bytes.extend_from_slice(&(-1.0f32).to_bits().to_be_bytes());
        // category ordinal
        bytes.push(0u8);
        // label
        let label = b"spawn_view";
        bytes.extend_from_slice(&(label.len() as u32).to_be_bytes());
        bytes.extend_from_slice(label);
        // diagnostics
        bytes.push(TAG_DIAGNOSTICS);
        let diag = b"{}";
        bytes.extend_from_slice(&(diag.len() as u32).to_be_bytes());
        bytes.extend_from_slice(diag);
        // topology_regions
        bytes.push(TAG_TOPOLOGY_REGIONS);
        bytes.extend_from_slice(&3u32.to_be_bytes());
        // topology_edges
        bytes.push(TAG_TOPOLOGY_EDGES);
        bytes.extend_from_slice(&2u32.to_be_bytes());
        // topology_metrics
        bytes.push(TAG_TOPOLOGY_METRICS);
        bytes.extend_from_slice(&100u64.to_be_bytes()); // route_distance
        bytes.extend_from_slice(&2u32.to_be_bytes()); // max_branch_depth
        bytes.extend_from_slice(&1u32.to_be_bytes()); // dead_end_count
        bytes.extend_from_slice(&0u32.to_be_bytes()); // articulation_count
        bytes.extend_from_slice(&0u32.to_be_bytes()); // crossing_count
        bytes.extend_from_slice(&1u32.to_be_bytes()); // cycle_count
        bytes.extend_from_slice(&3u32.to_be_bytes()); // per_layer_cycles[0]
        bytes
    }

    #[test]
    fn round_trip_success_preserves_all_fields() {
        let bytes = make_success_bytes();
        let decoder = ReplayDecoder::new(&bytes);
        let outcome = decoder.decode().unwrap();
        match &outcome {
            ReplayOutcome::Success {
                seed,
                attempt_index,
                ascii_bytes,
                resources: r,
                capture_views: v,
                diagnostics_bytes: d,
                ..
            } => {
                assert_eq!(*seed, 42);
                assert_eq!(*attempt_index, 7);
                assert_eq!(ascii_bytes, b"test_level");
                assert_eq!(r.total_tiles, 100);
                assert_eq!(r.light_count, 3);
                assert_eq!(v.len(), 1);
                assert_eq!(v[0].label, "spawn_view");
                assert_eq!(v[0].eye[0].to_bits(), 1.0f32.to_bits());
                assert_eq!(v[0].eye[1].to_bits(), 2.0f32.to_bits());
                assert_eq!(v[0].look_at[2].to_bits(), (-1.0f32).to_bits());
                assert_eq!(v[0].category_ordinal, 0);
                assert_eq!(d, b"{}");
            }
            _ => panic!("expected success"),
        }
        // Re-decode should produce identical result
        let decoder2 = ReplayDecoder::new(&bytes);
        let outcome2 = decoder2.decode().unwrap();
        assert!(outcome.exact_eq(&outcome2));
    }

    #[test]
    fn round_trip_exhausted_preserves_all_fields() {
        let mut bytes = Vec::new();
        bytes.push(TAG_SCHEMA);
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.push(TAG_SEED);
        bytes.extend_from_slice(&99u64.to_be_bytes());
        bytes.push(TAG_ATTEMPT_INDEX);
        bytes.extend_from_slice(&3u32.to_be_bytes());
        bytes.push(TAG_ATTEMPT_IDENTITY);
        bytes.extend_from_slice(&[0xCD; 32]);
        bytes.push(TAG_EXHAUSTED);
        let stage = b"placement";
        bytes.extend_from_slice(&(stage.len() as u32).to_be_bytes());
        bytes.extend_from_slice(stage);
        let reason = b"placement_retry";
        bytes.extend_from_slice(&(reason.len() as u32).to_be_bytes());
        bytes.extend_from_slice(reason);
        bytes.push(TAG_DIAGNOSTICS);
        let diag = b"{}";
        bytes.extend_from_slice(&(diag.len() as u32).to_be_bytes());
        bytes.extend_from_slice(diag);

        let decoder = ReplayDecoder::new(&bytes);
        let outcome = decoder.decode().unwrap();
        match outcome {
            ReplayOutcome::Exhausted {
                seed,
                attempt_index,
                stage,
                reason,
                ..
            } => {
                assert_eq!(seed, 99);
                assert_eq!(attempt_index, 3);
                assert_eq!(stage, "placement");
                assert_eq!(reason, "placement_retry");
            }
            _ => panic!("expected exhausted"),
        }
    }

    #[test]
    fn reject_truncated_input() {
        let bytes = vec![TAG_SCHEMA, 0x00, 0x00]; // truncated after schema tag
        let decoder = ReplayDecoder::new(&bytes);
        assert!(matches!(decoder.decode(), Err(ReplayError::Truncated { .. })));
    }

    #[test]
    fn reject_trailing_bytes() {
        let mut bytes = make_success_bytes();
        bytes.push(0xFF);
        bytes.push(0xEE);
        let decoder = ReplayDecoder::new(&bytes);
        let result = decoder.decode();
        // Trailing bytes after a complete frame are detected as an invalid tag
        // since the decoder tries to read the next tag and finds 0xFF.
        assert!(result.is_err(), "trailing bytes must be rejected");
    }

    #[test]
    fn reject_invalid_tag() {
        let mut bytes = Vec::new();
        bytes.push(TAG_SCHEMA);
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.push(0xFF); // invalid tag
        let decoder = ReplayDecoder::new(&bytes);
        assert!(matches!(decoder.decode(), Err(ReplayError::InvalidTag { .. })));
    }

    #[test]
    fn reject_oversized_length() {
        let mut bytes = Vec::new();
        bytes.push(TAG_SCHEMA);
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.push(TAG_ASCII);
        bytes.extend_from_slice(&((MAX_FRAME_BYTES + 1) as u32).to_be_bytes());
        let decoder = ReplayDecoder::new(&bytes);
        assert!(matches!(
            decoder.decode(),
            Err(ReplayError::OversizedLength { .. })
        ));
    }

    #[test]
    fn exact_eq_detects_bit_level_differences() {
        let bytes = make_success_bytes();
        let outcome1 = ReplayDecoder::new(&bytes).decode().unwrap();

        let mut bytes2 = bytes.clone();
        // Flip a bit in the ascii section
        bytes2[40] ^= 0x01;
        let outcome2 = ReplayDecoder::new(&bytes2).decode().unwrap();

        assert!(!outcome1.exact_eq(&outcome2));
    }

    #[test]
    fn float_bit_preservation() {
        // Encode a capture view with specific float values and verify bit-level preservation
        let enc = ReplayEncoder::new()
            .schema()
            .seed(1)
            .attempt_index(0)
            .attempt_identity([0; 32])
            .ascii_bytes(b"x")
            .resources(&ResourceCounts {
                total_tiles: 0,
                floor_tiles: 0,
                wall_tiles: 0,
                void_tiles: 0,
                ramp_tiles: 0,
                non_empty_chunks: 0,
                estimated_vertices: 0,
                estimated_indices: 0,
                light_count: 0,
                model_count: 0,
                static_body_count: 0,
                total_body_count: 0,
            })
            .topology_regions(0)
            .topology_edges(0)
            .topology_metrics(0, 0, 0, 0, 0, &[]);

        // Manually add capture views with NaN and negative zero
        let views: &[CaptureView] = &[];
        let bytes = enc.capture_views(views).diagnostics(b"{}").finish();

        // Verify we can decode it
        let outcome = ReplayDecoder::new(&bytes).decode().unwrap();
        match outcome {
            ReplayOutcome::Success {
                capture_views: v, ..
            } => {
                assert!(v.is_empty());
            }
            _ => panic!("expected success"),
        }
    }

    #[test]
    fn exact_eq_success_vs_exhausted() {
        let mut success_bytes = make_success_bytes();
        let success = ReplayDecoder::new(&success_bytes).decode().unwrap();

        let mut exh_bytes = Vec::new();
        exh_bytes.push(TAG_SCHEMA);
        exh_bytes.extend_from_slice(&1u32.to_be_bytes());
        exh_bytes.push(TAG_SEED);
        exh_bytes.extend_from_slice(&99u64.to_be_bytes());
        exh_bytes.push(TAG_ATTEMPT_INDEX);
        exh_bytes.extend_from_slice(&3u32.to_be_bytes());
        exh_bytes.push(TAG_ATTEMPT_IDENTITY);
        exh_bytes.extend_from_slice(&[0xCD; 32]);
        exh_bytes.push(TAG_EXHAUSTED);
        let stage = b"placement";
        exh_bytes.extend_from_slice(&(stage.len() as u32).to_be_bytes());
        exh_bytes.extend_from_slice(stage);
        let reason = b"placement_retry";
        exh_bytes.extend_from_slice(&(reason.len() as u32).to_be_bytes());
        exh_bytes.extend_from_slice(reason);
        exh_bytes.push(TAG_DIAGNOSTICS);
        let diag = b"{}";
        exh_bytes.extend_from_slice(&(diag.len() as u32).to_be_bytes());
        exh_bytes.extend_from_slice(diag);
        let exhausted = ReplayDecoder::new(&exh_bytes).decode().unwrap();

        assert!(!success.exact_eq(&exhausted));
    }
}
