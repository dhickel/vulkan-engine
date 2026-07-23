//! BSP cache identity: deterministic fingerprint of all inputs that change
//! extracted output.
//!
//! The cache identity is computed from every setting that changes the
//! extracted BSP DTOs. Identical inputs produce identical output; changing
//! any input produces a different cache key.

/// Components that form a BSP cache identity.
///
/// The cache identity includes every setting that changes extracted output:
/// BSP content, dialect, scale, palette, companions, texture roots,
/// replacements, light calibration, atlas policy, collision policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheIdentity {
    /// SHA-256 of raw .bsp bytes.
    pub bsp_content_hash: [u8; 32],
    /// Dialect / profile tag ("q1-portable-ericw" or exact variant).
    pub dialect_profile_tag: String,
    /// Resolved scale as canonical f32 bytes (little-endian).
    pub bsp_scale: [u8; 4],
    /// SHA-256 of palette bytes (empty if no palette).
    pub palette_content_hash: [u8; 32],
    /// Sorted list of (companion_kind, content_hash) pairs.
    pub companion_identities: Vec<CompanionId>,
    /// Sorted canonical paths for texture/WAD resolution roots.
    pub texture_resolution_roots: Vec<String>,
    /// Sorted (texture_name, resolved_path_hash) replacement mappings.
    pub replacement_mappings: Vec<ReplacementMapping>,
    /// Light calibration parameters as canonical bytes.
    pub light_calibration: LightCalibration,
    /// Atlas page size, padding, style count.
    pub atlas_policy: AtlasPolicy,
    /// Hull indices and convex decomposition limits.
    pub collision_policy: CollisionPolicy,
}

/// Identity of a companion file.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompanionId {
    /// Companion kind (e.g., "lit", "palette", "wad").
    pub kind: String,
    /// SHA-256 of companion file content.
    pub content_hash: [u8; 32],
}

/// A texture replacement mapping.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReplacementMapping {
    pub texture_name: String,
    pub resolved_path_hash: [u8; 32],
}

/// Light calibration parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LightCalibration {
    pub intensity_scale: [u8; 4], // f32 LE
    pub overbright: [u8; 4],      // f32 LE
}

impl Default for LightCalibration {
    fn default() -> Self {
        Self {
            intensity_scale: 2.0f32.to_le_bytes(),
            overbright: 2.0f32.to_le_bytes(),
        }
    }
}

/// Atlas page policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtlasPolicy {
    pub page_size: u32,
    pub padding: u32,
    pub style_count: u32,
}

impl Default for AtlasPolicy {
    fn default() -> Self {
        Self {
            page_size: 2048,
            padding: 2,
            style_count: 4,
        }
    }
}

/// Collision policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CollisionPolicy {
    pub hull_indices: Vec<u32>,
    pub convex_decomposition_limit: u32,
}

impl Default for CollisionPolicy {
    fn default() -> Self {
        Self {
            hull_indices: vec![0, 1, 2],
            convex_decomposition_limit: 16,
        }
    }
}

impl CacheIdentity {
    /// Compute a cache identity from BSP extraction inputs.
    ///
    /// This fingerprint is used to separate caches for different mounts.
    pub fn compute(
        bsp_content_hash: [u8; 32],
        dialect_profile_tag: impl Into<String>,
        bsp_scale: f32,
        palette_content_hash: [u8; 32],
        companion_identities: Vec<CompanionId>,
        texture_resolution_roots: Vec<String>,
        replacement_mappings: Vec<ReplacementMapping>,
        light_calibration: LightCalibration,
        atlas_policy: AtlasPolicy,
        collision_policy: CollisionPolicy,
    ) -> Self {
        Self {
            bsp_content_hash,
            dialect_profile_tag: dialect_profile_tag.into(),
            bsp_scale: bsp_scale.to_le_bytes(),
            palette_content_hash,
            companion_identities,
            texture_resolution_roots,
            replacement_mappings,
            light_calibration,
            atlas_policy,
            collision_policy,
        }
    }

    /// Produce a compact key string suitable for file-system safe cache directory naming.
    pub fn to_key_string(&self) -> String {
        // Hex-encode the BSP content hash as the primary discriminator,
        // appending a short hash of the remaining components.
        let primary = hex_encode(&self.bsp_content_hash);

        // Hash the remaining fields for a compact suffix
        let mut suffix_input = Vec::new();
        suffix_input.extend_from_slice(self.dialect_profile_tag.as_bytes());
        suffix_input.extend_from_slice(&self.bsp_scale);
        suffix_input.extend_from_slice(&self.palette_content_hash);
        for comp in &self.companion_identities {
            suffix_input.extend_from_slice(comp.kind.as_bytes());
            suffix_input.extend_from_slice(&comp.content_hash);
        }
        for root in &self.texture_resolution_roots {
            suffix_input.extend_from_slice(root.as_bytes());
        }
        for mapping in &self.replacement_mappings {
            suffix_input.extend_from_slice(mapping.texture_name.as_bytes());
            suffix_input.extend_from_slice(&mapping.resolved_path_hash);
        }
        suffix_input.extend_from_slice(&self.light_calibration.intensity_scale);
        suffix_input.extend_from_slice(&self.light_calibration.overbright);
        suffix_input.extend_from_slice(&self.atlas_policy.page_size.to_le_bytes());
        suffix_input.extend_from_slice(&self.atlas_policy.padding.to_le_bytes());
        suffix_input.extend_from_slice(&self.atlas_policy.style_count.to_le_bytes());
        for hull_idx in &self.collision_policy.hull_indices {
            suffix_input.extend_from_slice(&hull_idx.to_le_bytes());
        }
        suffix_input.extend_from_slice(
            &self
                .collision_policy
                .convex_decomposition_limit
                .to_le_bytes(),
        );

        let suffix_hash = compute_identity_hash(&suffix_input);
        let suffix_hex = hex_encode(&suffix_hash);
        format!("{}-{}", &primary[..16], &suffix_hex[..16])
    }
}

/// Compute a 32-byte identity hash from arbitrary input bytes using a
/// simple multi-lane hash (same algorithm as `bsp::compute_content_hash`).
fn compute_identity_hash(data: &[u8]) -> [u8; 32] {
    let mut lanes = [
        0xcbf2_9ce4_8422_2325u64,
        0x9e37_79b9_7f4a_7c15u64,
        0x94d0_49bb_1331_11ebu64,
        0x2545_f491_4f6c_dd1du64,
    ];
    for (i, &byte) in data.iter().enumerate() {
        let lane = i & 3;
        lanes[lane] ^= byte as u64;
        lanes[lane] = lanes[lane].wrapping_mul(0x100_0000_01b3);
        lanes[lane] ^= (i as u64).rotate_left((lane as u32) + 1);
    }
    let mut arr = [0u8; 32];
    for (i, lane) in lanes.iter().enumerate() {
        arr[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
    }
    arr
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_identity_key_is_deterministic() {
        let identity = CacheIdentity::compute(
            [0xabu8; 32],
            "q1-portable-ericw",
            0.0254,
            [0; 32],
            vec![],
            vec![],
            vec![],
            LightCalibration::default(),
            AtlasPolicy::default(),
            CollisionPolicy::default(),
        );

        let key1 = identity.to_key_string();
        let key2 = identity.to_key_string();
        assert_eq!(key1, key2);
    }

    #[test]
    fn cache_identity_differs_by_scale() {
        let id1 = CacheIdentity::compute(
            [0xabu8; 32],
            "q1-portable-ericw",
            0.0254,
            [0; 32],
            vec![],
            vec![],
            vec![],
            LightCalibration::default(),
            AtlasPolicy::default(),
            CollisionPolicy::default(),
        );

        let id2 = CacheIdentity::compute(
            [0xabu8; 32],
            "q1-portable-ericw",
            0.5,
            [0; 32],
            vec![],
            vec![],
            vec![],
            LightCalibration::default(),
            AtlasPolicy::default(),
            CollisionPolicy::default(),
        );

        assert_ne!(id1.to_key_string(), id2.to_key_string());
    }

    #[test]
    fn cache_identity_differs_by_hash() {
        let id1 = CacheIdentity::compute(
            [0xabu8; 32],
            "q1-portable-ericw",
            0.0254,
            [0; 32],
            vec![],
            vec![],
            vec![],
            LightCalibration::default(),
            AtlasPolicy::default(),
            CollisionPolicy::default(),
        );

        let id2 = CacheIdentity::compute(
            [0xcd; 32],
            "q1-portable-ericw",
            0.0254,
            [0; 32],
            vec![],
            vec![],
            vec![],
            LightCalibration::default(),
            AtlasPolicy::default(),
            CollisionPolicy::default(),
        );

        assert_ne!(id1.to_key_string(), id2.to_key_string());
    }
}
