//! Immutable Richness V1 request metadata.
//!
//! This is the request-bound metadata contract available before geometry,
//! assets, compiler evidence, or publication exist. It records exact canonical
//! authored and resolved request bytes plus their independently framed
//! identities; later phases add their own immutable result records rather than
//! mutating this provenance snapshot.

use super::request::ResolvedRichnessRequestV1;

/// Frozen schema tag for request provenance metadata.
pub const RICHNESS_METADATA_SCHEMA_VERSION: &str = "richness-v1";

/// Immutable metadata snapshot for one validated Richness request.
///
/// The canonical byte vectors retain inherited-versus-explicit provenance and
/// the resolved controls, while the hashes give downstream artifacts stable
/// compact references without involving baseline V3 metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RichnessMetadataV1 {
    schema_version: &'static str,
    request_identity: [u8; 32],
    resolved_request_identity: [u8; 32],
    canonical_request: Vec<u8>,
    canonical_resolved_request: Vec<u8>,
}

impl RichnessMetadataV1 {
    /// Capture immutable provenance from an already validated request.
    pub fn from_resolved(request: &ResolvedRichnessRequestV1) -> Self {
        Self {
            schema_version: RICHNESS_METADATA_SCHEMA_VERSION,
            request_identity: request.provenance().identity_hash(),
            resolved_request_identity: request.identity_hash(),
            canonical_request: request.provenance().to_canonical_bytes(),
            canonical_resolved_request: request.to_canonical_bytes(),
        }
    }

    /// Frozen schema tag for this provenance record.
    pub fn schema_version(&self) -> &'static str {
        self.schema_version
    }

    /// Identity of the authored request, including explicit-state markers.
    pub fn request_identity(&self) -> [u8; 32] {
        self.request_identity
    }

    /// Identity of the resolved request, including resolved values and sources.
    pub fn resolved_request_identity(&self) -> [u8; 32] {
        self.resolved_request_identity
    }

    /// Exact canonical authored request bytes.
    pub fn canonical_request(&self) -> &[u8] {
        &self.canonical_request
    }

    /// Exact canonical resolved request bytes.
    pub fn canonical_resolved_request(&self) -> &[u8] {
        &self.canonical_resolved_request
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::richness::request::{
        ResolvedRichnessRequestV1, RichnessDocumentV1, RichnessPreset, RichnessTheme,
    };

    #[test]
    fn metadata_is_an_immutable_request_provenance_snapshot() {
        let request = ResolvedRichnessRequestV1::resolve(
            RichnessDocumentV1::new(42, 2048, RichnessPreset::Sparse, RichnessTheme::Ancient)
                .unwrap(),
        )
        .unwrap();
        let metadata = RichnessMetadataV1::from_resolved(&request);

        assert_eq!(metadata.schema_version(), RICHNESS_METADATA_SCHEMA_VERSION);
        assert_eq!(
            metadata.request_identity(),
            request.provenance().identity_hash()
        );
        assert_eq!(
            metadata.resolved_request_identity(),
            request.identity_hash()
        );
        assert_eq!(
            metadata.canonical_request(),
            request.provenance().to_canonical_bytes()
        );
        assert_eq!(
            metadata.canonical_resolved_request(),
            request.to_canonical_bytes()
        );
    }
}
