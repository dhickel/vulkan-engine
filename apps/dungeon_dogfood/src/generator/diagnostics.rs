use serde::Serialize;

use super::config::NormalizedGeneratorConfig;
use super::determinism::{
    lowercase_hex, AttemptIdentity, GeneratorIdentity, GENERATOR_VERSION, RNG_VERSION,
};
use super::error::{ErrorStage, GeneratorError};

const DIAGNOSTIC_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct AttemptDiagnostic {
    attempt_index: u32,
    attempt_identity: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct FoundationFailureDiagnostic {
    stage: String,
    reason: String,
}

impl FoundationFailureDiagnostic {
    pub(super) fn from_error(error: &GeneratorError) -> Self {
        Self {
            stage: error.stage().code().to_owned(),
            reason: error.reason_code().to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct GeneratorDiagnostics {
    schema_version: u32,
    generator_version: u32,
    rng_version: u32,
    normalized_configuration: NormalizedGeneratorConfig,
    configuration_identity: String,
    catalog_identity: String,
    generator_identity: String,
    seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    attempt: Option<AttemptDiagnostic>,
    #[serde(skip_serializing_if = "Option::is_none")]
    foundation_failure: Option<FoundationFailureDiagnostic>,
}

impl GeneratorDiagnostics {
    pub(super) fn new(
        config: &NormalizedGeneratorConfig,
        catalog_digest: [u8; 32],
        seed: u64,
    ) -> Self {
        let generator_identity = GeneratorIdentity::new(config, catalog_digest, seed);
        Self {
            schema_version: DIAGNOSTIC_SCHEMA_VERSION,
            generator_version: GENERATOR_VERSION,
            rng_version: RNG_VERSION,
            normalized_configuration: config.clone(),
            configuration_identity: config.canonical_hash(),
            catalog_identity: lowercase_hex(&catalog_digest),
            generator_identity: generator_identity.hex(),
            seed,
            attempt: None,
            foundation_failure: None,
        }
    }

    pub(super) fn with_attempt(mut self, attempt: AttemptIdentity) -> Self {
        self.attempt = Some(AttemptDiagnostic {
            attempt_index: attempt.index(),
            attempt_identity: attempt.hex(),
        });
        self
    }

    pub(super) fn with_failure(mut self, error: &GeneratorError) -> Self {
        self.foundation_failure = Some(FoundationFailureDiagnostic::from_error(error));
        self
    }

    pub(super) fn canonical_json_bytes(&self) -> Result<Vec<u8>, GeneratorError> {
        serde_json::to_vec(self).map_err(|_| GeneratorError::CanonicalSerialization {
            stage: ErrorStage::Diagnostics,
            reason: "json_encoding_failed",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{GeneratorConfig, QualifiedProfile};
    use super::*;

    fn fixture() -> (NormalizedGeneratorConfig, GeneratorIdentity) {
        let config = GeneratorConfig::qualified(QualifiedProfile::Minimum)
            .normalize()
            .unwrap();
        let identity = GeneratorIdentity::new(&config, [0xcd; 32], 42);
        (config, identity)
    }

    #[test]
    fn canonical_json_is_exact_ordered_and_stable() {
        let (config, identity) = fixture();
        let diagnostics = GeneratorDiagnostics::new(&config, [0xcd; 32], 42)
            .with_attempt(AttemptIdentity::new(identity, 3));
        let first = diagnostics.canonical_json_bytes().unwrap();
        let second = diagnostics.canonical_json_bytes().unwrap();
        assert_eq!(first, second);
        let json = std::str::from_utf8(&first).unwrap();
        assert!(json.starts_with("{\"schema_version\":1,\"generator_version\":1,\"rng_version\":1,\"normalized_configuration\":"));
        assert!(json.contains("\"attempt_index\":3"));
        assert!(!json.contains("foundation_failure"));
    }

    #[test]
    fn hashes_are_lowercase_64_character_hex() {
        let (config, _) = fixture();
        let diagnostics = GeneratorDiagnostics::new(&config, [0xef; 32], 9);
        for value in [
            &diagnostics.configuration_identity,
            &diagnostics.catalog_identity,
            &diagnostics.generator_identity,
        ] {
            assert_eq!(value.len(), 64);
            assert!(value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        }
    }

    #[test]
    fn failure_context_is_structured_and_path_free() {
        let (config, _) = fixture();
        let error = GeneratorError::MandatoryInfeasibility {
            stage: ErrorStage::Configuration,
            constraint: "mandatory_roles",
            required: 7,
            available: 3,
        };
        let bytes = GeneratorDiagnostics::new(&config, [0; 32], 0)
            .with_failure(&error)
            .canonical_json_bytes()
            .unwrap();
        let json = std::str::from_utf8(&bytes).unwrap();
        assert!(json.contains("\"stage\":\"configuration\""));
        assert!(json.contains("\"reason\":\"mandatory_roles\""));
        assert!(!json.contains('/'));
        assert!(!json.contains('\\'));
    }

    #[test]
    fn canonical_dto_has_no_float_map_dynamic_or_host_fields() {
        let (config, _) = fixture();
        let json = String::from_utf8(
            GeneratorDiagnostics::new(&config, [1; 32], u64::MAX)
                .canonical_json_bytes()
                .unwrap(),
        )
        .unwrap();
        for forbidden in ["host", "path", "time", "thread", ".0", "e+"] {
            assert!(!json.contains(forbidden));
        }
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(value["normalized_configuration"].is_object());
        assert!(value["seed"].is_u64());
    }
}
