use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ErrorStage {
    Configuration,
    CanonicalConfiguration,
    Rng,
    Diagnostics,
    Prefab,
}

impl ErrorStage {
    pub(super) const fn code(self) -> &'static str {
        match self {
            Self::Configuration => "configuration",
            Self::CanonicalConfiguration => "canonical_configuration",
            Self::Rng => "rng",
            Self::Diagnostics => "diagnostics",
            Self::Prefab => "prefab",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum GeneratorError {
    UnsupportedConfiguration {
        stage: ErrorStage,
        reason: &'static str,
        value: u64,
    },
    ArithmeticOverflow {
        stage: ErrorStage,
        operation: &'static str,
    },
    MandatoryInfeasibility {
        stage: ErrorStage,
        constraint: &'static str,
        required: u64,
        available: u64,
    },
    InvalidRngRange {
        stage: ErrorStage,
        reason: &'static str,
        lower: u64,
        upper: u64,
    },
    CanonicalSerialization {
        stage: ErrorStage,
        reason: &'static str,
    },
    PrefabIntegrity {
        stage: ErrorStage,
        context: String,
        reason: &'static str,
    },
}

impl GeneratorError {
    pub(super) const fn stage(&self) -> ErrorStage {
        match self {
            Self::UnsupportedConfiguration { stage, .. }
            | Self::ArithmeticOverflow { stage, .. }
            | Self::MandatoryInfeasibility { stage, .. }
            | Self::InvalidRngRange { stage, .. }
            | Self::CanonicalSerialization { stage, .. }
            | Self::PrefabIntegrity { stage, .. } => *stage,
        }
    }

    pub(super) const fn reason_code(&self) -> &'static str {
        match self {
            Self::UnsupportedConfiguration { reason, .. }
            | Self::InvalidRngRange { reason, .. }
            | Self::CanonicalSerialization { reason, .. }
            | Self::PrefabIntegrity { reason, .. } => reason,
            Self::ArithmeticOverflow { operation, .. } => operation,
            Self::MandatoryInfeasibility { constraint, .. } => constraint,
        }
    }
}

impl fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedConfiguration { stage, reason, value } => write!(
                f,
                "generator error stage={} reason={} value={}",
                stage.code(), reason, value
            ),
            Self::ArithmeticOverflow { stage, operation } => write!(
                f,
                "generator error stage={} reason=arithmetic_overflow operation={}",
                stage.code(), operation
            ),
            Self::MandatoryInfeasibility {
                stage,
                constraint,
                required,
                available,
            } => write!(
                f,
                "generator error stage={} reason={} required={} available={}",
                stage.code(), constraint, required, available
            ),
            Self::InvalidRngRange {
                stage,
                reason,
                lower,
                upper,
            } => write!(
                f,
                "generator error stage={} reason={} lower={} upper={}",
                stage.code(), reason, lower, upper
            ),
            Self::CanonicalSerialization { stage, reason } => write!(
                f,
                "generator error stage={} reason={}",
                stage.code(), reason
            ),
            Self::PrefabIntegrity { stage, context, reason } => write!(
                f,
                "generator error stage={} context={} reason={}",
                stage.code(), context, reason
            ),
        }
    }
}

impl std::error::Error for GeneratorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_error_has_stable_structured_codes_and_no_path() {
        let errors = [
            GeneratorError::UnsupportedConfiguration {
                stage: ErrorStage::Configuration,
                reason: "dimension_out_of_range",
                value: 1,
            },
            GeneratorError::ArithmeticOverflow {
                stage: ErrorStage::CanonicalConfiguration,
                operation: "length_conversion",
            },
            GeneratorError::MandatoryInfeasibility {
                stage: ErrorStage::Configuration,
                constraint: "mandatory_roles",
                required: 2,
                available: 1,
            },
            GeneratorError::InvalidRngRange {
                stage: ErrorStage::Rng,
                reason: "empty_range",
                lower: 4,
                upper: 4,
            },
            GeneratorError::CanonicalSerialization {
                stage: ErrorStage::Diagnostics,
                reason: "json_encoding_failed",
            },
            GeneratorError::PrefabIntegrity {
                stage: ErrorStage::Prefab,
                context: "small-room-square".into(),
                reason: "invalid_token",
            },
        ];
        for error in errors {
            assert!(!error.to_string().contains('/'));
            assert!(!error.to_string().contains('\\'));
            assert!(!error.stage().code().is_empty());
            assert!(!error.reason_code().is_empty());
        }
    }
}
