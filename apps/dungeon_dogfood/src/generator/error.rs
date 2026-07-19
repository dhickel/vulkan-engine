use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorStage {
    Configuration,
    CanonicalConfiguration,
    Rng,
    Diagnostics,
    Prefab,
    Placement,
    Topology,
    Materialization,
    Ir,
}

impl ErrorStage {
    pub(super) const fn code(self) -> &'static str {
        match self {
            Self::Configuration => "configuration",
            Self::CanonicalConfiguration => "canonical_configuration",
            Self::Rng => "rng",
            Self::Diagnostics => "diagnostics",
            Self::Prefab => "prefab",
            Self::Placement => "placement",
            Self::Topology => "topology",
            Self::Materialization => "materialization",
            Self::Ir => "ir",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GeneratorError {
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
    PlacementExhausted {
        stage: ErrorStage,
        reason: &'static str,
        attempted: u64,
        placed: u64,
        target: u64,
    },
    TopologyInfeasible {
        stage: ErrorStage,
        constraint: &'static str,
        required: u64,
        available: u64,
    },
    TransitionInfeasible {
        stage: ErrorStage,
        lower_layer: u16,
        upper_layer: u16,
        required: u64,
        available: u64,
        rejected: u64,
    },
    TransitionBinding {
        stage: ErrorStage,
        transition: u32,
        reason: &'static str,
    },
    GraphBoundViolation {
        stage: ErrorStage,
        constraint: &'static str,
        minimum: u64,
        maximum: u64,
        actual: u64,
    },
    SearchExhausted {
        stage: ErrorStage,
        search: &'static str,
        attempted: u64,
        budget: u64,
    },
    IrInvariant {
        stage: ErrorStage,
        detail: String,
    },
    OccupancyConflict {
        stage: ErrorStage,
        detail: String,
    },
    TileBufferOverflow {
        stage: ErrorStage,
        detail: String,
    },
    TileBufferConflict {
        stage: ErrorStage,
        detail: String,
    },
    CorridorNoPath {
        stage: ErrorStage,
        edge: u32,
    },
    CorridorInvariant {
        stage: ErrorStage,
        edge: u32,
        detail: String,
    },
    MaterializationInfeasible {
        stage: ErrorStage,
        constraint: &'static str,
        detail: String,
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
            | Self::PrefabIntegrity { stage, .. }
            | Self::PlacementExhausted { stage, .. }
            | Self::TopologyInfeasible { stage, .. }
            | Self::TransitionInfeasible { stage, .. }
            | Self::TransitionBinding { stage, .. }
            | Self::GraphBoundViolation { stage, .. }
            | Self::SearchExhausted { stage, .. }
            | Self::IrInvariant { stage, .. }
            | Self::OccupancyConflict { stage, .. }
            | Self::TileBufferOverflow { stage, .. }
            | Self::TileBufferConflict { stage, .. }
            | Self::CorridorNoPath { stage, .. }
            | Self::CorridorInvariant { stage, .. }
            | Self::MaterializationInfeasible { stage, .. } => *stage,
        }
    }

    pub(super) fn reason_code(&self) -> &str {
        match self {
            Self::UnsupportedConfiguration { reason, .. }
            | Self::InvalidRngRange { reason, .. }
            | Self::CanonicalSerialization { reason, .. }
            | Self::PrefabIntegrity { reason, .. }
            | Self::PlacementExhausted { reason, .. } => reason,
            Self::ArithmeticOverflow { operation, .. } => operation,
            Self::MandatoryInfeasibility { constraint, .. }
            | Self::TopologyInfeasible { constraint, .. }
            | Self::GraphBoundViolation { constraint, .. } => constraint,
            Self::TransitionInfeasible { .. } => "transition_infeasible",
            Self::TransitionBinding { reason, .. } => reason,
            Self::SearchExhausted { search, .. } => search,
            Self::IrInvariant { detail, .. }
            | Self::OccupancyConflict { detail, .. }
            | Self::TileBufferOverflow { detail, .. }
            | Self::TileBufferConflict { detail, .. }
            | Self::CorridorInvariant { detail, .. } => detail.as_str(),
            Self::CorridorNoPath { .. } => "corridor_no_path",
            Self::MaterializationInfeasible { constraint, .. } => constraint,
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
            Self::PlacementExhausted { stage, reason, attempted, placed, target } => write!(
                f,
                "generator error stage={} reason={} attempted={} placed={} target={}",
                stage.code(), reason, attempted, placed, target
            ),
            Self::TopologyInfeasible { stage, constraint, required, available } => write!(
                f,
                "generator error stage={} constraint={} required={} available={}",
                stage.code(), constraint, required, available
            ),
            Self::TransitionInfeasible {
                stage,
                lower_layer,
                upper_layer,
                required,
                available,
                rejected,
            } => write!(
                f,
                "generator error stage={} reason=transition_infeasible lower_layer={} upper_layer={} required={} available={} rejected={}",
                stage.code(), lower_layer, upper_layer, required, available, rejected
            ),
            Self::TransitionBinding { stage, transition, reason } => write!(
                f,
                "generator error stage={} transition={} reason={}",
                stage.code(), transition, reason
            ),
            Self::GraphBoundViolation {
                stage,
                constraint,
                minimum,
                maximum,
                actual,
            } => write!(
                f,
                "generator error stage={} constraint={} minimum={} maximum={} actual={}",
                stage.code(), constraint, minimum, maximum, actual
            ),
            Self::SearchExhausted { stage, search, attempted, budget } => write!(
                f,
                "generator error stage={} search={} attempted={} budget={}",
                stage.code(), search, attempted, budget
            ),
            Self::IrInvariant { stage, detail } => write!(
                f,
                "generator error stage={} detail={}",
                stage.code(), detail
            ),
            Self::OccupancyConflict { stage, detail } => write!(
                f,
                "generator error stage={} detail={}",
                stage.code(), detail
            ),
            Self::TileBufferOverflow { stage, detail } => write!(
                f,
                "generator error stage={} detail={}",
                stage.code(), detail
            ),
            Self::TileBufferConflict { stage, detail } => write!(
                f,
                "generator error stage={} detail={}",
                stage.code(), detail
            ),
            Self::CorridorNoPath { stage, edge } => write!(
                f,
                "generator error stage={} reason=corridor_no_path edge={}",
                stage.code(), edge
            ),
            Self::CorridorInvariant { stage, edge, detail } => write!(
                f,
                "generator error stage={} reason=corridor_invariant edge={} detail={}",
                stage.code(), edge, detail
            ),
            Self::MaterializationInfeasible { stage, constraint, detail } => write!(
                f,
                "generator error stage={} constraint={} detail={}",
                stage.code(), constraint, detail
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
            GeneratorError::PlacementExhausted {
                stage: ErrorStage::Placement,
                reason: "grid_exhausted",
                attempted: 10,
                placed: 5,
                target: 20,
            },
            GeneratorError::TopologyInfeasible {
                stage: ErrorStage::Topology,
                constraint: "spine_distance",
                required: 100,
                available: 50,
            },
            GeneratorError::TransitionInfeasible {
                stage: ErrorStage::Placement,
                lower_layer: 0,
                upper_layer: 1,
                required: 2,
                available: 1,
                rejected: 9,
            },
            GeneratorError::TransitionBinding {
                stage: ErrorStage::Ir,
                transition: 4,
                reason: "missing_endpoint",
            },
            GeneratorError::GraphBoundViolation {
                stage: ErrorStage::Topology,
                constraint: "cycle_bounds",
                minimum: 1,
                maximum: 4,
                actual: 0,
            },
            GeneratorError::SearchExhausted {
                stage: ErrorStage::Topology,
                search: "topology_search",
                attempted: 32,
                budget: 32,
            },
            GeneratorError::IrInvariant {
                stage: ErrorStage::Ir,
                detail: "duplicate_region_id".into(),
            },
            GeneratorError::OccupancyConflict {
                stage: ErrorStage::Placement,
                detail: "reservation_overlap self region_0".into(),
            },
            GeneratorError::TileBufferOverflow {
                stage: ErrorStage::Materialization,
                detail: "buffer_capacity_exceeded".into(),
            },
            GeneratorError::TileBufferConflict {
                stage: ErrorStage::Materialization,
                detail: "overlapping_write at (0,5,5)".into(),
            },
            GeneratorError::CorridorNoPath {
                stage: ErrorStage::Materialization,
                edge: 7,
            },
            GeneratorError::CorridorInvariant {
                stage: ErrorStage::Materialization,
                edge: 7,
                detail: "legal_connector_missing".into(),
            },
            GeneratorError::MaterializationInfeasible {
                stage: ErrorStage::Materialization,
                constraint: "ramp_approach_blocked",
                detail: "lower approach cell not walkable".into(),
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
