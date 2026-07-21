//! Config validation: range checks, light budget, bounds invariants.

use crate::config::NormalizedConfig;

/// Allowed resolution values.
pub const VALID_RESOLUTIONS: &[u32] = &[64, 96, 128];

/// Maximum allowed light budget.
pub const MAX_LIGHT_BUDGET: u32 = 16;

/// Minimum shell thickness (must leave room for interior cave).
pub const MIN_SHELL_THICKNESS: u32 = 0;

/// Maximum shell thickness as a fraction of resolution.
pub const MAX_SHELL_THICKNESS_RATIO: f32 = 0.4;

/// Validation error variants.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// Resolution is not in the allowed set.
    InvalidResolution { got: u32, allowed: &'static [u32] },
    /// Light budget exceeds the maximum.
    LightBudgetExceeded { got: u32, max: u32 },
    /// Shell thickness is too small.
    ShellTooThin { got: u32, min: u32 },
    /// Shell thickness exceeds maximum ratio of resolution.
    ShellTooThick {
        got: u32,
        resolution: u32,
        max_ratio: f32,
    },
    /// Resolution is zero (empty lattice).
    ResolutionZero,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidResolution { got, allowed: _ } => {
                write!(
                    f,
                    "invalid resolution {got}: must be one of {:?}",
                    VALID_RESOLUTIONS
                )
            }
            Self::LightBudgetExceeded { got, max } => {
                write!(f, "light budget {got} exceeds maximum {max}")
            }
            Self::ShellTooThin { got, min } => {
                write!(f, "shell thickness {got} is below minimum {min}")
            }
            Self::ShellTooThick {
                got,
                resolution,
                max_ratio,
            } => {
                write!(
                    f,
                    "shell thickness {got} exceeds {:.0}% of resolution {resolution} (max {})",
                    max_ratio * 100.0,
                    (*resolution as f32 * max_ratio) as u32
                )
            }
            Self::ResolutionZero => {
                write!(f, "resolution must be positive")
            }
        }
    }
}

/// Validate a `NormalizedConfig`. Returns `Ok(())` or the first error.
pub fn validate_normalized(config: &NormalizedConfig) -> Result<(), ConfigError> {
    validate_resolution(config.resolution)?;
    validate_light_budget(config.light_budget)?;
    validate_shell_thickness(config.shell_thickness, config.resolution)?;
    Ok(())
}

fn validate_resolution(resolution: u32) -> Result<(), ConfigError> {
    if resolution == 0 {
        return Err(ConfigError::ResolutionZero);
    }
    if !VALID_RESOLUTIONS.contains(&resolution) {
        return Err(ConfigError::InvalidResolution {
            got: resolution,
            allowed: VALID_RESOLUTIONS,
        });
    }
    Ok(())
}

fn validate_light_budget(budget: u32) -> Result<(), ConfigError> {
    if budget > MAX_LIGHT_BUDGET {
        return Err(ConfigError::LightBudgetExceeded {
            got: budget,
            max: MAX_LIGHT_BUDGET,
        });
    }
    Ok(())
}

fn validate_shell_thickness(thickness: u32, resolution: u32) -> Result<(), ConfigError> {
    if thickness < MIN_SHELL_THICKNESS {
        return Err(ConfigError::ShellTooThin {
            got: thickness,
            min: MIN_SHELL_THICKNESS,
        });
    }
    let max_thickness = (resolution as f32 * MAX_SHELL_THICKNESS_RATIO) as u32;
    if thickness > max_thickness {
        return Err(ConfigError::ShellTooThick {
            got: thickness,
            resolution,
            max_ratio: MAX_SHELL_THICKNESS_RATIO,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> NormalizedConfig {
        NormalizedConfig {
            seed: 0,
            resolution: 64,
            shell_thickness: 2,
            light_budget: 4,
        }
    }

    #[test]
    fn valid_config_passes() {
        assert_eq!(validate_normalized(&valid_config()), Ok(()));
    }

    #[test]
    fn resolution_zero_rejected() {
        let cfg = NormalizedConfig {
            resolution: 0,
            ..valid_config()
        };
        assert!(matches!(
            validate_normalized(&cfg),
            Err(ConfigError::ResolutionZero)
        ));
    }

    #[test]
    fn invalid_resolution_rejected() {
        let cfg = NormalizedConfig {
            resolution: 50,
            ..valid_config()
        };
        assert!(matches!(
            validate_normalized(&cfg),
            Err(ConfigError::InvalidResolution { .. })
        ));
    }

    #[test]
    fn all_valid_resolutions_accepted() {
        for &r in VALID_RESOLUTIONS {
            let cfg = NormalizedConfig {
                resolution: r,
                ..valid_config()
            };
            assert_eq!(
                validate_normalized(&cfg),
                Ok(()),
                "resolution {r} should be valid"
            );
        }
    }

    #[test]
    fn light_budget_at_max_passes() {
        let cfg = NormalizedConfig {
            light_budget: MAX_LIGHT_BUDGET,
            ..valid_config()
        };
        assert_eq!(validate_normalized(&cfg), Ok(()));
    }

    #[test]
    fn light_budget_exceeded_rejected() {
        let cfg = NormalizedConfig {
            light_budget: MAX_LIGHT_BUDGET + 1,
            ..valid_config()
        };
        let err = validate_normalized(&cfg).unwrap_err();
        assert!(matches!(
            err,
            ConfigError::LightBudgetExceeded { got, max }
            if got == MAX_LIGHT_BUDGET + 1 && max == MAX_LIGHT_BUDGET
        ));
    }

    #[test]
    fn shell_thickness_zero_passes() {
        let cfg = NormalizedConfig {
            shell_thickness: 0,
            ..valid_config()
        };
        assert_eq!(validate_normalized(&cfg), Ok(()));
    }

    #[test]
    fn shell_too_thick_rejected() {
        let cfg = NormalizedConfig {
            shell_thickness: (64.0 * MAX_SHELL_THICKNESS_RATIO) as u32 + 1,
            resolution: 64,
            ..valid_config()
        };
        assert!(matches!(
            validate_normalized(&cfg),
            Err(ConfigError::ShellTooThick { .. })
        ));
    }

    #[test]
    fn display_messages_are_human_readable() {
        let err = ConfigError::InvalidResolution {
            got: 50,
            allowed: VALID_RESOLUTIONS,
        };
        let msg = err.to_string();
        assert!(msg.contains("50"));
        assert!(msg.contains("64"));

        let err = ConfigError::LightBudgetExceeded { got: 20, max: 16 };
        assert!(err.to_string().contains("20"));
        assert!(err.to_string().contains("16"));
    }
}
