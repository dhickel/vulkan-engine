//! Compound digital input: stable axis values from weighted action contributions.

use std::fmt;

use engine_events::ActionId;

use crate::InputSnapshot;

/// Validation error for compound axis configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisError {
    /// A contributor weight was NaN or infinite.
    NonFiniteWeight,
    /// An axis range endpoint was NaN or infinite, or the range was reversed.
    InvalidRange,
    /// The radial dead zone was NaN, infinite, negative, or not below one.
    InvalidDeadZone,
}

impl fmt::Display for AxisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteWeight => f.write_str("axis contributor weight must be finite"),
            Self::InvalidRange => f.write_str("axis range must be finite and ordered"),
            Self::InvalidDeadZone => f.write_str("axis dead zone must be finite and in 0.0..1.0"),
        }
    }
}

impl std::error::Error for AxisError {}

/// One contributing action for a [`CompoundAxis`].
#[derive(Clone, Debug, PartialEq)]
pub struct AxisContributor {
    action: ActionId,
    weight: f32,
}

impl AxisContributor {
    /// Create a finite weighted action contributor.
    pub fn new(action: ActionId, weight: f32) -> Result<Self, AxisError> {
        if !weight.is_finite() {
            return Err(AxisError::NonFiniteWeight);
        }
        Ok(Self { action, weight })
    }

    /// Action whose value contributes to the axis.
    pub fn action(&self) -> &ActionId {
        &self.action
    }

    /// Weight applied to the action value.
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

/// A one-dimensional compound axis built from weighted action contributions.
///
/// Reads action values from an [`InputSnapshot`], multiplies each by its
/// weight, sums the result, and clamps after aggregation to its configured
/// finite, ordered range.
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundAxis {
    contributors: Vec<AxisContributor>,
    min: f32,
    max: f32,
}

impl CompoundAxis {
    /// Create a compound axis clamped to the conventional `[-1.0, 1.0]` range.
    pub fn new(contributors: Vec<AxisContributor>) -> Result<Self, AxisError> {
        Self::with_range(contributors, -1.0, 1.0)
    }

    /// Create a compound axis with an explicit finite, ordered output range.
    pub fn with_range(
        contributors: Vec<AxisContributor>,
        min: f32,
        max: f32,
    ) -> Result<Self, AxisError> {
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(AxisError::InvalidRange);
        }
        Ok(Self {
            contributors,
            min,
            max,
        })
    }

    /// Evaluate the axis from an input snapshot.
    pub fn evaluate(&self, snapshot: &InputSnapshot) -> f32 {
        let raw = self
            .contributors
            .iter()
            .map(|contributor| snapshot.action_value(&contributor.action) * contributor.weight)
            .sum::<f32>();
        raw.clamp(self.min, self.max)
    }

    /// Contributing actions in evaluation order.
    pub fn contributors(&self) -> &[AxisContributor] {
        &self.contributors
    }

    /// Inclusive output range after aggregation.
    pub fn range(&self) -> (f32, f32) {
        (self.min, self.max)
    }
}

/// Two-axis compound input with radial dead zone and rescaling.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis2D {
    x_axis: CompoundAxis,
    y_axis: CompoundAxis,
    dead_zone: f32,
}

impl Axis2D {
    /// Create an `Axis2D` from two compound axes.
    ///
    /// `dead_zone` is a finite radial threshold in `0.0..1.0`. Aggregated
    /// components are first normalized to the unit circle, then values inside
    /// the dead zone are zeroed and values outside it are rescaled to the full
    /// unit range while preserving direction.
    pub fn new(
        x_axis: CompoundAxis,
        y_axis: CompoundAxis,
        dead_zone: f32,
    ) -> Result<Self, AxisError> {
        if !dead_zone.is_finite() || !(0.0..1.0).contains(&dead_zone) {
            return Err(AxisError::InvalidDeadZone);
        }
        Ok(Self {
            x_axis,
            y_axis,
            dead_zone,
        })
    }

    /// Evaluate both axes against a snapshot.
    ///
    /// Returns `(x, y)` with a radial magnitude no greater than one.
    pub fn evaluate(&self, snapshot: &InputSnapshot) -> (f32, f32) {
        let raw_x = self.x_axis.evaluate(snapshot);
        let raw_y = self.y_axis.evaluate(snapshot);
        let raw_magnitude = (raw_x * raw_x + raw_y * raw_y).sqrt();
        if raw_magnitude == 0.0 || raw_magnitude <= self.dead_zone {
            return (0.0, 0.0);
        }

        let magnitude = raw_magnitude.min(1.0);
        let rescaled_magnitude = (magnitude - self.dead_zone) / (1.0 - self.dead_zone);
        let direction_scale = rescaled_magnitude / raw_magnitude;
        (raw_x * direction_scale, raw_y * direction_scale)
    }

    pub fn x_axis(&self) -> &CompoundAxis {
        &self.x_axis
    }

    pub fn y_axis(&self) -> &CompoundAxis {
        &self.y_axis
    }

    pub fn dead_zone(&self) -> f32 {
        self.dead_zone
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InputLayer, LayerDescriptor, LayerPriority};

    fn contributor(action: &str, weight: f32) -> AxisContributor {
        AxisContributor::new(ActionId::new(action), weight).unwrap()
    }

    fn axis(contributors: Vec<AxisContributor>) -> CompoundAxis {
        CompoundAxis::new(contributors).unwrap()
    }

    fn action_snapshot(pressed: &[(&str, f32)]) -> InputSnapshot {
        let mut input = crate::InputSystem::new();

        struct DirectLayer {
            values: Vec<(ActionId, f32)>,
        }

        impl InputLayer for DirectLayer {
            fn on_frame_end(
                &mut self,
                _snapshot: &InputSnapshot,
                ctx: &mut crate::InputContext<'_>,
            ) {
                for (action, value) in &self.values {
                    ctx.set_action_value(action, *value);
                }
            }
        }

        input.add_layer(
            LayerDescriptor::new("direct", LayerPriority(10)),
            DirectLayer {
                values: pressed
                    .iter()
                    .map(|(name, value)| (ActionId::new(*name), *value))
                    .collect(),
            },
        );
        input.dispatch_frame();
        input.dispatch_frame();
        input.snapshot().clone()
    }

    #[test]
    fn rejects_non_finite_configuration() {
        assert_eq!(
            AxisContributor::new(ActionId::new("x"), f32::NAN),
            Err(AxisError::NonFiniteWeight)
        );
        assert_eq!(
            CompoundAxis::with_range(Vec::new(), 1.0, -1.0),
            Err(AxisError::InvalidRange)
        );
        assert_eq!(
            Axis2D::new(axis(Vec::new()), axis(Vec::new()), f32::INFINITY),
            Err(AxisError::InvalidDeadZone)
        );
        assert_eq!(
            Axis2D::new(axis(Vec::new()), axis(Vec::new()), 1.0),
            Err(AxisError::InvalidDeadZone)
        );
    }

    #[test]
    fn compound_axis_opposing_digital_actions_cancel() {
        let compound = axis(vec![contributor("right", 1.0), contributor("left", -1.0)]);
        let snapshot = action_snapshot(&[("right", 1.0), ("left", 1.0)]);
        assert_eq!(compound.evaluate(&snapshot), 0.0);
    }

    #[test]
    fn compound_axis_clamps_after_weighted_aggregation() {
        let compound = axis(vec![contributor("a", 1.0), contributor("b", 1.0)]);
        let snapshot = action_snapshot(&[("a", 1.0), ("b", 1.0)]);
        assert_eq!(compound.evaluate(&snapshot), 1.0);
    }

    #[test]
    fn compound_axis_supports_a_valid_custom_range() {
        let compound = CompoundAxis::with_range(vec![contributor("a", 2.0)], -0.5, 0.5).unwrap();
        let snapshot = action_snapshot(&[("a", 1.0)]);
        assert_eq!(compound.evaluate(&snapshot), 0.5);
        assert_eq!(compound.range(), (-0.5, 0.5));
    }

    #[test]
    fn axis2d_radial_dead_zone_and_rescaling_preserve_direction() {
        let axis2d = Axis2D::new(
            axis(vec![contributor("x", 1.0)]),
            axis(vec![contributor("y", 1.0)]),
            0.2,
        )
        .unwrap();
        let inside = action_snapshot(&[("x", 0.1), ("y", 0.1)]);
        assert_eq!(axis2d.evaluate(&inside), (0.0, 0.0));

        let above = action_snapshot(&[("x", 0.5)]);
        let (x, y) = axis2d.evaluate(&above);
        assert!((x - 0.375).abs() < 0.001);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn axis2d_normalizes_full_digital_diagonals() {
        let axis2d = Axis2D::new(
            axis(vec![contributor("x", 1.0)]),
            axis(vec![contributor("y", 1.0)]),
            0.0,
        )
        .unwrap();
        let snapshot = action_snapshot(&[("x", 1.0), ("y", 1.0)]);
        let (x, y) = axis2d.evaluate(&snapshot);
        assert!((x - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.001);
        assert!((y - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.001);
    }

    #[test]
    fn axis2d_zero_vector_stays_zero() {
        let axis2d = Axis2D::new(axis(Vec::new()), axis(Vec::new()), 0.0).unwrap();
        assert_eq!(axis2d.evaluate(&action_snapshot(&[])), (0.0, 0.0));
    }
}
