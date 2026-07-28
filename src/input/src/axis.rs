//! Compound digital input: stable Axis2D from weighted action contributions.

use engine_events::ActionId;

use crate::InputSnapshot;

/// One contributing action for a [`CompoundAxis`].
#[derive(Clone, Debug, PartialEq)]
pub struct AxisContributor {
    /// Action whose value contributes to the axis.
    pub action: ActionId,
    /// Weight applied to the action value (positive or negative).
    pub weight: f32,
}

impl AxisContributor {
    pub const fn new(action: ActionId, weight: f32) -> Self {
        Self { action, weight }
    }
}

/// A one-dimensional compound axis built from weighted action contributions.
///
/// Reads action values from an [`InputSnapshot`], multiplies each by its
/// weight, sums the result, and clamps to `[-1.0, 1.0]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundAxis {
    contributors: Vec<AxisContributor>,
}

impl CompoundAxis {
    pub fn new(contributors: Vec<AxisContributor>) -> Self {
        Self { contributors }
    }

    /// Evaluate the axis from an input snapshot.
    ///
    /// Returns a value in `[-1.0, 1.0]` after weighted sum and clamping.
    pub fn evaluate(&self, snapshot: &InputSnapshot) -> f32 {
        let raw: f32 = self
            .contributors
            .iter()
            .map(|c| snapshot.action_value(&c.action) * c.weight)
            .sum();
        raw.clamp(-1.0, 1.0)
    }

    pub fn contributors(&self) -> &[AxisContributor] {
        &self.contributors
    }
}

/// Two-axis compound input with radial dead zone and rescaling.
///
/// Combines two [`CompoundAxis`] instances (one per axis), evaluates them
/// against a snapshot, applies a radial dead zone, and rescales the output
/// so that edge-of-dead-zone maps to zero and full deflection maps to one.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis2D {
    x_axis: CompoundAxis,
    y_axis: CompoundAxis,
    dead_zone: f32,
}

impl Axis2D {
    /// Create an `Axis2D` from two compound axes.
    ///
    /// `dead_zone` is clamped to `[0.0, 1.0)`. Values whose radial magnitude
    /// falls below `dead_zone` are zeroed; values above are rescaled from
    /// `[dead_zone, 1.0]` to `[0.0, 1.0]`.
    pub fn new(x_axis: CompoundAxis, y_axis: CompoundAxis, dead_zone: f32) -> Self {
        let dead_zone = dead_zone.clamp(0.0, 0.9999);
        Self {
            x_axis,
            y_axis,
            dead_zone,
        }
    }

    /// Evaluate both axes against a snapshot.
    ///
    /// Returns `(x, y)` where each component is in `[-1.0, 1.0]` after dead-zone
    /// processing and rescaling.
    pub fn evaluate(&self, snapshot: &InputSnapshot) -> (f32, f32) {
        let raw_x = self.x_axis.evaluate(snapshot);
        let raw_y = self.y_axis.evaluate(snapshot);
        let mag = (raw_x * raw_x + raw_y * raw_y).sqrt();

        if mag <= self.dead_zone {
            return (0.0, 0.0);
        }

        let scale = (mag - self.dead_zone) / (1.0 - self.dead_zone);
        let scaled_mag = scale / mag;

        let x = (raw_x * scaled_mag).clamp(-1.0, 1.0);
        let y = (raw_y * scaled_mag).clamp(-1.0, 1.0);
        (x, y)
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
                    .map(|(name, val)| (ActionId::new(*name), *val))
                    .collect(),
            },
        );

        // First dispatch: on_frame_end sets the action values.
        input.dispatch_frame();
        // Second dispatch: refresh_action_snapshot picks up the new values.
        input.dispatch_frame();
        input.snapshot().clone()
    }

    #[test]
    fn compound_axis_single_positive() {
        let axis = CompoundAxis::new(vec![AxisContributor::new(ActionId::new("right"), 1.0)]);
        let snap = action_snapshot(&[("right", 0.7)]);
        let val = axis.evaluate(&snap);
        assert!((val - 0.7).abs() < 0.001);
    }

    #[test]
    fn compound_axis_bidirectional() {
        let axis = CompoundAxis::new(vec![
            AxisContributor::new(ActionId::new("right"), 1.0),
            AxisContributor::new(ActionId::new("left"), -1.0),
        ]);
        let snap_right = action_snapshot(&[("right", 1.0)]);
        let snap_left = action_snapshot(&[("left", 1.0)]);
        assert!((axis.evaluate(&snap_right) - 1.0).abs() < 0.001);
        assert!((axis.evaluate(&snap_left) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn compound_axis_clamps_to_one() {
        let axis = CompoundAxis::new(vec![AxisContributor::new(ActionId::new("a"), 0.7)]);
        let snap = action_snapshot(&[("a", 1.0)]);
        let val = axis.evaluate(&snap);
        assert!((val - 0.7).abs() < 0.001);
    }

    #[test]
    fn axis2d_dead_zone_zeroes_small_input() {
        let axis = Axis2D::new(
            CompoundAxis::new(vec![AxisContributor::new(ActionId::new("x"), 1.0)]),
            CompoundAxis::new(vec![AxisContributor::new(ActionId::new("y"), 1.0)]),
            0.2,
        );
        let snap = action_snapshot(&[("x", 0.1), ("y", 0.1)]);
        let (x, y) = axis.evaluate(&snap);
        assert!((x - 0.0).abs() < 0.001);
        assert!((y - 0.0).abs() < 0.001);
    }

    #[test]
    fn axis2d_rescales_above_dead_zone() {
        let axis = Axis2D::new(
            CompoundAxis::new(vec![AxisContributor::new(ActionId::new("x"), 1.0)]),
            CompoundAxis::new(vec![AxisContributor::new(ActionId::new("y"), 1.0)]),
            0.3,
        );
        // Full deflection on X only. Raw mag = 1.0, dead_zone = 0.3.
        // scale = (1.0 - 0.3) / (1.0 - 0.3) = 1.0, output = 1.0
        let snap = action_snapshot(&[("x", 1.0)]);
        let (x, y) = axis.evaluate(&snap);
        assert!((x - 1.0).abs() < 0.001);
        assert!((y - 0.0).abs() < 0.001);

        // Half deflection. Raw x=0.5, mag=0.5, dead_zone=0.3.
        // scale = (0.5 - 0.3) / (1.0 - 0.3) = 0.2 / 0.7 ≈ 0.2857
        // scaled_mag = 0.2857 / 0.5 = 0.5714
        // output x = 0.5 * 0.5714 = 0.2857
        let snap2 = action_snapshot(&[("x", 0.5)]);
        let (x2, y2) = axis.evaluate(&snap2);
        let expected = 0.2857143;
        assert!((x2 - expected).abs() < 0.001);
        assert!((y2 - 0.0).abs() < 0.001);
    }

    #[test]
    fn axis2d_zero_dead_zone_passes_through() {
        let axis = Axis2D::new(
            CompoundAxis::new(vec![AxisContributor::new(ActionId::new("x"), 1.0)]),
            CompoundAxis::new(vec![AxisContributor::new(ActionId::new("y"), -1.0)]),
            0.0,
        );
        let snap = action_snapshot(&[("x", 0.5), ("y", 1.0)]);
        let (x, y) = axis.evaluate(&snap);
        assert!((x - 0.5).abs() < 0.001);
        assert!((y - (-1.0)).abs() < 0.001);
    }
}
