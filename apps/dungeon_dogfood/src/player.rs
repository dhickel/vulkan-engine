use glam::{Vec2, Vec3};

pub const PLAYER_HEIGHT: f32 = 1.8;
pub const PLAYER_RADIUS: f32 = 0.3;
pub const PLAYER_EYE_HEIGHT: f32 = 1.6;
pub const MAX_HORIZONTAL_DISPLACEMENT_PER_FRAME: f32 = 3.0;

/// Character controller capsule geometry — must match the physics body shape.
pub const PLAYER_CAPSULE_HALF_HEIGHT: f32 = 0.55;
pub const PLAYER_CAPSULE_RADIUS: f32 = 0.30;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CameraIntentGuard {
    Accepted,
    Clamped {
        attempted_displacement: f32,
        applied_displacement: f32,
    },
    RejectedNonFinite,
}

/// Player state for the character-controller-driven player.
///
/// Position is read back from the physics body after each step; velocity is
/// managed by the [`physics::CharacterController`] and retained for
/// backward compatibility with the generator validation tool.
#[derive(Debug, Clone)]
pub struct PlayerState {
    /// Current world position (authoritative — read from physics body).
    pub position: Vec3,
    /// Velocity in world units per second (set from desired translation for
    /// backward compatibility with [`collision::resolve_player_step`]).
    pub velocity: Vec3,
    /// Whether noclip mode is active. When true, collision is bypassed
    /// and vertical input is applied.
    pub noclip: bool,
    /// Per-step desired translation in world space (set by [`ingest_movement_intent`]).
    pub desired_translation: Vec3,
}

impl PlayerState {
    pub fn new(spawn_eye_position: Vec3) -> Self {
        Self {
            position: spawn_eye_position,
            velocity: Vec3::ZERO,
            noclip: false,
            desired_translation: Vec3::ZERO,
        }
    }

    /// Compute per-step desired translation from input state.
    ///
    /// `world_move_dir` is the normalized horizontal movement direction in
    /// world space (derived from camera-local axes and yaw rotation).
    /// `vertical` is the raw vertical input (-1.0..1.0).
    /// `dt` is the fixed timestep duration.  `move_speed` is the
    /// base movement speed in world units per second.
    ///
    /// Stores `desired_translation` for the next fixed step and returns a
    /// [`CameraIntentGuard`] recording whether clamping was needed.
    pub fn ingest_movement_intent(
        &mut self,
        world_move_dir: Vec2,
        vertical: f32,
        dt: f32,
        move_speed: f32,
    ) -> CameraIntentGuard {
        if dt <= 0.0 || !dt.is_finite() {
            self.desired_translation = Vec3::ZERO;
            return CameraIntentGuard::Accepted;
        }

        if !self.position.is_finite() || !world_move_dir.is_finite() {
            self.desired_translation = Vec3::ZERO;
            return CameraIntentGuard::RejectedNonFinite;
        }

        let horizontal = world_move_dir * move_speed * dt;
        let vert = if self.noclip { vertical * move_speed * dt } else { 0.0 };

        let desired = Vec3::new(horizontal.x, vert, horizontal.y);
        let attempted_displacement = desired.length();

        let clamped = if attempted_displacement > MAX_HORIZONTAL_DISPLACEMENT_PER_FRAME {
            desired.normalize_or_zero() * MAX_HORIZONTAL_DISPLACEMENT_PER_FRAME
        } else {
            desired
        };

        self.desired_translation = clamped;
        // Maintain velocity for backward compatibility with collision module.
        self.velocity = if dt > 0.0 { clamped / dt } else { Vec3::ZERO };

        if attempted_displacement > MAX_HORIZONTAL_DISPLACEMENT_PER_FRAME {
            CameraIntentGuard::Clamped {
                attempted_displacement,
                applied_displacement: MAX_HORIZONTAL_DISPLACEMENT_PER_FRAME,
            }
        } else {
            CameraIntentGuard::Accepted
        }
    }

    pub fn has_finite_position(&self) -> bool {
        self.position.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_large_horizontal_frame_displacement() {
        let mut player = PlayerState::new(Vec3::ZERO);

        let guard = player.ingest_movement_intent(
            Vec2::new(3.0, 4.0),
            0.0,
            1.0,
            1.0,
        );

        assert_eq!(
            guard,
            CameraIntentGuard::Clamped {
                attempted_displacement: 5.0,
                applied_displacement: 3.0,
            }
        );
        // desired_translation is clamped to length 3 in the (3,0,4) direction
        let expected = Vec2::new(3.0, 4.0).normalize() * 3.0;
        assert!((player.desired_translation.x - expected.x).abs() < 1e-6);
        assert!((player.desired_translation.z - expected.y).abs() < 1e-6);
    }

    #[test]
    fn rejects_non_finite_input() {
        let mut player = PlayerState::new(Vec3::ZERO);

        let guard = player.ingest_movement_intent(
            Vec2::new(f32::NAN, 0.0),
            0.0,
            1.0,
            1.0,
        );

        assert_eq!(guard, CameraIntentGuard::RejectedNonFinite);
        assert_eq!(player.desired_translation, Vec3::ZERO);
    }

    #[test]
    fn noclip_applies_vertical() {
        let mut player = PlayerState::new(Vec3::ZERO);
        player.noclip = true;

        let guard = player.ingest_movement_intent(
            Vec2::new(0.0, 0.0),
            2.0,
            0.5,
            1.0,
        );

        assert_eq!(guard, CameraIntentGuard::Accepted);
        assert!((player.desired_translation.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normal_movement_applies_horizontal_without_vertical() {
        let mut player = PlayerState::new(Vec3::ZERO);

        let guard = player.ingest_movement_intent(
            Vec2::new(1.0, 0.0),
            2.0,
            0.5,
            1.0,
        );

        assert_eq!(guard, CameraIntentGuard::Accepted);
        assert!((player.desired_translation.x - 0.5).abs() < 1e-6);
        assert!((player.desired_translation.y - 0.0).abs() < 1e-6);
        assert!((player.desired_translation.z - 0.0).abs() < 1e-6);
    }

    #[test]
    fn zero_dt_results_in_no_movement() {
        let mut player = PlayerState::new(Vec3::ZERO);
        player.ingest_movement_intent(Vec2::new(1.0, 0.0), 0.0, 0.0, 1.0);
        assert_eq!(player.desired_translation, Vec3::ZERO);
    }
}
