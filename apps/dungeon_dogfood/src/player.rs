use glam::{Vec2, Vec3};

pub const PLAYER_HEIGHT: f32 = 1.8;
pub const PLAYER_RADIUS: f32 = 0.3;
pub const PLAYER_EYE_HEIGHT: f32 = 1.6;
pub const MAX_HORIZONTAL_DISPLACEMENT_PER_FRAME: f32 = 3.0;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CameraIntentGuard {
    Accepted,
    Clamped {
        attempted_displacement: f32,
        applied_displacement: f32,
    },
    RejectedNonFinite,
}

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub position: Vec3,
    pub velocity: Vec3,
    pub noclip: bool,
}

impl PlayerState {
    pub fn new(spawn_eye_position: Vec3) -> Self {
        Self {
            position: spawn_eye_position,
            velocity: Vec3::ZERO,
            noclip: false,
        }
    }

    pub fn ingest_camera_intent(&mut self, camera_position: Vec3, dt: f32) -> CameraIntentGuard {
        if dt <= 0.0 || !dt.is_finite() {
            self.velocity = Vec3::ZERO;
            return CameraIntentGuard::Accepted;
        }

        if !camera_position.is_finite() || !self.position.is_finite() {
            self.velocity = Vec3::ZERO;
            return CameraIntentGuard::RejectedNonFinite;
        }

        let delta = camera_position - self.position;
        if !delta.is_finite() {
            self.velocity = Vec3::ZERO;
            return CameraIntentGuard::RejectedNonFinite;
        }

        let horizontal_delta = Vec2::new(delta.x, delta.z);
        let attempted_displacement = horizontal_delta.length();
        let applied_horizontal = if attempted_displacement > MAX_HORIZONTAL_DISPLACEMENT_PER_FRAME {
            horizontal_delta.normalize_or_zero() * MAX_HORIZONTAL_DISPLACEMENT_PER_FRAME
        } else {
            horizontal_delta
        };

        let vertical_velocity = if self.noclip { delta.y / dt } else { 0.0 };
        self.velocity = Vec3::new(
            applied_horizontal.x / dt,
            vertical_velocity,
            applied_horizontal.y / dt,
        );

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

        let guard = player.ingest_camera_intent(Vec3::new(3.0, 0.0, 4.0), 1.0);

        assert_eq!(
            guard,
            CameraIntentGuard::Clamped {
                attempted_displacement: 5.0,
                applied_displacement: 3.0,
            }
        );
        assert!((player.velocity.x - 1.8).abs() < 1e-6);
        assert!((player.velocity.z - 2.4).abs() < 1e-6);
    }

    #[test]
    fn rejects_non_finite_camera_intent() {
        let mut player = PlayerState::new(Vec3::ZERO);

        let guard = player.ingest_camera_intent(Vec3::new(f32::NAN, 0.0, 0.0), 1.0);

        assert_eq!(guard, CameraIntentGuard::RejectedNonFinite);
        assert_eq!(player.velocity, Vec3::ZERO);
    }

    #[test]
    fn noclip_preserves_vertical_camera_intent() {
        let mut player = PlayerState::new(Vec3::ZERO);
        player.noclip = true;

        let guard = player.ingest_camera_intent(Vec3::new(0.0, 2.0, 0.0), 0.5);

        assert_eq!(guard, CameraIntentGuard::Accepted);
        assert_eq!(player.velocity, Vec3::new(0.0, 4.0, 0.0));
    }
}
