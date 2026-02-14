use glam::Vec3;

pub const PLAYER_HEIGHT: f32 = 1.8;
pub const PLAYER_RADIUS: f32 = 0.3;
pub const PLAYER_EYE_HEIGHT: f32 = 1.6;

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub position: Vec3,
    pub velocity: Vec3,
}

impl PlayerState {
    pub fn new(spawn_eye_position: Vec3) -> Self {
        Self {
            position: spawn_eye_position,
            velocity: Vec3::ZERO,
        }
    }

    pub fn ingest_camera_intent(&mut self, camera_position: Vec3, dt: f32) {
        if dt <= 0.0 {
            self.velocity = Vec3::ZERO;
            return;
        }

        let delta = camera_position - self.position;
        self.velocity = Vec3::new(delta.x / dt, 0.0, delta.z / dt);
    }
}
