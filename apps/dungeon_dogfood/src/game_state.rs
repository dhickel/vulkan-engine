use glam::Vec3;

/// Runtime game state
///
/// Phase 03: Basic player position tracking
/// Phase 04: Will add collision solver and movement logic
pub struct GameState {
    pub player_position: Vec3,
}

impl GameState {
    pub fn new(spawn_position: Vec3) -> Self {
        Self {
            player_position: spawn_position,
        }
    }

    /// Update game state for a frame
    ///
    /// Phase 03: Placeholder for collision update hook
    /// Phase 04: Will implement collision solver and movement
    pub fn update(&mut self, _delta_seconds: f32) {
        // Phase 04 will add:
        // - Read input movement intent
        // - Run collision solver
        // - Update player_position with resolved movement
        // - Update camera position via renderer.set_camera_position()
    }
}
