//! Camera and FPS controller utilities.
//!
//! The FPS controller no longer receives raw key events directly; it consumes
//! semantic actions from the input snapshot.

use glam::{Mat4, Quat, Vec3};
use input::{ActionId, InputSnapshot};

/// Camera with position and quaternion orientation.
pub struct Camera {
    position: Vec3,
    orientation: Quat,
    pitch: f32,
    yaw: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: glam::vec3(0.0, 0.0, 1.0),
            orientation: Default::default(),
            pitch: 0.0,
            yaw: 0.0,
        }
    }
}

impl Camera {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        let translation = Mat4::from_translation(self.position);
        let rotation = Mat4::from_quat(self.orientation);
        (translation * rotation).inverse()
    }

    pub fn get_position(&self) -> Vec3 {
        self.position
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    pub fn update_rotation(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw += delta_x;
        self.pitch += delta_y;

        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );

        let yaw_quat = Quat::from_rotation_y(self.yaw);
        let pitch_quat = Quat::from_rotation_x(self.pitch);
        self.orientation = yaw_quat * pitch_quat;
    }

    pub fn update_position(&mut self, direction: Vec3, amount: f32) {
        let yaw_rotation = Quat::from_rotation_y(self.yaw);
        let velocity = yaw_rotation * direction * amount;
        self.position += velocity;
    }
}

#[derive(Clone, Debug)]
pub struct FpsActionBindings {
    pub forward: ActionId,
    pub backward: ActionId,
    pub left: ActionId,
    pub right: ActionId,
    pub up: ActionId,
    pub down: ActionId,
}

impl Default for FpsActionBindings {
    fn default() -> Self {
        Self {
            forward: ActionId::new("move.forward"),
            backward: ActionId::new("move.backward"),
            left: ActionId::new("move.left"),
            right: ActionId::new("move.right"),
            up: ActionId::new("move.up"),
            down: ActionId::new("move.down"),
        }
    }
}

pub struct FPSController {
    m_sensitivity: f64,
    move_speed: f32,
    action_bindings: FpsActionBindings,
}

impl FPSController {
    pub fn new(m_sensitivity: f64, move_speed: f32) -> Self {
        Self {
            m_sensitivity,
            move_speed,
            action_bindings: FpsActionBindings::default(),
        }
    }

    pub fn with_bindings(mut self, bindings: FpsActionBindings) -> Self {
        self.action_bindings = bindings;
        self
    }

    pub fn update_from_snapshot(
        &mut self,
        snapshot: &InputSnapshot,
        delta_seconds: f32,
        camera: &mut Camera,
    ) {
        let m_delta = snapshot.mouse_delta();
        let rot_x = m_delta.0 * self.m_sensitivity;
        let rot_y = m_delta.1 * self.m_sensitivity;
        camera.update_rotation(-rot_x as f32, -rot_y as f32);

        let amount = delta_seconds * self.move_speed;
        let mut move_vec = Vec3::ZERO;

        if snapshot.action_pressed(&self.action_bindings.forward) {
            move_vec.z -= 1.0;
        }
        if snapshot.action_pressed(&self.action_bindings.backward) {
            move_vec.z += 1.0;
        }
        if snapshot.action_pressed(&self.action_bindings.left) {
            move_vec.x -= 1.0;
        }
        if snapshot.action_pressed(&self.action_bindings.right) {
            move_vec.x += 1.0;
        }
        if snapshot.action_pressed(&self.action_bindings.up) {
            move_vec.y += 1.0;
        }
        if snapshot.action_pressed(&self.action_bindings.down) {
            move_vec.y -= 1.0;
        }

        if move_vec.length_squared() > 0.0 {
            move_vec = move_vec.normalize();
        }

        camera.update_position(move_vec, amount);
    }
}
