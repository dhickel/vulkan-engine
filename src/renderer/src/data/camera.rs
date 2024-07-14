use glam::{vec3, Mat4, Quat, Vec3, Vec4};
use input::{KeyboardListener, ListenerType, MousePosListener};
use std::collections::HashSet;

use winit::event::Modifiers;
use winit::keyboard::KeyCode;

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
        let camera_rotation = Mat4::from_quat(self.orientation);
        let velocity = Vec4::new(direction.x, 0.0, direction.z, 0.0) * amount; // Assuming no vertical movement
        let transformed_velocity = camera_rotation * velocity;

        self.position += Vec3::new(transformed_velocity.x, transformed_velocity.y, transformed_velocity.z);
        println!("Updating position: {:?}", self.position)
    }
}

pub struct FPSController {
    id: u32,
    prev_m_pos: glam::Vec2,
    m_delta: (f64, f64),
    view_vec: glam::Vec2,
    m_sensitivity: f64,
    in_window: bool,
    camera: Camera,
    input_actions: [bool; 4],
    move_speed: f32,
    move_vec: glam::Vec3,
}

#[repr(C)]
pub enum InputAction {
    MoveUp = 0,
    MoveDown = 1,
    MoveLeft = 2,
    MoveRight = 3,
}

impl FPSController {
    pub fn new(id: u32, camera: Camera, m_sensitivity: f64, move_speed: f32) -> Self {
        Self {
            id: id,
            prev_m_pos: Default::default(),
            m_delta: Default::default(),
            view_vec: Default::default(),
            move_vec: Default::default(),
            in_window: true,
            input_actions: [false; 4],
            m_sensitivity,
            camera,
            move_speed,
        }
    }

    pub fn get_camera(&self) -> &Camera {
        &self.camera
    }

    pub fn update(&mut self, delta: u128) {
        let rot_x = self.m_delta.0 * self.m_sensitivity;
        let rot_y = self.m_delta.1 * self.m_sensitivity;

        // Update rotation
        self.camera.update_rotation(-rot_x as f32, -rot_y as f32);

        // Calculate movement direction and amount
        let amount = (delta as f64 * self.move_speed as f64 / 1000.0) as f32; // Assuming delta is in milliseconds
        self.move_vec = Vec3::ZERO;

        if self.input_actions[InputAction::MoveUp as usize] {
            self.move_vec.z -= 1.0;
        }
        if self.input_actions[InputAction::MoveDown as usize] {
            self.move_vec.z += 1.0;
        }
        if self.input_actions[InputAction::MoveLeft as usize] {
            self.move_vec.x -= 1.0;
        }
        if self.input_actions[InputAction::MoveRight as usize] {
            self.move_vec.x += 1.0;
        }

        if self.move_vec.length_squared() > 0.0 {
            self.move_vec = self.move_vec.normalize();
        }

        // Update position
        self.camera.update_position(self.move_vec, amount);
    }
}

impl MousePosListener for FPSController {
    fn listener_type(&self) -> ListenerType {
        ListenerType::GameInput
    }

    fn listener_id(&self) -> u32 {
        self.id
    }

    fn broadcast(&mut self, delta: (f64, f64), modifiers: &HashSet<Modifiers>) {
        self.m_delta = delta;
    }
}

impl KeyboardListener for FPSController {
    fn listener_type(&self) -> ListenerType {
        ListenerType::GameInput
    }

    fn listener_id(&self) -> u32 {
        self.id
    }

    fn listener_for(&self, key: KeyCode) -> bool {
        matches!(
            key,
            KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD
        )
    }

    fn broadcast(&mut self, key: KeyCode, pressed: bool, modifiers: &HashSet<Modifiers>) {
        match key {
            KeyCode::KeyW => self.input_actions[InputAction::MoveUp as usize] = pressed,
            KeyCode::KeyA => self.input_actions[InputAction::MoveLeft as usize] = pressed,
            KeyCode::KeyS => self.input_actions[InputAction::MoveDown as usize] = pressed,
            KeyCode::KeyD => self.input_actions[InputAction::MoveRight as usize] = pressed,
            _ => {}
        }
    }
}
