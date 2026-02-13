//! # Camera & FPS Controller
//!
//! ## Purpose
//! Implements first-person camera with Quake-style movement (WASD + mouse look).
//! Integrates with input system via listener pattern.
//!
//! ## Components
//! - **Camera**: Core camera with position/orientation, generates view matrix
//! - **FPSController**: Input handling + movement logic, owns Camera
//!
//! ## Movement Style
//! Quake-style FPS controls:
//! - Mouse: Yaw (Y-axis) and pitch (X-axis) rotation
//! - WASD: Forward/backward/strafe
//! - Space/Shift: Up/down (noclip)
//! - Pitch clamped to ±89° to prevent gimbal lock
//!
//! ## Input Integration
//! FPSController implements KeyboardListener and MousePosListener traits.
//! Registered with input system broadcaster (see input/lib.rs).

use glam::{vec3, Mat4, Quat, Vec3, Vec4};
use input::{KeyboardListener, ListenerType, MousePosListener};
use std::collections::HashSet;

use winit::event::Modifiers;
use winit::keyboard::KeyCode;

/// Camera with position and quaternion orientation.
///
/// ## View Matrix
/// Computed as inverse of translation * rotation. Standard FPS camera.
///
/// ## Rotation Storage
/// Stores pitch/yaw as floats for clamping, but orientation as Quat for smooth interpolation.
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

pub struct FPSController {
    id: u32,
    prev_m_pos: glam::Vec2,
    m_delta: (f64, f64),
    view_vec: glam::Vec2,
    m_sensitivity: f64,
    in_window: bool,
    camera: Camera,
    input_actions: [bool; 6],
    move_speed: f32,
    move_vec: glam::Vec3,
}

#[repr(C)]
pub enum InputAction {
    Forward = 0,
    Backward = 1,
    Left = 2,
    Right = 3,
    Up = 4,
    Down = 5,
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
            input_actions: [false; 6],
            m_sensitivity,
            camera,
            move_speed,
        }
    }

    pub fn get_camera(&self) -> &Camera {
        &self.camera
    }

    pub fn get_camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn update(&mut self, delta_seconds: f32) {
        let rot_x = self.m_delta.0 * self.m_sensitivity;
        let rot_y = self.m_delta.1 * self.m_sensitivity;

        // Update rotation
        self.camera.update_rotation(-rot_x as f32, -rot_y as f32);

        // Calculate movement direction and amount
        let amount = delta_seconds * self.move_speed;
        self.move_vec = Vec3::ZERO;

        if self.input_actions[InputAction::Forward as usize] {
            self.move_vec.z -= 1.0;
        }
        if self.input_actions[InputAction::Backward as usize] {
            self.move_vec.z += 1.0;
        }
        if self.input_actions[InputAction::Left as usize] {
            self.move_vec.x -= 1.0;
        }
        if self.input_actions[InputAction::Right as usize] {
            self.move_vec.x += 1.0;
        }
        if self.input_actions[InputAction::Up as usize] {
            self.move_vec.y += 1.0;
        }
        if self.input_actions[InputAction::Down as usize] {
            self.move_vec.y -= 1.0;
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
            KeyCode::KeyW
                | KeyCode::KeyA
                | KeyCode::KeyS
                | KeyCode::KeyD
                | KeyCode::Space
                | KeyCode::ShiftLeft
        )
    }

    fn broadcast(&mut self, key: KeyCode, pressed: bool, modifiers: &HashSet<Modifiers>) {
        match key {
            KeyCode::KeyW => self.input_actions[InputAction::Forward as usize] = pressed,
            KeyCode::KeyA => self.input_actions[InputAction::Left as usize] = pressed,
            KeyCode::KeyS => self.input_actions[InputAction::Backward as usize] = pressed,
            KeyCode::KeyD => self.input_actions[InputAction::Right as usize] = pressed,
            KeyCode::Space => self.input_actions[InputAction::Up as usize] = pressed,
            KeyCode::ShiftLeft => self.input_actions[InputAction::Down as usize] = pressed,
            _ => {}
        }
    }
}
