use input::{KeyboardListener, ListenerType, MousePosListener};
use std::collections::HashSet;
use glam::vec3;
use winit::event::Modifiers;
use winit::keyboard::KeyCode;


pub struct Camera {
    position: glam::Vec3,
    view_matrix: glam::Mat4,
    orientation: glam::Quat,
    delta_rotation: glam::Quat,
    pitch: f32,
    yaw: f32,
}


impl Default for Camera {
    fn default() -> Self {
        Self{
            position: glam::vec3(0.0, 0.0, -5.0),
            view_matrix: Default::default(),
            orientation: Default::default(),
            delta_rotation: Default::default(),
            pitch: 0.0,
            yaw: 0.0,
        }
    }
}

impl Camera {
    pub fn update_rotation(&mut self, delta_x: f32, delta_y: f32) {
        // Apply sensitivity to deltas


        // Update yaw and pitch
        self.yaw += delta_x;
        self.pitch += delta_y;

        // Clamp the pitch to avoid gimbal lock
        self.pitch = self.pitch.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);

        // Create quaternions for the yaw and pitch
        let yaw_quat = glam::Quat::from_rotation_y(self.yaw);
        let pitch_quat = glam::Quat::from_rotation_x(self.pitch);

        // Update orientation
        self.orientation = yaw_quat * pitch_quat;

        // Update the view matrix
        self.update_view_matrix();
    }

    pub fn update_position(&mut self, m_vec: glam::Vec3) {
        let transformed_vec = self.orientation * m_vec;
        self.position += transformed_vec;
        self.update_view_matrix();
    }

    pub fn get_view_matrix(&self) -> glam::Mat4 {
        self.view_matrix
    }

    fn update_view_matrix(&mut self) {
        let rotation_matrix = glam::Mat4::from_quat(self.orientation.conjugate());
        self.view_matrix = rotation_matrix * glam::Mat4::from_translation(-self.position);
    }
}

pub struct FPSController {
    id: u32,
    prev_m_pos: glam::Vec2,
    curr_m_pos: glam::Vec2,
    view_vec: glam::Vec2,
    m_sensitivity: f32,
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
    pub fn new(id: u32, camera: Camera, m_sensitivity: f32, move_speed: f32) -> Self {
        Self {
            id: id,
            prev_m_pos: Default::default(),
            curr_m_pos: Default::default(),
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
        // Mouse
       // if self.prev_m_pos.x > 0.0 && self.prev_m_pos.y > 0.0 && self.in_window {
        //     let dx = self.curr_m_pos.x - self.prev_m_pos.x;
        //     let dy = self.curr_m_pos.y - self.prev_m_pos.y;
        //     let rotate_x = dx != 0.0;
        //     let rotate_y = dy != 0.0;
        //
        //     if rotate_x {
        //         self.view_vec.y = dx;
        //     }
        //     if rotate_y {
        //         self.view_vec.x = dy;
        //     }
        // }
        //
        // let rot_x = (self.view_vec.x * self.m_sensitivity).to_radians();
        // let rot_y = (self.view_vec.y * self.m_sensitivity).to_radians();

        // Movement
        let amount = (delta as f64 * self.move_speed as f64) as f32;
        self.move_vec = glam::Vec3::ZERO;

        if self.input_actions[0] {
            self.move_vec.z -= amount;
        }
        if self.input_actions[1] {
            self.move_vec.z += amount;
        }
        if self.input_actions[2] {
            self.move_vec.x -= amount;
        }
        if self.input_actions[3] {
            self.move_vec.x += amount;
        }

        self.camera.update_rotation(self.curr_m_pos.x, self.curr_m_pos.y);

        if self.move_vec.length() > 0.0 {
            self.move_vec = self.move_vec.normalize() * amount;
        }
        
  
        self.camera.update_position(self.move_vec);

    }
}

impl MousePosListener for FPSController {
    fn listener_type(&self) -> ListenerType {
        ListenerType::GameInput
    }

    fn listener_id(&self) -> u32 {
        self.id
    }

    fn broadcast(&mut self, pos: glam::Vec2, modifiers: &HashSet<Modifiers>) {
        self.prev_m_pos = self.curr_m_pos;
        self.curr_m_pos = pos;
        
        // println!("Prev Pos: {:?}", self.curr_m_pos);
        // println!("curr Pos: {:?}", pos);
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