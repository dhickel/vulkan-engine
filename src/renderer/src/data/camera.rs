//! Camera and FPS controller utilities.
//!
//! The FPS controller no longer receives raw key events directly; it consumes
//! semantic actions from the input snapshot.

use glam::{Mat4, Quat, Vec3, Vec4};
use input::{ActionId, InputConsume, InputContext, InputEvent, InputLayer, InputSnapshot};

/// A ray in world space, defined by an origin and a direction.
#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Create a ray from screen coordinates using an inverse view-projection matrix.
    /// `screen_pos` is in pixel coordinates with (0,0) at top-left.
    /// `viewport_size` is (width, height) in pixels.
    pub fn from_screen(
        screen_pos: (f32, f32),
        viewport_size: (u32, u32),
        inv_view_proj: Mat4,
        camera_position: Vec3,
    ) -> Self {
        let ndc_x = (2.0 * screen_pos.0) / viewport_size.0 as f32 - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_pos.1) / viewport_size.1 as f32;
        let ndc_near = Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let ndc_far = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let world_near = inv_view_proj * ndc_near;
        let world_far = inv_view_proj * ndc_far;

        let near3 = Vec3::new(world_near.x, world_near.y, world_near.z) / world_near.w;
        let far3 = Vec3::new(world_far.x, world_far.y, world_far.z) / world_far.w;

        let direction = (far3 - near3).normalize_or_zero();
        Ray {
            origin: camera_position,
            direction,
        }
    }
}

/// Axis-aligned bounding box for intersection testing.
#[derive(Copy, Clone, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Ray-AABB intersection test (slab method). Returns the t-value of the
    /// entry point if the ray hits, or None.
    pub fn intersect_ray(&self, ray: &Ray) -> Option<f32> {
        if ray.direction.length_squared() == 0.0 {
            return None;
        }

        let (tx_min, tx_max) =
            ray_axis_interval(ray.origin.x, ray.direction.x, self.min.x, self.max.x)?;
        let (ty_min, ty_max) =
            ray_axis_interval(ray.origin.y, ray.direction.y, self.min.y, self.max.y)?;
        let (tz_min, tz_max) =
            ray_axis_interval(ray.origin.z, ray.direction.z, self.min.z, self.max.z)?;

        let tmin = tx_min.max(ty_min).max(tz_min);
        let tmax = tx_max.min(ty_max).min(tz_max);

        if tmax < 0.0 || tmin > tmax {
            return None;
        }

        Some(tmin.max(0.0))
    }
}

fn ray_axis_interval(origin: f32, direction: f32, min: f32, max: f32) -> Option<(f32, f32)> {
    if direction.abs() <= f32::EPSILON {
        return (origin >= min && origin <= max).then_some((f32::NEG_INFINITY, f32::INFINITY));
    }

    let t1 = (min - origin) / direction;
    let t2 = (max - origin) / direction;
    Some((t1.min(t2), t1.max(t2)))
}

/// View frustum for culling. Six planes (left, right, top, bottom, near, far).
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let m = vp.to_cols_array_2d();
        // Left, right, bottom, top, near, far
        let planes = [
            Vec4::new(
                m[0][3] + m[0][0],
                m[1][3] + m[1][0],
                m[2][3] + m[2][0],
                m[3][3] + m[3][0],
            ),
            Vec4::new(
                m[0][3] - m[0][0],
                m[1][3] - m[1][0],
                m[2][3] - m[2][0],
                m[3][3] - m[3][0],
            ),
            Vec4::new(
                m[0][3] + m[0][1],
                m[1][3] + m[1][1],
                m[2][3] + m[2][1],
                m[3][3] + m[3][1],
            ),
            Vec4::new(
                m[0][3] - m[0][1],
                m[1][3] - m[1][1],
                m[2][3] - m[2][1],
                m[3][3] - m[3][1],
            ),
            Vec4::new(
                m[0][3] + m[0][2],
                m[1][3] + m[1][2],
                m[2][3] + m[2][2],
                m[3][3] + m[3][2],
            ),
            Vec4::new(
                m[0][3] - m[0][2],
                m[1][3] - m[1][2],
                m[2][3] - m[2][2],
                m[3][3] - m[3][2],
            ),
        ];
        // Normalize planes
        let planes = planes.map(|p| {
            let len = Vec3::new(p.x, p.y, p.z).length();
            if len > 0.0 {
                p / len
            } else {
                p
            }
        });
        Self { planes }
    }

    /// Test if an AABB is inside (or intersecting) the frustum.
    /// Returns true if the AABB is at least partially inside.
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            let p = Vec3::new(
                if plane.x > 0.0 {
                    aabb.max.x
                } else {
                    aabb.min.x
                },
                if plane.y > 0.0 {
                    aabb.max.y
                } else {
                    aabb.min.y
                },
                if plane.z > 0.0 {
                    aabb.max.z
                } else {
                    aabb.min.z
                },
            );
            if plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

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

// ---------------------------------------------------------------------------
// Orbit / editor camera
// ---------------------------------------------------------------------------

/// Orbiting camera for editor use. Rotates around a target point using
/// spherical coordinates (theta, phi, radius).
pub struct OrbitCamera {
    pub target: Vec3,
    pub theta: f32, // azimuth (horizontal rotation)
    pub phi: f32,   // elevation (clamped to avoid gimbal lock)
    pub radius: f32,
    pub min_radius: f32,
    pub max_radius: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            theta: 0.0,
            phi: std::f32::consts::FRAC_PI_4,
            radius: 5.0,
            min_radius: 0.5,
            max_radius: 100.0,
        }
    }
}

impl OrbitCamera {
    /// Compute eye position from spherical coordinates.
    pub fn eye_position(&self) -> Vec3 {
        Vec3::new(
            self.target.x + self.radius * self.phi.cos() * self.theta.sin(),
            self.target.y + self.radius * self.phi.sin(),
            self.target.z + self.radius * self.phi.cos() * self.theta.cos(),
        )
    }

    /// Compute view matrix looking at the target from the calculated eye position.
    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.eye_position();
        Mat4::look_at_rh(eye, self.target, Vec3::Y)
    }

    /// Rotate around target by delta angles (in radians).
    pub fn rotate(&mut self, delta_theta: f32, delta_phi: f32) {
        self.theta += delta_theta;
        self.phi += delta_phi;
        self.phi = self.phi.clamp(0.05, std::f32::consts::FRAC_PI_2 - 0.05);
    }

    /// Zoom toward/away from target.
    pub fn zoom(&mut self, delta: f32) {
        self.radius = (self.radius - delta).clamp(self.min_radius, self.max_radius);
    }

    /// Pan the target in camera-local space.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let forward = (self.target - self.eye_position()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        self.target += right * delta_x + up * delta_y;
    }
}

/// Input layer for orbit camera controls.
/// - Left mouse drag: rotate
/// - Scroll: zoom
/// - Middle mouse drag: pan
pub struct OrbitController {
    pub sensitivity: f32,
    pub zoom_speed: f32,
    pub pan_speed: f32,
    last_mouse: Option<(f64, f64)>,
}

impl OrbitController {
    pub fn new() -> Self {
        Self {
            sensitivity: 0.005,
            zoom_speed: 0.5,
            pan_speed: 0.01,
            last_mouse: None,
        }
    }
}

impl InputLayer for OrbitController {
    fn on_event(&mut self, event: &InputEvent, _ctx: &mut InputContext<'_>) -> InputConsume {
        if let InputEvent::MouseMotion { delta } = event {
            self.last_mouse = Some(*delta);
        }
        InputConsume::Ignored
    }

    fn on_frame_end(&mut self, snapshot: &InputSnapshot, _ctx: &mut InputContext<'_>) {
        // Mouse delta consumed via snapshot polling in the update function
        let _ = snapshot.mouse_delta();
    }
}

impl OrbitController {
    /// Apply controller input to an OrbitCamera. Call once per frame.
    pub fn update(&mut self, camera: &mut OrbitCamera, snapshot: &InputSnapshot) {
        let m_delta = snapshot.mouse_delta();
        let scroll = snapshot.scroll_delta_lines();

        // Rotate with left mouse (polled via snapshot — the actual button state
        // would determine if rotation is active in a real implementation)
        if m_delta.0.abs() > 0.0 || m_delta.1.abs() > 0.0 {
            camera.rotate(
                -m_delta.0 as f32 * self.sensitivity,
                -m_delta.1 as f32 * self.sensitivity,
            );
        }

        // Zoom with scroll
        if scroll.abs() > 0.0 {
            camera.zoom(scroll * self.zoom_speed);
        }
    }
}
