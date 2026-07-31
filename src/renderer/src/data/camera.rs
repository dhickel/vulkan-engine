//! Camera and FPS controller utilities.
//!
//! The FPS controller no longer receives raw key events directly; it consumes
//! semantic actions from the input snapshot.

use glam::{Mat4, Quat, Vec3, Vec4};
use input::{ActionId, InputConsume, InputContext, InputEvent, InputLayer, InputSnapshot};

const LOOK_AT_EPSILON: f32 = 1.0e-6;

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

/// Result of a ray-vs-AABB intersection test with additional hit metadata.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AabbRayHit {
    /// World-unit distance from the ray origin to the entry point.
    pub distance: f32,
    /// The world-space hit point on (or near) the AABB surface.
    pub point: Vec3,
    /// The outward-facing entry-face normal. A ray starting inside the box has
    /// no entry face, so its zero-distance hit has no normal.
    pub normal: Option<Vec3>,
}

/// Axis-aligned bounding box for intersection testing.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// True when every component is finite.
    pub fn is_finite(&self) -> bool {
        self.min.is_finite() && self.max.is_finite()
    }

    /// True when min[i] <= max[i] for all axes.
    pub fn is_ordered(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }

    /// Center point of the AABB.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The eight corners of this AABB.
    pub fn corners(&self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// Conservative union of `self` and `other`. Returns `None` if either is non-finite.
    pub fn union(&self, other: &Aabb) -> Option<Aabb> {
        if !self.is_finite() || !self.is_ordered() || !other.is_finite() || !other.is_ordered() {
            return None;
        }
        Some(Aabb::from_min_max(
            self.min.min(other.min),
            self.max.max(other.max),
        ))
    }

    /// Extend `self` to enclose `other`. Returns false if either is non-finite.
    pub fn extend_to_enclose(&mut self, other: &Aabb) -> bool {
        if !self.is_finite() || !self.is_ordered() || !other.is_finite() || !other.is_ordered() {
            return false;
        }
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        true
    }

    /// Transform all eight corners by `transform` and recompute min/max.
    /// Returns `None` if the transform produces non-finite results.
    pub fn transformed(&self, transform: &Mat4) -> Option<Aabb> {
        if !self.is_finite() || !self.is_ordered() || !transform.is_finite() {
            return None;
        }
        let corners = self.corners();
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for corner in corners {
            let world = transform.transform_point3(corner);
            if !world.is_finite() {
                return None;
            }
            min = min.min(world);
            max = max.max(world);
        }
        if !min.is_finite() || !max.is_finite() {
            return None;
        }
        Some(Aabb::from_min_max(min, max))
    }

    /// Ray-AABB intersection test (slab method). Returns the entry distance
    /// if the ray hits, or `None`. This compatibility API delegates to the
    /// rich hit path so both APIs share validation and intersection semantics.
    pub fn intersect_ray(&self, ray: &Ray) -> Option<f32> {
        self.intersect_ray_hit(ray).map(|hit| hit.distance)
    }

    /// Ray-AABB intersection test with hit metadata.
    ///
    /// Invalid/non-finite bounds and rays are rejected. A ray beginning inside
    /// the box reports distance zero with no synthetic entry normal. Exact
    /// corner ties use X, then Y, then Z as the deterministic entry axis.
    pub fn intersect_ray_hit(&self, ray: &Ray) -> Option<AabbRayHit> {
        if !self.is_finite()
            || !self.is_ordered()
            || !ray.origin.is_finite()
            || !ray.direction.is_finite()
            || ray.direction.length_squared() == 0.0
        {
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

        let inside = tmin < 0.0;
        let distance = tmin.max(0.0);
        let point = ray.origin + ray.direction * distance;
        let normal = (!inside).then(|| {
            let axis = if tx_min >= ty_min && tx_min >= tz_min {
                0
            } else if ty_min >= tz_min {
                1
            } else {
                2
            };
            match axis {
                0 if ray.direction.x > 0.0 => Vec3::NEG_X,
                0 => Vec3::X,
                1 if ray.direction.y > 0.0 => Vec3::NEG_Y,
                1 => Vec3::Y,
                2 if ray.direction.z > 0.0 => Vec3::NEG_Z,
                _ => Vec3::Z,
            }
        });

        Some(AabbRayHit {
            distance,
            point,
            normal,
        })
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
#[derive(Clone, Debug)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix using Vulkan's
    /// `[0, 1]` normalized-device-coordinate depth range.
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
            Vec4::new(m[0][2], m[1][2], m[2][2], m[3][2]),
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CameraLookAtError {
    NonFiniteInput,
    DegenerateDirection,
    DegenerateUp,
    CollinearUp,
}

impl CameraLookAtError {
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::NonFiniteInput => "camera look-at requires finite eye, target, and up vectors",
            Self::DegenerateDirection => "camera look-at requires distinct eye and target points",
            Self::DegenerateUp => "camera look-at requires a non-zero up vector",
            Self::CollinearUp => "camera look-at up vector cannot be collinear with view direction",
        }
    }
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

    pub(crate) fn look_at(
        &mut self,
        eye: Vec3,
        target: Vec3,
        up: Vec3,
    ) -> Result<(), CameraLookAtError> {
        if !eye.is_finite() || !target.is_finite() || !up.is_finite() {
            return Err(CameraLookAtError::NonFiniteInput);
        }

        let view_direction = target - eye;
        let direction_length_squared = view_direction.length_squared();
        if direction_length_squared <= LOOK_AT_EPSILON {
            return Err(CameraLookAtError::DegenerateDirection);
        }

        let up_length_squared = up.length_squared();
        if up_length_squared <= LOOK_AT_EPSILON {
            return Err(CameraLookAtError::DegenerateUp);
        }

        let forward = view_direction / direction_length_squared.sqrt();
        let up_normalized = up / up_length_squared.sqrt();
        if forward.cross(up_normalized).length_squared() <= LOOK_AT_EPSILON {
            return Err(CameraLookAtError::CollinearUp);
        }

        let view = Mat4::look_at_rh(eye, target, up);
        let (_, orientation, _) = view.inverse().to_scale_rotation_translation();

        self.position = eye;
        self.orientation = orientation.normalize();
        self.pitch = forward.y.clamp(-1.0, 1.0).asin().clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
        self.yaw = (-forward.x).atan2(-forward.z);

        Ok(())
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

#[cfg(test)]
mod tests {
    use super::{Aabb, Camera, CameraLookAtError, Frustum, Ray};
    use glam::{Mat4, Vec3};

    const ASSERT_EPSILON: f32 = 1.0e-4;

    fn assert_matrix_approx_eq(actual: Mat4, expected: Mat4) {
        let actual = actual.to_cols_array();
        let expected = expected.to_cols_array();
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= ASSERT_EPSILON,
                "matrix element {index} differed: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn look_at_matches_glam_view_matrix() {
        let eye = Vec3::new(3.0, 2.0, 5.0);
        let target = Vec3::new(-1.0, 0.5, 0.25);
        let up = Vec3::Y;
        let mut camera = Camera::default();

        camera.look_at(eye, target, up).unwrap();

        assert_eq!(camera.get_position(), eye);
        assert_matrix_approx_eq(camera.get_view_matrix(), Mat4::look_at_rh(eye, target, up));
    }

    #[test]
    fn look_at_rejects_degenerate_vectors_without_mutating_camera() {
        let mut camera = Camera::new(Vec3::new(1.0, 2.0, 3.0));
        let original_view = camera.get_view_matrix();
        let original_position = camera.get_position();

        assert_eq!(
            camera.look_at(original_position, original_position, Vec3::Y),
            Err(CameraLookAtError::DegenerateDirection)
        );
        assert_eq!(
            camera.look_at(Vec3::ZERO, Vec3::NEG_Z, Vec3::ZERO),
            Err(CameraLookAtError::DegenerateUp)
        );
        assert_eq!(
            camera.look_at(Vec3::ZERO, Vec3::NEG_Z, Vec3::NEG_Z),
            Err(CameraLookAtError::CollinearUp)
        );

        assert_eq!(camera.get_position(), original_position);
        assert_matrix_approx_eq(camera.get_view_matrix(), original_view);
    }

    #[test]
    fn frustum_uses_vulkan_zero_to_one_depth_range() {
        let projection = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 1.0, 10.0);
        let frustum = Frustum::from_view_projection(&projection);

        let inside = Aabb::from_min_max(Vec3::new(-0.25, -0.25, -2.0), Vec3::new(0.25, 0.25, -1.5));
        let before_near =
            Aabb::from_min_max(Vec3::new(-0.1, -0.1, -0.9), Vec3::new(0.1, 0.1, -0.8));
        let across_near =
            Aabb::from_min_max(Vec3::new(-0.1, -0.1, -1.1), Vec3::new(0.1, 0.1, -0.9));
        let beyond_far =
            Aabb::from_min_max(Vec3::new(-0.25, -0.25, -11.0), Vec3::new(0.25, 0.25, -10.5));
        let outside_right =
            Aabb::from_min_max(Vec3::new(2.1, -0.1, -2.0), Vec3::new(2.2, 0.1, -1.9));

        assert!(frustum.intersects_aabb(&inside));
        assert!(!frustum.intersects_aabb(&before_near));
        assert!(frustum.intersects_aabb(&across_near));
        assert!(!frustum.intersects_aabb(&beyond_far));
        assert!(!frustum.intersects_aabb(&outside_right));
    }

    #[test]
    fn aabb_hit_reports_entry_normal_and_inside_without_one() {
        let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
        let hit = aabb
            .intersect_ray_hit(&Ray {
                origin: Vec3::new(0.0, 0.0, 3.0),
                direction: Vec3::NEG_Z,
            })
            .expect("outside ray hits");
        assert_eq!(hit.distance, 2.0);
        assert_eq!(hit.normal, Some(Vec3::Z));

        let inside = aabb
            .intersect_ray_hit(&Ray {
                origin: Vec3::ZERO,
                direction: Vec3::X,
            })
            .expect("inside ray hits");
        assert_eq!(inside.distance, 0.0);
        assert_eq!(inside.normal, None);
        assert_eq!(
            aabb.intersect_ray(&Ray {
                origin: Vec3::ZERO,
                direction: Vec3::X,
            }),
            Some(0.0)
        );
    }

    #[test]
    fn set_position_preserves_current_orientation() {
        let eye = Vec3::new(3.0, 2.0, 5.0);
        let target = Vec3::new(-1.0, 0.5, 0.25);
        let next_eye = Vec3::new(-2.0, 4.0, 8.0);
        let mut camera = Camera::default();
        camera.look_at(eye, target, Vec3::Y).unwrap();
        let orientation = camera.orientation;

        camera.set_position(next_eye);

        assert_eq!(camera.get_position(), next_eye);
        assert_eq!(camera.orientation, orientation);
    }

    #[test]
    fn fps_rotation_after_look_at_uses_current_yaw_pitch() {
        let eye = Vec3::new(0.0, 0.0, 3.0);
        let target = Vec3::ZERO;
        let mut camera = Camera::default();
        camera.look_at(eye, target, Vec3::Y).unwrap();
        let view_before = camera.get_view_matrix();

        camera.update_rotation(0.0, 0.0);

        assert_matrix_approx_eq(camera.get_view_matrix(), view_before);
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

    /// Mouse sensitivity (radians per pixel of mouse motion).
    pub fn sensitivity(&self) -> f64 {
        self.m_sensitivity
    }

    /// Movement speed in world units per second.
    pub fn move_speed(&self) -> f32 {
        self.move_speed
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
#[derive(Clone, Debug)]
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

// ── EditorCamera ─────────────────────────────────────────────────────

/// Editor camera wrapping an [`OrbitCamera`] with perspective or
/// orthographic projection support.
///
/// Provides `screen_to_ray` for picking, `focus_on(Aabb)` for framing,
/// and direct access to view/projection matrices.
#[derive(Clone, Debug)]
pub struct EditorCamera {
    orbit: OrbitCamera,
    /// Projection mode.
    pub projection_mode: EditorProjection,
    /// Near clipping plane.
    pub near_plane: f32,
    /// Far clipping plane.
    pub far_plane: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect_ratio: f32,
}

/// Projection mode for [`EditorCamera`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EditorProjection {
    /// Perspective projection with vertical FOV in radians.
    Perspective { fov_y: f32 },
    /// Orthographic projection with half-height in world units.
    Orthographic { half_height: f32 },
}

impl Default for EditorCamera {
    fn default() -> Self {
        Self {
            orbit: OrbitCamera::default(),
            projection_mode: EditorProjection::Perspective {
                fov_y: 60.0_f32.to_radians(),
            },
            near_plane: 0.1,
            far_plane: 1000.0,
            aspect_ratio: 16.0 / 9.0,
        }
    }
}

impl EditorCamera {
    /// Create a new [`EditorCamera`] with default perspective projection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific orbit configuration.
    pub fn with_orbit(orbit: OrbitCamera) -> Self {
        Self {
            orbit,
            ..Default::default()
        }
    }

    /// Set the projection to perspective with the given vertical FOV
    /// (radians).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `fov_y` is not in (0, PI) or if `near_plane` or
    /// `far_plane` is non-finite or `near_plane >= far_plane`.
    pub fn set_perspective(
        &mut self,
        fov_y: f32,
        near_plane: f32,
        far_plane: f32,
    ) -> Result<(), &'static str> {
        if fov_y <= 0.0 || fov_y >= std::f32::consts::PI {
            return Err("fov_y must be in (0, PI)");
        }
        if !near_plane.is_finite() || !far_plane.is_finite() {
            return Err("near_plane and far_plane must be finite");
        }
        if near_plane <= 0.0 || near_plane >= far_plane {
            return Err("near_plane must be > 0 and < far_plane");
        }
        self.projection_mode = EditorProjection::Perspective { fov_y };
        self.near_plane = near_plane;
        self.far_plane = far_plane;
        Ok(())
    }

    /// Set the projection to orthographic.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `half_height` is non-finite or ≤ 0, or if
    /// `near_plane` / `far_plane` are invalid.
    pub fn set_orthographic(
        &mut self,
        half_height: f32,
        near_plane: f32,
        far_plane: f32,
    ) -> Result<(), &'static str> {
        if !half_height.is_finite() || half_height <= 0.0 {
            return Err("half_height must be finite and > 0");
        }
        if !near_plane.is_finite() || !far_plane.is_finite() {
            return Err("near_plane and far_plane must be finite");
        }
        if near_plane >= far_plane {
            return Err("near_plane must be < far_plane");
        }
        self.projection_mode = EditorProjection::Orthographic { half_height };
        self.near_plane = near_plane;
        self.far_plane = far_plane;
        Ok(())
    }

    // ── Accessors ──────────────────────────────────────────────────

    /// Return the wrapped [`OrbitCamera`] for direct manipulation.
    pub fn orbit(&self) -> &OrbitCamera {
        &self.orbit
    }

    /// Return a mutable reference to the wrapped [`OrbitCamera`].
    pub fn orbit_mut(&mut self) -> &mut OrbitCamera {
        &mut self.orbit
    }

    /// Compute the projection matrix from the current mode.
    pub fn projection_matrix(&self) -> Mat4 {
        match self.projection_mode {
            EditorProjection::Perspective { fov_y } => {
                Mat4::perspective_rh(fov_y, self.aspect_ratio, self.near_plane, self.far_plane)
            }
            EditorProjection::Orthographic { half_height } => {
                let half_width = half_height * self.aspect_ratio;
                Mat4::orthographic_rh(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    self.near_plane,
                    self.far_plane,
                )
            }
        }
    }

    /// Compute the view matrix from the orbit camera.
    pub fn view_matrix(&self) -> Mat4 {
        self.orbit.view_matrix()
    }

    /// Combined view-projection matrix.
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Inverse view-projection matrix for screen-to-world transforms.
    pub fn inv_view_projection(&self) -> Mat4 {
        self.view_projection_matrix().inverse()
    }

    /// Camera eye position in world space.
    pub fn eye_position(&self) -> Vec3 {
        self.orbit.eye_position()
    }

    // ── Screen-to-ray ──────────────────────────────────────────────

    /// Build a world-space [`Ray`] from pixel coordinates.
    ///
    /// `screen_pos.0` = x pixels, `screen_pos.1` = y pixels, with
    /// (0,0) at top-left.  `viewport_size` = (width, height) in pixels.
    ///
    /// - **Perspective**: ray originates at the eye, direction toward
    ///   the un-projected world point.
    /// - **Orthographic**: ray originates at a per-pixel point on the
    ///   near plane, direction parallel to the view axis.  This ensures
    ///   intersection tests produce correct world-space distances even
    ///   though all ortho rays are parallel.
    ///
    /// Returns `None` if the view-projection matrix is singular or the
    /// ray cannot be constructed.
    pub fn screen_to_ray(&self, screen_pos: (f32, f32), viewport_size: (u32, u32)) -> Option<Ray> {
        let inv_vp = self.inv_view_projection();
        if !inv_vp.is_finite() {
            return None;
        }

        match self.projection_mode {
            EditorProjection::Perspective { .. } => {
                let eye = self.eye_position();
                let ray = Ray::from_screen(screen_pos, viewport_size, inv_vp, eye);
                if ray.direction.length_squared() < 1e-10 {
                    return None;
                }
                Some(ray)
            }
            EditorProjection::Orthographic { .. } => {
                // Orthographic: map screen pixel to near-plane world point.
                let w = viewport_size.0 as f32;
                let h = viewport_size.1 as f32;
                if w <= 0.0 || h <= 0.0 {
                    return None;
                }
                let ndc_x = (2.0 * screen_pos.0) / w - 1.0;
                let ndc_y = 1.0 - (2.0 * screen_pos.1) / h;
                // Near plane in NDC (Vulkan [0,1] depth).
                let ndc_near = glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
                let world_near = inv_vp * ndc_near;
                if world_near.w.abs() < 1e-10 {
                    return None;
                }
                let origin =
                    glam::Vec3::new(world_near.x, world_near.y, world_near.z) / world_near.w;
                // Direction is along the view axis (camera forward), not eye-relative.
                let forward = (self.orbit.target - self.eye_position()).normalize();
                if forward.length_squared() < 1e-10 {
                    return None;
                }
                Some(Ray {
                    origin,
                    direction: forward,
                })
            }
        }
    }

    // ── Focus on AABB ──────────────────────────────────────────────

    /// Reposition the orbit target to the center of `aabb` and adjust
    /// radius so the entire AABB fits in view.
    ///
    /// Returns `Err` if the AABB is non-finite or degenerate.
    pub fn focus_on(&mut self, aabb: &Aabb) -> Result<(), &'static str> {
        let center = aabb.center();
        let radius = self.compute_fit_radius(aabb)?;
        self.orbit.target = center;
        self.orbit.radius = radius;
        Ok(())
    }

    /// Reposition the orbit target to the aggregate center of multiple
    /// AABBs (e.g., all selected objects) and adjust radius so the union
    /// fits in view.
    ///
    /// Returns `Err` if any AABB is non-finite or the aggregate is
    /// degenerate.
    pub fn focus_on_many(&mut self, aabbs: &[Aabb]) -> Result<(), &'static str> {
        if aabbs.is_empty() {
            return Err("no AABBs provided");
        }
        // Compute the union AABB.
        let mut union = aabbs[0];
        for aabb in &aabbs[1..] {
            if !aabb.is_finite() || !aabb.is_ordered() {
                return Err("AABB is non-finite or not ordered");
            }
            union.extend_to_enclose(aabb);
        }
        if !union.is_finite() || !union.is_ordered() {
            return Err("union AABB is non-finite or not ordered");
        }
        self.focus_on(&union)
    }

    /// Convert to a [`CameraView`] for scene camera submission.
    ///
    /// `viewport_width` and `viewport_height` are in pixels and used
    /// to compute the aspect ratio for the projection matrix.
    pub fn to_camera_view(&self, viewport_width: u32, viewport_height: u32) -> crate::CameraView {
        let aspect = viewport_width as f32 / viewport_height as f32;
        let view = self.view_matrix();
        let proj = match self.projection_mode {
            EditorProjection::Perspective { fov_y } => {
                glam::Mat4::perspective_rh(fov_y, aspect, self.near_plane, self.far_plane)
            }
            EditorProjection::Orthographic { half_height } => {
                let half_width = half_height * aspect;
                glam::Mat4::orthographic_rh(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    self.near_plane,
                    self.far_plane,
                )
            }
        };
        crate::CameraView::new(view, proj, self.eye_position())
    }

    // ── Internal helper ────────────────────────────────────────────

    /// Compute a radius that fits `aabb` in view given the current
    /// projection mode.
    fn compute_fit_radius(&self, aabb: &Aabb) -> Result<f32, &'static str> {
        if !aabb.is_finite() || !aabb.is_ordered() {
            return Err("AABB is non-finite or not ordered");
        }
        let extents = aabb.max - aabb.min;
        let max_extent = extents.x.max(extents.y).max(extents.z);
        if max_extent <= 0.0 {
            return Err("AABB is degenerate (zero extents)");
        }
        let required_radius = match self.projection_mode {
            EditorProjection::Perspective { fov_y } => {
                let half_fov = fov_y * 0.5;
                (max_extent * 0.5) / half_fov.tan()
            }
            EditorProjection::Orthographic { half_height } => {
                // For ortho, radius doesn't affect visibility, but a sensible
                // distance helps frame the scene.
                max_extent * 0.5 / half_height * self.orbit.radius
            }
        };
        Ok((required_radius * 1.2).clamp(self.orbit.min_radius, self.orbit.max_radius))
    }
}
