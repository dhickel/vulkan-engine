//! QuakeToEngine coordinate transform: the single authoritative conversion from
//! Quake space to engine space.
//!
//! Contract: `bsp-spatial-physics.md` §2.

use glam::Vec3;

/// The canonical Quake-to-engine coordinate transform.
///
/// Every vector, plane, AABB, and angle conversion passes through this struct.
/// All subsystems use the same resolved scale. Cache keys include scale.
#[derive(Debug, Clone, Copy)]
pub struct QuakeToEngine {
    /// Scale factor: Quake units → engine units. Default `0.0254` (meters).
    pub scale: f32,
}

impl Default for QuakeToEngine {
    fn default() -> Self {
        QuakeToEngine { scale: 0.0254 }
    }
}

impl QuakeToEngine {
    /// Create a transform with a custom scale.
    pub fn new(scale: f32) -> Self {
        QuakeToEngine { scale }
    }

    /// Convert a position: `(x, y, z) → scale * (x, z, -y)`.
    #[inline]
    pub fn position(&self, qx: f32, qy: f32, qz: f32) -> Vec3 {
        Vec3::new(self.scale * qx, self.scale * qz, -self.scale * qy)
    }

    /// Convert a position vector.
    #[inline]
    pub fn position_vec3(&self, v: Vec3) -> Vec3 {
        self.position(v.x, v.y, v.z)
    }

    /// Convert a normal: `(nx, ny, nz) → (nx, nz, -ny)` then renormalize.
    /// Returns zero-vector if input is zero (no valid normal).
    #[inline]
    pub fn normal(&self, nx: f32, ny: f32, nz: f32) -> Vec3 {
        let result = Vec3::new(nx, nz, -ny);
        if result.length_squared() < 1e-12 {
            return Vec3::ZERO;
        }
        result.normalize()
    }

    /// Convert a normal vector.
    #[inline]
    pub fn normal_vec3(&self, n: Vec3) -> Vec3 {
        self.normal(n.x, n.y, n.z)
    }

    /// Convert a plane `(normal, dist)`: convert normal, `dist_engine = scale * dist`.
    #[inline]
    pub fn plane(&self, normal: Vec3, quake_dist: f32) -> (Vec3, f32) {
        let engine_normal = self.normal_vec3(normal);
        (engine_normal, self.scale * quake_dist)
    }

    /// Convert an AABB: convert both corners, recompute min/max.
    #[inline]
    pub fn aabb(&self, mins: Vec3, maxs: Vec3) -> (Vec3, Vec3) {
        let c0 = self.position_vec3(mins);
        let c1 = self.position_vec3(maxs);
        (c0.min(c1), c0.max(c1))
    }

    /// Convert a quake angle (0–360, -1 up, -2 down) to an engine-space direction vector.
    pub fn angle_to_direction(&self, angle: f32) -> Vec3 {
        match angle as i32 {
            -1 => Vec3::new(0.0, 1.0, 0.0),  // up → engine +Y
            -2 => Vec3::new(0.0, -1.0, 0.0), // down → engine -Y
            _ => {
                // Quake yaw: angle in degrees, 0 = east (+X), 90 = north (+Y in Quake, -Z in engine)
                let rad = angle.to_radians();
                Vec3::new(rad.cos(), 0.0, -rad.sin()).normalize()
            }
        }
    }

    /// Convert quake `angles` (pitch, yaw, roll) to an engine-space direction vector.
    ///
    /// Quake pitch: positive = look DOWN (`forward.z = -sin(p)`).
    /// After `QuakeToEngine`, that maps to negative engine Y.
    pub fn angles_to_direction(&self, pitch: f32, yaw: f32, _roll: f32) -> Vec3 {
        // Quake forward vector: pitch and yaw
        let p = pitch.to_radians();
        let y = yaw.to_radians();
        let qx = p.cos() * y.cos();
        let qy = p.cos() * y.sin();
        let qz = -p.sin(); // quake positive pitch = look down
        self.normal(qx, qy, qz)
    }

    /// Convert quake `mangle` (pitch, yaw, roll) to an engine-space direction vector.
    /// Same semantics as `angles`.
    #[inline]
    pub fn mangle_to_direction(&self, pitch: f32, yaw: f32, roll: f32) -> Vec3 {
        self.angles_to_direction(pitch, yaw, roll)
    }

    /// Convert quake Euler angles to engine Euler angles.
    ///
    /// - engine_pitch = -quake_pitch (positive-down → positive-up)
    /// - engine_yaw = quake_yaw (Z rotation unchanged)
    /// - engine_roll = -quake_roll (Y rotation negated by YZ swap)
    pub fn angles_to_engine_euler(&self, pitch: f32, yaw: f32, roll: f32) -> Vec3 {
        Vec3::new(-pitch, yaw, -roll)
    }

    /// Convert quake Euler angles to engine Euler (using mangle tuple order).
    #[inline]
    pub fn mangle_to_engine_euler(&self, pitch: f32, yaw: f32, roll: f32) -> Vec3 {
        self.angles_to_engine_euler(pitch, yaw, roll)
    }

    /// Rebase an inline model's local-space vertex to world space for the model origin.
    ///
    /// Model vertices are in local model space relative to the model origin.
    /// The model origin is in Quake space. The instance transform is:
    /// `translate(origin) * rotate(angle/angles) * scale`.
    #[inline]
    pub fn inline_model_origin(&self, model_origin: Vec3) -> Vec3 {
        self.position_vec3(model_origin)
    }

    /// Compute the push-off epsilon for plane-surface queries in engine units.
    #[inline]
    pub fn surface_epsilon(&self) -> f32 {
        1e-3
    }

    /// Point-equality epsilon in engine units.
    #[inline]
    pub fn point_epsilon(&self) -> f32 {
        1e-6
    }

    /// Planarity epsilon in engine units.
    #[inline]
    pub fn planarity_epsilon(&self) -> f32 {
        1e-4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scale_is_inches_to_meters() {
        let qte = QuakeToEngine::default();
        assert!((qte.scale - 0.0254).abs() < 1e-8);
    }

    #[test]
    fn position_axes_swap() {
        let qte = QuakeToEngine::default();
        let p = qte.position(10.0, 20.0, 30.0);
        assert!((p.x - 10.0 * 0.0254).abs() < 1e-6);
        assert!((p.y - 30.0 * 0.0254).abs() < 1e-6); // Z → Y
        assert!((p.z - (-20.0 * 0.0254)).abs() < 1e-6); // Y → -Z
    }

    #[test]
    fn normal_axes_swap_and_renormalize() {
        let qte = QuakeToEngine::default();
        let n = qte.normal(0.0, 1.0, 0.0); // quake up → engine -Z
        assert!((n.x - 0.0).abs() < 1e-6);
        assert!((n.y - 0.0).abs() < 1e-6);
        assert!((n.z - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn normal_identity_is_preserved() {
        let qte = QuakeToEngine::default();
        // quake east (1,0,0) → engine east (1,0,0)
        let n = qte.normal(1.0, 0.0, 0.0);
        assert!((n.x - 1.0).abs() < 1e-6);
        assert!((n.y).abs() < 1e-6);
        assert!((n.z).abs() < 1e-6);
    }

    #[test]
    fn angle_sentinels() {
        let qte = QuakeToEngine::default();
        let up = qte.angle_to_direction(-1.0);
        assert!((up.x).abs() < 1e-6);
        assert!((up.y - 1.0).abs() < 1e-6); // engine +Y
        assert!((up.z).abs() < 1e-6);

        let down = qte.angle_to_direction(-2.0);
        assert!((down.x).abs() < 1e-6);
        assert!((down.y + 1.0).abs() < 1e-6); // engine -Y
        assert!((down.z).abs() < 1e-6);
    }

    #[test]
    fn angle_0_faces_east() {
        let qte = QuakeToEngine::default();
        let dir = qte.angle_to_direction(0.0);
        assert!((dir.x - 1.0).abs() < 1e-6);
        assert!((dir.y).abs() < 1e-6);
        assert!((dir.z).abs() < 1e-6);
    }

    #[test]
    fn angle_90_faces_north_engine_negz() {
        let qte = QuakeToEngine::default();
        let dir = qte.angle_to_direction(90.0);
        assert!((dir.x).abs() < 1e-5);
        assert!((dir.y).abs() < 1e-5);
        assert!((dir.z + 1.0).abs() < 1e-5); // engine -Z
    }

    #[test]
    fn aabb_converts_both_corners() {
        let qte = QuakeToEngine::default();
        let (emins, emaxs) = qte.aabb(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(100.0, 200.0, 300.0),
        );
        assert!(emins.x < emaxs.x);
        assert!(emins.y < emaxs.y);
        assert!(emins.z < emaxs.z);
    }

    #[test]
    fn inline_model_origin() {
        let qte = QuakeToEngine::default();
        let origin = qte.inline_model_origin(Vec3::new(128.0, 256.0, 64.0));
        // (128*0.0254, 64*0.0254, -256*0.0254)
        assert!((origin.x - 128.0 * 0.0254).abs() < 1e-6);
        assert!((origin.y - 64.0 * 0.0254).abs() < 1e-6);
        assert!((origin.z + 256.0 * 0.0254).abs() < 1e-6);
    }

    #[test]
    fn angles_to_engine_euler() {
        let qte = QuakeToEngine::default();
        let euler = qte.angles_to_engine_euler(30.0, 45.0, 10.0);
        assert!((euler.x + 30.0).abs() < 1e-6); // pitch negated
        assert!((euler.y - 45.0).abs() < 1e-6); // yaw unchanged
        assert!((euler.z + 10.0).abs() < 1e-6); // roll negated
    }
}
