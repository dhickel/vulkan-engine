//! # Data Utilities and Synchronization Primitives
//!
//! Shared helpers for image/format math and lightweight synchronization utilities used by
//! upload workers and Vulkan transfer orchestration.
//!
//! Internal utility module with many future-facing helpers; dead code allowed.

use crate::data::gpu_data::Vertex;
use glam::{Vec3, Vec4};
use std::cmp::max;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub fn mb_to_bytes(mb: u64) -> u64 {
    mb * 1_048_576
}

pub fn calc_mips_count(width: u32, height: u32) -> u32 {
    let max_dimension = max(width, height) as f64;
    (max_dimension.log2().floor() as u32) + 1
}

/// Resolves the effective mip count for a 2D texture.
///
/// - `requested == None` means auto full chain.
/// - `requested == Some(v)` where `v > 0` is clamped to the hardware-valid max.
/// - `requested == Some(0)` or negative-equivalent falls back to auto.
pub fn resolve_texture_mip_count(width: u32, height: u32, requested: Option<u32>) -> u32 {
    let auto = calc_mips_count(width, height);
    match requested {
        Some(v) if v > 0 => v.min(auto),
        _ => auto,
    }
}

pub fn get_skybox_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let vertices = vec![
        // Front face
        Vertex {
            position: Vec3::new(-1.0, -1.0, 1.0),
            uv0_x: 0.0,
            normal: Vec3::Z,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, -1.0, 1.0),
            uv0_x: 1.0,
            normal: Vec3::Z,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, 1.0, 1.0),
            uv0_x: 1.0,
            normal: Vec3::Z,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(-1.0, 1.0, 1.0),
            uv0_x: 0.0,
            normal: Vec3::Z,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        // Back face
        Vertex {
            position: Vec3::new(-1.0, -1.0, -1.0),
            uv0_x: 1.0,
            normal: -Vec3::Z,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: -Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(-1.0, 1.0, -1.0),
            uv0_x: 1.0,
            normal: -Vec3::Z,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: -Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, 1.0, -1.0),
            uv0_x: 0.0,
            normal: -Vec3::Z,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: -Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, -1.0, -1.0),
            uv0_x: 0.0,
            normal: -Vec3::Z,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: -Vec4::X,
            ..Default::default()
        },
        // Top face
        Vertex {
            position: Vec3::new(-1.0, 1.0, -1.0),
            uv0_x: 0.0,
            normal: Vec3::Y,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(-1.0, 1.0, 1.0),
            uv0_x: 0.0,
            normal: Vec3::Y,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, 1.0, 1.0),
            uv0_x: 1.0,
            normal: Vec3::Y,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, 1.0, -1.0),
            uv0_x: 1.0,
            normal: Vec3::Y,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        // Bottom face
        Vertex {
            position: Vec3::new(-1.0, -1.0, -1.0),
            uv0_x: 0.0,
            normal: -Vec3::Y,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, -1.0, -1.0),
            uv0_x: 1.0,
            normal: -Vec3::Y,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, -1.0, 1.0),
            uv0_x: 1.0,
            normal: -Vec3::Y,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(-1.0, -1.0, 1.0),
            uv0_x: 0.0,
            normal: -Vec3::Y,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: Vec4::X,
            ..Default::default()
        },
        // Right face
        Vertex {
            position: Vec3::new(1.0, -1.0, -1.0),
            uv0_x: 1.0,
            normal: Vec3::X,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: -Vec4::Z,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, 1.0, -1.0),
            uv0_x: 1.0,
            normal: Vec3::X,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: -Vec4::Z,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, 1.0, 1.0),
            uv0_x: 0.0,
            normal: Vec3::X,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: -Vec4::Z,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(1.0, -1.0, 1.0),
            uv0_x: 0.0,
            normal: Vec3::X,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: -Vec4::Z,
            ..Default::default()
        },
        // Left face
        Vertex {
            position: Vec3::new(-1.0, -1.0, -1.0),
            uv0_x: 0.0,
            normal: -Vec3::X,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: Vec4::Z,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(-1.0, -1.0, 1.0),
            uv0_x: 1.0,
            normal: -Vec3::X,
            uv0_y: 0.0,
            color: Vec4::ONE,
            tangent: Vec4::Z,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(-1.0, 1.0, 1.0),
            uv0_x: 1.0,
            normal: -Vec3::X,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: Vec4::Z,
            ..Default::default()
        },
        Vertex {
            position: Vec3::new(-1.0, 1.0, -1.0),
            uv0_x: 0.0,
            normal: -Vec3::X,
            uv0_y: 1.0,
            color: Vec4::ONE,
            tangent: Vec4::Z,
            ..Default::default()
        },
    ];

    let indices = vec![
        0, 1, 2, 2, 3, 0, // front
        4, 5, 6, 6, 7, 4, // back
        8, 9, 10, 10, 11, 8, // top
        12, 13, 14, 14, 15, 12, // bottom
        16, 17, 18, 18, 19, 16, // right
        20, 21, 22, 22, 23, 20, // left
    ];

    (vertices, indices)
}

#[derive(Debug)]
pub struct CountdownLatch {
    count: Arc<(Mutex<usize>, Condvar)>,
}

impl CountdownLatch {
    pub fn new() -> Self {
        CountdownLatch {
            count: Arc::new((Mutex::new(0), Condvar::new())),
        }
    }

    fn increment(&self) {
        let (lock, _) = &*self.count;
        let mut count = lock.lock().unwrap();
        *count += 1;
    }

    pub fn await_zero(&self, timeout: Duration) -> Result<(), LatchTimeOutError> {
        let (lock, cvar) = &*self.count;
        let mut count = lock.lock().unwrap();
        let start = Instant::now();

        while *count > 0 {
            if start.elapsed() >= timeout {
                return Err(LatchTimeOutError);
            }

            let remaining = timeout
                .checked_sub(start.elapsed())
                .unwrap_or(Duration::from_secs(0));
            let (new_count, timeout_result) = cvar.wait_timeout(count, remaining).unwrap();
            count = new_count;

            if timeout_result.timed_out() {
                return Err(LatchTimeOutError);
            }
        }

        Ok(())
    }

    pub fn get_count(&self) -> usize {
        let (lock, _) = &*self.count;
        *lock.lock().unwrap()
    }

    pub fn create_guard(&self) -> CountDownDropGuard {
        self.increment();
        CountDownDropGuard::new(self.count.clone())
    }
}

#[derive(Debug)]
pub struct CountDownDropGuard {
    latch: Arc<(Mutex<usize>, Condvar)>,
}

impl CountDownDropGuard {
    pub fn new(latch: Arc<(Mutex<usize>, Condvar)>) -> Self {
        CountDownDropGuard { latch }
    }
}

impl Drop for CountDownDropGuard {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.latch;
        let mut count = lock.lock().unwrap();
        if *count > 0 {
            *count -= 1;
            if *count == 0 {
                cvar.notify_all();
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LatchTimeOutError;

impl std::fmt::Display for LatchTimeOutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Latch wait operation timed out")
    }
}

impl std::error::Error for LatchTimeOutError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_mips_count_1x1() {
        assert_eq!(calc_mips_count(1, 1), 1);
    }

    #[test]
    fn calc_mips_count_2x2() {
        assert_eq!(calc_mips_count(2, 2), 2);
    }

    #[test]
    fn calc_mips_count_1024x1024() {
        // log2(1024) = 10, so 10 + 1 = 11
        assert_eq!(calc_mips_count(1024, 1024), 11);
    }

    #[test]
    fn calc_mips_count_1024x256() {
        // max(1024, 256) = 1024, log2(1024) = 10, so 11
        assert_eq!(calc_mips_count(1024, 256), 11);
    }

    #[test]
    fn calc_mips_count_non_power_of_two() {
        // max dim 300, log2(300) = 8.22, floor = 8, + 1 = 9
        assert_eq!(calc_mips_count(300, 200), 9);
    }

    #[test]
    fn resolve_mip_count_auto_full_chain() {
        assert_eq!(resolve_texture_mip_count(1024, 1024, None), 11);
    }

    #[test]
    fn resolve_mip_count_explicit_clamped() {
        // Request 20 but max is 11 for 1024x1024
        assert_eq!(resolve_texture_mip_count(1024, 1024, Some(20)), 11);
    }

    #[test]
    fn resolve_mip_count_explicit_within_range() {
        // Request 5 which is less than auto=11
        assert_eq!(resolve_texture_mip_count(1024, 1024, Some(5)), 5);
    }

    #[test]
    fn resolve_mip_count_zero_falls_back_to_auto() {
        assert_eq!(resolve_texture_mip_count(1024, 1024, Some(0)), 11);
    }

    #[test]
    fn resolve_mip_count_1x1_always_1() {
        assert_eq!(resolve_texture_mip_count(1, 1, None), 1);
        assert_eq!(resolve_texture_mip_count(1, 1, Some(5)), 1);
    }
}
