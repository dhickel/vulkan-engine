use std::cell::Cell;
use std::cmp::max;
use crate::data::gpu_data;
use crate::data::gpu_data::{MaterialMeta, MeshMeta, MetRoughUniform, Sampler, SurfaceMeta, TextureMeta, Vertex, VkGpuMeshBuffers, VkGpuTextureBuffer};
use crate::vulkan::vk_types::{VkBuffer, VkImageAlloc, VkPipeline};
use crate::vulkan::vk_util;
use ash::vk;
use glam::{vec4, Vec4};
use std::collections::HashMap;
use std::sync::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use half::f16;
use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use vk_mem::Alloc;

pub const EXTENT3D_ONE: vk::Extent3D = vk::Extent3D {
    width: 1,
    height: 1,
    depth: 1,
};

pub trait PackUnorm {
    fn pack_unorm_4x8(&self) -> u32;
}

impl PackUnorm for Vec4 {
    fn pack_unorm_4x8(&self) -> u32 {
        let x = (self.x.clamp(0.0, 1.0) * 255.0).round() as u32;
        let y = (self.y.clamp(0.0, 1.0) * 255.0).round() as u32;
        let z = (self.z.clamp(0.0, 1.0) * 255.0).round() as u32;
        let w = (self.w.clamp(0.0, 1.0) * 255.0).round() as u32;

        (x << 0) | (y << 8) | (z << 16) | (w << 24)
    }
}

pub fn convert_rgb32f_to_rgba32f(img: ImageBuffer<Rgb<f32>, Vec<f32>>) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
    let (width, height) = img.dimensions();

    ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = img.get_pixel(x, y);
        Rgba([pixel[0], pixel[1], pixel[2], 1.0])
    })
}

pub fn calc_mips_count(width: u32, height: u32) -> u32 {
    let max_dimension = max(width, height) as f64;
    (max_dimension.log2().floor() as u32) + 1
}

pub fn bytes_per_pixel(format : vk::Format) -> u32 {
    match format {
        vk::Format::R8_UNORM => 1,
        vk::Format::R8G8_UNORM => 2,
        vk::Format::R8G8B8_UNORM => 3,
        vk::Format::R8G8B8A8_UNORM => 4,
        vk::Format::R16_SFLOAT => 2,
        vk::Format::R16G16_SFLOAT => 4,
        vk::Format::R16G16B16_SFLOAT => 6,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        vk::Format::R32_SFLOAT => 4,
        vk::Format::R32G32_SFLOAT => 8,
        vk::Format::R32G32B32_SFLOAT => 12,
        vk::Format::R32G32B32A32_SFLOAT => 16,
        _ => panic!("Cannot calculate bytes per pixel: Unsupported format")
    }
}

pub struct Semaphore {
    permits: Mutex<usize>,
    condvar: Condvar,
}

pub struct SemaphorePermit<'a> {
    semaphore: &'a Semaphore,
    released: Cell<bool>,
}

impl<'a> SemaphorePermit<'a> {
    // Take self to consume permit to avoid duplicated release calls on drop
    pub fn release(self) {
        if !self.released.get() {
            let mut permits = self.semaphore.permits.lock().unwrap();
            *permits += 1;
            self.semaphore.condvar.notify_one();
            self.released.set(true);
        }
    }
}

impl<'a> Drop for SemaphorePermit<'a> {
    fn drop(&mut self) {
        // Release on drop if not already released
        if !self.released.get() {
            let mut permits = self.semaphore.permits.lock().unwrap();
            *permits += 1;
            self.semaphore.condvar.notify_one();
            self.released.set(true);
        }
    }
}

impl Semaphore {
    pub fn new(count: usize) -> Self {
        Semaphore {
            permits: Mutex::new(count),
            condvar: Condvar::new(),
        }
    }

    pub fn acquire(&self) -> SemaphorePermit {
        let mut permits = self.permits.lock().unwrap();
        while *permits == 0 {
            permits = self.condvar.wait(permits).unwrap();
        }
        *permits -= 1;
        SemaphorePermit {
            semaphore: self,
            released: Cell::new(false),
        }
    }

    pub fn try_acquire(&self) -> Option<SemaphorePermit> {
        let mut permits = self.permits.lock().unwrap();
        if *permits > 0 {
            *permits -= 1;
            Some(SemaphorePermit {
                semaphore: self,
                released: Cell::new(false),
            })
        } else {
            None
        }
    }

    pub fn available_permits(&self) -> usize {
        *self.permits.lock().unwrap()
    }
}


pub struct SimpleLock {
    locked: AtomicBool,
    condvar: Condvar,
    mutex: Mutex<()>,
}

pub struct SimpleLockGuard<'a> {
    lock: &'a SimpleLock,
    released: Cell<bool>,
}

impl SimpleLock {
    pub fn new() -> Self {
        SimpleLock {
            locked: AtomicBool::new(false),
            condvar: Condvar::new(),
            mutex: Mutex::new(()),
        }
    }

    pub fn acquire(&self) -> SimpleLockGuard {
        let mut guard = self.mutex.lock().unwrap();
        while self.locked.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            guard = self.condvar.wait(guard).unwrap();
        }
        SimpleLockGuard {
            lock: self,
            released: Cell::new(false),
        }
    }

    pub fn try_acquire(&self) -> Option<SimpleLockGuard> {
        if self.locked.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
            Some(SimpleLockGuard {
                lock: self,
                released: Cell::new(false),
            })
        } else {
            None
        }
    }

    fn release(&self) {
        self.locked.store(false, Ordering::Release);
        self.condvar.notify_one();
    }
}

impl<'a> SimpleLockGuard<'a> {
    pub fn release(self) {
        if !self.released.get() {
            self.lock.release();
            self.released.set(true);
        }
    }
}

impl<'a> Drop for SimpleLockGuard<'a> {
    fn drop(&mut self) {
        if !self.released.get() {
            self.lock.release();
            self.released.set(true);
        }
    }
}