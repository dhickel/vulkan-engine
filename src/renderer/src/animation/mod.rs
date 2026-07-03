//! Animation system for glTF animation playback.
//!
//! Provides keyframe interpolation and per-frame bone-matrix updates.
//! Integrates with the renderer to upload joint transforms to the GPU
//! skinning buffer each frame.

use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;

/// A single keyframe value (translation, rotation, or scale).
#[derive(Clone, Debug)]
pub enum KeyframeValue {
    Translation(Vec3),
    Rotation(Quat),
    Scale(Vec3),
}

/// A channel targets a specific node and property.
#[derive(Clone, Debug)]
pub struct AnimationChannel {
    pub node_index: usize,
    pub target_path: AnimationTarget,
    pub sampler_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationTarget {
    Translation,
    Rotation,
    Scale,
}

/// Keyframe timestamps and values for one property.
#[derive(Clone, Debug)]
pub struct AnimationSampler {
    pub input: Vec<f32>, // timestamps in seconds
    pub output: Vec<KeyframeValue>,
    pub interpolation: Interpolation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline,
}

/// A named animation clip with channels and samplers.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimationChannel>,
    pub samplers: Vec<AnimationSampler>,
}

impl AnimationClip {
    pub fn new(name: impl Into<String>, duration: f32) -> Self {
        Self {
            name: name.into(),
            duration,
            channels: Vec::new(),
            samplers: Vec::new(),
        }
    }
}

/// Plays back an AnimationClip at a given time, computing per-node transforms.
pub struct AnimationPlayer {
    clip: Option<AnimationClip>,
    current_time: f32,
    playing: bool,
    loop_enabled: bool,
    speed: f32,
}

impl AnimationPlayer {
    pub fn new() -> Self {
        Self {
            clip: None,
            current_time: 0.0,
            playing: false,
            loop_enabled: false,
            speed: 1.0,
        }
    }

    /// Set the clip to play and reset time.
    pub fn set_clip(&mut self, clip: AnimationClip) {
        self.clip = Some(clip);
        self.current_time = 0.0;
    }

    pub fn play(&mut self) {
        self.playing = true;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn set_looping(&mut self, looping: bool) {
        self.loop_enabled = looping;
    }

    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Advance time by `dt` seconds and compute per-node transforms.
    /// Returns a map of node_index → local transform matrix, or empty if no clip is set.
    pub fn update(&mut self, dt: f32) -> HashMap<usize, Mat4> {
        let clip = match &self.clip {
            Some(c) => c,
            None => return HashMap::new(),
        };

        if !self.playing {
            return self.evaluate_at(self.current_time);
        }

        self.current_time += dt * self.speed;

        if self.loop_enabled && self.current_time > clip.duration {
            self.current_time %= clip.duration;
        } else if self.current_time > clip.duration {
            self.current_time = clip.duration;
            self.playing = false;
        }

        self.evaluate_at(self.current_time)
    }

    /// Evaluate the animation at a specific time, returning per-node transforms.
    fn evaluate_at(&self, time: f32) -> HashMap<usize, Mat4> {
        let clip = match &self.clip {
            Some(c) => c,
            None => return HashMap::new(),
        };

        let mut transforms: HashMap<usize, (Option<Vec3>, Option<Quat>, Option<Vec3>)> =
            HashMap::new();

        for channel in &clip.channels {
            let sampler = match clip.samplers.get(channel.sampler_index) {
                Some(s) => s,
                None => continue,
            };

            let value = interpolate_sampler(sampler, time);

            let entry = transforms.entry(channel.node_index).or_default();
            match channel.target_path {
                AnimationTarget::Translation => {
                    entry.0 = Some(match value {
                        KeyframeValue::Translation(v) => v,
                        _ => Vec3::ZERO,
                    })
                }
                AnimationTarget::Rotation => {
                    entry.1 = Some(match value {
                        KeyframeValue::Rotation(q) => q,
                        _ => Quat::IDENTITY,
                    })
                }
                AnimationTarget::Scale => {
                    entry.2 = Some(match value {
                        KeyframeValue::Scale(v) => v,
                        _ => Vec3::ONE,
                    })
                }
            }
        }

        transforms
            .into_iter()
            .map(|(idx, (t, r, s))| {
                let translation = t.unwrap_or(Vec3::ZERO);
                let rotation = r.unwrap_or(Quat::IDENTITY);
                let scale = s.unwrap_or(Vec3::ONE);
                (
                    idx,
                    Mat4::from_scale_rotation_translation(scale, rotation, translation),
                )
            })
            .collect()
    }
}

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Interpolate a sampler's value at a given time.
fn interpolate_sampler(sampler: &AnimationSampler, time: f32) -> KeyframeValue {
    if sampler.input.is_empty() || sampler.output.is_empty() {
        return sampler
            .output
            .first()
            .cloned()
            .unwrap_or(KeyframeValue::Translation(Vec3::ZERO));
    }

    // Find surrounding keyframes
    let idx = match sampler
        .input
        .binary_search_by(|t| t.partial_cmp(&time).unwrap())
    {
        Ok(i) => i,
        Err(i) => i
            .saturating_sub(1)
            .min(sampler.input.len().saturating_sub(2)),
    };

    let t0 = sampler.input[idx];
    let t1 = sampler.input[(idx + 1).min(sampler.input.len() - 1)];
    let v0 = &sampler.output[idx];
    let v1 = &sampler.output[(idx + 1).min(sampler.output.len() - 1)];

    if (t1 - t0).abs() < f32::EPSILON {
        return v0.clone();
    }

    let factor = ((time - t0) / (t1 - t0)).clamp(0.0, 1.0);

    match sampler.interpolation {
        Interpolation::Step => v0.clone(),
        Interpolation::Linear | Interpolation::CubicSpline => interpolate_linear(v0, v1, factor),
    }
}

fn interpolate_linear(a: &KeyframeValue, b: &KeyframeValue, t: f32) -> KeyframeValue {
    match (a, b) {
        (KeyframeValue::Translation(va), KeyframeValue::Translation(vb)) => {
            KeyframeValue::Translation(va.lerp(*vb, t))
        }
        (KeyframeValue::Rotation(qa), KeyframeValue::Rotation(qb)) => {
            KeyframeValue::Rotation(qa.lerp(*qb, t))
        }
        (KeyframeValue::Scale(va), KeyframeValue::Scale(vb)) => {
            KeyframeValue::Scale(va.lerp(*vb, t))
        }
        _ => a.clone(),
    }
}
