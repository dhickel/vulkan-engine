//! Animation system for glTF animation playback.
//!
//! Provides keyframe interpolation and per-frame bone-matrix updates.
//! Integrates with the renderer to upload joint transforms to the GPU
//! skinning buffer each frame.
//!
//! ## Durable animation targets
//! Channels reference scene nodes by durable [`SceneNodeId`] rather than
//! by temporary index. Stale targets are detected before evaluation and
//! return distinct errors.
//!
//! ## Interpolation modes
//! - **Step**: previous key value.
//! - **Linear**: `lerp` for translation/scale, shortest-path normalized
//!   quaternion `slerp` for rotation.
//! - **CubicSpline**: three output elements per key (in-tangent, value,
//!   out-tangent), Hermite interpolation with tangents scaled by the key
//!   interval. Quaternion tangents are applied after normalization.

use crate::api::AnimationError;
use crate::scene::scene_world::SceneNodeId;
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;

/// A single keyframe value (translation, rotation, or scale).
#[derive(Clone, Debug, PartialEq)]
pub enum KeyframeValue {
    Translation(Vec3),
    Rotation(Quat),
    Scale(Vec3),
}

/// A channel targets a specific scene node and property by durable
/// [`SceneNodeId`] instead of a volatile index.
#[derive(Clone, Debug)]
pub struct AnimationChannel {
    /// Durable scene node target.
    pub target: SceneNodeId,
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
    /// Timestamps in seconds, must be monotonic and finite.
    pub input: Vec<f32>,
    /// Output values. For `CubicSpline` there are three output elements
    /// per input key (in-tangent, value, out-tangent).
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

    /// Validate the clip before acceptance.
    ///
    /// Checks duration, sampler indices, target cardinality, finite
    /// timestamps, and interpolation output counts.
    pub fn validate(&self) -> Result<(), AnimationError> {
        if !self.duration.is_finite() || self.duration < 0.0 {
            return Err(AnimationError::InvalidDuration(format!(
                "duration must be finite and >= 0, got {}",
                self.duration
            )));
        }
        if self.name.trim().is_empty() {
            return Err(AnimationError::InvalidClip(
                "clip name must not be empty".to_string(),
            ));
        }

        for (ci, channel) in self.channels.iter().enumerate() {
            let sampler = self.samplers.get(channel.sampler_index).ok_or_else(|| {
                AnimationError::InvalidChannel(format!(
                    "channel {} sampler index {} is out of bounds (samplers len={})",
                    ci,
                    channel.sampler_index,
                    self.samplers.len()
                ))
            })?;

            // Validate sampler inputs are monotonic and finite.
            if sampler.input.is_empty() {
                return Err(AnimationError::InvalidSampler(format!(
                    "sampler {} has no input timestamps",
                    channel.sampler_index
                )));
            }
            for (i, window) in sampler.input.windows(2).enumerate() {
                let t0 = window[0];
                let t1 = window[1];
                if !t0.is_finite() {
                    return Err(AnimationError::InvalidTimestamp(format!(
                        "sampler {} input[{}]={} is non-finite",
                        channel.sampler_index, i, t0
                    )));
                }
                if t1 < t0 {
                    return Err(AnimationError::InvalidTimestamp(format!(
                        "sampler {} input is not monotonic: input[{}]={} > input[{}]={}",
                        channel.sampler_index,
                        i,
                        t0,
                        i + 1,
                        t1
                    )));
                }
            }
            if let Some(last) = sampler.input.last() {
                if !last.is_finite() {
                    return Err(AnimationError::InvalidTimestamp(format!(
                        "sampler {} last input={} is non-finite",
                        channel.sampler_index, last
                    )));
                }
            }

            // Validate output cardinality.
            match sampler.interpolation {
                Interpolation::Step | Interpolation::Linear => {
                    if sampler.output.len() != sampler.input.len() {
                        return Err(AnimationError::CardinalityMismatch(format!(
                            "sampler {} has {} outputs for {} inputs (need exactly inputs for {:?})",
                            channel.sampler_index,
                            sampler.output.len(),
                            sampler.input.len(),
                            sampler.interpolation
                        )));
                    }
                }
                Interpolation::CubicSpline => {
                    let expected = sampler.input.len().checked_mul(3).ok_or_else(|| {
                        AnimationError::CardinalityMismatch(format!(
                            "sampler {} input count overflows cubic output cardinality",
                            channel.sampler_index
                        ))
                    })?;
                    if sampler.output.len() != expected {
                        return Err(AnimationError::CardinalityMismatch(format!(
                            "sampler {} has {} outputs for {} inputs (need exactly {} = inputs*3 for CubicSpline)",
                            channel.sampler_index,
                            sampler.output.len(),
                            sampler.input.len(),
                            expected
                        )));
                    }
                }
            }

            // Validate output type consistency.
            let expected_kind = match channel.target_path {
                AnimationTarget::Translation => "Translation",
                AnimationTarget::Rotation => "Rotation",
                AnimationTarget::Scale => "Scale",
            };
            for (oi, output) in sampler.output.iter().enumerate() {
                if !keyframe_value_is_valid(output) {
                    return Err(AnimationError::InvalidKeyframe(format!(
                        "sampler {} output[{}] is non-finite or invalid",
                        channel.sampler_index, oi
                    )));
                }
                let kind = match output {
                    KeyframeValue::Translation(_) => "Translation",
                    KeyframeValue::Rotation(_) => "Rotation",
                    KeyframeValue::Scale(_) => "Scale",
                };
                if kind != expected_kind {
                    return Err(AnimationError::CardinalityMismatch(format!(
                        "sampler {} output[{}] is {} but channel {} expects {}",
                        channel.sampler_index,
                        oi,
                        kind,
                        ci,
                        expected_kind
                    )));
                }
            }
        }

        Ok(())
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
    /// Returns an error if the clip fails validation.
    pub fn set_clip(&mut self, clip: AnimationClip) -> Result<(), AnimationError> {
        clip.validate()?;
        self.clip = Some(clip);
        self.current_time = 0.0;
        Ok(())
    }

    /// Returns a reference to the current clip, if any.
    pub fn clip(&self) -> Option<&AnimationClip> {
        self.clip.as_ref()
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

    /// Returns the current playback time in seconds.
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Returns whether the player is currently playing.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Advance time by `dt` seconds and compute per-node transforms.
    ///
    /// Returns a map of [`SceneNodeId`] → local transform matrix, or an
    /// empty map if no clip is set. Values are finite-checked and
    /// normalized before return.
    pub fn update(
        &mut self,
        dt: f32,
        valid_targets: &HashMap<SceneNodeId, ()>,
    ) -> Result<HashMap<SceneNodeId, Mat4>, AnimationError> {
        let clip = match &self.clip {
            Some(c) => c,
            None => return Ok(HashMap::new()),
        };

        if !dt.is_finite() || dt < 0.0 {
            return Err(AnimationError::InvalidTimestamp(format!(
                "delta time must be finite and non-negative, got {dt}"
            )));
        }

        if !self.playing {
            return self.evaluate_at(self.current_time, valid_targets);
        }

        self.current_time += dt * self.speed;

        if !self.current_time.is_finite() {
            return Err(AnimationError::InvalidTimestamp(format!(
                "current time became non-finite after advance: dt={dt}, speed={}",
                self.speed
            )));
        }

        if self.loop_enabled && clip.duration > 0.0 {
            if self.current_time > clip.duration {
                self.current_time %= clip.duration;
            } else if self.current_time < 0.0 {
                self.current_time = clip.duration + (self.current_time % clip.duration);
            }
        } else if self.current_time > clip.duration {
            self.current_time = clip.duration;
            self.playing = false;
        } else if self.current_time < 0.0 {
            self.current_time = 0.0;
        }

        self.evaluate_at(self.current_time, valid_targets)
    }

    /// Evaluate the animation at a specific time, returning per-node transforms.
    fn evaluate_at(
        &self,
        time: f32,
        valid_targets: &HashMap<SceneNodeId, ()>,
    ) -> Result<HashMap<SceneNodeId, Mat4>, AnimationError> {
        let clip = match &self.clip {
            Some(c) => c,
            None => return Ok(HashMap::new()),
        };

        let mut transforms: HashMap<SceneNodeId, (Option<Vec3>, Option<Quat>, Option<Vec3>)> =
            HashMap::new();

        for channel in &clip.channels {
            // Check that the target node is still valid.
            if !valid_targets.contains_key(&channel.target) {
                return Err(AnimationError::StaleTarget(channel.target));
            }

            let sampler = match clip.samplers.get(channel.sampler_index) {
                Some(s) => s,
                None => {
                    return Err(AnimationError::InvalidChannel(format!(
                        "sampler index {} out of bounds",
                        channel.sampler_index
                    )))
                }
            };

            let value = interpolate_sampler(sampler, time)?;

            let entry = transforms.entry(channel.target).or_default();
            match channel.target_path {
                AnimationTarget::Translation => {
                    if let KeyframeValue::Translation(v) = value {
                        if !v.is_finite() {
                            return Err(AnimationError::NonFiniteOutput(
                                "translation interpolation produced a non-finite value".to_string(),
                            ));
                        }
                        entry.0 = Some(v);
                    }
                }
                AnimationTarget::Rotation => {
                    if let KeyframeValue::Rotation(q) = value {
                        if !q.is_finite() || !q.is_normalized() {
                            return Err(AnimationError::NonFiniteOutput(
                                "rotation interpolation produced an invalid quaternion".to_string(),
                            ));
                        }
                        entry.1 = Some(q);
                    }
                }
                AnimationTarget::Scale => {
                    if let KeyframeValue::Scale(v) = value {
                        if !v.is_finite() {
                            return Err(AnimationError::NonFiniteOutput(
                                "scale interpolation produced a non-finite value".to_string(),
                            ));
                        }
                        entry.2 = Some(v);
                    }
                }
            }
        }

        let mut result = HashMap::with_capacity(transforms.len());
        for (id, (t, r, s)) in transforms {
            let translation = t.unwrap_or(Vec3::ZERO);
            let rotation = r.unwrap_or(Quat::IDENTITY);
            let scale = s.unwrap_or(Vec3::ONE);

            if !translation.is_finite() || !scale.is_finite() || !rotation.is_finite() {
                return Err(AnimationError::NonFiniteOutput(format!(
                    "non-finite transform for node (slot={}, gen={})",
                    id.slot, id.generation
                )));
            }

            result.insert(
                id,
                Mat4::from_scale_rotation_translation(scale, rotation, translation),
            );
        }

        Ok(result)
    }
}

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Interpolate a sampler's value at a given time.
fn interpolate_sampler(
    sampler: &AnimationSampler,
    time: f32,
) -> Result<KeyframeValue, AnimationError> {
    if !time.is_finite() {
        return Err(AnimationError::InvalidTimestamp(format!(
            "evaluation time {time} is non-finite"
        )));
    }

    if sampler.input.is_empty() || sampler.output.is_empty() {
        return sampler.output.first().cloned().ok_or_else(|| {
            AnimationError::InvalidSampler("sampler has no output values".to_string())
        });
    }

    match sampler.interpolation {
        Interpolation::Step => interpolate_step(sampler, time),
        Interpolation::Linear => interpolate_linear(sampler, time),
        Interpolation::CubicSpline => interpolate_cubic_spline(sampler, time),
    }
}

/// Step interpolation: return the value at the most recent key ≤ time.
fn interpolate_step(
    sampler: &AnimationSampler,
    time: f32,
) -> Result<KeyframeValue, AnimationError> {
    let idx = match sampler
        .input
        .binary_search_by(|t| t.partial_cmp(&time).unwrap_or(std::cmp::Ordering::Less))
    {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) => i - 1,
    };
    Ok(sampler.output[idx].clone())
}

/// Linear interpolation: lerp for translation/scale, shortest-path slerp
/// for quaternion rotation.
fn interpolate_linear(
    sampler: &AnimationSampler,
    time: f32,
) -> Result<KeyframeValue, AnimationError> {
    let (t0, t1, v0, v1) = find_surrounding_keys(sampler, time)?;
    if (t1 - t0).abs() < f32::EPSILON {
        return Ok(v0.clone());
    }
    let factor = ((time - t0) / (t1 - t0)).clamp(0.0, 1.0);

    match (v0, v1) {
        (KeyframeValue::Translation(va), KeyframeValue::Translation(vb)) => {
            Ok(KeyframeValue::Translation(va.lerp(*vb, factor)))
        }
        (KeyframeValue::Rotation(qa), KeyframeValue::Rotation(qb)) => {
            let qa = normalize_quat(*qa, "linear rotation start")?;
            let qb = normalize_quat(*qb, "linear rotation end")?;
            Ok(KeyframeValue::Rotation(shortest_path_slerp(qa, qb, factor)?))
        }
        (KeyframeValue::Scale(va), KeyframeValue::Scale(vb)) => {
            Ok(KeyframeValue::Scale(va.lerp(*vb, factor)))
        }
        _ => Err(AnimationError::CardinalityMismatch(
            "output type mismatch in linear interpolation".to_string(),
        )),
    }
}

/// Cubic spline interpolation using the glTF Hermite formulation.
///
/// Output layout for cubic spline: for each of the N input keys there are
/// three output elements: in-tangent, value, out-tangent.
///
/// For two adjacent keys at t_k and t_{k+1}:
///   p0 = value at t_k
///   m0 = out-tangent at t_k (scaled by (t_{k+1} - t_k))
///   p1 = value at t_{k+1}
///   m1 = in-tangent at t_{k+1} (scaled by (t_{k+1} - t_k))
///
/// Hermite basis:
///   H00 =  2t³ - 3t² + 1
///   H10 =   t³ - 2t² + t
///   H01 = -2t³ + 3t²
///   H11 =   t³ -  t²
fn interpolate_cubic_spline(
    sampler: &AnimationSampler,
    time: f32,
) -> Result<KeyframeValue, AnimationError> {
    if sampler.output.len() < sampler.input.len() * 3 {
        return Err(AnimationError::CardinalityMismatch(format!(
            "cubic spline sampler has {} outputs for {} inputs (need {} = inputs*3)",
            sampler.output.len(),
            sampler.input.len(),
            sampler.input.len() * 3
        )));
    }

    let idx = match sampler
        .input
        .binary_search_by(|t| t.partial_cmp(&time).unwrap_or(std::cmp::Ordering::Less))
    {
        Ok(i) => i.saturating_sub(1).min(sampler.input.len().saturating_sub(2)),
        Err(0) => 0,
        Err(i) => (i - 1).min(sampler.input.len().saturating_sub(2)),
    };

    let t0 = sampler.input[idx];
    let t1 = sampler.input[(idx + 1).min(sampler.input.len() - 1)];

    if (t1 - t0).abs() < f32::EPSILON {
        // Return the value at t0 (middle element of the triple).
        return Ok(sampler.output[idx * 3 + 1].clone());
    }

    let dt = t1 - t0;
    let t = ((time - t0) / dt).clamp(0.0, 1.0);

    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    // For key k: output[3k] = in-tangent, output[3k+1] = value, output[3k+2] = out-tangent
    let p0 = sampler.output[idx * 3 + 1].clone(); // value at k
    let m0 = sampler.output[idx * 3 + 2].clone(); // out-tangent at k
    let p1 = sampler.output[(idx + 1) * 3 + 1].clone(); // value at k+1
    let m1 = sampler.output[(idx + 1) * 3].clone(); // in-tangent at k+1

    match (&p0, &m0, &p1, &m1) {
        (
            KeyframeValue::Translation(v0),
            KeyframeValue::Translation(tan_out),
            KeyframeValue::Translation(v1),
            KeyframeValue::Translation(tan_in),
        ) => {
            let m0_scaled = *tan_out * dt;
            let m1_scaled = *tan_in * dt;
            let result = *v0 * h00 + m0_scaled * h10 + *v1 * h01 + m1_scaled * h11;
            Ok(KeyframeValue::Translation(result))
        }
        (
            KeyframeValue::Scale(v0),
            KeyframeValue::Scale(tan_out),
            KeyframeValue::Scale(v1),
            KeyframeValue::Scale(tan_in),
        ) => {
            let m0_scaled = *tan_out * dt;
            let m1_scaled = *tan_in * dt;
            let result = *v0 * h00 + m0_scaled * h10 + *v1 * h01 + m1_scaled * h11;
            Ok(KeyframeValue::Scale(result))
        }
        (
            KeyframeValue::Rotation(v0_q),
            KeyframeValue::Rotation(tan_out_q),
            KeyframeValue::Rotation(v1_q),
            KeyframeValue::Rotation(tan_in_q),
        ) => {
            // Reconstruct Quat from components to avoid glam Deref ambiguity.
            let v0_n = normalize_quat(*v0_q, "cubic rotation start")?;
            let v1_n = normalize_quat(*v1_q, "cubic rotation end")?;
            let m0_scaled = glam::Quat::from_xyzw(
                tan_out_q.x * dt,
                tan_out_q.y * dt,
                tan_out_q.z * dt,
                tan_out_q.w * dt,
            );
            let m1_scaled = glam::Quat::from_xyzw(
                tan_in_q.x * dt,
                tan_in_q.y * dt,
                tan_in_q.z * dt,
                tan_in_q.w * dt,
            );

            // Hermite blend in Vec4 space, then normalize.
            let v0v: glam::Vec4 = glam::Vec4::new(v0_n.x, v0_n.y, v0_n.z, v0_n.w);
            let v1v: glam::Vec4 = glam::Vec4::new(v1_n.x, v1_n.y, v1_n.z, v1_n.w);
            let m0v: glam::Vec4 =
                glam::Vec4::new(m0_scaled.x, m0_scaled.y, m0_scaled.z, m0_scaled.w);
            let m1v: glam::Vec4 =
                glam::Vec4::new(m1_scaled.x, m1_scaled.y, m1_scaled.z, m1_scaled.w);
            let blended = v0v * h00 + m0v * h10 + v1v * h01 + m1v * h11;
            let q0 = glam::Quat::from_xyzw(blended.x, blended.y, blended.z, blended.w);
            let q0 = match normalize_quat(q0, "cubic rotation output") {
                Ok(q) => q,
                Err(_) => shortest_path_slerp(v0_n, v1_n, t)?,
            };

            Ok(KeyframeValue::Rotation(q0))
        }
        _ => Err(AnimationError::CardinalityMismatch(
            "output type mismatch in cubic spline interpolation".to_string(),
        )),
    }
}

/// Find the pair of surrounding key indices for a given time.
fn find_surrounding_keys(
    sampler: &AnimationSampler,
    time: f32,
) -> Result<(f32, f32, &KeyframeValue, &KeyframeValue), AnimationError> {
    let idx = match sampler
        .input
        .binary_search_by(|t| t.partial_cmp(&time).unwrap_or(std::cmp::Ordering::Less))
    {
        Ok(i) => i.saturating_sub(1).min(sampler.input.len().saturating_sub(2)),
        Err(0) => 0,
        Err(i) => (i - 1).min(sampler.input.len().saturating_sub(2)),
    };

    let next = (idx + 1).min(sampler.input.len() - 1);
    let t0 = sampler.input[idx];
    let t1 = sampler.input[next];
    let v0 = &sampler.output[idx];
    let v1 = &sampler.output[next];
    Ok((t0, t1, v0, v1))
}

fn keyframe_value_is_valid(value: &KeyframeValue) -> bool {
    match value {
        KeyframeValue::Translation(v) | KeyframeValue::Scale(v) => v.is_finite(),
        KeyframeValue::Rotation(q) => q.is_finite() && q.length_squared() > 1e-12,
    }
}

fn normalize_quat(q: Quat, context: &str) -> Result<Quat, AnimationError> {
    if !q.is_finite() || q.length_squared() <= 1e-12 {
        return Err(AnimationError::NonFiniteOutput(format!(
            "{context} quaternion is non-finite or zero-length"
        )));
    }
    let normalized = q.normalize();
    if !normalized.is_finite() {
        return Err(AnimationError::NonFiniteOutput(format!(
            "{context} quaternion normalization produced a non-finite value"
        )));
    }
    Ok(normalized)
}

/// Shortest-path quaternion slerp.
///
/// If the dot product of the two quaternions is negative, one is negated
/// to take the shorter arc. The result is normalized.
fn shortest_path_slerp(a: Quat, b: Quat, t: f32) -> Result<Quat, AnimationError> {
    let mut b = b;
    let mut dot = a.dot(b);

    // Take the shortest path.
    if dot < 0.0 {
        b = -b;
        dot = -dot;
    }

    // Clamp dot to [-1, 1] to avoid acos domain errors.
    dot = dot.clamp(-1.0, 1.0);

    if dot > 1.0 - f32::EPSILON {
        // Quaternions are nearly identical; use lerp.
        let result = a + (b - a) * t;
        return normalize_quat(result, "slerp lerp result");
    }

    let theta = dot.acos();
    let sin_theta = theta.sin();

    if sin_theta.abs() < f32::EPSILON {
        return normalize_quat(a, "slerp degenerate result");
    }

    let w_a = ((1.0 - t) * theta).sin() / sin_theta;
    let w_b = (t * theta).sin() / sin_theta;

    normalize_quat(a * w_a + b * w_b, "slerp result")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_clip_node_id() -> SceneNodeId {
        SceneNodeId::new(1, 0)
    }

    fn make_valid_targets() -> HashMap<SceneNodeId, ()> {
        let mut map = HashMap::new();
        map.insert(make_test_clip_node_id(), ());
        map
    }

    fn make_translation_sampler_linear(
        times: Vec<f32>,
        values: Vec<Vec3>,
    ) -> AnimationSampler {
        AnimationSampler {
            input: times,
            output: values.into_iter().map(KeyframeValue::Translation).collect(),
            interpolation: Interpolation::Linear,
        }
    }

    fn make_translation_sampler_step(
        times: Vec<f32>,
        values: Vec<Vec3>,
    ) -> AnimationSampler {
        AnimationSampler {
            input: times,
            output: values.into_iter().map(KeyframeValue::Translation).collect(),
            interpolation: Interpolation::Step,
        }
    }

    fn make_translation_sampler_cubic(
        times: Vec<f32>,
        // Triples: (in_tangent, value, out_tangent) for each key
        triples: Vec<(Vec3, Vec3, Vec3)>,
    ) -> AnimationSampler {
        let mut output = Vec::with_capacity(triples.len() * 3);
        for (tin, val, tout) in triples {
            output.push(KeyframeValue::Translation(tin));
            output.push(KeyframeValue::Translation(val));
            output.push(KeyframeValue::Translation(tout));
        }
        AnimationSampler {
            input: times,
            output,
            interpolation: Interpolation::CubicSpline,
        }
    }

    // ── Clip validation ──────────────────────────────────────────────

    #[test]
    fn clip_validation_rejects_empty_name() {
        let clip = AnimationClip::new("", 1.0);
        assert!(clip.validate().is_err());
    }

    #[test]
    fn clip_validation_rejects_non_finite_duration() {
        let clip = AnimationClip::new("test", f32::NAN);
        assert!(matches!(
            clip.validate(),
            Err(AnimationError::InvalidDuration(_))
        ));
    }

    #[test]
    fn clip_validation_rejects_out_of_bounds_sampler_index() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 5,
        });
        assert!(matches!(
            clip.validate(),
            Err(AnimationError::InvalidChannel(_))
        ));
    }

    #[test]
    fn clip_validation_rejects_non_monotonic_timestamps() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.samplers.push(AnimationSampler {
            input: vec![0.0, 0.5, 0.3],
            output: vec![
                KeyframeValue::Translation(Vec3::ZERO),
                KeyframeValue::Translation(Vec3::ONE),
                KeyframeValue::Translation(Vec3::new(2.0, 0.0, 0.0)),
            ],
            interpolation: Interpolation::Linear,
        });
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 0,
        });
        assert!(matches!(
            clip.validate(),
            Err(AnimationError::InvalidTimestamp(_))
        ));
    }

    #[test]
    fn clip_validation_rejects_cubic_spline_wrong_cardinality() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.samplers.push(AnimationSampler {
            input: vec![0.0, 1.0],
            output: vec![
                KeyframeValue::Translation(Vec3::ZERO),
                KeyframeValue::Translation(Vec3::ONE),
                // Missing 4 more for 2 keys * 3 = 6
            ],
            interpolation: Interpolation::CubicSpline,
        });
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 0,
        });
        assert!(matches!(
            clip.validate(),
            Err(AnimationError::CardinalityMismatch(_))
        ));
    }

    #[test]
    fn clip_validation_rejects_type_mismatch() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.samplers.push(AnimationSampler {
            input: vec![0.0, 1.0],
            output: vec![
                KeyframeValue::Rotation(Quat::IDENTITY),
                KeyframeValue::Rotation(Quat::IDENTITY),
            ],
            interpolation: Interpolation::Linear,
        });
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 0,
        });
        assert!(matches!(
            clip.validate(),
            Err(AnimationError::CardinalityMismatch(_))
        ));
    }

    #[test]
    fn valid_clip_passes_validation() {
        let mut clip = AnimationClip::new("valid", 2.0);
        clip.samplers.push(make_translation_sampler_linear(
            vec![0.0, 1.0, 2.0],
            vec![Vec3::ZERO, Vec3::ONE, Vec3::new(2.0, 0.0, 0.0)],
        ));
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 0,
        });
        clip.validate().unwrap();
    }

    // ── Step interpolation ───────────────────────────────────────────

    #[test]
    fn step_interpolation_returns_previous_key() {
        let sampler = make_translation_sampler_step(
            vec![0.0, 1.0, 2.0],
            vec![Vec3::ZERO, Vec3::ONE, Vec3::new(2.0, 0.0, 0.0)],
        );
        let v = interpolate_sampler(&sampler, 0.5).unwrap();
        assert_eq!(v, KeyframeValue::Translation(Vec3::ZERO));

        let v = interpolate_sampler(&sampler, 1.5).unwrap();
        assert_eq!(v, KeyframeValue::Translation(Vec3::ONE));

        let v = interpolate_sampler(&sampler, 0.0).unwrap();
        assert_eq!(v, KeyframeValue::Translation(Vec3::ZERO));
    }

    // ── Linear interpolation ─────────────────────────────────────────

    #[test]
    fn linear_translation_midpoint() {
        let sampler = make_translation_sampler_linear(
            vec![0.0, 2.0],
            vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)],
        );
        let v = interpolate_sampler(&sampler, 1.0).unwrap();
        assert_eq!(v, KeyframeValue::Translation(Vec3::new(5.0, 0.0, 0.0)));
    }

    #[test]
    fn linear_rotation_slerp_halfway() {
        let q0 = Quat::IDENTITY;
        let q1 = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);

        let sampler = AnimationSampler {
            input: vec![0.0, 1.0],
            output: vec![KeyframeValue::Rotation(q0), KeyframeValue::Rotation(q1)],
            interpolation: Interpolation::Linear,
        };

        let v = interpolate_sampler(&sampler, 0.5).unwrap();
        if let KeyframeValue::Rotation(q) = v {
            // Should be roughly 45 degrees around X.
            let (axis, angle) = q.to_axis_angle();
            assert!((angle - std::f32::consts::FRAC_PI_4).abs() < 0.01);
            assert!((axis.x - 1.0).abs() < 0.01);
        } else {
            panic!("expected rotation");
        }
    }

    #[test]
    fn linear_rotation_slerp_shortest_path() {
        let q0 = Quat::IDENTITY;
        // 270 degrees around Y (shortest path is -90).
        let q1 = Quat::from_rotation_y(std::f32::consts::PI * 1.5);

        let sampler = AnimationSampler {
            input: vec![0.0, 1.0],
            output: vec![KeyframeValue::Rotation(q0), KeyframeValue::Rotation(q1)],
            interpolation: Interpolation::Linear,
        };

        let v = interpolate_sampler(&sampler, 0.5).unwrap();
        if let KeyframeValue::Rotation(q) = v {
            // Should take the shorter path (~ -45 degrees around Y).
            let (axis, angle) = q.to_axis_angle();
            assert!(angle < std::f32::consts::FRAC_PI_2 + 0.1);
        } else {
            panic!("expected rotation");
        }
    }

    // ── Cubic spline interpolation ──────────────────────────────────

    #[test]
    fn cubic_spline_zero_tangents_is_hermite_smooth_step() {
        let sampler = make_translation_sampler_cubic(
            vec![0.0, 1.0],
            vec![
                (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO), // key 0: tangents 0
                (Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO), // key 1: tangents 0
            ],
        );

        let v = interpolate_sampler(&sampler, 0.0).unwrap();
        assert_eq!(v, KeyframeValue::Translation(Vec3::ZERO));

        let v = interpolate_sampler(&sampler, 0.5).unwrap();
        if let KeyframeValue::Translation(p) = v {
            assert!(p.x > 0.0 && p.x < 10.0);
        } else {
            panic!("expected translation");
        }

        let v = interpolate_sampler(&sampler, 1.0).unwrap();
        assert_eq!(v, KeyframeValue::Translation(Vec3::new(10.0, 0.0, 0.0)));
    }

    #[test]
    fn cubic_spline_hermite_midpoint_with_tangents() {
        // Two keys with known tangents. At t=0.5 the Hermite result
        // should match the standard formula.
        let sampler = make_translation_sampler_cubic(
            vec![0.0, 2.0],
            vec![
                (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
                (Vec3::new(-2.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)),
            ],
        );

        let v = interpolate_sampler(&sampler, 1.0).unwrap();
        if let KeyframeValue::Translation(p) = v {
            // dt=2. m0_scaled=(2,0,0)*2=4, m1_scaled=(-2,0,0)*2=-4.
            // At t=0.5: h00=0.5, h10=0.125, h01=0.5, h11=-0.125.
            // result = 0*0.5 + 4*0.125 + 4*0.5 + (-4)*(-0.125)
            //       = 0 + 0.5 + 2.0 + 0.5 = 3.0
            assert!((p.x - 3.0).abs() < 0.001, "got x={}", p.x);
        } else {
            panic!("expected translation");
        }
    }

    #[test]
    fn cubic_spline_rotation_normalized_output() {
        let q_ident = Quat::IDENTITY;
        let q_90 = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);

        let sampler = AnimationSampler {
            input: vec![0.0, 1.0],
            output: vec![
                KeyframeValue::Rotation(q_ident), // in-tan
                KeyframeValue::Rotation(q_ident), // value
                KeyframeValue::Rotation(q_ident), // out-tan
                KeyframeValue::Rotation(q_ident), // in-tan
                KeyframeValue::Rotation(q_90), // value
                KeyframeValue::Rotation(q_ident), // out-tan
            ],
            interpolation: Interpolation::CubicSpline,
        };

        let v = interpolate_sampler(&sampler, 0.5).unwrap();
        if let KeyframeValue::Rotation(q) = v {
            assert!(q.is_normalized());
            assert!(q.is_finite());
        } else {
            panic!("expected rotation");
        }
    }

    // ── Player tests ─────────────────────────────────────────────────

    #[test]
    fn player_advances_time_correctly() {
        let mut clip = AnimationClip::new("advance", 2.0);
        clip.samplers.push(make_translation_sampler_linear(
            vec![0.0, 2.0],
            vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)],
        ));
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 0,
        });

        let mut player = AnimationPlayer::new();
        player.set_clip(clip).unwrap();
        player.play();

        let targets = make_valid_targets();
        let result = player.update(1.0, &targets).unwrap();
        assert_eq!(player.current_time(), 1.0);
        // At t=1.0, should be midpoint.
        let m = result[&make_test_clip_node_id()];
        let (_, _, t) = m.to_scale_rotation_translation();
        assert!((t.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn player_stops_at_clip_end() {
        let mut clip = AnimationClip::new("finite", 2.0);
        clip.samplers.push(make_translation_sampler_linear(
            vec![0.0, 2.0],
            vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)],
        ));
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 0,
        });

        let mut player = AnimationPlayer::new();
        player.set_clip(clip).unwrap();
        player.play();

        let targets = make_valid_targets();
        let _ = player.update(3.0, &targets).unwrap();
        assert!(!player.is_playing());
        assert_eq!(player.current_time(), 2.0);
    }

    #[test]
    fn player_loops_correctly() {
        let mut clip = AnimationClip::new("loop", 2.0);
        clip.samplers.push(make_translation_sampler_linear(
            vec![0.0, 2.0],
            vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)],
        ));
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 0,
        });

        let mut player = AnimationPlayer::new();
        player.set_clip(clip).unwrap();
        player.set_looping(true);
        player.play();

        let targets = make_valid_targets();
        let _ = player.update(3.0, &targets).unwrap();
        assert!(player.is_playing());
        assert_eq!(player.current_time(), 1.0);
    }

    #[test]
    fn player_rejects_stale_target() {
        let mut clip = AnimationClip::new("stale", 1.0);
        clip.samplers.push(make_translation_sampler_linear(
            vec![0.0, 1.0],
            vec![Vec3::ZERO, Vec3::ONE],
        ));
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 0,
        });

        let mut player = AnimationPlayer::new();
        player.set_clip(clip).unwrap();
        player.play();

        // Empty targets — the node ID is not valid.
        let empty_targets = HashMap::new();
        let result = player.update(0.1, &empty_targets);
        assert!(matches!(result, Err(AnimationError::StaleTarget(_))));
    }

    #[test]
    fn player_rejects_negative_dt() {
        let mut clip = AnimationClip::new("neg_dt", 1.0);
        clip.samplers.push(make_translation_sampler_linear(
            vec![0.0, 1.0],
            vec![Vec3::ZERO, Vec3::ONE],
        ));
        clip.channels.push(AnimationChannel {
            target: make_test_clip_node_id(),
            target_path: AnimationTarget::Translation,
            sampler_index: 0,
        });

        let mut player = AnimationPlayer::new();
        player.set_clip(clip).unwrap();
        player.play();

        let targets = make_valid_targets();
        let result = player.update(-0.1, &targets);
        assert!(matches!(result, Err(AnimationError::InvalidTimestamp(_))));
    }

    // ── Shortest-path slerp ──────────────────────────────────────────

    #[test]
    fn shortest_path_slerp_identity_to_identity() {
        let result = shortest_path_slerp(Quat::IDENTITY, Quat::IDENTITY, 0.5).unwrap();
        assert!(result.is_normalized());
        assert!((result.w - 1.0).abs() < 0.001);
    }

    #[test]
    fn shortest_path_slerp_handles_negative_dot() {
        let a = Quat::IDENTITY;
        let b = -Quat::from_rotation_y(std::f32::consts::FRAC_PI_2); // negated version
        let result = shortest_path_slerp(a, b, 0.0).unwrap();
        // At t=0, should be a (identity).
        assert!((result.w - 1.0).abs() < 0.01);
    }
}
