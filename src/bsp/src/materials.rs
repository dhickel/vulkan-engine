//! Surface material classification and animated texture cycles.
//!
//! Contract: `bsp-renderer-lighting.md` §3, §5.

use crate::lumps;

/// Texinfo flags from Quake BSP.
pub mod tex_flags {
    pub const TEX_SPECIAL: u32 = 0x01;
    pub const SURF_SKY: u32 = 0x04;
    pub const SURF_WARP: u32 = 0x08;
    pub const SURF_TRANS33: u32 = 0x10;
    pub const SURF_TRANS66: u32 = 0x20;
    pub const SURF_FLOWING: u32 = 0x40;
    pub const SURF_NODRAW: u32 = 0x80;
}

/// Classification of a BSP surface for rendering and collision.
///
/// Fullbright is per-pixel via the emissive mask, not a separate surface class.
/// All lightmapped surfaces (Opaque) carry a fullbright mask overlaying the lightmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceClass {
    /// Standard opaque surface — rendered with lightmap + fullbright mask overlay.
    Opaque,
    /// Alpha-mask surface — alpha-tested (fences, grates).
    AlphaMask,
    /// Sky surface — depth-preserving, no lightmap.
    Sky,
    /// Liquid/warp surface — animated, blended, two-sided.
    Liquid,
    /// Nodraw surface — no rendering.
    NoDraw,
    /// Clip surface — no rendering, collision only.
    Clip,
    /// Trigger surface — no rendering, trigger sensor.
    Trigger,
    /// Skip/hint surface — no rendering, no collision.
    Skip,
}

impl SurfaceClass {
    /// Whether this surface class contributes to rendering.
    pub fn is_visible(self) -> bool {
        !matches!(self, SurfaceClass::NoDraw | SurfaceClass::Clip | SurfaceClass::Trigger | SurfaceClass::Skip)
    }

    /// Whether this surface class generates a collider.
    pub fn contributes_collision(self) -> bool {
        matches!(self, SurfaceClass::Opaque | SurfaceClass::AlphaMask)
    }

    /// Whether this surface class is a trigger sensor.
    pub fn is_trigger(self) -> bool {
        matches!(self, SurfaceClass::Trigger)
    }

    /// Get the render class for batch grouping.
    pub fn render_class(self) -> crate::geometry::RenderClass {
        match self {
            SurfaceClass::Opaque => crate::geometry::RenderClass::Opaque,
            SurfaceClass::AlphaMask => crate::geometry::RenderClass::AlphaMask,
            SurfaceClass::Sky => crate::geometry::RenderClass::Sky,
            SurfaceClass::Liquid => crate::geometry::RenderClass::Liquid,
            SurfaceClass::NoDraw | SurfaceClass::Clip | SurfaceClass::Trigger | SurfaceClass::Skip => {
                crate::geometry::RenderClass::Hidden
            }
        }
    }
}

/// Classify a surface by its texinfo flags and texture name.
///
/// Texture name conventions (Quake standard) take precedence over flags in
/// some cases, but both are checked for defense-in-depth.
pub fn classify_surface(tex_flags: u32, texture_name: &str) -> SurfaceClass {
    let name_lower = texture_name.to_ascii_lowercase();

    // Check texture name conventions first
    if name_lower.starts_with("sky") || name_lower.starts_with("*sky") {
        return SurfaceClass::Sky;
    }

    if name_lower == "clip" {
        return SurfaceClass::Clip;
    }
    if name_lower == "trigger" {
        return SurfaceClass::Trigger;
    }
    if name_lower == "skip" || name_lower == "hint" || name_lower == "origin" {
        return SurfaceClass::Skip;
    }
    if name_lower == "nodraw" || name_lower == "null" {
        return SurfaceClass::NoDraw;
    }

    // Liquid surfaces: *water, *slime, *lava, *04water, etc.
    if name_lower.starts_with('*')
        && (name_lower.contains("water")
            || name_lower.contains("slime")
            || name_lower.contains("lava")
            || name_lower.contains("tele"))
    {
        return SurfaceClass::Liquid;
    }

    // Alpha-mask: `{` prefix (Quake convention)
    if name_lower.starts_with('{') {
        return SurfaceClass::AlphaMask;
    }

    // Check texinfo flags
    if tex_flags & tex_flags::SURF_SKY != 0 {
        return SurfaceClass::Sky;
    }
    if tex_flags & tex_flags::SURF_NODRAW != 0 {
        return SurfaceClass::NoDraw;
    }
    if tex_flags & tex_flags::SURF_WARP != 0 {
        return SurfaceClass::Liquid;
    }

    SurfaceClass::Opaque
}

/// Check whether palette index is in the fullbright range.
///
/// Default: indices 224–255 (last 32 colors).
pub fn is_fullbright(palette_index: u8, fullbright_start: u8, fullbright_end: u8) -> bool {
    palette_index >= fullbright_start && palette_index <= fullbright_end
}

/// An animated texture cycle.
#[derive(Debug, Clone)]
pub struct AnimatedTexture {
    /// Base texture name (without numeric prefix).
    pub base_name: String,
    /// Frame texture names in order.
    pub frames: Vec<String>,
    /// Frame duration in seconds (default 0.1).
    pub frame_duration: f32,
    /// Cycle type.
    pub cycle_type: AnimationCycleType,
}

/// Type of animation cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationCycleType {
    /// Sequential numeric: +0name, +1name, +2name, ...
    Sequential,
    /// Alternating: +aname, +bname cycle.
    Alternate,
}

/// Detect and build an animation cycle for a texture name.
///
/// Returns None if the texture name does not indicate animation.
pub fn detect_animation(texture_name: &str, available_textures: &[String]) -> Option<AnimatedTexture> {
    // Quake convention: `+<N><name>` for sequential, `+<a|b><name>` for alternate
    if !texture_name.starts_with('+') || texture_name.len() < 3 {
        return None;
    }

    let after_plus = &texture_name[1..];
    let first_char = after_plus.chars().next()?;

    if first_char.is_ascii_digit() {
        // Sequential: +0name, +1name, +2name, ...
        // Extract the number and base name
        let num_end = after_plus
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after_plus.len());
        let base_name: String = after_plus[num_end..].to_string();
        if base_name.is_empty() {
            return None;
        }

        let mut frames = Vec::new();
        for i in 0u32.. {
            let frame_name = format!("+{}{}", i, base_name);
            if available_textures.iter().any(|t| t == &frame_name) {
                frames.push(frame_name);
            } else {
                break;
            }
        }
        if frames.len() < 2 {
            return None;
        }
        Some(AnimatedTexture {
            base_name,
            frames,
            frame_duration: 0.1,
            cycle_type: AnimationCycleType::Sequential,
        })
    } else if first_char == 'a' || first_char == 'b' {
        // Alternate: +aname, +bname
        let base_name: String = after_plus[1..].to_string();
        if base_name.is_empty() {
            return None;
        }
        let frame_a = format!("+a{}", base_name);
        let frame_b = format!("+b{}", base_name);
        let has_a = available_textures.iter().any(|t| t == &frame_a);
        let has_b = available_textures.iter().any(|t| t == &frame_b);
        if !has_a || !has_b {
            return None;
        }
        Some(AnimatedTexture {
            base_name,
            frames: vec![frame_a, frame_b],
            frame_duration: 0.1,
            cycle_type: AnimationCycleType::Alternate,
        })
    } else {
        None
    }
}

/// Compute the current frame index for an animation cycle given engine time.
pub fn animation_frame_index(
    cycle: &AnimatedTexture,
    engine_time_ticks: f32,
) -> usize {
    let frame_count = cycle.frames.len();
    if frame_count == 0 {
        return 0;
    }
    let ticks = engine_time_ticks.max(0.0);
    match cycle.cycle_type {
        AnimationCycleType::Sequential => {
            (ticks / cycle.frame_duration) as usize % frame_count
        }
        AnimationCycleType::Alternate => {
            (ticks / cycle.frame_duration) as usize % 2
        }
    }
}

/// Resolve material identity from texture and surface class.
///
/// Material identity is a deterministic u64 key for batch grouping.
/// It encodes the texture reference and surface class.
pub fn material_identity(texture_index: u32, surface_class: SurfaceClass) -> u64 {
    let class_bits = (surface_class as u64) & 0xFF;
    ((texture_index as u64) << 8) | class_bits
}

/// BSP material descriptor — the neutral ABI between extraction and renderer.
///
/// Each material associates a resolved texture, fullbright mask, lightmap page,
/// surface classification, animation data, and rendering metadata.
#[derive(Debug, Clone)]
pub struct BspMaterial {
    /// Material index (stable within an extraction).
    pub material_index: u32,
    /// Resolved texture identity (name + source).
    pub texture_identity: String,
    /// Whether this material has an associated fullbright mask.
    pub has_fullbright_mask: bool,
    /// Fullbright mask dimensions (width, height), same as texture dimensions.
    pub fullbright_mask_dims: (u32, u32),
    /// Lightmap atlas page for this material (u32::MAX = no lightmap).
    pub lightmap_page: u32,
    /// Surface classification.
    pub surface_class: SurfaceClass,
    /// Whether this surface is alpha-masked (palette index 255 treated as alpha).
    pub has_alpha_mask: bool,
    /// Whether this surface uses warp animation (liquid).
    pub has_warp: bool,
    /// Whether this surface uses flowing animation.
    pub has_flow: bool,
    /// Transparency flags from texinfo.
    pub trans33: bool,
    pub trans66: bool,
    /// Associated animation cycle (if any).
    pub animation: Option<AnimatedTexture>,
    /// Overbright factor from calibration.
    pub overbright: f32,
    /// Linear light scale from calibration.
    pub light_scale: f32,
    /// Receive mask defaults.
    pub receive_ibl: bool,
    pub receive_csm: bool,
    pub receive_dynamic: bool,
}

impl Default for BspMaterial {
    fn default() -> Self {
        BspMaterial {
            material_index: 0,
            texture_identity: String::new(),
            has_fullbright_mask: false,
            fullbright_mask_dims: (0, 0),
            lightmap_page: u32::MAX,
            surface_class: SurfaceClass::Opaque,
            has_alpha_mask: false,
            has_warp: false,
            has_flow: false,
            trans33: false,
            trans66: false,
            animation: None,
            overbright: 2.0,
            light_scale: 1.0,
            receive_ibl: false,
            receive_csm: false,
            receive_dynamic: true,
        }
    }
}

/// Collect unique surface classes from a set of faces.
pub fn classify_faces(
    texinfos: &[lumps::Texinfo],
    texture_names: &[String], // per-texinfo texture name
    faces: &[lumps::Face],
) -> Vec<SurfaceClass> {
    faces
        .iter()
        .map(|face| {
            let ti = texinfos.get(face.texinfo_id as usize);
            let name = ti
                .and_then(|t| texture_names.get(t.miptex as usize))
                .map(|s| s.as_str())
                .unwrap_or("");
            let flags = ti.map(|t| t.flags).unwrap_or(0);
            classify_surface(flags, name)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_sky_by_name() {
        assert_eq!(classify_surface(0, "sky4"), SurfaceClass::Sky);
        assert_eq!(classify_surface(0, "*sky1"), SurfaceClass::Sky);
    }

    #[test]
    fn classify_liquid_by_name() {
        assert_eq!(classify_surface(0, "*water01"), SurfaceClass::Liquid);
        assert_eq!(classify_surface(0, "*slime"), SurfaceClass::Liquid);
        assert_eq!(classify_surface(0, "*lava"), SurfaceClass::Liquid);
    }

    #[test]
    fn classify_clip_trigger_skip() {
        assert_eq!(classify_surface(0, "clip"), SurfaceClass::Clip);
        assert_eq!(classify_surface(0, "trigger"), SurfaceClass::Trigger);
        assert_eq!(classify_surface(0, "skip"), SurfaceClass::Skip);
        assert_eq!(classify_surface(0, "hint"), SurfaceClass::Skip);
    }

    #[test]
    fn classify_nodraw() {
        assert_eq!(classify_surface(0, "nodraw"), SurfaceClass::NoDraw);
        assert_eq!(classify_surface(0, "null"), SurfaceClass::NoDraw);
    }

    #[test]
    fn classify_alpha_mask() {
        assert_eq!(classify_surface(0, "{fence"), SurfaceClass::AlphaMask);
    }

    #[test]
    fn classify_sky_by_flag() {
        assert_eq!(
            classify_surface(tex_flags::SURF_SKY, "wall1"),
            SurfaceClass::Sky
        );
    }

    #[test]
    fn classify_nodraw_by_flag() {
        assert_eq!(
            classify_surface(tex_flags::SURF_NODRAW, "wall1"),
            SurfaceClass::NoDraw
        );
    }

    #[test]
    fn classify_warp_by_flag() {
        assert_eq!(
            classify_surface(tex_flags::SURF_WARP, "wall1"),
            SurfaceClass::Liquid
        );
    }

    #[test]
    fn classify_default_opaque() {
        assert_eq!(classify_surface(0, "wall1"), SurfaceClass::Opaque);
    }

    #[test]
    fn detect_sequential_animation() {
        let available = vec![
            "+0test".to_string(),
            "+1test".to_string(),
            "+2test".to_string(),
        ];
        let anim = detect_animation("+0test", &available).unwrap();
        assert_eq!(anim.frames.len(), 3);
        assert_eq!(anim.cycle_type, AnimationCycleType::Sequential);
    }

    #[test]
    fn detect_alternate_animation() {
        let available = vec![
            "+awall".to_string(),
            "+bwall".to_string(),
        ];
        let anim = detect_animation("+awall", &available).unwrap();
        assert_eq!(anim.frames.len(), 2);
        assert_eq!(anim.cycle_type, AnimationCycleType::Alternate);
    }

    #[test]
    fn no_animation_without_alternates() {
        let available = vec!["+awall".to_string()];
        assert!(detect_animation("+awall", &available).is_none());
    }

    #[test]
    fn animation_frame_index_wraps() {
        let cycle = AnimatedTexture {
            base_name: "test".into(),
            frames: vec!["+0test".into(), "+1test".into(), "+2test".into()],
            frame_duration: 0.1,
            cycle_type: AnimationCycleType::Sequential,
        };
        assert_eq!(animation_frame_index(&cycle, 0.0), 0);
        assert_eq!(animation_frame_index(&cycle, 0.1), 1);
        assert_eq!(animation_frame_index(&cycle, 0.2), 2);
        assert_eq!(animation_frame_index(&cycle, 0.3), 0);
    }

    #[test]
    fn fullbright_range() {
        assert!(is_fullbright(224, 224, 255));
        assert!(is_fullbright(255, 224, 255));
        assert!(!is_fullbright(0, 224, 255));
        assert!(!is_fullbright(223, 224, 255));
    }

    #[test]
    fn surface_class_visibility() {
        assert!(SurfaceClass::Opaque.is_visible());
        assert!(SurfaceClass::Sky.is_visible());
        assert!(!SurfaceClass::Clip.is_visible());
        assert!(!SurfaceClass::NoDraw.is_visible());
    }

    #[test]
    fn surface_class_collision() {
        assert!(SurfaceClass::Opaque.contributes_collision());
        assert!(SurfaceClass::AlphaMask.contributes_collision());
        assert!(!SurfaceClass::Sky.contributes_collision());
        assert!(!SurfaceClass::Liquid.contributes_collision());
        assert!(!SurfaceClass::NoDraw.contributes_collision());
    }
}
