//! ThemeDefinition: semantic role → WAD identity + companion filename mapping.
//!
//! Maps each of the nine semantic roles (wall, floor, ceiling, accent, portal,
//! vertical, cave, prop, emissive) to exact lowercase-ASCII WAD identities and
//! `_basecolor.png` / `_norm.png` / `_gloss.png` companion filenames for each
//! of the three Richness V1 themes (Ancient, Egyptian, Brutalist).
//!
//! # Isolation rule
//!
//! Theme selection is crate-private and is never exposed through blueprint,
//! topology, or generation APIs. The mapping is a sealed enumeration that the
//! Richness pipeline references internally; callers outside the `richness`
//! module cannot inspect or alter theme identity.
//!
//! # Case stability
//!
//! Every WAD identity and companion filename in this module is a Rust `&str`
//! literal. The compiler guarantees exact-case stability: there are no format
//! strings, no runtime concatenation, and no ASCII-insensitive fallback paths.
//! The validation layer in `engine_pack::richness_assets` enforces exact-case
//! filename matching for every asset on disk.

// ── Semantic role ──────────────────────────────────────────────────────────

/// Semantic surface role — exactly the nine roles declared in each theme's
/// `theme.toml` `[roles]` section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SemanticRole {
    Wall,
    Floor,
    Ceiling,
    Accent,
    Portal,
    Vertical,
    Cave,
    Prop,
    Emissive,
}

impl SemanticRole {
    /// Return the lowercase ASCII WAD identity for this role.
    ///
    /// Every theme uses the same identity strings because the WAD entries
    /// are self-contained within the theme archive — no cross-theme identity
    /// collision is possible.
    pub fn wad_identity(self) -> &'static str {
        match self {
            Self::Wall => "wall",
            Self::Floor => "floor",
            Self::Ceiling => "ceiling",
            Self::Accent => "accent",
            Self::Portal => "portal",
            Self::Vertical => "vertical",
            Self::Cave => "cave",
            Self::Prop => "prop",
            Self::Emissive => "emissive",
        }
    }

    /// Ordered slice of all nine roles.
    pub fn all() -> &'static [SemanticRole] {
        ALL_ROLES
    }
}

/// Static slice of all nine semantic roles.
const ALL_ROLES: &[SemanticRole] = &[
    SemanticRole::Wall,
    SemanticRole::Floor,
    SemanticRole::Ceiling,
    SemanticRole::Accent,
    SemanticRole::Portal,
    SemanticRole::Vertical,
    SemanticRole::Cave,
    SemanticRole::Prop,
    SemanticRole::Emissive,
];

// ── Companion file suffix constants ────────────────────────────────────────

const BASECOLOR_SUFFIX: &str = "_basecolor.png";
const NORM_SUFFIX: &str = "_norm.png";
const GLOSS_SUFFIX: &str = "_gloss.png";

// ── Theme definition ───────────────────────────────────────────────────────

/// Sealed mapping from semantic roles to exact WAD identities and companion
/// PNG filenames for a single theme.
///
/// Every `ThemeDefinition` is constructed once and is immutable. The identity
/// strings are embedded in the binary as compile-time constants.
#[derive(Debug, Clone)]
pub struct ThemeDefinition {
    /// Display name (e.g. "Richness Ancient v1").
    pub name: &'static str,
    /// Theme directory name within `src/bsp_generator/themes/`.
    pub dir_name: &'static str,
    /// WAD filename (basename only, e.g. "richness_ancient_v1.wad").
    pub wad_filename: &'static str,
    /// Palette filename (basename only, always "palette.lmp").
    pub palette_filename: &'static str,
    /// License type (always "CC0-1.0").
    pub license: &'static str,
    /// Texture dimension in pixels (always 256).
    pub texture_size: u32,
    /// Master seed used by build.py.
    pub master_seed: u32,
    /// Expected PNG count (27: 9 roles × 3 files).
    pub png_count: usize,
    /// Expected palette size in bytes (768).
    pub palette_size: usize,
    /// Expected WAD entry count (10: 9 roles + "skip").
    pub wad_entry_count: usize,
    /// Semantic roles present in this theme.
    pub roles: &'static [SemanticRole],
    /// Per-role companion filenames relative to `textures/`.
    roles_companions: &'static [(&'static str, &'static str, &'static str, &'static str)],
}

impl ThemeDefinition {
    /// Return the basecolor companion filename for a role (e.g. "wall_basecolor.png").
    pub fn basecolor_filename(&self, role: SemanticRole) -> &'static str {
        self.lookup_companion(role).1
    }

    /// Return the normal companion filename for a role (e.g. "wall_norm.png").
    pub fn norm_filename(&self, role: SemanticRole) -> &'static str {
        self.lookup_companion(role).2
    }

    /// Return the gloss companion filename for a role (e.g. "wall_gloss.png").
    pub fn gloss_filename(&self, role: SemanticRole) -> &'static str {
        self.lookup_companion(role).3
    }

    /// Return all three companion filenames for a role.
    pub fn companion_filenames(
        &self,
        role: SemanticRole,
    ) -> (&'static str, &'static str, &'static str) {
        let entry = self.lookup_companion(role);
        (entry.1, entry.2, entry.3)
    }

    /// Return all expected output filenames for this theme (PNGs only, no WAD/palette/static).
    pub fn all_png_filenames(&self) -> Vec<&'static str> {
        let mut out = Vec::with_capacity(self.png_count);
        for (_role, basecolor, norm, gloss) in self.roles_companions {
            out.push(*basecolor);
            out.push(*norm);
            out.push(*gloss);
        }
        out.sort_unstable();
        out
    }

    /// Return all declared WAD identities (including "skip").
    pub fn all_wad_identities(&self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = self.roles.iter().map(|r| r.wad_identity()).collect();
        out.push("skip");
        out.sort_unstable();
        out
    }

    fn lookup_companion(
        &self,
        role: SemanticRole,
    ) -> &(&'static str, &'static str, &'static str, &'static str) {
        let wad_id = role.wad_identity();
        self.roles_companions
            .iter()
            .find(|(identity, _, _, _)| *identity == wad_id)
            .expect("ThemeDefinition: role not found in companion table")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Theme constants
// ═══════════════════════════════════════════════════════════════════════════

/// Richness Ancient v1 theme definition.
pub static THEME_ANCIENT: ThemeDefinition = ThemeDefinition {
    name: "Richness Ancient v1",
    dir_name: "richness_ancient_v1",
    wad_filename: "richness_ancient_v1.wad",
    palette_filename: "palette.lmp",
    license: "CC0-1.0",
    texture_size: 256,
    master_seed: 0x5241_5631,
    png_count: 27,
    palette_size: 768,
    wad_entry_count: 10,
    roles: ALL_ROLES,
    roles_companions: &[
        (
            "wall",
            "wall_basecolor.png",
            "wall_norm.png",
            "wall_gloss.png",
        ),
        (
            "floor",
            "floor_basecolor.png",
            "floor_norm.png",
            "floor_gloss.png",
        ),
        (
            "ceiling",
            "ceiling_basecolor.png",
            "ceiling_norm.png",
            "ceiling_gloss.png",
        ),
        (
            "accent",
            "accent_basecolor.png",
            "accent_norm.png",
            "accent_gloss.png",
        ),
        (
            "portal",
            "portal_basecolor.png",
            "portal_norm.png",
            "portal_gloss.png",
        ),
        (
            "vertical",
            "vertical_basecolor.png",
            "vertical_norm.png",
            "vertical_gloss.png",
        ),
        (
            "cave",
            "cave_basecolor.png",
            "cave_norm.png",
            "cave_gloss.png",
        ),
        (
            "prop",
            "prop_basecolor.png",
            "prop_norm.png",
            "prop_gloss.png",
        ),
        (
            "emissive",
            "emissive_basecolor.png",
            "emissive_norm.png",
            "emissive_gloss.png",
        ),
    ],
};

/// Richness Egyptian v1 theme definition.
pub static THEME_EGYPTIAN: ThemeDefinition = ThemeDefinition {
    name: "Richness Egyptian v1",
    dir_name: "richness_egyptian_v1",
    wad_filename: "richness_egyptian_v1.wad",
    palette_filename: "palette.lmp",
    license: "CC0-1.0",
    texture_size: 256,
    master_seed: 0x5245_5631,
    png_count: 27,
    palette_size: 768,
    wad_entry_count: 10,
    roles: ALL_ROLES,
    roles_companions: &[
        (
            "wall",
            "wall_basecolor.png",
            "wall_norm.png",
            "wall_gloss.png",
        ),
        (
            "floor",
            "floor_basecolor.png",
            "floor_norm.png",
            "floor_gloss.png",
        ),
        (
            "ceiling",
            "ceiling_basecolor.png",
            "ceiling_norm.png",
            "ceiling_gloss.png",
        ),
        (
            "accent",
            "accent_basecolor.png",
            "accent_norm.png",
            "accent_gloss.png",
        ),
        (
            "portal",
            "portal_basecolor.png",
            "portal_norm.png",
            "portal_gloss.png",
        ),
        (
            "vertical",
            "vertical_basecolor.png",
            "vertical_norm.png",
            "vertical_gloss.png",
        ),
        (
            "cave",
            "cave_basecolor.png",
            "cave_norm.png",
            "cave_gloss.png",
        ),
        (
            "prop",
            "prop_basecolor.png",
            "prop_norm.png",
            "prop_gloss.png",
        ),
        (
            "emissive",
            "emissive_basecolor.png",
            "emissive_norm.png",
            "emissive_gloss.png",
        ),
    ],
};

/// Richness Brutalist v1 theme definition.
pub static THEME_BRUTALIST: ThemeDefinition = ThemeDefinition {
    name: "Richness Brutalist v1",
    dir_name: "richness_brutalist_v1",
    wad_filename: "richness_brutalist_v1.wad",
    palette_filename: "palette.lmp",
    license: "CC0-1.0",
    texture_size: 256,
    master_seed: 0x5242_5631,
    png_count: 27,
    palette_size: 768,
    wad_entry_count: 10,
    roles: ALL_ROLES,
    roles_companions: &[
        (
            "wall",
            "wall_basecolor.png",
            "wall_norm.png",
            "wall_gloss.png",
        ),
        (
            "floor",
            "floor_basecolor.png",
            "floor_norm.png",
            "floor_gloss.png",
        ),
        (
            "ceiling",
            "ceiling_basecolor.png",
            "ceiling_norm.png",
            "ceiling_gloss.png",
        ),
        (
            "accent",
            "accent_basecolor.png",
            "accent_norm.png",
            "accent_gloss.png",
        ),
        (
            "portal",
            "portal_basecolor.png",
            "portal_norm.png",
            "portal_gloss.png",
        ),
        (
            "vertical",
            "vertical_basecolor.png",
            "vertical_norm.png",
            "vertical_gloss.png",
        ),
        (
            "cave",
            "cave_basecolor.png",
            "cave_norm.png",
            "cave_gloss.png",
        ),
        (
            "prop",
            "prop_basecolor.png",
            "prop_norm.png",
            "prop_gloss.png",
        ),
        (
            "emissive",
            "emissive_basecolor.png",
            "emissive_norm.png",
            "emissive_gloss.png",
        ),
    ],
};

/// Static slice of all three Richness V1 theme definitions.
const ALL_THEMES: &[&ThemeDefinition] = &[&THEME_ANCIENT, &THEME_EGYPTIAN, &THEME_BRUTALIST];

/// All three Richness V1 theme definitions.
pub fn all_themes() -> &'static [&'static ThemeDefinition] {
    ALL_THEMES
}

// ── Test-only constructor ──────────────────────────────────────────────────

/// Construct a minimal ThemeDefinition for testing.
#[doc(hidden)]
pub fn test_theme_def(name: &'static str) -> ThemeDefinition {
    ThemeDefinition {
        name,
        dir_name: name,
        wad_filename: "test.wad",
        palette_filename: "palette.lmp",
        license: "CC0-1.0",
        texture_size: 256,
        master_seed: 0,
        png_count: 0,
        palette_size: 768,
        wad_entry_count: 0,
        roles: &[],
        roles_companions: &[],
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_roles_have_companions() {
        for theme in all_themes() {
            for role in SemanticRole::all() {
                let wad = role.wad_identity();
                let (bc, norm, gloss) = theme.companion_filenames(*role);
                assert!(
                    bc.starts_with(wad) && bc.ends_with(BASECOLOR_SUFFIX),
                    "{theme_name}: {role:?} basecolor mismatch: {bc}",
                    theme_name = theme.name
                );
                assert!(
                    norm.starts_with(wad) && norm.ends_with(NORM_SUFFIX),
                    "{theme_name}: {role:?} norm mismatch: {norm}",
                    theme_name = theme.name
                );
                assert!(
                    gloss.starts_with(wad) && gloss.ends_with(GLOSS_SUFFIX),
                    "{theme_name}: {role:?} gloss mismatch: {gloss}",
                    theme_name = theme.name
                );
            }
        }
    }

    #[test]
    fn wad_identities_are_lowercase_ascii() {
        for role in SemanticRole::all() {
            let id = role.wad_identity();
            assert!(
                id.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "WAD identity '{id}' is not lowercase ASCII"
            );
        }
    }

    #[test]
    fn companion_filenames_are_exact_case() {
        for theme in all_themes() {
            for filename in theme.all_png_filenames() {
                assert_eq!(
                    filename,
                    filename.to_ascii_lowercase(),
                    "companion filename '{filename}' must be lowercase ASCII"
                );
                assert!(
                    filename.ends_with(".png"),
                    "companion filename '{filename}' must end with .png"
                );
                assert!(
                    filename.contains('_'),
                    "companion filename '{filename}' must have role_ suffix pattern"
                );
            }
        }
    }

    #[test]
    fn themes_have_distinct_dir_names() {
        let mut names: Vec<&str> = all_themes().iter().map(|t| t.dir_name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 3, "theme dir names must be distinct");
    }

    #[test]
    fn themes_have_distinct_wad_filenames() {
        let mut wads: Vec<&str> = all_themes().iter().map(|t| t.wad_filename).collect();
        wads.sort_unstable();
        wads.dedup();
        assert_eq!(wads.len(), 3, "WAD filenames must be distinct");
    }

    #[test]
    fn themes_have_distinct_master_seeds() {
        let mut seeds: Vec<u32> = all_themes().iter().map(|t| t.master_seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(seeds.len(), 3, "master seeds must be distinct");
    }

    #[test]
    fn png_count_is_27() {
        for theme in all_themes() {
            assert_eq!(theme.png_count, 27);
            assert_eq!(theme.all_png_filenames().len(), 27);
        }
    }

    #[test]
    fn wad_entry_count_is_10() {
        for theme in all_themes() {
            assert_eq!(theme.all_wad_identities().len(), 10);
        }
    }

    #[test]
    fn license_is_cc0() {
        for theme in all_themes() {
            assert_eq!(theme.license, "CC0-1.0");
        }
    }

    #[test]
    fn texture_size_is_256() {
        for theme in all_themes() {
            assert_eq!(theme.texture_size, 256);
        }
    }

    #[test]
    fn palette_size_is_768() {
        for theme in all_themes() {
            assert_eq!(theme.palette_size, 768);
        }
    }
}
