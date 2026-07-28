//! Version-1 physics component configuration DTOs.
//!
//! These are renderer-independent serializable descriptions of bodies, collider
//! shapes, and character controllers. They carry no Rapier handles, slot indices,
//! or GPU descriptors.

use serde::{Deserialize, Serialize};

// ── Body ─────────────────────────────────────────────────────────────

/// Version-1 body configuration DTO.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BodyConfigV1 {
    pub body_id: String,
    pub kind: BodyKindConfigV1,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub sleeping: bool,
    pub gravity_scale: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
}

impl Default for BodyConfigV1 {
    fn default() -> Self {
        Self {
            body_id: String::new(),
            kind: BodyKindConfigV1::Dynamic,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            sleeping: false,
            gravity_scale: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
        }
    }
}

/// Body kind for serialization.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum BodyKindConfigV1 {
    Static,
    Dynamic,
    Kinematic,
}

// ── Collider shapes ──────────────────────────────────────────────────

/// Version-1 collider shape configuration DTO.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ColliderShapeConfigV1 {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
    CapsuleY { half_height: f32, radius: f32 },
}

/// Version-1 collider configuration DTO.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ColliderConfigV1 {
    pub collider_id: String,
    pub parent_body_id: String,
    pub shape: ColliderShapeConfigV1,
    pub is_trigger: bool,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub friction: f32,
    pub restitution: f32,
}

impl Default for ColliderConfigV1 {
    fn default() -> Self {
        Self {
            collider_id: String::new(),
            parent_body_id: String::new(),
            shape: ColliderShapeConfigV1::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
            is_trigger: false,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            friction: 0.5,
            restitution: 0.0,
        }
    }
}

// ── Character controller ─────────────────────────────────────────────

/// Version-1 character controller configuration DTO.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CharacterConfigV1 {
    pub character_id: String,
    pub body_id: String,
    pub collider_id: String,
    pub offset: CharacterOffsetConfigV1,
    pub max_slope_angle_radians: f32,
    pub step_height: f32,
    pub min_width: f32,
    pub max_width: f32,
}

impl Default for CharacterConfigV1 {
    fn default() -> Self {
        Self {
            character_id: String::new(),
            body_id: String::new(),
            collider_id: String::new(),
            offset: CharacterOffsetConfigV1::default(),
            max_slope_angle_radians: 0.785398, // ~45 degrees
            step_height: 0.3,
            min_width: 0.2,
            max_width: 1.0,
        }
    }
}

/// Offset for character controller relative to body.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CharacterOffsetConfigV1 {
    pub translation: [f32; 3],
}

impl Default for CharacterOffsetConfigV1 {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
        }
    }
}

// ── Conversions ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_config_roundtrip_json() {
        let config = BodyConfigV1 {
            body_id: "body.player".into(),
            kind: BodyKindConfigV1::Dynamic,
            translation: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            sleeping: false,
            gravity_scale: 1.0,
            linear_damping: 0.5,
            angular_damping: 0.1,
        };
        let json = serde_json::to_string(&config).unwrap();
        let round: BodyConfigV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(config, round);
    }

    #[test]
    fn collider_config_roundtrip_json() {
        let config = ColliderConfigV1 {
            collider_id: "collider.player".into(),
            parent_body_id: "body.player".into(),
            shape: ColliderShapeConfigV1::CapsuleY {
                half_height: 0.8,
                radius: 0.4,
            },
            is_trigger: false,
            translation: [0.0, 0.5, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            friction: 0.6,
            restitution: 0.1,
        };
        let json = serde_json::to_string(&config).unwrap();
        let round: ColliderConfigV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(config, round);
    }

    #[test]
    fn character_config_roundtrip_json() {
        let config = CharacterConfigV1 {
            character_id: "char.player".into(),
            body_id: "body.player".into(),
            collider_id: "collider.player".into(),
            offset: CharacterOffsetConfigV1 {
                translation: [0.0, 0.0, 0.0],
            },
            max_slope_angle_radians: 0.785398,
            step_height: 0.3,
            min_width: 0.2,
            max_width: 1.0,
        };
        let json = serde_json::to_string(&config).unwrap();
        let round: CharacterConfigV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(config, round);
    }

    #[test]
    fn defaults_are_usable() {
        let body = BodyConfigV1::default();
        assert_eq!(body.kind, BodyKindConfigV1::Dynamic);
        assert_eq!(body.gravity_scale, 1.0);

        let collider = ColliderConfigV1::default();
        assert!(!collider.is_trigger);

        let character = CharacterConfigV1::default();
        assert!(character.max_slope_angle_radians > 0.0);
    }
}
