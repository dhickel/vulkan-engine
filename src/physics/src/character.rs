//! Rapier-backed kinematic character controller.
//!
//! Wraps `rapier3d::control::KinematicCharacterController` behind durable
//! engine IDs.  Rapier handles are never exposed publicly.

use crate::{
    validate_positive, validate_vec3, PhysicsBodyId, PhysicsColliderId, PhysicsError, PhysicsWorld,
};
use rapier3d::control::{
    CharacterAutostep, CharacterCollision, CharacterLength, EffectiveCharacterMovement,
    KinematicCharacterController,
};
use rapier3d::na;
use rapier3d::prelude::*;

/// Configuration for a character controller.
///
/// All fields map to Rapier controller parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct CharacterConfig {
    /// Maximum slope angle the character can climb (radians).
    pub max_slope_climb_angle: f32,
    /// Minimum slope angle that triggers sliding (radians).
    pub min_slope_slide_angle: f32,
    /// Whether sliding is enabled on steep slopes.
    pub slide: bool,
    /// Maximum step height for auto-stepping (absolute world units).
    pub autostep_max_height: f32,
    /// Minimum step width for auto-stepping (absolute world units).
    pub autostep_min_width: f32,
    /// Whether auto-stepping is enabled.
    pub autostep: bool,
    /// Offset from the body origin to the character's base.
    pub offset: [f32; 3],
}

impl Default for CharacterConfig {
    fn default() -> Self {
        Self {
            max_slope_climb_angle: 0.785398, // ~45 degrees
            min_slope_slide_angle: 0.523599, // ~30 degrees
            slide: true,
            autostep_max_height: 0.3,
            autostep_min_width: 0.2,
            autostep: true,
            offset: [0.0; 3],
        }
    }
}

impl CharacterConfig {
    /// Build from the serializable V1 DTO.
    pub fn from_config_v1(
        config: &crate::components::CharacterConfigV1,
    ) -> Result<Self, PhysicsError> {
        validate_positive(
            "character.max_slope_angle_radians",
            config.max_slope_angle_radians,
        )?;
        validate_positive("character.step_height", config.step_height)?;
        validate_positive("character.min_width", config.min_width)?;
        validate_positive("character.max_width", config.max_width)?;
        validate_vec3("character.offset.translation", config.offset.translation)?;
        if config.min_width > config.max_width {
            return Err(PhysicsError::NonPositiveDimension {
                field: "character.min_width > max_width",
            });
        }
        Ok(Self {
            max_slope_climb_angle: config.max_slope_angle_radians,
            min_slope_slide_angle: config.max_slope_angle_radians * 0.7,
            slide: true,
            autostep_max_height: config.step_height,
            autostep_min_width: config.min_width,
            autostep: true,
            offset: config.offset.translation,
        })
    }
}

/// A kinematic character controller backed by Rapier.
///
/// The controller references an existing body+collider pair in a
/// [`PhysicsWorld`].  Call [`move_and_slide`](Self::move_and_slide) each
/// fixed step to apply movement while resolving collisions.
pub struct CharacterController {
    body_id: PhysicsBodyId,
    collider_id: PhysicsColliderId,
    config: CharacterConfig,
    controller: KinematicCharacterController,
    on_floor: bool,
    is_sliding: bool,
}

impl CharacterController {
    /// Create a new controller for an existing body+collider pair.
    ///
    /// The body must be kinematic and the collider must exist in `world`
    /// at construction time.
    pub fn new(
        world: &PhysicsWorld,
        body_id: PhysicsBodyId,
        collider_id: PhysicsColliderId,
        config: CharacterConfig,
    ) -> Result<Self, PhysicsError> {
        // Verify body, collider, and their ownership before constructing the
        // controller. Character motion only drives position-based kinematics.
        let body_handle = world
            .body_handle_for(&body_id)
            .ok_or_else(|| PhysicsError::MissingBody(body_id.clone()))?;
        if !world.body_is_kinematic(&body_id) {
            return Err(PhysicsError::CharacterRequiresKinematicBody(body_id));
        }
        let collider_handle = world
            .collider_handle_for(&collider_id)
            .ok_or_else(|| PhysicsError::MissingCollider(collider_id.clone()))?;

        if world.collider_parent_body(collider_handle) != Some(body_handle) {
            return Err(PhysicsError::MissingCollider(collider_id));
        }
        validate_character_config(&config)?;

        let mut controller = KinematicCharacterController::default();
        controller.max_slope_climb_angle = config.max_slope_climb_angle;
        controller.min_slope_slide_angle = config.min_slope_slide_angle;
        controller.slide = config.slide;
        controller.autostep = config.autostep.then(|| CharacterAutostep {
            max_height: CharacterLength::Absolute(config.autostep_max_height),
            min_width: CharacterLength::Absolute(config.autostep_min_width),
            include_dynamic_bodies: false,
        });

        Ok(Self {
            body_id,
            collider_id,
            config,
            controller,
            on_floor: false,
            is_sliding: false,
        })
    }

    /// Move the character by `desired_translation` over `dt` seconds,
    /// resolving collisions against the world.
    ///
    /// Returns the actual translation applied.  After calling this,
    /// [`is_on_floor`](Self::is_on_floor) reports ground state.
    pub fn move_and_slide(
        &mut self,
        world: &mut PhysicsWorld,
        desired_translation: [f32; 3],
        dt: f32,
    ) -> Result<[f32; 3], PhysicsError> {
        validate_positive("character.dt", dt).map_err(|_| PhysicsError::NonPositiveDeltaTime)?;
        validate_vec3("character.desired_translation", desired_translation)?;

        let body_handle = world
            .body_handle_for(&self.body_id)
            .ok_or_else(|| PhysicsError::MissingBody(self.body_id.clone()))?;
        let collider_handle = world
            .collider_handle_for(&self.collider_id)
            .ok_or_else(|| PhysicsError::MissingBody(self.body_id.clone()))?;

        // Get the collider's shape for the character controller.
        let collider = world
            .colliders
            .get(collider_handle)
            .ok_or_else(|| PhysicsError::MissingBody(self.body_id.clone()))?;
        let body_pos = *collider.position();

        let desired = na::Vector3::new(
            desired_translation[0],
            desired_translation[1],
            desired_translation[2],
        );

        let character_shape = collider.shape();

        // Ensure query pipeline is current before movement.
        world.query_pipeline.update(&world.colliders);

        let result: EffectiveCharacterMovement = self.controller.move_shape(
            dt,
            &world.bodies,
            &world.colliders,
            &world.query_pipeline,
            character_shape,
            &body_pos,
            desired,
            QueryFilter::default().exclude_collider(collider_handle),
            |_collision: CharacterCollision| {
                // Collision events are consumed internally; callers use
                // PhysicsWorld::last_contact_records() for contact data.
            },
        );

        // `result.translation` is a delta. Move the body by that delta rather
        // than assigning the collider's world-space position, which preserves
        // authored collider-local offsets.
        let new_translation = result.translation;
        if let Some(body) = world.bodies.get_mut(body_handle) {
            let mut new_pos = *body.position();
            new_pos.translation.vector += new_translation;
            body.set_next_kinematic_position(new_pos);
            body.set_position(new_pos, true);
        }

        // Keep query pipeline current for next operation.
        world.query_pipeline.update(&world.colliders);

        self.on_floor = result.grounded;
        self.is_sliding = result.is_sliding_down_slope;

        Ok([new_translation.x, new_translation.y, new_translation.z])
    }

    /// Whether the character was on the ground after the last movement.
    pub fn is_on_floor(&self) -> bool {
        self.on_floor
    }

    /// Whether the character was sliding down a slope after the last movement.
    pub fn is_sliding(&self) -> bool {
        self.is_sliding
    }

    /// The durable body ID this controller moves.
    pub fn body_id(&self) -> &PhysicsBodyId {
        &self.body_id
    }

    /// The durable collider ID used for collision detection.
    pub fn collider_id(&self) -> &PhysicsColliderId {
        &self.collider_id
    }

    /// The current configuration (read-only).
    pub fn config(&self) -> &CharacterConfig {
        &self.config
    }

    /// Update the step/slope configuration.
    ///
    /// Changes take effect on the next [`move_and_slide`](Self::move_and_slide) call.
    pub fn set_config(&mut self, config: CharacterConfig) {
        // Configuration is validated at construction. This infallible legacy
        // setter retains its API; invalid values are ignored.
        if validate_character_config(&config).is_err() {
            return;
        }
        self.controller.max_slope_climb_angle = config.max_slope_climb_angle;
        self.controller.min_slope_slide_angle = config.min_slope_slide_angle;
        self.controller.slide = config.slide;
        self.controller.autostep = config.autostep.then(|| CharacterAutostep {
            max_height: CharacterLength::Absolute(config.autostep_max_height),
            min_width: CharacterLength::Absolute(config.autostep_min_width),
            include_dynamic_bodies: false,
        });
        self.config = config;
    }
}

fn validate_character_config(config: &CharacterConfig) -> Result<(), PhysicsError> {
    validate_positive(
        "character.max_slope_climb_angle",
        config.max_slope_climb_angle,
    )?;
    validate_positive(
        "character.min_slope_slide_angle",
        config.min_slope_slide_angle,
    )?;
    validate_vec3("character.offset", config.offset)?;
    if config.autostep {
        validate_positive("character.autostep_max_height", config.autostep_max_height)?;
        validate_positive("character.autostep_min_width", config.autostep_min_width)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::character::{CharacterConfig, CharacterController};
    use crate::{
        BodyDescriptor, BodyKind, ColliderDescriptor, ColliderShape, PhysicsBodyId,
        PhysicsColliderId, PhysicsError, PhysicsWorld,
    };

    fn setup_character_world() -> (PhysicsWorld, PhysicsBodyId, PhysicsColliderId) {
        let mut world = PhysicsWorld::new();
        world.set_gravity(0.0, -9.81, 0.0);

        // Static floor
        world
            .create_body(BodyDescriptor::new(
                "body.floor",
                BodyKind::Static,
                [0.0, -0.5, 0.0],
            ))
            .unwrap();
        world
            .create_collider(ColliderDescriptor::new(
                "collider.floor",
                "body.floor",
                ColliderShape::Cuboid {
                    half_extents: [10.0, 0.5, 10.0],
                },
            ))
            .unwrap();

        // Kinematic character body + capsule collider
        let body_id = world
            .create_body(BodyDescriptor::new(
                "body.character",
                BodyKind::Kinematic,
                [0.0, 1.5, 0.0],
            ))
            .unwrap();
        let collider_id = world
            .create_collider(ColliderDescriptor::new(
                "collider.character",
                body_id.clone(),
                ColliderShape::CapsuleY {
                    half_height: 0.8,
                    radius: 0.4,
                },
            ))
            .unwrap();

        (world, body_id, collider_id)
    }

    #[test]
    fn character_controller_creation_and_basic_move() {
        let (world, body_id, collider_id) = setup_character_world();
        let config = CharacterConfig::default();
        let mut controller =
            CharacterController::new(&world, body_id.clone(), collider_id.clone(), config).unwrap();

        let mut world = world;
        world.step(1.0 / 60.0).unwrap();

        // Move right
        let actual = controller
            .move_and_slide(&mut world, [1.0, 0.0, 0.0], 1.0 / 60.0)
            .unwrap();
        assert!(actual[0] > 0.0, "character should move right");
    }

    #[test]
    fn is_on_floor_detected() {
        let (world, body_id, collider_id) = setup_character_world();
        let config = CharacterConfig::default();
        let mut controller =
            CharacterController::new(&world, body_id.clone(), collider_id.clone(), config).unwrap();

        let mut world = world;
        world.step(1.0 / 60.0).unwrap();

        // Move down — should land on floor
        controller
            .move_and_slide(&mut world, [0.0, -2.0, 0.0], 1.0 / 60.0)
            .unwrap();
        assert!(controller.is_on_floor(), "should be on floor after landing");
    }

    #[test]
    fn missing_body_rejected() {
        let (world, _body_id, collider_id) = setup_character_world();
        let config = CharacterConfig::default();
        let result = CharacterController::new(
            &world,
            PhysicsBodyId::new("body.nonexistent"),
            collider_id,
            config,
        );
        assert!(result.is_err());
    }

    #[test]
    fn zero_dt_rejected() {
        let (world, body_id, collider_id) = setup_character_world();
        let config = CharacterConfig::default();
        let mut controller =
            CharacterController::new(&world, body_id, collider_id, config).unwrap();

        let mut world = world;
        world.step(1.0 / 60.0).unwrap();
        let err = controller
            .move_and_slide(&mut world, [1.0, 0.0, 0.0], 0.0)
            .unwrap_err();
        assert_eq!(err, PhysicsError::NonPositiveDeltaTime);
    }

    #[test]
    fn config_from_v1_dto() {
        let dto = crate::components::CharacterConfigV1 {
            character_id: "char.hero".into(),
            body_id: "body.hero".into(),
            collider_id: "collider.hero".into(),
            offset: crate::components::CharacterOffsetConfigV1 {
                translation: [0.0; 3],
            },
            max_slope_angle_radians: 0.5,
            step_height: 0.3,
            min_width: 0.2,
            max_width: 1.0,
        };
        let config = CharacterConfig::from_config_v1(&dto).unwrap();
        assert_eq!(config.max_slope_climb_angle, 0.5);
        assert_eq!(config.autostep_max_height, 0.3);
    }

    #[test]
    fn config_v1_invalid_width_rejected() {
        let dto = crate::components::CharacterConfigV1 {
            character_id: "char.hero".into(),
            body_id: "body.hero".into(),
            collider_id: "collider.hero".into(),
            offset: crate::components::CharacterOffsetConfigV1 {
                translation: [0.0; 3],
            },
            max_slope_angle_radians: 0.5,
            step_height: 0.3,
            min_width: 1.0,
            max_width: 0.2,
        };
        let result = CharacterConfig::from_config_v1(&dto);
        assert!(result.is_err());
    }

    #[test]
    fn set_config_updates_behavior() {
        let (world, body_id, collider_id) = setup_character_world();
        let config = CharacterConfig::default();
        let mut controller =
            CharacterController::new(&world, body_id.clone(), collider_id.clone(), config).unwrap();

        let mut new_config = CharacterConfig::default();
        new_config.max_slope_climb_angle = 0.2;
        controller.set_config(new_config.clone());

        assert_eq!(controller.config().max_slope_climb_angle, 0.2);
    }
}
