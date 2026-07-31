//! App-owned mesh collider recipe bridge.
//!
//! Converts renderer-neutral [`MeshGeometryDto`] into per-mesh physics recipes
//! (static trimesh, convex hull, or no-collider) and manages the recipe lifecycle:
//! synchronous registration, deferred completion, unload/cancellation, and
//! fence-aware retirement through the renderer's [`RetirementClass::ColliderRecipe`].
//!
//! Recipe geometry is stored in model space; the instance transform is carried
//! separately and decomposed at collider instantiation time.

use std::collections::HashMap;
use std::sync::Arc;

use physics::{
    validate_collider_shape, BodyDescriptor, BodyKind, ColliderDescriptor, ColliderShape,
    PhysicsBodyId, PhysicsColliderId, PhysicsError, PhysicsWorld,
};
use renderer::api::{FrameSerial, GpuRetirementQueue, RetirementClass};
use renderer::prelude::{LoadStatus, MeshGeometryDto, MeshHandle};
use renderer::SceneNodeId;

// ---------------------------------------------------------------------------
// Collider policy
// ---------------------------------------------------------------------------

/// Per-mesh collider policy assigned explicitly by the app.
///
/// - `StaticTrimesh`: a static triangle-mesh collider (static bodies only).
/// - `ConvexHull`: a convex-hull collider (static, dynamic, or kinematic bodies).
/// - `None`: intentional absence — no recipe is allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColliderPolicy {
    StaticTrimesh,
    ConvexHull,
    None,
}

// ---------------------------------------------------------------------------
// Recipe handle
// ---------------------------------------------------------------------------

/// Slot+generation handle for a live mesh collider recipe.
///
/// Stale handles are rejected immediately; payload retirement follows the
/// renderer's fence-aware serial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshColliderRecipeHandle {
    pub slot: u32,
    pub generation: u32,
}

impl MeshColliderRecipeHandle {
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

// ---------------------------------------------------------------------------
// Recipe
// ---------------------------------------------------------------------------

/// Immutable model-space collider recipe.
///
/// Created from a [`MeshGeometryDto`] and a [`ColliderPolicy`]. The instance
/// transform is carried separately at collider instantiation time.
#[derive(Debug, Clone)]
pub struct MeshColliderRecipe {
    pub handle: MeshColliderRecipeHandle,
    pub mesh: MeshHandle,
    pub policy: ColliderPolicy,
    /// Model-space positions (Arc-shared with the DTO).
    pub positions: Arc<[[f32; 3]]>,
    /// Triangle indices, preserved as flat `[u32]`; for trimesh, interpreted as `[u32; 3]` triples.
    pub indices: Arc<[u32]>,
}

// ---------------------------------------------------------------------------
// Bridge error
// ---------------------------------------------------------------------------

/// Errors produced by the mesh collider bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// `ColliderPolicy::None` intentionally has no recipe handle.
    NoColliderRecipe,
    /// A recipe is already registered for this mesh generation.
    RecipeAlreadyExists { slot: u32, generation: u32 },
    /// No recipe exists for this recipe handle.
    RecipeNotFound { slot: u32, generation: u32 },
    /// A recipe slot generation could not advance without wrapping.
    GenerationExhausted { slot: u32 },
    /// The DTO index count is not a multiple of 3.
    InvalidIndices { index_count: usize },
    /// One or more DTO positions are non-finite.
    NonFinitePositions,
    /// StaticTrimesh was requested but the body kind is non-static.
    TrimeshOnNonStaticBody,
    /// The convex hull is empty, degenerate, or has insufficient unique points.
    ConvexHullInvalid(String),
    /// Model-to-instance transform is singular or non-invertible.
    TransformNotInvertible,
    /// Model-to-instance transform contains shear components.
    TransformHasShear,
    /// Model-to-instance transform contains non-finite elements.
    TransformNonFinite,
    /// An error from the physics crate.
    Physics(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoColliderRecipe => write!(f, "None policy does not allocate a recipe"),
            Self::RecipeAlreadyExists { slot, generation } => {
                write!(
                    f,
                    "recipe already exists: slot {slot} generation {generation}"
                )
            }
            Self::RecipeNotFound { slot, generation } => {
                write!(f, "recipe not found: slot {slot} generation {generation}")
            }
            Self::GenerationExhausted { slot } => {
                write!(f, "recipe generation exhausted for slot {slot}")
            }
            Self::InvalidIndices { index_count } => {
                write!(
                    f,
                    "invalid indices: count {index_count} not a multiple of 3"
                )
            }
            Self::NonFinitePositions => write!(f, "non-finite positions in DTO"),
            Self::TrimeshOnNonStaticBody => {
                write!(f, "static trimesh requires a static body")
            }
            Self::ConvexHullInvalid(msg) => write!(f, "convex hull invalid: {msg}"),
            Self::TransformNotInvertible => write!(f, "transform is not invertible"),
            Self::TransformHasShear => write!(f, "transform contains shear"),
            Self::TransformNonFinite => write!(f, "transform contains non-finite elements"),
            Self::Physics(msg) => write!(f, "physics error: {msg}"),
        }
    }
}

impl From<PhysicsError> for BridgeError {
    fn from(e: PhysicsError) -> Self {
        BridgeError::Physics(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Transform decomposition result
// ---------------------------------------------------------------------------

/// Decomposed model-to-instance transform with shear/non-invertible rejection.
#[derive(Debug, Clone, Copy)]
struct DecomposedTransform {
    scale: glam::Vec3,
    rotation: glam::Quat,
    translation: glam::Vec3,
}

// ---------------------------------------------------------------------------
// Recipe store
// ---------------------------------------------------------------------------

/// Generation-validated recipe store keyed by mesh `(slot, generation)`.
///
/// Lookups validate the mesh generation. Invalidation is immediate; retirement
/// reaping follows the renderer frame serial.
#[derive(Default)]
struct RecipeSlot {
    generation: u32,
    recipe: Option<MeshColliderRecipe>,
    retiring: bool,
}

#[derive(Default)]
struct RecipeStore {
    slots: Vec<RecipeSlot>,
    mesh_to_handle: HashMap<(u32, u32), MeshColliderRecipeHandle>,
    free_slots: Vec<u32>,
    retirement: GpuRetirementQueue<MeshColliderRecipe>,
}

impl RecipeStore {
    fn insert(
        &mut self,
        mesh: MeshHandle,
        policy: ColliderPolicy,
        positions: Arc<[[f32; 3]]>,
        indices: Arc<[u32]>,
    ) -> Result<MeshColliderRecipeHandle, BridgeError> {
        let key = (mesh.slot, mesh.generation);
        if self.mesh_to_handle.contains_key(&key) {
            return Err(BridgeError::RecipeAlreadyExists {
                slot: mesh.slot,
                generation: mesh.generation,
            });
        }

        let slot = self.free_slots.pop().unwrap_or(self.slots.len() as u32);
        if slot as usize == self.slots.len() {
            self.slots.push(RecipeSlot::default());
        }
        let recipe_slot = &mut self.slots[slot as usize];
        debug_assert!(recipe_slot.recipe.is_none() && !recipe_slot.retiring);
        let handle = MeshColliderRecipeHandle::new(slot, recipe_slot.generation);
        let recipe = MeshColliderRecipe {
            handle,
            mesh,
            policy,
            positions,
            indices,
        };
        recipe_slot.recipe = Some(recipe);
        self.mesh_to_handle.insert(key, handle);
        Ok(handle)
    }

    fn get_by_mesh(&self, mesh: MeshHandle) -> Result<&MeshColliderRecipe, BridgeError> {
        let handle = self
            .mesh_to_handle
            .get(&(mesh.slot, mesh.generation))
            .copied()
            .ok_or(BridgeError::RecipeNotFound {
                slot: mesh.slot,
                generation: mesh.generation,
            })?;
        self.get_by_handle(handle)
    }

    fn get_by_handle(
        &self,
        handle: MeshColliderRecipeHandle,
    ) -> Result<&MeshColliderRecipe, BridgeError> {
        self.slots
            .get(handle.slot as usize)
            .filter(|slot| slot.generation == handle.generation)
            .and_then(|slot| slot.recipe.as_ref())
            .ok_or(BridgeError::RecipeNotFound {
                slot: handle.slot,
                generation: handle.generation,
            })
    }

    fn retire(
        &mut self,
        mesh: MeshHandle,
        retire_after: FrameSerial,
    ) -> Result<Option<MeshColliderRecipeHandle>, BridgeError> {
        let key = (mesh.slot, mesh.generation);
        let Some(handle) = self.mesh_to_handle.get(&key).copied() else {
            return Ok(None);
        };
        let slot = &mut self.slots[handle.slot as usize];
        let next_generation = slot
            .generation
            .checked_add(1)
            .ok_or(BridgeError::GenerationExhausted { slot: handle.slot })?;
        let recipe = slot
            .recipe
            .take()
            .expect("live mesh map must reference a recipe");
        slot.generation = next_generation;
        slot.retiring = true;
        self.mesh_to_handle.remove(&key);
        self.retirement
            .enqueue(RetirementClass::ColliderRecipe, retire_after, recipe);
        Ok(Some(handle))
    }

    fn reap(&mut self, completed: FrameSerial) -> Result<usize, BridgeError> {
        let records = self
            .retirement
            .reap_through(completed)
            .map_err(|err| BridgeError::Physics(format!("retirement error: {err:?}")))?;
        let count = records.len();
        for record in records {
            debug_assert_eq!(record.class, RetirementClass::ColliderRecipe);
            let slot_index = record.payload.handle.slot;
            let slot = &mut self.slots[slot_index as usize];
            debug_assert!(slot.recipe.is_none() && slot.retiring);
            slot.retiring = false;
            self.free_slots.push(slot_index);
        }
        Ok(count)
    }

    fn pending_retirement_count(&self) -> usize {
        self.retirement
            .pending_by_class(RetirementClass::ColliderRecipe)
    }

    fn is_empty(&self) -> bool {
        self.mesh_to_handle.is_empty()
    }

    fn len(&self) -> usize {
        self.mesh_to_handle.len()
    }
}

// ---------------------------------------------------------------------------
// Body-to-node mapping for transform writeback
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct BodyNodeBinding {
    node: SceneNodeId,
    /// Initial model rotation and scale relative to the physics body's pose.
    visual_from_body: glam::Mat4,
}

/// Maps dynamic/kinematic physics bodies to scene nodes for full-pose writeback.
#[derive(Default)]
pub struct BodyNodeMap {
    mappings: HashMap<PhysicsBodyId, BodyNodeBinding>,
}

impl BodyNodeMap {
    fn insert(
        &mut self,
        body_id: PhysicsBodyId,
        node_id: SceneNodeId,
        visual_from_body: glam::Mat4,
    ) {
        self.mappings.insert(
            body_id,
            BodyNodeBinding {
                node: node_id,
                visual_from_body,
            },
        );
    }

    pub fn remove(&mut self, body_id: &PhysicsBodyId) {
        self.mappings.remove(body_id);
    }

    pub fn get(&self, body_id: &PhysicsBodyId) -> Option<SceneNodeId> {
        self.mappings.get(body_id).map(|binding| binding.node)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&PhysicsBodyId, SceneNodeId)> + '_ {
        self.mappings
            .iter()
            .map(|(body, binding)| (body, binding.node))
    }
}

#[derive(Debug, Clone)]
struct RecipeInstance {
    body: PhysicsBodyId,
    collider: PhysicsColliderId,
    created_body: bool,
}

// ---------------------------------------------------------------------------
// Mesh collider bridge
// ---------------------------------------------------------------------------

/// App-owned bridge that converts mesh DTOs into physics collider recipes,
/// instantiates colliders, manages the recipe lifecycle, and performs
/// dynamic transform writeback.
pub struct MeshColliderBridge {
    store: RecipeStore,
    body_node_map: BodyNodeMap,
    /// Physics world owned by the bridge.
    pub world: PhysicsWorld,
    instances: HashMap<MeshColliderRecipeHandle, Vec<RecipeInstance>>,
    /// Counter for generating unique body/collider string IDs.
    next_body_idx: u64,
}

impl MeshColliderBridge {
    /// Create an empty bridge with a fresh physics world.
    pub fn new() -> Self {
        Self {
            store: RecipeStore::default(),
            body_node_map: BodyNodeMap::default(),
            world: PhysicsWorld::new(),
            instances: HashMap::new(),
            next_body_idx: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Recipe lifecycle
    // -----------------------------------------------------------------------

    /// Register a concrete collider recipe synchronously.
    pub fn register_recipe(
        &mut self,
        dto: &MeshGeometryDto,
        policy: ColliderPolicy,
    ) -> Result<MeshColliderRecipeHandle, BridgeError> {
        if policy == ColliderPolicy::None {
            return Err(BridgeError::NoColliderRecipe);
        }
        validate_dto_for_policy(dto, policy)?;
        self.store.insert(
            dto.mesh,
            policy,
            Arc::clone(&dto.positions),
            Arc::clone(&dto.indices),
        )
    }

    /// Apply any policy. `None` records intentional absence by returning
    /// `Ok(None)` without consuming a handle slot.
    pub fn register_policy(
        &mut self,
        dto: &MeshGeometryDto,
        policy: ColliderPolicy,
    ) -> Result<Option<MeshColliderRecipeHandle>, BridgeError> {
        if policy == ColliderPolicy::None {
            return Ok(None);
        }
        self.register_recipe(dto, policy).map(Some)
    }

    /// Register an uploaded deferred result. Duplicate polling is idempotent.
    pub fn register_deferred_recipe(
        &mut self,
        dto: &MeshGeometryDto,
        policy: ColliderPolicy,
    ) -> Result<MeshColliderRecipeHandle, BridgeError> {
        match self.register_recipe(dto, policy) {
            Ok(handle) => Ok(handle),
            Err(BridgeError::RecipeAlreadyExists { .. }) => {
                self.store.get_by_mesh(dto.mesh).map(|recipe| recipe.handle)
            }
            Err(error) => Err(error),
        }
    }

    /// Explicit deferred lifecycle entrypoint. Pending, queued cancellation, and
    /// failed completion create no recipe; only `Uploaded` registers one.
    pub fn register_deferred_status(
        &mut self,
        status: &LoadStatus<MeshGeometryDto>,
        policy: ColliderPolicy,
    ) -> Result<Option<MeshColliderRecipeHandle>, BridgeError> {
        match status {
            LoadStatus::Uploaded { value } => {
                if policy == ColliderPolicy::None {
                    Ok(None)
                } else {
                    self.register_deferred_recipe(value, policy).map(Some)
                }
            }
            LoadStatus::Pending { .. } | LoadStatus::Cancelled | LoadStatus::Failed { .. } => {
                Ok(None)
            }
        }
    }

    /// Immediately invalidates a recipe and queues its payload until the last
    /// submitted frame that could reference the scene instance completes.
    pub fn unload_recipe(
        &mut self,
        mesh: MeshHandle,
        retire_after: FrameSerial,
    ) -> Result<bool, BridgeError> {
        let retired = self.store.retire(mesh, retire_after)?;
        let Some(handle) = retired else {
            return Ok(false);
        };
        self.remove_recipe_instances(handle);
        Ok(true)
    }

    /// Failure/cancellation after registration follows the same safe invalidation path.
    pub fn cancel_or_fail_recipe(
        &mut self,
        mesh: MeshHandle,
        retire_after: FrameSerial,
    ) -> Result<bool, BridgeError> {
        self.unload_recipe(mesh, retire_after)
    }

    pub fn reap_retired(&mut self, completed: FrameSerial) -> Result<usize, BridgeError> {
        self.store.reap(completed)
    }

    pub fn pending_retirement_count(&self) -> usize {
        self.store.pending_retirement_count()
    }

    /// Look up a recipe by mesh handle.
    pub fn recipe_for_mesh(&self, mesh: MeshHandle) -> Result<&MeshColliderRecipe, BridgeError> {
        self.store.get_by_mesh(mesh)
    }

    /// Look up a recipe by recipe handle.
    pub fn recipe_by_handle(
        &self,
        handle: MeshColliderRecipeHandle,
    ) -> Result<&MeshColliderRecipe, BridgeError> {
        self.store.get_by_handle(handle)
    }

    /// True when no live recipes exist.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Number of live recipes.
    pub fn recipe_count(&self) -> usize {
        self.store.len()
    }

    // -----------------------------------------------------------------------
    // Collider instantiation
    // -----------------------------------------------------------------------

    /// Instantiate a physics collider from a recipe and instance transform.
    ///
    /// Creates the parent body if it doesn't exist. The transform is decomposed
    /// into scale (baked into the collider geometry), rotation, and translation.
    /// Shear and non-invertible transforms are rejected.
    ///
    /// Returns the body and collider IDs on success. The recipe must be live;
    /// stale handles are rejected.
    pub fn instantiate_collider(
        &mut self,
        recipe_handle: MeshColliderRecipeHandle,
        body_kind: BodyKind,
        body_id: &str,
        collider_id: &str,
        model_to_instance: glam::Mat4,
        node_id: Option<SceneNodeId>,
    ) -> Result<(PhysicsBodyId, PhysicsColliderId), BridgeError> {
        let recipe = self.store.get_by_handle(recipe_handle)?.clone();
        let decomposed = decompose_transform(model_to_instance)?;

        if recipe.policy == ColliderPolicy::StaticTrimesh && body_kind != BodyKind::Static {
            return Err(BridgeError::TrimeshOnNonStaticBody);
        }

        // Build and fully validate before mutating PhysicsWorld.
        let shape = build_scaled_shape(&recipe, decomposed.scale)?;
        validate_collider_shape(&shape, body_kind)?;

        let body_id: PhysicsBodyId = body_id.into();
        let collider_id: PhysicsColliderId = collider_id.into();
        let created_body = self.world.body_position_by_id(&body_id).is_none();
        if created_body {
            self.world.create_body(BodyDescriptor::new(
                body_id.clone(),
                body_kind,
                decomposed.translation.to_array(),
            ))?;
        }

        let body_pose = self
            .world
            .body_pose_by_id(&body_id)
            .expect("new or existing body must have a pose");
        let body_world = glam::Mat4::from_rotation_translation(
            glam::Quat::from_array(body_pose.rotation),
            glam::Vec3::from_array(body_pose.translation),
        );
        let instance_pose =
            glam::Mat4::from_rotation_translation(decomposed.rotation, decomposed.translation);
        let collider_from_body = body_world.inverse() * instance_pose;
        let (_, collider_rotation, collider_translation) =
            collider_from_body.to_scale_rotation_translation();
        let desc = ColliderDescriptor::new(collider_id.clone(), body_id.clone(), shape)
            .translation(collider_translation.to_array())
            .rotation(collider_rotation.to_array());
        if let Err(error) = self.world.create_collider(desc) {
            if created_body {
                self.world.remove_body(&body_id);
            }
            return Err(error.into());
        }

        if body_kind != BodyKind::Static {
            if let Some(node) = node_id {
                let visual_from_body = body_world.inverse() * model_to_instance;
                self.body_node_map
                    .insert(body_id.clone(), node, visual_from_body);
            }
        }
        self.instances
            .entry(recipe_handle)
            .or_default()
            .push(RecipeInstance {
                body: body_id.clone(),
                collider: collider_id.clone(),
                created_body,
            });

        Ok((body_id, collider_id))
    }

    /// Write back the pose of every mapped dynamic/kinematic body into the
    /// scene via `set_transform`. Static bodies are skipped.
    pub fn writeback_dynamic_transforms(
        &self,
        scene: &mut renderer::Scene,
    ) -> Result<usize, BridgeError> {
        let mut count = 0;
        for (body_id, binding) in &self.body_node_map.mappings {
            if let Some(pose) = self.world.body_pose_by_id(body_id) {
                let body_from_world = glam::Mat4::from_rotation_translation(
                    glam::Quat::from_array(pose.rotation),
                    glam::Vec3::from_array(pose.translation),
                );
                scene
                    .set_transform(binding.node, body_from_world * binding.visual_from_body)
                    .map_err(|e| BridgeError::Physics(format!("set_transform failed: {e}")))?;
                count += 1;
            }
        }
        Ok(count)
    }

    fn remove_recipe_instances(&mut self, recipe: MeshColliderRecipeHandle) {
        for instance in self.instances.remove(&recipe).unwrap_or_default() {
            self.body_node_map.remove(&instance.body);
            if instance.created_body {
                self.world.remove_body(&instance.body);
            } else {
                self.world.remove_collider(&instance.collider);
            }
        }
    }

    /// Access the body-node map.
    pub fn body_node_map(&self) -> &BodyNodeMap {
        &self.body_node_map
    }

    /// Allocate a unique body index for generating IDs.
    pub fn next_body_index(&mut self) -> u64 {
        let idx = self.next_body_idx;
        self.next_body_idx += 1;
        idx
    }

    /// Export all dynamic/kinematic body-node mappings for integration with
    /// a [`PhysicsBridge`]. Callers should invoke this after seeding so the
    /// component-driven bridge can participate in unified transform writeback.
    pub fn export_body_node_mappings_to_physics_bridge(
        &self,
        physics_bridge: &mut crate::physics_bridge::PhysicsBridge,
    ) {
        for (body_id, node_id) in self.body_node_map.iter() {
            physics_bridge.register_external_body_node(
                body_id.clone(),
                node_id,
                glam::Mat4::IDENTITY,
            );
        }
    }
}

impl Default for MeshColliderBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DTO validation
// ---------------------------------------------------------------------------

/// Validate a DTO against a policy before recipe creation.
fn validate_dto_for_policy(
    dto: &MeshGeometryDto,
    policy: ColliderPolicy,
) -> Result<(), BridgeError> {
    let shape = match policy {
        ColliderPolicy::None => return Ok(()),
        ColliderPolicy::StaticTrimesh => {
            if dto.indices.is_empty() || dto.indices.len() % 3 != 0 {
                return Err(BridgeError::InvalidIndices {
                    index_count: dto.indices.len(),
                });
            }
            ColliderShape::TriMeshStatic {
                vertices: dto.positions.to_vec(),
                indices: dto
                    .indices
                    .chunks_exact(3)
                    .map(|triangle| [triangle[0], triangle[1], triangle[2]])
                    .collect(),
            }
        }
        ColliderPolicy::ConvexHull => ColliderShape::ConvexHull {
            points: dto.positions.to_vec(),
        },
    };
    validate_collider_shape(&shape, BodyKind::Static).map_err(|error| match error {
        PhysicsError::ConvexHullEmpty
        | PhysicsError::ConvexHullInsufficientPoints { .. }
        | PhysicsError::ConvexHullDegenerate => BridgeError::ConvexHullInvalid(error.to_string()),
        PhysicsError::ConvexHullNonFiniteVertex { .. }
        | PhysicsError::TrimeshNonFiniteVertex { .. } => BridgeError::NonFinitePositions,
        other => BridgeError::Physics(other.to_string()),
    })
}

// ---------------------------------------------------------------------------
// Transform decomposition
// ---------------------------------------------------------------------------

/// Decompose a `Mat4` into scale, rotation, translation.
///
/// Rejects shear, non-invertible, and non-finite transforms.
fn decompose_transform(matrix: glam::Mat4) -> Result<DecomposedTransform, BridgeError> {
    // Check all elements are finite.
    for col in matrix.to_cols_array_2d().iter() {
        for &v in col.iter() {
            if !v.is_finite() {
                return Err(BridgeError::TransformNonFinite);
            }
        }
    }

    // Check determinant is non-zero (invertible).
    let det = matrix.determinant();
    if !det.is_finite() || det == 0.0 {
        return Err(BridgeError::TransformNotInvertible);
    }

    // Decompose using glam's built-in.
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();

    if !scale.is_finite()
        || !rotation.is_finite()
        || !translation.is_finite()
        || scale.x == 0.0
        || scale.y == 0.0
        || scale.z == 0.0
    {
        return Err(BridgeError::TransformNotInvertible);
    }

    // Reject shear/projective components rather than silently approximating them.
    let recomposed = glam::Mat4::from_scale_rotation_translation(scale, rotation, translation);
    let source = matrix.to_cols_array();
    let rebuilt = recomposed.to_cols_array();
    for (actual, expected) in source.into_iter().zip(rebuilt) {
        let tolerance = 1e-5 * actual.abs().max(expected.abs()).max(1.0);
        if (actual - expected).abs() > tolerance {
            return Err(BridgeError::TransformHasShear);
        }
    }

    Ok(DecomposedTransform {
        scale,
        rotation,
        translation,
    })
}

// ---------------------------------------------------------------------------
// Shape construction with scale baking
// ---------------------------------------------------------------------------

/// Build a `ColliderShape` from a recipe, baking scale into a temporary
/// vertex copy. Model-space geometry (`recipe.positions`) is scaled; the
/// caller applies the rotation+translation via the collider pose.
fn build_scaled_shape(
    recipe: &MeshColliderRecipe,
    scale: glam::Vec3,
) -> Result<ColliderShape, BridgeError> {
    match recipe.policy {
        ColliderPolicy::None => Err(BridgeError::Physics(
            "cannot build shape for None policy".to_string(),
        )),
        ColliderPolicy::StaticTrimesh => {
            let scaled_positions: Vec<[f32; 3]> = recipe
                .positions
                .iter()
                .map(|p| [p[0] * scale.x, p[1] * scale.y, p[2] * scale.z])
                .collect();

            let indices: Vec<[u32; 3]> = recipe
                .indices
                .chunks_exact(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect();

            Ok(ColliderShape::TriMeshStatic {
                vertices: scaled_positions,
                indices,
            })
        }
        ColliderPolicy::ConvexHull => {
            let points: Vec<[f32; 3]> = recipe
                .positions
                .iter()
                .map(|p| [p[0] * scale.x, p[1] * scale.y, p[2] * scale.z])
                .collect();

            Ok(ColliderShape::ConvexHull { points })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;
    use renderer::prelude::{MeshDeformation, MeshLocalAabb};

    fn make_dto(
        slot: u32,
        gen: u32,
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
    ) -> MeshGeometryDto {
        let aabb = compute_test_aabb(&positions);
        MeshGeometryDto {
            mesh: MeshHandle::new(slot, gen),
            positions: Arc::from(positions.into_boxed_slice()),
            indices: Arc::from(indices.into_boxed_slice()),
            local_aabb: aabb,
            deformation: MeshDeformation::Rigid,
        }
    }

    fn compute_test_aabb(positions: &[[f32; 3]]) -> Option<MeshLocalAabb> {
        if positions.is_empty() {
            return None;
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in positions {
            for i in 0..3 {
                if !p[i].is_finite() {
                    return None;
                }
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        if !min[0].is_finite() {
            return None;
        }
        Some(renderer::prelude::MeshLocalAabb::new(min, max))
    }

    fn tetrahedron_positions() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    fn floor_positions() -> Vec<[f32; 3]> {
        vec![
            [-5.0, 0.0, -5.0],
            [5.0, 0.0, -5.0],
            [-5.0, 0.0, 5.0],
            [5.0, 0.0, 5.0],
        ]
    }

    fn floor_indices() -> Vec<u32> {
        vec![0, 1, 2, 1, 3, 2]
    }

    // --- policy validation ---

    #[test]
    fn static_trimesh_policy_with_valid_mesh() {
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        assert!(validate_dto_for_policy(&dto, ColliderPolicy::StaticTrimesh).is_ok());
    }

    #[test]
    fn static_trimesh_policy_rejects_empty_indices() {
        let dto = make_dto(10, 0, floor_positions(), vec![]);
        let err = validate_dto_for_policy(&dto, ColliderPolicy::StaticTrimesh).unwrap_err();
        assert!(matches!(err, BridgeError::InvalidIndices { .. }));
    }

    #[test]
    fn static_trimesh_policy_rejects_non_multiple_of_three_indices() {
        let dto = make_dto(10, 0, floor_positions(), vec![0, 1]);
        let err = validate_dto_for_policy(&dto, ColliderPolicy::StaticTrimesh).unwrap_err();
        assert!(matches!(err, BridgeError::InvalidIndices { .. }));
    }

    #[test]
    fn static_trimesh_policy_rejects_non_finite_positions() {
        let dto = make_dto(
            10,
            0,
            vec![[0.0, 0.0, 0.0], [1.0, f32::NAN, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        );
        let err = validate_dto_for_policy(&dto, ColliderPolicy::StaticTrimesh).unwrap_err();
        assert!(matches!(err, BridgeError::NonFinitePositions));
    }

    #[test]
    fn convex_hull_policy_with_valid_mesh() {
        let dto = make_dto(10, 0, tetrahedron_positions(), vec![0, 1, 2]);
        assert!(validate_dto_for_policy(&dto, ColliderPolicy::ConvexHull).is_ok());
    }

    #[test]
    fn convex_hull_policy_rejects_empty_positions() {
        let dto = make_dto(10, 0, vec![], vec![]);
        let err = validate_dto_for_policy(&dto, ColliderPolicy::ConvexHull).unwrap_err();
        assert!(matches!(err, BridgeError::ConvexHullInvalid(_)));
    }

    #[test]
    fn none_policy_always_passes_validation() {
        let dto = make_dto(10, 0, vec![], vec![]);
        assert!(validate_dto_for_policy(&dto, ColliderPolicy::None).is_ok());
    }

    // --- recipe registration ---

    #[test]
    fn register_and_lookup_recipe() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        let handle = bridge
            .register_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();
        let recipe = bridge.recipe_by_handle(handle).unwrap();
        assert_eq!(recipe.policy, ColliderPolicy::StaticTrimesh);
        assert_eq!(recipe.mesh.slot, 10);
    }

    #[test]
    fn duplicate_recipe_rejected() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        bridge
            .register_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();
        let err = bridge
            .register_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap_err();
        assert!(matches!(err, BridgeError::RecipeAlreadyExists { .. }));
    }

    #[test]
    fn deferred_recipe_is_idempotent() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        let h1 = bridge
            .register_deferred_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();
        let h2 = bridge
            .register_deferred_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();
        assert_eq!(h1, h2);
        assert_eq!(bridge.recipe_count(), 1);
    }

    #[test]
    fn recipe_for_mesh_by_generation() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        bridge
            .register_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();
        assert!(bridge.recipe_for_mesh(MeshHandle::new(10, 0)).is_ok());
    }

    #[test]
    fn stale_mesh_generation_rejected() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        bridge
            .register_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();
        let err = bridge.recipe_for_mesh(MeshHandle::new(10, 1)).unwrap_err();
        assert!(matches!(err, BridgeError::RecipeNotFound { .. }));
    }

    #[test]
    fn unload_removes_recipe() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        bridge
            .register_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();
        let removed = bridge
            .unload_recipe(MeshHandle::new(10, 0), FrameSerial::new(3))
            .unwrap();
        assert!(removed);
        assert!(bridge.is_empty());
        assert_eq!(bridge.pending_retirement_count(), 1);
    }

    #[test]
    fn unload_nonexistent_is_noop() {
        let mut bridge = MeshColliderBridge::new();
        assert!(!bridge
            .unload_recipe(MeshHandle::new(99, 0), FrameSerial::ZERO)
            .unwrap());
    }

    #[test]
    fn cancelled_recipe_does_not_appear() {
        // Queued cancellation produces no recipe. Simulate: never register.
        let bridge = MeshColliderBridge::new();
        assert!(bridge.recipe_for_mesh(MeshHandle::new(10, 0)).is_err());
        assert!(bridge.is_empty());
    }

    // --- transform decomposition ---

    #[test]
    fn identity_transform_decomposes() {
        let t = decompose_transform(Mat4::IDENTITY).unwrap();
        assert_eq!(t.scale, glam::Vec3::ONE);
        assert!((t.rotation.w - 1.0).abs() < 1e-6);
        assert_eq!(t.translation, glam::Vec3::ZERO);
    }

    #[test]
    fn translation_only() {
        let m = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let t = decompose_transform(m).unwrap();
        assert_eq!(t.translation, glam::Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn uniform_scale_baked() {
        let m = Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(2.0),
            glam::Quat::IDENTITY,
            glam::Vec3::ZERO,
        );
        let t = decompose_transform(m).unwrap();
        assert!((t.scale.x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn non_uniform_scale_works() {
        let m = Mat4::from_scale_rotation_translation(
            glam::Vec3::new(2.0, 1.0, 3.0),
            glam::Quat::IDENTITY,
            glam::Vec3::new(1.0, 0.0, 0.0),
        );
        let t = decompose_transform(m).unwrap();
        assert_eq!(t.scale, glam::Vec3::new(2.0, 1.0, 3.0));
    }

    #[test]
    fn shear_rejected() {
        // Construct a shear matrix manually.
        let mut m = Mat4::IDENTITY;
        m.col_mut(0)[1] = 0.5; // X-shear in Y
        let err = decompose_transform(m).unwrap_err();
        assert!(matches!(err, BridgeError::TransformHasShear));
    }

    #[test]
    fn reflected_non_uniform_scale_is_preserved() {
        let matrix = Mat4::from_scale_rotation_translation(
            glam::Vec3::new(-2.0, 3.0, 4.0),
            glam::Quat::from_rotation_y(0.4),
            glam::Vec3::new(1.0, 2.0, 3.0),
        );
        let decomposed = decompose_transform(matrix).unwrap();
        let rebuilt = Mat4::from_scale_rotation_translation(
            decomposed.scale,
            decomposed.rotation,
            decomposed.translation,
        );
        assert!(matrix.abs_diff_eq(rebuilt, 1e-5));
    }

    #[test]
    fn non_invertible_rejected() {
        let m = Mat4::from_scale(glam::Vec3::ZERO);
        let err = decompose_transform(m).unwrap_err();
        assert!(matches!(err, BridgeError::TransformNotInvertible));
    }

    #[test]
    fn nan_transform_rejected() {
        let m = Mat4::from_cols_array(&[f32::NAN; 16]);
        let err = decompose_transform(m).unwrap_err();
        assert!(matches!(err, BridgeError::TransformNonFinite));
    }

    // --- collider instantiation ---

    #[test]
    fn instantiate_static_trimesh() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        let recipe_handle = bridge
            .register_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();

        let (body, collider) = bridge
            .instantiate_collider(
                recipe_handle,
                BodyKind::Static,
                "body.floor",
                "collider.floor",
                Mat4::IDENTITY,
                None,
            )
            .unwrap();

        assert_eq!(body, PhysicsBodyId::new("body.floor"));
        assert_eq!(collider, PhysicsColliderId::new("collider.floor"));
        assert!(bridge.world.collider_exists(&collider));
    }

    #[test]
    fn instantiate_convex_hull_dynamic() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, tetrahedron_positions(), vec![]);
        let recipe_handle = bridge
            .register_recipe(&dto, ColliderPolicy::ConvexHull)
            .unwrap();

        let (body, collider) = bridge
            .instantiate_collider(
                recipe_handle,
                BodyKind::Dynamic,
                "body.ball",
                "collider.ball",
                Mat4::IDENTITY,
                None,
            )
            .unwrap();

        assert_eq!(body, PhysicsBodyId::new("body.ball"));
        assert!(bridge.world.collider_exists(&collider));
    }

    #[test]
    fn static_trimesh_on_dynamic_rejected() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        let recipe_handle = bridge
            .register_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();

        let err = bridge
            .instantiate_collider(
                recipe_handle,
                BodyKind::Dynamic,
                "body.bad",
                "collider.bad",
                Mat4::IDENTITY,
                None,
            )
            .unwrap_err();
        assert!(matches!(err, BridgeError::TrimeshOnNonStaticBody));
    }

    #[test]
    #[allow(unused_variables)]
    fn scale_baked_into_shape() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, tetrahedron_positions(), vec![]);
        let recipe_handle = bridge
            .register_recipe(&dto, ColliderPolicy::ConvexHull)
            .unwrap();

        let scale = glam::Vec3::new(2.0, 2.0, 2.0);
        let m =
            Mat4::from_scale_rotation_translation(scale, glam::Quat::IDENTITY, glam::Vec3::ZERO);
        let (body, collider) = bridge
            .instantiate_collider(
                recipe_handle,
                BodyKind::Static,
                "body.scaled",
                "collider.scaled",
                m,
                None,
            )
            .unwrap();

        // Ray from above should hit the scaled hull (extends to z=2 now)
        let hit = bridge
            .world
            .cast_ray(physics::RayQuery::new(
                [0.25, 0.25, 5.0],
                [0.0, 0.0, -1.0],
                10.0,
            ))
            .unwrap()
            .unwrap();
        assert_eq!(hit.collider, collider);
    }

    #[test]
    fn stale_recipe_handle_rejected() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        let recipe_handle = bridge
            .register_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();

        bridge
            .unload_recipe(MeshHandle::new(10, 0), FrameSerial::new(4))
            .unwrap();

        let err = bridge
            .instantiate_collider(
                recipe_handle,
                BodyKind::Static,
                "body.bad",
                "collider.bad",
                Mat4::IDENTITY,
                None,
            )
            .unwrap_err();
        assert!(matches!(err, BridgeError::RecipeNotFound { .. }));
    }

    #[test]
    fn collision_contact_detected() {
        let mut bridge = MeshColliderBridge::new();
        bridge.world.set_gravity(0.0, -10.0, 0.0);

        // Static floor via trimesh recipe
        let floor_dto = make_dto(20, 0, floor_positions(), floor_indices());
        let floor_handle = bridge
            .register_recipe(&floor_dto, ColliderPolicy::StaticTrimesh)
            .unwrap();
        bridge
            .instantiate_collider(
                floor_handle,
                BodyKind::Static,
                "body.floor",
                "collider.floor",
                Mat4::IDENTITY,
                None,
            )
            .unwrap();

        // Dynamic convex-hull recipe proves the app bridge's moving-body path.
        let hull_dto = make_dto(21, 0, tetrahedron_positions(), vec![]);
        let hull_handle = bridge
            .register_recipe(&hull_dto, ColliderPolicy::ConvexHull)
            .unwrap();
        bridge
            .instantiate_collider(
                hull_handle,
                BodyKind::Dynamic,
                "body.hull",
                "collider.hull",
                Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
                None,
            )
            .unwrap();

        // Step several times to let the sphere fall onto the floor.
        for _ in 0..120 {
            bridge.world.step(1.0 / 60.0).unwrap();
        }
        let contacts = bridge.world.last_contact_records();
        assert!(
            !contacts.is_empty(),
            "contact should be detected after sphere falls onto trimesh floor"
        );
    }

    #[test]
    fn transform_writeback_map_tracks_only_explicit_dynamic_binding() {
        let mut bridge = MeshColliderBridge::new();
        let body_id = PhysicsBodyId::new("body.dynamic");
        let node_id = SceneNodeId::new(42, 0);

        bridge.body_node_map.insert(
            body_id.clone(),
            node_id,
            Mat4::from_scale(glam::Vec3::splat(2.0)),
        );
        assert_eq!(bridge.body_node_map.get(&body_id), Some(node_id));
        bridge.body_node_map.remove(&body_id);
        assert_eq!(bridge.body_node_map.get(&body_id), None);
    }

    #[test]
    fn deferred_completion_and_fence_retirement_lifecycle() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        let handle = bridge
            .register_deferred_recipe(&dto, ColliderPolicy::StaticTrimesh)
            .unwrap();

        assert!(bridge.unload_recipe(dto.mesh, FrameSerial::new(5)).unwrap());
        assert!(bridge.recipe_by_handle(handle).is_err());
        assert_eq!(bridge.pending_retirement_count(), 1);
        assert_eq!(bridge.reap_retired(FrameSerial::new(4)).unwrap(), 0);
        assert_eq!(bridge.pending_retirement_count(), 1);
        assert_eq!(bridge.reap_retired(FrameSerial::new(5)).unwrap(), 1);
        assert_eq!(bridge.pending_retirement_count(), 0);

        let next = make_dto(11, 0, floor_positions(), floor_indices());
        let next_handle = bridge
            .register_recipe(&next, ColliderPolicy::StaticTrimesh)
            .unwrap();
        assert_eq!(next_handle.slot, handle.slot);
        assert_eq!(next_handle.generation, handle.generation + 1);
    }

    #[test]
    fn deferred_status_registers_only_uploaded_and_is_idempotent() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        assert!(bridge
            .register_deferred_status(&LoadStatus::Cancelled, ColliderPolicy::StaticTrimesh)
            .unwrap()
            .is_none());
        let uploaded = LoadStatus::Uploaded { value: dto };
        let first = bridge
            .register_deferred_status(&uploaded, ColliderPolicy::StaticTrimesh)
            .unwrap();
        let second = bridge
            .register_deferred_status(&uploaded, ColliderPolicy::StaticTrimesh)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(bridge.recipe_count(), 1);
    }

    #[test]
    fn none_policy_no_recipe_allocated() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, floor_positions(), floor_indices());
        assert!(bridge
            .register_policy(&dto, ColliderPolicy::None)
            .unwrap()
            .is_none());
        assert_eq!(bridge.recipe_count(), 0);
        assert!(matches!(
            bridge.register_recipe(&dto, ColliderPolicy::None),
            Err(BridgeError::NoColliderRecipe)
        ));
    }

    #[test]
    fn instantiation_failure_rolls_back_new_body() {
        let mut bridge = MeshColliderBridge::new();
        let first = make_dto(10, 0, tetrahedron_positions(), vec![]);
        let second = make_dto(11, 0, tetrahedron_positions(), vec![]);
        let first_handle = bridge
            .register_recipe(&first, ColliderPolicy::ConvexHull)
            .unwrap();
        let second_handle = bridge
            .register_recipe(&second, ColliderPolicy::ConvexHull)
            .unwrap();
        bridge
            .instantiate_collider(
                first_handle,
                BodyKind::Dynamic,
                "body.first",
                "collider.duplicate",
                Mat4::IDENTITY,
                None,
            )
            .unwrap();

        assert!(bridge
            .instantiate_collider(
                second_handle,
                BodyKind::Dynamic,
                "body.rollback",
                "collider.duplicate",
                Mat4::IDENTITY,
                None,
            )
            .is_err());
        assert!(bridge
            .world
            .body_pose_by_id(&PhysicsBodyId::new("body.rollback"))
            .is_none());
    }

    #[test]
    fn unload_removes_instances_and_node_mapping_before_retirement() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, tetrahedron_positions(), vec![]);
        let handle = bridge
            .register_recipe(&dto, ColliderPolicy::ConvexHull)
            .unwrap();
        let node = SceneNodeId::new(9, 0);
        let (body, collider) = bridge
            .instantiate_collider(
                handle,
                BodyKind::Dynamic,
                "body.unload",
                "collider.unload",
                Mat4::IDENTITY,
                Some(node),
            )
            .unwrap();
        assert_eq!(bridge.body_node_map.get(&body), Some(node));

        assert!(bridge
            .cancel_or_fail_recipe(dto.mesh, FrameSerial::new(2))
            .unwrap());
        assert!(bridge.world.body_pose_by_id(&body).is_none());
        assert!(!bridge.world.collider_exists(&collider));
        assert!(bridge.body_node_map.get(&body).is_none());
        assert_eq!(bridge.pending_retirement_count(), 1);
    }

    #[test]
    fn writeback_preserves_initial_rotation_scale_and_translation() {
        let mut bridge = MeshColliderBridge::new();
        let dto = make_dto(10, 0, tetrahedron_positions(), vec![]);
        let handle = bridge
            .register_recipe(&dto, ColliderPolicy::ConvexHull)
            .unwrap();
        let mut scene = renderer::Scene::new();
        let node = scene.create_node(None, Mat4::IDENTITY).unwrap();
        let expected = Mat4::from_scale_rotation_translation(
            glam::Vec3::new(2.0, 3.0, 4.0),
            glam::Quat::from_rotation_y(0.6),
            glam::Vec3::new(5.0, 6.0, 7.0),
        );
        bridge
            .instantiate_collider(
                handle,
                BodyKind::Dynamic,
                "body.writeback",
                "collider.writeback",
                expected,
                Some(node),
            )
            .unwrap();
        assert_eq!(bridge.writeback_dynamic_transforms(&mut scene).unwrap(), 1);
        assert!(scene.transform(node).unwrap().abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn recipe_store_default_is_empty() {
        let bridge = MeshColliderBridge::new();
        assert!(bridge.is_empty());
        assert_eq!(bridge.recipe_count(), 0);
    }
}
