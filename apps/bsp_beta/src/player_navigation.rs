//! BSP player navigation and the app-owned fixed-step controller.
//!
//! [`PlayerMover`] is the legacy point-trace helper retained for existing
//! direct navigation tests. [`BspPlayerMovementController`] is the active
//! Richness boundary used by the BSP beta frame loop when a map contains
//! qualified climb/drop descriptors. It traces the player origin through the
//! compiler-preexpanded player hull and owns gravity, stepping, jumping,
//! ladder, overlap, reset, and one-way-drop state.

use bsp::coords::QuakeToEngine;
use bsp::{self, StoredHull, TraceResult};
use glam::Vec3;
use std::collections::BTreeSet;

/// Player hull half-height in Quake units.
pub const PLAYER_HALF_HEIGHT_QUAKE: f32 = 24.0;
/// Player hull half-extents at the default 0.0254 scale.
pub const PLAYER_HALF_EXTENTS_ENGINE: Vec3 = Vec3::new(
    16.0 * 0.0254,
    PLAYER_HALF_HEIGHT_QUAKE * 0.0254, // Z extent → engine Y (up)
    16.0 * 0.0254,                     // Y extent → engine -Z
);

/// Fixed-step player mover using BSP clipnode traces.
pub struct PlayerMover {
    /// Current position in engine space.
    pub position: Vec3,
    /// Player hull half-extents in engine units.
    pub half_extents: Vec3,
    /// Current velocity (engine units per second).
    pub velocity: Vec3,
    /// Whether the mover is on the ground.
    pub on_ground: bool,
}

impl PlayerMover {
    /// Create a new player mover at the given engine-space position.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            half_extents: PLAYER_HALF_EXTENTS_ENGINE,
            velocity: Vec3::ZERO,
            on_ground: false,
        }
    }

    /// Check whether a point in engine space is in empty (walkable) space.
    pub fn is_clear(
        &self,
        position: Vec3,
        nodes: &[bsp::lumps::Node],
        leaves: &[bsp::lumps::Leaf],
        planes: &[bsp::lumps::Plane],
    ) -> bool {
        let contents = bsp::point_contents(position, nodes, leaves, planes);
        !contents.is_solid()
    }

    /// Trace a line from current position by a delta, returning the
    /// trace result for point hull (hull 0).
    pub fn trace_move(
        &self,
        delta: Vec3,
        clipnodes: &[bsp::lumps::Clipnode],
        planes: &[bsp::lumps::Plane],
        models: &[bsp::lumps::Model],
        qte: &QuakeToEngine,
    ) -> TraceResult {
        let end = self.position + delta;
        bsp::trace_line(
            self.position,
            end,
            StoredHull::Point,
            clipnodes,
            planes,
            models,
            qte,
        )
    }

    /// Attempt a fixed-step move. Returns the new position after resolving
    /// collisions via simple slide-along-wall.
    ///
    /// If `resolve_sliding` is true, the mover will attempt to slide along
    /// the hit plane instead of stopping dead.
    pub fn step(
        &mut self,
        delta: Vec3,
        clipnodes: &[bsp::lumps::Clipnode],
        planes_data: &[bsp::lumps::Plane],
        models: &[bsp::lumps::Model],
        _nodes: &[bsp::lumps::Node],
        _leaves: &[bsp::lumps::Leaf],
        _bsp_planes: &[bsp::lumps::Plane],
        qte: &QuakeToEngine,
        resolve_sliding: bool,
    ) {
        if delta.length_squared() < 1e-10 {
            return;
        }

        let result = self.trace_move(delta, clipnodes, planes_data, models, qte);

        if result.starts_solid {
            // Try to nudge out — should not happen in normal operation
            return;
        }

        if result.no_hit {
            // Full move succeeded
            self.position += delta;
            self.on_ground = false;
        } else if result.hit_fraction < 1e-6 {
            // Blocked immediately; don't move
        } else if resolve_sliding {
            // Move to the hit point, then attempt to slide
            let move_fraction = (result.hit_fraction - 0.001).max(0.0);
            self.position += delta * move_fraction;

            // Compute slide vector: remove the component along the hit normal
            let remaining = delta * (1.0 - move_fraction);
            let normal = result.plane_normal;
            let dot = remaining.dot(normal);
            if dot < 0.0 {
                let slide = remaining - normal * dot;
                // Recurse with the slide vector (one level, no infinite recursion)
                if slide.length_squared() > 1e-10 {
                    let slide_result = self.trace_move(slide, clipnodes, planes_data, models, qte);
                    if slide_result.no_hit {
                        self.position += slide;
                    } else if slide_result.hit_fraction > 1e-6 {
                        self.position += slide * (slide_result.hit_fraction - 0.001).max(0.0);
                    }
                }
            }
        } else {
            // Move up to the hit point but not through
            let move_fraction = (result.hit_fraction - 0.001).max(0.0);
            self.position += delta * move_fraction;
        }
    }

    /// Verify that the current position is in non-solid space.
    pub fn validate_position(
        &self,
        nodes: &[bsp::lumps::Node],
        leaves: &[bsp::lumps::Leaf],
        planes: &[bsp::lumps::Plane],
    ) -> bool {
        self.is_clear(self.position, nodes, leaves, planes)
    }
}

/// Active-controller fixed step.
pub const BSP_FIXED_DT: f32 = 1.0 / 60.0;
/// Existing BSP beta horizontal speed, retained for Richness movement.
pub const WALK_SPEED_ENGINE: f32 = 1.0;
/// Maximum automatic step height.
pub const STEP_HEIGHT_QUAKE: f32 = 24.0;
/// Upward jump impulse.
pub const JUMP_SPEED_ENGINE: f32 = 4.0;
/// Downward acceleration while airborne or dropping.
pub const GRAVITY_ENGINE: f32 = 9.8;
/// Maximum downward speed.
pub const TERMINAL_FALL_SPEED_ENGINE: f32 = 20.0;
/// Fraction of the desired horizontal velocity acquired per airborne tick.
pub const AIR_CONTROL_FACTOR: f32 = 0.25;
/// Constant ladder speed; ladders apply no gravity or vertical acceleration.
pub const LADDER_SPEED_ENGINE: f32 = 1.5;
/// Minimum forward alignment accepted for climb/drop entry.
pub const VOLUME_ENTRY_DOT: f32 = 0.5;

const CONVENTION_REVISION: &str = "enhanced-v3-richness-conventions/v1";
const GROUND_PROBE_QUAKE: f32 = 2.0;
const DROP_LANDING_MIN_QUAKE: f32 = 32.0;

/// Input sampled once per display frame and replayed at the fixed-step boundary.
#[derive(Debug, Clone, Copy, Default)]
pub struct MovementInput {
    /// Camera-relative horizontal intent transformed into engine world space.
    pub wish_direction: Vec3,
    /// Forward/backward axis. Positive forward climbs up; negative climbs down.
    pub forward_axis: f32,
    /// Initial press of the jump action.
    pub jump_pressed: bool,
}

impl MovementInput {
    pub fn new(wish_direction: Vec3, forward_axis: f32, jump_pressed: bool) -> Self {
        let mut wish_direction = Vec3::new(wish_direction.x, 0.0, wish_direction.z);
        if wish_direction.length_squared() > 1.0 {
            wish_direction = wish_direction.normalize();
        }
        Self {
            wish_direction,
            forward_axis: forward_axis.clamp(-1.0, 1.0),
            jump_pressed,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BspMovementState {
    Grounded,
    Airborne,
    Climbing {
        volume_id: String,
        retained_horizontal: Vec3,
    },
    OneWayDropping {
        volume_id: String,
        entry_height: f32,
        retained_horizontal: Vec3,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BspMovementDiagnostic {
    pub code: &'static str,
    pub volume_id: Option<String>,
    pub detail: &'static str,
}

impl BspMovementDiagnostic {
    fn blocked(volume_id: Option<String>, detail: &'static str) -> Self {
        Self {
            code: "BspMovementBlocked",
            volume_id,
            detail,
        }
    }
}

#[derive(Debug, Clone)]
struct MovementVolume {
    source_entity: u32,
    id: String,
    mins: Vec3,
    maxs: Vec3,
    entry_normal: Vec3,
    priority: i32,
}

impl MovementVolume {
    fn contains(&self, point: Vec3) -> bool {
        point.x >= self.mins.x
            && point.x <= self.maxs.x
            && point.y >= self.mins.y
            && point.y <= self.maxs.y
            && point.z >= self.mins.z
            && point.z <= self.maxs.z
    }
}

/// Raw collision records and qualified volume descriptors retained by the app.
#[derive(Debug, Clone)]
pub struct BspMovementWorld {
    clipnodes: Vec<bsp::lumps::Clipnode>,
    planes: Vec<bsp::lumps::Plane>,
    models: Vec<bsp::lumps::Model>,
    nodes: Vec<bsp::lumps::Node>,
    leaves: Vec<bsp::lumps::Leaf>,
    qte: QuakeToEngine,
    climb_volumes: Vec<MovementVolume>,
    drop_volumes: Vec<MovementVolume>,
}

impl BspMovementWorld {
    /// Extract only compiler-qualified Richness descriptors. Ordinary maps have
    /// no volumes and retain the pre-existing free-camera path.
    pub fn from_bsp(world: &bsp::BspWorld, scale: f32) -> Result<Self, String> {
        let qte = QuakeToEngine::new(scale);
        let mut climb_volumes = Vec::new();
        let mut drop_volumes = Vec::new();
        let mut ids = BTreeSet::new();

        for entity in &world.entities {
            let Some(kind) = bsp::entities::get_singleton(entity, "richness_volume") else {
                continue;
            };
            let revision =
                bsp::entities::get_singleton(entity, "convention_revision").ok_or_else(|| {
                    format!(
                        "richness volume {} has no convention_revision",
                        entity.source_index
                    )
                })?;
            if revision != CONVENTION_REVISION {
                return Err(format!(
                    "richness volume {} uses unsupported convention revision '{revision}'",
                    entity.source_index
                ));
            }
            let id = bsp::entities::get_singleton(entity, "richness_volume_id")
                .filter(|id| !id.is_empty())
                .ok_or_else(|| format!("richness volume {} has no stable id", entity.source_index))?
                .to_string();
            if !ids.insert(id.clone()) {
                return Err(format!("duplicate richness volume id '{id}'"));
            }
            let model_index = bsp::entities::get_singleton(entity, "model")
                .and_then(|model| model.strip_prefix('*'))
                .and_then(|model| model.parse::<usize>().ok())
                .ok_or_else(|| format!("richness volume '{id}' has no brush model"))?;
            let model = world
                .models
                .get(model_index)
                .ok_or_else(|| format!("richness volume '{id}' model *{model_index} is missing"))?;
            let (mins, maxs) = qte.aabb(model.mins, model.maxs);
            if !mins.is_finite()
                || !maxs.is_finite()
                || mins.x >= maxs.x
                || mins.y >= maxs.y
                || mins.z >= maxs.z
            {
                return Err(format!("richness volume '{id}' has invalid bounds"));
            }

            let (entry_key, default_priority) = match kind {
                "climb" => ("climb_normal", 0),
                "one_way_drop" => {
                    if bsp::entities::get_singleton(entity, "one_way") != Some("1") {
                        return Err(format!("drop volume '{id}' is not marked one_way=1"));
                    }
                    ("entry_normal", 0)
                }
                other => return Err(format!("unsupported richness volume kind '{other}'")),
            };
            let normal_q = parse_vec3(
                bsp::entities::get_singleton(entity, entry_key)
                    .ok_or_else(|| format!("richness volume '{id}' has no {entry_key}"))?,
            )
            .ok_or_else(|| format!("richness volume '{id}' has malformed {entry_key}"))?;
            let entry_normal = qte.normal_vec3(normal_q);
            let cardinal = ((entry_normal.x.abs() - 1.0).abs() <= 1.0e-4
                && entry_normal.z.abs() <= 1.0e-4)
                || ((entry_normal.z.abs() - 1.0).abs() <= 1.0e-4 && entry_normal.x.abs() <= 1.0e-4);
            if entry_normal.y.abs() > 1.0e-4
                || (entry_normal.length_squared() - 1.0).abs() > 1.0e-4
                || !cardinal
            {
                return Err(format!(
                    "richness volume '{id}' entry normal must be horizontal unit cardinal"
                ));
            }
            let priority = bsp::entities::get_singleton(entity, "climb_priority")
                .map(str::parse::<i32>)
                .transpose()
                .map_err(|_| format!("richness volume '{id}' has malformed climb_priority"))?
                .unwrap_or(default_priority);
            let volume = MovementVolume {
                source_entity: entity.source_index,
                id,
                mins,
                maxs,
                entry_normal,
                priority,
            };
            match kind {
                "climb" => climb_volumes.push(volume),
                "one_way_drop" => drop_volumes.push(volume),
                _ => unreachable!(),
            }
        }

        climb_volumes.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.source_entity.cmp(&right.source_entity))
                .then_with(|| left.id.cmp(&right.id))
        });
        drop_volumes.sort_by(|left, right| left.source_entity.cmp(&right.source_entity));

        Ok(Self {
            clipnodes: world.clipnodes.clone(),
            planes: world.planes.clone(),
            models: world.models.clone(),
            nodes: world.nodes.clone(),
            leaves: world.leaves.clone(),
            qte,
            climb_volumes,
            drop_volumes,
        })
    }

    pub fn has_richness_volumes(&self) -> bool {
        !self.climb_volumes.is_empty() || !self.drop_volumes.is_empty()
    }
}

fn parse_vec3(value: &str) -> Option<Vec3> {
    let mut values = value.split_whitespace().map(str::parse::<f32>);
    let result = Vec3::new(
        values.next()?.ok()?,
        values.next()?.ok()?,
        values.next()?.ok()?,
    );
    (values.next().is_none() && result.is_finite()).then_some(result)
}

/// App-owned movement boundary called by the BSP beta fixed-step loop.
pub struct BspPlayerMovementController {
    world: BspMovementWorld,
    position: Vec3,
    velocity: Vec3,
    state: BspMovementState,
    diagnostics: Vec<BspMovementDiagnostic>,
}

impl BspPlayerMovementController {
    pub fn new(position: Vec3, world: BspMovementWorld) -> Self {
        Self {
            world,
            position,
            velocity: Vec3::ZERO,
            state: BspMovementState::Airborne,
            diagnostics: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.world.has_richness_volumes()
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }

    pub fn state(&self) -> &BspMovementState {
        &self.state
    }

    pub fn active_volume_id(&self) -> Option<&str> {
        match &self.state {
            BspMovementState::Climbing { volume_id, .. }
            | BspMovementState::OneWayDropping { volume_id, .. } => Some(volume_id),
            _ => None,
        }
    }

    pub fn validate_position(&self) -> bool {
        let trace = self.trace(Vec3::ZERO);
        !trace.starts_solid && !trace.all_solid
    }

    pub fn take_diagnostics(&mut self) -> Vec<BspMovementDiagnostic> {
        std::mem::take(&mut self.diagnostics)
    }

    /// Teleport/focus/spawn reset. Every out-of-band position change clears
    /// volume references and velocity.
    pub fn teleport(&mut self, position: Vec3) {
        self.position = position;
        self.velocity = Vec3::ZERO;
        self.state = BspMovementState::Airborne;
        self.diagnostics.clear();
    }

    /// Atomic generation replacement. No descriptor from the retired world is
    /// retained.
    pub fn reset_for_regeneration(&mut self, position: Vec3, world: BspMovementWorld) {
        self.world = world;
        self.teleport(position);
    }

    /// Keep MCP/test camera teleports and the controller authoritative state in
    /// agreement without interpreting them as movement.
    pub fn synchronize_external_position(&mut self, position: Vec3) {
        if self.position.distance_squared(position) > 1.0e-8 {
            self.teleport(position);
        }
    }

    pub fn fixed_step(&mut self, input: MovementInput, dt: f32) {
        if !self.is_active() || !dt.is_finite() || dt <= 0.0 {
            return;
        }

        match self.state.clone() {
            BspMovementState::Climbing {
                volume_id,
                retained_horizontal,
            } => self.step_climbing(input, dt, &volume_id, retained_horizontal),
            BspMovementState::OneWayDropping {
                volume_id,
                entry_height,
                retained_horizontal,
            } => self.step_drop(dt, &volume_id, entry_height, retained_horizontal),
            BspMovementState::Grounded | BspMovementState::Airborne => {
                self.try_enter_volume(input);
                match self.state.clone() {
                    BspMovementState::Climbing {
                        volume_id,
                        retained_horizontal,
                    } => self.step_climbing(input, dt, &volume_id, retained_horizontal),
                    BspMovementState::OneWayDropping {
                        volume_id,
                        entry_height,
                        retained_horizontal,
                    } => self.step_drop(dt, &volume_id, entry_height, retained_horizontal),
                    _ => self.step_standard(input, dt),
                }
            }
        }
    }

    fn trace(&self, delta: Vec3) -> TraceResult {
        bsp::trace_line(
            self.position,
            self.position + delta,
            StoredHull::Player,
            &self.world.clipnodes,
            &self.world.planes,
            &self.world.models,
            &self.world.qte,
        )
    }

    fn grounded(&self) -> bool {
        let trace = self.trace(Vec3::NEG_Y * self.world.qte.scale * GROUND_PROBE_QUAKE);
        !trace.no_hit && !trace.starts_solid && trace.plane_normal.y > 0.5
    }

    fn move_fraction(&mut self, delta: Vec3) -> TraceResult {
        let trace = self.trace(delta);
        if trace.no_hit {
            self.position += delta;
        } else if !trace.starts_solid && trace.hit_fraction > 1.0e-6 {
            self.position += delta * (trace.hit_fraction - 0.001).max(0.0);
        }
        trace
    }

    fn try_enter_volume(&mut self, input: MovementInput) {
        let entry_direction = input.wish_direction.normalize_or_zero();
        if input.forward_axis > 0.0 {
            if let Some(volume) = self
                .world
                .climb_volumes
                .iter()
                .find(|volume| {
                    volume.contains(self.position)
                        && entry_direction.dot(volume.entry_normal) >= VOLUME_ENTRY_DOT
                })
                .cloned()
            {
                let retained_horizontal = Vec3::new(self.velocity.x, 0.0, self.velocity.z);
                self.velocity = Vec3::ZERO;
                self.state = BspMovementState::Climbing {
                    volume_id: volume.id,
                    retained_horizontal,
                };
                return;
            }
        }

        if let Some(volume) = self
            .world
            .drop_volumes
            .iter()
            .find(|volume| {
                volume.contains(self.position)
                    && entry_direction.dot(volume.entry_normal) >= VOLUME_ENTRY_DOT
            })
            .cloned()
        {
            let retained_horizontal = entry_direction * WALK_SPEED_ENGINE;
            self.velocity = retained_horizontal;
            self.state = BspMovementState::OneWayDropping {
                volume_id: volume.id,
                entry_height: self.position.y,
                retained_horizontal,
            };
        }
    }

    fn step_standard(&mut self, input: MovementInput, dt: f32) {
        let was_grounded = self.grounded();
        let desired = input.wish_direction * WALK_SPEED_ENGINE;
        if was_grounded {
            self.velocity.x = desired.x;
            self.velocity.z = desired.z;
            self.velocity.y = 0.0;
            if input.jump_pressed {
                self.velocity.y = JUMP_SPEED_ENGINE;
            }
        } else {
            self.velocity.x += (desired.x - self.velocity.x) * AIR_CONTROL_FACTOR;
            self.velocity.z += (desired.z - self.velocity.z) * AIR_CONTROL_FACTOR;
        }
        if !was_grounded || input.jump_pressed {
            self.velocity.y =
                (self.velocity.y - GRAVITY_ENGINE * dt).max(-TERMINAL_FALL_SPEED_ENGINE);
        }

        let horizontal = Vec3::new(self.velocity.x * dt, 0.0, self.velocity.z * dt);
        if horizontal.length_squared() > 1.0e-10 {
            self.move_horizontal(horizontal, was_grounded);
        }

        let vertical = Vec3::Y * self.velocity.y * dt;
        if vertical.length_squared() > 1.0e-10 {
            let trace = self.move_fraction(vertical);
            if !trace.no_hit {
                if self.velocity.y < 0.0 && trace.plane_normal.y > 0.5 {
                    self.velocity.y = 0.0;
                } else if self.velocity.y > 0.0 && trace.plane_normal.y < -0.5 {
                    self.velocity.y = 0.0;
                    self.diagnostics.push(BspMovementDiagnostic::blocked(
                        None,
                        "standing headroom rejected upward movement",
                    ));
                }
            }
        }

        self.state = if self.grounded() && self.velocity.y <= 0.0 {
            BspMovementState::Grounded
        } else {
            BspMovementState::Airborne
        };
    }

    fn move_horizontal(&mut self, delta: Vec3, allow_step: bool) {
        let start = self.position;
        let direct = self.move_fraction(delta);
        if direct.no_hit || direct.starts_solid || !allow_step {
            return;
        }

        self.position = start;
        let step = self.world.qte.scale * STEP_HEIGHT_QUAKE;
        let up = self.move_fraction(Vec3::Y * step);
        if !up.no_hit {
            self.position = start;
            return;
        }
        let across = self.move_fraction(delta);
        if !across.no_hit {
            self.position = start;
            return;
        }
        let down = self.move_fraction(Vec3::NEG_Y * (step + self.world.qte.scale));
        if down.starts_solid || down.no_hit || down.plane_normal.y <= 0.5 {
            self.position = start;
        }
    }

    fn step_climbing(
        &mut self,
        input: MovementInput,
        dt: f32,
        volume_id: &str,
        retained_horizontal: Vec3,
    ) {
        if input.jump_pressed {
            self.velocity = retained_horizontal + Vec3::Y * JUMP_SPEED_ENGINE;
            self.state = BspMovementState::Airborne;
            return;
        }
        let Some(volume) = self
            .world
            .climb_volumes
            .iter()
            .find(|volume| volume.id == volume_id)
            .cloned()
        else {
            self.velocity = Vec3::ZERO;
            self.state = BspMovementState::Airborne;
            return;
        };
        if !volume.contains(self.position) {
            self.velocity = retained_horizontal;
            self.state = BspMovementState::Airborne;
            return;
        }

        self.velocity = Vec3::Y * input.forward_axis * LADDER_SPEED_ENGINE;
        let trace = self.move_fraction(self.velocity * dt);
        if !trace.no_hit {
            if input.forward_axis > 0.0 && trace.plane_normal.y < -0.5 {
                // The ladder shaft is capped by solid structure: the climb is
                // complete at the physical top. Release to airborne so the
                // player exits deterministically instead of stalling.
                self.velocity = retained_horizontal;
                self.state = BspMovementState::Airborne;
                return;
            }
            self.velocity = Vec3::ZERO;
            self.diagnostics.push(BspMovementDiagnostic::blocked(
                Some(volume.id.clone()),
                "ladder movement intersected BSP collision",
            ));
        }
        let half_height = PLAYER_HALF_HEIGHT_QUAKE * self.world.qte.scale;
        if input.forward_axis > 0.0 && self.position.y >= volume.maxs.y - half_height {
            self.velocity = retained_horizontal;
            self.state = BspMovementState::Airborne;
        } else if input.forward_axis < 0.0 && self.position.y <= volume.mins.y + half_height {
            self.velocity = retained_horizontal;
            self.state = if self.grounded() {
                BspMovementState::Grounded
            } else {
                BspMovementState::Airborne
            };
        }
    }

    fn step_drop(
        &mut self,
        dt: f32,
        volume_id: &str,
        entry_height: f32,
        retained_horizontal: Vec3,
    ) {
        self.velocity.x = retained_horizontal.x;
        self.velocity.z = retained_horizontal.z;
        self.velocity.y = (self.velocity.y - GRAVITY_ENGINE * dt).max(-TERMINAL_FALL_SPEED_ENGINE);
        let horizontal = Vec3::new(self.velocity.x * dt, 0.0, self.velocity.z * dt);
        self.move_fraction(horizontal);
        let trace = self.move_fraction(Vec3::Y * self.velocity.y * dt);
        let minimum_descent = self.world.qte.scale * DROP_LANDING_MIN_QUAKE;
        if !trace.no_hit
            && self.velocity.y < 0.0
            && trace.plane_normal.y > 0.5
            && self.position.y <= entry_height - minimum_descent
        {
            self.velocity = Vec3::ZERO;
            self.state = BspMovementState::Grounded;
        } else {
            self.state = BspMovementState::OneWayDropping {
                volume_id: volume_id.to_string(),
                entry_height,
                retained_horizontal,
            };
        }
    }

    /// Point contents remains an independent strict-world witness; movement
    /// collision itself always uses the compiler-preexpanded player hull.
    pub fn point_is_clear(&self) -> bool {
        !bsp::point_contents_with_transform(
            self.position,
            &self.world.nodes,
            &self.world.leaves,
            &self.world.planes,
            &self.world.qte,
        )
        .is_solid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bsp::coords::QuakeToEngine;
    use bsp::{BspLoader, LoadOptions};
    use glam::Vec3;
    use std::path::Path;

    fn load_fixture(name: &str) -> bsp::BspWorld {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/bsp/tests/fixtures/compiled")
            .join(name);
        let data = std::fs::read(&path).unwrap();
        let options = LoadOptions {
            strict: true,
            source_identity: name.into(),
            ..LoadOptions::default()
        };
        BspLoader::load(&data, &options).unwrap()
    }

    #[test]
    fn player_mover_spawn_valid() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();
        let spawn_q = Vec3::new(-128.0, 0.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mover = PlayerMover::new(spawn_eng);
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }

    #[test]
    fn player_mover_simple_east_move() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();
        let spawn_q = Vec3::new(-128.0, 100.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mut mover = PlayerMover::new(spawn_eng);

        // Move east by 200 quake units, north of the pillar (clear path)
        let delta_q = Vec3::new(200.0, 0.0, 0.0);
        let delta_eng = Vec3::new(
            qte.scale * delta_q.x,
            qte.scale * delta_q.z,
            -qte.scale * delta_q.y,
        );

        mover.step(
            delta_eng,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &world.nodes,
            &world.leaves,
            &world.planes,
            &qte,
            true,
        );

        // Should have moved east
        let moved_q = Vec3::new(
            mover.position.x / qte.scale,
            -mover.position.z / qte.scale,
            mover.position.y / qte.scale,
        );
        assert!(
            moved_q.x > -100.0,
            "mover should have moved east, x={}",
            moved_q.x
        );
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }

    #[test]
    fn player_mover_blocked_by_pillar() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();
        let spawn_q = Vec3::new(-128.0, 0.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mut mover = PlayerMover::new(spawn_eng);

        // Move east directly toward the pillar
        let delta_eng = Vec3::new(qte.scale * 200.0, 0.0, 0.0);
        mover.step(
            delta_eng,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &world.nodes,
            &world.leaves,
            &world.planes,
            &qte,
            true,
        );

        // Should have been stopped by the pillar
        let moved_q = Vec3::new(
            mover.position.x / qte.scale,
            -mover.position.z / qte.scale,
            mover.position.y / qte.scale,
        );
        // The pillar is at x=-16..16. Mover should not have reached past x=-17
        assert!(
            moved_q.x < 0.0,
            "mover should be blocked by pillar, x={}",
            moved_q.x
        );
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }

    #[test]
    fn player_mover_slide_along_wall() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();
        // Start near but not at the west wall, heading north-west at an angle.
        let spawn_q = Vec3::new(-220.0, -100.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mut mover = PlayerMover::new(spawn_eng);

        // Move north-west (-20 X, +200 Y in Quake)
        let delta_q = Vec3::new(-20.0, 200.0, 0.0);
        let delta_eng = Vec3::new(
            qte.scale * delta_q.x,
            qte.scale * delta_q.z,
            -qte.scale * delta_q.y,
        );

        mover.step(
            delta_eng,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &world.nodes,
            &world.leaves,
            &world.planes,
            &qte,
            true,
        );

        let pos_q = Vec3::new(
            mover.position.x / qte.scale,
            -mover.position.z / qte.scale,
            mover.position.y / qte.scale,
        );
        // Should slide north along west wall
        assert!(
            pos_q.x <= -210.0,
            "mover must not pass through west wall, x={}",
            pos_q.x
        );
        assert!(pos_q.y > -90.0, "mover must slide north, y={}", pos_q.y);
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }

    #[test]
    fn player_mover_validate_all_positions() {
        let world = load_fixture("dungeon-navigation-bsp2.bsp");
        let qte = QuakeToEngine::default();

        // Test multiple clear positions around the room
        let clear_positions = [
            Vec3::new(-200.0, 0.0, 0.0),
            Vec3::new(200.0, 0.0, 0.0),
            Vec3::new(0.0, 200.0, 0.0),
            Vec3::new(0.0, -200.0, 0.0),
            Vec3::new(-128.0, 128.0, 0.0),
            Vec3::new(128.0, -128.0, 0.0),
        ];

        for q_pos in &clear_positions {
            let eng_pos = qte.position_vec3(*q_pos);
            let mover = PlayerMover::new(eng_pos);
            assert!(
                mover.validate_position(&world.nodes, &world.leaves, &world.planes),
                "position {:?} should be clear",
                q_pos
            );
        }
    }

    #[test]
    fn player_mover_straight_junction_traversal() {
        let world = load_fixture("dungeon-junction-straight-bsp2.bsp");
        let qte = QuakeToEngine::default();
        let spawn_q = Vec3::new(-192.0, 0.0, 0.0);
        let spawn_eng = qte.position_vec3(spawn_q);

        let mut mover = PlayerMover::new(spawn_eng);

        // Move east through the corridor to the other room
        let delta_eng = Vec3::new(qte.scale * 384.0, 0.0, 0.0);
        mover.step(
            delta_eng,
            &world.clipnodes,
            &world.planes,
            &world.models,
            &world.nodes,
            &world.leaves,
            &world.planes,
            &qte,
            true,
        );

        let moved_q = Vec3::new(
            mover.position.x / qte.scale,
            -mover.position.z / qte.scale,
            mover.position.y / qte.scale,
        );
        assert!(
            moved_q.x > 100.0,
            "mover should reach east room, x={}",
            moved_q.x
        );
        assert!(mover.validate_position(&world.nodes, &world.leaves, &world.planes));
    }
}
