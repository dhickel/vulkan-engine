//! Reusable structural behavior adapters for BSP entities.
//!
//! Provides deterministic state machines for doors (`func_door`), buttons
//! (`func_button`), platforms (`func_plat`), triggers (`trigger_once`,
//! `trigger_multiple`), target relays, and light-style activation.
//!
//! All gameplay (AI, items, weapons, HUD) remains app-owned. This module
//! exposes only structural adapters that apps opt into.
//!
//! # Deterministic Update Ordering
//!
//! Entities are processed in order of their BSP entity index. Within a
//! single frame, triggers fire first, then doors/buttons/platforms update
//! their physics state. Activation cascades are resolved depth-first with
//! cycle detection.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// An activation signal sent through the target relay system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Turn on / open / activate.
    On,
    /// Turn off / close / deactivate.
    Off,
    /// Toggle between states.
    Toggle,
}

// ── Door ──────────────────────────────────────────────────────────────

/// Movement phase for a `func_door`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoorPhase {
    Closed,
    Opening,
    Open,
    Closing,
}

/// Runtime state for a single `func_door` entity.
#[derive(Debug, Clone)]
pub struct DoorState {
    pub entity_index: u32,
    pub phase: DoorPhase,
    /// Target name for activation.
    pub targetname: Option<String>,
    /// Target entity to fire on state change.
    pub target: Option<String>,
    /// Origin in engine space.
    pub origin: [f32; 3],
    /// Movement direction (unit vector).
    pub movedir: [f32; 3],
    /// Movement speed in engine units per second.
    pub speed: f32,
    /// How long to wait before auto-closing (seconds).
    pub wait: f32,
    /// How far the door sticks out from the wall (lip).
    pub lip: f32,
    /// Current travel fraction: 0.0 = closed, 1.0 = fully open.
    pub travel: f32,
    /// Timer for auto-close wait.
    pub wait_timer: f32,
    /// Start position (origin).
    pub start_position: [f32; 3],
    /// End position (origin + movedir * travel_distance).
    pub end_position: [f32; 3],
    /// Total travel distance.
    pub travel_distance: f32,
}

impl DoorState {
    pub fn new(
        entity_index: u32,
        targetname: Option<String>,
        target: Option<String>,
        origin: [f32; 3],
        movedir: [f32; 3],
        speed: f32,
        wait: f32,
        lip: f32,
    ) -> Self {
        let speed = if speed <= 0.0 { 100.0 } else { speed };
        let travel_distance = if lip > 0.0 { lip } else { 1.0 };
        let end = [
            origin[0] + movedir[0] * travel_distance,
            origin[1] + movedir[1] * travel_distance,
            origin[2] + movedir[2] * travel_distance,
        ];
        Self {
            entity_index,
            phase: DoorPhase::Closed,
            targetname,
            target,
            origin,
            movedir,
            speed,
            wait,
            lip,
            travel_distance,
            travel: 0.0,
            wait_timer: 0.0,
            start_position: origin,
            end_position: end,
        }
    }

    /// Advance the door state machine by `dt` seconds.
    /// Returns the current world-space position for scene/collider sync.
    pub fn update(&mut self, dt: f32) -> [f32; 3] {
        match self.phase {
            DoorPhase::Closed => self.origin,
            DoorPhase::Opening => {
                self.travel += self.speed * dt / self.travel_distance;
                if self.travel >= 1.0 {
                    self.travel = 1.0;
                    self.phase = DoorPhase::Open;
                    self.wait_timer = self.wait;
                }
                self.interpolate_position(self.travel)
            }
            DoorPhase::Open => {
                if self.wait > 0.0 {
                    self.wait_timer -= dt;
                    if self.wait_timer <= 0.0 {
                        self.phase = DoorPhase::Closing;
                    }
                }
                self.interpolate_position(1.0)
            }
            DoorPhase::Closing => {
                self.travel -= self.speed * dt / self.travel_distance;
                if self.travel <= 0.0 {
                    self.travel = 0.0;
                    self.phase = DoorPhase::Closed;
                }
                self.interpolate_position(self.travel)
            }
        }
    }

    /// Activate the door: toggle between closed and open.
    pub fn activate(&mut self, _activation: Activation) {
        match self.phase {
            DoorPhase::Closed => {
                self.phase = DoorPhase::Opening;
                self.travel = 0.0;
            }
            DoorPhase::Opening | DoorPhase::Open => {
                self.phase = DoorPhase::Closing;
            }
            DoorPhase::Closing => {
                self.phase = DoorPhase::Opening;
            }
        }
    }

    /// Get the target to fire when the door reaches a terminal state.
    pub fn terminal_target(&self) -> Option<&str> {
        match self.phase {
            DoorPhase::Open | DoorPhase::Closed => self.target.as_deref(),
            _ => None,
        }
    }

    fn interpolate_position(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + self.movedir[0] * self.travel_distance * t,
            self.origin[1] + self.movedir[1] * self.travel_distance * t,
            self.origin[2] + self.movedir[2] * self.travel_distance * t,
        ]
    }
}

// ── Button ────────────────────────────────────────────────────────────

/// Movement phase for a `func_button`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonPhase {
    Up,
    Pressing,
    Down,
    Returning,
}

/// Runtime state for a single `func_button` entity.
#[derive(Debug, Clone)]
pub struct ButtonState {
    pub entity_index: u32,
    pub phase: ButtonPhase,
    pub targetname: Option<String>,
    pub target: Option<String>,
    pub origin: [f32; 3],
    pub movedir: [f32; 3],
    pub speed: f32,
    pub wait: f32,
    pub lip: f32,
    pub travel: f32,
    pub travel_distance: f32,
    pub wait_timer: f32,
    pub start_position: [f32; 3],
    pub end_position: [f32; 3],
}

impl ButtonState {
    pub fn new(
        entity_index: u32,
        targetname: Option<String>,
        target: Option<String>,
        origin: [f32; 3],
        movedir: [f32; 3],
        speed: f32,
        wait: f32,
        lip: f32,
    ) -> Self {
        let speed = if speed <= 0.0 { 40.0 } else { speed };
        let travel_distance = if lip > 0.0 { lip } else { 4.0 };
        let end = [
            origin[0] + movedir[0] * travel_distance,
            origin[1] + movedir[1] * travel_distance,
            origin[2] + movedir[2] * travel_distance,
        ];
        Self {
            entity_index,
            phase: ButtonPhase::Up,
            targetname,
            target,
            origin,
            movedir,
            speed,
            wait,
            lip,
            travel_distance,
            travel: 0.0,
            wait_timer: 0.0,
            start_position: origin,
            end_position: end,
        }
    }

    pub fn update(&mut self, dt: f32) -> [f32; 3] {
        match self.phase {
            ButtonPhase::Up => self.origin,
            ButtonPhase::Pressing => {
                self.travel += self.speed * dt / self.travel_distance;
                if self.travel >= 1.0 {
                    self.travel = 1.0;
                    self.phase = ButtonPhase::Down;
                    self.wait_timer = self.wait;
                }
                self.interpolate_position(self.travel)
            }
            ButtonPhase::Down => {
                if self.wait > 0.0 {
                    self.wait_timer -= dt;
                    if self.wait_timer <= 0.0 {
                        self.phase = ButtonPhase::Returning;
                    }
                }
                self.interpolate_position(1.0)
            }
            ButtonPhase::Returning => {
                self.travel -= self.speed * dt / self.travel_distance;
                if self.travel <= 0.0 {
                    self.travel = 0.0;
                    self.phase = ButtonPhase::Up;
                }
                self.interpolate_position(self.travel)
            }
        }
    }

    pub fn activate(&mut self, _activation: Activation) {
        if self.phase == ButtonPhase::Up {
            self.phase = ButtonPhase::Pressing;
            self.travel = 0.0;
        }
    }

    fn interpolate_position(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + self.movedir[0] * self.travel_distance * t,
            self.origin[1] + self.movedir[1] * self.travel_distance * t,
            self.origin[2] + self.movedir[2] * self.travel_distance * t,
        ]
    }
}

// ── Platform ──────────────────────────────────────────────────────────

/// Movement phase for a `func_plat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformPhase {
    Low,
    Raising,
    High,
    Lowering,
}

/// Runtime state for a single `func_plat` (elevator/lift) entity.
#[derive(Debug, Clone)]
pub struct PlatformState {
    pub entity_index: u32,
    pub phase: PlatformPhase,
    pub targetname: Option<String>,
    pub target: Option<String>,
    pub origin: [f32; 3],
    pub height: f32,
    pub speed: f32,
    pub lip: f32,
    pub travel: f32,
    pub start_position: [f32; 3],
    pub end_position: [f32; 3],
    pub wait_timer: f32,
}

impl PlatformState {
    pub fn new(
        entity_index: u32,
        targetname: Option<String>,
        target: Option<String>,
        origin: [f32; 3],
        height: f32,
        speed: f32,
        lip: f32,
    ) -> Self {
        let speed = if speed <= 0.0 { 150.0 } else { speed };
        let effective_height = if height > 0.0 { height } else { lip };
        let end = [origin[0], origin[1] + effective_height, origin[2]];
        Self {
            entity_index,
            phase: PlatformPhase::Low,
            targetname,
            target,
            origin,
            height: effective_height,
            speed,
            lip,
            travel: 0.0,
            start_position: origin,
            end_position: end,
            wait_timer: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32) -> [f32; 3] {
        match self.phase {
            PlatformPhase::Low => self.origin,
            PlatformPhase::Raising => {
                self.travel += self.speed * dt / self.height.max(0.001);
                if self.travel >= 1.0 {
                    self.travel = 1.0;
                    self.phase = PlatformPhase::High;
                    self.wait_timer = 3.0;
                }
                self.interpolate_position(self.travel)
            }
            PlatformPhase::High => {
                self.wait_timer -= dt;
                if self.wait_timer <= 0.0 {
                    self.phase = PlatformPhase::Lowering;
                }
                self.interpolate_position(1.0)
            }
            PlatformPhase::Lowering => {
                self.travel -= self.speed * dt / self.height.max(0.001);
                if self.travel <= 0.0 {
                    self.travel = 0.0;
                    self.phase = PlatformPhase::Low;
                }
                self.interpolate_position(self.travel)
            }
        }
    }

    pub fn activate(&mut self, _activation: Activation) {
        match self.phase {
            PlatformPhase::Low => {
                self.phase = PlatformPhase::Raising;
                self.travel = 0.0;
            }
            PlatformPhase::High => {
                self.phase = PlatformPhase::Lowering;
                self.travel = 1.0;
            }
            _ => {}
        }
    }

    fn interpolate_position(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0],
            self.origin[1] + self.height * t,
            self.origin[2],
        ]
    }
}

// ── Trigger ───────────────────────────────────────────────────────────

/// Runtime state for a trigger entity.
#[derive(Debug, Clone)]
pub struct TriggerState {
    pub entity_index: u32,
    /// Classname: "trigger_once" or "trigger_multiple".
    pub classname: String,
    /// Target entity name to fire when triggered.
    pub target: Option<String>,
    /// Target entity name to remove when triggered.
    pub killtarget: Option<String>,
    /// Set of entities currently intersecting the trigger volume.
    pub current_occupants: HashSet<u32>,
    /// Whether the trigger has already fired (for trigger_once).
    pub fired: bool,
}

impl TriggerState {
    pub fn new(
        entity_index: u32,
        classname: String,
        target: Option<String>,
        killtarget: Option<String>,
    ) -> Self {
        Self {
            entity_index,
            classname,
            target,
            killtarget,
            current_occupants: HashSet::new(),
            fired: false,
        }
    }

    /// Process occupant changes. Returns `true` if the trigger fired.
    pub fn update_occupants(&mut self, new_occupants: HashSet<u32>) -> TriggerEvent {
        // Determine enter/stay/exit
        let entered: HashSet<_> = new_occupants
            .difference(&self.current_occupants)
            .copied()
            .collect();
        let _staying: HashSet<_> = new_occupants
            .intersection(&self.current_occupants)
            .copied()
            .collect();
        let _exited: HashSet<_> = self
            .current_occupants
            .difference(&new_occupants)
            .copied()
            .collect();

        self.current_occupants = new_occupants;

        let should_fire = if self.classname == "trigger_once" {
            if self.fired {
                false
            } else if !entered.is_empty() {
                self.fired = true;
                true
            } else {
                false
            }
        } else {
            // trigger_multiple: fire on any new entry, or if anyone is still inside
            !self.current_occupants.is_empty() && !entered.is_empty()
        };

        if should_fire {
            TriggerEvent::Fired {
                trigger_entity: self.entity_index,
                target: self.target.clone(),
                killtarget: self.killtarget.clone(),
            }
        } else if self.current_occupants.is_empty() {
            TriggerEvent::Idle
        } else {
            TriggerEvent::Occupied
        }
    }

    /// Reset the trigger (for reload/unload).
    pub fn reset(&mut self) {
        self.current_occupants.clear();
        self.fired = false;
    }
}

/// Events emitted by trigger processing.
#[derive(Debug, Clone)]
pub enum TriggerEvent {
    /// Trigger is idle (no occupants, not fired).
    Idle,
    /// Trigger is occupied but not firing.
    Occupied,
    /// Trigger fired. Contains target and killtarget to activate.
    Fired {
        trigger_entity: u32,
        target: Option<String>,
        killtarget: Option<String>,
    },
}

// ── Light Style ───────────────────────────────────────────────────────

/// Light style activation state.
#[derive(Debug, Clone)]
pub struct LightStyleState {
    /// Whether the style is currently active (on).
    pub active: bool,
    /// Base intensity multiplier.
    pub intensity: f32,
}

impl Default for LightStyleState {
    fn default() -> Self {
        Self {
            active: true,
            intensity: 1.0,
        }
    }
}

// ── Structural Behavior Adapter ───────────────────────────────────────

/// Top-level adapter that manages all structural behaviors.
///
/// Doors, buttons, platforms, triggers, targets, and light styles are
/// updated deterministically by entity index. Activation cascades are
/// resolved depth-first with cycle detection.
#[derive(Debug, Clone)]
pub struct StructuralBehaviorAdapter {
    /// Entity index → door state.
    pub doors: HashMap<u32, DoorState>,
    /// Entity index → button state.
    pub buttons: HashMap<u32, ButtonState>,
    /// Entity index → platform state.
    pub platforms: HashMap<u32, PlatformState>,
    /// Entity index → trigger state.
    pub triggers: HashMap<u32, TriggerState>,
    /// Light style name → state.
    pub light_styles: HashMap<String, LightStyleState>,
    /// targetname → deterministic set of entity indices that have this targetname.
    target_map: HashMap<String, BTreeSet<u32>>,
    /// entity_index → target string (who this entity targets).
    entity_targets: HashMap<u32, String>,
    /// entity_index → killtarget string.
    entity_killtargets: HashMap<u32, String>,
    /// Activation queue for resolving cascades.
    activation_queue: VecDeque<(u32, Activation)>,
}

impl StructuralBehaviorAdapter {
    pub fn new() -> Self {
        Self {
            doors: HashMap::new(),
            buttons: HashMap::new(),
            platforms: HashMap::new(),
            triggers: HashMap::new(),
            light_styles: HashMap::new(),
            target_map: HashMap::new(),
            entity_targets: HashMap::new(),
            entity_killtargets: HashMap::new(),
            activation_queue: VecDeque::new(),
        }
    }

    /// Register behavioral entities from BSP entity descriptors.
    ///
    /// Called during prepare or when a map is loaded. Builds target lookup
    /// tables and initializes default state for recognized entities.
    pub fn register_entities<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = BehaviorEntityInfo>,
    {
        for info in entities {
            // Register target mappings
            if let Some(ref tn) = info.targetname {
                self.target_map
                    .entry(tn.clone())
                    .or_default()
                    .insert(info.entity_index);
            }
            if let Some(ref t) = info.target {
                self.entity_targets.insert(info.entity_index, t.clone());
            }
            if let Some(ref kt) = info.killtarget {
                self.entity_killtargets
                    .insert(info.entity_index, kt.clone());
            }

            // Initialize state machines based on classname
            match info.classname.as_str() {
                "func_door" | "func_door_secret" => {
                    let door = DoorState::new(
                        info.entity_index,
                        info.targetname,
                        info.target,
                        info.origin,
                        info.movedir.unwrap_or([1.0, 0.0, 0.0]),
                        info.speed.unwrap_or(100.0),
                        info.wait.unwrap_or(3.0),
                        info.lip.unwrap_or(0.0),
                    );
                    self.doors.insert(info.entity_index, door);
                }
                "func_button" => {
                    let button = ButtonState::new(
                        info.entity_index,
                        info.targetname,
                        info.target,
                        info.origin,
                        info.movedir.unwrap_or([0.0, 0.0, 1.0]),
                        info.speed.unwrap_or(40.0),
                        info.wait.unwrap_or(1.0),
                        info.lip.unwrap_or(4.0),
                    );
                    self.buttons.insert(info.entity_index, button);
                }
                "func_plat" => {
                    let plat = PlatformState::new(
                        info.entity_index,
                        info.targetname,
                        info.target,
                        info.origin,
                        info.height.unwrap_or(128.0),
                        info.speed.unwrap_or(150.0),
                        info.lip.unwrap_or(8.0),
                    );
                    self.platforms.insert(info.entity_index, plat);
                }
                "trigger_once" | "trigger_multiple" => {
                    let trigger = TriggerState::new(
                        info.entity_index,
                        info.classname.clone(),
                        info.target,
                        info.killtarget,
                    );
                    self.triggers.insert(info.entity_index, trigger);
                }
                "light" | "light_flame" | "light_fluoro" | "light_torch" => {
                    if let Some(ref style) = info.light_style {
                        self.light_styles.entry(style.clone()).or_default();
                    }
                }
                _ => {}
            }
        }
    }

    /// Advance all state machines by `dt` seconds.
    ///
    /// Returns a list of `(entity_index, new_world_position)` for entities
    /// whose position changed (for scene/collider transform sync).
    pub fn update(&mut self, dt: f32) -> Vec<(u32, [f32; 3])> {
        // Process activation queue (resolve cascades)
        self.process_activation_queue();

        let mut updates = Vec::new();

        // Sort keys for deterministic ordering
        // Collect target activations to avoid double-borrow during door iteration
        let mut pending_targets: Vec<String> = Vec::new();
        let mut door_indices: Vec<u32> = self.doors.keys().copied().collect();
        door_indices.sort();
        for ei in door_indices {
            let door = self.doors.get_mut(&ei).unwrap();
            let prev_phase = door.phase;
            let pos = door.update(dt);
            if door.phase != prev_phase
                || door.phase == DoorPhase::Opening
                || door.phase == DoorPhase::Closing
            {
                updates.push((ei, pos));
            }
            // Fire targets once when a door reaches a terminal state.
            let entered_terminal = !matches!(prev_phase, DoorPhase::Open | DoorPhase::Closed)
                && matches!(door.phase, DoorPhase::Open | DoorPhase::Closed);
            if entered_terminal {
                if let Some(target_name) = door.terminal_target() {
                    pending_targets.push(target_name.to_string());
                }
            }
        }
        for target_name in &pending_targets {
            self.queue_target_activation(target_name, Activation::On);
        }

        let mut button_indices: Vec<u32> = self.buttons.keys().copied().collect();
        button_indices.sort();
        for ei in button_indices {
            let button = self.buttons.get_mut(&ei).unwrap();
            let prev_phase = button.phase;
            let pos = button.update(dt);
            if button.phase != prev_phase
                || button.phase == ButtonPhase::Pressing
                || button.phase == ButtonPhase::Returning
            {
                updates.push((ei, pos));
            }
        }

        let mut plat_indices: Vec<u32> = self.platforms.keys().copied().collect();
        plat_indices.sort();
        for ei in plat_indices {
            let plat = self.platforms.get_mut(&ei).unwrap();
            let prev_phase = plat.phase;
            let pos = plat.update(dt);
            if plat.phase != prev_phase
                || plat.phase == PlatformPhase::Raising
                || plat.phase == PlatformPhase::Lowering
            {
                updates.push((ei, pos));
            }
        }

        // Collect all updates then sort by entity index for deterministic ordering
        updates.sort_by_key(|(ei, _)| *ei);

        // Process activation queue again for cascades from terminal targets
        self.process_activation_queue();

        updates
    }

    /// Activate an entity by target name.
    pub fn activate_by_target(&mut self, target_name: &str, activation: Activation) {
        self.queue_target_activation(target_name, activation);
        self.process_activation_queue();
    }

    /// Activate an entity by entity index.
    pub fn activate_by_index(&mut self, entity_index: u32, activation: Activation) {
        if let Some(door) = self.doors.get_mut(&entity_index) {
            door.activate(activation);
        } else if let Some(button) = self.buttons.get_mut(&entity_index) {
            button.activate(activation);
        } else if let Some(plat) = self.platforms.get_mut(&entity_index) {
            plat.activate(activation);
        }
        // Fire any target this entity has
        if let Some(target_name) = self.entity_targets.get(&entity_index).cloned() {
            self.queue_target_activation(&target_name, activation);
        }
        self.process_activation_queue();
    }

    /// Set light style active state.
    pub fn set_light_style(&mut self, style: &str, active: bool) {
        if let Some(state) = self.light_styles.get_mut(style) {
            state.active = active;
        }
    }

    /// Set light style intensity.
    pub fn set_light_style_intensity(&mut self, style: &str, intensity: f32) {
        if let Some(state) = self.light_styles.get_mut(style) {
            state.intensity = intensity.clamp(0.0, 1.0);
            state.active = intensity > 0.0;
        }
    }

    /// Get light style active state.
    pub fn light_style_active(&self, style: &str) -> bool {
        self.light_styles
            .get(style)
            .map(|s| s.active)
            .unwrap_or(true)
    }

    /// Reset all state (for reload/unload).
    pub fn reset(&mut self) {
        for trigger in self.triggers.values_mut() {
            trigger.reset();
        }
        for door in self.doors.values_mut() {
            door.phase = DoorPhase::Closed;
            door.travel = 0.0;
            door.wait_timer = 0.0;
        }
        for button in self.buttons.values_mut() {
            button.phase = ButtonPhase::Up;
            button.travel = 0.0;
            button.wait_timer = 0.0;
        }
        for plat in self.platforms.values_mut() {
            plat.phase = PlatformPhase::Low;
            plat.travel = 0.0;
            plat.wait_timer = 0.0;
        }
        self.activation_queue.clear();
    }

    /// Check if an entity is currently in motion.
    pub fn is_moving(&self, entity_index: u32) -> bool {
        if let Some(door) = self.doors.get(&entity_index) {
            matches!(door.phase, DoorPhase::Opening | DoorPhase::Closing)
        } else if let Some(button) = self.buttons.get(&entity_index) {
            matches!(button.phase, ButtonPhase::Pressing | ButtonPhase::Returning)
        } else if let Some(plat) = self.platforms.get(&entity_index) {
            matches!(plat.phase, PlatformPhase::Raising | PlatformPhase::Lowering)
        } else {
            false
        }
    }

    /// Get the current world position of a moving entity.
    pub fn entity_position(&self, entity_index: u32) -> Option<[f32; 3]> {
        if let Some(door) = self.doors.get(&entity_index) {
            Some(door.interpolate_position(door.travel))
        } else if let Some(button) = self.buttons.get(&entity_index) {
            Some(button.interpolate_position(button.travel))
        } else if let Some(plat) = self.platforms.get(&entity_index) {
            Some(plat.interpolate_position(plat.travel))
        } else {
            None
        }
    }

    /// Get the trigger state for an entity (if it's a trigger).
    pub fn trigger_state(&self, entity_index: u32) -> Option<&TriggerState> {
        self.triggers.get(&entity_index)
    }

    /// Update trigger occupants and return any firing events.
    pub fn update_trigger_occupants(
        &mut self,
        entity_index: u32,
        new_occupants: HashSet<u32>,
    ) -> Option<TriggerEvent> {
        let trigger = self.triggers.get_mut(&entity_index)?;
        let event = trigger.update_occupants(new_occupants);

        if let TriggerEvent::Fired {
            ref target,
            ref killtarget,
            ..
        } = &event
        {
            if let Some(t) = target {
                self.queue_target_activation(t, Activation::On);
            }
            if let Some(kt) = killtarget {
                self.queue_target_kill(kt);
            }
            self.process_activation_queue();
        }

        Some(event)
    }

    // ── Internal ───────────────────────────────────────────────────────

    fn queue_target_activation(&mut self, target_name: &str, activation: Activation) {
        if let Some(targets) = self.target_map.get(target_name) {
            for &ei in targets {
                self.activation_queue.push_back((ei, activation));
            }
        }
    }

    fn queue_target_kill(&mut self, target_name: &str) {
        if let Some(targets) = self.target_map.get(target_name) {
            for &ei in targets {
                self.activation_queue.push_back((ei, Activation::Off));
            }
        }
    }

    fn process_activation_queue(&mut self) {
        // Cycle detection: track how many times each entity has been activated
        // in this cascade to prevent infinite loops.
        let mut activation_count: HashMap<u32, usize> = HashMap::new();
        const MAX_CASCADE_DEPTH: usize = 3;

        while let Some((ei, activation)) = self.activation_queue.pop_front() {
            let count = activation_count.entry(ei).or_insert(0);
            *count += 1;
            if *count > MAX_CASCADE_DEPTH {
                // Cycle detected, stop cascading
                continue;
            }

            // Activate the entity itself
            if let Some(door) = self.doors.get_mut(&ei) {
                door.activate(activation);
            } else if let Some(button) = self.buttons.get_mut(&ei) {
                button.activate(activation);
            } else if let Some(plat) = self.platforms.get_mut(&ei) {
                plat.activate(activation);
            }

            // Fire its target chain
            if let Some(target_name) = self.entity_targets.get(&ei).cloned() {
                if let Some(targets) = self.target_map.get(&target_name) {
                    for &t_ei in targets {
                        self.activation_queue.push_back((t_ei, activation));
                    }
                }
            }
        }
    }

    // ── Persistence ─────────────────────────────────────────────────

    /// Export current mutable behavior state for persistence.
    pub fn export_state(&self) -> crate::source_link::MutableBehaviorState {
        use crate::source_link::{
            CanonicalFloat, SerializedButtonState, SerializedDoorState,
            SerializedPlatformState, SerializedTriggerState,
        };
        let mut state = crate::source_link::MutableBehaviorState::default();

        let mut door_indices: Vec<u32> = self.doors.keys().copied().collect();
        door_indices.sort();
        for ei in door_indices {
            let d = &self.doors[&ei];
            let phase: u8 = match d.phase {
                DoorPhase::Closed => 0,
                DoorPhase::Opening => 1,
                DoorPhase::Open => 2,
                DoorPhase::Closing => 3,
            };
            state.doors.push(SerializedDoorState {
                entity_index: ei,
                phase,
                travel: CanonicalFloat(d.travel),
                wait_timer: CanonicalFloat(d.wait_timer),
            });
        }

        let mut button_indices: Vec<u32> = self.buttons.keys().copied().collect();
        button_indices.sort();
        for ei in button_indices {
            let b = &self.buttons[&ei];
            let phase: u8 = match b.phase {
                ButtonPhase::Up => 0,
                ButtonPhase::Pressing => 1,
                ButtonPhase::Down => 2,
                ButtonPhase::Returning => 3,
            };
            state.buttons.push(SerializedButtonState {
                entity_index: ei,
                phase,
                travel: CanonicalFloat(b.travel),
                wait_timer: CanonicalFloat(b.wait_timer),
            });
        }

        let mut plat_indices: Vec<u32> = self.platforms.keys().copied().collect();
        plat_indices.sort();
        for ei in plat_indices {
            let p = &self.platforms[&ei];
            let phase: u8 = match p.phase {
                PlatformPhase::Low => 0,
                PlatformPhase::Raising => 1,
                PlatformPhase::High => 2,
                PlatformPhase::Lowering => 3,
            };
            state.platforms.push(SerializedPlatformState {
                entity_index: ei,
                phase,
                travel: CanonicalFloat(p.travel),
                wait_timer: CanonicalFloat(p.wait_timer),
            });
        }

        let mut trigger_indices: Vec<u32> = self.triggers.keys().copied().collect();
        trigger_indices.sort();
        for ei in trigger_indices {
            let t = &self.triggers[&ei];
            state.triggers.push(SerializedTriggerState {
                entity_index: ei,
                fired: t.fired,
            });
        }

        let mut style_keys: Vec<&String> = self.light_styles.keys().collect();
        style_keys.sort();
        for key in style_keys {
            if let Ok(idx) = key.parse::<u32>() {
                if idx <= 63 {
                    let ls = &self.light_styles[key];
                    state.light_styles.insert(idx, CanonicalFloat(ls.intensity));
                }
            }
        }

        state
    }

    /// Import mutable behavior state from a persistence payload.
    ///
    /// Only entities that exist in the adapter (from the current BSP)
    /// are restored. Entities in the payload that don't match the current
    /// BSP are ignored.
    pub fn import_state(&mut self, state: &crate::source_link::MutableBehaviorState) {
        for sd in &state.doors {
            if let Some(door) = self.doors.get_mut(&sd.entity_index) {
                door.phase = match sd.phase {
                    0 => DoorPhase::Closed,
                    1 => DoorPhase::Opening,
                    2 => DoorPhase::Open,
                    3 => DoorPhase::Closing,
                    _ => DoorPhase::Closed,
                };
                door.travel = sd.travel.0.clamp(0.0, 1.0);
                door.wait_timer = sd.wait_timer.0.max(0.0);
            }
        }
        for sb in &state.buttons {
            if let Some(button) = self.buttons.get_mut(&sb.entity_index) {
                button.phase = match sb.phase {
                    0 => ButtonPhase::Up,
                    1 => ButtonPhase::Pressing,
                    2 => ButtonPhase::Down,
                    3 => ButtonPhase::Returning,
                    _ => ButtonPhase::Up,
                };
                button.travel = sb.travel.0.clamp(0.0, 1.0);
                button.wait_timer = sb.wait_timer.0.max(0.0);
            }
        }
        for sp in &state.platforms {
            if let Some(plat) = self.platforms.get_mut(&sp.entity_index) {
                plat.phase = match sp.phase {
                    0 => PlatformPhase::Low,
                    1 => PlatformPhase::Raising,
                    2 => PlatformPhase::High,
                    3 => PlatformPhase::Lowering,
                    _ => PlatformPhase::Low,
                };
                plat.travel = sp.travel.0.clamp(0.0, 1.0);
                plat.wait_timer = sp.wait_timer.0.max(0.0);
            }
        }
        for st in &state.triggers {
            if let Some(trigger) = self.triggers.get_mut(&st.entity_index) {
                trigger.fired = st.fired;
            }
        }
        for (style_id, intensity) in &state.light_styles {
            let style_key = style_id.to_string();
            // Light styles are keyed by string name; try to find the matching style.
            for (name, ls) in self.light_styles.iter_mut() {
                if *name == style_key {
                    ls.intensity = intensity.0.clamp(0.0, 1.0);
                    ls.active = intensity.0 > 0.0;
                }
            }
        }
    }
}

impl Default for StructuralBehaviorAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Behavior Entity Info ──────────────────────────────────────────────

/// Information needed to register a structural behavior entity.
///
/// This is a neutral DTO that the app bridge or coordinator can construct
/// from BSP entity descriptors.
#[derive(Debug, Clone)]
pub struct BehaviorEntityInfo {
    pub entity_index: u32,
    pub classname: String,
    pub targetname: Option<String>,
    pub target: Option<String>,
    pub killtarget: Option<String>,
    pub origin: [f32; 3],
    pub movedir: Option<[f32; 3]>,
    pub speed: Option<f32>,
    pub wait: Option<f32>,
    pub lip: Option<f32>,
    pub height: Option<f32>,
    pub light_style: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn door_cycle() {
        let mut door = DoorState::new(
            0,
            None,
            None,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            100.0,
            1.0,
            0.0,
        );
        // Closed initially
        assert_eq!(door.phase, DoorPhase::Closed);
        assert_eq!(door.update(0.1), [0.0, 0.0, 0.0]);

        // Activate → Opening
        door.activate(Activation::Toggle);
        assert_eq!(door.phase, DoorPhase::Opening);
        let pos_half = door.update(0.005); // Should move half-way at speed 100, dist 1
        assert!(pos_half[0] > 0.0 && pos_half[0] < 1.0);

        // Complete opening
        let pos_open = door.update(0.1); // Should finish
        assert_eq!(door.phase, DoorPhase::Open);
        assert!((pos_open[0] - 1.0).abs() < 0.001);

        // Wait timer expires — set timer directly to skip wait
        door.wait_timer = 0.0;
        door.update(0.1);
        assert_eq!(door.phase, DoorPhase::Closing);

        // Complete closing
        door.update(0.1);
        assert_eq!(door.phase, DoorPhase::Closed);
        assert!((door.interpolate_position(0.0)[0]).abs() < 0.001);
    }

    #[test]
    fn button_press_and_return() {
        let mut button = ButtonState::new(
            1,
            None,
            None,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            80.0,
            0.5,
            4.0,
        );
        assert_eq!(button.phase, ButtonPhase::Up);
        button.activate(Activation::On);
        assert_eq!(button.phase, ButtonPhase::Pressing);

        // Press fully
        button.update(1.0);
        assert_eq!(button.phase, ButtonPhase::Down);

        // Wait then return
        button.wait_timer = 0.0; // skip wait
        button.update(0.1);
        assert_eq!(button.phase, ButtonPhase::Returning);
    }

    #[test]
    fn platform_raise_lower() {
        let mut plat = PlatformState::new(
            2,
            Some("lift1".into()),
            Some("door_exit".into()),
            [0.0, 0.0, 0.0],
            64.0,
            150.0,
            8.0,
        );
        assert_eq!(plat.phase, PlatformPhase::Low);
        plat.activate(Activation::Toggle);
        assert_eq!(plat.phase, PlatformPhase::Raising);

        // Raise fully
        plat.update(1.0);
        assert_eq!(plat.phase, PlatformPhase::High);

        // Wait expires, start lowering
        plat.wait_timer = 0.0;
        plat.update(0.1);
        assert_eq!(plat.phase, PlatformPhase::Lowering);
    }

    #[test]
    fn trigger_once_fires_once() {
        let mut trigger = TriggerState::new(3, "trigger_once".into(), Some("door1".into()), None);
        let event = trigger.update_occupants(HashSet::from([1, 2]));
        assert!(matches!(event, TriggerEvent::Fired { .. }));
        assert!(trigger.fired);

        // Second time should not fire
        let event2 = trigger.update_occupants(HashSet::from([1, 2]));
        assert!(matches!(event2, TriggerEvent::Occupied));
    }

    #[test]
    fn trigger_multiple_fires_on_each_entry() {
        let mut trigger =
            TriggerState::new(4, "trigger_multiple".into(), Some("door2".into()), None);
        // First entry fires
        let event = trigger.update_occupants(HashSet::from([1]));
        assert!(matches!(event, TriggerEvent::Fired { .. }));

        // Same occupant staying → no fire
        let event2 = trigger.update_occupants(HashSet::from([1]));
        assert!(matches!(event2, TriggerEvent::Occupied));

        // New occupant enters → fires again
        let event3 = trigger.update_occupants(HashSet::from([1, 2]));
        assert!(matches!(event3, TriggerEvent::Fired { .. }));
    }

    #[test]
    fn trigger_empty_goes_idle() {
        let mut trigger = TriggerState::new(5, "trigger_multiple".into(), None, None);
        trigger.update_occupants(HashSet::from([1]));
        let event = trigger.update_occupants(HashSet::new());
        assert!(matches!(event, TriggerEvent::Idle));
    }

    #[test]
    fn adapter_deterministic_update_order() {
        let mut adapter = StructuralBehaviorAdapter::new();
        adapter.register_entities(vec![
            BehaviorEntityInfo {
                entity_index: 10,
                classname: "func_door".into(),
                targetname: Some("door_a".into()),
                target: None,
                killtarget: None,
                origin: [0.0, 0.0, 0.0],
                movedir: Some([1.0, 0.0, 0.0]),
                speed: Some(200.0),
                wait: Some(2.0),
                lip: Some(0.0),
                height: None,
                light_style: None,
            },
            BehaviorEntityInfo {
                entity_index: 5,
                classname: "func_button".into(),
                targetname: Some("btn_a".into()),
                target: Some("door_a".into()),
                killtarget: None,
                origin: [10.0, 0.0, 5.0],
                movedir: Some([0.0, 0.0, 1.0]),
                speed: Some(40.0),
                wait: Some(1.0),
                lip: Some(4.0),
                height: None,
                light_style: None,
            },
        ]);

        // Button 5 targets door 10
        adapter.activate_by_index(5, Activation::On);
        let door = adapter.doors.get(&10).unwrap();
        // Door should be opening because button activated it
        assert_eq!(door.phase, DoorPhase::Opening);
    }

    #[test]
    fn activation_cascade_cycle_detection() {
        let mut adapter = StructuralBehaviorAdapter::new();
        adapter.register_entities(vec![
            BehaviorEntityInfo {
                entity_index: 1,
                classname: "func_door".into(),
                targetname: Some("a".into()),
                target: Some("b".into()),
                killtarget: None,
                origin: [0.0, 0.0, 0.0],
                movedir: Some([1.0, 0.0, 0.0]),
                speed: Some(100.0),
                wait: Some(1.0),
                lip: Some(0.0),
                height: None,
                light_style: None,
            },
            BehaviorEntityInfo {
                entity_index: 2,
                classname: "func_door".into(),
                targetname: Some("b".into()),
                target: Some("a".into()),
                killtarget: None,
                origin: [5.0, 0.0, 0.0],
                movedir: Some([1.0, 0.0, 0.0]),
                speed: Some(100.0),
                wait: Some(1.0),
                lip: Some(0.0),
                height: None,
                light_style: None,
            },
        ]);

        // This should not infinite loop
        adapter.activate_by_target("a", Activation::On);
        // Both doors should be opening (activated)
        let door_a = adapter.doors.get(&1).unwrap();
        let door_b = adapter.doors.get(&2).unwrap();
        assert_eq!(door_a.phase, DoorPhase::Opening);
        assert_eq!(door_b.phase, DoorPhase::Opening);
    }

    #[test]
    fn light_style_toggle() {
        let mut adapter = StructuralBehaviorAdapter::new();
        adapter.register_entities(vec![BehaviorEntityInfo {
            entity_index: 3,
            classname: "light".into(),
            targetname: None,
            target: None,
            killtarget: None,
            origin: [0.0, 5.0, 0.0],
            movedir: None,
            speed: None,
            wait: None,
            lip: None,
            height: None,
            light_style: Some("flicker".into()),
        }]);

        assert!(adapter.light_style_active("flicker"));
        adapter.set_light_style("flicker", false);
        assert!(!adapter.light_style_active("flicker"));
        adapter.set_light_style("flicker", true);
        assert!(adapter.light_style_active("flicker"));
    }

    #[test]
    fn adapter_reset_clears_all_state() {
        let mut adapter = StructuralBehaviorAdapter::new();
        adapter.register_entities(vec![BehaviorEntityInfo {
            entity_index: 1,
            classname: "func_door".into(),
            targetname: None,
            target: None,
            killtarget: None,
            origin: [0.0, 0.0, 0.0],
            movedir: Some([1.0, 0.0, 0.0]),
            speed: Some(100.0),
            wait: Some(1.0),
            lip: Some(0.0),
            height: None,
            light_style: None,
        }]);

        adapter.activate_by_index(1, Activation::On);
        assert_eq!(adapter.doors.get(&1).unwrap().phase, DoorPhase::Opening);

        adapter.reset();
        assert_eq!(adapter.doors.get(&1).unwrap().phase, DoorPhase::Closed);
    }

    #[test]
    fn is_moving_reports_correctly() {
        let mut adapter = StructuralBehaviorAdapter::new();
        adapter.register_entities(vec![BehaviorEntityInfo {
            entity_index: 1,
            classname: "func_door".into(),
            targetname: None,
            target: None,
            killtarget: None,
            origin: [0.0, 0.0, 0.0],
            movedir: Some([1.0, 0.0, 0.0]),
            speed: Some(100.0),
            wait: Some(1.0),
            lip: Some(0.0),
            height: None,
            light_style: None,
        }]);

        assert!(!adapter.is_moving(1));
        adapter.activate_by_index(1, Activation::On);
        assert!(adapter.is_moving(1));
    }
}
