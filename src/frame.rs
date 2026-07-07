//! Small frame timing helpers for app-owned runtime loops.

use std::time::{Duration, Instant};

use engine_events::{DispatchReport, EventBus};

use crate::events::RuntimeEventDispatcher;
use crate::input::{InputActionEventEmitter, InputSystem};

/// Frame timing snapshot returned by [`FrameClock`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameInfo {
    pub index: u64,
    pub delta: Duration,
    pub delta_seconds: f32,
}

/// Report returned after beginning one app-owned frame.
#[derive(Clone, Debug, PartialEq)]
pub struct AppFrameBeginReport {
    pub frame: FrameInfo,
    pub action_events_emitted: usize,
    pub input_dispatch: DispatchReport,
    pub frame_started: DispatchReport,
}

/// Report returned after ending one app-owned frame.
#[derive(Clone, Debug, PartialEq)]
pub struct AppFrameEndReport {
    pub frame_index: u64,
    pub frame_ended: DispatchReport,
}

/// Fixed-step simulation timing configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedStepConfig {
    pub step: Duration,
    pub max_steps_per_frame: u32,
}

/// Result of advancing a [`FixedStepClock`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedStepUpdate {
    pub steps: u32,
    pub alpha: f32,
    pub accumulated: Duration,
    pub dropped_time: Duration,
}

/// Pure fixed-step accumulator for app-owned update loops.
#[derive(Clone, Debug)]
pub struct FixedStepClock {
    config: FixedStepConfig,
    accumulator: Duration,
}

/// Monotonic frame clock for app-owned loops.
#[derive(Clone, Debug)]
pub struct FrameClock {
    next_index: u64,
    last_tick: Instant,
}

impl Default for FrameClock {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameClock {
    pub fn new() -> Self {
        Self::from_instant(Instant::now())
    }

    pub fn from_instant(start: Instant) -> Self {
        Self {
            next_index: 0,
            last_tick: start,
        }
    }

    pub fn tick(&mut self) -> FrameInfo {
        self.tick_at(Instant::now())
    }

    pub fn tick_at(&mut self, now: Instant) -> FrameInfo {
        let delta = now
            .checked_duration_since(self.last_tick)
            .unwrap_or(Duration::ZERO);
        self.last_tick = now;

        let info = FrameInfo {
            index: self.next_index,
            delta,
            delta_seconds: delta.as_secs_f32(),
        };
        self.next_index = self.next_index.saturating_add(1);
        info
    }

    pub fn next_index(&self) -> u64 {
        self.next_index
    }
}

/// Begin one app-owned frame by ticking time, dispatching input, emitting input
/// actions, draining input events, and broadcasting `FrameStarted`.
pub fn begin_app_frame(
    input: &mut InputSystem,
    action_events: &mut InputActionEventEmitter,
    events: &mut EventBus,
    frame_clock: &mut FrameClock,
) -> AppFrameBeginReport {
    let frame = frame_clock.tick();
    input.dispatch_frame();
    let action_events_emitted =
        action_events.emit_from_snapshot(events, input.snapshot(), frame.index);
    let input_dispatch = RuntimeEventDispatcher::drain_input(events);
    let frame_started = RuntimeEventDispatcher::frame_started(events, frame.index);

    AppFrameBeginReport {
        frame,
        action_events_emitted,
        input_dispatch,
        frame_started,
    }
}

/// End one app-owned frame by broadcasting `FrameEnded`.
pub fn end_app_frame(events: &mut EventBus, frame_index: u64) -> AppFrameEndReport {
    let frame_ended = RuntimeEventDispatcher::frame_ended(events, frame_index);

    AppFrameEndReport {
        frame_index,
        frame_ended,
    }
}

impl FixedStepClock {
    pub fn new(config: FixedStepConfig) -> Self {
        Self {
            config,
            accumulator: Duration::ZERO,
        }
    }

    pub fn update(&mut self, delta: Duration) -> FixedStepUpdate {
        let delta = delta.max(Duration::ZERO);
        if delta == Duration::ZERO {
            return FixedStepUpdate {
                steps: 0,
                alpha: 0.0,
                accumulated: self.accumulator,
                dropped_time: Duration::ZERO,
            };
        }

        if self.config.step == Duration::ZERO || self.config.max_steps_per_frame == 0 {
            let accumulated = self.accumulator.saturating_add(delta);
            self.accumulator = Duration::ZERO;
            return FixedStepUpdate {
                steps: 0,
                alpha: 0.0,
                accumulated: self.accumulator,
                dropped_time: accumulated,
            };
        }

        let max_step_time = duration_mul(self.config.step, self.config.max_steps_per_frame);
        let mut accumulated = self.accumulator.saturating_add(delta);
        let dropped_time = if accumulated > max_step_time {
            let dropped = accumulated - max_step_time;
            accumulated = max_step_time;
            dropped
        } else {
            Duration::ZERO
        };

        let step_nanos = self.config.step.as_nanos();
        let steps =
            ((accumulated.as_nanos() / step_nanos) as u32).min(self.config.max_steps_per_frame);
        let stepped_time = duration_mul(self.config.step, steps);
        self.accumulator = accumulated.saturating_sub(stepped_time);
        let alpha =
            (self.accumulator.as_secs_f32() / self.config.step.as_secs_f32()).clamp(0.0, 1.0);

        FixedStepUpdate {
            steps,
            alpha,
            accumulated: self.accumulator,
            dropped_time,
        }
    }

    pub fn reset(&mut self) {
        self.accumulator = Duration::ZERO;
    }
}

fn duration_mul(duration: Duration, multiplier: u32) -> Duration {
    duration.checked_mul(multiplier).unwrap_or(Duration::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::{Arc, Mutex};

    use crate::events::{EngineEvent, EventStage, FrameId, LifecycleEvent};
    use crate::input::{ActionId, ActionMap, LayerDescriptor, LayerPriority};
    use winit::event::ElementState;
    use winit::keyboard::{KeyCode, ModifiersState};

    fn fixed_clock(step_ms: u64, max_steps_per_frame: u32) -> FixedStepClock {
        FixedStepClock::new(FixedStepConfig {
            step: Duration::from_millis(step_ms),
            max_steps_per_frame,
        })
    }

    fn assert_alpha(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 0.0001,
            "expected alpha {expected}, got {actual}"
        );
    }

    fn add_jump_action(input: &mut InputSystem) {
        let mut map = ActionMap::new();
        map.bind_key("jump", KeyCode::Space);
        input.add_layer(
            LayerDescriptor::new("actions", LayerPriority(0)),
            map.into_layer(),
        );
    }

    #[test]
    fn frame_clock_advances_indices_and_deltas() {
        let start = Instant::now();
        let mut clock = FrameClock::from_instant(start);

        let first = clock.tick_at(start + Duration::from_millis(16));
        let second = clock.tick_at(start + Duration::from_millis(33));

        assert_eq!(first.index, 0);
        assert_eq!(first.delta, Duration::from_millis(16));
        assert_eq!(second.index, 1);
        assert_eq!(second.delta, Duration::from_millis(17));
        assert_eq!(clock.next_index(), 2);
    }

    #[test]
    fn frame_clock_clamps_backwards_ticks_to_zero_delta() {
        let start = Instant::now();
        let mut clock = FrameClock::from_instant(start);

        let info = clock.tick_at(start - Duration::from_millis(1));

        assert_eq!(info.index, 0);
        assert_eq!(info.delta, Duration::ZERO);
    }

    #[test]
    fn fixed_step_clock_runs_one_exact_step() {
        let mut clock = fixed_clock(16, 4);

        let update = clock.update(Duration::from_millis(16));

        assert_eq!(update.steps, 1);
        assert_eq!(update.accumulated, Duration::ZERO);
        assert_eq!(update.dropped_time, Duration::ZERO);
        assert_alpha(update.alpha, 0.0);
    }

    #[test]
    fn fixed_step_clock_accumulates_remainder() {
        let mut clock = fixed_clock(16, 4);

        let update = clock.update(Duration::from_millis(30));

        assert_eq!(update.steps, 1);
        assert_eq!(update.accumulated, Duration::from_millis(14));
        assert_eq!(update.dropped_time, Duration::ZERO);
        assert_alpha(update.alpha, 14.0 / 16.0);
    }

    #[test]
    fn fixed_step_clock_caps_catch_up_and_drops_excess_time() {
        let mut clock = fixed_clock(16, 3);

        let update = clock.update(Duration::from_millis(100));

        assert_eq!(update.steps, 3);
        assert_eq!(update.accumulated, Duration::ZERO);
        assert_eq!(update.dropped_time, Duration::from_millis(52));
        assert_alpha(update.alpha, 0.0);
    }

    #[test]
    fn fixed_step_clock_zero_delta_reports_no_steps_and_zero_alpha() {
        let mut clock = fixed_clock(16, 4);

        let update = clock.update(Duration::ZERO);

        assert_eq!(update.steps, 0);
        assert_eq!(update.accumulated, Duration::ZERO);
        assert_eq!(update.dropped_time, Duration::ZERO);
        assert_alpha(update.alpha, 0.0);
    }

    #[test]
    fn fixed_step_clock_clamped_backward_delta_reports_no_steps_and_zero_alpha() {
        let mut clock = fixed_clock(16, 4);
        let backward_equivalent_delta = Duration::ZERO.max(Duration::ZERO);

        let update = clock.update(backward_equivalent_delta);

        assert_eq!(update.steps, 0);
        assert_eq!(update.accumulated, Duration::ZERO);
        assert_eq!(update.dropped_time, Duration::ZERO);
        assert_alpha(update.alpha, 0.0);
    }

    #[test]
    fn fixed_step_clock_accumulates_across_multiple_frames() {
        let mut clock = fixed_clock(16, 4);

        let first = clock.update(Duration::from_millis(10));
        let second = clock.update(Duration::from_millis(10));

        assert_eq!(first.steps, 0);
        assert_eq!(first.accumulated, Duration::from_millis(10));
        assert_alpha(first.alpha, 10.0 / 16.0);
        assert_eq!(second.steps, 1);
        assert_eq!(second.accumulated, Duration::from_millis(4));
        assert_alpha(second.alpha, 4.0 / 16.0);
    }

    #[test]
    fn fixed_step_clock_reset_clears_accumulator() {
        let mut clock = fixed_clock(16, 4);
        let _ = clock.update(Duration::from_millis(10));

        clock.reset();
        let update = clock.update(Duration::from_millis(6));

        assert_eq!(update.steps, 0);
        assert_eq!(update.accumulated, Duration::from_millis(6));
        assert_alpha(update.alpha, 6.0 / 16.0);
    }

    #[test]
    fn begin_app_frame_ticks_dispatches_input_actions_and_frame_started() {
        let mut input = InputSystem::new();
        add_jump_action(&mut input);
        input.queue_event(crate::input::InputEvent::Key {
            code: KeyCode::Space,
            state: ElementState::Pressed,
            repeat: false,
            modifiers: ModifiersState::empty(),
        });

        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_listener = Arc::clone(&seen);
        let mut events = EventBus::new();
        events.subscribe(move |event| {
            match &event.event {
                EngineEvent::Input(action) => seen_listener.lock().unwrap().push((
                    event.stage,
                    "input".to_string(),
                    Some(action.action.clone()),
                    event.frame,
                )),
                EngineEvent::Lifecycle(LifecycleEvent::FrameStarted) => {
                    seen_listener.lock().unwrap().push((
                        event.stage,
                        "frame_started".to_string(),
                        None,
                        event.frame,
                    ));
                }
                _ => {}
            }
            Ok(())
        });

        let mut action_events = InputActionEventEmitter::new();
        let mut frame_clock = FrameClock::new();

        let report = begin_app_frame(
            &mut input,
            &mut action_events,
            &mut events,
            &mut frame_clock,
        );

        assert_eq!(report.frame.index, 0);
        assert_eq!(frame_clock.next_index(), 1);
        assert_eq!(report.action_events_emitted, 1);
        assert_eq!(report.input_dispatch.dispatched, 1);
        assert!(report.input_dispatch.failures.is_empty());
        assert_eq!(report.frame_started.dispatched, 1);
        assert!(report.frame_started.failures.is_empty());
        assert_eq!(events.pending_len(), 0);
        assert!(input.snapshot().action_pressed(&ActionId::new("jump")));

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].0, EventStage::Input);
        assert_eq!(seen[0].1, "input");
        assert_eq!(seen[0].2, Some(ActionId::new("jump")));
        assert_eq!(seen[0].3, Some(FrameId(0)));
        assert_eq!(seen[1].0, EventStage::PreUpdate);
        assert_eq!(seen[1].1, "frame_started");
        assert_eq!(seen[1].3, Some(FrameId(0)));
    }

    #[test]
    fn end_app_frame_emits_frame_ended() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_listener = Arc::clone(&seen);
        let mut events = EventBus::new();
        events.subscribe(move |event| {
            if let EngineEvent::Lifecycle(LifecycleEvent::FrameEnded) = &event.event {
                seen_listener
                    .lock()
                    .unwrap()
                    .push((event.stage, event.frame));
            }
            Ok(())
        });

        let report = end_app_frame(&mut events, 7);

        assert_eq!(report.frame_index, 7);
        assert_eq!(report.frame_ended.dispatched, 1);
        assert!(report.frame_ended.failures.is_empty());
        assert_eq!(events.pending_len(), 0);
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            &[(EventStage::PostUpdate, Some(FrameId(7)))]
        );
    }
}
