use std::collections::{BTreeMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::panic::{catch_unwind, AssertUnwindSafe};

use imgui::{Condition, Ui};
use input::InputDebugSnapshot;
use log::error;

use crate::data::handles::EnvironmentHandle;

const MAX_CONSOLE_HISTORY: usize = 128;
const MAX_CONSOLE_OUTPUT_LINES: usize = 256;

pub type DebugViewCallback = Box<dyn FnMut(&Ui, &DebugUiFrameContext) + 'static>;

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DebugViewId(String);

impl DebugViewId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl Display for DebugViewId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for DebugViewId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for DebugViewId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Debug)]
pub struct DebugViewDescriptor {
    pub id: DebugViewId,
    pub label: String,
    pub enabled_by_default: bool,
    pub order: i32,
}

impl DebugViewDescriptor {
    pub fn new(id: impl Into<DebugViewId>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            enabled_by_default: true,
            order: 0,
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled_by_default = enabled;
        self
    }

    pub fn with_order(mut self, order: i32) -> Self {
        self.order = order;
        self
    }
}

#[derive(Clone, Debug)]
pub struct DebugUiFrameContext {
    pub frame_index: u64,
    pub delta_seconds: f32,
    pub fps: f32,
    pub viewport_size: (u32, u32),
    pub resize_pending: bool,
    pub environment_requested: Option<EnvironmentHandle>,
    pub environment_active: EnvironmentHandle,
    pub environment_transitioning: bool,
    pub draw_item_count: usize,
    pub point_light_count: usize,
    pub draw_skybox: bool,
    pub draw_geometry: bool,
    pub draw_imgui: bool,
    pub asset_tasks_pumped_last: usize,
    pub input_debug: InputDebugSnapshot,
}

impl Default for DebugUiFrameContext {
    fn default() -> Self {
        Self {
            frame_index: 0,
            delta_seconds: 0.0,
            fps: 0.0,
            viewport_size: (0, 0),
            resize_pending: false,
            environment_requested: None,
            environment_active: EnvironmentHandle::new(0, 0),
            environment_transitioning: false,
            draw_item_count: 0,
            point_light_count: 0,
            draw_skybox: true,
            draw_geometry: true,
            draw_imgui: true,
            asset_tasks_pumped_last: 0,
            input_debug: InputDebugSnapshot::default(),
        }
    }
}

struct DebugViewEntry {
    descriptor: DebugViewDescriptor,
    callback: DebugViewCallback,
    enabled: bool,
}

pub struct DebugUiManager {
    visible: bool,
    views: BTreeMap<DebugViewId, DebugViewEntry>,
    frame_context: DebugUiFrameContext,
    console_input: String,
    console_history: Vec<String>,
    console_history_cursor: Option<usize>,
    console_output: VecDeque<String>,
    view_filter: String,
}

impl Default for DebugUiManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugUiManager {
    pub fn new() -> Self {
        let mut manager = Self {
            visible: false,
            views: BTreeMap::new(),
            frame_context: DebugUiFrameContext::default(),
            console_input: String::new(),
            console_history: Vec::new(),
            console_history_cursor: None,
            console_output: VecDeque::new(),
            view_filter: String::new(),
        };

        manager.register_builtin_views();
        manager
    }

    pub fn update_frame_context(&mut self, frame_context: DebugUiFrameContext) {
        self.frame_context = frame_context;
    }

    pub fn register_view(
        &mut self,
        descriptor: DebugViewDescriptor,
        callback: DebugViewCallback,
    ) -> bool {
        if self.views.contains_key(&descriptor.id) {
            return false;
        }

        let enabled = descriptor.enabled_by_default;
        self.views.insert(
            descriptor.id.clone(),
            DebugViewEntry {
                descriptor,
                callback,
                enabled,
            },
        );
        true
    }

    pub fn unregister_view(&mut self, id: &DebugViewId) -> bool {
        self.views.remove(id).is_some()
    }

    pub fn set_view_enabled(&mut self, id: &DebugViewId, enabled: bool) -> bool {
        let Some(view) = self.views.get_mut(id) else {
            return false;
        };

        view.enabled = enabled;
        true
    }

    pub fn toggle_view(&mut self, id: &DebugViewId) -> bool {
        let Some(view) = self.views.get_mut(id) else {
            return false;
        };

        view.enabled = !view.enabled;
        true
    }

    pub fn view_enabled(&self, id: &DebugViewId) -> Option<bool> {
        self.views.get(id).map(|view| view.enabled)
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn toggle_visible(&mut self) {
        self.visible = !self.visible;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn render(&mut self, ui: &Ui) {
        if !self.visible {
            return;
        }

        self.render_views_window(ui);
        self.render_console_window(ui);
        self.render_enabled_views(ui);
    }

    fn register_builtin_views(&mut self) {
        let _ = self.register_view(
            DebugViewDescriptor::new("view.runtime", "Runtime").with_order(100),
            Box::new(|ui, ctx| {
                ui.window("Runtime")
                    .size([340.0, 170.0], Condition::FirstUseEver)
                    .build(|| {
                        ui.text(format!("Frame: {}", ctx.frame_index));
                        ui.text(format!(
                            "Delta: {:.3}ms ({:.2} fps)",
                            ctx.delta_seconds * 1000.0,
                            ctx.fps
                        ));
                        ui.text(format!(
                            "Viewport: {} x {}",
                            ctx.viewport_size.0, ctx.viewport_size.1
                        ));
                        ui.text(format!("Resize pending: {}", ctx.resize_pending));
                    });
            }),
        );

        let _ = self.register_view(
            DebugViewDescriptor::new("view.environment", "Environment").with_order(200),
            Box::new(|ui, ctx| {
                ui.window("Environment")
                    .size([360.0, 140.0], Condition::FirstUseEver)
                    .build(|| {
                        ui.text(format!(
                            "Active: {}",
                            format_env_handle(ctx.environment_active)
                        ));

                        if let Some(requested) = ctx.environment_requested {
                            ui.text(format!("Requested: {}", format_env_handle(requested)));
                        } else {
                            ui.text("Requested: none");
                        }

                        ui.text(format!("Transitioning: {}", ctx.environment_transitioning));
                    });
            }),
        );

        let _ = self.register_view(
            DebugViewDescriptor::new("view.input", "Input").with_order(300),
            Box::new(|ui, ctx| {
                ui.window("Input")
                    .size([360.0, 170.0], Condition::FirstUseEver)
                    .build(|| {
                        ui.text(format!("Queued events: {}", ctx.input_debug.queued_events));
                        ui.text(format!("Layer count: {}", ctx.input_debug.layer_count));
                        ui.text(format!(
                            "Active layers: {}",
                            ctx.input_debug.active_layer_count
                        ));
                        ui.text(format!(
                            "Last dispatch consumed: {}",
                            ctx.input_debug.last_dispatch_consumed_events
                        ));
                    });
            }),
        );

        let _ = self.register_view(
            DebugViewDescriptor::new("view.submission", "Submission").with_order(400),
            Box::new(|ui, ctx| {
                ui.window("Submission")
                    .size([360.0, 180.0], Condition::FirstUseEver)
                    .build(|| {
                        ui.text(format!("Draw items: {}", ctx.draw_item_count));
                        ui.text(format!("Point lights: {}", ctx.point_light_count));
                        ui.separator();
                        ui.text(format!("draw_skybox: {}", ctx.draw_skybox));
                        ui.text(format!("draw_geometry: {}", ctx.draw_geometry));
                        ui.text(format!("draw_imgui: {}", ctx.draw_imgui));
                    });
            }),
        );

        let _ = self.register_view(
            DebugViewDescriptor::new("view.assets", "Assets").with_order(500),
            Box::new(|ui, ctx| {
                ui.window("Assets")
                    .size([360.0, 120.0], Condition::FirstUseEver)
                    .build(|| {
                        ui.text(format!(
                            "Asset tasks pumped last frame: {}",
                            ctx.asset_tasks_pumped_last
                        ));
                        ui.text("Deep cache diagnostics are deferred post-alpha.");
                    });
            }),
        );
    }

    fn render_views_window(&mut self, ui: &Ui) {
        ui.window("Views")
            .size([360.0, 460.0], Condition::FirstUseEver)
            .build(|| {
                ui.input_text("Filter", &mut self.view_filter).build();
                ui.separator();

                let query = self.view_filter.to_ascii_lowercase();
                for id in self.ordered_view_ids() {
                    let Some(view) = self.views.get_mut(&id) else {
                        continue;
                    };

                    if !query.is_empty() {
                        let id_match = view
                            .descriptor
                            .id
                            .as_str()
                            .to_ascii_lowercase()
                            .contains(query.as_str());
                        let label_match = view
                            .descriptor
                            .label
                            .to_ascii_lowercase()
                            .contains(query.as_str());

                        if !id_match && !label_match {
                            continue;
                        }
                    }

                    let mut enabled = view.enabled;
                    let toggle_id = format!("{}##{}", view.descriptor.label, view.descriptor.id);
                    if ui.checkbox(toggle_id.as_str(), &mut enabled) {
                        view.enabled = enabled;
                    }

                    ui.same_line();
                    ui.text_disabled(format!("({})", view.descriptor.id));
                }
            });
    }

    fn render_console_window(&mut self, ui: &Ui) {
        ui.window("Console")
            .size([640.0, 320.0], Condition::FirstUseEver)
            .build(|| {
                ui.child_window("console_output")
                    .size([0.0, -30.0])
                    .build(|| {
                        for line in &self.console_output {
                            ui.text(line);
                        }
                    });

                ui.separator();
                if ui.small_button("Prev") {
                    self.history_prev();
                }
                ui.same_line();
                if ui.small_button("Next") {
                    self.history_next();
                }
                ui.same_line();
                let submitted = ui
                    .input_text("##console_input", &mut self.console_input)
                    .enter_returns_true(true)
                    .build();

                if submitted {
                    self.submit_console_line();
                }
            });
    }

    fn render_enabled_views(&mut self, ui: &Ui) {
        let frame_context = self.frame_context.clone();
        for id in self.ordered_view_ids() {
            let mut panic_message = None;

            if let Some(entry) = self.views.get_mut(&id) {
                if !entry.enabled {
                    continue;
                }

                let result =
                    catch_unwind(AssertUnwindSafe(|| (entry.callback)(ui, &frame_context)));
                if let Err(payload) = result {
                    panic_message = Some(panic_payload_to_string(payload));
                }
            }

            if let Some(message) = panic_message {
                error!("debug view '{}' panicked: {}", id, message);
                self.push_console_output(format!("view '{}' panicked: {}", id, message));
            }
        }
    }

    fn ordered_view_ids(&self) -> Vec<DebugViewId> {
        let mut ids: Vec<DebugViewId> = self.views.keys().cloned().collect();
        ids.sort_by(|a, b| {
            let left = self.views.get(a).unwrap();
            let right = self.views.get(b).unwrap();

            left.descriptor
                .order
                .cmp(&right.descriptor.order)
                .then_with(|| left.descriptor.label.cmp(&right.descriptor.label))
                .then_with(|| left.descriptor.id.cmp(&right.descriptor.id))
        });

        ids
    }

    fn submit_console_line(&mut self) {
        let line = self.console_input.trim().to_string();
        self.console_input.clear();
        if line.is_empty() {
            return;
        }

        self.push_console_output(format!("> {}", line));
        self.console_history.push(line.clone());
        if self.console_history.len() > MAX_CONSOLE_HISTORY {
            let drop_count = self.console_history.len() - MAX_CONSOLE_HISTORY;
            self.console_history.drain(0..drop_count);
        }
        self.console_history_cursor = None;

        self.execute_console_command(line.as_str());
    }

    fn execute_console_command(&mut self, line: &str) {
        let args: Vec<&str> = line.split_whitespace().collect();
        if args.is_empty() {
            return;
        }

        match args[0] {
            "help" => {
                self.push_console_output("commands: help clear history view.list view.enable <id> view.disable <id> view.toggle <id>".to_string());
            }
            "clear" => {
                self.console_output.clear();
            }
            "history" => {
                if self.console_history.is_empty() {
                    self.push_console_output("history is empty".to_string());
                } else {
                    let lines: Vec<String> = self
                        .console_history
                        .iter()
                        .enumerate()
                        .map(|(idx, item)| format!("{}: {}", idx, item))
                        .collect();
                    for line in lines {
                        self.push_console_output(line);
                    }
                }
            }
            "view.list" => {
                let mut lines = Vec::new();
                for id in self.ordered_view_ids() {
                    if let Some(entry) = self.views.get(&id) {
                        lines.push(format!(
                            "{} ({}) = {}",
                            entry.descriptor.id,
                            entry.descriptor.label,
                            if entry.enabled { "enabled" } else { "disabled" }
                        ));
                    }
                }
                for line in lines {
                    self.push_console_output(line);
                }
            }
            "view.enable" => {
                self.run_view_command(args.as_slice(), true, false);
            }
            "view.disable" => {
                self.run_view_command(args.as_slice(), false, false);
            }
            "view.toggle" => {
                self.run_view_command(args.as_slice(), false, true);
            }
            _ => {
                self.push_console_output(format!("unknown command: {}", args[0]));
            }
        }
    }

    fn run_view_command(&mut self, args: &[&str], enabled: bool, toggle: bool) {
        if args.len() != 2 {
            self.push_console_output(format!("{} expects exactly one <id> argument", args[0]));
            return;
        }

        let id = DebugViewId::new(args[1]);
        let ok = if toggle {
            self.toggle_view(&id)
        } else {
            self.set_view_enabled(&id, enabled)
        };

        if ok {
            let state = self.view_enabled(&id).unwrap_or(false);
            self.push_console_output(format!(
                "{} => {}",
                id,
                if state { "enabled" } else { "disabled" }
            ));
        } else {
            self.push_console_output(format!("unknown view id: {}", id));
        }
    }

    fn push_console_output(&mut self, line: String) {
        self.console_output.push_back(line);
        while self.console_output.len() > MAX_CONSOLE_OUTPUT_LINES {
            self.console_output.pop_front();
        }
    }

    fn history_prev(&mut self) {
        if self.console_history.is_empty() {
            return;
        }

        let next_index = match self.console_history_cursor {
            None => self.console_history.len() - 1,
            Some(0) => 0,
            Some(index) => index.saturating_sub(1),
        };

        self.console_history_cursor = Some(next_index);
        self.console_input = self.console_history[next_index].clone();
    }

    fn history_next(&mut self) {
        let Some(curr) = self.console_history_cursor else {
            return;
        };

        let next = curr + 1;
        if next >= self.console_history.len() {
            self.console_history_cursor = None;
            self.console_input.clear();
            return;
        }

        self.console_history_cursor = Some(next);
        self.console_input = self.console_history[next].clone();
    }
}

fn format_env_handle(handle: EnvironmentHandle) -> String {
    format!("{}:{}", handle.slot, handle.generation)
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    let payload = payload.as_ref();
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }

    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_string();
    }

    "unknown panic payload".to_string()
}

#[cfg(test)]
mod tests {
    use super::{DebugUiManager, DebugViewDescriptor, DebugViewId};

    #[test]
    fn register_unregister_and_toggle_view_state() {
        let mut manager = DebugUiManager::new();
        let id = DebugViewId::new("user.test");

        let registered = manager.register_view(
            DebugViewDescriptor::new(id.clone(), "User Test"),
            Box::new(|_, _| {}),
        );
        assert!(registered);

        assert!(manager.set_view_enabled(&id, false));
        assert_eq!(manager.view_enabled(&id), Some(false));

        assert!(manager.toggle_view(&id));
        assert_eq!(manager.view_enabled(&id), Some(true));

        assert!(manager.unregister_view(&id));
        assert_eq!(manager.view_enabled(&id), None);
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut manager = DebugUiManager::new();
        let id = DebugViewId::new("user.dup");

        assert!(manager.register_view(
            DebugViewDescriptor::new(id.clone(), "First"),
            Box::new(|_, _| {}),
        ));

        assert!(
            !manager.register_view(DebugViewDescriptor::new(id, "Second"), Box::new(|_, _| {}),)
        );
    }
}
