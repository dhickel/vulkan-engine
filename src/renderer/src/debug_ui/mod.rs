use std::collections::{BTreeMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use imgui::{Condition, Ui};
use input::InputDebugSnapshot;
use log::error;

use crate::data::handles::EnvironmentHandle;

const MAX_CONSOLE_HISTORY: usize = 128;
const MAX_CONSOLE_OUTPUT_LINES: usize = 256;
const MAX_TIMING_HISTORY: usize = 180;
const TIMING_AVG_WINDOW: usize = 60;
const FRAME_HISTORY_SAMPLE_SIZE: usize = 8;
const MAX_SPIKE_HISTORY: usize = 32;
const SPIKE_ABSOLUTE_THRESHOLD_MS: f32 = 20.0;
const SPIKE_RELATIVE_THRESHOLD_SCALE: f32 = 1.5;
const SIDEBAR_MIN_WIDTH: f32 = 380.0;
const SIDEBAR_MAX_WIDTH: f32 = 640.0;
const DEFAULT_TIMING_REPORT_PREFIX: &str = "timing_report";
const DEFAULT_TIMING_RECORD_INTERVAL_MS: u64 = 250;
const DEFAULT_TIMING_RECORD_DURATION_SECS: u64 = 10;

pub type DebugViewCallback = Box<dyn FnMut(&Ui, &DebugUiFrameContext) + 'static>;

#[derive(Clone, Debug, Default)]
pub struct DebugTimingRow {
    pub label: &'static str,
    pub cpu_ms: f32,
    pub gpu_ms: Option<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct DebugTimingSnapshot {
    pub gpu_supported: bool,
    pub frame_cpu_ms: f32,
    pub frame_gpu_ms: Option<f32>,
    pub stage_timings: Vec<DebugTimingRow>,
    pub pass_timings: Vec<DebugTimingRow>,
}

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
    pub timings: DebugTimingSnapshot,
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
            timings: DebugTimingSnapshot::default(),
        }
    }
}

struct DebugViewEntry {
    descriptor: DebugViewDescriptor,
    callback: DebugViewCallback,
    enabled: bool,
}

#[derive(Default)]
struct TimingWindow {
    cpu: VecDeque<f32>,
    gpu: VecDeque<f32>,
}

impl TimingWindow {
    fn push(&mut self, cpu_ms: f32, gpu_ms: Option<f32>) {
        self.cpu.push_back(cpu_ms.max(0.0));
        while self.cpu.len() > TIMING_AVG_WINDOW {
            self.cpu.pop_front();
        }

        self.gpu.push_back(gpu_ms.unwrap_or(f32::NAN));
        while self.gpu.len() > TIMING_AVG_WINDOW {
            self.gpu.pop_front();
        }
    }

    fn avg_cpu(&self) -> Option<f32> {
        average_iter(self.cpu.iter().copied())
    }

    fn avg_gpu(&self) -> Option<f32> {
        average_iter(self.gpu.iter().copied())
    }

    fn max_cpu(&self) -> Option<f32> {
        max_iter(self.cpu.iter().copied())
    }

    fn max_gpu(&self) -> Option<f32> {
        max_iter(self.gpu.iter().copied())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TimingPeak {
    cpu_ms: Option<f32>,
    gpu_ms: Option<f32>,
}

impl TimingPeak {
    fn observe(&mut self, cpu_ms: f32, gpu_ms: Option<f32>) {
        self.cpu_ms = Some(match self.cpu_ms {
            Some(existing) => existing.max(cpu_ms.max(0.0)),
            None => cpu_ms.max(0.0),
        });

        if let Some(gpu_ms) = gpu_ms.filter(|value| value.is_finite()) {
            self.gpu_ms = Some(match self.gpu_ms {
                Some(existing) => existing.max(gpu_ms.max(0.0)),
                None => gpu_ms.max(0.0),
            });
        }
    }
}

#[derive(Clone, Debug)]
struct SpikeTimingSource {
    label: &'static str,
    cpu_ms: f32,
    gpu_ms: Option<f32>,
}

#[derive(Clone, Debug)]
struct SpikeAttribution {
    frame_index: u64,
    frame_cpu_ms: f32,
    frame_gpu_ms: Option<f32>,
    threshold_cpu_ms: f32,
    top_stage: Option<SpikeTimingSource>,
    top_pass: Option<SpikeTimingSource>,
}

pub struct DebugUiManager {
    console_visible: bool,
    debug_visible: bool,
    views: BTreeMap<DebugViewId, DebugViewEntry>,
    frame_context: DebugUiFrameContext,
    console_input: String,
    console_history: Vec<String>,
    console_history_cursor: Option<usize>,
    console_output: VecDeque<String>,
    view_filter: String,
    frame_timing_window: TimingWindow,
    frame_timing_peak: TimingPeak,
    stage_timing_windows: BTreeMap<&'static str, TimingWindow>,
    pass_timing_windows: BTreeMap<&'static str, TimingWindow>,
    stage_timing_peaks: BTreeMap<&'static str, TimingPeak>,
    pass_timing_peaks: BTreeMap<&'static str, TimingPeak>,
    frame_sample_count: usize,
    frame_sample_cpu_sum: f32,
    frame_sample_gpu_sum: f32,
    frame_sample_gpu_count: usize,
    cpu_frame_history: VecDeque<f32>,
    gpu_frame_history: VecDeque<f32>,
    spike_attribution_enabled: bool,
    spike_record_while_hidden: bool,
    spike_history: VecDeque<SpikeAttribution>,
    timing_report_path: String,
    timing_report_status: Option<String>,
    timing_record_interval_ms: u64,
    timing_record_duration_secs: u64,
    timing_recording_active: bool,
    timing_record_started_at: Option<Instant>,
    timing_record_next_snapshot_at: Option<Instant>,
    timing_record_end_at: Option<Instant>,
    timing_record_samples_written: u64,
    timing_record_active_path: Option<String>,
}

impl Default for DebugUiManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugUiManager {
    pub fn new() -> Self {
        let mut manager = Self {
            console_visible: false,
            debug_visible: false,
            views: BTreeMap::new(),
            frame_context: DebugUiFrameContext::default(),
            console_input: String::new(),
            console_history: Vec::new(),
            console_history_cursor: None,
            console_output: VecDeque::new(),
            view_filter: String::new(),
            frame_timing_window: TimingWindow::default(),
            frame_timing_peak: TimingPeak::default(),
            stage_timing_windows: BTreeMap::new(),
            pass_timing_windows: BTreeMap::new(),
            stage_timing_peaks: BTreeMap::new(),
            pass_timing_peaks: BTreeMap::new(),
            frame_sample_count: 0,
            frame_sample_cpu_sum: 0.0,
            frame_sample_gpu_sum: 0.0,
            frame_sample_gpu_count: 0,
            cpu_frame_history: VecDeque::new(),
            gpu_frame_history: VecDeque::new(),
            spike_attribution_enabled: false,
            spike_record_while_hidden: false,
            spike_history: VecDeque::new(),
            timing_report_path: default_timing_report_path(),
            timing_report_status: None,
            timing_record_interval_ms: DEFAULT_TIMING_RECORD_INTERVAL_MS,
            timing_record_duration_secs: DEFAULT_TIMING_RECORD_DURATION_SECS,
            timing_recording_active: false,
            timing_record_started_at: None,
            timing_record_next_snapshot_at: None,
            timing_record_end_at: None,
            timing_record_samples_written: 0,
            timing_record_active_path: None,
        };

        manager.register_builtin_views();
        manager
    }

    pub fn update_frame_context(&mut self, frame_context: DebugUiFrameContext) {
        self.frame_context = frame_context;
        self.record_timing_history();
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

    pub fn set_console_visible(&mut self, visible: bool) {
        self.console_visible = visible;
    }

    pub fn toggle_console_visible(&mut self) {
        self.console_visible = !self.console_visible;
    }

    pub fn is_console_visible(&self) -> bool {
        self.console_visible
    }

    pub fn set_debug_visible(&mut self, visible: bool) {
        self.debug_visible = visible;
    }

    pub fn toggle_debug_visible(&mut self) {
        self.debug_visible = !self.debug_visible;
    }

    pub fn is_debug_visible(&self) -> bool {
        self.debug_visible
    }

    pub fn is_any_visible(&self) -> bool {
        self.console_visible || self.debug_visible
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.console_visible = visible;
        self.debug_visible = visible;
    }

    pub fn toggle_visible(&mut self) {
        if self.is_any_visible() {
            self.set_visible(false);
        } else {
            self.set_visible(true);
        }
    }

    pub fn is_visible(&self) -> bool {
        self.is_any_visible()
    }

    pub fn configure_timing_recording_options(
        &mut self,
        duration_secs: Option<u64>,
        interval_ms: Option<u64>,
        report_path: Option<String>,
    ) -> Result<(), String> {
        if let Some(duration_secs) = duration_secs {
            if duration_secs == 0 {
                return Err("record duration must be >= 1 second".to_string());
            }
            self.timing_record_duration_secs = duration_secs;
        }

        if let Some(interval_ms) = interval_ms {
            if interval_ms == 0 {
                return Err("snapshot interval must be >= 1 ms".to_string());
            }
            self.timing_record_interval_ms = interval_ms;
        }

        if let Some(report_path) = report_path {
            let report_path = report_path.trim();
            if report_path.is_empty() {
                return Err("output path is empty".to_string());
            }
            self.timing_report_path = report_path.to_string();
        }

        Ok(())
    }

    pub fn start_timing_recording_now(&mut self) -> Result<String, String> {
        if self.timing_recording_active {
            return Err("timing recording is already active".to_string());
        }

        let timings = self.frame_context.timings.clone();
        self.start_timing_recording(&timings)
    }

    pub fn render(&mut self, ui: &Ui) {
        if !self.is_any_visible() {
            return;
        }

        let display_size = ui.io().display_size;
        let screen_width = display_size[0].max(1.0);
        let screen_height = display_size[1].max(1.0);
        let sidebar_width = (screen_width * 0.32).clamp(SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH);

        if self.console_visible {
            self.render_console_window(ui, sidebar_width, screen_height);
        }

        if self.debug_visible {
            self.render_debug_window(ui, screen_width, screen_height, sidebar_width);
        }
    }

    fn register_builtin_views(&mut self) {
        let _ = self.register_view(
            DebugViewDescriptor::new("view.runtime", "Runtime").with_order(100),
            Box::new(|ui, ctx| {
                ui.text("Runtime");
                ui.text(format!("Frame: {}", ctx.frame_index));
                ui.text(format!(
                    "Viewport: {} x {}",
                    ctx.viewport_size.0, ctx.viewport_size.1
                ));
                ui.text(format!("Resize pending: {}", ctx.resize_pending));
            }),
        );

        let _ = self.register_view(
            DebugViewDescriptor::new("view.environment", "Environment").with_order(200),
            Box::new(|ui, ctx| {
                ui.text("Environment");
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
            }),
        );

        let _ = self.register_view(
            DebugViewDescriptor::new("view.input", "Input").with_order(300),
            Box::new(|ui, ctx| {
                ui.text("Input");
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
            }),
        );

        let _ = self.register_view(
            DebugViewDescriptor::new("view.submission", "Submission").with_order(400),
            Box::new(|ui, ctx| {
                ui.text("Submission");
                ui.text(format!("Draw items: {}", ctx.draw_item_count));
                ui.text(format!("Point lights: {}", ctx.point_light_count));
                ui.text(format!("draw_skybox: {}", ctx.draw_skybox));
                ui.text(format!("draw_geometry: {}", ctx.draw_geometry));
                ui.text(format!("draw_imgui: {}", ctx.draw_imgui));
            }),
        );

        let _ = self.register_view(
            DebugViewDescriptor::new("view.assets", "Assets").with_order(500),
            Box::new(|ui, ctx| {
                ui.text("Assets");
                ui.text(format!(
                    "Asset tasks pumped last frame: {}",
                    ctx.asset_tasks_pumped_last
                ));
                ui.text("Deep cache diagnostics are deferred post-alpha.");
            }),
        );
    }

    fn render_console_window(&mut self, ui: &Ui, sidebar_width: f32, screen_height: f32) {
        ui.window("Console###engine_console")
            .position([0.0, 0.0], Condition::Always)
            .size([sidebar_width, screen_height], Condition::Always)
            .movable(false)
            .resizable(false)
            .collapsible(false)
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

    fn render_debug_window(
        &mut self,
        ui: &Ui,
        screen_width: f32,
        screen_height: f32,
        sidebar_width: f32,
    ) {
        let x = (screen_width - sidebar_width).max(0.0);

        ui.window("Debug###engine_debug")
            .position([x, 0.0], Condition::Always)
            .size([sidebar_width, screen_height], Condition::Always)
            .movable(false)
            .resizable(false)
            .collapsible(false)
            .build(|| {
                ui.text("Views");
                ui.input_text("Filter", &mut self.view_filter).build();
                ui.separator();
                self.render_view_toggles(ui);
                ui.separator();
                self.render_performance_section(ui);
                ui.separator();

                ui.child_window("debug_views").size([0.0, 0.0]).build(|| {
                    self.render_enabled_views(ui);
                });
            });
    }

    fn render_view_toggles(&mut self, ui: &Ui) {
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
    }

    fn render_performance_section(&mut self, ui: &Ui) {
        let timings = self.frame_context.timings.clone();
        ui.text("Performance");
        ui.same_line();
        if ui.button("Reset Maxima##timing_max_reset") {
            self.reset_timing_maxima();
        }
        let Some(_tab_bar) = ui.tab_bar("performance_tabs##engine_debug") else {
            self.render_performance_timing_tab(ui, &timings);
            return;
        };

        if let Some(_tab) = ui.tab_item("Timing##performance_tab_timing") {
            self.render_performance_timing_tab(ui, &timings);
        }

        if let Some(_tab) = ui.tab_item("Graph##performance_tab_graph") {
            self.render_performance_graph_tab(ui);
        }

        if let Some(_tab) = ui.tab_item("Record##performance_tab_record") {
            self.render_performance_record_tab(ui, &timings);
        }
    }

    fn render_performance_timing_tab(&self, ui: &Ui, timings: &DebugTimingSnapshot) {
        let mode_label = if !timings.gpu_supported {
            "CPU-only fallback (GPU timestamps unsupported)"
        } else if timings.frame_gpu_ms.is_some() {
            "CPU + GPU"
        } else {
            "CPU-only fallback (GPU timing pending/unavailable)"
        };

        ui.text(format!("Mode: {mode_label}"));
        let frame_cpu_avg = self
            .frame_timing_window
            .avg_cpu()
            .unwrap_or(timings.frame_cpu_ms);
        let frame_gpu_avg = self.frame_timing_window.avg_gpu();
        let frame_cpu_max = self.frame_timing_peak.cpu_ms.unwrap_or(frame_cpu_avg);
        let frame_gpu_max = self.frame_timing_peak.gpu_ms.or(frame_gpu_avg);
        ui.text(format!(
            "Frame CPU avg/max ({}f): {:.3} / {:.3} ms",
            TIMING_AVG_WINDOW, frame_cpu_avg, frame_cpu_max
        ));
        if frame_gpu_avg.is_some() || frame_gpu_max.is_some() {
            ui.text(format!(
                "Frame GPU avg/max ({}f): {} / {}",
                TIMING_AVG_WINDOW,
                format_gpu_ms(frame_gpu_avg),
                format_gpu_ms(frame_gpu_max)
            ));
        } else {
            ui.text("Frame GPU avg/max: cpu-only/pending");
        }

        if !timings.stage_timings.is_empty() {
            ui.separator();
            ui.text(format!(
                "Pipeline stages (avg {}f, max since reset)",
                TIMING_AVG_WINDOW
            ));
            for row in &timings.stage_timings {
                let window = self.stage_timing_windows.get(row.label);
                let peak = self
                    .stage_timing_peaks
                    .get(row.label)
                    .copied()
                    .unwrap_or_default();
                ui.separator();
                ui.text(row.label);
                ui.text(format!(
                    "  CPU avg {:.3} ms | max {:.3} ms",
                    window.and_then(TimingWindow::avg_cpu).unwrap_or(row.cpu_ms),
                    peak.cpu_ms
                        .unwrap_or(window.and_then(TimingWindow::max_cpu).unwrap_or(row.cpu_ms),)
                ));
                ui.text(format!(
                    "  GPU avg {} | max {}",
                    format_gpu_ms(window.and_then(TimingWindow::avg_gpu).or(row.gpu_ms)),
                    format_gpu_ms(
                        peak.gpu_ms
                            .or(window.and_then(TimingWindow::max_gpu))
                            .or(row.gpu_ms),
                    )
                ));
            }
        }

        if !timings.pass_timings.is_empty() {
            ui.separator();
            ui.text(format!(
                "Render pass timings (avg {}f, max since reset)",
                TIMING_AVG_WINDOW
            ));
            for row in &timings.pass_timings {
                let window = self.pass_timing_windows.get(row.label);
                let peak = self
                    .pass_timing_peaks
                    .get(row.label)
                    .copied()
                    .unwrap_or_default();
                ui.separator();
                ui.text(row.label);
                ui.text(format!(
                    "  CPU avg {:.3} ms | max {:.3} ms",
                    window.and_then(TimingWindow::avg_cpu).unwrap_or(row.cpu_ms),
                    peak.cpu_ms
                        .unwrap_or(window.and_then(TimingWindow::max_cpu).unwrap_or(row.cpu_ms),)
                ));
                ui.text(format!(
                    "  GPU avg {} | max {}",
                    format_gpu_ms(window.and_then(TimingWindow::avg_gpu).or(row.gpu_ms)),
                    format_gpu_ms(
                        peak.gpu_ms
                            .or(window.and_then(TimingWindow::max_gpu))
                            .or(row.gpu_ms),
                    )
                ));
            }
        }
    }

    fn render_performance_graph_tab(&self, ui: &Ui) {
        ui.text("Frame pacing history");
        let cpu_samples: Vec<f32> = self.cpu_frame_history.iter().copied().collect();
        if cpu_samples.len() > 2 {
            ui.plot_lines("Frame pacing history (CPU ms, sampled)", &cpu_samples)
                .graph_size([0.0, 64.0])
                .build();
        } else {
            ui.text_disabled("Need more CPU samples.");
        }

        let gpu_samples: Vec<f32> = self.gpu_frame_history.iter().copied().collect();
        let has_gpu_samples = gpu_samples.iter().any(|value| value.is_finite());
        if has_gpu_samples {
            ui.plot_lines("Frame pacing history (GPU ms, sampled)", &gpu_samples)
                .graph_size([0.0, 64.0])
                .build();
        } else {
            ui.text_disabled("No GPU samples yet.");
        }
    }

    fn render_performance_record_tab(&mut self, ui: &Ui, timings: &DebugTimingSnapshot) {
        ui.text("Timing JSONL");
        ui.input_text(
            "Output path##timing_report_path",
            &mut self.timing_report_path,
        )
        .build();
        if ui.small_button("New Timestamp Path##timing_report_reset_path") {
            self.timing_report_path = default_timing_report_path();
        }

        let mut interval_ms = self.timing_record_interval_ms as i32;
        if ui
            .input_int(
                "Snapshot interval (ms)##timing_record_interval",
                &mut interval_ms,
            )
            .build()
        {
            self.timing_record_interval_ms = interval_ms.max(1) as u64;
        }

        let mut duration_secs = self.timing_record_duration_secs as i32;
        if ui
            .input_int(
                "Record length (sec)##timing_record_duration",
                &mut duration_secs,
            )
            .build()
        {
            self.timing_record_duration_secs = duration_secs.max(1) as u64;
        }

        ui.same_line();
        if self.timing_recording_active {
            if ui.button("Stop Recording##timing_report_stop") {
                self.stop_timing_recording("stopped by user");
            }
        } else if ui.button("Start Recording##timing_report_start") {
            match self.start_timing_recording(timings) {
                Ok(saved_path) => {
                    let message = format!("recording JSONL to: {saved_path}");
                    self.timing_report_status = Some(message.clone());
                    self.push_console_output(message);
                }
                Err(err) => {
                    let message = format!("timing recording failed: {err}");
                    self.timing_report_status = Some(message.clone());
                    self.push_console_output(message);
                }
            }
        }

        if self.timing_recording_active {
            if let (Some(started_at), Some(end_at)) =
                (self.timing_record_started_at, self.timing_record_end_at)
            {
                let now = Instant::now();
                let elapsed = now.saturating_duration_since(started_at).as_secs_f32();
                let remaining = end_at.saturating_duration_since(now).as_secs_f32();
                ui.text(format!(
                    "Recording: {:.2}s elapsed, {:.2}s remaining, {} samples",
                    elapsed.max(0.0),
                    remaining.max(0.0),
                    self.timing_record_samples_written
                ));
            }
        }

        if let Some(status) = self.timing_report_status.as_ref() {
            ui.text(status);
        } else {
            ui.text_disabled("No timing log recorded yet.");
        }

        ui.separator();
        ui.text("Cause table (spike attribution)");
        ui.checkbox(
            "Spike Attribution##toggle_spike_attr",
            &mut self.spike_attribution_enabled,
        );
        ui.same_line();
        ui.checkbox(
            "Record while hidden##toggle_spike_hidden",
            &mut self.spike_record_while_hidden,
        );

        if !self.spike_attribution_enabled {
            ui.text_disabled("Disabled");
            return;
        }

        let baseline = self.frame_timing_window.avg_cpu().unwrap_or(0.0);
        let relative_threshold = baseline * SPIKE_RELATIVE_THRESHOLD_SCALE;
        let effective_threshold = SPIKE_ABSOLUTE_THRESHOLD_MS.max(relative_threshold);
        ui.text(format!(
            "Trigger: frame CPU > max({:.1} ms, {:.1}x avg {:.3} ms) => {:.3} ms",
            SPIKE_ABSOLUTE_THRESHOLD_MS,
            SPIKE_RELATIVE_THRESHOLD_SCALE,
            baseline,
            effective_threshold
        ));

        if self.spike_history.is_empty() {
            ui.text("No spikes recorded yet.");
            return;
        }

        ui.child_window("spike_attribution_history")
            .size([0.0, 96.0])
            .build(|| {
                for spike in self.spike_history.iter() {
                    let stage = spike
                        .top_stage
                        .as_ref()
                        .map(|item| {
                            format!(
                                "{} CPU {:.3}ms GPU {}",
                                item.label,
                                item.cpu_ms,
                                format_gpu_ms(item.gpu_ms)
                            )
                        })
                        .unwrap_or_else(|| "n/a".to_string());
                    let pass = spike
                        .top_pass
                        .as_ref()
                        .map(|item| {
                            format!(
                                "{} CPU {:.3}ms GPU {}",
                                item.label,
                                item.cpu_ms,
                                format_gpu_ms(item.gpu_ms)
                            )
                        })
                        .unwrap_or_else(|| "n/a".to_string());
                    ui.text(format!(
                        "#{} CPU {:.3} ms GPU {} (thr {:.3}) | Stage {} | Pass {}",
                        spike.frame_index,
                        spike.frame_cpu_ms,
                        format_gpu_ms(spike.frame_gpu_ms),
                        spike.threshold_cpu_ms,
                        stage,
                        pass
                    ));
                }
            });
    }

    fn start_timing_recording(&mut self, timings: &DebugTimingSnapshot) -> Result<String, String> {
        let report_path = self.timing_report_path.trim().to_string();
        if report_path.is_empty() {
            return Err("output path is empty".to_string());
        }

        let now = Instant::now();
        let interval = Duration::from_millis(self.timing_record_interval_ms.max(1));
        let duration = Duration::from_secs(self.timing_record_duration_secs.max(1));
        let end_at = now + duration;

        self.prepare_timing_record_path(report_path.as_str())?;
        fs::write(report_path.as_str(), "")
            .map_err(|err| format!("failed to initialize '{}': {err}", report_path))?;

        self.timing_recording_active = true;
        self.timing_record_started_at = Some(now);
        self.timing_record_next_snapshot_at = Some(now + interval);
        self.timing_record_end_at = Some(end_at);
        self.timing_record_samples_written = 0;
        self.timing_record_active_path = Some(report_path.clone());

        self.append_timing_jsonl_snapshot(timings, now, "start")?;
        Ok(report_path)
    }

    fn stop_timing_recording(&mut self, reason: &str) {
        self.timing_recording_active = false;
        self.timing_record_started_at = None;
        self.timing_record_next_snapshot_at = None;
        self.timing_record_end_at = None;
        self.timing_record_active_path = None;
        self.timing_report_status = Some(format!(
            "timing recording {reason}; samples={}",
            self.timing_record_samples_written
        ));
    }

    fn update_timing_recording(&mut self, timings: &DebugTimingSnapshot) {
        if !self.timing_recording_active {
            return;
        }

        let now = Instant::now();
        let Some(end_at) = self.timing_record_end_at else {
            self.stop_timing_recording("stopped (missing end time)");
            return;
        };

        if now >= end_at {
            let _ = self.append_timing_jsonl_snapshot(timings, now, "end");
            self.stop_timing_recording("completed");
            return;
        }

        let Some(next_snapshot_at) = self.timing_record_next_snapshot_at else {
            self.stop_timing_recording("stopped (missing next snapshot)");
            return;
        };

        if now < next_snapshot_at {
            return;
        }

        let interval = Duration::from_millis(self.timing_record_interval_ms.max(1));
        self.timing_record_next_snapshot_at = Some(now + interval);

        if let Err(err) = self.append_timing_jsonl_snapshot(timings, now, "interval") {
            self.stop_timing_recording("aborted");
            let message = format!("timing recording failed: {err}");
            self.timing_report_status = Some(message.clone());
            self.push_console_output(message);
        }
    }

    fn prepare_timing_record_path(&self, report_path: &str) -> Result<(), String> {
        if report_path.is_empty() {
            return Err("output path is empty".to_string());
        }

        let path = Path::new(report_path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|err| {
                    format!(
                        "failed to create parent directory '{}': {err}",
                        parent.display()
                    )
                })?;
            }
        }
        Ok(())
    }

    fn append_timing_jsonl_snapshot(
        &mut self,
        timings: &DebugTimingSnapshot,
        now: Instant,
        reason: &str,
    ) -> Result<(), String> {
        let report_path = self
            .timing_record_active_path
            .as_deref()
            .unwrap_or(self.timing_report_path.trim());
        self.prepare_timing_record_path(report_path)?;

        let elapsed_ms = self
            .timing_record_started_at
            .map(|started| now.saturating_duration_since(started).as_millis() as u64)
            .unwrap_or(0);
        let line = self.build_timing_jsonl_line(timings, elapsed_ms, reason);

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(report_path)
            .map_err(|err| format!("failed to open '{}': {err}", report_path))?;
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|err| format!("failed to append '{}': {err}", report_path))?;

        self.timing_record_samples_written += 1;
        Ok(())
    }

    fn build_timing_jsonl_line(
        &self,
        timings: &DebugTimingSnapshot,
        elapsed_ms: u64,
        reason: &str,
    ) -> String {
        let mut fields = Vec::new();

        let mode_label = if !timings.gpu_supported {
            "CPU-only fallback (GPU timestamps unsupported)"
        } else if timings.frame_gpu_ms.is_some() {
            "CPU + GPU"
        } else {
            "CPU-only fallback (GPU timing pending/unavailable)"
        };

        fields.push(format!(
            "\"record_type\":\"{}\"",
            json_escape("timing_snapshot")
        ));
        fields.push(format!(
            "\"frame_index\":{}",
            self.frame_context.frame_index
        ));
        fields.push(format!(
            "\"wall_timestamp_unix_s\":{}",
            unix_timestamp_seconds().unwrap_or(0)
        ));
        fields.push(format!("\"elapsed_ms\":{}", elapsed_ms));
        fields.push(format!("\"reason\":\"{}\"", json_escape(reason)));
        fields.push(format!("\"mode\":\"{}\"", json_escape(mode_label)));
        fields.push(format!("\"avg_window_frames\":{}", TIMING_AVG_WINDOW));
        fields.push(format!(
            "\"frame_cpu_ms\":{{\"current\":{:.3},\"avg\":{:.3},\"max\":{:.3}}}",
            timings.frame_cpu_ms.max(0.0),
            self.frame_timing_window
                .avg_cpu()
                .unwrap_or(timings.frame_cpu_ms.max(0.0)),
            self.frame_timing_peak
                .cpu_ms
                .or_else(|| self.frame_timing_window.max_cpu())
                .unwrap_or(timings.frame_cpu_ms.max(0.0))
        ));
        fields.push(format!(
            "\"frame_gpu_ms\":{{\"current\":{},\"avg\":{},\"max\":{}}}",
            json_number_or_null(timings.frame_gpu_ms),
            json_number_or_null(self.frame_timing_window.avg_gpu()),
            json_number_or_null(
                self.frame_timing_peak
                    .gpu_ms
                    .or_else(|| self.frame_timing_window.max_gpu())
            )
        ));

        let mut stage_rows = Vec::new();
        for row in timings.stage_timings.iter() {
            let window = self.stage_timing_windows.get(row.label);
            let peak = self
                .stage_timing_peaks
                .get(row.label)
                .copied()
                .unwrap_or_default();
            stage_rows.push(format!(
                "{{\"label\":\"{}\",\"cpu_ms\":{{\"current\":{:.3},\"avg\":{:.3},\"max\":{:.3}}},\"gpu_ms\":{{\"current\":{},\"avg\":{},\"max\":{}}}}}",
                json_escape(row.label),
                row.cpu_ms.max(0.0),
                window
                    .and_then(TimingWindow::avg_cpu)
                    .unwrap_or(row.cpu_ms.max(0.0)),
                peak.cpu_ms.unwrap_or(
                    window
                        .and_then(TimingWindow::max_cpu)
                        .unwrap_or(row.cpu_ms.max(0.0))
                ),
                json_number_or_null(row.gpu_ms),
                json_number_or_null(window.and_then(TimingWindow::avg_gpu).or(row.gpu_ms)),
                json_number_or_null(
                    peak.gpu_ms
                        .or(window.and_then(TimingWindow::max_gpu))
                        .or(row.gpu_ms)
                ),
            ));
        }
        fields.push(format!("\"stages\":[{}]", stage_rows.join(",")));

        let mut pass_rows = Vec::new();
        for row in timings.pass_timings.iter() {
            let window = self.pass_timing_windows.get(row.label);
            let peak = self
                .pass_timing_peaks
                .get(row.label)
                .copied()
                .unwrap_or_default();
            pass_rows.push(format!(
                "{{\"label\":\"{}\",\"cpu_ms\":{{\"current\":{:.3},\"avg\":{:.3},\"max\":{:.3}}},\"gpu_ms\":{{\"current\":{},\"avg\":{},\"max\":{}}}}}",
                json_escape(row.label),
                row.cpu_ms.max(0.0),
                window
                    .and_then(TimingWindow::avg_cpu)
                    .unwrap_or(row.cpu_ms.max(0.0)),
                peak.cpu_ms.unwrap_or(
                    window
                        .and_then(TimingWindow::max_cpu)
                        .unwrap_or(row.cpu_ms.max(0.0))
                ),
                json_number_or_null(row.gpu_ms),
                json_number_or_null(window.and_then(TimingWindow::avg_gpu).or(row.gpu_ms)),
                json_number_or_null(
                    peak.gpu_ms
                        .or(window.and_then(TimingWindow::max_gpu))
                        .or(row.gpu_ms)
                ),
            ));
        }
        fields.push(format!("\"passes\":[{}]", pass_rows.join(",")));

        let mut cause_rows = Vec::new();
        for spike in self.spike_history.iter() {
            let stage_label = spike.top_stage.as_ref().map_or("n/a", |item| item.label);
            let stage_cpu = spike.top_stage.as_ref().map(|item| item.cpu_ms);
            let stage_gpu = spike.top_stage.as_ref().and_then(|item| item.gpu_ms);
            let pass_label = spike.top_pass.as_ref().map_or("n/a", |item| item.label);
            let pass_cpu = spike.top_pass.as_ref().map(|item| item.cpu_ms);
            let pass_gpu = spike.top_pass.as_ref().and_then(|item| item.gpu_ms);
            cause_rows.push(format!(
                "{{\"frame_index\":{},\"frame_cpu_ms\":{:.3},\"frame_gpu_ms\":{},\"threshold_cpu_ms\":{:.3},\"top_stage\":{{\"label\":\"{}\",\"cpu_ms\":{},\"gpu_ms\":{}}},\"top_pass\":{{\"label\":\"{}\",\"cpu_ms\":{},\"gpu_ms\":{}}}}}",
                spike.frame_index,
                spike.frame_cpu_ms,
                json_number_or_null(spike.frame_gpu_ms),
                spike.threshold_cpu_ms,
                json_escape(stage_label),
                json_number_or_null(stage_cpu),
                json_number_or_null(stage_gpu),
                json_escape(pass_label),
                json_number_or_null(pass_cpu),
                json_number_or_null(pass_gpu)
            ));
        }
        fields.push(format!("\"cause_table\":[{}]", cause_rows.join(",")));

        format!("{{{}}}", fields.join(","))
    }

    fn render_enabled_views(&mut self, ui: &Ui) {
        for id in self.ordered_view_ids() {
            let mut panic_message = None;

            if let Some(entry) = self.views.get_mut(&id) {
                if !entry.enabled {
                    continue;
                }

                let result = catch_unwind(AssertUnwindSafe(|| {
                    (entry.callback)(ui, &self.frame_context)
                }));
                if let Err(payload) = result {
                    panic_message = Some(panic_payload_to_string(payload));
                }
            }

            if let Some(message) = panic_message {
                error!("debug view '{}' panicked: {}", id, message);
                self.push_console_output(format!("view '{}' panicked: {}", id, message));
            }

            ui.separator();
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

    fn record_timing_history(&mut self) {
        let timings = self.frame_context.timings.clone();
        let baseline_cpu_ms = self.frame_timing_window.avg_cpu();
        if self.should_record_spike_attribution() {
            self.maybe_record_spike_attribution(baseline_cpu_ms);
        }

        self.frame_timing_window
            .push(timings.frame_cpu_ms, timings.frame_gpu_ms);
        self.frame_timing_peak
            .observe(timings.frame_cpu_ms, timings.frame_gpu_ms);

        for row in timings.stage_timings.iter() {
            self.stage_timing_windows
                .entry(row.label)
                .or_default()
                .push(row.cpu_ms, row.gpu_ms);
            self.stage_timing_peaks
                .entry(row.label)
                .or_default()
                .observe(row.cpu_ms, row.gpu_ms);
        }

        for row in timings.pass_timings.iter() {
            self.pass_timing_windows
                .entry(row.label)
                .or_default()
                .push(row.cpu_ms, row.gpu_ms);
            self.pass_timing_peaks
                .entry(row.label)
                .or_default()
                .observe(row.cpu_ms, row.gpu_ms);
        }

        self.frame_sample_count += 1;
        self.frame_sample_cpu_sum += timings.frame_cpu_ms.max(0.0);
        if let Some(gpu_ms) = timings.frame_gpu_ms {
            self.frame_sample_gpu_sum += gpu_ms.max(0.0);
            self.frame_sample_gpu_count += 1;
        }

        self.update_timing_recording(&timings);

        if self.frame_sample_count >= FRAME_HISTORY_SAMPLE_SIZE {
            self.push_sampled_frame_history(
                self.frame_sample_cpu_sum / self.frame_sample_count as f32,
                if self.frame_sample_gpu_count > 0 {
                    Some(self.frame_sample_gpu_sum / self.frame_sample_gpu_count as f32)
                } else {
                    None
                },
            );

            self.frame_sample_count = 0;
            self.frame_sample_cpu_sum = 0.0;
            self.frame_sample_gpu_sum = 0.0;
            self.frame_sample_gpu_count = 0;
        }
    }

    fn reset_timing_maxima(&mut self) {
        self.frame_timing_peak = TimingPeak::default();
        self.stage_timing_peaks.clear();
        self.pass_timing_peaks.clear();
    }

    fn push_sampled_frame_history(&mut self, cpu_sample_ms: f32, gpu_sample_ms: Option<f32>) {
        self.cpu_frame_history.push_back(cpu_sample_ms.max(0.0));
        while self.cpu_frame_history.len() > MAX_TIMING_HISTORY {
            self.cpu_frame_history.pop_front();
        }

        self.gpu_frame_history
            .push_back(gpu_sample_ms.unwrap_or(f32::NAN));
        while self.gpu_frame_history.len() > MAX_TIMING_HISTORY {
            self.gpu_frame_history.pop_front();
        }
    }

    fn should_record_spike_attribution(&self) -> bool {
        self.spike_attribution_enabled && (self.debug_visible || self.spike_record_while_hidden)
    }

    fn maybe_record_spike_attribution(&mut self, baseline_cpu_ms: Option<f32>) {
        let timings = &self.frame_context.timings;
        let frame_cpu_ms = timings.frame_cpu_ms.max(0.0);
        let baseline_cpu_ms = baseline_cpu_ms.unwrap_or(frame_cpu_ms);
        let threshold_cpu_ms =
            SPIKE_ABSOLUTE_THRESHOLD_MS.max(baseline_cpu_ms * SPIKE_RELATIVE_THRESHOLD_SCALE);
        if frame_cpu_ms <= threshold_cpu_ms {
            return;
        }

        let top_stage = timings
            .stage_timings
            .iter()
            .max_by(|left, right| left.cpu_ms.total_cmp(&right.cpu_ms))
            .map(|row| SpikeTimingSource {
                label: row.label,
                cpu_ms: row.cpu_ms,
                gpu_ms: row.gpu_ms,
            });
        let top_pass = timings
            .pass_timings
            .iter()
            .max_by(|left, right| left.cpu_ms.total_cmp(&right.cpu_ms))
            .map(|row| SpikeTimingSource {
                label: row.label,
                cpu_ms: row.cpu_ms,
                gpu_ms: row.gpu_ms,
            });

        self.spike_history.push_front(SpikeAttribution {
            frame_index: self.frame_context.frame_index,
            frame_cpu_ms,
            frame_gpu_ms: timings.frame_gpu_ms,
            threshold_cpu_ms,
            top_stage,
            top_pass,
        });
        while self.spike_history.len() > MAX_SPIKE_HISTORY {
            self.spike_history.pop_back();
        }
    }
}

fn format_env_handle(handle: EnvironmentHandle) -> String {
    format!("{}:{}", handle.slot, handle.generation)
}

fn format_gpu_ms(gpu_ms: Option<f32>) -> String {
    if let Some(ms) = gpu_ms {
        format!("{ms:.3} ms")
    } else {
        "cpu-only/pending".to_string()
    }
}

fn json_number_or_null(value: Option<f32>) -> String {
    value
        .filter(|candidate| candidate.is_finite())
        .map(|number| format!("{number:.3}"))
        .unwrap_or_else(|| "null".to_string())
}

fn json_escape(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn unix_timestamp_seconds() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

fn default_timing_report_path() -> String {
    let suffix = unix_timestamp_seconds().unwrap_or(0);
    format!("{DEFAULT_TIMING_REPORT_PREFIX}_{suffix}.jsonl")
}

fn average_iter(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for value in values {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }

    (count > 0).then_some(sum / count as f32)
}

fn max_iter(values: impl Iterator<Item = f32>) -> Option<f32> {
    values
        .filter(|value| value.is_finite())
        .max_by(|a, b| a.total_cmp(b))
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

    #[test]
    fn console_and_debug_visibility_are_independent() {
        let mut manager = DebugUiManager::new();

        assert!(!manager.is_console_visible());
        assert!(!manager.is_debug_visible());
        assert!(!manager.is_any_visible());

        manager.set_console_visible(true);
        assert!(manager.is_console_visible());
        assert!(!manager.is_debug_visible());
        assert!(manager.is_any_visible());

        manager.set_debug_visible(true);
        assert!(manager.is_console_visible());
        assert!(manager.is_debug_visible());
        assert!(manager.is_any_visible());

        manager.set_console_visible(false);
        assert!(!manager.is_console_visible());
        assert!(manager.is_debug_visible());
        assert!(manager.is_any_visible());
    }
}
