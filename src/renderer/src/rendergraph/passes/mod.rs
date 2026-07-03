//! # RenderGraph Pass Set
//!
//! Canonical frame-pass list used by the renderer's default rendergraph.

mod debug_capture_pass;
mod geometry_pass;
mod imgui_pass;
mod prepare_targets_pass;
mod present_copy_pass;
mod skybox_pass;
mod terminal_present_pass;

pub use debug_capture_pass::DebugCapturePass;
pub use geometry_pass::GeometryPass;
pub use imgui_pass::ImguiPass;
pub use prepare_targets_pass::PrepareTargetsPass;
pub use present_copy_pass::PresentCopyPass;
pub use skybox_pass::SkyboxPass;
pub use terminal_present_pass::TerminalPresentPass;
