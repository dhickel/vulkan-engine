//! # RenderGraph Pass Set
//!
//! Canonical frame-pass list used by the renderer's default rendergraph.

mod debug_capture_pass;
#[cfg(feature = "debug-draw")]
mod debug_lines;
mod geometry_pass;
mod imgui_pass;
mod prepare_targets_pass;
mod present_copy_pass;
mod shadow_pass;
mod skybox_pass;
#[cfg(feature = "sprites-2d")]
mod sprites;
mod terminal_present_pass;

pub use debug_capture_pass::DebugCapturePass;
#[cfg(feature = "debug-draw")]
pub use debug_lines::DebugLinesPass;
pub use geometry_pass::GeometryPass;
pub use imgui_pass::ImguiPass;
pub use prepare_targets_pass::PrepareTargetsPass;
pub use present_copy_pass::PresentCopyPass;
pub use shadow_pass::ShadowPass;
pub use skybox_pass::SkyboxPass;
#[cfg(feature = "sprites-2d")]
pub use sprites::SpritesPass;
pub use terminal_present_pass::TerminalPresentPass;
