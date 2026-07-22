//! Asynchronous asset-loading ticket and status types.
//!
//! `LoadTicket` is an opaque handle returned by async load requests. The engine
//! pumps background tasks each frame and callers poll `LoadStatus` to observe
//! completion.
//!
//! ## Error propagation (Phase 05)
//!
//! `LoadStatus::Failed` carries the structured [`AssetError`], which preserves
//! asynchronous asset-load failure domains such as Assimp import failures
//! ([`AssimpImportError`]) and texture decode errors. Shader module loading uses
//! a separate typed [`ShaderLoadError`] at the Vulkan utility boundary instead of
//! stringly validation inside `load_shader_module`.

use std::time::Instant;

use super::errors::AssetError;

/// Opaque handle for an in-flight asynchronous asset load.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LoadTicket(u64);

impl LoadTicket {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

/// Observed state of an asynchronous load identified by a [`LoadTicket`].
///
/// `Failed` carries a structured [`AssetError`] that preserves the original
/// asset-load domain (e.g., Assimp import or texture decode) for deterministic
/// handling.
pub enum LoadStatus<T> {
    /// Load is queued and has not yet started background processing.
    Pending { queued_at: Instant },
    /// Data has been uploaded to the GPU but not yet merged into a scene.
    Uploaded { value: T },
    /// The load failed with a structured error.
    Failed { error: AssetError },
    /// The load was cancelled by the caller.
    Cancelled,
}
