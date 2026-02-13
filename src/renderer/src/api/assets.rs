use super::errors::AssetError;

/// Asset subsystem facade placeholder.
#[derive(Default)]
pub struct AssetManager;

impl AssetManager {
    /// Thread: Main
    /// May Stall: No
    pub fn new() -> Self {
        Self
    }

    /// Thread: Main
    /// May Stall: Yes
    pub fn blocking_upload_all(&mut self) -> Result<(), AssetError> {
        Err(AssetError::Unsupported(
            "asset manager public API is not implemented yet".to_string(),
        ))
    }
}
