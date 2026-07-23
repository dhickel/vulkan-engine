//! # Startup Debug Scene Selection
//!
//! Small runtime helper for deterministic scene startup scenarios used in render-path testing.

use crate::api::AssetPolicyConfig;
use crate::data::assimp_util::{self, AssimpImportError, ModelMeta};
use crate::data::data_cache::VkDataCache;
use crate::data::gpu_data::MaterialShadingModel;
use std::path::Path;
use std::sync::Arc;

pub const DEFAULT_STARTUP_MODEL_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb";

pub fn load_startup_scene(
    data_cache: Arc<VkDataCache>,
    force_unlit_materials: bool,
    model_path: &Path,
) -> Result<ModelMeta, AssimpImportError> {
    let path_str = model_path.to_str().ok_or_else(|| {
        AssimpImportError::InvalidPath(format!(
            "startup model path is not valid UTF-8: {}",
            model_path.display()
        ))
    })?;
    let loaded_scene = assimp_util::load_model(
        path_str,
        data_cache.clone(),
        false,
        &AssetPolicyConfig::default(),
    )?;

    if force_unlit_materials {
        let mut tex_cache = data_cache
            .texture_cache
            .lock()
            .map_err(|_| AssimpImportError::Internal("texture_cache lock poisoned".to_string()))?;

        tex_cache
            .set_unloaded_material_shading_model(
                &loaded_scene.material_ids,
                MaterialShadingModel::Unlit,
            )
            .map_err(AssimpImportError::Internal)?;
    }

    Ok(loaded_scene)
}
