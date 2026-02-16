//! # Startup Debug Scene Selection
//!
//! Small runtime helper for deterministic scene startup scenarios used in render-path testing.

use crate::api::AssetPolicyConfig;
use crate::data::assimp_util::{self, AssimpImportError, ModelMeta};
use crate::data::data_cache::VkDataCache;
use crate::data::gpu_data::MaterialShadingModel;
use std::sync::Arc;

pub const DEFAULT_STARTUP_MODEL_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb";

pub fn load_startup_scene(
    data_cache: Arc<VkDataCache>,
    force_unlit_materials: bool,
) -> Result<ModelMeta, AssimpImportError> {
    let loaded_scene = assimp_util::load_model(
        DEFAULT_STARTUP_MODEL_PATH,
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
            .map_err(|e| AssimpImportError::Internal(e))?;
    }

    Ok(loaded_scene)
}
