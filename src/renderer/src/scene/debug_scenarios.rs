//! # Startup Debug Scene Selection
//!
//! Small runtime helper for deterministic scene startup scenarios used in render-path testing.

use crate::data::assimp_util::{self, ModelMeta};
use crate::data::data_cache::VkDataCache;
use crate::data::gpu_data::MaterialShadingModel;
use std::sync::Arc;

pub const DEFAULT_STARTUP_MODEL_PATH: &str = "src/renderer/src/assets/DamagedHelmet.glb";

pub fn load_startup_scene(
    data_cache: Arc<VkDataCache>,
    force_unlit_materials: bool,
) -> Result<ModelMeta, String> {
    let loaded_scene =
        assimp_util::load_model(DEFAULT_STARTUP_MODEL_PATH, data_cache.clone(), false)?;

    if force_unlit_materials {
        data_cache
            .texture_cache
            .lock()
            .unwrap()
            .set_unloaded_material_shading_model(
                &loaded_scene.material_ids,
                MaterialShadingModel::Unlit,
            )?;
    }

    Ok(loaded_scene)
}
