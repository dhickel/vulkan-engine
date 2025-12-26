use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RendererConfig {
    pub shader_dir: String,
    pub shader_files: ShaderFiles,
    pub assets: AssetPaths,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ShaderFiles {
    pub pbr_vert: String,
    pub pbr_frag: String,
    pub pbr_frag_unlit: String,
    pub brdf_lut_frag: String,
    pub brdf_lut_vert: String,
    pub skybox_frag: String,
    pub skybox_vert: String,
    pub cube_filter_vert: String,
    pub env_irradiance_frag: String,
    pub env_prefilter_frag: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AssetPaths {
    pub skybox_dir: String,
    pub default_model: String,
    pub cache_dir: String,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            shader_dir: "src/renderer/src/shaders".to_string(),
            shader_files: ShaderFiles {
                pbr_vert: "pbr_base.vert.spv".to_string(),
                pbr_frag: "material_pbr.frag.spv".to_string(),
                pbr_frag_unlit: "material_unlit.frag.spv".to_string(),
                brdf_lut_frag: "gen_brd_flut.frag.spv".to_string(),
                brdf_lut_vert: "gen_brd_flut.vert.spv".to_string(),
                skybox_frag: "skybox.frag.spv".to_string(),
                skybox_vert: "skybox.vert.spv".to_string(),
                cube_filter_vert: "filtered_cube.vert.spv".to_string(),
                env_irradiance_frag: "env_irradiance_cube.frag.spv".to_string(),
                env_prefilter_frag: "env_prefilter_cube.frag.spv".to_string(),
            },
            assets: AssetPaths {
                skybox_dir: "src/renderer/src/assets/sky_maps/sky".to_string(),
                default_model: "src/renderer/src/assets/DamagedHelmet.glb".to_string(),
                cache_dir: "assets/cache".to_string(),
            },
        }
    }
}

impl RendererConfig {
    pub fn load(path: &str) -> Self {
        if let Ok(content) = fs::read_to_string(path) {
            match toml::from_str(&content) {
                Ok(config) => {
                    println!("Loaded configuration from {}", path);
                    return config;
                },
                Err(e) => eprintln!("Failed to parse config: {}, using defaults", e),
            }
        } else {
            eprintln!("Config file not found at {}, using defaults", path);
        }
        Self::default()
    }

    pub fn get_shader_path(&self, filename: &str) -> String {
        Path::new(&self.shader_dir).join(filename).to_string_lossy().to_string()
    }
}
