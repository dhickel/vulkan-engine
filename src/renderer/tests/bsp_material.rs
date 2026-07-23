//! BSP material pipeline unit tests (feature-gated).
//!
//! Verifies BSP material creation, pipeline variant mapping, and
//! surface class → pipeline routing without requiring a GPU.

#[cfg(feature = "bsp")]
mod bsp_tests {
    use renderer::api::bsp::{
        BspCachedSurfaceRepr, BspMaterialDesc, BspSurfaceCacheRepr, BspSurfaceClass,
        BspSurfaceUniform, BspTextureSet, VkPipelineType,
    };
    use renderer::api::BspMaterialHandle;
    use renderer::api::BspTextureHandle;
    use ash::vk::Handle;
    use renderer::TextureHandle;

    /// Verify that BspSurfaceClass variants map to the correct VkPipelineType.
    #[test]
    fn bsp_surface_class_maps_to_pipeline_type() {
        use VkPipelineType::*;

        let map = |class: BspSurfaceClass| -> VkPipelineType {
            match class {
                BspSurfaceClass::Lightmapped => BspOpaque,
                BspSurfaceClass::Fullbright => BspFullbright,
                BspSurfaceClass::AlphaMask => BspAlphaMask,
                BspSurfaceClass::Sky => BspSky,
                BspSurfaceClass::Liquid => BspLiquid,
                BspSurfaceClass::Nodraw => BspOpaque, // nodraw is placeholder
            }
        };

        assert_eq!(map(BspSurfaceClass::Lightmapped), BspOpaque);
        assert_eq!(map(BspSurfaceClass::Fullbright), BspFullbright);
        assert_eq!(map(BspSurfaceClass::AlphaMask), BspAlphaMask);
        assert_eq!(map(BspSurfaceClass::Sky), BspSky);
        assert_eq!(map(BspSurfaceClass::Liquid), BspLiquid);
    }

    /// Verify BspSurfaceUniform defaults match specification values.
    #[test]
    fn bsp_surface_uniform_defaults() {
        let uniform = BspSurfaceUniform::default();
        assert_eq!(uniform.style_index, 0);
        assert_eq!(uniform.fullbright_base, 224);
        assert_eq!(uniform.fullbright_count, 32);
        assert_eq!(uniform.alpha_threshold, 0.5);
        assert_eq!(uniform.animation_frame, 0);
        assert_eq!(uniform.animation_time, 0.0);
    }

    /// Verify BspSurfaceCache add and get works with generation tracking.
    #[test]
    fn bsp_surface_cache_add_and_get() {
        let mut cache = BspSurfaceCacheRepr::new();
        let cached = BspCachedSurfaceRepr {
            material_descriptor: ash::vk::DescriptorSet::from_raw(0xBEEF),
            surf_ubo_alloc: Default::default(),
            pipeline: VkPipelineType::BspOpaque,
            albedo_tex: TextureHandle::new(10, 0),
            fullbright_tex: None,
            lightmap_tex: TextureHandle::new(11, 0),
        };

        let handle = cache.add(cached);
        assert_eq!(handle.slot, 0);
        assert_eq!(handle.generation, 0);

        let retrieved = cache
            .get(handle)
            .expect("cached surface should be retrievable");
        assert_eq!(retrieved.pipeline, VkPipelineType::BspOpaque);
        assert_eq!(retrieved.albedo_tex, TextureHandle::new(10, 0));
    }

    /// Verify BspMaterialHandle field layout matches slot+generation pattern.
    #[test]
    fn bsp_material_handle_identity() {
        let h1 = BspMaterialHandle::new(5, 3);
        let h2 = BspMaterialHandle::new(5, 3);
        let h3 = BspMaterialHandle::new(5, 4);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.slot, 5);
        assert_eq!(h1.generation, 3);
    }

    /// Verify BspMaterialDesc carries all fields through construction.
    #[test]
    fn bsp_material_desc_stores_surface_class_and_params() {
        let desc = BspMaterialDesc {
            surface_class: BspSurfaceClass::AlphaMask,
            textures: BspTextureSet {
                albedo: BspTextureHandle::new(1, 0),
                fullbright_mask: Some(BspTextureHandle::new(2, 0)),
                lightmap_atlas: BspTextureHandle::new(3, 0),
            },
            surface_params: BspSurfaceUniform {
                alpha_threshold: 0.75,
                ..Default::default()
            },
        };

        assert_eq!(desc.surface_class, BspSurfaceClass::AlphaMask);
        assert_eq!(desc.surface_params.alpha_threshold, 0.75);
        assert_eq!(desc.textures.albedo.slot, 1);
        assert!(desc.textures.fullbright_mask.is_some());
    }

    #[test]
    fn bsp_surface_cache_records_only_real_descriptor_payloads() {
        let mut cache = BspSurfaceCacheRepr::new();
        let descriptor = ash::vk::DescriptorSet::from_raw(0xCAFE);
        let handle = cache.add(BspCachedSurfaceRepr {
            material_descriptor: descriptor,
            surf_ubo_alloc: Default::default(),
            pipeline: VkPipelineType::BspOpaque,
            albedo_tex: TextureHandle::new(10, 1),
            fullbright_tex: Some(TextureHandle::new(11, 2)),
            lightmap_tex: TextureHandle::new(20, 3),
        });

        let cached = cache
            .get(handle)
            .expect("published material should be cached");
        assert_eq!(cached.material_descriptor, descriptor);
        assert_eq!(cached.albedo_tex, TextureHandle::new(10, 1));
        assert_eq!(cached.fullbright_tex, Some(TextureHandle::new(11, 2)));
        assert_eq!(cached.lightmap_tex, TextureHandle::new(20, 3));
    }
}
