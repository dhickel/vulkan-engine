//! Pure target-device resource-budget calculation plus a local Vulkan evidence probe.

use ash::vk;

const CSM_CASCADE_COUNT: u32 = 3;
const CSM_CASCADE_DIM: u32 = 1024;
const CSM_DEPTH_BYTES_PER_TEXEL: u64 = 4;
const FRAME_RESOURCE_SLOTS: u32 = 3;
const INSTANCE_COUNT_BUDGET: u32 = 16_384;
const INSTANCE_STRIDE: u32 = 64;
const INSTANCE_BUFFER_BYTES_PER_FRAME: u64 = INSTANCE_COUNT_BUDGET as u64 * INSTANCE_STRIDE as u64;
const PLANNED_SAMPLED_IMAGES_PER_STAGE: u32 = 9;
const PLANNED_UNIFORM_BUFFERS_PER_STAGE: u32 = 3;
const PLANNED_STORAGE_BUFFERS_PER_STAGE: u32 = 1;
const REQUIRED_PUSH_CONSTANT_BYTES: u32 = 144;
const REQUIRED_BOUND_DESCRIPTOR_SETS: u32 = 4;

#[derive(Debug, Clone)]
struct DeviceBudget {
    device_name: String,
    device_id: u32,
    device_type: String,
    max_image_dimension_2d: u32,
    max_image_array_layers: u32,
    max_framebuffer_layers: u32,
    max_push_constants_size: u32,
    max_bound_descriptor_sets: u32,
    max_per_stage_descriptor_sampled_images: u32,
    max_descriptor_set_sampled_images: u32,
    max_per_stage_descriptor_uniform_buffers: u32,
    max_descriptor_set_uniform_buffers: u32,
    max_per_stage_descriptor_storage_buffers: u32,
    max_descriptor_set_storage_buffers: u32,
    max_storage_buffer_range: u32,
    csm_dimension_ok: bool,
    csm_layers_ok: bool,
    csm_depth_format_ok: bool,
    sampled_image_descriptors_ok: bool,
    uniform_buffer_descriptors_ok: bool,
    storage_buffer_descriptors_ok: bool,
    instance_buffer_range_ok: bool,
    push_constants_ok: bool,
    bound_descriptor_sets_ok: bool,
    shadow_bytes_per_frame: u64,
    shadow_bytes_aggregate: u64,
    instance_bytes_per_frame: u64,
    instance_bytes_aggregate: u64,
    aggregate_feature_bytes: u64,
    overall_pass: bool,
}

impl DeviceBudget {
    fn calculate(props: &vk::PhysicalDeviceProperties, d32_sampled_array_supported: bool) -> Self {
        let limits = &props.limits;
        let device_name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let device_type = format!("{:?}", props.device_type);
        let csm_dimension_ok = limits.max_image_dimension2_d >= CSM_CASCADE_DIM;
        let csm_layers_ok = limits.max_image_array_layers >= CSM_CASCADE_COUNT
            && limits.max_framebuffer_layers >= CSM_CASCADE_COUNT;
        let sampled_image_descriptors_ok = limits.max_per_stage_descriptor_sampled_images
            >= PLANNED_SAMPLED_IMAGES_PER_STAGE
            && limits.max_descriptor_set_sampled_images >= PLANNED_SAMPLED_IMAGES_PER_STAGE;
        let uniform_buffer_descriptors_ok = limits.max_per_stage_descriptor_uniform_buffers
            >= PLANNED_UNIFORM_BUFFERS_PER_STAGE
            && limits.max_descriptor_set_uniform_buffers >= PLANNED_UNIFORM_BUFFERS_PER_STAGE;
        let storage_buffer_descriptors_ok = limits.max_per_stage_descriptor_storage_buffers
            >= PLANNED_STORAGE_BUFFERS_PER_STAGE
            && limits.max_descriptor_set_storage_buffers >= PLANNED_STORAGE_BUFFERS_PER_STAGE;
        let instance_buffer_range_ok =
            limits.max_storage_buffer_range as u64 >= INSTANCE_BUFFER_BYTES_PER_FRAME;
        let push_constants_ok = limits.max_push_constants_size >= REQUIRED_PUSH_CONSTANT_BYTES;
        let bound_descriptor_sets_ok =
            limits.max_bound_descriptor_sets >= REQUIRED_BOUND_DESCRIPTOR_SETS;
        let shadow_bytes_per_frame = CSM_CASCADE_COUNT as u64
            * CSM_CASCADE_DIM as u64
            * CSM_CASCADE_DIM as u64
            * CSM_DEPTH_BYTES_PER_TEXEL;
        let shadow_bytes_aggregate = shadow_bytes_per_frame * FRAME_RESOURCE_SLOTS as u64;
        let instance_bytes_aggregate =
            INSTANCE_BUFFER_BYTES_PER_FRAME * FRAME_RESOURCE_SLOTS as u64;
        let aggregate_feature_bytes = shadow_bytes_aggregate + instance_bytes_aggregate;
        let overall_pass = csm_dimension_ok
            && csm_layers_ok
            && d32_sampled_array_supported
            && sampled_image_descriptors_ok
            && uniform_buffer_descriptors_ok
            && storage_buffer_descriptors_ok
            && instance_buffer_range_ok
            && push_constants_ok
            && bound_descriptor_sets_ok;

        Self {
            device_name,
            device_id: props.device_id,
            device_type,
            max_image_dimension_2d: limits.max_image_dimension2_d,
            max_image_array_layers: limits.max_image_array_layers,
            max_framebuffer_layers: limits.max_framebuffer_layers,
            max_push_constants_size: limits.max_push_constants_size,
            max_bound_descriptor_sets: limits.max_bound_descriptor_sets,
            max_per_stage_descriptor_sampled_images: limits.max_per_stage_descriptor_sampled_images,
            max_descriptor_set_sampled_images: limits.max_descriptor_set_sampled_images,
            max_per_stage_descriptor_uniform_buffers: limits
                .max_per_stage_descriptor_uniform_buffers,
            max_descriptor_set_uniform_buffers: limits.max_descriptor_set_uniform_buffers,
            max_per_stage_descriptor_storage_buffers: limits
                .max_per_stage_descriptor_storage_buffers,
            max_descriptor_set_storage_buffers: limits.max_descriptor_set_storage_buffers,
            max_storage_buffer_range: limits.max_storage_buffer_range,
            csm_dimension_ok,
            csm_layers_ok,
            csm_depth_format_ok: d32_sampled_array_supported,
            sampled_image_descriptors_ok,
            uniform_buffer_descriptors_ok,
            storage_buffer_descriptors_ok,
            instance_buffer_range_ok,
            push_constants_ok,
            bound_descriptor_sets_ok,
            shadow_bytes_per_frame,
            shadow_bytes_aggregate,
            instance_bytes_per_frame: INSTANCE_BUFFER_BYTES_PER_FRAME,
            instance_bytes_aggregate,
            aggregate_feature_bytes,
            overall_pass,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "phase": "Phase 0",
            "device": {
                "name": self.device_name,
                "id": self.device_id,
                "type": self.device_type,
            },
            "limits": {
                "max_image_dimension_2d": self.max_image_dimension_2d,
                "max_image_array_layers": self.max_image_array_layers,
                "max_framebuffer_layers": self.max_framebuffer_layers,
                "max_push_constants_size": self.max_push_constants_size,
                "max_bound_descriptor_sets": self.max_bound_descriptor_sets,
                "max_per_stage_descriptor_sampled_images": self.max_per_stage_descriptor_sampled_images,
                "max_descriptor_set_sampled_images": self.max_descriptor_set_sampled_images,
                "max_per_stage_descriptor_uniform_buffers": self.max_per_stage_descriptor_uniform_buffers,
                "max_descriptor_set_uniform_buffers": self.max_descriptor_set_uniform_buffers,
                "max_per_stage_descriptor_storage_buffers": self.max_per_stage_descriptor_storage_buffers,
                "max_descriptor_set_storage_buffers": self.max_descriptor_set_storage_buffers,
                "max_storage_buffer_range": self.max_storage_buffer_range,
            },
            "planned": {
                "frame_resource_slots": FRAME_RESOURCE_SLOTS,
                "csm": {
                    "cascade_count": CSM_CASCADE_COUNT,
                    "dimension": CSM_CASCADE_DIM,
                    "format": "D32_SFLOAT",
                    "dimension_ok": self.csm_dimension_ok,
                    "layers_ok": self.csm_layers_ok,
                    "sampled_depth_array_format_ok": self.csm_depth_format_ok,
                    "bytes_per_frame": self.shadow_bytes_per_frame,
                    "aggregate_bytes": self.shadow_bytes_aggregate,
                },
                "descriptors": {
                    "sampled_images_per_stage": PLANNED_SAMPLED_IMAGES_PER_STAGE,
                    "uniform_buffers_per_stage": PLANNED_UNIFORM_BUFFERS_PER_STAGE,
                    "storage_buffers_per_stage": PLANNED_STORAGE_BUFFERS_PER_STAGE,
                    "sampled_images_ok": self.sampled_image_descriptors_ok,
                    "uniform_buffers_ok": self.uniform_buffer_descriptors_ok,
                    "storage_buffers_ok": self.storage_buffer_descriptors_ok,
                },
                "push_constant_bytes": REQUIRED_PUSH_CONSTANT_BYTES,
                "push_constants_ok": self.push_constants_ok,
                "bound_descriptor_sets": REQUIRED_BOUND_DESCRIPTOR_SETS,
                "bound_descriptor_sets_ok": self.bound_descriptor_sets_ok,
                "instances": {
                    "count": INSTANCE_COUNT_BUDGET,
                    "stride": INSTANCE_STRIDE,
                    "bytes_per_frame": self.instance_bytes_per_frame,
                    "aggregate_bytes": self.instance_bytes_aggregate,
                    "buffer_range_ok": self.instance_buffer_range_ok,
                },
                "aggregate_shadow_and_instance_bytes": self.aggregate_feature_bytes,
            },
            "overall_pass": self.overall_pass,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_budget_calculator_rejects_unresolved_depth_format() {
        let mut props = vk::PhysicalDeviceProperties::default();
        props.limits.max_image_dimension2_d = CSM_CASCADE_DIM;
        props.limits.max_image_array_layers = CSM_CASCADE_COUNT;
        props.limits.max_framebuffer_layers = CSM_CASCADE_COUNT;
        props.limits.max_per_stage_descriptor_sampled_images = PLANNED_SAMPLED_IMAGES_PER_STAGE;
        props.limits.max_descriptor_set_sampled_images = PLANNED_SAMPLED_IMAGES_PER_STAGE;
        props.limits.max_per_stage_descriptor_uniform_buffers = PLANNED_UNIFORM_BUFFERS_PER_STAGE;
        props.limits.max_descriptor_set_uniform_buffers = PLANNED_UNIFORM_BUFFERS_PER_STAGE;
        props.limits.max_per_stage_descriptor_storage_buffers = PLANNED_STORAGE_BUFFERS_PER_STAGE;
        props.limits.max_descriptor_set_storage_buffers = PLANNED_STORAGE_BUFFERS_PER_STAGE;
        props.limits.max_storage_buffer_range = INSTANCE_BUFFER_BYTES_PER_FRAME as u32;
        props.limits.max_push_constants_size = REQUIRED_PUSH_CONSTANT_BYTES;
        props.limits.max_bound_descriptor_sets = REQUIRED_BOUND_DESCRIPTOR_SETS;
        assert!(!DeviceBudget::calculate(&props, false).overall_pass);
        assert!(DeviceBudget::calculate(&props, true).overall_pass);
    }

    #[test]
    #[ignore = "requires a local Vulkan device"]
    fn device_budget_test() {
        let entry = unsafe { ash::Entry::load() }.expect("load Vulkan entry");
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance =
            unsafe { entry.create_instance(&create_info, None) }.expect("create Vulkan instance");
        let device = unsafe { instance.enumerate_physical_devices() }
            .expect("enumerate Vulkan devices")
            .into_iter()
            .next()
            .expect("no Vulkan physical device; CSM/spatial gates unresolved");
        let props = unsafe { instance.get_physical_device_properties(device) };
        let depth_properties = unsafe {
            instance.get_physical_device_image_format_properties(
                device,
                vk::Format::D32_SFLOAT,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageCreateFlags::empty(),
            )
        };
        let depth_ok = depth_properties.is_ok_and(|properties| {
            properties.max_extent.width >= CSM_CASCADE_DIM
                && properties.max_extent.height >= CSM_CASCADE_DIM
                && properties.max_array_layers >= CSM_CASCADE_COUNT
        });
        let budget = DeviceBudget::calculate(&props, depth_ok);

        let report_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".internal-dev/debug_reports/engine-integration-sprint/phase-01");
        std::fs::create_dir_all(&report_dir).expect("create report directory");
        let json = serde_json::to_string_pretty(&budget.to_json()).unwrap();
        std::fs::write(report_dir.join("device-budget.json"), json).unwrap();
        let markdown = format!(
            "# Device Budget Report — Phase 01\n\n- **Device:** {} (id: {})\n- **Type:** {}\n- **Overall:** {}\n\n## Planned gates\n\n| Gate | Planned | Device limit / support | Result |\n|---|---:|---:|---|\n| D32 sampled depth array | 3 × 1024² | queried image-format properties | {} |\n| Sampled images per stage | {} | {} | {} |\n| Uniform buffers per stage | {} | {} | {} |\n| Storage buffers per stage | {} | {} | {} |\n| Push constants | {} bytes | {} bytes | {} |\n| Bound descriptor sets | {} | {} | {} |\n| Instance buffer per frame | {} bytes | {} bytes | {} |\n\n## Aggregate planned resources\n\n- Shadow depth: {} bytes/frame; {} bytes across {} frame slots.\n- Instance data: {} bytes/frame; {} bytes across {} frame slots.\n- Aggregate shadow + instance allocation allowance: {} bytes.\n",
            budget.device_name,
            budget.device_id,
            budget.device_type,
            if budget.overall_pass { "PASS" } else { "FAIL" },
            pass_fail(budget.csm_depth_format_ok),
            PLANNED_SAMPLED_IMAGES_PER_STAGE,
            budget.max_per_stage_descriptor_sampled_images,
            pass_fail(budget.sampled_image_descriptors_ok),
            PLANNED_UNIFORM_BUFFERS_PER_STAGE,
            budget.max_per_stage_descriptor_uniform_buffers,
            pass_fail(budget.uniform_buffer_descriptors_ok),
            PLANNED_STORAGE_BUFFERS_PER_STAGE,
            budget.max_per_stage_descriptor_storage_buffers,
            pass_fail(budget.storage_buffer_descriptors_ok),
            REQUIRED_PUSH_CONSTANT_BYTES,
            budget.max_push_constants_size,
            pass_fail(budget.push_constants_ok),
            REQUIRED_BOUND_DESCRIPTOR_SETS,
            budget.max_bound_descriptor_sets,
            pass_fail(budget.bound_descriptor_sets_ok),
            budget.instance_bytes_per_frame,
            budget.max_storage_buffer_range,
            pass_fail(budget.instance_buffer_range_ok),
            budget.shadow_bytes_per_frame,
            budget.shadow_bytes_aggregate,
            FRAME_RESOURCE_SLOTS,
            budget.instance_bytes_per_frame,
            budget.instance_bytes_aggregate,
            FRAME_RESOURCE_SLOTS,
            budget.aggregate_feature_bytes,
        );
        std::fs::write(report_dir.join("device-budget.md"), markdown).unwrap();
        unsafe { instance.destroy_instance(None) };
        assert!(budget.overall_pass, "device budget failed: {budget:#?}");
    }

    fn pass_fail(value: bool) -> &'static str {
        if value {
            "PASS"
        } else {
            "FAIL"
        }
    }

    use std::path::PathBuf;
}
