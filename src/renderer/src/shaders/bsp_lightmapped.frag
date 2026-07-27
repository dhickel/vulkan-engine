#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

#include "tonemapping.glsl"

// BSP lightmapped fragment shader — opaque / fullbright / alpha-mask.
//
// Surface classification is driven by `surf.surfaceFlags`:
//   SURF_ALPHA_MASK (1<<0): discard pixels below alphaThreshold
//   SURF_FULLBRIGHT (1<<3): additive emission on top of lit albedo
//
// Lightmap composition:
//   For each of 4 face style slots (styleIds[i] != 255):
//     sample lightmap array layer i
//     multiply by packed frameValues intensity for styleIds[i]
//     accumulate
//
// Overbright (2.0) applied once after style sum.
// Fullbright emission added after diffuse.
// IBL / CSM / dynamic lights controlled by receiveMask.

layout (location = 0) in vec3 inWorldPos;
layout (location = 1) in vec2 inUV0;
layout (location = 2) in vec2 inUV1;
layout (location = 3) in vec3 inNormal;

// Scene bindings (set 0) — shared with PBR path.
layout (set = 0, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    vec3 camPos;
} ubo;

layout (set = 0, binding = 1) uniform EnvironmentParams {
    vec4 light_dir;
    vec4 light_color;
    mat4 light_view_proj;
    float exposure;
    float gamma;
    float prefilter_mips_levels;
    float ibl_ambient_scale;
    // … remainder of EnvironmentUBO is declared but not all fields used here
} env;

layout (set = 0, binding = 3) uniform samplerCube prefilteredMap;

// Material bindings (set 1).
layout (set = 1, binding = 0) uniform sampler2D albedoTex;
layout (set = 1, binding = 1) uniform sampler2D fullbrightMask;
layout (set = 1, binding = 2) uniform sampler2DArray lightmapAtlas;

layout (set = 1, binding = 3) uniform BspSurfaceParams {
    vec4 lightmapScaleBias;
    uvec4 styleIds;          // 4 style slot indices, 255 = unused
    uint fullbrightBase;
    uint fullbrightCount;
    float alphaThreshold;
    uint animationFrame;
    float animationTime;
    uint surfaceFlags;
    uint receiveMask;
    uint lightmapLayerBase;
    float liquidWarpScale;
    float liquidFlowSpeed;
    uvec2 _pad1;
} surf;

// Frame-varying values (set 2).
layout (set = 2, binding = 0) uniform BspFrameValues {
    vec4 styleIntensityPacked[16];
    float liquidWarpTime;
    float liquidFlowTime;
    float globalAnimationTime;
} frameValues;

layout (location = 0) out vec4 outColor;

// ── Surface flag constants ─────────────────────────────────────────────

const uint SURF_ALPHA_MASK       = 1u << 0;
const uint SURF_SKY              = 1u << 1;
const uint SURF_LIQUID           = 1u << 2;
const uint SURF_FULLBRIGHT       = 1u << 3;
const uint SURF_UNLIT_FALLBACK   = 1u << 7;

const uint RECEIVE_IBL            = 1u << 8;
const uint RECEIVE_CSM            = 1u << 9;
const uint RECEIVE_DYNAMIC_LIGHTS = 1u << 10;

// ── Lighting constants ─────────────────────────────────────────────────

const float OVERBRIGHT   = 2.0;
const float PI_INV       = 1.0 / 3.14159265359;

float styleIntensity(uint styleId)
{
    if (styleId == 255u || styleId >= 64u) {
        return 0.0;
    }
    uint vectorIndex = styleId >> 2;
    uint componentIndex = styleId & 3u;
    return frameValues.styleIntensityPacked[vectorIndex][int(componentIndex)];
}

vec3 decodeLightmap(vec3 encoded)
{
    return pow(max(encoded, vec3(0.0)), vec3(2.2));
}

void main()
{
    vec4 albedoSample = texture(albedoTex, inUV0);
    vec3 albedo = albedoSample.rgb;
    float alpha  = albedoSample.a;

    // ── 1. Alpha-mask test ─────────────────────────────────────────

    if ((surf.surfaceFlags & SURF_ALPHA_MASK) != 0u) {
        if (alpha < surf.alphaThreshold) {
            discard;
        }
    }

    // ── 2. Lightmap: 4-style weighted sum ──────────────────────────

    bool hasBakedLightmap = (surf.surfaceFlags & SURF_UNLIT_FALLBACK) == 0u;
    vec2 atlasUv = inUV1 * surf.lightmapScaleBias.xy + surf.lightmapScaleBias.zw;
    vec3 lightmapIrradiance = vec3(0.0);

    if (hasBakedLightmap) {
        // Style slot 0
        if (surf.styleIds.x != 255u) {
            float intensity = styleIntensity(surf.styleIds.x);
            vec3 sample0 = texture(lightmapAtlas, vec3(atlasUv, float(surf.lightmapLayerBase))).rgb;
            lightmapIrradiance += decodeLightmap(sample0) * intensity;
        }
        // Style slot 1
        if (surf.styleIds.y != 255u) {
            float intensity = styleIntensity(surf.styleIds.y);
            vec3 sample1 = texture(lightmapAtlas, vec3(atlasUv, float(surf.lightmapLayerBase + 1u))).rgb;
            lightmapIrradiance += decodeLightmap(sample1) * intensity;
        }
        // Style slot 2
        if (surf.styleIds.z != 255u) {
            float intensity = styleIntensity(surf.styleIds.z);
            vec3 sample2 = texture(lightmapAtlas, vec3(atlasUv, float(surf.lightmapLayerBase + 2u))).rgb;
            lightmapIrradiance += decodeLightmap(sample2) * intensity;
        }
        // Style slot 3
        if (surf.styleIds.w != 255u) {
            float intensity = styleIntensity(surf.styleIds.w);
            vec3 sample3 = texture(lightmapAtlas, vec3(atlasUv, float(surf.lightmapLayerBase + 3u))).rgb;
            lightmapIrradiance += decodeLightmap(sample3) * intensity;
        }
    }

    // Apply transfer function and overbright once after sum.
    vec3 irradiance = lightmapIrradiance * OVERBRIGHT;

    // ── 3. Diffuse Lambertian ──────────────────────────────────────

    vec3 color = hasBakedLightmap ? irradiance * albedo * PI_INV : albedo * 3.0;

    // ── 4. Fullbright emissive ─────────────────────────────────────

    float fullbright = texture(fullbrightMask, inUV0).r;
    // Palette fullbrights emit their source color. Adding a white scalar
    // destroys colored lava, lamps, and trim even when the palette decoded correctly.
    color += fullbright * albedo;

    // ── 5. IBL ambient contribution ────────────────────────────────

    if ((surf.receiveMask & RECEIVE_IBL) != 0u) {
        vec3 N = normalize(inNormal);
        vec3 V = normalize(ubo.camPos - inWorldPos);
        vec3 R = reflect(-V, N);
        float ambientScale = env.ibl_ambient_scale;
        vec3 ibl = texture(prefilteredMap, R).rgb * ambientScale * albedo * PI_INV;
        color += ibl;
    }

    // ── 6. Dynamic direct light (placeholder — Phase 09 CSM impl) ──
    // When RECEIVE_DYNAMIC_LIGHTS is set, sampled via set 0 binding 5
    // (sampler2DArrayShadow) when CSM is active. This path is reserved.

    // ── 7. Output ──────────────────────────────────────────────────

    outColor = tonemap(vec4(color, 1.0), env.exposure, env.gamma);
}
