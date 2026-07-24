#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

// BSP liquid/warp fragment shader — animated UV, two-sided, translucent.
//
// Computes:
//   warped uv = inUV0 + sin/cos warp from liquidFrame.warpTime
//   albedo = texture(albedoTex, warpedUv).rgb
//   lightmap = 4-style weighted sum (same as opaque)
//   diffuse = lightmap * overbright * albedo / PI
//   outColor = vec4(diffuse, liquidAlpha)

layout (location = 0) in vec3 inWorldPos;
layout (location = 1) in vec2 inUV0;
layout (location = 2) in vec2 inUV1;
layout (location = 3) in vec3 inNormal;

// Scene bindings (set 0) — shared.
layout (set = 0, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    vec3 camPos;
} ubo;

// Material bindings (set 1).
layout (set = 1, binding = 0) uniform sampler2D albedoTex;
layout (set = 1, binding = 1) uniform sampler2D fullbrightMask;
layout (set = 1, binding = 2) uniform sampler2DArray lightmapAtlas;

layout (set = 1, binding = 3) uniform BspSurfaceParams {
    vec4 lightmapScaleBias;
    uvec4 styleIds;
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

const float OVERBRIGHT = 2.0;
const float PI_INV     = 1.0 / 3.14159265359;
const float LIQUID_ALPHA = 0.6;

float styleIntensity(uint styleId)
{
    if (styleId == 255u || styleId >= 64u) {
        return 0.0;
    }
    uint vectorIndex = styleId >> 2;
    uint componentIndex = styleId & 3u;
    return frameValues.styleIntensityPacked[vectorIndex][int(componentIndex)];
}

void main()
{
    // ── Warped UV coordinate driven by frame-local time ─────────────

    float warpScale = surf.liquidWarpScale;
    float flowSpeed = surf.liquidFlowSpeed;

    float warpX = sin(inUV1.y * 4.0 + frameValues.liquidWarpTime * 2.3) * warpScale
                + cos(inUV1.x * 5.0 + frameValues.liquidWarpTime * 1.7) * warpScale;
    float warpY = cos(inUV1.x * 4.0 + frameValues.liquidWarpTime * 2.1) * warpScale
                + sin(inUV1.y * 5.0 + frameValues.liquidWarpTime * 1.9) * warpScale;

    // Flow scroll along UV
    float flowU = frameValues.liquidFlowTime * flowSpeed * 0.1;
    float flowV = frameValues.liquidFlowTime * flowSpeed * 0.07;

    vec2 warpedUV = inUV0 + vec2(warpX + flowU, warpY + flowV);

    vec3 albedo = texture(albedoTex, warpedUV).rgb;

    // ── Lightmap 4-style weighted sum ───────────────────────────────

    vec2 atlasUv = inUV1 * surf.lightmapScaleBias.xy + surf.lightmapScaleBias.zw;
    vec3 lightmapIrradiance = vec3(0.0);

    if (surf.styleIds.x != 255u) {
        float intensity = styleIntensity(surf.styleIds.x);
        lightmapIrradiance += texture(lightmapAtlas, vec3(atlasUv, float(surf.lightmapLayerBase))).rgb * intensity;
    }
    if (surf.styleIds.y != 255u) {
        float intensity = styleIntensity(surf.styleIds.y);
        lightmapIrradiance += texture(lightmapAtlas, vec3(atlasUv, float(surf.lightmapLayerBase + 1u))).rgb * intensity;
    }
    if (surf.styleIds.z != 255u) {
        float intensity = styleIntensity(surf.styleIds.z);
        lightmapIrradiance += texture(lightmapAtlas, vec3(atlasUv, float(surf.lightmapLayerBase + 2u))).rgb * intensity;
    }
    if (surf.styleIds.w != 255u) {
        float intensity = styleIntensity(surf.styleIds.w);
        lightmapIrradiance += texture(lightmapAtlas, vec3(atlasUv, float(surf.lightmapLayerBase + 3u))).rgb * intensity;
    }

    vec3 irradiance = lightmapIrradiance * OVERBRIGHT;
    vec3 diffuse = irradiance * albedo * PI_INV;

    // ── Output with alpha for post-blend depth sort ─────────────────

    outColor = vec4(diffuse, LIQUID_ALPHA);
}
