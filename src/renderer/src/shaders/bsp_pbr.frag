#version 450
#extension GL_GOOGLE_include_directive: require

#include "tonemapping.glsl"

// BSP PBR fragment path. The baked BSP lightmap replaces the environment's
// diffuse irradiance term; the existing prefiltered environment + BRDF LUT
// provide roughness-dependent specular IBL.

layout (location = 0) in vec3 inWorldPos;
layout (location = 1) in vec2 inUV0;
layout (location = 2) in vec2 inUV1;
layout (location = 3) in vec3 inNormal;

layout (set = 0, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    vec3 camPos;
} ubo;

layout (set = 0, binding = 1) uniform EnvironmentParams {
    vec4 lightDir;
    vec4 lightColor;
    mat4 lightViewProj;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
} env;

layout (set = 0, binding = 3) uniform samplerCube prefilteredMap;
layout (set = 0, binding = 4) uniform sampler2D samplerBRDFLUT;

layout (set = 1, binding = 0) uniform sampler2D albedoTex;
// R = fullbright mask, G/B = tangent normal X/Y, A = gloss.
layout (set = 1, binding = 1) uniform sampler2D materialDataTex;
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

layout (set = 2, binding = 0) uniform BspFrameValues {
    vec4 styleIntensityPacked[16];
    float liquidWarpTime;
    float liquidFlowTime;
    float globalAnimationTime;
} frameValues;

layout (location = 0) out vec4 outColor;

const uint SURF_ALPHA_MASK = 1u << 0;
const uint SURF_PBR_NORMAL = 1u << 5;
const uint SURF_PBR_GLOSS = 1u << 6;
const uint SURF_UNLIT_FALLBACK = 1u << 7;
const uint RECEIVE_IBL = 1u << 8;

const float OVERBRIGHT = 2.0;
const float PI_INV = 1.0 / 3.14159265359;
const float MIN_ROUGHNESS = 0.04;

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

vec3 sampleBakedIrradiance(vec2 atlasUv)
{
    vec3 irradiance = vec3(0.0);
    for (uint slot = 0u; slot < 4u; ++slot) {
        uint styleId = surf.styleIds[int(slot)];
        if (styleId == 255u) {
            continue;
        }
        vec3 encoded = texture(
            lightmapAtlas,
            vec3(atlasUv, float(surf.lightmapLayerBase + slot))
        ).rgb;
        irradiance += decodeLightmap(encoded) * styleIntensity(styleId);
    }
    return irradiance * OVERBRIGHT;
}

vec3 mappedNormal(vec4 materialData)
{
    vec3 N = normalize(inNormal);
    if ((surf.surfaceFlags & SURF_PBR_NORMAL) == 0u) {
        return N;
    }

    vec2 tangentXY = materialData.gb * 2.0 - 1.0;
    float tangentZ = sqrt(max(1.0 - dot(tangentXY, tangentXY), 0.0));
    vec3 tangentNormal = normalize(vec3(tangentXY, tangentZ));

    vec3 q1 = dFdx(inWorldPos);
    vec3 q2 = dFdy(inWorldPos);
    vec2 st1 = dFdx(inUV0);
    vec2 st2 = dFdy(inUV0);
    vec3 tangent = q1 * st2.y - q2 * st1.y;
    tangent -= N * dot(N, tangent);
    if (dot(tangent, tangent) < 1e-8) {
        vec3 axis = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        tangent = cross(axis, N);
    }
    vec3 T = normalize(tangent);
    vec3 B = -normalize(cross(N, T));
    return normalize(mat3(T, B, N) * tangentNormal);
}

void main()
{
    vec4 albedoSample = texture(albedoTex, inUV0);
    if ((surf.surfaceFlags & SURF_ALPHA_MASK) != 0u
        && albedoSample.a < surf.alphaThreshold) {
        discard;
    }

    vec4 materialData = texture(materialDataTex, inUV0);
    vec3 N = mappedNormal(materialData);
    vec3 V = normalize(ubo.camPos - inWorldPos);
    float NdotV = clamp(abs(dot(N, V)), 0.001, 1.0);

    float gloss = (surf.surfaceFlags & SURF_PBR_GLOSS) != 0u
        ? materialData.a
        : 0.0;
    float perceptualRoughness = clamp(1.0 - gloss, MIN_ROUGHNESS, 1.0);

    bool hasBakedLightmap = (surf.surfaceFlags & SURF_UNLIT_FALLBACK) == 0u;
    vec2 atlasUv = inUV1 * surf.lightmapScaleBias.xy + surf.lightmapScaleBias.zw;
    vec3 bakedIrradiance = vec3(0.0);
    if (hasBakedLightmap) {
        bakedIrradiance = sampleBakedIrradiance(atlasUv);
    }

    // Dielectric BSP materials are non-metallic. The baked lightmap remains the
    // only diffuse irradiance source and is energy-balanced against specular IBL.
    vec3 f0 = vec3(0.04);
    vec3 brdf = texture(
        samplerBRDFLUT,
        vec2(NdotV, 1.0 - perceptualRoughness)
    ).rgb;
    vec3 kS = clamp(f0 * brdf.x + brdf.y, vec3(0.0), vec3(1.0));
    vec3 kD = vec3(1.0) - kS;
    vec3 color = hasBakedLightmap
        ? bakedIrradiance * albedoSample.rgb * kD * PI_INV
        : albedoSample.rgb * 3.0;

    if ((surf.receiveMask & RECEIVE_IBL) != 0u) {
        vec3 reflection = normalize(reflect(-V, N));
        float lod = perceptualRoughness * env.prefilteredCubeMipLevels;
        vec3 specularLight = textureLod(prefilteredMap, reflection, lod).rgb;
        color += specularLight * kS * env.scaleIBLAmbient;
    }

    // Palette fullbright remains additive and hue-preserving on PBR surfaces.
    color += materialData.r * albedoSample.rgb;

    outColor = tonemap(vec4(color, albedoSample.a), env.exposure, env.gamma);
}
