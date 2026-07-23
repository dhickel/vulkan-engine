#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

// BSP liquid/warp fragment shader — animated UV, two-sided, transparency.

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
layout (set = 1, binding = 2) uniform sampler2DArray lightmapAtlas;

layout (set = 1, binding = 3) uniform BspSurfaceParams {
    vec4 lightmapScaleBias;
    uint styleIndex;
    uint fullbrightBase;
    uint fullbrightCount;
    float alphaThreshold;
    uint animationFrame;
    float animationTime;
    uint _pad0;
    uint _pad1;
} surf;

layout (location = 0) out vec4 outColor;

const float OVERBRIGHT = 2.0;
const float PI_INV = 1.0 / 3.14159265359;

void main()
{
    // Warped UV coordinate driven by time.
    float warp = sin(inUV1.x * 4.0 + surf.animationTime * 2.0) * 0.02 +
                 cos(inUV1.y * 4.0 + surf.animationTime * 1.7) * 0.02;
    vec2 warpedUV0 = inUV0 + vec2(warp);

    vec3 albedo = texture(albedoTex, warpedUV0).rgb;

    // Lightmap irradiance from style-indexed array layer.
    vec3 lightmapIrradiance = texture(lightmapAtlas, vec3(inUV1, float(surf.styleIndex))).rgb;
    vec3 irradiance = lightmapIrradiance * OVERBRIGHT;

    vec3 diffuse = irradiance * albedo * PI_INV;

    // Liquid surfaces are always partially transparent for post-blend depth sort.
    outColor = vec4(diffuse, 0.6);
}
