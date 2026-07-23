#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

// BSP lightmapped fragment shader — opaque / alpha-mask RGB-style modulation.
//
// Computes:
//   albedo  = texture(albedo_tex, uv0).rgb
//   emissive = fullbright mask check (palette index range)
//   lightmap = texture(lightmap_atlas, vec3(uv1, style_index)).rgb
//   irradiance = lightmap * overbright (2.0)
//   diffuse = irradiance * albedo / PI
//   color = diffuse + emissive * fullbright_color
//
// Dynamic light integration and CSM shadow sampling are reserved for later passes.

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

// Material bindings (set 1).
layout (set = 1, binding = 0) uniform sampler2D albedoTex;
layout (set = 1, binding = 1) uniform sampler2D fullbrightMask;
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
const vec4 FULLBRIGHT_EMISSIVE = vec4(1.0, 1.0, 1.0, 0.0);

void main()
{
    vec4 albedoSample = texture(albedoTex, inUV0);
    vec3 albedo = albedoSample.rgb;
    float alpha  = albedoSample.a;

    // Lightmap irradiance from style-indexed array layer.
    vec2 atlasUv = inUV1 * surf.lightmapScaleBias.xy + surf.lightmapScaleBias.zw;
    vec3 lightmapIrradiance = texture(lightmapAtlas, vec3(atlasUv, float(surf.styleIndex))).rgb;

    // Decode sRGB-like lightmap byte → linear.
    // Lightmap bytes are stored as pow(value, 2.2) in atlas.
    // The atlas format is already linear (uploaded as-is from CPU decode),
    // so we apply overbright directly.
    vec3 irradiance = lightmapIrradiance * OVERBRIGHT;

    // Diffuse Lambertian from baked lightmap only (no specular).
    vec3 diffuse = irradiance * albedo * PI_INV;

    // Fullbright emissive mask: palette-index based.
    // The fullbright mask texture stores 0 (lit) or 1 (fullbright).
    float fullbright = texture(fullbrightMask, inUV0).r;
    vec3 emissiveOut = fullbright * FULLBRIGHT_EMISSIVE.rgb;

    vec3 color = diffuse + emissiveOut;

    // Alpha-mask: discard pixels below threshold when alpha test is active.
    // Alpha from albedo texture's a channel; opaque surfaces have a == 1.
    if (alpha < surf.alphaThreshold) {
        discard;
    }

    outColor = vec4(color, 1.0);
}
