/* Copyright (c) 2018-2024, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

#include "vertex_struct.glsl"
#include "shader_material.glsl"
#include "srgbtolinear.glsl"

layout (location = 0) in vec3 inWorldPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec2 inUV1;
layout (location = 4) in vec4 inColor0;

layout (set = 0, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    vec3 camPos;
} ubo;

layout (set = 0, binding = 1) uniform UBOParams {
    vec4 lightDir;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    float debugViewInputs;
    float debugViewEquation;
} uboParams;

layout (set = 2, binding = 0) uniform sampler2D colorMap;
layout (set = 2, binding = 1) uniform sampler2D roughnessMap;
layout (set = 2, binding = 2) uniform sampler2D normalMap;
layout (set = 2, binding = 3) uniform sampler2D aoMap;
layout (set = 2, binding = 4) uniform sampler2D emissiveMap;

layout (push_constant) uniform constants {
    mat4 modelMatrix;
    VertexBuffer vertexBuffer;
    MaterialMeta mataterialMeta;
    uint jointCount;
    uint _pad1;
    uint _pad2;
    uint _pad3;
} pc;

layout (location = 0) out vec4 outColor;

void main()
{
    MaterialMeta material = pc.mataterialMeta;

    vec4 baseColor = material.baseColorFactor;
    if (material.baseColorTextureSet > -1) {
        vec2 uv = material.baseColorTextureSet == 0 ? inUV0 : inUV1;
        baseColor = SRGBtoLINEAR(texture(colorMap, uv)) * baseColor;
    }
    baseColor *= inColor0;

    if (material.alphaMask == 1.0f && baseColor.a < material.alphaMaskCutoff) {
        discard;
    }

    vec3 emissive = material.emissiveFactor.rgb * material.emissiveStrength;
    if (material.emissiveTextureSet > -1) {
        vec2 uv = material.emissiveTextureSet == 0 ? inUV0 : inUV1;
        emissive *= SRGBtoLINEAR(texture(emissiveMap, uv)).rgb;
    }

    outColor = vec4(baseColor.rgb + emissive, baseColor.a);
}
