/* Copyright (c) 2018-2023, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

#include "vertex_struct.glsl"
#include "shader_material.glsl"

layout (set = 0, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    vec3 camPos;
} ubo;

#define MAX_NUM_JOINTS 128

layout (set = 1, binding = 0) uniform UBONode {
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout (push_constant) uniform constants {
    mat4 modelMatrix;
    VertexBuffer vertexBuffer;
    MaterialMeta mataterialMeta;
    uint jointCount;
} pc;

layout (location = 0) out vec3 outWorldPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV0;
layout (location = 3) out vec2 outUV1;
layout (location = 4) out vec4 outColor0;

void main()
{
    Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
    outColor0 = v.color;

    vec4 worldPos;
    if (pc.jointCount > 0) {
        // Mesh is skinned
        vec4 weights = v.weights;
        uvec4 joints = v.joints;
        mat4 skinMat =
        weights.x * node.jointMatrix[joints.x] +
        weights.y * node.jointMatrix[joints.y] +
        weights.z * node.jointMatrix[joints.z] +
        weights.w * node.jointMatrix[joints.w];

        worldPos = pc.modelMatrix * skinMat * vec4(v.position, 1.0);
        outNormal = normalize(transpose(inverse(mat3(pc.modelMatrix * skinMat))) * v.normal);
    } else {
        worldPos = pc.modelMatrix * vec4(v.position, 1.0);
        outNormal = normalize(transpose(inverse(mat3(pc.modelMatrix))) * v.normal);
    }

    worldPos.y = -worldPos.y;
    outWorldPos = worldPos.xyz / worldPos.w;
    outUV0 = vec2(v.uv0_x, v.uv0_y);
    outUV1 = vec2(v.uv1_x, v.uv1_y);
    gl_Position = ubo.projection * ubo.view * worldPos;
}