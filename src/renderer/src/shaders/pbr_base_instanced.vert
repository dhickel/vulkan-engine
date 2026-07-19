/* Copyright (c) 2018-2023, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 * Instanced PBR vertex shader variant.
 * Differs from pbr_base.vert in using a frame-local storage buffer
 * (set 0, binding 1) for per-instance model matrices instead of the
 * push-constant model matrix. Intended for rigid opaque same-mesh/material draws.
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

// Per-instance model matrix. Normal matrix is derived via inverse-transpose
// of the upper 3x3 in the shader.
struct InstanceData {
    mat4 modelMatrix;
};

layout (set = 0, binding = 1) readonly buffer InstanceBuffer {
    InstanceData instances[];
} instanceBuffer;

#define MAX_NUM_JOINTS 128

layout (set = 1, binding = 0) uniform UBONode {
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout (push_constant) uniform constants {
    VertexBuffer vertexBuffer;
    MaterialMeta materialMeta;
    uint jointCount;
    uint hasUV1;
    uint _pad1;
    uint _pad2;
} pc;

layout (location = 0) out vec3 outWorldPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV0;
layout (location = 3) out vec2 outUV1;
layout (location = 4) out vec4 outColor0;

void main()
{
    InstanceData inst = instanceBuffer.instances[gl_InstanceIndex];
    Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
    outColor0 = v.color;

    vec4 worldPos;
    mat3 normalMatrix3 = mat3(inst.modelMatrix);
    // For rigid-body transforms, inverse-transpose equals the rotation part.
    // For non-uniform scale (don't instance those), use the general form.
    mat3 normalMat = transpose(inverse(normalMatrix3));

    if (pc.jointCount > 0) {
        vec4 weights = v.weights;
        uvec4 joints = v.joints;
        mat4 skinMat =
        weights.x * node.jointMatrix[joints.x] +
        weights.y * node.jointMatrix[joints.y] +
        weights.z * node.jointMatrix[joints.z] +
        weights.w * node.jointMatrix[joints.w];

        worldPos = inst.modelMatrix * skinMat * vec4(v.position, 1.0);
        outNormal = normalize(normalMat * mat3(skinMat) * v.normal);
    } else {
        worldPos = inst.modelMatrix * vec4(v.position, 1.0);
        outNormal = normalize(normalMat * v.normal);
    }

    outWorldPos = worldPos.xyz / worldPos.w;
    outUV0 = vec2(v.uv0_x, v.uv0_y);
    outUV1 = vec2(v.uv1_x, v.uv1_y);
    gl_Position = ubo.projection * ubo.view * worldPos;
}
