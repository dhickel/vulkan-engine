#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

#include "vertex_struct.glsl"

layout (set = 0, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    vec3 camPos;
} ubo;

layout (push_constant) uniform constants {
    mat4 modelMatrix;
    VertexBuffer vertexBuffer;
    // No MaterialMeta — BSP materials are push-constant-free beyond model
    // and vertex buffer address.
} pc;

layout (location = 0) out vec3 outWorldPos;
layout (location = 1) out vec2 outUV0;
layout (location = 2) out vec2 outUV1;
layout (location = 3) out vec3 outNormal;

void main()
{
    Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];

    vec4 worldPos = pc.modelMatrix * vec4(v.position, 1.0);
    outWorldPos = worldPos.xyz / worldPos.w;

    // Quake/BSP surfaces use UV mapping from the first two UV channels.
    outUV0 = vec2(v.uv0_x, v.uv0_y);
    outUV1 = vec2(v.uv1_x, v.uv1_y);

    vec3 normal = normalize(transpose(inverse(mat3(pc.modelMatrix))) * v.normal);
    outNormal = normal;

    gl_Position = ubo.projection * ubo.view * worldPos;
}
