#version 450
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

#include "vertex_struct.glsl"

layout (set = 0, binding = 0) uniform ShadowUBO {
    mat4 lightViewProj;
} shadowUbo;

layout (push_constant) uniform constants {
    mat4 modelMatrix;
    VertexBuffer vertexBuffer;
    uint jointCount;
    uint pad1;
    uint pad2;
    uint pad3;
} pc;

void main()
{
    Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];

    vec4 worldPos = pc.modelMatrix * vec4(v.position, 1.0);
    gl_Position = shadowUbo.lightViewProj * worldPos;
}
