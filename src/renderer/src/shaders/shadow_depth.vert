#version 450
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

#include "vertex_struct.glsl"

layout (push_constant) uniform constants {
    mat4 lightModelViewProjection;
    VertexBuffer vertexBuffer;
    uint pad1;
    uint pad2;
} pc;

void main()
{
    Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];

    gl_Position = pc.lightModelViewProjection * vec4(v.position, 1.0);
}
