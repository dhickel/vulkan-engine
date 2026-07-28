#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_buffer_reference : require

// Debug line vertex: position and color passed as interleaved attributes
// read from a buffer via buffer_reference.

struct DebugLineVertex {
    vec3 position;
    vec3 color;
};

layout (buffer_reference, std430) readonly buffer DebugLineVertexBuffer {
    DebugLineVertex vertices[];
};

layout (push_constant) uniform constants {
    mat4 viewProjection;
    DebugLineVertexBuffer vertexBuffer;
} pc;

layout (location = 0) out vec3 outColor;

void main()
{
    DebugLineVertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
    gl_Position = pc.viewProjection * vec4(v.position, 1.0);
    outColor = v.color;
}
