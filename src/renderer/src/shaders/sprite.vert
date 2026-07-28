#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_buffer_reference : require

// Sprite vertex shader: pulls per-vertex data from a buffer_reference
// and transforms through an orthographic camera matrix in push constants.

struct SpriteVertex {
    vec2 position;   // world-space position
    vec2 texcoord;   // UV (reserved for future texture support)
    vec4 color;      // per-vertex RGBA
};

layout (buffer_reference, std430) readonly buffer SpriteVertexBuffer {
    SpriteVertex vertices[];
};

layout (push_constant) uniform constants {
    mat4 viewProjection;
    SpriteVertexBuffer vertexBuffer;
} pc;

layout (location = 0) out vec4 outColor;

void main()
{
    SpriteVertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
    gl_Position = pc.viewProjection * vec4(v.position, 0.0, 1.0);
    outColor = v.color;
}
