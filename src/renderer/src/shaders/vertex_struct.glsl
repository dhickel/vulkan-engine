#ifndef VERTEX_STRUCTURES_GLSL
#define VERTEX_STRUCTURES_GLSL

struct Vertex {
    vec3 position;
    float uv0_x;
    vec3 normal;
    float uv0_y;
    vec4 color;
    vec4 tangent;
    float uv1_x;
    float uv1_y;
    uvec4 joints;
    vec4 weights;
    int _pad1;
    int _pad2;
};

layout (buffer_reference, std430) readonly buffer VertexBuffer {
    Vertex vertices[];
};

#endif // VERTEX_STRUCTURES_GLSL