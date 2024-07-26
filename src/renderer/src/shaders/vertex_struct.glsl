#ifndef VERTEX_STRUCTURES_GLSL
#define VERTEX_STRUCTURES_GLSL

struct Vertex {
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 color;
    vec4 tangent;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer {
    Vertex vertices[];
};

#endif // VERTEX_STRUCTURES_GLSL