#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_buffer_reference : require

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

layout(push_constant) uniform PushConstants {
	mat4 projection;
	mat4 model;
	VertexBuffer vertexBuffer;
	float exposure;
	float gamma;
} pc;

layout (location = 0) out vec3 outUVW;

void main()
{
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	outUVW = v.position;

	mat4 viewMat = mat4(mat3(pc.model));
	gl_Position = pc.projection * viewMat * vec4(v.position.xyz, 1.0);
}