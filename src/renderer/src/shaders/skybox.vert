#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_buffer_reference : require
#extension GL_GOOGLE_include_directive : enable

#include "vertex_struct.glsl"

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