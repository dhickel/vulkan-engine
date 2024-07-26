#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_GOOGLE_include_directive : enable

#include "vertex_struct.glsl"

layout(push_constant) uniform PushConstants {
	layout (offset = 0) mat4 mvp;
	layout (offset = 64) VertexBuffer vertexBuffer;
} pc;

layout (location = 0) out vec3 outUVW;

void main()
{
	Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];
	vec3 inPos = v.position;
	outUVW = inPos;
	gl_Position = pc.mvp * vec4(inPos.xyz, 1.0);
}