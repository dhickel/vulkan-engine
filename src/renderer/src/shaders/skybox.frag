#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_GOOGLE_include_directive : require

layout (location = 0) in vec3 inUVW;
layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
	mat4 projection;
	mat4 model;
	uint vertex_buffer_addr_low;  // Lower 32 bits of the address
	uint vertex_buffer_addr_high; // Upper 32 bits of the address
	float exposure;
	float gamma;
} pc;

layout (binding = 0) uniform samplerCube samplerEnv;

#include "tonemapping.glsl"
#include "srgbtolinear.glsl"

void main()
{
	vec4 color = textureLod(samplerEnv, inUVW, 1);
	vec3 tonemapped = SRGBtoLINEAR(tonemap(color, pc.exposure, pc.gamma)).rgb;
	outColor = vec4(tonemapped * 1.0, 1.0);
}