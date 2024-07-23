#version 450
#extension GL_GOOGLE_include_directive : require

layout (location = 0) in vec3 inUVW;
layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
	mat4 projection_model;
	layout(offset = 64) float exposure;
	float gamma;
} pc;

layout (binding = 0) uniform samplerCube samplerEnv;

#include "tonemapping.glsl"
#include "srgbtolinear.glsl"

void main()
{
	vec4 color = textureLod(samplerEnv, inUVW, 1.5);
	vec3 tonemapped = SRGBtoLINEAR(tonemap(color, pc.exposure, pc.gamma)).rgb;
	outColor = vec4(tonemapped * 1.0, 1.0);
}