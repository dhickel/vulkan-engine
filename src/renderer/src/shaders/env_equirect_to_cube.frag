#version 450

// Equirectangular to cubemap conversion fragment shader.
// Paired with filtered_cube.vert which outputs cube direction in inPos.

layout (location = 0) in vec3 inPos;
layout (location = 0) out vec4 outColor;
layout (binding = 0) uniform sampler2D samplerEquirect;

#define PI 3.1415926535897932384626433832795

void main()
{
	vec3 d = normalize(inPos);
	float u = atan(d.z, d.x) * (0.5 / PI) + 0.5;
	float v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) / PI;
	vec3 color = texture(samplerEquirect, vec2(u, v)).rgb;
	outColor = vec4(color, 1.0);
}
