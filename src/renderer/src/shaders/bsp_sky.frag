#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

// BSP sky fragment shader — depth=1.0 pass with environment sampling.
//
// Sky surfaces write maximum depth so the engine sky/environment renders
// behind them. The fragment color is taken from the sky/environment cube map.

layout (location = 0) in vec3 inWorldPos;

// Scene bindings (set 0) — shared.
layout (set = 0, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    vec3 camPos;
} ubo;

// Sky/environment cube map sampler.
layout (set = 0, binding = 3) uniform samplerCube prefilteredMap;

layout (location = 0) out vec4 outColor;

void main()
{
    // Sky surfaces render the environment color and write max depth.
    vec3 viewDir = normalize(inWorldPos - ubo.camPos);
    vec3 envColor = texture(prefilteredMap, viewDir).rgb;
    outColor = vec4(envColor, 1.0);
    gl_FragDepth = 1.0;
}
