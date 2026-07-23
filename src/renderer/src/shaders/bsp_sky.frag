#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable

// BSP sky fragment shader — environment sampling, depth-tested, no depth write.
//
// Sky surfaces render the environment color sampled from the prefiltered
// cube map. They depth-test against scene geometry but do NOT write depth
// and do NOT set gl_FragDepth. The skybox/environment pass draws behind them.

layout (location = 0) in vec3 inWorldPos;

// Scene bindings (set 0) — shared.
layout (set = 0, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    vec3 camPos;
} ubo;

// Prefiltered environment cube map at set 0 binding 3.
layout (set = 0, binding = 3) uniform samplerCube prefilteredMap;

// Material bindings (set 1) — declared for layout compatibility.
layout (set = 1, binding = 3) uniform BspSurfaceParams {
    vec4 lightmapScaleBias;
    uvec4 styleIds;
    uint fullbrightBase;
    uint fullbrightCount;
    float alphaThreshold;
    uint animationFrame;
    float animationTime;
    uint surfaceFlags;
    uint receiveMask;
    uint _pad0;
    float liquidWarpScale;
    float liquidFlowSpeed;
    uvec2 _pad1;
} surf;

// Frame-varying values (set 2) — declared for layout compatibility.
layout (set = 2, binding = 0) uniform BspFrameValues {
    vec4 styleIntensityPacked[16];
    float liquidWarpTime;
    float liquidFlowTime;
    float globalAnimationTime;
} frameValues;

const uint RECEIVE_IBL = 1u << 8;

layout (location = 0) out vec4 outColor;

void main()
{
    // Sky surfaces: sample environment color.
    // No gl_FragDepth — depth is written (or not) by pipeline state.
    vec3 viewDir = normalize(inWorldPos - ubo.camPos);
    vec3 envColor = texture(prefilteredMap, viewDir).rgb;
    outColor = vec4(envColor, 1.0);
}
