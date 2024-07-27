/* Copyright (c) 2018-2023, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

struct ShaderMaterial {
	vec4 baseColorFactor;
	vec4 emissiveFactor;
	int baseColorTextureSet;
	int metallicRoughnessSet;
	int normalTextureSet;	
	int occlusionTextureSet;
	int emissiveTextureSet;
	float metallicFactor;	
	float roughnessFactor;
	float emissiveStrength;
	float alphaMask;	
	float alphaMaskCutoff;

};