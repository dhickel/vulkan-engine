# 11 - Alpha Limits and Near-Term Roadmap

This chapter records current alpha constraints and what to prioritize next for dogfooding quality.

## Current Alpha Limits (As-Is)

- Headless mode API flag exists but is not implemented.
- Some low-level cleanup/destruction paths are still incomplete.
- Runtime contains many `unwrap`/panic hot-path calls in Vulkan core.
- Environment switch path may produce frame-time hitches on first activation.
- Renderer crate currently emits a very large warning set in build/test output.
- App-side content packs still need stronger machine-readable conventions in downstream integrations.

## What Is Good Enough for Alpha Dogfooding

- Facade API is usable for basic game loop integration.
- Typed error surface exists across renderer/scene/assets/hooks.
- Scene fragment model enables composition from imported assets.
- Deferred loading exists with ticket polling semantics and bounded in-flight scheduling.
- Texture policy pipeline supports sidecar manifests and per-call overrides.
- Example suite covers key usage patterns and compiles.

## Recommended Near-Term Priorities

1. Eliminate high-risk runtime panics in render/sync hot paths.
2. Improve transform propagation correctness and scene hierarchy mutation confidence.
3. Continue swapchain/environment transition hardening with targeted runtime tests.
4. Push downstream apps toward manifest-driven content definitions (paths, fallback policy, validation).
5. Trim warning surface to increase signal in CI and developer workflows.

## Release Guidance

Before broad alpha announcement, run:
- Example smoke matrix on at least one Vulkan-capable Linux machine.
- Example smoke matrix on at least one Vulkan-capable Windows machine.
- Stress runs for resize spam + environment switching + deferred load churn.

## Learn More

- Dogfooding playbook: `08_examples_dogfooding_playbook.md`
- Review artifact: `.internal-dev/reviews/2026-02-13-rendering-api-alpha-readiness-review.md`
