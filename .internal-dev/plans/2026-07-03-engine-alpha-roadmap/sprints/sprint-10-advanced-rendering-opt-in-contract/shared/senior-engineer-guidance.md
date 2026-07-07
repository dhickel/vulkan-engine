# Senior Engineer Guidance

## Key Judgments

- Keep the beginner facade boring. Sprint 10 succeeds by making advanced paths explicit, not by making every renderer internal convenient to reach.
- Treat the current rendergraph as ordered pass execution, not a dependency graph. Custom pass APIs require resource declarations and synchronization clarity before they can be trusted.
- Prefer named descriptors over handles. A descriptor can say "depth texture is available for observation"; a raw handle silently transfers lifecycle and synchronization obligations to the user.
- Feature gates are part of the contract. Validate both default and `advanced-interop` builds.
- Documentation drift is a product bug here. Old docs that encourage custom command recording through safe hooks are dangerous because the live hook context does not support that.

## Direct Targets

- `src/renderer/src/api/advanced.rs`: feature-gated unsafe interop and any new advanced named types.
- `src/renderer/src/api/hooks.rs`: safe hook context contract, only if fields/docs/tests need correction.
- `src/renderer/src/api/mod.rs` and `src/renderer/src/lib.rs`: export and feature-gate boundaries.
- `src/renderer/src/rendergraph/mod.rs`: only if a worker adds a constrained feature-gated extension; otherwise read-only for contract validation.
- `docs/api/05-render-hooks-and-extension-points.md` and `docs/api/05-hooks.md`: duplicate hook contract cleanup.
- `docs/api/00-index.md`: public tiering.
- `docs/internal/07-rendergraph-dependencies-and-aliasing.md`: internal current-truth update if implementation changes or residuals need recording.

## Likely Failure Modes

- Accidentally adding `advanced` or rendergraph items to `renderer::prelude`.
- Letting feature-enabled public rendergraph look stable without warning labels.
- Adding a "temporary" raw handle field that becomes a de facto contract.
- Running only default `cargo check` and missing feature-gated compilation failures.
- Treating present-target capture or desktop screenshot as visual proof.
- Updating old docs but leaving duplicate pages with contradictory claims.

## Decision Rules

- If a feature can be done through existing safe hooks/debug views, document that path first.
- If a feature requires raw backend access, keep it behind unsafe `advanced-interop` or defer it.
- If implementation must touch pass order or capture/readback, require runtime smoke and, when visible/readback behavior changes, headless draw capture.
- If Sprint 09 facade changes are unresolved, stop before editing overlapping API exports.
