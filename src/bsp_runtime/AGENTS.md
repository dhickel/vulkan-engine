# bsp_runtime — Transactional Source-Linked BSP Runtime

## Scope

`bsp_runtime` is the dedicated integration coordinator for BSP map data in the engine. It owns the two-step prepare/validate/commit transaction, generation-token guards, idempotent rollback, unload/reload/reimport semantics, app bridge orchestration, cache identity computation, and source-link persistence.

## Dependency Rules

- `bsp_runtime` depends on `bsp`, `renderer` (with `bsp` feature), and `engine_events`.
- It does NOT depend on `physics`, any app crate, or the root `engine` crate.
- The `renderer` does NOT depend on `bsp_runtime` (one-way dependency).

## Module Map

| module | role |
|--------|------|
| `lib.rs` | crate root, re-exports |
| `error.rs` | `BspRuntimeError` — all coordinator errors |
| `generation.rs` | `BspGenerationToken`, `BspGenerationCounter`, cancellation |
| `bridge.rs` | `AppBridge` trait, `BridgeAggregator`, DTO types |
| `cache.rs` | `CacheIdentity` — deterministic cache fingerprint |
| `source_link.rs` | `BspSourceLink`, `BspSourceReference`, override reconciliation |
| `coordinator.rs` | `BspCoordinator` — the main transaction coordinator |

## Key Contracts

### Two-Step Transaction
1. `prepare(bsp_bytes)` — parse, extract, stage (hidden)
2. `set_renderer_mount_ready(token, mount)` — accept the move-only prepared lease
3. `validate_for_scene(token, scene)` — check generation, renderer readiness, bridges, and scene preflight
4. `commit(token, scene)` — publish the already validated lease
5. `rollback()` — idempotent staged-candidate cleanup

`commit_with_mount` is a legacy convenience wrapper; it must not be used after a lease was already set ready.

### Generation Tokens
- Monotonic increment on each `prepare()`
- Cancellation: newer prepare invalidates previous staged state
- `validate()` and `commit()` check generation matches

### Bridge Hooks
- `prepare` — app creates resources from DTOs
- `validate` — app confirms readiness
- `commit` — app publishes to simulation
- `rollback` — app removes resources (idempotent)

### Poisoning
- Panic in bridge commit/rollback → coordinator poisoned
- `is_poisoned()` query, no further BSP operations accepted

### Retirement Handoff API (Phase A — EnhancedV3)

Every `DetachedBspMount` produced by replacement, unload, rollback, stale upload, or
teardown is deposited into the coordinator's pending-retirement queue exactly once.
The coordinator never silently drops a detached receipt. Callers must drain and submit
each mount to `Renderer::retire_bsp_mount`.

| method | role |
|--------|------|
| `pending_retirement_count()` | current queue depth |
| `retired_mount_count()` | cumulative detachment count (diagnostic only) |
| `retirement_diagnostics()` | alias for `retired_mount_count` |
| `drain_pending_retirements()` | take all queued `DetachedBspMount`s, leaving queue empty |
| `requeue_retirement(detached)` | return a previously drained receipt to the queue (e.g. after rejection retry) |

If `Renderer::retire_bsp_mount` rejects with a `BspRetirementRejection`, reconstruct
the `DetachedBspMount` via `BspRetirementRejection::into_detached()` and requeue it.
`Scene::clear_bsp_mount()` likewise returns (rather than drops) its detached receipt,
so direct scene users must submit or requeue that receipt as well.

### Current Lifecycle Boundary
- `PreparedBspMount` is move-only and `Scene::retire_bsp_mount()` detaches it from scene submission.
- Detached mounts are now queued for explicit renderer fence-aware retirement via the handoff API above.
- Generic committed-bridge teardown is blocked on GitHub #60.

## Validation

```bash
cargo check -p bsp_runtime
cargo test -p bsp_runtime
cargo check -p renderer --features bsp
```

## Related

- Transaction/Ownership spec: `.internal-dev/specifications/bsp-transaction-ownership.md`
- Acceptance spec: `.internal-dev/specifications/bsp-acceptance.md`
- Dungeon generation spec: `.internal-dev/specifications/bsp-dungeon-generation.md`
- Renderer BSP guide: `src/renderer/AGENTS.md`
- BSP crate: `src/bsp/`
