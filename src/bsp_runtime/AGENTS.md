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
2. `validate(token)` — check generation, bridges
3. `commit_with_mount(token, scene, mount)` — publish atomically
4. `rollback()` — idempotent cleanup

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

## Validation

```bash
cargo check -p bsp_runtime
cargo test -p bsp_runtime
cargo check -p renderer --features bsp
```

## Related

- Transaction/Ownership spec: `.internal-dev/specifications/bsp-transaction-ownership.md`
- Acceptance spec: `.internal-dev/specifications/bsp-acceptance.md`
- Renderer BSP guide: `src/renderer/AGENTS.md`
- BSP crate: `src/bsp/`
