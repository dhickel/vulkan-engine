# package_io — Package Trust Boundary

## Scope

`package_io` is the canonical package trust boundary crate shared by the engine
runtime and tooling. It defines confined resource loading, checked budget
reservations, content identity, and a shared resolver that normalizes, validates,
and reads package-relative paths under a trusted root.

## Crate Contract

- **Zero downstream dependencies**: Only `std`. No renderer, Vulkan, BSP, physics,
  app, windowing, async, or filesystem-watcher dependencies.
- **Fail-closed**: Every rejection is an error. Failed reservations and failed
  reads do not mutate cumulative counters.
- **Shared resolver**: Runtime and tooling use the same resolver. Path checks
  are not duplicated elsewhere.
- **No symlink traversal**: Symlinks are detected and rejected at every component.
- **No archive member, device, pipe, or socket access**: All non-regular-file
  resources are rejected.
- **Budget enforcement**: File count, source bytes, decompressed bytes, image
  pixels/dimensions, data-URI decoded bytes, nesting/recursion depth,
  external-model buffers/images, and aggregate package/mount totals.
- **Stable diagnostics**: Every rejection carries a stable machine-readable code.

## Module Map

| module | role |
|--------|------|
| `lib.rs` | Public API: `PackageRoot`, `LogicalResourceId`, `ConfinedResource`, `AuthorizedBytes`, `ContentIdentity`, re-exports |
| `budget.rs` | `BudgetLedger`, `BudgetSnapshot`, `ResourceBudget`, checked reservation |
| `resolver.rs` | `PackageResolver`, path normalization, percent-decode, symlink-aware containment, read/hash |

## Validation

```bash
cargo check -p package_io
cargo test -p package_io
```

## Related

- Workspace root: `Cargo.toml` (workspace members)
- Engine pack: `tools/engine_pack/`
- BSP runtime: `src/bsp_runtime/`
