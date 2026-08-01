# bsp_beta — tmpfs Quota Fix for M3 Package Builds

## Date
2026-08-01

## Change Summary
Fixed the `bsp_beta --m3-generate` bridge-proof failure caused by system
temp-dir (tmpfs) exhaustion: `write .map: Disk quota exceeded (os error 122)`.
The M3 pipeline reserves a process-scoped package root under the system temp
dir (`bsp-beta-m3-<pid>-<nonce>-<seq>`); killed/crashed runs previously left
compiled packages behind, and long-lived agent sessions had accumulated ~13 GB
of stale artifacts on a 16 GB tmpfs, leaving insufficient headroom for the
ericw-tools compile of a new package.

## Files
- `apps/bsp_beta/src/main.rs` — added `sweep_stale_package_roots()`, invoked
  before the first package root is reserved in `build_initial_generated_import`.
  The sweep reaps `bsp-beta-m3-*` roots whose owning PID is no longer alive
  (`/proc/<pid>` liveness check) and whose mtime is older than a 5-minute
  threshold; fresh roots owned by live processes are never touched.

## Validation
- `cargo test -p bsp_beta --bin bsp_beta`: 47 passed (incl. new
  `sweep_stale_roots_tests::sweep_removes_stale_roots_and_keeps_fresh_and_live`).
- `cargo test -p bsp_beta --lib`: 51 passed (incl. the previously-failing
  disk-quota M3 build test).
- Live launch `bsp_beta --m3-generate --preset moderate --seed 42`: initial M3
  package build succeeded; BSP extraction 3386 faces / 22 entities / 20 lights
  / 4 batches with PVS; BSP proof passed; swapchain created; zero ERROR lines.
- Environment: /tmp freed from 81% → 17% usage (13 GB available).

## Specification Impact
none — runtime resilience hardening; no public contract, config, or output
format changed.
