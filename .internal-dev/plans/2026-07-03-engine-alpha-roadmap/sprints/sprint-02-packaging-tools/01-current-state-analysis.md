# Current State Analysis

## Verified Inputs

- Root workspace currently lists renderer, input, audio, physics, scripting, editor, and dogfood crates, but no `tools/engine_pack` member.
- Package schema and durable record parsing live in `src/renderer/src/data/asset_registry.rs`.
- Project schema also lives in `asset_registry.rs`, but `Project::load` currently parses TOML without the same explicit validation level used for packages.
- Scene schema and graph validation live in private serialized scene structures in `src/renderer/src/api/scene.rs`.
- Editor sample project exists at `apps/editor/sample_project/engine.project.toml`.
- Editor package loading uses `Project::load`, then calls `load_package_manifest_with_expected_id` for enabled packages.

## Code Facts To Recheck Before Editing

- `src/renderer/src/data/asset_registry.rs:13-14` defines package and project format versions.
- `src/renderer/src/data/asset_registry.rs:90-149` defines `AssetKind` and its string serialization.
- `src/renderer/src/data/asset_registry.rs:151-177` defines package manifest and package asset record fields.
- `src/renderer/src/data/asset_registry.rs:379-441` parses and validates package manifests.
- `src/renderer/src/data/asset_registry.rs:472-520` rejects empty, absolute, and escaping asset paths, and rejects asset IDs containing slash characters.
- `src/renderer/src/data/asset_registry.rs:535-634` defines project schema and load/save behavior.
- `src/renderer/src/api/scene.rs:1053-1132` defines private serialized scene structures.
- `src/renderer/src/api/scene.rs:1306-1378` validates scene format version, duplicate node IDs, parent graph, roots, and asset references.
- `apps/editor/src/main.rs:327-375` loads the project and enabled package manifests.

## Architecture Fit

- The canonical packaging path should be Rust because runtime/editor schemas are Rust types.
- Shared validation should initially remain renderer-owned unless a dependency cycle forces a separate crate. A split into a support crate is allowed only if it reduces coupling and keeps the editor/runtime/CLI using the same code.
- CLI commands should operate on files and folders without Vulkan initialization. Validation must not require creating `Renderer` unless a later phase intentionally performs runtime/capture proof.
- Scene validation should avoid real asset loading when the command is schema/reference validation. It can parse scene JSON and resolve durable asset IDs against loaded package metadata.

## Gaps

- No CLI workspace member exists for package authoring.
- Project validation lacks clear error types and field/path checks.
- Scene validation is private and tied to loader-oriented deserialization, making CLI validation likely to duplicate logic unless exposed/refactored carefully.
- Package validation checks manifest shape but may need file-existence checks, dependency/reference checks, and richer diagnostics for CLI use.
- Existing `AssetKind` does not include future Track A kinds such as audio, scripts, or collision descriptors. Sprint 02 can either add alpha metadata/kinds with tests or record a migration gap if too large.

## Risks

- Overcoupling CLI to renderer internals can make future package tools require Vulkan-era dependencies or broad renderer build costs.
- Duplicated validation can produce editor/CLI disagreement, which is a Sprint 02 blocking bug.
- Generators can accidentally use paths as identity. Generated IDs must be stable durable strings, not path strings.
- `pack` can overreach into binary archives. Folder copy output is enough for alpha.
- Existing sample project may need normalization. If so, changes must be narrow and tested.

## Validation Blind Spots

- Compile checks do not prove visual placement or rendered asset readiness.
- Scene JSON validation without package metadata can miss unknown durable asset IDs.
- Package manifest parsing without file-existence checks can accept missing files that fail later in the editor.
- CLI stdout/stderr wording can drift unless tests assert stable diagnostic content.
