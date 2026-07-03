# Asset Loading

This legacy chapter is retained as a stable link target. The current asset API
reference lives in [04-assets-sync-deferred-and-handles.md](04-assets-sync-deferred-and-handles.md).

Use that chapter for the implemented contracts around:

- synchronous model, texture, material, and environment loading;
- deferred `LoadTicket` workflows with `request_model_load` and `poll_model_load`;
- durable package IDs, asset records, and editor asset browser listing;
- model, prefab, wall chunk, texture, and environment lookup by package asset ID;
- slot/generation handle semantics and stale handle behavior.

Scene integration for loaded fragments is documented in
[03-scene-graph-and-fragment-workflows.md](03-scene-graph-and-fragment-workflows.md).
