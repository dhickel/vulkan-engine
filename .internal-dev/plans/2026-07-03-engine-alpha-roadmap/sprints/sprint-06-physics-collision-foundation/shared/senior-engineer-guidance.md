# Senior Engineer Guidance

- Treat durable IDs and runtime handles as separate layers. Durable IDs are authored and serialized; Rapier handles are internal lookup keys.
- Keep `physics` boring and Vulkan-free. If a test needs a window, the boundary is wrong.
- Prefer typed descriptor structs over ad hoc maps in public APIs; maps are acceptable only at package metadata boundaries with typed validation layered on top.
- Backward compatibility matters for existing scenes and packages. Add defaults so old files without collision metadata still validate.
- Validator diagnostics should be specific enough for authors: invalid shape kind, invalid dimensions, duplicate collision id, runtime handle identity, unknown collision asset id.
- Do not overbuild character movement. Dogfood ramps/floors are specialized; a written debt record is better than a risky half-migration.
- If generated mesh bounds are named, call them a placeholder/deferred generation path unless actual CPU bound generation and tests are added.
- Event bridge should convert engine-neutral physics records into `engine_events` types. Avoid moving simulation concerns into `engine_events`.
- Existing warning noise should be documented in validation reports; new warnings introduced by this sprint are defects.
- Capture proof is evidence for visible behavior only. For physics and metadata, unit/CLI tests are stronger evidence than screenshots.

