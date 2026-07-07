# Senior Engineer Guidance

- Keep the CLI boring: file input, deterministic output, nonzero failures, and no Vulkan startup.
- Make Rust validation shared first, polished CLI UX second. A nice CLI that disagrees with editor/runtime is worse than a rough CLI with correct contracts.
- Treat path normalization as security-sensitive. Reject absolute paths and parent traversal for package-relative asset paths and pack outputs.
- Preserve identity separation. IDs are identity; paths are diagnostics and load locations; runtime handles are transient outputs.
- Avoid broad crate extraction unless necessary. The renderer crate already owns the schemas, and Sprint 02 can expose a narrow validation API without creating a new architecture.
- Do not turn `pack` into release packaging. Folder copy with a report is enough for alpha and easier to validate.
- Prefer fixtures over mocked narratives. Every supported invalid case should have a fixture or in-test temp file that proves the diagnostic.
- Keep Sprint 01 blocked status visible but out of scope. Do not create its changelog or close it.
