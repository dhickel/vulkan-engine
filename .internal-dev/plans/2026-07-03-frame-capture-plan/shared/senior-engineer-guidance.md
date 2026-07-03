# Senior Engineer Guidance

## Core Direction

- Treat capture as renderer-owned infrastructure, not a public Vulkan hook. This preserves facade safety while giving the backend enough access to images, layouts, queues, and allocators.
- Prefer a present-equivalent target for proof. Capturing the draw image is useful diagnostics, but the required proof should show the final scene path after present-copy and UI where applicable.
- Split final present transition before adding capture. The current `ImguiPass` transition makes the capture point ambiguous and encourages illegal `PRESENT_SRC_KHR -> TRANSFER_SRC` churn.
- Headless/offscreen is a core deliverable. Do not finish with windowed-only capture unless the explicit user gate is triggered and approved.

## Gotchas

- Swapchain images currently lack `TRANSFER_SRC`; do not assume present-image readback works until swapchain usage/capabilities are checked.
- Draw images are `R16G16B16A16_SFLOAT`; PNG output requires conversion, not raw byte reinterpretation.
- Present images are likely BGRA/RGBA 8-bit; swizzle must be explicit.
- Row pitch and buffer copy layout are common corruption points. Validation must inspect images, not only file existence.
- `vk_debug::capture_and_save_image_view(...)` is not production quality. Reusing its shape without fixing layout restoration, allocation cleanup, and error handling is a defect.
- Existing frame fence and frame-slot reuse rules are sensitive. Do not reuse or clear frame-local readback resources before GPU completion.
- Headless initialization may require device selection without surface-present support. Keep that change bounded and stop if it turns into a broad renderer rewrite.
- Manual capture must coexist with current debug key handling and ImGui keyboard capture behavior.

## Implementation Reasoning Cues

- If a capture needs the exact user-visible image, capture the present/offscreen-present target after UI and before terminal present transition.
- If a capture needs reliable headless operation, prefer engine-owned images over swapchain images.
- If capture output is blank, first verify frame warmup/start index, target layout at copy time, and whether the selected example has completed async asset/environment loading.
- If validation layers complain, fix image layout/stage/access ordering before touching image conversion code.
- If PNG colors are visibly wrong but nonblank, inspect format swizzle and linear/sRGB assumptions before changing scene lighting.
- If command timeout happens before capture, raise the scheduled frame only when startup logs show the scene is still loading; do not make timeout unbounded.

## Model And Validation Expectations

- Implementation workers: default `gpt-5.5`, high reasoning unless the main thread overrides.
- Phase validators: default `gpt-5.5`, high reasoning.
- Final quality validator for this large suite: default `gpt-5.5`, xhigh reasoning.
- No browser/Playwright validation applies.
