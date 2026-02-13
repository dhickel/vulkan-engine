#!/usr/bin/env bash
set -euo pipefail

REPO="dhickel/vulkan-engine"

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. Run: gh auth login -h github.com"
  exit 1
fi

create_issue() {
  local title="$1"
  shift
  local -a label_args=("$@")
  local body
  body="$(cat)"

  if ! gh issue create --repo "$REPO" --title "$title" "${label_args[@]}" --body "$body"; then
    echo "Label create failed for: $title ; retrying without labels..."
    gh issue create --repo "$REPO" --title "$title" --body "$body"
  fi
}

create_issue \
  "Renderer: async/staged environment preparation to remove frame-time hitches on env switch" \
  --label renderer --label performance --label alpha-backlog <<'EOF'
Problem
Environment switches currently perform synchronous readiness work in the active frame path (`prepare_submission_environment` / `ensure_environment_ready`), which can cause frame-time spikes.

Goal
Introduce staged/async environment preparation and only activate requested environments once descriptors + env maps are fully ready.

Scope
- `src/renderer/src/vulkan/vk_render.rs`
- Environment transition state machine and activation handoff logic.

Acceptance Criteria
- First-time environment activation no longer causes large visible hitch in normal path.
- Fallback behavior remains safe when async preparation is unavailable/fails.
- State transitions remain observable via existing runtime status APIs/logging.

Filed by Codex agent under the direction of @dhickel.
EOF

create_issue \
  "Asset API: bounded parallel deferred loading with deterministic ticket semantics" \
  --label assets --label performance --label api <<'EOF'
Problem
Deferred loading currently serializes to a single in-flight task, which limits throughput for streaming/loading-screen scenarios.

Goal
Add bounded parallelism (e.g. configurable `max_in_flight`, default 2-4) while preserving deterministic ticket terminal state behavior.

Scope
- `src/renderer/src/api/assets.rs`
- Queue scheduler and in-flight tracking.
- Facade status surface to expose queue depth/in-flight count for UI progress.

Acceptance Criteria
- Multiple deferred loads can progress concurrently up to configured bound.
- Ticket status transitions remain deterministic and stable.
- Public API exposes enough state for progress UI.

Filed by Codex agent under the direction of @dhickel.
EOF

create_issue \
  "Vulkan runtime hardening: reduce unwrap/panic exposure in frame/render hot paths" \
  --label vulkan --label reliability --label tech-debt <<'EOF'
Problem
Panic/unwrap density in hot paths makes runtime fault handling brittle and can hard-crash on recoverable failures.

Goal
Prioritize targeted hardening in frame/render/swapchain/environment-prep paths by replacing unwraps with typed error propagation where practical, and improving contextual panic/logging where not recoverable.

Scope
- Start with `src/renderer/src/vulkan/vk_render.rs`
- Focus slices: render loop, swapchain rebuild path, environment preparation path.

Acceptance Criteria
- Reduced unwrap count in targeted high-risk regions.
- Improved error context in propagated failures/logs.
- No behavior regressions in existing example smoke runs.

Filed by Codex agent under the direction of @dhickel.
EOF

create_issue \
  "Sync asset load wait loop: bound transfer pumping and reduce busy-wait CPU churn" \
  --label assets --label performance --label tech-debt <<'EOF'
Problem
Sync asset load waits currently use unbounded transfer pumping (`usize::MAX`) with fixed 1ms sleep, which can cause unnecessary CPU churn for long waits.

Goal
Bound transfer pump work per iteration and evaluate condition-based wake strategy to reduce busy-wait overhead.

Scope
- `src/renderer/src/api/assets.rs`
- Sync wait loop transfer pump cadence and wake policy.

Acceptance Criteria
- Lower CPU churn during long synchronous waits.
- No regressions in correctness or completion ordering.
- Behavior remains predictable for bootstrap workflows.

Filed by Codex agent under the direction of @dhickel.
EOF

echo "Created 4 issues in $REPO"
