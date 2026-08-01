//! Focused tests: retirement handoff API (Phase A — EnhancedV3 live-generation).
//!
//! Every replacement, unload, rollback, stale upload, and teardown that
//! produces a DetachedBspMount must deposit it into the coordinator's
//! pending queue. The coordinator never silently drops a detached receipt.
//! The caller drains and submits to Renderer::retire_bsp_mount.

use bsp_runtime::coordinator::BspCoordinator;

use renderer::api::bsp::PreparedBspMount;
use renderer::api::Scene;

// ── shared helpers ────────────────────────────────────────────────────

fn minimal_bsp_bytes() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&29u32.to_le_bytes());
    let mut current_offset: u32 = 124;
    let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
    let entity_offset = current_offset;
    let entity_size = entity_bytes.len() as u32;
    current_offset += entity_size;
    let plane_offset = current_offset;
    let plane_size = 20u32;
    current_offset += plane_size;
    let lumps: [(u32, u32); 15] = [
        (entity_offset, entity_size),
        (plane_offset, plane_size),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ];
    for (off, sz) in &lumps {
        data.extend_from_slice(&off.to_le_bytes());
        data.extend_from_slice(&sz.to_le_bytes());
    }
    data.extend_from_slice(entity_bytes);
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());
    data.extend_from_slice(&0i32.to_le_bytes());
    data
}

fn empty_mount() -> PreparedBspMount {
    PreparedBspMount::new()
}

fn prepare_and_commit(coordinator: &mut BspCoordinator, scene: &mut Scene, identity: &str) {
    let bsp_bytes = minimal_bsp_bytes();
    let prepare = coordinator
        .prepare(&bsp_bytes, None, identity)
        .expect("prepare should succeed");
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .expect("set mount ready");
    coordinator.validate(prepare.token).expect("validate");
    coordinator.commit(prepare.token, scene).expect("commit");
}

// ── tests ─────────────────────────────────────────────────────────────

#[test]
fn replacement_queues_exactly_one_receipt() {
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // First mount — no prior active, so no retirement.
    prepare_and_commit(&mut coordinator, &mut scene, "maps/a");
    assert_eq!(coordinator.pending_retirement_count(), 0);

    // Second mount replaces first — exactly one detached receipt queued.
    prepare_and_commit(&mut coordinator, &mut scene, "maps/b");
    assert_eq!(coordinator.pending_retirement_count(), 1);
    // Diagnostic counter increments on every detachment.
    assert!(coordinator.retired_mount_count() >= 1);
}

#[test]
fn unload_queues_one_receipt() {
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Commit one mount (no prior — no retirement).
    prepare_and_commit(&mut coordinator, &mut scene, "maps/a");
    assert_eq!(coordinator.pending_retirement_count(), 0);

    // Unload the active mount — exactly one detached receipt queued.
    coordinator.unload(&mut scene).expect("unload");
    assert_eq!(coordinator.pending_retirement_count(), 1);
    assert!(coordinator.retired_mount_count() >= 1);
}

#[test]
fn stale_unpublished_rollback_queues_one_receipt() {
    let mut coordinator = BspCoordinator::new();
    let _scene = Scene::new(); // no commit needed — just prepare + rollback

    let bsp_bytes = minimal_bsp_bytes();
    let prepare = coordinator
        .prepare(&bsp_bytes, None, "maps/a")
        .expect("prepare");

    // Attach a renderer mount (staged, not committed).
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .expect("set mount ready");

    assert_eq!(coordinator.pending_retirement_count(), 0);

    // Rollback detaches the unpublished mount.
    coordinator.rollback().expect("rollback");
    assert_eq!(coordinator.pending_retirement_count(), 1);
}

#[test]
fn drain_moves_receipts_out() {
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // First mount, then replace — replacement queues one.
    prepare_and_commit(&mut coordinator, &mut scene, "maps/a");
    prepare_and_commit(&mut coordinator, &mut scene, "maps/b");

    assert_eq!(coordinator.pending_retirement_count(), 1);

    // Drain takes them all.
    let receipts = coordinator.drain_pending_retirements();
    assert_eq!(receipts.len(), 1);
    assert_eq!(coordinator.pending_retirement_count(), 0);

    // Second drain returns empty.
    let receipts2 = coordinator.drain_pending_retirements();
    assert!(receipts2.is_empty());
}

#[test]
fn multiple_retirements_accumulate_and_drain_together() {
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Mount A → replace with B → replace with C → unload C
    // Expected: A retired during B commit, B retired during C commit, C retired during unload = 3
    prepare_and_commit(&mut coordinator, &mut scene, "maps/a");
    assert_eq!(coordinator.pending_retirement_count(), 0);

    prepare_and_commit(&mut coordinator, &mut scene, "maps/b");
    assert_eq!(coordinator.pending_retirement_count(), 1); // A retired

    prepare_and_commit(&mut coordinator, &mut scene, "maps/c");
    assert_eq!(coordinator.pending_retirement_count(), 2); // A + B

    coordinator.unload(&mut scene).expect("unload");
    assert_eq!(coordinator.pending_retirement_count(), 3); // A + B + C

    let receipts = coordinator.drain_pending_retirements();
    assert_eq!(receipts.len(), 3);
    assert_eq!(coordinator.pending_retirement_count(), 0);
}

#[test]
fn requeue_preserves_receipt_for_next_drain() {
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Replace to get one detached receipt.
    prepare_and_commit(&mut coordinator, &mut scene, "maps/a");
    prepare_and_commit(&mut coordinator, &mut scene, "maps/b");

    let mut receipts = coordinator.drain_pending_retirements();
    assert_eq!(receipts.len(), 1);
    assert_eq!(coordinator.pending_retirement_count(), 0);

    // Requeue the receipt (simulating a retry after rejection).
    let detached = receipts.pop().unwrap();
    coordinator.requeue_retirement(detached);
    assert_eq!(coordinator.pending_retirement_count(), 1);

    // Drain again — the same receipt is back.
    let receipts2 = coordinator.drain_pending_retirements();
    assert_eq!(receipts2.len(), 1);
}

#[test]
fn pending_count_reflects_queue_depth() {
    let mut coordinator = BspCoordinator::new();

    // No retirements initially.
    assert_eq!(coordinator.pending_retirement_count(), 0);
    assert_eq!(coordinator.retired_mount_count(), 0);
    assert_eq!(coordinator.retirement_diagnostics(), 0);

    // Prepare and rollback adds one.
    let bsp_bytes = minimal_bsp_bytes();
    let prepare = coordinator
        .prepare(&bsp_bytes, None, "maps/a")
        .expect("prepare");
    coordinator
        .set_renderer_mount_ready(prepare.token, empty_mount())
        .expect("set mount ready");
    coordinator.rollback().expect("rollback");

    assert_eq!(coordinator.pending_retirement_count(), 1);
    assert_eq!(coordinator.retired_mount_count(), 1);

    // Drain clears pending but not the diagnostic counter.
    let _ = coordinator.drain_pending_retirements();
    assert_eq!(coordinator.pending_retirement_count(), 0);
    assert_eq!(coordinator.retired_mount_count(), 1); // cumulative, not decremented
    assert_eq!(coordinator.retirement_diagnostics(), 1);
}

#[test]
fn replacement_preserves_bridge_and_source_link_after_handoff() {
    // Verify that the handoff queue does not disturb bridge ordering,
    // source-link updates, or old-world behavior during replacement.
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    // Mount A
    prepare_and_commit(&mut coordinator, &mut scene, "maps/a");
    assert!(coordinator.is_active());
    assert!(scene.has_bsp_mount());
    assert!(coordinator.source_link().is_some());

    // Mount B replaces A
    prepare_and_commit(&mut coordinator, &mut scene, "maps/b");
    assert!(coordinator.is_active());
    assert!(scene.has_bsp_mount());
    // Source link updated to B
    assert!(coordinator
        .source_link()
        .unwrap()
        .asset_id
        .contains("maps/b"));

    // A's mount is in the pending queue.
    assert_eq!(coordinator.pending_retirement_count(), 1);
}

#[test]
fn old_world_preserved_on_prepare_failure_with_handoff() {
    // When a replacement prepare fails, the old mount and its scene
    // state are unchanged, and no new retirement is queued.
    let mut coordinator = BspCoordinator::new();
    let mut scene = Scene::new();

    prepare_and_commit(&mut coordinator, &mut scene, "maps/a");
    assert!(coordinator.is_active());
    let ret_before = coordinator.retired_mount_count();
    let pending_before = coordinator.pending_retirement_count();

    // Try a prepare that fails (invalid BSP bytes) — old world survives.
    let result = coordinator.prepare(&[0u8; 4], None, "maps/bad");
    assert!(result.is_err());
    assert!(coordinator.is_active());
    assert!(scene.has_bsp_mount());
    assert!(coordinator.source_link().is_some());
    assert_eq!(coordinator.retired_mount_count(), ret_before);
    assert_eq!(coordinator.pending_retirement_count(), pending_before);
}
