//! Hard and aggregate budget limits with checked arithmetic helpers.
//!
//! Every allocation path must check limits *before* allocating. Budget failures
//! use `DiagnosticCode::AllocationExceeded` or the specific structural count code.
//!
//! Limits match `bsp-compatibility.md` §3.

use crate::diagnostic::{BspReport, DiagnosticCode};

// ── Structural limits ──

/// Maximum vertices for BSP29 (u16 index space).
pub const MAX_VERTICES_BSP29: u32 = 65_535;
/// Maximum edges for BSP29 (u16 index space).
pub const MAX_EDGES_BSP29: u32 = 65_535;
/// Maximum surfedges for BSP29.
pub const MAX_SURFEDGES_BSP29: u32 = 65_535;
/// Maximum faces for BSP29.
pub const MAX_FACES_BSP29: u32 = 65_535;
/// Maximum markfaces for BSP29.
pub const MAX_MARKFACES_BSP29: u32 = 65_535;
/// Maximum nodes for BSP29.
pub const MAX_NODES_BSP29: u32 = 32_767;
/// Maximum leaves for BSP29.
pub const MAX_LEAVES_BSP29: u32 = 8_191;
/// Maximum clipnodes for BSP29.
pub const MAX_CLIPNODES_BSP29: u32 = 32_767;
/// Maximum models for BSP29 (index 255 is reserved).
pub const MAX_MODELS_BSP29: u32 = 256;

/// BSP2 uses 32-bit indices; practical limit is i32::MAX.
pub const MAX_ELEMENTS_BSP2: u32 = i32::MAX as u32;

// ── Aggregate allocation budgets ──

/// Total lump allocation budget (256 MiB).
pub const TOTAL_LUMP_ALLOCATION: u64 = 256 * 1024 * 1024;
/// Total face vertex allocation budget (16 MiB).
pub const TOTAL_FACE_VERTEX_ALLOCATION: u64 = 16 * 1024 * 1024;
/// Maximum entity count.
pub const MAX_ENTITY_COUNT: u32 = 65_536;
/// Maximum texture/miptex count.
pub const MAX_TEXTURE_COUNT: u32 = 4_096;
/// Maximum light styles per face.
pub const MAX_LIGHT_STYLES_PER_FACE: usize = 4;
/// Reserved light style sentinel (unused slot).
pub const LIGHT_STYLE_SENTINEL: u8 = 255;
/// Maximum supported style identifier.
pub const MAX_STYLE_IDENTIFIER: u8 = 63;
/// Maximum entity string length (1 MiB).
pub const MAX_ENTITY_STRING_LENGTH: u32 = 1_048_576;
/// Maximum WAD entry count per archive.
pub const MAX_WAD_ENTRY_COUNT: u32 = 4_096;
/// Maximum BSPX entries.
pub const MAX_BSPX_ENTRIES: u32 = 64;
/// Maximum depth for tree traversal (guard against cycles via depth limit).
pub const MAX_TREE_DEPTH: u32 = 65_535;

/// Maximum absolute vertex component (32,768 Quake units).
pub const MAX_VERTEX_COMPONENT: f32 = 32_768.0;

// ── Checked helpers ──

/// Check that `count * stride` does not overflow u32 and is within the
/// per-lump element budget for the given profile-limit code.
pub fn checked_count_stride_u32(
    count: u32,
    stride: u32,
    max_count: u32,
    code: DiagnosticCode,
) -> Result<u32, BspReport> {
    if count > max_count {
        return Err(BspReport::fatal(
            code,
            format!("count {} exceeds profile limit {}", count, max_count),
        ));
    }
    count.checked_mul(stride).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::StructuralCorruptOverflow,
            format!("count {} * stride {} overflowed u32", count, stride),
        )
    })
}

/// Check that `count * stride` does not overflow usize.
pub fn checked_count_stride_usize(
    count: u32,
    stride: usize,
    max_count: u32,
    code: DiagnosticCode,
) -> Result<usize, BspReport> {
    if count > max_count {
        return Err(BspReport::fatal(
            code,
            format!("count {} exceeds profile limit {}", count, max_count),
        ));
    }
    let count_usize = count as usize;
    count_usize.checked_mul(stride).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::StructuralCorruptOverflow,
            format!("count {} * stride {} overflowed usize", count, stride),
        )
    })
}

/// Check that cumulative allocation does not exceed the total budget.
pub fn check_cumulative_allocation(
    current: u64,
    additional: u64,
    budget: u64,
) -> Result<u64, BspReport> {
    let total = current.checked_add(additional).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::StructuralCorruptOverflow,
            "cumulative allocation overflowed u64",
        )
    })?;
    if total > budget {
        return Err(BspReport::fatal(
            DiagnosticCode::AllocationExceeded,
            format!("cumulative allocation {} exceeds budget {}", total, budget),
        ));
    }
    Ok(total)
}

/// Validate that a byte range `[offset, offset + size)` is within `total_len`.
pub fn check_byte_range(
    offset: u32,
    size: u32,
    total_len: usize,
    context: &str,
) -> Result<(), BspReport> {
    let end = offset.checked_add(size).ok_or_else(|| {
        BspReport::fatal(
            DiagnosticCode::StructuralCorruptOverflow,
            format!("{}: offset + size overflowed u32", context),
        )
    })?;
    let total = total_len as u64;
    if (end as u64) > total {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!(
                "{}: range [{}, {}) exceeds file length {}",
                context, offset, end, total_len
            ),
        ));
    }
    Ok(())
}

/// Check that count is divisible by expected divisor (for fixed-size element lumps).
pub fn check_count_divisible(count: u32, divisor: u32, lump_name: &str) -> Result<(), BspReport> {
    if divisor == 0 {
        return Ok(());
    }
    if count % divisor != 0 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!(
                "lump {} size {} is not divisible by element size {}",
                lump_name, count, divisor
            ),
        ));
    }
    Ok(())
}

/// Check that `index` is strictly less than `max`.
pub fn check_index(index: u32, max: u32, context: &str) -> Result<(), BspReport> {
    if index >= max {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptIndex,
            format!(
                "{}: index {} out of range (max {})",
                context,
                index,
                max.saturating_sub(1)
            ),
        ));
    }
    Ok(())
}

/// Check that `index` as i32 is valid (non-negative, within range).
pub fn check_index_signed(index: i32, max: u32, context: &str) -> Result<u32, BspReport> {
    if index < 0 {
        // Negative values may be valid in some contexts (e.g., surfedge sign, child sentinels).
        // This function treats them as errors; callers should handle sentinels themselves.
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptIndex,
            format!("{}: negative index {}", context, index),
        ));
    }
    let u = index as u32;
    check_index(u, max, context)?;
    Ok(u)
}

/// Check that a node child index is valid: either a valid node index (positive)
/// or a valid leaf index (negative, mapped). Returns the absolute leaf offset or
/// the positive node index.
pub fn check_node_child(
    child: i32,
    num_nodes: u32,
    num_leaves: u32,
    context: &str,
) -> Result<NodeChild, BspReport> {
    if child >= 0 {
        let node_idx = child as u32;
        if node_idx >= num_nodes && node_idx != 0 {
            // node_idx == 0 is valid only if num_nodes > 0
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "{}: node child {} out of range (max nodes {})",
                    context,
                    child,
                    num_nodes.saturating_sub(1)
                ),
            ));
        }
        if node_idx == 0 && num_nodes > 0 {
            // Node 0 is the root; child pointing to root is a cycle
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptCycle,
                format!("{}: node child {} points to root (cycle)", context, child),
            ));
        }
        Ok(NodeChild::Node(node_idx))
    } else {
        let leaf_idx = (-1 - child) as u32;
        if leaf_idx >= num_leaves {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptIndex,
                format!(
                    "{}: leaf child {} out of range (max leaves {})",
                    context,
                    child,
                    num_leaves.saturating_sub(1)
                ),
            ));
        }
        Ok(NodeChild::Leaf(leaf_idx))
    }
}

/// Result of node child validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeChild {
    Node(u32),
    Leaf(u32),
}

/// Check graph acyclicity via depth-bounded DFS.
/// `children` is a function returning child node indices for a given node.
/// Returns Ok if no cycles found within max_depth.
pub fn check_graph_acyclic(
    node_count: u32,
    children: impl Fn(u32) -> (Option<u32>, Option<u32>),
    max_depth: u32,
    context: &str,
) -> Result<(), BspReport> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Gray,
        Black,
    }

    let nc = node_count as usize;
    let mut colors = vec![Color::White; nc];
    let mut depths = vec![0u32; nc];

    fn dfs(
        node: u32,
        colors: &mut [Color],
        depths: &mut [u32],
        depth: u32,
        max_depth: u32,
        children: &impl Fn(u32) -> (Option<u32>, Option<u32>),
        context: &str,
    ) -> Result<(), BspReport> {
        if depth > max_depth {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptCycle,
                format!(
                    "{}: exceeded max depth {} at node {}",
                    context, max_depth, node
                ),
            ));
        }
        let idx = node as usize;
        colors[idx] = Color::Gray;
        depths[idx] = depth;

        let (left, right) = children(node);
        for &child_opt in &[left, right] {
            if let Some(child) = child_opt {
                if child >= colors.len() as u32 {
                    continue; // Index out of range, let caller validate
                }
                let ci = child as usize;
                match colors[ci] {
                    Color::Gray => {
                        return Err(BspReport::fatal(
                            DiagnosticCode::StructuralCorruptCycle,
                            format!("{}: cycle detected at node {} -> {}", context, node, child),
                        ));
                    }
                    Color::White => {
                        dfs(
                            child,
                            colors,
                            depths,
                            depth + 1,
                            max_depth,
                            children,
                            context,
                        )?;
                    }
                    Color::Black => {} // already processed
                }
            }
        }
        colors[idx] = Color::Black;
        Ok(())
    }

    // Start DFS from root (node 0)
    if node_count > 0 {
        dfs(
            0,
            &mut colors,
            &mut depths,
            0,
            max_depth,
            &children,
            context,
        )?;
    }
    Ok(())
}

/// Check that decompression output size is within budget.
pub fn check_decompression_output(
    output_bytes: u64,
    budget: u64,
    context: &str,
) -> Result<(), BspReport> {
    if output_bytes > budget {
        return Err(BspReport::fatal(
            DiagnosticCode::AllocationExceeded,
            format!(
                "{}: decompression output {} exceeds budget {}",
                context, output_bytes, budget
            ),
        ));
    }
    Ok(())
}

/// Check that a count fits within a u16 (for BSP29 index fields).
pub fn check_u16_count(count: u32, context: &str) -> Result<(), BspReport> {
    if count > u16::MAX as u32 {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("{}: count {} exceeds u16 range", context, count),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_count_stride_ok() {
        assert_eq!(
            checked_count_stride_u32(10, 20, 100, DiagnosticCode::StructuralVertexCount).unwrap(),
            200
        );
    }

    #[test]
    fn checked_count_stride_exceeds_limit() {
        let r = checked_count_stride_u32(200, 20, 100, DiagnosticCode::StructuralVertexCount);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralVertexCount);
    }

    #[test]
    fn checked_count_stride_overflow() {
        let r =
            checked_count_stride_u32(u32::MAX, 2, u32::MAX, DiagnosticCode::StructuralVertexCount);
        assert!(r.is_err());
        assert_eq!(
            r.unwrap_err().code,
            DiagnosticCode::StructuralCorruptOverflow
        );
    }

    #[test]
    fn byte_range_valid() {
        assert!(check_byte_range(0, 100, 200, "test").is_ok());
    }

    #[test]
    fn byte_range_past_end() {
        let r = check_byte_range(150, 100, 200, "test");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptLump);
    }

    #[test]
    fn byte_range_overflow() {
        let r = check_byte_range(u32::MAX, 10, 100, "test");
        assert!(r.is_err());
        assert_eq!(
            r.unwrap_err().code,
            DiagnosticCode::StructuralCorruptOverflow
        );
    }

    #[test]
    fn check_index_valid() {
        assert!(check_index(5, 10, "test").is_ok());
    }

    #[test]
    fn check_index_out_of_range() {
        let r = check_index(10, 10, "test");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptIndex);
    }

    #[test]
    fn node_child_leaf() {
        let r = check_node_child(-1, 10, 5, "test");
        assert_eq!(r.unwrap(), NodeChild::Leaf(0));
    }

    #[test]
    fn node_child_leaf_out_of_range() {
        let r = check_node_child(-10, 10, 5, "test");
        assert!(r.is_err());
    }

    #[test]
    fn node_child_cycle_to_root() {
        let r = check_node_child(0, 10, 5, "test");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptCycle);
    }

    #[test]
    fn acyclic_graph_passes() {
        // Simple tree: 0 -> {1, 2}, 1 -> {leaf, leaf}, 2 -> {leaf, leaf}
        let children = |n: u32| -> (Option<u32>, Option<u32>) {
            match n {
                0 => (Some(1), Some(2)),
                _ => (None, None),
            }
        };
        assert!(check_graph_acyclic(3, children, 100, "test").is_ok());
    }

    #[test]
    fn cyclic_graph_detected() {
        // 0 -> {1, leaf}, 1 -> {0, leaf} — cycle
        let children = |n: u32| -> (Option<u32>, Option<u32>) {
            match n {
                0 => (Some(1), None),
                1 => (Some(0), None),
                _ => (None, None),
            }
        };
        let r = check_graph_acyclic(2, children, 100, "test");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptCycle);
    }

    #[test]
    fn depth_limit_exceeded() {
        // Chain: 0 -> 1 -> 2 -> 3 -> 4
        let children = |n: u32| -> (Option<u32>, Option<u32>) {
            if n < 4 {
                (Some(n + 1), None)
            } else {
                (None, None)
            }
        };
        let r = check_graph_acyclic(5, children, 2, "test");
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::StructuralCorruptCycle);
    }
}
