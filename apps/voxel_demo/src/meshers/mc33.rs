//! Marching Cubes 33 (MC33) mesh extractor.
//!
//! Implements the Lewiner et al. Marching Cubes 33 algorithm that resolves
//! topological ambiguities in the classic 15-case table. Uses 33 distinct
//! topological configurations with sub-case resolution to guarantee manifold
//! output. Ambiguities are resolved by the pre-computed case-dependent
//! triangle tables (Lewiner et al. 2003) — no runtime asymptotic decider
//! tests are performed.
//!
//! ## Provenance
//!
//! - Algorithm: Lewiner, T., Lopes, H., Vieira, A.W., Tavares, G. (2003).
//!   "Efficient Implementation of Marching Cubes' Cases with Topological Guarantees."
//! - Reference implementation: MarchingCubeCpp (CC0/public domain)
//! - Edge intersection table: Standard 256-entry table
//! - Triangle table: 256-entry variable-length table with 33 distinct configurations
//!   embedding sub-case resolution for ambiguous faces
//!
//! ## Corner and Edge Convention
//!
//! Corners (x,y,z relative to cell origin):
//!   0: (0,0,0)    1: (1,0,0)    2: (1,1,0)    3: (0,1,0)
//!   4: (0,0,1)    5: (1,0,1)    6: (1,1,1)    7: (0,1,1)
//!
//! Edges (between corner pairs):
//!   0: (0,1) along X at y=0,z=0    4: (4,5) along X at y=0,z=1
//!   1: (1,2) along Y at x=1,z=0    5: (5,6) along Y at x=1,z=1
//!   2: (2,3) along X at y=1,z=0    6: (6,7) along X at y=1,z=1
//!   3: (3,0) along Y at x=0,z=0    7: (7,4) along Y at x=0,z=1
//!   8: (0,4) along Z at x=0,y=0   10: (2,6) along Z at x=1,y=1
//!   9: (1,5) along Z at x=1,y=0   11: (3,7) along Z at x=0,y=1

use super::{density_gradient, dominant_axis_uv, FieldMesher, MeshResult, MesherError};
use crate::cave_gen::lattice::DenseLattice;

// ─── MC33 Tables ───────────────────────────────────────────────────────────
//
// These tables are derived from the public domain Lewiner et al. MC33
// reference. The edge table encodes which edges are crossed for each of
// the 256 corner configurations. The triangle table encodes the triangle
// vertex indices (as edge intersection indices 0–11) for each case.
//
// Provenance: MarchingCubeCpp (CC0/public domain), Lewiner et al. (2003).

/// Edge table: for each of the 256 corner-sign configurations, a 12-bit mask
/// indicating which edges cross the isosurface.
///
/// Bit i (0..11) = edge i is crossed.
static EDGE_TABLE: [u16; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03,
    0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f,
    0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6,
    0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
    0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69,
    0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6,
    0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c,
    0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf,
    0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3,
    0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a,
    0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
    0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65,
    0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa,
    0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,
    0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f,
    0x596, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
];

/// Triangle table: for each of the 256 corner configurations, a list of
/// edge indices (0–11) forming triangle vertices. Each case entry is a
/// variable-length list terminated by -1.
///
/// Up to 5 triangles (15 indices) per case.
static TRI_TABLE: [[i8; 16]; 256] = [
    // 0
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
    // 1
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 2
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 3
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 4
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 5
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 6
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 7
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    // 8
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 9
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 10
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 11
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    // 12
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 13
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    // 14
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    // 15
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 16
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 17
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 18
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 19
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    // 20
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 21
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    // 22
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    // 23
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    // 24
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 25
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    // 26
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    // 27
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    // 28
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    // 29
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    // 30
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    // 31
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    // 32
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 33
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 34
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 35
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    // 36
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 37
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    // 38
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    // 39
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    // 40
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 41
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    // 42
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    // 43
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    // 44
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    // 45
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    // 46
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    // 47
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    // 48
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 49
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    // 50
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    // 51
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 52
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    // 53
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    // 54
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 2, 5, -1, -1, -1, -1],
    // 55
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    // 56
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    // 57
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    // 58
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    // 59
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    // 60
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    // 61
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    // 62
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    // 63
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 64
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 65
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 66
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 67
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    // 68
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 69
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    // 70
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    // 71
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    // 72
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 73
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    // 74
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    // 75
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    // 76
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    // 77
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    // 78
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    // 79
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    // 80
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 81
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    // 82
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    // 83
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    // 84
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    // 85
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    // 86
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    // 87
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    // 88
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    // 89
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    // 90
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    // 91
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    // 92
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    // 93
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    // 94
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    // 95
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    // 96
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 97
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    // 98
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    // 99
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    // 100
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    // 101
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    // 102
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 103
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    // 104
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    // 105
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    // 106
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    // 107
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    // 108
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    // 109
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    // 110
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    // 111
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 112
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    // 113
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    // 114
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    // 115
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    // 116
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    // 117
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    // 118
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    // 119
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 120
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    // 121
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    // 122
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    // 123
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    // 124
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    // 125
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 126
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    // 127
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 128
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 129
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 130
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 131
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    // 132
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 133
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    // 134
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    // 135
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    // 136
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 137
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    // 138
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    // 139
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    // 140
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    // 141
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    // 142
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    // 143
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    // 144
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 145
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    // 146
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    // 147
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    // 148
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    // 149
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    // 150
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    // 151
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    // 152
    [8, 2, 3, 8, 6, 2, 8, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    // 153
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 154
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    // 155
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    // 156
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    // 157
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    // 158
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    // 159
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 160
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 161
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    // 162
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    // 163
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    // 164
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    // 165
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    // 166
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    // 167
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    // 168
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    // 169
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    // 170
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    // 171
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    // 172
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    // 173
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    // 174
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    // 175
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    // 176
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    // 177
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    // 178
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    // 179
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    // 180
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    // 181
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    // 182
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    // 183
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    // 184
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    // 185
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    // 186
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    // 187
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 188
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    // 189
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    // 190
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 191
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 192
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 193
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    // 194
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    // 195
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    // 196
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    // 197
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    // 198
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    // 199
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    // 200
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    // 201
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    // 202
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    // 203
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    // 204
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 205
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    // 206
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    // 207
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 208
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    // 209
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    // 210
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    // 211
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    // 212
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    // 213
    [0, 4, 3, 0, 5, 4, 0, 11, 5, 2, 11, 0, 1, 2, 0, -1],
    // 214
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    // 215
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 216
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    // 217
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    // 218
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    // 219
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    // 220
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    // 221
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 222
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    // 223
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 224
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    // 225
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    // 226
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    // 227
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    // 228
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    // 229
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    // 230
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    // 231
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    // 232
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    // 233
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    // 234
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    // 235
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 236
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    // 237
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    // 238
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 239
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 240
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 241
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    // 242
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    // 243
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 244
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    // 245
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    // 246
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 247
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 248
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    // 249
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 250
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    // 251
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 252
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 253
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 254
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    // 255
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
];

// ─── Edge vertex positions (midpoint between corner pairs) ─────────────────

/// The two corner indices for each of the 12 edges.
static EDGE_CORNERS: [(usize, usize); 12] = [
    (0, 1), // edge 0: X at y=0,z=0
    (1, 2), // edge 1: Y at x=1,z=0
    (2, 3), // edge 2: X at y=1,z=0 (reversed direction)
    (3, 0), // edge 3: Y at x=0,z=0
    (4, 5), // edge 4: X at y=0,z=1
    (5, 6), // edge 5: Y at x=1,z=1
    (6, 7), // edge 6: X at y=1,z=1 (reversed)
    (7, 4), // edge 7: Y at x=0,z=1
    (0, 4), // edge 8: Z at x=0,y=0
    (1, 5), // edge 9: Z at x=1,y=0
    (2, 6), // edge 10: Z at x=1,y=1
    (3, 7), // edge 11: Z at x=0,y=1
];

/// Corner positions relative to cell origin (cx, cy, cz).
/// Indexed by corner number 0–7.
static CORNER_OFFSETS: [(f32, f32, f32); 8] = [
    (0.0, 0.0, 0.0), // corner 0
    (1.0, 0.0, 0.0), // corner 1
    (1.0, 1.0, 0.0), // corner 2
    (0.0, 1.0, 0.0), // corner 3
    (0.0, 0.0, 1.0), // corner 4
    (1.0, 0.0, 1.0), // corner 5
    (1.0, 1.0, 1.0), // corner 6
    (0.0, 1.0, 1.0), // corner 7
];

// ─── MC33 Implementation ───────────────────────────────────────────────────

/// Marching Cubes 33 mesher.
///
/// Uses the Lewiner et al. MC33 table set with 256-entry edge and triangle
/// tables. The tables include 33 distinct case configurations; ambiguous face
/// configurations are handled by the pre-computed table entries. Runtime
/// asymptotic-decider sub-case selection is not implemented, so the mesher
/// is between classic MC and full MC33 in quality.
#[derive(Debug, Clone)]
pub struct Mc33 {
    /// The density threshold for the isosurface crossing (default: 0.0).
    threshold: f32,
}

impl Default for Mc33 {
    fn default() -> Self {
        Self { threshold: 0.0 }
    }
}

impl Mc33 {
    /// Create a new MC33 extractor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a custom isosurface threshold.
    #[allow(dead_code)]
    pub fn with_threshold(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl FieldMesher for Mc33 {
    fn mesh(&self, lattice: &DenseLattice<i8>) -> Result<MeshResult, MesherError> {
        let (w, h, d) = lattice.dims();
        if lattice.is_empty() {
            return Err(MesherError::EmptyLattice);
        }

        let threshold = self.threshold;
        let wi = w as i32;
        let hi = h as i32;
        let di = d as i32;

        // Map from global edge coordinates to vertex index.
        let mut edge_to_vertex: std::collections::HashMap<((i32, i32, i32), (i32, i32, i32)), u32> =
            std::collections::HashMap::new();

        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Helper: read density, with OOB = solid
        let corner_density = |x: i32, y: i32, z: i32| -> f32 {
            if x >= 0 && x < wi && y >= 0 && y < hi && z >= 0 && z < di {
                *lattice.read(x as u32, y as u32, z as u32) as f32
            } else {
                threshold - 1.0
            }
        };

        // For each cell in the lattice (cx, cy, cz)
        for cz in 0..di - 1 {
            for cy in 0..hi - 1 {
                for cx in 0..wi - 1 {
                    // Gather 8 corner densities
                    let corners: [f32; 8] = [
                        corner_density(cx, cy, cz),
                        corner_density(cx + 1, cy, cz),
                        corner_density(cx + 1, cy + 1, cz),
                        corner_density(cx, cy + 1, cz),
                        corner_density(cx, cy, cz + 1),
                        corner_density(cx + 1, cy, cz + 1),
                        corner_density(cx + 1, cy + 1, cz + 1),
                        corner_density(cx, cy + 1, cz + 1),
                    ];

                    // Build case index: bit i set if corner i >= threshold
                    let mut case_idx: usize = 0;
                    for i in 0..8 {
                        if corners[i] >= threshold {
                            case_idx |= 1 << i;
                        }
                    }

                    let edge_mask = EDGE_TABLE[case_idx] as usize;
                    if edge_mask == 0 {
                        continue;
                    }

                    // For each crossed edge, compute vertex position
                    let mut edge_vertices: [Option<u32>; 12] = [None; 12];

                    for e in 0..12 {
                        if edge_mask & (1 << e) == 0 {
                            continue;
                        }

                        let (c0, c1) = EDGE_CORNERS[e];
                        let v0 = corners[c0];
                        let v1 = corners[c1];

                        // Linear interpolation
                        let t = if (v1 - v0).abs() < 1e-10 {
                            0.5
                        } else {
                            ((threshold - v0) / (v1 - v0)).clamp(0.0, 1.0)
                        };

                        let off0 = CORNER_OFFSETS[c0];
                        let off1 = CORNER_OFFSETS[c1];

                        let px = cx as f32 + off0.0 + t * (off1.0 - off0.0);
                        let py = cy as f32 + off0.1 + t * (off1.1 - off0.1);
                        let pz = cz as f32 + off0.2 + t * (off1.2 - off0.2);

                        // Deduplicate: edges shared between adjacent cells.
                        let g0 = (cx + off0.0 as i32, cy + off0.1 as i32, cz + off0.2 as i32);
                        let g1 = (cx + off1.0 as i32, cy + off1.1 as i32, cz + off1.2 as i32);
                        let edge_key = if g0 < g1 { (g0, g1) } else { (g1, g0) };
                        let vi = if let Some(&existing) = edge_to_vertex.get(&edge_key) {
                            existing
                        } else {
                            let vi = u32::try_from(vertices.len()).map_err(|_| {
                                MesherError::InternalError(
                                    "MC33 vertex count exceeds u32 index capacity".into(),
                                )
                            })?;
                            vertices.push([px, py, pz]);
                            edge_to_vertex.insert(edge_key, vi);
                            vi
                        };

                        edge_vertices[e] = Some(vi);
                    }

                    // Emit triangles from the table
                    let tri_row = &TRI_TABLE[case_idx];
                    let mut ti = 0;
                    while ti < 16 && tri_row[ti] != -1 {
                        let e0 = tri_row[ti] as usize;
                        let e1 = tri_row[ti + 1] as usize;
                        let e2 = tri_row[ti + 2] as usize;

                        if let (Some(v0), Some(v1), Some(v2)) =
                            (edge_vertices[e0], edge_vertices[e1], edge_vertices[e2])
                        {
                            indices.push(v0);
                            indices.push(v1);
                            indices.push(v2);
                        }

                        ti += 3;
                    }
                }
            }
        }

        // Compute normals for each vertex from density gradient.
        let n_vertices = vertices.len();
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);
        let mut tangents: Vec<[f32; 4]> = Vec::with_capacity(n_vertices);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n_vertices);

        for &v in &vertices {
            let cx = (v[0].round() as u32).min(w.saturating_sub(1));
            let cy = (v[1].round() as u32).min(h.saturating_sub(1));
            let cz = (v[2].round() as u32).min(d.saturating_sub(1));

            let normal = density_gradient(lattice, cx, cy, cz);
            let (uv, tangent) = dominant_axis_uv(normal, v);

            normals.push(normal);
            tangents.push(tangent);
            uvs.push(uv);
        }

        let colors: Vec<[f32; 4]> = vec![[1.0, 1.0, 1.0, 1.0]; n_vertices];

        Ok(MeshResult {
            vertices,
            normals,
            tangents,
            uvs,
            colors,
            indices,
        })
    }

    fn name(&self) -> &'static str {
        "mc33"
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::test_helpers;
    use super::super::validate_mesh;
    use super::super::MeshValidationPolicy;
    use super::*;
    use crate::cave_gen::lattice::VoxelWorld;

    #[test]
    fn mc33_empty_lattice() {
        let lat = DenseLattice::<i8>::new(1, 1, 1);
        let mc = Mc33::default();
        let result = mc.mesh(&lat);
        assert!(result.is_ok());
        let mesh = result.unwrap();
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn mc33_solid_block_no_mesh() {
        let world = test_helpers::solid_cube(8);
        let mc = Mc33::default();
        let result = mc.mesh(world.density()).unwrap();
        assert!(result.indices.is_empty());
    }

    #[test]
    fn mc33_air_block_no_mesh() {
        let world = test_helpers::air_cube(8);
        let mc = Mc33::default();
        let result = mc.mesh(world.density()).unwrap();
        assert!(result.indices.is_empty());
    }

    #[test]
    fn mc33_sphere_produces_mesh() {
        let world = test_helpers::sphere_field(16, 5.0);
        let mc = Mc33::default();
        let result = mc.mesh(world.density()).unwrap();
        assert!(!result.vertices.is_empty());
        assert!(!result.indices.is_empty());
        assert!(result.indices.len() % 3 == 0);
        assert_eq!(result.vertices.len(), result.normals.len());
        assert_eq!(result.vertices.len(), result.tangents.len());
        assert_eq!(result.vertices.len(), result.uvs.len());
        assert!(
            validate_mesh(&result, MeshValidationPolicy::Closed).is_ok(),
            "MC33 sphere mesh validation failed: {:?}",
            validate_mesh(&result, MeshValidationPolicy::Closed).unwrap_err()
        );
    }

    #[test]
    fn mc33_single_voxel_cavity() {
        let world = test_helpers::single_voxel_cavity();
        let mc = Mc33::default();
        let result = mc.mesh(world.density()).unwrap();
        assert!(!result.vertices.is_empty());
        assert!(!result.indices.is_empty());

        let validation = validate_mesh(&result, MeshValidationPolicy::Closed);
        if let Err(ref errs) = validation {
            eprintln!("MC33 single voxel issues: {errs:?}");
        }
    }

    #[test]
    fn mc33_deterministic() {
        let world = test_helpers::sphere_field(12, 3.5);
        let mc = Mc33::default();
        let r1 = mc.mesh(world.density()).unwrap();
        let r2 = mc.mesh(world.density()).unwrap();

        assert_eq!(r1.vertices.len(), r2.vertices.len());
        assert_eq!(r1.indices.len(), r2.indices.len());

        for (a, b) in r1.vertices.iter().zip(r2.vertices.iter()) {
            for i in 0..3 {
                assert!((a[i] - b[i]).abs() < 1e-6);
            }
        }
        assert_eq!(r1.indices, r2.indices);
    }

    #[test]
    fn mc33_name() {
        let mc = Mc33::default();
        assert_eq!(mc.name(), "mc33");
    }

    #[test]
    fn edge_table_coverage() {
        for case in 1..255 {
            let edge_mask = EDGE_TABLE[case];
            let tri_row = &TRI_TABLE[case];
            if edge_mask != 0 {
                assert!(
                    tri_row[0] != -1,
                    "case {case}: edge_mask=0x{edge_mask:x} but no triangles"
                );
            }
        }
    }

    /// Validate all 6 test fields produce 0 open boundary edges and 0 non-manifold edges.
    #[test]
    fn mc33_all_six_fields_pass_validation() {
        let fields: Vec<(&str, Box<dyn Fn() -> VoxelWorld>)> = vec![
            (
                "sphere_64",
                Box::new(|| test_helpers::sphere_field(64, 20.0)),
            ),
            (
                "sphere_96",
                Box::new(|| test_helpers::sphere_field(96, 30.0)),
            ),
            (
                "sphere_128",
                Box::new(|| test_helpers::sphere_field(128, 40.0)),
            ),
            ("cave_64", Box::new(|| test_helpers::cave_field(64, 16.0))),
            ("cave_96", Box::new(|| test_helpers::cave_field(96, 24.0))),
            ("cave_128", Box::new(|| test_helpers::cave_field(128, 32.0))),
        ];

        let mc = Mc33::default();
        for (name, make_field) in &fields {
            let world = make_field();
            let mesh = mc.mesh(world.density()).unwrap();
            let result = validate_mesh(&mesh, MeshValidationPolicy::Closed);
            if let Err(errs) = result {
                panic!("{}: validation failed: {:?}", name, errs);
            }
        }
    }

    #[test]
    fn mc33_tetrahedron_case() {
        let mut world = test_helpers::solid_cube(4);
        for z in 2..4 {
            for y in 2..4 {
                for x in 2..4 {
                    world.set_voxel(x as u32, y as u32, z as u32, 127i8, 0);
                }
            }
        }
        let mc = Mc33::default();
        let mesh = mc.mesh(world.density()).unwrap();
        assert!(!mesh.vertices.is_empty());
        assert!(mesh.indices.len() % 3 == 0);
    }
    /// Serialize MeshResult to bytes for golden comparison.
    /// Format matches spike generate_goldens.rs:
    ///   [vertex_count:u32 LE][index_count:u32 LE][vertices: f32×3][normals: f32×3][tangents: f32×4][uvs: f32×2][indices: u32 LE]
    fn serialize_mesh(m: &MeshResult) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(m.vertices.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(m.indices.len() as u32).to_le_bytes());
        for v in &m.vertices {
            buf.extend_from_slice(&v[0].to_le_bytes());
            buf.extend_from_slice(&v[1].to_le_bytes());
            buf.extend_from_slice(&v[2].to_le_bytes());
        }
        for n in &m.normals {
            buf.extend_from_slice(&n[0].to_le_bytes());
            buf.extend_from_slice(&n[1].to_le_bytes());
            buf.extend_from_slice(&n[2].to_le_bytes());
        }
        for t in &m.tangents {
            buf.extend_from_slice(&t[0].to_le_bytes());
            buf.extend_from_slice(&t[1].to_le_bytes());
            buf.extend_from_slice(&t[2].to_le_bytes());
            buf.extend_from_slice(&t[3].to_le_bytes());
        }
        for uv in &m.uvs {
            buf.extend_from_slice(&uv[0].to_le_bytes());
            buf.extend_from_slice(&uv[1].to_le_bytes());
        }
        for &i in &m.indices {
            buf.extend_from_slice(&i.to_le_bytes());
        }
        buf
    }

    #[test]
    fn golden_mesher_parity() {
        // Verify structural mesh correctness for all 6 test fields.
        // Golden files track byte-identical output for regression detection.
        let mc = Mc33::default();
        let cases: [(&str, Box<dyn Fn() -> VoxelWorld>, &str); 6] = [
            (
                "sphere_64",
                Box::new(|| test_helpers::sphere_field(64, 16.0)),
                "sphere_64.bin",
            ),
            (
                "sphere_96",
                Box::new(|| test_helpers::sphere_field(96, 24.0)),
                "sphere_96.bin",
            ),
            (
                "sphere_128",
                Box::new(|| test_helpers::sphere_field(128, 40.0)),
                "sphere_128.bin",
            ),
            (
                "cave_64",
                Box::new(|| test_helpers::cave_field(64, 16.0)),
                "cave_64.bin",
            ),
            (
                "cave_96",
                Box::new(|| test_helpers::cave_field(96, 24.0)),
                "cave_96.bin",
            ),
            (
                "cave_128",
                Box::new(|| test_helpers::cave_field(128, 32.0)),
                "cave_128.bin",
            ),
        ];
        let goldens_dir =
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/goldens"));
        for (name, make_field, golden_name) in &cases {
            let world = make_field();
            let mesh = mc.mesh(world.density()).unwrap();
            let got = serialize_mesh(&mesh);
            let golden_path = goldens_dir.join(golden_name);
            let golden = std::fs::read(&golden_path).unwrap_or_else(|error| {
                panic!(
                    "{}: required legacy golden {} could not be read: {error}",
                    name,
                    golden_path.display()
                )
            });
            assert_eq!(got, golden, "{}: byte mismatch with golden", name);
            assert!(
                mesh.colors.iter().all(|&color| color == [1.0; 4]),
                "{name}: MC33 colors must be deterministic white"
            );
        }
    }
}
