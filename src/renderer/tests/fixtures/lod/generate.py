#!/usr/bin/env python3
"""Generate LOD fixture glTF files: high (sphere-like), medium (cube), low (tetrahedron).
No external dependencies beyond Python stdlib."""
import struct
import json
import os
import math

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def vec3_add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def vec3_sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def vec3_scale(v, s):
    return (v[0]*s, v[1]*s, v[2]*s)

def vec3_cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def vec3_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vec3_len(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def vec3_normalize(v):
    l = vec3_len(v)
    if l < 1e-10:
        return (0.0, 0.0, 0.0)
    return (v[0]/l, v[1]/l, v[2]/l)

def write_gltf(name, positions, normals, indices):
    pos_bytes = b''
    for p in positions:
        pos_bytes += struct.pack('<fff', *p)

    nrm_bytes = b''
    for n in normals:
        nrm_bytes += struct.pack('<fff', *n)

    idx_bytes = b''
    for i in indices:
        idx_bytes += struct.pack('<H', i)

    # Pad to 4-byte alignment
    while len(pos_bytes) % 4 != 0:
        pos_bytes += b'\x00'

    buffer_data = pos_bytes + nrm_bytes + idx_bytes

    bin_path = os.path.join(OUT_DIR, f'{name}.bin')
    with open(bin_path, 'wb') as f:
        f.write(buffer_data)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1
                },
                "indices": 2,
                "mode": 4
            }]
        }],
        "buffers": [{"uri": f"{name}.bin", "byteLength": len(buffer_data)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(nrm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes) + len(nrm_bytes), "byteLength": len(idx_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "max": [max(p[i] for p in positions) for i in range(3)],
             "min": [min(p[i] for p in positions) for i in range(3)]},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ]
    }

    gltf_path = os.path.join(OUT_DIR, f'{name}.gltf')
    with open(gltf_path, 'w') as f:
        json.dump(gltf, f, indent=2)
    print(f"Wrote {name}.gltf ({len(positions)} verts, {len(indices)//3} tris)")

def compute_flat_normals(positions, indices):
    """Compute per-face flat normals, duplicating vertices."""
    new_pos = []
    new_nrm = []
    new_idx = []
    for fi in range(0, len(indices), 3):
        i0, i1, i2 = indices[fi], indices[fi+1], indices[fi+2]
        p0, p1, p2 = positions[i0], positions[i1], positions[i2]
        n = vec3_normalize(vec3_cross(vec3_sub(p1, p0), vec3_sub(p2, p0)))
        base = len(new_pos)
        new_pos.extend([p0, p1, p2])
        new_nrm.extend([n, n, n])
        new_idx.extend([base, base+1, base+2])
    return new_pos, new_nrm, new_idx

# ---- Tetrahedron (low) ----
# 4 vertices, 4 faces
tetra_positions = [
    (0.0, 0.5, 0.0),
    (0.0, -0.1667, 0.4714),
    (-0.4082, -0.1667, -0.2357),
    (0.4082, -0.1667, -0.2357),
]
tetra_indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
tp, tn, ti = compute_flat_normals(tetra_positions, tetra_indices)
write_gltf('low', tp, tn, ti)

# ---- Cube (medium) ----
s = 0.5
cube_faces = [
    ([( s, -s, -s), ( s, -s,  s), ( s,  s,  s), ( s,  s, -s)], (1,0,0)),
    ([(-s, -s,  s), (-s, -s, -s), (-s,  s, -s), (-s,  s,  s)], (-1,0,0)),
    ([(-s,  s, -s), ( s,  s, -s), ( s,  s,  s), (-s,  s,  s)], (0,1,0)),
    ([(-s, -s,  s), ( s, -s,  s), ( s, -s, -s), (-s, -s, -s)], (0,-1,0)),
    ([(-s, -s,  s), (-s,  s,  s), ( s,  s,  s), ( s, -s,  s)], (0,0,1)),
    ([( s, -s, -s), ( s,  s, -s), (-s,  s, -s), (-s, -s, -s)], (0,0,-1)),
]
cube_pos = []
cube_nrm = []
cube_idx = []
for face_verts, n in cube_faces:
    base = len(cube_pos)
    cube_pos.extend(face_verts)
    cube_nrm.extend([n, n, n, n])
    cube_idx.extend([base, base+1, base+2, base, base+2, base+3])
write_gltf('medium', cube_pos, cube_nrm, cube_idx)

# ---- High (subdivided octahedron, 2 levels → 128 faces) ----
# Octahedron vertices
v = [
    (0.0, 0.5, 0.0), (0.0, -0.5, 0.0),
    (0.5, 0.0, 0.0), (-0.5, 0.0, 0.0),
    (0.0, 0.0, 0.5), (0.0, 0.0, -0.5),
]
faces = [
    (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
    (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),
]

def midpoint(a, b):
    return vec3_normalize(vec3_scale(vec3_add(v[a], v[b]), 0.5))

mid_cache = {}
def get_mid(a, b):
    key = (min(a,b), max(a,b))
    if key not in mid_cache:
        mid_cache[key] = len(v)
        v.append(midpoint(a, b))
    return mid_cache[key]

for _ in range(2):
    new_faces = []
    for a, b, c in faces:
        ab = get_mid(a, b)
        bc = get_mid(b, c)
        ca = get_mid(c, a)
        new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
    faces = new_faces

high_idx = []
for a, b, c in faces:
    high_idx.extend([a, b, c])

hp, hn, hi = compute_flat_normals(v, high_idx)
write_gltf('high', hp, hn, hi)

print("Done! All LOD fixture files generated.")
