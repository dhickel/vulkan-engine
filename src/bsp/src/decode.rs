//! Little-endian integer and float decoders from byte slices.
//!
//! All access is bounds-checked. No transmute, pointer casts, unchecked
//! indexing, unwrap on input data, or unbounded recursion.

use crate::diagnostic::{BspReport, DiagnosticCode};

/// Read a `u8` from `data` at `offset`.
#[inline]
pub fn read_u8(data: &[u8], offset: usize) -> Result<u8, BspReport> {
    data.get(offset)
        .copied()
        .ok_or_else(|| oob("u8", offset, data.len()))
}

/// Read a `u16` LE from `data` at `offset`.
#[inline]
pub fn read_u16_le(data: &[u8], offset: usize) -> Result<u16, BspReport> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| overflow("u16", offset))?;
    let slice = data
        .get(offset..end)
        .ok_or_else(|| oob("u16", offset, data.len()))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

/// Read an `i16` LE from `data` at `offset`.
#[inline]
pub fn read_i16_le(data: &[u8], offset: usize) -> Result<i16, BspReport> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| overflow("i16", offset))?;
    let slice = data
        .get(offset..end)
        .ok_or_else(|| oob("i16", offset, data.len()))?;
    Ok(i16::from_le_bytes([slice[0], slice[1]]))
}

/// Read a `u32` LE from `data` at `offset`.
#[inline]
pub fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, BspReport> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| overflow("u32", offset))?;
    let slice = data
        .get(offset..end)
        .ok_or_else(|| oob("u32", offset, data.len()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Read an `i32` LE from `data` at `offset`.
#[inline]
pub fn read_i32_le(data: &[u8], offset: usize) -> Result<i32, BspReport> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| overflow("i32", offset))?;
    let slice = data
        .get(offset..end)
        .ok_or_else(|| oob("i32", offset, data.len()))?;
    Ok(i32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Read an `f32` LE from `data` at `offset`.
#[inline]
pub fn read_f32_le(data: &[u8], offset: usize) -> Result<f32, BspReport> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| overflow("f32", offset))?;
    let slice = data
        .get(offset..end)
        .ok_or_else(|| oob("f32", offset, data.len()))?;
    Ok(f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Read an `f32` and reject non-finite values (NaN, infinity).
#[inline]
pub fn read_f32_finite(data: &[u8], offset: usize, context: &str) -> Result<f32, BspReport> {
    let val = read_f32_le(data, offset)?;
    if !val.is_finite() {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("{}: non-finite f32 value at offset {}", context, offset),
        ));
    }
    // Also reject excessively large values per the vertex component limit
    if val.abs() > crate::limits::MAX_VERTEX_COMPONENT {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!(
                "{}: f32 value {} exceeds max component {} at offset {}",
                context,
                val,
                crate::limits::MAX_VERTEX_COMPONENT,
                offset
            ),
        ));
    }
    Ok(val)
}

/// Read a `glam::Vec3` (3 × f32 LE) from `data` at `offset`.
#[inline]
pub fn read_vec3(data: &[u8], offset: usize) -> Result<glam::Vec3, BspReport> {
    let x = read_f32_le(data, offset)?;
    let y = read_f32_le(data, offset + 4)?;
    let z = read_f32_le(data, offset + 8)?;
    Ok(glam::Vec3::new(x, y, z))
}

/// Read a `glam::Vec3` with finiteness and magnitude checks.
#[inline]
pub fn read_vec3_finite(
    data: &[u8],
    offset: usize,
    context: &str,
) -> Result<glam::Vec3, BspReport> {
    let v = read_vec3(data, offset)?;
    if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
        return Err(BspReport::fatal(
            DiagnosticCode::StructuralCorruptLump,
            format!("{}: non-finite vec3 at offset {}", context, offset),
        ));
    }
    for (i, &comp) in [v.x, v.y, v.z].iter().enumerate() {
        if comp.abs() > crate::limits::MAX_VERTEX_COMPONENT {
            return Err(BspReport::fatal(
                DiagnosticCode::StructuralCorruptLump,
                format!(
                    "{}: vec3 component {} exceeds max at offset {}",
                    context, i, offset
                ),
            ));
        }
    }
    Ok(v)
}

/// Read a `[i16; 3]` tuple from `data` at `offset` (for mins/maxs in BSP29).
#[inline]
pub fn read_i16x3(data: &[u8], offset: usize) -> Result<[i16; 3], BspReport> {
    let a = read_i16_le(data, offset)?;
    let b = read_i16_le(data, offset + 2)?;
    let c = read_i16_le(data, offset + 4)?;
    Ok([a, b, c])
}

/// Read a `[i32; 3]` tuple from `data` at `offset` (for mins/maxs in BSP2).
#[inline]
pub fn read_i32x3(data: &[u8], offset: usize) -> Result<[i32; 3], BspReport> {
    let a = read_i32_le(data, offset)?;
    let b = read_i32_le(data, offset + 4)?;
    let c = read_i32_le(data, offset + 8)?;
    Ok([a, b, c])
}

#[inline]
fn oob(ty: &str, offset: usize, len: usize) -> BspReport {
    BspReport::fatal(
        DiagnosticCode::StructuralCorruptLump,
        format!(
            "out of bounds read {} at offset {} (file length {})",
            ty, offset, len
        ),
    )
}

#[inline]
fn overflow(ty: &str, offset: usize) -> BspReport {
    BspReport::fatal(
        DiagnosticCode::StructuralCorruptOverflow,
        format!("offset overflow reading {} at {}", ty, offset),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_primitives() {
        let data: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(read_u8(data, 0).unwrap(), 0x01);
        assert_eq!(read_u16_le(data, 0).unwrap(), 0x0201);
        assert_eq!(read_u32_le(data, 0).unwrap(), 0x04030201);
        assert_eq!(read_i32_le(data, 0).unwrap(), 0x04030201);
    }

    #[test]
    fn read_f32_finite_ok() {
        let data = 1.5f32.to_le_bytes();
        assert_eq!(read_f32_finite(&data, 0, "test").unwrap(), 1.5);
    }

    #[test]
    fn read_f32_finite_rejects_nan() {
        let data = f32::NAN.to_le_bytes();
        let r = read_f32_finite(&data, 0, "test");
        assert!(r.is_err());
    }

    #[test]
    fn read_f32_finite_rejects_inf() {
        let data = f32::INFINITY.to_le_bytes();
        let r = read_f32_finite(&data, 0, "test");
        assert!(r.is_err());
    }

    #[test]
    fn read_f32_finite_rejects_large() {
        let data = 100_000.0f32.to_le_bytes();
        let r = read_f32_finite(&data, 0, "test");
        assert!(r.is_err());
    }

    #[test]
    fn read_oob() {
        let data: &[u8] = &[0, 1, 2];
        assert!(read_u32_le(data, 0).is_err());
        assert!(read_u32_le(data, 1).is_err());
        assert!(read_u16_le(data, 2).is_err());
    }

    #[test]
    fn read_i16x3_ok() {
        let data: &[u8] = &[0x01, 0x00, 0x02, 0x00, 0x03, 0x00];
        assert_eq!(read_i16x3(data, 0).unwrap(), [1, 2, 3]);
    }

    #[test]
    fn read_i32x3_ok() {
        let mut data = Vec::new();
        data.extend_from_slice(&1i32.to_le_bytes());
        data.extend_from_slice(&2i32.to_le_bytes());
        data.extend_from_slice(&3i32.to_le_bytes());
        assert_eq!(read_i32x3(&data, 0).unwrap(), [1, 2, 3]);
    }
}
