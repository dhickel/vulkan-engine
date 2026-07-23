//! Lightmap atlas layout, style slot allocation, and luxel data.
//!
//! Contract: `bsp-renderer-lighting.md` §2.

use crate::diagnostic::{BspReport, DiagnosticCode};

/// Maximum atlas page size (4096² texels).
pub const ATLAS_PAGE_SIZE: u32 = 4096;
/// Padding between face luxel blocks in the atlas (2 texels).
pub const ATLAS_PADDING: u32 = 2;
/// Maximum number of atlas pages.
pub const MAX_ATLAS_PAGES: usize = 4;
/// Maximum style identifier.
pub const MAX_STYLE_IDENTIFIER: u8 = 63;
/// Style sentinel (unused slot).
pub const STYLE_SENTINEL: u8 = 255;
/// Maximum styles per face.
pub const MAX_STYLES_PER_FACE: usize = 4;

/// A single RGB luxel value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Luxel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Luxel {
    /// Create a luxel from equal RGB channels (monochrome).
    #[inline]
    pub fn from_gray(gray: u8) -> Self {
        Luxel {
            r: gray,
            g: gray,
            b: gray,
        }
    }

    /// Create a luxel from separate RGB channels.
    #[inline]
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Luxel { r, g, b }
    }
}

/// Layout of a face's lightmap luxels within an atlas page.
#[derive(Debug, Clone)]
pub struct FaceLightmapLayout {
    /// Which atlas page this face's lightmap lives on.
    pub page_index: u32,
    /// Pixel coordinates of the bottom-left luxel in the atlas.
    pub atlas_offset: (u32, u32),
    /// Luxel count (width, height).
    pub luxel_extents: (u32, u32),
    /// Whether this face has any lightmap data.
    pub has_data: bool,
}

/// A single atlas page as a flat RGB8 array.
#[derive(Debug, Clone)]
pub struct AtlasPage {
    /// Page index.
    pub index: u32,
    /// RGB8 pixel data (width × height × 3 bytes).
    pub data: Vec<u8>,
    /// Page width.
    pub width: u32,
    /// Page height.
    pub height: u32,
    /// Current write cursor position (x, y) for next face.
    cursor: (u32, u32),
    /// Current row height (used during packing).
    row_height: u32,
}

impl AtlasPage {
    /// Create a new atlas page.
    pub fn new(index: u32, width: u32, height: u32) -> Self {
        let size = (width * height * 3) as usize;
        AtlasPage {
            index,
            data: vec![0u8; size],
            width,
            height,
            cursor: (0, 0),
            row_height: 0,
        }
    }

    /// Try to allocate a rectangle of (w, h) luxels with padding.
    /// Returns the atlas offset (x, y) if successful, or None if full.
    pub fn allocate(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
        let padded_w = w + ATLAS_PADDING * 2;
        let padded_h = h + ATLAS_PADDING * 2;

        // Try to fit on current row
        if self.cursor.0 + padded_w <= self.width && self.cursor.1 + padded_h <= self.height {
            let offset = (self.cursor.0 + ATLAS_PADDING, self.cursor.1 + ATLAS_PADDING);
            self.cursor.0 += padded_w;
            self.row_height = self.row_height.max(padded_h);
            return Some(offset);
        }

        // Move to next row
        self.cursor.0 = 0;
        self.cursor.1 += self.row_height;
        self.row_height = 0;

        if self.cursor.1 + padded_h <= self.height && padded_w <= self.width {
            let offset = (ATLAS_PADDING, self.cursor.1 + ATLAS_PADDING);
            self.cursor.0 = padded_w;
            self.row_height = padded_h;
            return Some(offset);
        }

        None
    }

    /// Write a luxel block at the given atlas offset.
    pub fn write_luxels(
        &mut self,
        offset: (u32, u32),
        luxels: &[Luxel],
        width: u32,
        height: u32,
    ) {
        for y in 0..height {
            for x in 0..width {
                let lx = (offset.0 + x) as usize;
                let ly = (offset.1 + y) as usize;
                if lx >= self.width as usize || ly >= self.height as usize {
                    continue;
                }
                let src_idx = (y * width + x) as usize;
                if src_idx < luxels.len() {
                    let dst_idx = (ly * self.width as usize + lx) * 3;
                    self.data[dst_idx] = luxels[src_idx].r;
                    self.data[dst_idx + 1] = luxels[src_idx].g;
                    self.data[dst_idx + 2] = luxels[src_idx].b;
                }
            }
        }
    }
}

/// Lightmap atlas composed of multiple pages.
#[derive(Debug, Clone)]
pub struct LightmapAtlas {
    /// Atlas pages.
    pub pages: Vec<AtlasPage>,
    /// Face → layout mapping.
    pub face_layouts: Vec<FaceLightmapLayout>,
    /// Style layers present in the atlas.
    pub styles: Vec<u8>,
}

impl LightmapAtlas {
    /// Create an empty atlas.
    pub fn new() -> Self {
        LightmapAtlas {
            pages: Vec::new(),
            face_layouts: Vec::new(),
            styles: vec![0], // style 0 is always present
        }
    }

    /// Add a needed style (if not already present).
    pub fn add_style(&mut self, style: u8) {
        if style == STYLE_SENTINEL || style > MAX_STYLE_IDENTIFIER {
            return;
        }
        if !self.styles.contains(&style) {
            self.styles.push(style);
            self.styles.sort_unstable();
        }
    }

    /// Allocate and write lightmap data for a face.
    pub fn allocate_face(
        &mut self,
        face_index: u32,
        luxel_data: &[Luxel],
        luxel_width: u32,
        luxel_height: u32,
    ) -> Result<FaceLightmapLayout, BspReport> {
        if luxel_data.is_empty() || luxel_width == 0 || luxel_height == 0 {
            let layout = FaceLightmapLayout {
                page_index: 0,
                atlas_offset: (0, 0),
                luxel_extents: (0, 0),
                has_data: false,
            };
            // Extend face_layouts up to face_index
            while self.face_layouts.len() <= face_index as usize {
                self.face_layouts.push(FaceLightmapLayout {
                    page_index: 0,
                    atlas_offset: (0, 0),
                    luxel_extents: (0, 0),
                    has_data: false,
                });
            }
            self.face_layouts[face_index as usize] = layout.clone();
            return Ok(layout);
        }

        // Clamp luxel dimensions
        let w = luxel_width.min(ATLAS_PAGE_SIZE - ATLAS_PADDING * 2);
        let h = luxel_height.min(ATLAS_PAGE_SIZE - ATLAS_PADDING * 2);

        // Try to allocate into existing pages
        for page_idx in 0..self.pages.len() {
            if let Some(offset) = self.pages[page_idx].allocate(w, h) {
                self.pages[page_idx].write_luxels(offset, luxel_data, w, h);
                let layout = FaceLightmapLayout {
                    page_index: page_idx as u32,
                    atlas_offset: offset,
                    luxel_extents: (w, h),
                    has_data: true,
                };
                while self.face_layouts.len() <= face_index as usize {
                    self.face_layouts.push(FaceLightmapLayout {
                        page_index: 0,
                        atlas_offset: (0, 0),
                        luxel_extents: (0, 0),
                        has_data: false,
                    });
                }
                self.face_layouts[face_index as usize] = layout.clone();
                return Ok(layout);
            }
        }

        // Create new page if within budget
        if self.pages.len() >= MAX_ATLAS_PAGES {
            return Err(BspReport::fatal(
                DiagnosticCode::AllocationExceeded,
                format!("lightmap atlas page budget {} exceeded", MAX_ATLAS_PAGES),
            ));
        }

        let page_index = self.pages.len() as u32;
        let mut page = AtlasPage::new(page_index, ATLAS_PAGE_SIZE, ATLAS_PAGE_SIZE);
        let offset = page.allocate(w, h).ok_or_else(|| {
            BspReport::fatal(
                DiagnosticCode::AllocationExceeded,
                "cannot fit luxel block even on new atlas page",
            )
        })?;
        page.write_luxels(offset, luxel_data, w, h);
        self.pages.push(page);

        let layout = FaceLightmapLayout {
            page_index,
            atlas_offset: offset,
            luxel_extents: (w, h),
            has_data: true,
        };
        while self.face_layouts.len() <= face_index as usize {
            self.face_layouts.push(FaceLightmapLayout {
                page_index: 0,
                atlas_offset: (0, 0),
                luxel_extents: (0, 0),
                has_data: false,
            });
        }
        self.face_layouts[face_index as usize] = layout.clone();
        Ok(layout)
    }

    /// Get the face layout for a given face index.
    pub fn get_layout(&self, face_index: u32) -> Option<&FaceLightmapLayout> {
        self.face_layouts.get(face_index as usize)
    }
}

impl Default for LightmapAtlas {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode lightmap data from BSP raw bytes into luxel arrays.
///
/// For monochrome source: each byte becomes equal R, G, B.
/// For colored source (BSPX/.lit): 3 bytes per luxel (R, G, B).
pub fn decode_lightmaps_monochrome(
    raw_data: &[u8],
    face_lightofs: &[i32],
    face_luxel_extents: &[(u32, u32)],
) -> Vec<Vec<Luxel>> {
    let mut result: Vec<Vec<Luxel>> = Vec::with_capacity(face_lightofs.len());

    for (fi, &lightofs) in face_lightofs.iter().enumerate() {
        if lightofs < 0 {
            result.push(Vec::new());
            continue;
        }
        let start = lightofs as usize;
        let extents = face_luxel_extents.get(fi).copied().unwrap_or((0, 0));
        let luxel_count = (extents.0 * extents.1) as usize;
        if luxel_count == 0 || start >= raw_data.len() {
            result.push(Vec::new());
            continue;
        }
        let end = (start + luxel_count).min(raw_data.len());
        let luxels: Vec<Luxel> = raw_data[start..end]
            .iter()
            .map(|&b| Luxel::from_gray(b))
            .collect();
        result.push(luxels);
    }

    result
}

/// Decode colored lightmap data from RGB source (BSPX/.lit).
pub fn decode_lightmaps_rgb(
    rgb_data: &[u8],
    face_lightofs: &[i32],
    face_luxel_extents: &[(u32, u32)],
) -> Vec<Vec<Luxel>> {
    let mut result: Vec<Vec<Luxel>> = Vec::with_capacity(face_lightofs.len());

    for (fi, &lightofs) in face_lightofs.iter().enumerate() {
        if lightofs < 0 {
            result.push(Vec::new());
            continue;
        }
        let start = lightofs as usize * 3; // RGB data is 3 bytes per luxel
        let extents = face_luxel_extents.get(fi).copied().unwrap_or((0, 0));
        let luxel_count = (extents.0 * extents.1) as usize;
        if luxel_count == 0 || start >= rgb_data.len() {
            result.push(Vec::new());
            continue;
        }
        let end = (start + luxel_count * 3).min(rgb_data.len());
        let luxels: Vec<Luxel> = rgb_data[start..end]
            .chunks(3)
            .filter(|c| c.len() == 3)
            .map(|c| Luxel::from_rgb(c[0], c[1], c[2]))
            .collect();
        result.push(luxels);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_page_allocates_and_writes() {
        let mut page = AtlasPage::new(0, 256, 256);
        let offset = page.allocate(16, 16).expect("should allocate");
        assert_eq!(offset.0, ATLAS_PADDING);
        assert_eq!(offset.1, ATLAS_PADDING);

        let luxels: Vec<Luxel> = (0..256).map(|i| Luxel::from_rgb(i as u8, 0, 0)).collect();
        page.write_luxels(offset, &luxels, 16, 16);

        // Check first luxel was written
        let idx = ((offset.1 as usize) * 256 + offset.0 as usize) * 3;
        assert_eq!(page.data[idx], 0); // R
        assert_eq!(page.data[idx + 1], 0); // G
        assert_eq!(page.data[idx + 2], 0); // B
    }

    #[test]
    fn atlas_page_wraps_row() {
        let mut page = AtlasPage::new(0, 64, 256);
        // First block: 16+4=20, fits at x=0
        let o1 = page.allocate(16, 16).unwrap();
        assert_eq!(o1.0, ATLAS_PADDING);

        // Second block: same row, x=20
        let o2 = page.allocate(16, 16).unwrap();
        assert_eq!(o2.0, 20 + ATLAS_PADDING);

        // Third block: same row, x=40
        let o3 = page.allocate(16, 16).unwrap();
        assert_eq!(o3.0, 40 + ATLAS_PADDING);

        // Fourth block: should wrap to next row (since 60+20 > 64)
        let o4 = page.allocate(16, 16).unwrap();
        assert_eq!(o4.0, ATLAS_PADDING);
        assert_eq!(o4.1, 20 + ATLAS_PADDING); // row height 20 (16+4)
    }

    #[test]
    fn altas_full_reports_none() {
        let mut page = AtlasPage::new(0, 32, 32);
        // Can fit at most one 28x28 block (with padding = 32x32)
        let _o1 = page.allocate(28, 28);
        let o2 = page.allocate(4, 4);
        assert!(o2.is_none());
    }

    #[test]
    fn decode_monochrome_lightmaps() {
        let raw = vec![128u8, 200, 64];
        let lightofs = vec![0, -1, 1];
        let extents = vec![(1, 1), (0, 0), (2, 1)];
        let result = decode_lightmaps_monochrome(&raw, &lightofs, &extents);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0], Luxel::from_gray(128));
        assert!(result[1].is_empty());
        assert_eq!(result[2].len(), 2);
        assert_eq!(result[2][0], Luxel::from_gray(200));
    }

    #[test]
    fn decode_rgb_lightmaps() {
        let rgb = vec![255, 128, 64, 100, 150, 200];
        let lightofs = vec![0, 1];
        let extents = vec![(1, 1), (1, 1)];
        let result = decode_lightmaps_rgb(&rgb, &lightofs, &extents);
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0], Luxel::from_rgb(255, 128, 64));
        assert_eq!(result[1][0], Luxel::from_rgb(100, 150, 200));
    }

    #[test]
    fn styles_sort_deterministically() {
        let mut atlas = LightmapAtlas::new();
        atlas.add_style(5);
        atlas.add_style(1);
        atlas.add_style(3);
        assert_eq!(atlas.styles, vec![0, 1, 3, 5]);
    }

    #[test]
    fn sentinel_style_not_added() {
        let mut atlas = LightmapAtlas::new();
        atlas.add_style(STYLE_SENTINEL);
        assert_eq!(atlas.styles, vec![0]);
    }
}
