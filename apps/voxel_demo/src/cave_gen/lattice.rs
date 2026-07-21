//! Signed 8-bit density lattice with separate material tag byte.
//!
//! `DenseLattice<T>` is a generic 3D grid. The port uses `DenseLattice<i8>` for
//! signed density and a companion array of `u8` material tags stored as a separate
//! field on a concrete `VoxelWorld` struct.

/// A dense 3D lattice generic over the cell type.
#[derive(Debug, Clone)]
pub struct DenseLattice<T> {
    width: u32,
    height: u32,
    depth: u32,
    cells: Vec<T>,
}

impl<T: Clone + Default> DenseLattice<T> {
    /// Create a new lattice filled with the default value.
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        assert!(
            width > 0 && height > 0 && depth > 0,
            "dimensions must be positive"
        );
        let len = (width as usize) * (height as usize) * (depth as usize);
        Self {
            width,
            height,
            depth,
            cells: vec![T::default(); len],
        }
    }

    /// Fill every cell with the given value.
    pub fn fill(&mut self, value: T) {
        self.cells.fill(value);
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn dims(&self) -> (u32, u32, u32) {
        (self.width, self.height, self.depth)
    }

    /// Linear index for (x, y, z). Z is the slowest axis (layer), X is fastest.
    #[inline]
    fn index(&self, x: u32, y: u32, z: u32) -> Option<usize> {
        if x < self.width && y < self.height && z < self.depth {
            Some(
                (x as usize)
                    + (y as usize) * (self.width as usize)
                    + (z as usize) * (self.width as usize) * (self.height as usize),
            )
        } else {
            None
        }
    }

    /// Get a reference, or None if out of bounds.
    pub fn get(&self, x: u32, y: u32, z: u32) -> Option<&T> {
        self.index(x, y, z).map(|i| &self.cells[i])
    }

    /// Get a mutable reference, or None if out of bounds.
    pub fn get_mut(&mut self, x: u32, y: u32, z: u32) -> Option<&mut T> {
        self.index(x, y, z).map(|i| &mut self.cells[i])
    }

    /// Set a cell's value. Debug-asserts bounds.
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: T) {
        let i = self.index(x, y, z).expect("bounds check");
        self.cells[i] = value;
    }

    /// Read a cell's value. Debug-asserts bounds.
    pub fn read(&self, x: u32, y: u32, z: u32) -> &T {
        let i = self.index(x, y, z).expect("bounds check");
        &self.cells[i]
    }

    /// Iterate over all cells in linear order (x fastest, z slowest).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.cells.iter()
    }

    /// Iterate over all cells with their 3D coordinates.
    pub fn iter_coords(&self) -> impl Iterator<Item = (u32, u32, u32, &T)> {
        let (w, h, _d) = (self.width, self.height, self.depth);
        self.cells.iter().enumerate().map(move |(i, v)| {
            let x = (i % (w as usize)) as u32;
            let y = ((i / (w as usize)) % (h as usize)) as u32;
            let z = (i / (w as usize * h as usize)) as u32;
            (x, y, z, v)
        })
    }
}

/// A signed 8-bit density value. Negative = solid (cave wall), non-negative = air.
/// The value range −128..=127 maps to solidness: −128 is fully solid, 127 is empty air.
/// The zero-crossing (0) is the isosurface.
pub type Density = i8;

/// Material tag byte stored in a companion array parallel to the density lattice.
/// 0 = default/rock, 1–255 available for material variants.
pub type MaterialTag = u8;

/// Default material tag for unassigned cells.
pub const DEFAULT_MATERIAL: MaterialTag = 0;

/// A voxel world: a signed density lattice plus a parallel material tag array.
#[derive(Debug, Clone)]
pub struct VoxelWorld {
    density: DenseLattice<Density>,
    material: DenseLattice<MaterialTag>,
}

impl VoxelWorld {
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            density: DenseLattice::new(width, height, depth),
            material: DenseLattice::new(width, height, depth),
        }
    }

    pub fn dims(&self) -> (u32, u32, u32) {
        self.density.dims()
    }

    pub fn density(&self) -> &DenseLattice<Density> {
        &self.density
    }

    pub fn density_mut(&mut self) -> &mut DenseLattice<Density> {
        &mut self.density
    }

    pub fn material(&self) -> &DenseLattice<MaterialTag> {
        &self.material
    }

    pub fn material_mut(&mut self) -> &mut DenseLattice<MaterialTag> {
        &mut self.material
    }

    /// Set both density and material at a given coordinate.
    pub fn set_voxel(&mut self, x: u32, y: u32, z: u32, density: Density, material: MaterialTag) {
        self.density.set(x, y, z, density);
        self.material.set(x, y, z, material);
    }

    /// Fill the world with solid rock (−128 density, default material).
    pub fn fill_solid(&mut self) {
        self.density.fill(-128i8);
        self.material.fill(DEFAULT_MATERIAL);
    }

    /// Fill the world with empty air (127 density, default material).
    pub fn fill_air(&mut self) {
        self.density.fill(127i8);
        self.material.fill(DEFAULT_MATERIAL);
    }

    /// Generate a sphere test body: a solid sphere of the given radius centered in the world.
    /// The sphere interior gets `density` (negative = solid), exterior stays as-is.
    /// Returns the center (cx, cy, cz) in voxel coordinates.
    pub fn stamp_sphere(
        &mut self,
        radius: f32,
        density: Density,
        material: MaterialTag,
    ) -> (f32, f32, f32) {
        let (w, h, d) = self.dims();
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let cz = d as f32 / 2.0;
        let r2 = radius * radius;
        for z in 0..d {
            let dz = z as f32 - cz;
            for y in 0..h {
                let dy = y as f32 - cy;
                for x in 0..w {
                    let dx = x as f32 - cx;
                    if dx * dx + dy * dy + dz * dz <= r2 {
                        self.set_voxel(x, y, z, density, material);
                    }
                }
            }
        }
        (cx, cy, cz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_construction_and_bounds() {
        let lat = DenseLattice::<i8>::new(4, 4, 4);
        assert_eq!(lat.len(), 64);
        assert_eq!(lat.dims(), (4, 4, 4));
        assert!(lat.get(0, 0, 0).is_some());
        assert!(lat.get(3, 3, 3).is_some());
        assert!(lat.get(4, 0, 0).is_none());
        assert!(lat.get(0, 0, 4).is_none());
    }

    #[test]
    #[should_panic(expected = "dimensions must be positive")]
    fn lattice_zero_dimension_panics() {
        DenseLattice::<i8>::new(0, 4, 4);
    }

    #[test]
    fn lattice_fill_and_read() {
        let mut lat = DenseLattice::new(2, 2, 2);
        lat.fill(42i8);
        assert_eq!(lat.read(0, 0, 0), &42);
        assert_eq!(lat.read(1, 1, 1), &42);
        lat.set(0, 0, 0, 100);
        assert_eq!(lat.read(0, 0, 0), &100);
    }

    #[test]
    #[should_panic(expected = "bounds check")]
    fn lattice_set_out_of_bounds_panics() {
        let mut lat = DenseLattice::new(2, 2, 2);
        lat.set(2, 2, 2, 0);
    }

    #[test]
    fn lattice_index_order_x_fastest() {
        let mut lat = DenseLattice::new(3, 3, 3);
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    let val = (x + y * 3 + z * 9) as i8;
                    lat.set(x, y, z, val);
                }
            }
        }
        let coords: Vec<_> = lat.iter_coords().map(|(x, y, z, _)| (x, y, z)).collect();
        assert_eq!(coords[0], (0, 0, 0));
        assert_eq!(coords[1], (1, 0, 0));
        assert_eq!(coords[2], (2, 0, 0));
        assert_eq!(coords[3], (0, 1, 0));
    }

    #[test]
    fn voxel_world_fill_modes() {
        let mut world = VoxelWorld::new(8, 8, 8);
        world.fill_solid();
        assert_eq!(*world.density().read(3, 3, 3), -128i8);
        assert_eq!(*world.material().read(3, 3, 3), DEFAULT_MATERIAL);

        world.fill_air();
        assert_eq!(*world.density().read(3, 3, 3), 127i8);
    }

    #[test]
    fn voxel_world_material_companion() {
        let mut world = VoxelWorld::new(8, 8, 8);
        world.fill_solid();
        world.set_voxel(3, 3, 3, 0i8, 42);
        assert_eq!(world.material().read(3, 3, 3), &42);
        assert_eq!(world.density().read(3, 3, 3), &0);
    }

    #[test]
    fn lattice_iter_coords_completeness() {
        let lat = DenseLattice::<i8>::new(5, 5, 5);
        let count = lat.iter_coords().count();
        assert_eq!(count, 125);
    }

    #[test]
    fn voxel_world_sphere_stamp() {
        let mut world = VoxelWorld::new(32, 32, 32);
        world.fill_air();
        world.stamp_sphere(8.0, -128i8, 1);
        let center = world.density().read(16, 16, 16);
        assert_eq!(*center, -128i8);
        let corner = world.density().read(0, 0, 0);
        assert_eq!(*corner, 127i8);
    }
}
