//! Standalone 3D Perlin noise with fractal Brownian motion.
//!
//! Uses the classic Perlin algorithm with a seed-derived permutation table.
//! No external dependencies beyond the port's own PCG32V1 RNG.

use crate::cave_gen::rng::Pcg32V1;

/// 3D Perlin noise generator with a seed-derived permutation table.
#[derive(Debug, Clone)]
pub struct PerlinNoise {
    perm: [u8; 512],
}

impl PerlinNoise {
    /// Create a new Perlin noise generator from an RNG stream.
    /// The RNG is used to shuffle a 0..256 permutation table.
    pub fn from_rng(rng: &mut Pcg32V1) -> Self {
        let mut perm: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut perm);
        let mut table = [0u8; 512];
        for (i, &v) in perm.iter().enumerate() {
            table[i] = v;
            table[i + 256] = v;
        }
        Self { perm: table }
    }

    /// Sample 3D Perlin noise at (x, y, z). Output in approximately [-1, 1].
    pub fn noise_3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let xi = (x.floor() as i32) & 255;
        let yi = (y.floor() as i32) & 255;
        let zi = (z.floor() as i32) & 255;

        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();

        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let aaa = self.perm[(self.perm[(self.perm[xi as usize] as usize + yi as usize) & 255]
            as usize
            + zi as usize)
            & 255];
        let aba = self.perm[(self.perm[(self.perm[xi as usize] as usize + yi as usize + 1) & 255]
            as usize
            + zi as usize)
            & 255];
        let aab = self.perm[(self.perm[(self.perm[xi as usize] as usize + yi as usize) & 255]
            as usize
            + zi as usize
            + 1)
            & 255];
        let abb = self.perm[(
            self.perm[(self.perm[xi as usize] as usize + yi as usize + 1) & 255] as usize
                + zi as usize
                + 1
        ) & 255];
        let baa = self.perm[(
            self.perm[(self.perm[xi as usize + 1] as usize + yi as usize) & 255] as usize
                + zi as usize
        ) & 255];
        let bba = self.perm[(
            self.perm[(self.perm[xi as usize + 1] as usize + yi as usize + 1) & 255] as usize
                + zi as usize
        ) & 255];
        let bab = self.perm[(
            self.perm[(self.perm[xi as usize + 1] as usize + yi as usize) & 255] as usize
                + zi as usize
                + 1
        ) & 255];
        let bbb = self.perm[(
            self.perm[(self.perm[xi as usize + 1] as usize + yi as usize + 1) & 255] as usize
                + zi as usize
                + 1
        ) & 255];

        let x1 = lerp(
            grad(aaa, xf, yf, zf),
            grad(baa, xf - 1.0, yf, zf),
            u,
        );
        let x2 = lerp(
            grad(aba, xf, yf - 1.0, zf),
            grad(bba, xf - 1.0, yf - 1.0, zf),
            u,
        );
        let y1 = lerp(x1, x2, v);

        let x1 = lerp(
            grad(aab, xf, yf, zf - 1.0),
            grad(bab, xf - 1.0, yf, zf - 1.0),
            u,
        );
        let x2 = lerp(
            grad(abb, xf, yf - 1.0, zf - 1.0),
            grad(bbb, xf - 1.0, yf - 1.0, zf - 1.0),
            u,
        );
        let y2 = lerp(x1, x2, v);

        lerp(y1, y2, w)
    }

    /// Fractal Brownian motion: sum of octaves of noise.
    /// Returns a value in approximately [-1, 1].
    pub fn fbm_3d(
        &self,
        x: f64,
        y: f64,
        z: f64,
        octaves: u32,
        lacunarity: f64,
        gain: f64,
    ) -> f64 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for _ in 0..octaves {
            value += self.noise_3d(x * frequency, y * frequency, z * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= gain;
            frequency *= lacunarity;
        }

        if max_value > 0.0 {
            value / max_value
        } else {
            value
        }
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────

#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

fn grad(hash: u8, x: f64, y: f64, z: f64) -> f64 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    let u = if (h & 1) == 0 { u } else { -u };
    let v = if (h & 2) == 0 { v } else { -v };
    u + v
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cave_gen::rng::Pcg32V1;

    #[test]
    fn noise_deterministic() {
        let mut rng1 = Pcg32V1::from_phase(42, "noise-test");
        let n1 = PerlinNoise::from_rng(&mut rng1);

        let mut rng2 = Pcg32V1::from_phase(42, "noise-test");
        let n2 = PerlinNoise::from_rng(&mut rng2);

        for i in 0..20 {
            let x = i as f64 * 0.37;
            let y = i as f64 * 0.73;
            let z = i as f64 * 0.53;
            assert_eq!(n1.noise_3d(x, y, z), n2.noise_3d(x, y, z));
        }
    }

    #[test]
    fn noise_different_seeds_different_output() {
        let mut rng1 = Pcg32V1::from_phase(1, "noise-test");
        let n1 = PerlinNoise::from_rng(&mut rng1);

        let mut rng2 = Pcg32V1::from_phase(2, "noise-test");
        let n2 = PerlinNoise::from_rng(&mut rng2);

        let mut any_different = false;
        for i in 0..20 {
            let x = i as f64 * 1.73;
            let y = i as f64 * 2.17;
            let z = i as f64 * 3.31;
            if n1.noise_3d(x, y, z) != n2.noise_3d(x, y, z) {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "different seeds should produce different noise at some point");
    }

    #[test]
    fn noise_value_in_range() {
        let mut rng = Pcg32V1::from_phase(42, "noise-range-test");
        let noise = PerlinNoise::from_rng(&mut rng);
        for i in 0..100 {
            let x = i as f64 * 1.73;
            let y = i as f64 * 2.17;
            let z = i as f64 * 3.31;
            let v = noise.noise_3d(x, y, z);
            assert!(v >= -1.5 && v <= 1.5, "noise {v} out of expected range at ({x}, {y}, {z})");
        }
    }

    #[test]
    fn fbm_in_range() {
        let mut rng = Pcg32V1::from_phase(42, "fbm-test");
        let noise = PerlinNoise::from_rng(&mut rng);
        for i in 0..20 {
            let x = i as f64 * 0.5;
            let y = i as f64 * 0.7;
            let z = i as f64 * 1.1;
            let v = noise.fbm_3d(x, y, z, 4, 2.0, 0.5);
            assert!(v >= -1.5 && v <= 1.5, "fbm {v} out of range");
        }
    }

    #[test]
    fn fbm_deterministic() {
        let mut rng1 = Pcg32V1::from_phase(99, "fbm-det");
        let n1 = PerlinNoise::from_rng(&mut rng1);
        let mut rng2 = Pcg32V1::from_phase(99, "fbm-det");
        let n2 = PerlinNoise::from_rng(&mut rng2);
        assert_eq!(
            n1.fbm_3d(3.0, 4.0, 5.0, 3, 2.0, 0.5),
            n2.fbm_3d(3.0, 4.0, 5.0, 3, 2.0, 0.5)
        );
    }

    #[test]
    fn periodic_boundary_behavior() {
        let mut rng = Pcg32V1::from_phase(42, "periodic");
        let noise = PerlinNoise::from_rng(&mut rng);
        let v_before = noise.noise_3d(0.999, 0.0, 0.0);
        let v_after = noise.noise_3d(1.001, 0.0, 0.0);
        assert!(v_before >= -1.5 && v_before <= 1.5);
        assert!(v_after >= -1.5 && v_after <= 1.5);
    }
}
