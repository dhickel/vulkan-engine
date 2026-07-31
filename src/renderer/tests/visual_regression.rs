//! Decoded-pixel visual-regression comparison harness.
//!
//! Baseline update (review decoded output before retaining the result):
//! `VISUAL_REGRESSION_UPDATE=1 cargo test -p renderer visual_regression -- --nocapture`

use image::{GenericImageView, Pixel};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CaptureSidecar {
    target: String,
    extent: [u32; 2],
    frame: u32,
    scene_preset: String,
    #[serde(default)]
    declared_regions: Vec<DeclaredRegion>,
    #[serde(default)]
    notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DeclaredRegion {
    name: String,
    bounds: [u32; 4], // [x, y, width, height]
    expected_content: String,
}

#[derive(Debug, Clone)]
struct PixelDiff {
    max_abs_error: f32,
    differing_pixels: usize,
    total_pixels: usize,
    differing_ratio: f64,
}

impl PixelDiff {
    fn within_tolerance(&self, max_error: f32, max_ratio: f64) -> bool {
        self.max_abs_error <= max_error && self.differing_ratio <= max_ratio
    }
}

struct RegressionCase {
    name: &'static str,
    baseline_png: &'static str,
    baseline_json: &'static str,
    capture_png: &'static str,
    capture_json: &'static str,
    max_per_channel_error: f32,
    max_differing_ratio: f64,
}

fn compare_pngs(baseline_path: &Path, capture_path: &Path) -> Result<PixelDiff, String> {
    let baseline = image::open(baseline_path)
        .map_err(|e| format!("failed to decode baseline {}: {e}", baseline_path.display()))?;
    let capture = image::open(capture_path)
        .map_err(|e| format!("failed to decode capture {}: {e}", capture_path.display()))?;

    let baseline_extent = baseline.dimensions();
    let capture_extent = capture.dimensions();
    if baseline_extent != capture_extent {
        return Err(format!(
            "dimension mismatch: baseline {}×{}, capture {}×{}",
            baseline_extent.0, baseline_extent.1, capture_extent.0, capture_extent.1
        ));
    }
    if baseline.color() != capture.color() {
        return Err(format!(
            "color model mismatch: baseline {:?}, capture {:?}",
            baseline.color(),
            capture.color()
        ));
    }

    let total_pixels = baseline_extent.0 as usize * baseline_extent.1 as usize;
    let mut max_abs_error = 0.0f32;
    let mut differing_pixels = 0usize;
    for y in 0..baseline_extent.1 {
        for x in 0..baseline_extent.0 {
            let baseline_pixel = baseline.get_pixel(x, y);
            let capture_pixel = capture.get_pixel(x, y);
            let pixel_error = baseline_pixel
                .channels()
                .iter()
                .zip(capture_pixel.channels())
                .map(|(a, b)| (*a as f32 - *b as f32).abs())
                .fold(0.0f32, f32::max);
            max_abs_error = max_abs_error.max(pixel_error);
            differing_pixels += usize::from(pixel_error > 0.5);
        }
    }

    Ok(PixelDiff {
        max_abs_error,
        differing_pixels,
        total_pixels,
        differing_ratio: differing_pixels as f64 / total_pixels as f64,
    })
}

fn load_sidecar(path: &Path) -> Result<CaptureSidecar, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read sidecar {}: {e}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse sidecar {}: {e}", path.display()))
}

fn validate_sidecar(
    capture: &CaptureSidecar,
    baseline: &CaptureSidecar,
    decoded_extent: [u32; 2],
) -> Result<(), String> {
    if capture.extent != decoded_extent {
        return Err(format!(
            "sidecar extent {:?} does not match decoded PNG {:?}",
            capture.extent, decoded_extent
        ));
    }
    if capture.target != baseline.target {
        return Err(format!(
            "target mismatch: baseline {:?}, capture {:?}",
            baseline.target, capture.target
        ));
    }
    if capture.extent != baseline.extent {
        return Err(format!(
            "extent mismatch: baseline {:?}, capture {:?}",
            baseline.extent, capture.extent
        ));
    }
    if capture.frame != baseline.frame {
        return Err(format!(
            "frame mismatch: baseline {}, capture {}",
            baseline.frame, capture.frame
        ));
    }
    if capture.scene_preset != baseline.scene_preset {
        return Err(format!(
            "scene preset mismatch: baseline {:?}, capture {:?}",
            baseline.scene_preset, capture.scene_preset
        ));
    }
    if capture.declared_regions.is_empty() {
        return Err("sidecar declares no structural regions".to_string());
    }
    if capture.declared_regions != baseline.declared_regions {
        return Err("declared regions differ from the reviewed baseline".to_string());
    }

    for region in &capture.declared_regions {
        let [x, y, width, height] = region.bounds;
        let valid = !region.name.is_empty()
            && !region.expected_content.is_empty()
            && width > 0
            && height > 0
            && x.checked_add(width)
                .is_some_and(|right| right <= capture.extent[0])
            && y.checked_add(height)
                .is_some_and(|bottom| bottom <= capture.extent[1]);
        if !valid {
            return Err(format!("invalid declared region {region:?}"));
        }
    }
    Ok(())
}

fn update_baseline(case: &RegressionCase, fixture_dir: &Path) -> Result<(), String> {
    let capture_png = fixture_dir.join(case.capture_png);
    let capture_json = fixture_dir.join(case.capture_json);
    if !capture_png.exists() || !capture_json.exists() {
        return Err(format!(
            "baseline update requires capture PNG and sidecar: {}, {}",
            capture_png.display(),
            capture_json.display()
        ));
    }
    std::fs::copy(&capture_png, fixture_dir.join(case.baseline_png))
        .map_err(|e| format!("copy baseline PNG: {e}"))?;
    std::fs::copy(&capture_json, fixture_dir.join(case.baseline_json))
        .map_err(|e| format!("copy baseline sidecar: {e}"))?;
    Ok(())
}

fn regression_cases() -> [RegressionCase; 2] {
    [
        RegressionCase {
            name: "capture_geometry",
            baseline_png: "capture_geometry.baseline.png",
            baseline_json: "capture_geometry.baseline.json",
            capture_png: "capture_geometry.capture.png",
            capture_json: "capture_geometry.capture.json",
            max_per_channel_error: 5.0,
            max_differing_ratio: 0.02,
        },
        RegressionCase {
            name: "capture_shadows",
            baseline_png: "capture_shadows.baseline.png",
            baseline_json: "capture_shadows.baseline.json",
            capture_png: "capture_shadows.capture.png",
            capture_json: "capture_shadows.capture.json",
            max_per_channel_error: 5.0,
            max_differing_ratio: 0.02,
        },
    ]
}

fn run_regression_case(case: &RegressionCase, fixture_dir: &Path) -> Result<(), String> {
    let baseline_png = fixture_dir.join(case.baseline_png);
    let capture_png = fixture_dir.join(case.capture_png);
    let baseline_json = fixture_dir.join(case.baseline_json);
    let capture_json = fixture_dir.join(case.capture_json);
    for path in [&baseline_png, &capture_png, &baseline_json, &capture_json] {
        if !path.exists() {
            return Err(format!(
                "required visual-regression fixture missing: {}",
                path.display()
            ));
        }
    }

    let diff = compare_pngs(&baseline_png, &capture_png)?;
    if !diff.within_tolerance(case.max_per_channel_error, case.max_differing_ratio) {
        return Err(format!(
            "{} exceeded tolerance: max error {:.3}/{:.3}, differing ratio {:.6}/{:.6}",
            case.name,
            diff.max_abs_error,
            case.max_per_channel_error,
            diff.differing_ratio,
            case.max_differing_ratio
        ));
    }

    let baseline_sidecar = load_sidecar(&baseline_json)?;
    let capture_sidecar = load_sidecar(&capture_json)?;
    let extent = image::image_dimensions(&capture_png)
        .map_err(|e| format!("failed to read capture dimensions: {e}"))?;
    validate_sidecar(&capture_sidecar, &baseline_sidecar, [extent.0, extent.1])?;

    println!(
        "[{}] PASS max_abs_error={:.3} differing_pixels={}/{} ({:.4}%)",
        case.name,
        diff.max_abs_error,
        diff.differing_pixels,
        diff.total_pixels,
        diff.differing_ratio * 100.0
    );
    Ok(())
}

fn run_named_case(name: &str) {
    let fixture_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/visual_regression");
    let cases = regression_cases();
    let case = cases.iter().find(|case| case.name == name).unwrap();
    if std::env::var("VISUAL_REGRESSION_UPDATE").as_deref() == Ok("1") {
        update_baseline(case, &fixture_dir).expect("update reviewed baseline inputs");
    } else {
        run_regression_case(case, &fixture_dir).expect("visual regression");
    }
}

#[test]
fn visual_regression_geometry() {
    run_named_case("capture_geometry");
}

#[test]
fn visual_regression_shadows() {
    run_named_case("capture_shadows");
}

#[test]
fn visual_regression_rejects_structural_sidecar_drift() {
    let baseline = CaptureSidecar {
        target: "draw".into(),
        extent: [4, 4],
        frame: 5,
        scene_preset: "fixture".into(),
        declared_regions: vec![DeclaredRegion {
            name: "frame".into(),
            bounds: [0, 0, 4, 4],
            expected_content: "fixture".into(),
        }],
        notes: String::new(),
    };
    for mut drifted in [baseline.clone(), baseline.clone(), baseline.clone()] {
        if drifted.target == "draw" {
            drifted.target = "present".into();
        }
        assert!(validate_sidecar(&drifted, &baseline, [4, 4]).is_err());
    }
    let mut bad_extent = baseline.clone();
    bad_extent.extent = [8, 4];
    assert!(validate_sidecar(&bad_extent, &baseline, [4, 4]).is_err());
    let mut bad_frame = baseline.clone();
    bad_frame.frame = 6;
    assert!(validate_sidecar(&bad_frame, &baseline, [4, 4]).is_err());
    let mut bad_regions = baseline.clone();
    bad_regions.declared_regions.clear();
    assert!(validate_sidecar(&bad_regions, &baseline, [4, 4]).is_err());
}
