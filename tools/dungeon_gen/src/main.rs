use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("dungeon_gen: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut seed: Option<u64> = None;
    let mut class: Option<String> = None;
    let mut out: Option<PathBuf> = None;
    let mut preset: Option<String> = None;
    let mut extent: Option<u32> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                let value = args.next().ok_or("--seed requires a value")?;
                seed = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_| format!("invalid --seed value: {value}"))?,
                );
            }
            "--class" => {
                class = Some(args.next().ok_or("--class requires a value")?);
            }
            "--out" => {
                out = Some(PathBuf::from(args.next().ok_or("--out requires a value")?));
            }
            "--preset" => {
                preset = Some(args.next().ok_or("--preset requires a value")?);
            }
            "--extent" => {
                let value = args.next().ok_or("--extent requires a value")?;
                extent = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| format!("invalid --extent value: {value}"))?,
                );
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let seed = seed.unwrap_or(0);
    match class.as_deref().unwrap_or("m1") {
        "m1" => {
            if preset.is_some() {
                return Err("--preset is not valid for class m1".to_string());
            }
            if extent.is_some() {
                return Err("--extent is not valid for class m1".to_string());
            }
            let config = bsp_generator::DungeonConfig::nominal_m1();
            let (map_text, meta) = bsp_generator::generate(seed, config)
                .map_err(|err| format!("generation failed: {err:?}"))?;
            write_output(
                map_text,
                meta.room_count,
                meta.corridor_count,
                meta.face_count_estimate,
                seed,
                out,
            )?;
        }
        "m2" => {
            if preset.is_some() {
                return Err("--preset is not valid for class m2".to_string());
            }
            if extent.is_some() {
                return Err("--extent is not valid for class m2".to_string());
            }
            let config = bsp_generator::enhanced::config::EnhancedConfig::nominal();
            let (map_text, meta) = bsp_generator::generate_enhanced(seed, config)
                .map_err(|err| format!("enhanced generation failed: {err}"))?;
            write_output(
                map_text,
                meta.room_count,
                meta.route_count,
                (meta.room_count + meta.route_count + meta.transition_count) * 6,
                seed,
                out,
            )?;
        }
        "m3" => {
            let preset_tag = preset.as_deref().unwrap_or("sparse");
            let v3_preset = bsp_generator::V3Preset::from_tag(preset_tag).ok_or_else(|| {
                format!("unknown --preset '{preset_tag}'. Use sparse, moderate, or rich")
            })?;

            // Default extent: Sparse/Moderate 2048, Rich 3072
            let default_extent = match v3_preset {
                bsp_generator::V3Preset::Sparse | bsp_generator::V3Preset::Moderate => 2048,
                bsp_generator::V3Preset::Rich => 3072,
            };
            let xy_extent = extent.unwrap_or(default_extent);

            let config = bsp_generator::V3Config::new(seed, v3_preset, xy_extent)
                .map_err(|err| format!("v3 config invalid: {err}"))?;
            let (map_text, meta) = bsp_generator::generate_enhanced_v3(&config)
                .map_err(|err| format!("v3 generation failed: {err}"))?;
            write_output(
                map_text,
                meta.room_count(),
                meta.route_count(),
                meta.actual_faces(),
                seed,
                out,
            )?;
        }
        other => return Err(format!("--class must be m1, m2, or m3, got {other}")),
    }

    Ok(())
}

fn write_output(
    map_text: String,
    rooms: u32,
    corridors: u32,
    estimated_faces: u32,
    seed: u64,
    out: Option<PathBuf>,
) -> Result<(), String> {
    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("create output directory {}: {err}", parent.display()))?;
        }
        std::fs::write(&path, map_text)
            .map_err(|err| format!("write {}: {err}", path.display()))?;
        eprintln!(
            "generated {}: seed={seed} rooms={rooms} corridors={corridors} estimated_faces={estimated_faces}",
            path.display(),
        );
    } else {
        print!("{map_text}");
    }
    Ok(())
}

fn print_usage() {
    eprintln!("Usage: dungeon_gen [--seed <u64>] [--class m1|m2|m3] [--out <path>] [--preset sparse|moderate|rich] [--extent <1024..3072>]");
    eprintln!("  m1: Legacy v1 single-layer dungeon");
    eprintln!("  m2: Enhanced v2 two-layer dungeon with stairs");
    eprintln!("  m3: Enhanced v3 two-layer dungeon with cardinal+45° geometry");
    eprintln!("  --preset: (m3 only) sparse (default), moderate, or rich");
    eprintln!("  --extent: (m3 only) XY extent (1024–3072, multiple of 16)");
    eprintln!("            Sparse/Moderate default: 2048, Rich default: 3072");
}
