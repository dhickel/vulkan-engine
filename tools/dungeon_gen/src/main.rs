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
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let seed = seed.unwrap_or(0);
    let config = match class.as_deref().unwrap_or("m1") {
        "m1" => bsp_generator::DungeonConfig::nominal_m1(),
        "m2" => bsp_generator::DungeonConfig::nominal_m2(),
        other => return Err(format!("--class must be m1 or m2, got {other}")),
    };

    let (map_text, meta) = bsp_generator::generate(seed, config)
        .map_err(|err| format!("generation failed: {err:?}"))?;

    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("create output directory {}: {err}", parent.display()))?;
        }
        std::fs::write(&path, map_text)
            .map_err(|err| format!("write {}: {err}", path.display()))?;
        eprintln!(
            "generated {}: seed={seed} rooms={} corridors={} estimated_faces={}",
            path.display(),
            meta.room_count,
            meta.corridor_count,
            meta.face_count_estimate
        );
    } else {
        print!("{map_text}");
    }

    Ok(())
}

fn print_usage() {
    eprintln!("Usage: dungeon_gen [--seed <u64>] [--class m1|m2] [--out <path>]");
}
