use std::env;
use std::process;

fn print_runtime_migration_help() {
    eprintln!("The legacy `renderer::run()` and `debug_runtime` path has been removed.");
    eprintln!("Run renderer example binaries directly:");
    eprintln!("  cargo run -p renderer --example api_test");
    eprintln!("  cargo run -p renderer --example demo_pbr");
    eprintln!("  cargo run -p renderer --example demo_unlit");
    eprintln!("  cargo run -p renderer --example demo_model_load");
    eprintln!("  cargo run -p renderer --example demo_async_loading");
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if !args.is_empty() {
        eprintln!("Unsupported engine binary arguments: {}", args.join(" "));
    }

    print_runtime_migration_help();
    process::exit(2);
}
