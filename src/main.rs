use std::env;
use std::process;

use engine::launch::{self, LaunchCommand, LaunchError, LaunchOptions};

fn main() {
    init_logging();

    match launch::parse_command(env::args().skip(1)) {
        Ok(LaunchCommand::Help) => {
            print!("{}", launch::usage());
        }
        Ok(LaunchCommand::Run(options)) => match run(options) {
            Ok(()) => {}
            Err(err) => exit_with_error(err),
        },
        Err(err) => exit_with_error(err),
    }
}

fn run(options: LaunchOptions) -> Result<(), LaunchError> {
    engine::runtime::run(options).map_err(LaunchError::Runtime)
}

fn exit_with_error(err: LaunchError) -> ! {
    eprintln!("{err}");
    if err.is_usage() {
        eprintln!();
        eprint!("{}", launch::usage());
    }
    process::exit(err.exit_code());
}

fn init_logging() {
    let _ = env_logger::Builder::new()
        .target(env_logger::Target::Stdout)
        .parse_filters(&env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .try_init();
}
