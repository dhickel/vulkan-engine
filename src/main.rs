use std::env;
use std::process;

mod launch;

use launch::{LaunchCommand, LaunchError, LaunchOptions};

fn main() {
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

fn run(_options: LaunchOptions) -> Result<(), LaunchError> {
    Err(LaunchError::Runtime(
        "runtime project loading is not wired until Sprint 04 Phase 02".to_string(),
    ))
}

fn exit_with_error(err: LaunchError) -> ! {
    eprintln!("{err}");
    if err.is_usage() {
        eprintln!();
        eprint!("{}", launch::usage());
    }
    process::exit(err.exit_code());
}
