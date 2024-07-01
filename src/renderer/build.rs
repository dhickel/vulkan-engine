
use std::env;
use std::fs;
use std::path::Path;
use shaderc::Compiler;

fn main() {
    // Directory containing the shaders
    let shader_dir = "src/shaders";

    // Get the output directory
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_dir = shader_dir;
    // Create a shader compiler
    let compiler = Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("GL_EXT_buffer_reference", Some("1"));
    options.add_macro_definition("GL_GOOGLE_include_directive", Some("1"));
    options.set_include_callback(|name, _include_type, _source_file, _depth| {
        let path = Path::new(shader_dir).join(name);
        if path.exists() {
            Ok(shaderc::ResolvedInclude {
                resolved_name: path.to_str().unwrap().to_string(),
                content: fs::read_to_string(path).unwrap(),
            })
        } else {
            Err(format!("Failed to find include file: {}", name))
        }
    });

    // Iterate over the shader files in the shader directory
    for entry in fs::read_dir(shader_dir).unwrap() {
        let entry = entry.expect("Failed");
        let path = entry.path();

        if path.extension().unwrap() == "vert" || path.extension().unwrap() == "frag" || path.extension().unwrap() == "comp" {
            // Read the shader source code
            let shader_source = fs::read_to_string(&path).expect("Failed");

            // Compile the shader
            let binary_result = compiler.compile_into_spirv(
                &shader_source,
                match path.extension().unwrap().to_str().expect("Failed") {
                    "vert" => shaderc::ShaderKind::Vertex,
                    "frag" => shaderc::ShaderKind::Fragment,
                    "comp" => shaderc::ShaderKind::Compute,
                    _ => panic!("Unsupported shader type"),
                },
                path.file_name().unwrap().to_str().unwrap(),
                "main",
                Some(&options),
            ).unwrap_or_else(|e| {
                eprintln!("Shader compilation failed: {}", e);
                panic!("Shader compilation failed");
            });

            // Write the compiled SPIR-V to the output directory
            let output_path = Path::new(&out_dir).join(format!(
                "{}.spv",
                path.file_name().unwrap().to_str().unwrap()
            ));
            fs::write(&output_path, binary_result.as_binary_u8()).unwrap();
            println!("Built shaders");

            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}