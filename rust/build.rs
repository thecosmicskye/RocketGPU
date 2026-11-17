use std::env;
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("..");
    for p in ["CMakeLists.txt", "include", "src", "rust/shim"] {
        println!("cargo:rerun-if-changed={}", root.join(p).display());
    }
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let mut cfg = cmake::Config::new(&root);
    cfg.profile("Release")
        .define("ROCKETGPU_BUILD_RUST", "ON")
        .define("ROCKETGPU_USE_CUDA", if cuda { "ON" } else { "OFF" })
        .build_target("rocketgpu_rs");
    let build = cfg.build().join("build");

    for dir in [build.clone(), build.join("Release"), build.join("rust/shim"), build.join("rust/shim/Release")] {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static=rocketgpu_rs");
    println!("cargo:rustc-link-lib=static=rocketgpu");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-lib=c++");
    } else if target_env != "msvc" {
        println!("cargo:rustc-link-lib=stdc++");
    }
    if cuda {
        for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
            println!("cargo:rerun-if-env-changed={var}");
            if let Some(dir) = env::var_os(var) {
                let dir = PathBuf::from(dir);
                println!("cargo:rustc-link-search=native={}", dir.join("lib64").display());
                println!("cargo:rustc-link-search=native={}", dir.join("lib").join("x64").display());
            }
        }
        println!("cargo:rustc-link-lib=cudart");
    }
    println!("cargo:rustc-env=ROCKETGPU_RS_PRESETS={}", build.join("rust/shim/presets.rs").display());
}
