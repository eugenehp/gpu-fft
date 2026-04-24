fn main() {
    #[cfg(feature = "mlx")]
    build_mlx();
}

#[cfg(feature = "mlx")]
fn build_mlx() {
    println!("cargo:rerun-if-changed=ffi/mlx_fft.c");
    println!("cargo:rerun-if-changed=ffi/mlx_fft.h");
    println!("cargo:rerun-if-env-changed=MLX_C_PREFIX");

    let prefix = find_mlx_c_prefix().unwrap_or_else(|| {
        panic!(
            "\n\
            ══════════════════════════════════════════════════════════════\n\
            Could not find the MLX-C library (libmlxc.a).\n\
            \n\
            Install it with:\n\
            \n\
              git clone https://github.com/ml-explore/mlx-c /tmp/mlx-c\n\
              cd /tmp/mlx-c && mkdir build && cd build\n\
              cmake .. && make -j$(sysctl -n hw.ncpu)\n\
              cmake --install . --prefix $HOME/mlx-c-install\n\
            \n\
            Or set MLX_C_PREFIX to point to your installation:\n\
            \n\
              MLX_C_PREFIX=/path/to/mlx-c-install cargo build --features mlx\n\
            \n\
            Searched locations:\n\
              - $MLX_C_PREFIX (not set)\n\
              - $HOME/mlx-c-install\n\
              - /opt/homebrew\n\
              - /usr/local\n\
            ══════════════════════════════════════════════════════════════"
        )
    });

    eprintln!("cargo:warning=Using MLX-C from {}", prefix.display());

    cc::Build::new()
        .file("ffi/mlx_fft.c")
        .include("ffi")
        .include(prefix.join("include"))
        .compile("mlx_fft");

    println!(
        "cargo:rustc-link-search=native={}",
        prefix.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=jaccl");

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=c++");
}

#[cfg(feature = "mlx")]
fn find_mlx_c_prefix() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Ok(p) = std::env::var("MLX_C_PREFIX") {
        let path = PathBuf::from(p);
        if is_valid_mlx_prefix(&path) {
            return Some(path);
        }
    }

    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/mlx-c-install"),
        "/opt/homebrew".to_string(),
        "/usr/local".to_string(),
    ];

    for candidate in &candidates {
        let path = PathBuf::from(candidate);
        if is_valid_mlx_prefix(&path) {
            return Some(path);
        }
    }

    None
}

#[cfg(feature = "mlx")]
fn is_valid_mlx_prefix(prefix: &std::path::Path) -> bool {
    prefix.join("lib/libmlxc.a").exists() && prefix.join("include/mlx/c/fft.h").exists()
}
