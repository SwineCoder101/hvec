//! macOS link fix for the local embedder.
//!
//! The prebuilt ONNX Runtime that `fastembed` links through `ort-sys` refers to
//! CoreML classes (`MLComputePlan`, `MLOptimizationHints`) that exist only in
//! the macOS 14.4 SDK and newer. A Command Line Tools install often ships an
//! older SDK while Xcode.app carries a current one, and the link step then
//! fails with undefined `_OBJC_CLASS_$_MLComputePlan` symbols.
//!
//! When the default SDK is too old and `SDKROOT` is unset, this looks for a
//! newer SDK inside Xcode.app and passes it to the linker with `-isysroot`.
//! Setting `SDKROOT` yourself always wins. Nothing happens off macOS or when
//! the `local` feature is off.
//!
//! Cargo applies a build script's link arguments only to that package's own
//! targets, so `hvec-connect` (for its tests) and `hvec` (for the binary) both
//! run this from their `build.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// First SDK that declares the CoreML classes ONNX Runtime needs.
const MIN_SDK: (u32, u32) = (14, 4);

const XCODE_SDKS: &[&str] = &[
    "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
    "/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
];

pub fn apply() {
    println!("cargo:rerun-if-env-changed=SDKROOT");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || std::env::var_os("CARGO_FEATURE_LOCAL").is_none()
        || std::env::var_os("SDKROOT").is_some_and(|v| !v.is_empty())
    {
        return;
    }
    let Some(default) = default_sdk_version() else {
        return;
    };
    if default >= MIN_SDK {
        return;
    }
    match XCODE_SDKS
        .iter()
        .map(Path::new)
        .find(|p| sdk_version(p).is_some_and(|v| v >= MIN_SDK))
    {
        Some(sdk) => {
            println!("cargo:rustc-link-arg=-isysroot");
            println!("cargo:rustc-link-arg={}", sdk.display());
        }
        None => println!(
            "cargo:warning=the default macOS SDK is {}.{} and ONNX Runtime needs {}.{} or newer; \
             set SDKROOT to a newer SDK or build with --no-default-features (see README, Troubleshooting)",
            default.0, default.1, MIN_SDK.0, MIN_SDK.1
        ),
    }
}

fn default_sdk_version() -> Option<(u32, u32)> {
    let out = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-version"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_version(String::from_utf8_lossy(&out.stdout).trim())
}

/// Reads `"Version":"26.5"` out of the SDK's `SDKSettings.json` without a JSON parser.
fn sdk_version(sdk: &Path) -> Option<(u32, u32)> {
    let settings = std::fs::read_to_string(PathBuf::from(sdk).join("SDKSettings.json")).ok()?;
    let rest = &settings[settings.find("\"Version\"")? + "\"Version\"".len()..];
    let start = rest.find('"')? + 1;
    let end = start + rest[start..].find('"')?;
    parse_version(&rest[start..end])
}

fn parse_version(s: &str) -> Option<(u32, u32)> {
    let mut it = s.split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}
