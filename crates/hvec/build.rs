//! Shares the macOS SDK probe with `hvec-connect`; see that file for why the
//! binary has to repeat it.
#[path = "../hvec-connect/build/macos_sdk.rs"]
mod macos_sdk;

fn main() {
    println!("cargo:rerun-if-changed=../hvec-connect/build/macos_sdk.rs");
    macos_sdk::apply();
}
