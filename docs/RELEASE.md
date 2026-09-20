# openjev binary releases

Tagged releases contain the `openjev` executable, documentation and notices, plus the two MPL-2.0 covered-source archives described below. They do **not** contain GGUF model weights, API keys, cache contents, or probe receipts. Model artifacts remain separately downloaded into the verified openjev cache.

## Supported archives

| Archive target | Build/runtime contract |
| --- | --- |
| `aarch64-apple-darwin` | Apple Silicon, macOS 14.0 or newer, Metal enabled, embedded Metal library |
| `x86_64-unknown-linux-gnu` | x86-64 baseline CPU, glibc 2.35 or newer, system `libstdc++` and `libgcc` |

The Linux archive is a GNU/glibc build, not a static-musl portability claim. The macOS binary is not Developer ID signed or Apple notarized. Both builds use statically linked bundled llama.cpp/ggml libraries but retain normal operating-system shared-library dependencies.

Running `openjev` does not require Python, CMake, a compiler, Xcode, or Homebrew. Those tools may be used only while building or validating an archive in CI.

## Verify and install

Download the archive for your platform and the release's `SHA256SUMS` from the same GitHub Release. Verify the archive bytes before extraction:

```sh
tag=v0.1.0
target=aarch64-apple-darwin # or x86_64-unknown-linux-gnu
archive="openjev-${tag}-${target}.tar.gz"

# Linux
grep -F "  $archive" SHA256SUMS | sha256sum --check -

# macOS
grep -F "  $archive" SHA256SUMS | shasum -a 256 --check -
```

List the archive before extracting it. A valid archive has one versioned root directory, no absolute or `..` paths, and exactly these files:

```text
openjev
LICENSE
THIRD_PARTY.md
THIRD_PARTY_LICENSES.html
RUST-COPYRIGHT-library.html
colored-3.1.1.crate
option-ext-0.2.0.crate
README.md
SERVE.md
BUILD-INFO.json
```

`THIRD_PARTY_LICENSES.html` contains direct license/copyright texts for the pinned Cargo dependency closures and the bundled native llama.cpp/ggml sources. `RUST-COPYRIGHT-library.html` is the unmodified official Rust 1.95.0 compiler-payload notice for the standard library and compiled runtime. `colored-3.1.1.crate` and `option-ext-0.2.0.crate` are complete, unmodified original crates.io source archives for the MPL-2.0 dependencies. Recipients may redistribute them under the MPL-2.0 terms included in each crate archive and reproduced in the notice bundle. `BUILD-INFO.json` records the Cargo version, tag/ref, source commit, workflow run URL, target, Rust/native versions, system requirements, and SHA-256 hashes of every packaged payload file, including the Rust notice and both source archives. The source archive hashes are also checked against the exact package checksums in `Cargo.lock` before release packaging.

A user-local installation needs no `sudo`:

```sh
root="openjev-${tag}-${target}"

mkdir -p "$HOME/.local/share/openjev/releases/$tag" "$HOME/.local/bin"
tar -xzf "$archive" -C "$HOME/.local/share/openjev/releases/$tag"
ln -sfn "$HOME/.local/share/openjev/releases/$tag/$root/openjev" \
  "$HOME/.local/bin/openjev"
"$HOME/.local/bin/openjev" --version
```

Before replacing an existing `~/.local/bin/openjev`, inspect it and back it up; do not overwrite an unrelated file or symlink. Add `~/.local/bin` to `PATH` if needed.

## Runtime use

The CLI emits machine-readable JSON/JSONL on stdout and diagnostics on stderr. See the project README for decision commands and `SERVE.md` for the resident loopback HTTP API. A first inference requires a separately verified model cache; release archives never redistribute model weights.

The release workflow checks formatting, warnings-as-errors clippy, workspace tests with native/integration surfaces compiled but model integration disabled, the checked notice bundle against the pinned `Cargo.lock`, native release builds, archive contents, executable help/version JSON, and dynamic linkage. Actual downloaded-release offline model/HTTP/official-SDK acceptance is a separate post-publication gate and must not be inferred from packaging CI alone.
