# openjev binary releases

Tagged releases contain the `openjev` executable, documentation and notices, plus the two MPL-2.0 covered-source archives described below. They do **not** contain GGUF model weights, API keys, cache contents, or probe receipts. Model artifacts remain separately downloaded into the verified openjev cache.

## Supported archives

| Archive target | Build/runtime contract |
| --- | --- |
| `aarch64-apple-darwin` | Apple Silicon, macOS 14.0 or newer, Metal enabled, embedded Metal library |
| `x86_64-unknown-linux-gnu` | x86-64 baseline CPU, glibc 2.35 or newer, system `libstdc++` and `libgcc` |

The Linux archive is a GNU/glibc build, not a static-musl portability claim. The macOS binary is not Developer ID signed or Apple notarized. Both builds use statically linked bundled llama.cpp/ggml libraries but retain normal operating-system shared-library dependencies.

Running `openjev` does not require Python, CMake, a compiler, Xcode, or Homebrew. Those tools may be used only while building or validating an archive in CI.

## Automatic installation

The root [`install.sh`](../install.sh) installs prebuilt release artifacts; it does not build from source or download models. For public GitHub access:

```sh
curl -fsSL https://raw.githubusercontent.com/codesoda/openjev-rs/main/install.sh | sh
```

The repository is currently private. Use an authenticated GitHub CLI account with access:

```sh
gh api --hostname github.com -H 'Accept: application/vnd.github.raw' \
  repos/codesoda/openjev-rs/contents/install.sh | sh
```

Run `gh auth login` first if needed. Append `-s -- --version v0.1.0` to `sh` to select an exact release; the default is latest. The installer uses authenticated `gh` when available, otherwise public HTTPS downloads through `curl`. Installation requires neither Python nor `jq`.

Payloads are retained at `~/.openjev/bin/openjev-v<VERSION>-<TARGET>/`, including all notices and covered-source archives. `~/.openjev/bin/openjev` selects the active executable, and `~/.local/bin/openjev` points to that stable path. Managed symlinks are updated on upgrade; unrelated files/links are refused. The earlier `~/.local/share/openjev/releases/` installation layout can be migrated without deleting its payload.

The installer verifies the selected archive against `SHA256SUMS` before extraction, validates the allowed archive contents, and checks the executable's version before activation. On macOS it removes `com.apple.quarantine` from the verified downloaded executable with `xattr -d` before executing it. An absent quarantine attribute is normal; failure to remove an existing attribute is an error. This does not sign/notarize the executable or disable Gatekeeper globally. No model cache or shell startup files are modified.

## Manual verification and installation

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

For a **new** installation, after verifying and inspecting the archive, install without `sudo`:

```sh
(
  set -eu
  root="openjev-${tag}-${target}"
  destination="$HOME/.openjev/bin"
  for path in "$destination/$root" "$destination/openjev" "$HOME/.local/bin/openjev"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      echo "Already exists: $path; use the installer for a managed upgrade." >&2
      exit 1
    fi
  done
  mkdir -p "$destination" "$HOME/.local/bin"
  tar -xzf "$archive" -C "$destination"
  if [ "$(uname -s)" = Darwin ]; then
    if xattr "$destination/$root/openjev" | grep -qx com.apple.quarantine; then
      xattr -d com.apple.quarantine "$destination/$root/openjev"
    fi
  fi
  "$destination/$root/openjev" --version
  ln -s "$destination/$root/openjev" "$destination/openjev"
  ln -s "$destination/openjev" "$HOME/.local/bin/openjev"
)
```

Before replacing an existing installation, inspect and back it up; never overwrite an unrelated file or symlink. Add `~/.local/bin` to `PATH` if needed. Prefer `install.sh` for archive-safety checks and managed upgrades.

## Runtime use

The CLI emits machine-readable JSON/JSONL on stdout and diagnostics on stderr. See the project README for decision commands and `SERVE.md` for the resident loopback HTTP API. A first inference requires a separately verified model cache; release archives never redistribute model weights.

The release workflow checks formatting, warnings-as-errors clippy, workspace tests with native/integration surfaces compiled but model integration disabled, the checked notice bundle against the pinned `Cargo.lock`, native release builds, archive contents, executable help/version JSON, and dynamic linkage. Actual downloaded-release offline model/HTTP/official-SDK acceptance is a separate post-publication gate and must not be inferred from packaging CI alone.
