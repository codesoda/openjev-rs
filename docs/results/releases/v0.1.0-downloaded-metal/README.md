# v0.1.0 downloaded-release Metal acceptance evidence

This directory records the bounded acceptance of the installed macOS artifact downloaded from the private GitHub release [`v0.1.0`](https://github.com/codesoda/openjev-rs/releases/tag/v0.1.0). The immutable release source is tag `v0.1.0` at `bb23406606e423fb35f5e62fdf5f170a6b14ad3f`. These captures and this explanation were produced afterward as a separate documentation-only follow-up on `main`; they are not part of the immutable release tag. Independent Astra final acceptance review passed with no release blockers, and parent verified the documentation against the captures.

## Provenance and identity

- Main CI: [run 35493609481](https://github.com/codesoda/openjev-rs/actions/runs/35493609481), successful on the exact source SHA (`main-ci-run.json`).
- Tagged CI/publication: [run 35494477837](https://github.com/codesoda/openjev-rs/actions/runs/35494477837), successful native macOS/Linux and publication jobs (`ci-run.json`).
- Release metadata and GitHub asset digests: `release.json`.
- Exact acquisition command (`download-verification.json`):

  ```sh
  gh release download v0.1.0 --repo codesoda/openjev-rs \
    --dir /Users/chrisraethke/.cache/openjev/releases/v0.1.0
  ```

- Downloaded archive SHA-256 values:
  - macOS: `b2b26ee4ed33b584f01b22b5ebec5d38745c4cd53a59e2e9c8fd303031896251`
  - Linux: `4430be0d48e77248b3e170bcb572f521f794fb5a39ea34e99f6285be37a9b8e6`
  - `SHA256SUMS`: `3cf1f0d65b5379af1cce4c01c27dcc640331a3cb1ff5296431fa609acec34967`
- Installed macOS binary SHA-256: `b9999f65f936fdd17e193af90e57bd568c2888f98889b15c3ce98d1a319fc66a`, matching the packaged `BUILD-INFO.json` (`artifact-identity.json`, `installation.json`).
- Install root: `~/.local/share/openjev/releases/v0.1.0/openjev-v0.1.0-aarch64-apple-darwin/`; `~/.local/bin/openjev` resolves to its unchanged executable. No pre-existing binary was replaced.

`download-verification.json` records matching `SHA256SUMS` and GitHub digests, safe archive verification, the macOS linkage check, and Linux `--skip-execute` on the local Mac. `installed-linkage.txt` records the installed Mach-O arm64 identity and only system framework/library dependencies. Hosted Linux CI—not the local Mac—ran Linux help/version/linkage/package checks. No Linux model inference was run.

## Runtime commands and checks

`smoke.py` is the retained acceptance harness. `command-manifest.json` records the exact installed-binary `--version`, `--help`, resident server, `npm ci`, and official SDK smoke argv/cwd/environment shape, with the ephemeral bearer secret redacted. The harness removed DYLD/GGML/library-path overrides, ran from a temporary directory outside the checkout, selected the existing verified model cache with `--offline`, and forced `--device metal`.

The smoke started at `2026-09-20T06:37:12Z` and passed:

- eight raw HTTP checks: health, readiness, three equal decoded JSON mixed Choice/Noul/Score response bodies, wrong bearer 401, unknown model 404, and unsupported float 422 (`http-responses.json`);
- one resident PID and one pinned Qwen3-0.6B load, embedded Metal library, `MTL0 (Apple M3 Pro)`, empty stdout, and SIGTERM exit 0 (`summary.json`, `server.stderr.txt`, `server.stdout.txt`);
- raw mixed requests: 313 input tokens, 0 output tokens;
- official `@typesafe-ai/sdk` 0.6.0 smoke: passed with 338 input tokens, 0 output tokens (`sdk-result.json`, `sdk.stdout.txt`);
- explicit `requested=shared; effective=serial` and serial full-prompt fallback disclosure on mixed requests;
- smoke process exited and temporary `node_modules` was removed.

## Evidence integrity

`EVIDENCE.sha256` covers all 28 raw evidence files. It intentionally excludes this explanatory README and the checksum manifest itself. Verify it with:

```sh
cd docs/results/releases/v0.1.0-downloaded-metal
shasum -a 256 -c EVIDENCE.sha256
```

All 28 entries were independently verified. A credential-pattern scan was clean. Empty stdout/stderr captures are intentionally retained and hash to the standard empty-file SHA-256.

Key file groups:

- CI/release/download: `main-ci-run.json`, `ci-run.json`, `release.json`, `download-verification.json`, `SHA256SUMS`.
- Package/install/linkage: `BUILD-INFO.json`, `macos-verify.json`, `linux-verify.json`, `installed-linkage.txt`, `installation.json`, `artifact-identity.json`.
- Runtime: `smoke.py`, `command-manifest.json`, `summary.json`, `http-responses.json`, version/help/server/npm/SDK captures.
- Final local gates: `local-clippy.log`, `local-tests.log`; parent also reported passing fmt, seven Python packaging/native-check tests, the license/source checker, and `actionlint`.

## Scope limits

This proves release/source identity, archive and payload integrity, macOS system linkage, user-local installation, one real installed-release offline Qwen Metal resident HTTP smoke, and the pinned SDK subset. It does **not** prove acceleration, latency or throughput, shared/batch numerical parity, Linux model inference, every model/OS combination, hosted Jev parity, Apple Developer ID signing/notarization, M6 completion, M7 completion, or completion of the full implementation brief. Shared execution remained explicit serial fallback. No model was downloaded during this acceptance run.
