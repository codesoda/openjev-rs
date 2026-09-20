#!/usr/bin/env python3
"""Generate the checked-in release dependency notice bundle.

This intentionally runs cargo-about once per distributed target and adds native
sources that Cargo metadata cannot see. Regeneration may access crates.io,
ClearlyDefined, GitHub, and the exact pinned native source URLs below.
"""

import hashlib
import html
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "THIRD_PARTY_LICENSES.html"
METADATA = REPO_ROOT / "licenses" / "THIRD_PARTY_LICENSES.metadata.json"
RUST_NOTICE = REPO_ROOT / "licenses" / "RUST-COPYRIGHT-library.html"
CARGO_ABOUT_VERSION = "0.9.2"
RUST_RELEASE = "1.95.0"
RUST_COMMIT_HASH = "59807616e1fa2540724bfbac14d7976d7e4a3860"
RUST_NOTICE_SHA256 = "90567e2718bf7fd65a71a3a43c5596488e80e5f51ed02bfea6fec54458b5f3d1"
RUST_NOTICE_MANIFEST_ENTRY = "file:share/doc/rust/COPYRIGHT-library.html"
FEATURES = "openjev-cli/metal openjev-cli/native"
TARGETS = {
    "aarch64-apple-darwin": "macOS Metal release",
    "x86_64-unknown-linux-gnu": "Linux native release",
}
LLAMA_CPP_COMMIT = "e79e4bf660e19f2ad851e06c6913f7a8c5852621"
SOURCE_ARCHIVES = {
    "colored-3.1.1.crate": {
        "package": "colored",
        "version": "3.1.1",
        "sha256": "faf9468729b8cbcea668e36183cb69d317348c2e08e994829fb56ebfdfbaac34",
    },
    "option-ext-0.2.0.crate": {
        "package": "option-ext",
        "version": "0.2.0",
        "sha256": "04744f49eae99ab78e0d5c0b603ab218f515ea8cfe5a456d7629ad883a3b6e7d",
    },
}

# Whole-file hashes make extraction from the exact reviewed sources fail closed.
NATIVE_SOURCES = {
    "aws_lc_license": (
        "https://raw.githubusercontent.com/aws/aws-lc-rs/7943223c99d909bc399bdf1b856821bb04f1f3c5/aws-lc-sys/LICENSE",
        "728536b4160e051f86d7c9c388f704866b3d512cd7df97ac3516c65279523c4e",
    ),
    "llama_license": (
        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_COMMIT}/LICENSE",
        "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d",
    ),
    "cpp_httplib_license": (
        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_COMMIT}/vendor/cpp-httplib/LICENSE",
        "4b45cbe16d7b71b89ae6127e26e0d90a029198ca5e958ad8e3d0b8bbed364d8b",
    ),
    "cpp_httplib_header": (
        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_COMMIT}/vendor/cpp-httplib/httplib.h",
        "98cb1849d1eaa547c18af978f3b238720d6d2fb00967bdb8b6144dca8aa2f49e",
    ),
    "nlohmann_header": (
        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_COMMIT}/vendor/nlohmann/json.hpp",
        "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63",
    ),
    "nlohmann_license": (
        "https://raw.githubusercontent.com/nlohmann/json/v3.12.0/LICENSE.MIT",
        "46a65cffd1ea955132d95a8dd921640714a8d6b537d2e4e482d31145ae95b603",
    ),
    "base64_header": (
        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_COMMIT}/common/base64.hpp",
        "57f595aa0a206c4dec9a84b90a3416028a242da4dd8f219afc0859a6ccb7efe7",
    ),
    "subprocess_header": (
        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_COMMIT}/vendor/sheredom/subprocess.h",
        "13a70780fd107b4c6e7ca9af5738215718a18fb778f462f0ec47732b39899255",
    ),
    "llamafile_sgemm": (
        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_COMMIT}/ggml/src/ggml-cpu/llamafile/sgemm.cpp",
        "e226380b05393302ba91b38d22718d9521700ee2cb7943f83ad316a7a9dcd597",
    ),
    "ggml_cpu_ops": (
        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_COMMIT}/ggml/src/ggml-cpu/ops.cpp",
        "dd7265a7402515002d4679f18d5d1d423456e87f256f7cc11af3bbb4148bf773",
    ),
    "ggml_metal": (
        f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_CPP_COMMIT}/ggml/src/ggml-metal/ggml-metal.metal",
        "5d577d20a699016d108b83d517fe49c716c635e4dee46a0538157542a8789130",
    ),
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def installed_rust_notice():
    rustc_vv = subprocess.run(
        ["rustc", "-vV"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    ).stdout
    fields = {}
    for line in rustc_vv.splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            fields[key] = value
    if fields.get("release") != RUST_RELEASE or fields.get("commit-hash") != RUST_COMMIT_HASH:
        raise RuntimeError(
            "expected rustc %s commit %s, got release %r commit %r"
            % (RUST_RELEASE, RUST_COMMIT_HASH, fields.get("release"), fields.get("commit-hash"))
        )
    host = fields.get("host")
    if not host:
        raise RuntimeError("rustc -vV did not report a host")
    sysroot = Path(
        subprocess.run(
            ["rustc", "--print", "sysroot"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.strip()
    )
    manifest = sysroot / "lib" / "rustlib" / ("manifest-rustc-" + host)
    if RUST_NOTICE_MANIFEST_ENTRY not in manifest.read_text(encoding="utf-8").splitlines():
        raise RuntimeError("Rust library notice is not part of the installed rustc component manifest")
    source = sysroot / "share" / "doc" / "rust" / "COPYRIGHT-library.html"
    notice = source.read_bytes()
    if sha256_bytes(notice) != RUST_NOTICE_SHA256:
        raise RuntimeError("installed Rust library notice does not match the reviewed Rust 1.95.0 file")
    provenance = {
        "commit_hash": fields["commit-hash"],
        "installer_component": "rustc",
        "installer_manifest_entry": RUST_NOTICE_MANIFEST_ENTRY,
        "notice": str(RUST_NOTICE.relative_to(REPO_ROOT)),
        "notice_sha256": RUST_NOTICE_SHA256,
        "provenance_command": "rustc -vV",
        "release": fields["release"],
    }
    return notice, provenance


def cargo_about_binary():
    override = os.environ.get("CARGO_ABOUT")
    if override:
        return override
    installed = Path.home() / ".cargo" / "bin" / "cargo-about"
    if installed.is_file():
        return str(installed)
    found = shutil.which("cargo-about")
    if found:
        return found
    raise RuntimeError("cargo-about 0.9.2 is required")


def cargo_licenses(binary, target):
    command = [
        binary,
        "-L",
        "warn",
        "generate",
        "--config",
        "about.toml",
        "--locked",
        "--workspace",
        "--features",
        FEATURES,
        "--target",
        target,
        "--format",
        "json",
        "--fail",
    ]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise RuntimeError("cargo-about failed for %s:\n%s" % (target, stderr))
    if stderr:
        raise RuntimeError("cargo-about emitted unresolved warnings for %s:\n%s" % (target, stderr))
    return json.loads(completed.stdout), shlex.join(command[3:])


def fetch_native_sources():
    sources = {}
    for name, (url, expected_hash) in NATIVE_SOURCES.items():
        request = Request(url, headers={"User-Agent": "openjev-license-generator/1"})
        with urlopen(request, timeout=30) as response:
            data = response.read()
        actual_hash = sha256_bytes(data)
        if actual_hash != expected_hash:
            raise RuntimeError("native license source hash mismatch for %s" % url)
        sources[name] = data.decode("utf-8")
    return sources


def block_comment(text, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = text.find("/*", start + 1)
        if start < 0:
            raise RuntimeError("expected native block comment was not found")
    end = text.find("*/", start + 2)
    if end < 0:
        raise RuntimeError("unterminated native block comment")
    return text[start + 2 : end].strip()


def line_comment_prefix(text, line_count):
    lines = text.splitlines()[:line_count]
    if len(lines) != line_count or any(not line.startswith("//") for line in lines):
        raise RuntimeError("expected native line comment changed")
    return "\n".join(line[2:].lstrip() for line in lines).strip()


def exact_notice_line(text, needle):
    matches = [line.strip() for line in text.splitlines() if needle in line]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one native notice containing %r" % needle)
    return matches[0].removeprefix("//").strip()


def native_notices(sources):
    cpp_header = line_comment_prefix(sources["cpp_httplib_header"], 6)
    nlohmann_header = line_comment_prefix(sources["nlohmann_header"], 7)
    subprocess_text = block_comment(sources["subprocess_header"], 0) + "\n\n" + block_comment(
        sources["subprocess_header"], 1
    )
    shared_ggml_notice = exact_notice_line(
        sources["ggml_cpu_ops"], "Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng"
    )
    metal_notice = exact_notice_line(
        sources["ggml_metal"], "Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng"
    )
    if shared_ggml_notice != metal_notice:
        raise RuntimeError("CPU and Metal embedded MIT notices diverged")
    return [
        {
            "name": "AWS-LC and compiled-in BoringSSL/OpenSSL/Fiat/s2n-bignum/Jitter Entropy sources",
            "license": "ISC, Apache-2.0, MIT, MIT-0, BSD-3-Clause, and CC0 notices as detailed below",
            "source": NATIVE_SOURCES["aws_lc_license"][0],
            "text": sources["aws_lc_license"].strip(),
        },
        {
            "name": "llama.cpp and ggml",
            "license": "MIT",
            "source": NATIVE_SOURCES["llama_license"][0],
            "text": sources["llama_license"].strip(),
        },
        {
            "name": "llamafile SGEMM embedded in ggml CPU",
            "license": "MIT",
            "source": NATIVE_SOURCES["llamafile_sgemm"][0],
            "text": line_comment_prefix(sources["llamafile_sgemm"], 21),
        },
        {
            "name": "ggml CPU and Metal adapted kernels",
            "license": "MIT (notice; terms in llama.cpp/ggml entry above)",
            "source": "%s and %s"
            % (NATIVE_SOURCES["ggml_cpu_ops"][0], NATIVE_SOURCES["ggml_metal"][0]),
            "text": shared_ggml_notice,
        },
        {
            "name": "cpp-httplib 0.53.0 bundled by llama.cpp common",
            "license": "MIT",
            "source": NATIVE_SOURCES["cpp_httplib_license"][0],
            "text": cpp_header + "\n\n" + sources["cpp_httplib_license"].strip(),
        },
        {
            "name": "JSON for Modern C++ 3.12.0 bundled by llama.cpp common",
            "license": "MIT",
            "source": "%s and %s"
            % (NATIVE_SOURCES["nlohmann_header"][0], NATIVE_SOURCES["nlohmann_license"][0]),
            "text": nlohmann_header + "\n\n" + sources["nlohmann_license"].strip(),
        },
        {
            "name": "base64.hpp bundled by llama.cpp common",
            "license": "Public domain dedication / Unlicense notice",
            "source": NATIVE_SOURCES["base64_header"][0],
            "text": block_comment(sources["base64_header"], 0),
        },
        {
            "name": "sheredom/subprocess.h bundled by llama.cpp common",
            "license": "Public domain dedication / Unlicense notice",
            "source": NATIVE_SOURCES["subprocess_header"][0],
            "text": subprocess_text,
        },
    ]


def merge_cargo_outputs(outputs):
    packages = {}
    licenses = {}
    for target, document in outputs.items():
        target_packages = {
            (entry["package"]["name"], entry["package"]["version"])
            for entry in document["crates"]
            if not entry["package"]["name"].startswith("openjev-")
        }
        used_packages = set()
        for license_entry in document["licenses"]:
            text = license_entry.get("text", "")
            if not text.strip():
                raise RuntimeError("empty license text for %s" % license_entry.get("id"))
            key = (license_entry["id"], text)
            merged = licenses.setdefault(
                key,
                {"id": license_entry["id"], "name": license_entry["name"], "text": text, "used_by": {}},
            )
            for used in license_entry["used_by"]:
                package = used["crate"]
                package_key = (package["name"], package["version"])
                if package["name"].startswith("openjev-"):
                    continue
                used_packages.add(package_key)
                packages.setdefault(package_key, package)
                merged["used_by"].setdefault(package_key, set()).add(target)
        missing = target_packages - used_packages
        if missing:
            raise RuntimeError("packages without license text for %s: %r" % (target, sorted(missing)))
    return packages, licenses


def render_html(outputs, commands, sources):
    lock_hash = sha256_bytes((REPO_ROOT / "Cargo.lock").read_bytes())
    packages, licenses = merge_cargo_outputs(outputs)
    target_labels = {target: label for target, label in TARGETS.items()}
    chunks = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>openjev third-party licenses</title>",
        "<style>body{font:15px/1.45 system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#222}",
        "pre{white-space:pre-wrap;border:1px solid #ccc;background:#f7f7f7;padding:1rem;overflow:auto}",
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.35rem;text-align:left}",
        "code{overflow-wrap:anywhere}.review{border-left:5px solid #a65d00;padding:.5rem 1rem;background:#fff7e6}</style></head><body>",
        "<h1>openjev third-party licenses and notices</h1>",
        "<p>This file accompanies the distributed openjev CLI. It contains license texts and attributable notices for the Rust dependency closure of both release targets, plus native llama.cpp/ggml and bundled sources that Cargo metadata cannot enumerate.</p>",
        '<div class="review"><strong>Release review note:</strong> <code>colored 3.1.1</code> and <code>option-ext 0.2.0</code> are MPL-2.0-only transitive dependencies through <code>hf-hub</code>/<code>hf-xet</code>. Their complete MPL-2.0 terms appear below. Their corresponding source is available from their linked repositories and crates.io packages. This generated inclusion records the actual graph; it is not a blanket approval of future copyleft dependencies.</div>',
        "<h2>Generation record</h2>",
        "<ul><li>cargo-about version: <code>%s</code></li><li>Cargo.lock SHA-256: <code>%s</code></li><li>llama.cpp/ggml pin: <code>%s</code></li></ul>"
        % (CARGO_ABOUT_VERSION, lock_hash, LLAMA_CPP_COMMIT),
        "<p>Commands (run from the repository root):</p><pre>%s</pre>"
        % html.escape("\n".join(commands)),
        "<h2>Dependency inventory (%d packages)</h2><table><thead><tr><th>Package</th><th>Version</th><th>Repository</th></tr></thead><tbody>"
        % len(packages),
    ]
    for key, package in sorted(packages.items()):
        repository = package.get("repository") or ("https://crates.io/crates/" + package["name"])
        chunks.append(
            "<tr><td>%s</td><td>%s</td><td><a href=\"%s\">%s</a></td></tr>"
            % tuple(html.escape(value, quote=True) for value in (package["name"], package["version"], repository, repository))
        )
    chunks.append("</tbody></table><h2>Rust dependency license texts</h2>")
    for index, entry in enumerate(sorted(licenses.values(), key=lambda item: (item["id"], item["text"]))):
        chunks.append("<section><h3 id=\"rust-%d\">%s (%s)</h3><p>Used by:</p><ul>" % (index, html.escape(entry["name"]), html.escape(entry["id"])))
        for package_key, targets in sorted(entry["used_by"].items()):
            package = packages[package_key]
            repository = package.get("repository") or ("https://crates.io/crates/" + package["name"])
            target_text = ", ".join(target_labels[target] for target in sorted(targets))
            chunks.append(
                '<li><a href="%s">%s %s</a> — %s</li>'
                % tuple(
                    html.escape(value, quote=True)
                    for value in (repository, package["name"], package["version"], target_text)
                )
            )
        chunks.append("</ul><pre>%s</pre></section>" % html.escape(entry["text"]))
    chunks.append("<h2>Native llama.cpp/ggml and embedded-source notices</h2>")
    for index, notice in enumerate(native_notices(sources)):
        chunks.append(
            '<section><h3 id="native-%d">%s</h3><p>License: %s<br>Exact source: <code>%s</code></p><pre>%s</pre></section>'
            % tuple(
                [index]
                + [html.escape(notice[key], quote=True) for key in ("name", "license", "source", "text")]
            )
        )
    chunks.append("</body></html>\n")
    return "".join(chunks).encode("utf-8"), lock_hash, len(packages), len(licenses)


def main():
    binary = cargo_about_binary()
    version = subprocess.run(
        [binary, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    ).stdout.strip()
    if version != "cargo-about " + CARGO_ABOUT_VERSION:
        raise RuntimeError("expected cargo-about %s, got %s" % (CARGO_ABOUT_VERSION, version))

    outputs = {}
    commands = []
    for target in TARGETS:
        outputs[target], command = cargo_licenses(binary, target)
        commands.append("cargo about " + command)
    sources = fetch_native_sources()
    rendered, lock_hash, package_count, license_count = render_html(outputs, commands, sources)
    rust_notice, rust_provenance = installed_rust_notice()
    OUTPUT.write_bytes(rendered)
    RUST_NOTICE.write_bytes(rust_notice)

    metadata = {
        "schema": "openjev-third-party-licenses-v1",
        "bundle": OUTPUT.name,
        "bundle_sha256": sha256_bytes(rendered),
        "cargo_about_version": CARGO_ABOUT_VERSION,
        "cargo_lock_sha256": lock_hash,
        "features": FEATURES,
        "targets": list(TARGETS),
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "native_source_sha256": {name: value[1] for name, value in sorted(NATIVE_SOURCES.items())},
        "package_count": package_count,
        "license_text_variant_count": license_count,
        "rust_library_notice": rust_provenance,
        "source_archives": SOURCE_ARCHIVES,
    }
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s (%d bytes)" % (OUTPUT, len(rendered)))
    print("wrote %s (%d bytes)" % (RUST_NOTICE, len(rust_notice)))
    print("wrote %s" % METADATA)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as error:
        print("generate_third_party_licenses.py: error: %s" % error, file=sys.stderr)
        sys.exit(1)
