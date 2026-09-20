#!/usr/bin/env python3
"""Fail closed when the checked-in release license bundle is stale or incomplete."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "THIRD_PARTY_LICENSES.html"
METADATA = REPO_ROOT / "licenses" / "THIRD_PARTY_LICENSES.metadata.json"
GENERATOR = REPO_ROOT / "scripts" / "generate_third_party_licenses.py"
RELEASE = REPO_ROOT / "scripts" / "release.py"
SOURCE_DIRECTORY = REPO_ROOT / "licenses" / "sources"
RUST_NOTICE = REPO_ROOT / "licenses" / "RUST-COPYRIGHT-library.html"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-release.yml"
REQUIRED_TEXT = (
    "Copyright (c) Tokio Contributors",
    "Copyright (c) Dial AI",
    "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.",
    "Copyright (c) 1998-2011 The OpenSSL Project. All rights reserved.",
    "Copyright (c) 2023-2026 The ggml authors",
    "Copyright 2024 Mozilla Foundation",
    "Copyright (c) 2026 Yuji Hirose",
    "SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann",
    "colored 3.1.1",
    "option-ext 0.2.0",
    "Mozilla Public License Version 2.0",
    "sheredom/subprocess.h bundled by llama.cpp common",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message):
    raise RuntimeError(message)


def cargo_lock_checksum(package, version):
    lock_text = (REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8")
    matches = []
    for block in re.split(r"(?m)^\[\[package\]\]\s*$", lock_text)[1:]:
        fields = {}
        for key in ("name", "version", "source", "checksum"):
            match = re.search(r'^%s = "([^"]+)"$' % key, block, re.MULTILINE)
            if match is not None:
                fields[key] = match.group(1)
        if fields.get("name") == package and fields.get("version") == version:
            matches.append(fields)
    if len(matches) != 1:
        fail("expected exactly one Cargo.lock package for %s %s" % (package, version))
    entry = matches[0]
    if not entry.get("source", "").startswith("registry+"):
        fail("covered source package is not registry-backed: %s %s" % (package, version))
    checksum = entry.get("checksum", "")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        fail("covered source package has no valid Cargo.lock checksum: %s %s" % (package, version))
    return checksum


def main():
    if not BUNDLE.is_file() or not METADATA.is_file() or not GENERATOR.is_file():
        fail("license bundle, metadata, or generator is missing")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("schema") != "openjev-third-party-licenses-v1":
        fail("unknown license metadata schema")
    if metadata.get("bundle") != BUNDLE.name or metadata.get("bundle_sha256") != sha256(BUNDLE):
        fail("THIRD_PARTY_LICENSES.html does not match its metadata")
    lock_hash = sha256(REPO_ROOT / "Cargo.lock")
    if metadata.get("cargo_lock_sha256") != lock_hash:
        fail("Cargo.lock changed; regenerate THIRD_PARTY_LICENSES.html")

    spec = importlib.util.spec_from_file_location("openjev_license_generator", GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    release_spec = importlib.util.spec_from_file_location("openjev_release", RELEASE)
    release = importlib.util.module_from_spec(release_spec)
    release_spec.loader.exec_module(release)
    if metadata.get("cargo_about_version") != generator.CARGO_ABOUT_VERSION:
        fail("cargo-about version metadata does not match the generator")
    expected_native_hashes = {name: value[1] for name, value in sorted(generator.NATIVE_SOURCES.items())}
    if metadata.get("native_source_sha256") != expected_native_hashes:
        fail("native source hashes do not match the generator")
    if metadata.get("llama_cpp_commit") != generator.LLAMA_CPP_COMMIT:
        fail("native source pin does not match the generator")
    if metadata.get("source_archives") != generator.SOURCE_ARCHIVES:
        fail("covered source archive metadata does not match the generator")
    expected_rust_notice = {
        "commit_hash": generator.RUST_COMMIT_HASH,
        "installer_component": "rustc",
        "installer_manifest_entry": generator.RUST_NOTICE_MANIFEST_ENTRY,
        "notice": str(RUST_NOTICE.relative_to(REPO_ROOT)),
        "notice_sha256": generator.RUST_NOTICE_SHA256,
        "provenance_command": "rustc -vV",
        "release": generator.RUST_RELEASE,
    }
    if metadata.get("rust_library_notice") != expected_rust_notice:
        fail("Rust library notice metadata does not match the pinned compiler")
    if not RUST_NOTICE.is_file() or sha256(RUST_NOTICE) != generator.RUST_NOTICE_SHA256:
        fail("Rust library notice does not match the reviewed compiler payload")
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    workflow_toolchains = []
    for line in workflow_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("RUSTUP_TOOLCHAIN:"):
            workflow_toolchains.append(stripped.partition(":")[2].strip().strip("\"'"))
    if workflow_toolchains != [generator.RUST_RELEASE]:
        fail("release workflow RUSTUP_TOOLCHAIN does not match the Rust notice compiler")
    for filename, source in sorted(generator.SOURCE_ARCHIVES.items()):
        archive = SOURCE_DIRECTORY / filename
        if not archive.is_file():
            fail("covered source archive is missing: %s" % archive.relative_to(REPO_ROOT))
        lock_checksum = cargo_lock_checksum(source["package"], source["version"])
        if source["sha256"] != lock_checksum:
            fail("covered source metadata does not match Cargo.lock for %s %s" % (source["package"], source["version"]))
        if sha256(archive) != lock_checksum:
            fail("covered source archive does not match Cargo.lock: %s" % archive.relative_to(REPO_ROOT))

    release_script = RELEASE.read_text(encoding="utf-8")
    pin_match = re.search(r'^LLAMA_CPP_COMMIT = "([0-9a-f]{40})"$', release_script, re.MULTILINE)
    if pin_match is None or pin_match.group(1) != metadata.get("llama_cpp_commit"):
        fail("release backend pin does not match the license bundle")
    if "THIRD_PARTY_LICENSES.html" not in release.MEMBER_FILES:
        fail("release archive does not include THIRD_PARTY_LICENSES.html")
    if "RUST-COPYRIGHT-library.html" not in release.MEMBER_FILES:
        fail("release archive does not include the Rust library notice")
    if release.RUST_NOTICE_PATH != ("licenses", "RUST-COPYRIGHT-library.html"):
        fail("release Rust library notice path does not match license metadata")
    expected_release_paths = {
        filename: ("licenses", "sources", filename) for filename in generator.SOURCE_ARCHIVES
    }
    if release.SOURCE_ARCHIVE_PATHS != expected_release_paths:
        fail("release covered-source mapping does not match license metadata")
    if not set(generator.SOURCE_ARCHIVES).issubset(release.MEMBER_FILES):
        fail("release member list omits a covered source archive")

    bundle_text = BUNDLE.read_text(encoding="utf-8")
    for required in REQUIRED_TEXT:
        if required not in bundle_text:
            fail("required attributable notice is missing: %s" % required)
    if len(BUNDLE.read_bytes()) > 3 * 1024 * 1024:
        fail("license bundle unexpectedly exceeds 3 MiB")
    print(
        "verified THIRD_PARTY_LICENSES.html sha256=%s Rust-notice=%s Cargo.lock=%s packages=%s source_archives=%s"
        % (
            metadata["bundle_sha256"],
            metadata["rust_library_notice"]["notice_sha256"],
            lock_hash,
            metadata.get("package_count"),
            len(generator.SOURCE_ARCHIVES),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print("check_third_party_licenses.py: error: %s" % error, file=sys.stderr)
        sys.exit(1)
