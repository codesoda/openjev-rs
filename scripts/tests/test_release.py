#!/usr/bin/env python3

import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_SCRIPT = REPO_ROOT / "scripts" / "release.py"
SOURCE_SHA = "0123456789abcdef0123456789abcdef01234567"
EPOCH = "1700000000"
_VERSION_MATCH = re.search(
    r'(?ms)^\[workspace\.package\]\s*$.*?^version\s*=\s*"([^"]+)"',
    (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"),
)
if _VERSION_MATCH is None:
    raise RuntimeError("workspace package version not found")
WORKSPACE_VERSION = _VERSION_MATCH.group(1)


class ReleasePackagingTests(unittest.TestCase):
    def run_release(self, *arguments, expect=0):
        completed = subprocess.run(
            [sys.executable, str(RELEASE_SCRIPT), "--repo-root", str(REPO_ROOT)] + list(arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, expect, completed.stderr)
        return completed

    def fake_binary(self, directory, version=WORKSPACE_VERSION):
        binary = directory / "openjev"
        binary.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  --version) printf '%s\\n' '{\"schema\":\"openjev-version-v1\",\"version\":\"__VERSION__\",\"build\":\"test-native\"}' ;;\n"
            "  --help) printf '%s\\n' '{\"schema\":\"openjev-help-v1\",\"command\":\"openjev\",\"usage\":\"Usage: openjev\",\"text\":\"test\"}' ;;\n"
            "  *) exit 2 ;;\n"
            "esac\n".replace("__VERSION__", version),
            encoding="utf-8",
        )
        binary.chmod(0o755)
        return binary

    def package(self, binary, output, source_ref="refs/heads/main"):
        result = self.run_release(
            "package",
            "--binary",
            str(binary),
            "--target",
            "aarch64-apple-darwin",
            "--source-sha",
            SOURCE_SHA,
            "--source-ref",
            source_ref,
            "--source-date-epoch",
            EPOCH,
            "--rust-version",
            "rustc 1.95.0 (test)",
            "--workflow-run-url",
            "https://github.example/actions/runs/1",
            "--output-dir",
            str(output),
        )
        return Path(result.stdout.strip())

    def test_fake_binary_package_is_reproducible_and_smokes_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = self.fake_binary(root)
            first = self.package(binary, root / "first")
            second = self.package(binary, root / "second")
            self.assertEqual(hashlib.sha256(first.read_bytes()).digest(), hashlib.sha256(second.read_bytes()).digest())

            verified = self.run_release(
                "verify",
                "--archive",
                str(first),
                "--expected-version",
                WORKSPACE_VERSION,
                "--expected-target",
                "aarch64-apple-darwin",
                "--expected-source-sha",
                SOURCE_SHA,
            )
            self.assertIn(
                "verified openjev-v%s-aarch64-apple-darwin.tar.gz" % WORKSPACE_VERSION,
                verified.stdout,
            )

            with tarfile.open(str(first), "r:gz") as archive:
                names = archive.getnames()
                self.assertEqual(
                    names,
                    [
                        "openjev-v%s-aarch64-apple-darwin" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/openjev" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/LICENSE" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/THIRD_PARTY.md" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/THIRD_PARTY_LICENSES.html" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/RUST-COPYRIGHT-library.html" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/colored-3.1.1.crate" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/option-ext-0.2.0.crate" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/README.md" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/SERVE.md" % WORKSPACE_VERSION,
                        "openjev-v%s-aarch64-apple-darwin/BUILD-INFO.json" % WORKSPACE_VERSION,
                    ],
                )
                build_info = json.load(archive.extractfile(names[-1]))
                self.assertEqual(build_info["source_sha"], SOURCE_SHA)
                self.assertIsNone(build_info["tag"])
                self.assertFalse(build_info["distribution"]["contains_model_weights"])
                self.assertEqual(build_info["backend"]["llama_cpp_sys_crate"], "0.1.156")
                self.assertIn("THIRD_PARTY_LICENSES.html", build_info["files_sha256"])
                notice_member = "openjev-v%s-aarch64-apple-darwin/THIRD_PARTY_LICENSES.html" % WORKSPACE_VERSION
                notices = archive.extractfile(notice_member).read().decode("utf-8")
                self.assertIn("Copyright (c) Tokio Contributors", notices)
                self.assertIn("Copyright 2024 Mozilla Foundation", notices)
                rust_notice = "RUST-COPYRIGHT-library.html"
                rust_source = REPO_ROOT / "licenses" / rust_notice
                rust_member = "openjev-v%s-aarch64-apple-darwin/%s" % (WORKSPACE_VERSION, rust_notice)
                packaged_rust_notice = archive.extractfile(rust_member).read()
                rust_notice_hash = "90567e2718bf7fd65a71a3a43c5596488e80e5f51ed02bfea6fec54458b5f3d1"
                self.assertEqual(packaged_rust_notice, rust_source.read_bytes())
                self.assertEqual(hashlib.sha256(packaged_rust_notice).hexdigest(), rust_notice_hash)
                self.assertEqual(build_info["files_sha256"][rust_notice], rust_notice_hash)
                for filename, expected_hash in {
                    "colored-3.1.1.crate": "faf9468729b8cbcea668e36183cb69d317348c2e08e994829fb56ebfdfbaac34",
                    "option-ext-0.2.0.crate": "04744f49eae99ab78e0d5c0b603ab218f515ea8cfe5a456d7629ad883a3b6e7d",
                }.items():
                    source = REPO_ROOT / "licenses" / "sources" / filename
                    member = "openjev-v%s-aarch64-apple-darwin/%s" % (WORKSPACE_VERSION, filename)
                    packaged = archive.extractfile(member).read()
                    self.assertEqual(packaged, source.read_bytes())
                    self.assertEqual(hashlib.sha256(packaged).hexdigest(), expected_hash)
                    self.assertEqual(build_info["files_sha256"][filename], expected_hash)

    def test_tag_and_binary_must_exactly_match_cargo_version(self):
        good_tag = "v" + WORKSPACE_VERSION
        good = self.run_release("validate-ref", "--ref", "refs/tags/" + good_tag, "--require-tag")
        self.assertEqual(good.stdout.strip(), good_tag)
        bad = self.run_release("validate-ref", "--ref", "refs/tags/v999.0.0", "--require-tag", expect=1)
        self.assertIn("does not match workspace version", bad.stderr)
        branch = self.run_release("validate-ref", "--ref", "refs/heads/main", "--require-tag", expect=1)
        self.assertIn("publication requires", branch.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_release(
                "package",
                "--binary",
                str(self.fake_binary(root, version="0.0.0-mismatch")),
                "--target",
                "aarch64-apple-darwin",
                "--source-sha",
                SOURCE_SHA,
                "--source-ref",
                "refs/heads/main",
                "--source-date-epoch",
                EPOCH,
                "--rust-version",
                "rustc 1.95.0 (test)",
                "--output-dir",
                str(root / "dist"),
                expect=1,
            )
            self.assertIn("binary version", result.stderr)

    def test_verify_rejects_source_mismatch_and_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.package(self.fake_binary(root), root / "dist")
            mismatch = self.run_release(
                "verify",
                "--archive",
                str(archive),
                "--expected-source-sha",
                "f" * 40,
                "--skip-execute",
                expect=1,
            )
            self.assertIn("source SHA does not match", mismatch.stderr)

            evil = root / "evil.tar.gz"
            with tarfile.open(str(evil), "w:gz") as output:
                info = tarfile.TarInfo("../outside")
                payload = b"bad"
                info.size = len(payload)
                output.addfile(info, io.BytesIO(payload))
            rejected = self.run_release("verify", "--archive", str(evil), "--skip-execute", expect=1)
            self.assertIn("unsafe archive member", rejected.stderr)
            self.assertFalse((root.parent / "outside").exists())

    def test_checksums_cover_exactly_both_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mac = root / ("openjev-v%s-aarch64-apple-darwin.tar.gz" % WORKSPACE_VERSION)
            linux = root / ("openjev-v%s-x86_64-unknown-linux-gnu.tar.gz" % WORKSPACE_VERSION)
            mac.write_bytes(b"mac")
            linux.write_bytes(b"linux")
            output = root / "SHA256SUMS"
            self.run_release("checksums", "--output", str(output), str(linux), str(mac))
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].endswith("  openjev-v%s-aarch64-apple-darwin.tar.gz" % WORKSPACE_VERSION))
            self.assertTrue(lines[1].endswith("  openjev-v%s-x86_64-unknown-linux-gnu.tar.gz" % WORKSPACE_VERSION))


if __name__ == "__main__":
    unittest.main()
