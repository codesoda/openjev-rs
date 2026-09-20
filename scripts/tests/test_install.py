#!/usr/bin/env python3
"""Deterministic, network-free tests for install.sh."""

import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "install.sh"
MEMBERS = (
    "openjev",
    "LICENSE",
    "THIRD_PARTY.md",
    "THIRD_PARTY_LICENSES.html",
    "RUST-COPYRIGHT-library.html",
    "colored-3.1.1.crate",
    "option-ext-0.2.0.crate",
    "README.md",
    "SERVE.md",
    "BUILD-INFO.json",
)
MAC_TARGET = "aarch64-apple-darwin"
LINUX_TARGET = "x86_64-unknown-linux-gnu"


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.home = self.root / "home"
        self.home.mkdir()
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir()
        self.stubs = self.root / "stubs"
        self.stubs.mkdir()
        self.tmp = self.root / "tmp"
        self.tmp.mkdir()
        self.log = self.root / "calls.log"
        self.events = self.root / "events.log"
        self._write_stubs()

    def tearDown(self):
        self.temporary.cleanup()

    def _stub(self, name, body):
        path = self.stubs / name
        path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
        path.chmod(0o755)

    def _write_stubs(self):
        self._stub(
            "uname",
            "case \"$1\" in\n"
            "  -s) printf '%s\\n' \"${TEST_OS:-Darwin}\" ;;\n"
            "  -m) printf '%s\\n' \"${TEST_ARCH:-arm64}\" ;;\n"
            "  *) exit 2 ;;\n"
            "esac\n",
        )
        self._stub("sw_vers", "printf '%s\\n' \"${TEST_MAC_VERSION:-14.4}\"\n")
        self._stub("getconf", "printf 'glibc %s\\n' \"${TEST_GLIBC:-2.35}\"\n")
        self._stub(
            "xattr",
            "printf 'xattr %s\\n' \"$*\" >>\"$CALL_LOG\"\n"
            "printf 'xattr\\n' >>\"$EVENT_LOG\"\n"
            "if [ \"${1:-}\" = -d ]; then\n"
            "  [ \"${XATTR_DELETE_FAIL:-0}\" != 1 ] || exit 9\n"
            "  exit 0\n"
            "fi\n"
            "[ \"${XATTR_LIST_FAIL:-0}\" != 1 ] || exit 8\n"
            "if [ -n \"${XATTR_HAS_MATCH:-}\" ]; then\n"
            "  case \"$1\" in *\"$XATTR_HAS_MATCH\"*) printf 'com.apple.quarantine\\n' ;; esac\n"
            "elif [ \"${XATTR_HAS:-0}\" = 1 ]; then\n"
            "  printf 'com.apple.quarantine\\n'\n"
            "fi\n",
        )
        self._stub(
            "curl",
            "printf 'curl %s\\n' \"$*\" >>\"$CALL_LOG\"\n"
            "output= url= latest=0\n"
            "while [ $# -gt 0 ]; do\n"
            "  case \"$1\" in\n"
            "    --output|-o) output=$2; shift 2 ;;\n"
            "    --write-out|-w) shift 2 ;;\n"
            "    http*) url=$1; shift ;;\n"
            "    *) shift ;;\n"
            "  esac\n"
            "done\n"
            "case \"$url\" in\n"
            "  */releases/latest) printf '%s' \"${LATEST_URL:-https://github.com/codesoda/openjev-rs/releases/tag/$LATEST_TAG}\"; exit 0 ;;\n"
            "esac\n"
            "case \"$url\" in *\"${CURL_FAIL_MATCH:-__never__}\"*) exit 22 ;; esac\n"
            "name=${url##*/}\n"
            "cp \"$FIXTURES/$LATEST_TAG/$name\" \"$output\"\n",
        )
        self._stub(
            "gh",
            "printf 'gh unexpectedly called\n' >>\"$CALL_LOG\"\nexit 99\n",
        )

    def make_release(
        self,
        tag="v0.1.0",
        target=MAC_TARGET,
        mutation=None,
        binary_mode="good",
        checksum_mode="good",
    ):
        release = self.fixtures / tag
        release.mkdir(parents=True, exist_ok=True)
        root = f"openjev-{tag}-{target}"
        archive = release / f"{root}.tar.gz"
        version = tag[1:]
        if binary_mode == "good":
            result = f'{{"schema":"openjev-version-v1","version":"{version}","build":"test"}}'
            exit_line = "exit 0"
        elif binary_mode == "wrong":
            result = '{"schema":"openjev-version-v1","version":"9.9.9","build":"test"}'
            exit_line = "exit 0"
        elif binary_mode == "bad-json":
            result = "not-json"
            exit_line = "exit 0"
        elif binary_mode == "staged-only":
            result = f'{{"schema":"openjev-version-v1","version":"{version}","build":"test"}}'
            exit_line = 'case "$0" in */.install.*) exit 0 ;; *) exit 7 ;; esac'
        else:
            result = ""
            exit_line = "exit 7"
        binary = (
            "#!/bin/sh\n"
            "printf 'version %s\\n' \"$0\" >>\"${EVENT_LOG:-/dev/null}\"\n"
            f"printf '%s\\n' '{result}'\n"
            f"{exit_line}\n"
        ).encode()
        data = {
            "openjev": binary,
            "LICENSE": b"license\n",
            "THIRD_PARTY.md": b"third party\n",
            "THIRD_PARTY_LICENSES.html": b"full notices\n",
            "RUST-COPYRIGHT-library.html": b"rust notices\n",
            "colored-3.1.1.crate": b"colored source\n",
            "option-ext-0.2.0.crate": b"option-ext source\n",
            "README.md": b"release readme\n",
            "SERVE.md": b"serve docs\n",
            "BUILD-INFO.json": b'{"schema":"openjev-build-info-v1"}\n',
        }
        with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as output:
            directory = tarfile.TarInfo(root + "/")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            output.addfile(directory)
            member_order = reversed(MEMBERS) if mutation == "reordered" else MEMBERS
            for name in member_order:
                if mutation in ("symlink", "hardlink") and name == "LICENSE":
                    info = tarfile.TarInfo(f"{root}/{name}")
                    info.type = tarfile.SYMTYPE if mutation == "symlink" else tarfile.LNKTYPE
                    info.linkname = f"{root}/openjev"
                    output.addfile(info)
                    continue
                info = tarfile.TarInfo(f"{root}/{name}")
                payload = data[name]
                info.size = len(payload)
                info.mode = 0o755 if name == "openjev" else 0o644
                output.addfile(info, io.BytesIO(payload))
                if mutation == "duplicate" and name == "LICENSE":
                    duplicate = tarfile.TarInfo(f"{root}/{name}")
                    duplicate.size = len(payload)
                    output.addfile(duplicate, io.BytesIO(payload))
            if mutation == "extra":
                info = tarfile.TarInfo(f"{root}/EXTRA")
                info.size = 1
                output.addfile(info, io.BytesIO(b"x"))
            elif mutation == "traversal":
                info = tarfile.TarInfo("../outside")
                info.size = 1
                output.addfile(info, io.BytesIO(b"x"))
            elif mutation == "absolute":
                info = tarfile.TarInfo("/absolute-outside")
                info.size = 1
                output.addfile(info, io.BytesIO(b"x"))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum = release / "SHA256SUMS"
        line = f"{digest}  {archive.name}\n"
        if checksum_mode == "wrong":
            line = f"{'0' * 64}  {archive.name}\n"
        elif checksum_mode == "missing":
            line = f"{digest}  another.tar.gz\n"
        elif checksum_mode == "duplicate":
            line = line + line
        elif checksum_mode == "malformed":
            line = f"xyz  {archive.name}\n"
        elif checksum_mode == "extra-field":
            line = f"{digest}  {archive.name} extra\n"
        elif checksum_mode == "valid-plus-malformed":
            line += f"xyz  {archive.name} extra\n"
        checksum.write_text(line, encoding="utf-8")
        return archive

    def run_installer(self, *arguments, expect=0, **environment):
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(self.home),
                "TMPDIR": str(self.tmp),
                "PATH": f"{self.stubs}:/usr/bin:/bin",
                "FIXTURES": str(self.fixtures),
                "LATEST_TAG": environment.pop("LATEST_TAG", "v0.1.0"),
                "CALL_LOG": str(self.log),
                "EVENT_LOG": str(self.events),
            }
        )
        env.update({key: str(value) for key, value in environment.items()})
        result = subprocess.run(
            ["/bin/sh", str(INSTALLER), *arguments],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, expect, result.stderr)
        self.assertEqual(result.stdout, "")
        return result

    def installed_root(self, tag="v0.1.0", target=MAC_TARGET):
        return self.home / ".openjev" / "bin" / f"openjev-{tag}-{target}"

    def assert_links(self, tag="v0.1.0", target=MAC_TARGET):
        payload = self.installed_root(tag, target)
        active = self.home / ".openjev" / "bin" / "openjev"
        command = self.home / ".local" / "bin" / "openjev"
        self.assertTrue(active.is_symlink())
        self.assertEqual(os.readlink(active), str(payload / "openjev"))
        self.assertTrue(command.is_symlink())
        self.assertEqual(os.readlink(command), str(active))

    def test_fresh_public_latest_installs_exact_payload_and_notices(self):
        archive = self.make_release()
        with tarfile.open(archive, "r:gz") as packaged:
            packaged_binary = packaged.extractfile(
                f"openjev-v0.1.0-{MAC_TARGET}/openjev"
            ).read()
        result = self.run_installer(XATTR_HAS="1")
        payload = self.installed_root()
        self.assertEqual({path.name for path in payload.iterdir()}, set(MEMBERS))
        self.assertEqual((payload / "openjev").read_bytes(), packaged_binary)
        self.assertEqual((payload / "THIRD_PARTY_LICENSES.html").read_bytes(), b"full notices\n")
        self.assertEqual((payload / "RUST-COPYRIGHT-library.html").read_bytes(), b"rust notices\n")
        self.assertEqual((payload / "colored-3.1.1.crate").read_bytes(), b"colored source\n")
        self.assertTrue(os.access(payload / "openjev", os.X_OK))
        self.assert_links()
        calls = self.log.read_text()
        self.assertIn("/releases/latest", calls)
        self.assertIn("--proto =https", calls)
        self.assertIn("--proto-redir =https", calls)
        self.assertIn("SHA256SUMS", calls)
        self.assertIn("is not on PATH", result.stderr)
        self.assertLess(self.events.read_text().index("xattr"), self.events.read_text().index("version"))

    def test_exact_archive_set_is_order_independent(self):
        self.make_release(mutation="reordered")
        self.run_installer("--version", "v0.1.0")
        self.assert_links()

    def test_public_latest_rejects_an_unexpected_redirect(self):
        self.run_installer(
            LATEST_URL="https://github.com/attacker/project/releases/tag/v0.1.0",
            expect=1,
        )
        calls = self.log.read_text()
        self.assertEqual(calls.count("curl "), 1)
        self.assertFalse(self.installed_root().exists())

    def test_public_download_ignores_github_credentials_and_cli(self):
        self.make_release()
        self.run_installer(GH_TOKEN="unused-test-token", GH_HOST="enterprise.example")
        calls = self.log.read_text()
        self.assertNotIn("gh ", calls)
        self.assertNotIn("unused-test-token", calls)
        self.assertNotIn("enterprise.example", calls)
        self.assertNotIn("Authorization", calls)
        self.assertEqual(calls.count("curl "), 3)
        self.assertIn("https://github.com/codesoda/openjev-rs/releases/latest", calls)
        self.assert_links()

    def test_signal_during_link_staging_cleans_temporary_directories(self):
        self.make_release("v0.1.0")
        self.run_installer("--version", "v0.1.0")
        old_link = os.readlink(self.home / ".openjev/bin/openjev")
        self.make_release("v0.2.0")
        self._stub("ln", 'kill -TERM "$PPID"\nexit 1\n')
        self.run_installer("--version", "v0.2.0", LATEST_TAG="v0.2.0", expect=1)
        self.assertEqual(os.readlink(self.home / ".openjev/bin/openjev"), old_link)
        self.assertEqual(list(self.home.rglob(".openjev-link.*")), [])
        self.assertEqual(list(self.home.rglob(".install.*")), [])
        self.assertFalse((self.home / ".openjev/install.lock").exists())
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_linux_version_pin_skips_latest_and_xattr(self):
        self.make_release(target=LINUX_TARGET)
        self.run_installer(
            "--version",
            "v0.1.0",
            TEST_OS="Linux",
            TEST_ARCH="x86_64",
            TEST_GLIBC="2.35",
        )
        calls = self.log.read_text()
        self.assertNotIn("/releases/latest", calls)
        self.assertNotIn("xattr ", calls)
        self.assert_links(target=LINUX_TARGET)

    def test_repeat_is_idempotent_and_upgrade_retains_old_payload(self):
        self.make_release("v0.1.0")
        self.run_installer("--version", "v0.1.0")
        old = self.installed_root()
        self.run_installer("--version", "v0.1.0")
        self.make_release("v0.2.0")
        self.run_installer("--version", "v0.2.0", LATEST_TAG="v0.2.0")
        self.assertTrue(old.is_dir())
        self.assertTrue(self.installed_root("v0.2.0").is_dir())
        self.assert_links("v0.2.0")

    def test_existing_same_version_must_be_byte_identical(self):
        self.make_release()
        self.run_installer("--version", "v0.1.0")
        active = os.readlink(self.home / ".openjev/bin/openjev")
        existing_binary = self.installed_root() / "openjev"
        existing_binary.chmod(0o644)
        (self.installed_root() / "LICENSE").write_text("changed", encoding="utf-8")
        result = self.run_installer("--version", "v0.1.0", expect=1)
        self.assertIn("differs", result.stderr)
        self.assertFalse(os.access(existing_binary, os.X_OK))
        self.assertEqual(os.readlink(self.home / ".openjev/bin/openjev"), active)

    def test_identical_existing_payload_repairs_and_checks_actual_binary(self):
        self.make_release()
        self.run_installer("--version", "v0.1.0")
        existing_binary = self.installed_root() / "openjev"
        existing_binary.chmod(0o644)
        self.events.write_text("", encoding="utf-8")
        self.run_installer("--version", "v0.1.0")
        self.assertTrue(os.access(existing_binary, os.X_OK))
        events = self.events.read_text().splitlines()
        self.assertTrue(any(str(existing_binary) in event for event in events), events)
        self.assert_links()

    def test_identical_existing_payload_clears_quarantine_on_actual_binary(self):
        self.make_release()
        self.run_installer("--version", "v0.1.0")
        existing_binary = self.installed_root() / "openjev"
        self.log.write_text("", encoding="utf-8")
        self.run_installer(
            "--version",
            "v0.1.0",
            XATTR_HAS_MATCH=str(existing_binary),
        )
        self.assertIn(
            f"xattr -d com.apple.quarantine {existing_binary}",
            self.log.read_text(),
        )
        self.assert_links()

    def test_reused_binary_runtime_failure_preserves_existing_links(self):
        archive = self.make_release(binary_mode="staged-only")
        bin_dir = self.home / ".openjev/bin"
        bin_dir.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as packaged:
            packaged.extractall(bin_dir)
        old_payload = bin_dir / f"openjev-v0.0.9-{MAC_TARGET}"
        old_payload.mkdir()
        old_binary = old_payload / "openjev"
        old_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        old_binary.chmod(0o755)
        active = bin_dir / "openjev"
        active.symlink_to(old_binary)
        command = self.home / ".local/bin/openjev"
        command.parent.mkdir(parents=True)
        command.symlink_to(active)
        result = self.run_installer("--version", "v0.1.0", expect=1)
        self.assertIn("failed its --version check", result.stderr)
        self.assertEqual(os.readlink(active), str(old_binary))
        self.assertEqual(os.readlink(command), str(active))

    def test_previous_layout_link_is_migrated_without_deleting_payload(self):
        self.make_release()
        old = self.home / ".local/share/openjev/releases/v0.0.9/openjev-v0.0.9-aarch64-apple-darwin"
        old.mkdir(parents=True)
        (old / "openjev").write_text("old", encoding="utf-8")
        link_dir = self.home / ".local/bin"
        link_dir.mkdir(parents=True)
        (link_dir / "openjev").symlink_to(old / "openjev")
        self.run_installer("--version", "v0.1.0")
        self.assertEqual((old / "openjev").read_text(), "old")
        self.assert_links()

    def test_managed_looking_links_through_symlinked_parents_are_refused(self):
        for layout in ("new", "previous"):
            with self.subTest(layout=layout):
                case_home = self.root / ("canonical-" + layout)
                case_home.mkdir()
                if layout == "new":
                    outside = self.root / "outside-new"
                    outside.mkdir(exist_ok=True)
                    (outside / "openjev").write_text("outside", encoding="utf-8")
                    bin_dir = case_home / ".openjev/bin"
                    bin_dir.mkdir(parents=True)
                    declared_parent = bin_dir / f"openjev-v0.0.9-{MAC_TARGET}"
                    declared_parent.symlink_to(outside, target_is_directory=True)
                    link = bin_dir / "openjev"
                    link.symlink_to(declared_parent / "openjev")
                else:
                    outside = self.root / "outside-previous"
                    old = outside / f"openjev/releases/v0.0.9/openjev-v0.0.9-{MAC_TARGET}"
                    old.mkdir(parents=True)
                    (old / "openjev").write_text("outside", encoding="utf-8")
                    local = case_home / ".local"
                    local.mkdir()
                    (local / "share").symlink_to(outside, target_is_directory=True)
                    link = local / "bin/openjev"
                    link.parent.mkdir()
                    link.symlink_to(
                        case_home
                        / f".local/share/openjev/releases/v0.0.9/openjev-v0.0.9-{MAC_TARGET}/openjev"
                    )
                self.run_installer("--version", "v0.1.0", HOME=str(case_home), expect=1)
                self.assertTrue(link.is_symlink())
        calls = self.log.read_text() if self.log.exists() else ""
        self.assertNotIn("curl ", calls)

    def test_unrelated_file_and_links_are_refused_before_network(self):
        cases = ("file", "local-link", "active-link")
        for case in cases:
            with self.subTest(case=case):
                case_home = self.root / ("home-" + case)
                case_home.mkdir()
                if case == "file":
                    path = case_home / ".local/bin/openjev"
                    path.parent.mkdir(parents=True)
                    path.write_text("mine", encoding="utf-8")
                elif case == "local-link":
                    path = case_home / ".local/bin/openjev"
                    path.parent.mkdir(parents=True)
                    path.symlink_to("/tmp/not-openjev")
                else:
                    path = case_home / ".openjev/bin/openjev"
                    path.parent.mkdir(parents=True)
                    path.symlink_to("/tmp/not-openjev")
                self.run_installer("--version", "v0.1.0", HOME=str(case_home), expect=1)
                calls = self.log.read_text() if self.log.exists() else ""
                self.assertNotIn("curl ", calls)
                self.assertTrue(path.exists() or path.is_symlink())

    def test_truncated_definitions_cannot_begin_installation(self):
        text = INSTALLER.read_text(encoding="utf-8")
        definitions = text.rsplit('\nmain "$@"', 1)[0] + "\n"
        truncated = self.root / "truncated-install.sh"
        truncated.write_text(definitions, encoding="utf-8")
        missing_home = self.root / "truncated-home"
        result = subprocess.run(
            ["/bin/sh", str(truncated)],
            env={
                **os.environ,
                "HOME": str(missing_home),
                "PATH": f"{self.stubs}:/usr/bin:/bin",
                "CALL_LOG": str(self.log),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertFalse(missing_home.exists())
        self.assertFalse(self.log.exists())

    def test_help_and_invalid_version_do_not_install_or_use_network(self):
        missing_home = self.root / "does-not-exist"
        result = self.run_installer("--help", HOME=str(missing_home))
        self.assertIn("Usage:", result.stderr)
        self.assertFalse(missing_home.exists())
        self.assertFalse(self.log.exists())
        result = self.run_installer("--version", "main", expect=1)
        self.assertIn("tag like", result.stderr)
        self.assertFalse((self.home / ".openjev").exists())

    def test_unsupported_platform_and_minimum_versions_fail_without_network(self):
        cases = (
            {"TEST_OS": "FreeBSD", "TEST_ARCH": "x86_64"},
            {"TEST_OS": "Darwin", "TEST_ARCH": "x86_64"},
            {"TEST_OS": "Darwin", "TEST_ARCH": "arm64", "TEST_MAC_VERSION": "13.6"},
            {"TEST_OS": "Linux", "TEST_ARCH": "x86_64", "TEST_GLIBC": "2.34"},
        )
        for values in cases:
            with self.subTest(values=values):
                self.run_installer("--version", "v0.1.0", expect=1, **values)
                calls = self.log.read_text() if self.log.exists() else ""
                self.assertNotIn("curl ", calls)

    def test_checksum_failures_preserve_old_install(self):
        self.make_release("v0.1.0")
        self.run_installer("--version", "v0.1.0")
        old_link = os.readlink(self.home / ".openjev/bin/openjev")
        for mode in (
            "wrong",
            "missing",
            "duplicate",
            "malformed",
            "extra-field",
            "valid-plus-malformed",
        ):
            with self.subTest(mode=mode):
                self.make_release("v0.2.0", checksum_mode=mode)
                result = self.run_installer(
                    "--version", "v0.2.0", LATEST_TAG="v0.2.0", expect=1
                )
                if mode == "extra-field":
                    self.assertIn("invalid SHA-256", result.stderr)
                elif mode == "valid-plus-malformed":
                    self.assertIn("exactly one entry", result.stderr)
                self.assertEqual(os.readlink(self.home / ".openjev/bin/openjev"), old_link)
                self.assertFalse(self.installed_root("v0.2.0").exists())

    def test_malicious_archive_members_are_rejected(self):
        for mutation in ("traversal", "absolute", "symlink", "hardlink", "extra", "duplicate"):
            with self.subTest(mutation=mutation):
                self.make_release(mutation=mutation)
                self.run_installer("--version", "v0.1.0", expect=1)
                self.assertFalse(self.installed_root().exists())
                self.assertFalse((self.root / "outside").exists())

    def test_invalid_or_failing_binary_preserves_old_install(self):
        self.make_release("v0.1.0")
        self.run_installer("--version", "v0.1.0")
        old_link = os.readlink(self.home / ".openjev/bin/openjev")
        for mode in ("wrong", "bad-json", "fail"):
            with self.subTest(mode=mode):
                self.make_release("v0.2.0", binary_mode=mode)
                self.run_installer("--version", "v0.2.0", LATEST_TAG="v0.2.0", expect=1)
                self.assertEqual(os.readlink(self.home / ".openjev/bin/openjev"), old_link)

    def test_interrupted_and_corrupt_downloads_preserve_old_install(self):
        self.make_release("v0.1.0")
        self.run_installer("--version", "v0.1.0")
        old_link = os.readlink(self.home / ".openjev/bin/openjev")
        self.make_release("v0.2.0")
        self.run_installer(
            "--version", "v0.2.0", LATEST_TAG="v0.2.0", CURL_FAIL_MATCH=".tar.gz", expect=1
        )
        self.assertEqual(os.readlink(self.home / ".openjev/bin/openjev"), old_link)
        archive = self.fixtures / "v0.2.0" / f"openjev-v0.2.0-{MAC_TARGET}.tar.gz"
        archive.write_bytes(b"truncated")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (archive.parent / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        self.run_installer("--version", "v0.2.0", LATEST_TAG="v0.2.0", expect=1)
        self.assertEqual(os.readlink(self.home / ".openjev/bin/openjev"), old_link)

    def test_symlinked_home_or_install_component_is_refused(self):
        actual = self.root / "actual-home"
        actual.mkdir()
        linked = self.root / "linked-home"
        linked.symlink_to(actual, target_is_directory=True)
        self.run_installer("--version", "v0.1.0", HOME=str(linked), expect=1)
        unsafe = self.root / "unsafe"
        unsafe.mkdir()
        (self.home / ".openjev").symlink_to(unsafe, target_is_directory=True)
        self.run_installer("--version", "v0.1.0", expect=1)
        self.assertEqual(list(unsafe.iterdir()), [])

    def test_lock_contention_fails_before_network(self):
        lock = self.home / ".openjev/install.lock"
        lock.mkdir(parents=True)
        self.run_installer("--version", "v0.1.0", expect=1)
        self.assertTrue(lock.is_dir())
        calls = self.log.read_text() if self.log.exists() else ""
        self.assertNotIn("curl ", calls)

    def test_macos_no_quarantine_attribute_succeeds(self):
        self.make_release()
        self.run_installer("--version", "v0.1.0", XATTR_HAS="0")
        calls = self.log.read_text()
        self.assertIn("xattr ", calls)
        self.assertNotIn("xattr -d", calls)
        self.assert_links()

    def test_reused_binary_xattr_failure_preserves_existing_links(self):
        self.make_release("v0.1.0")
        self.run_installer("--version", "v0.1.0")
        active = self.home / ".openjev/bin/openjev"
        old_link = os.readlink(active)
        existing_binary = self.installed_root() / "openjev"
        result = self.run_installer(
            "--version",
            "v0.1.0",
            XATTR_HAS_MATCH=str(existing_binary),
            XATTR_DELETE_FAIL="1",
            expect=1,
        )
        self.assertIn("quarantine", result.stderr)
        self.assertEqual(os.readlink(active), old_link)

    def test_macos_xattr_delete_failure_preserves_old_install(self):
        self.make_release("v0.1.0")
        self.run_installer("--version", "v0.1.0")
        old_link = os.readlink(self.home / ".openjev/bin/openjev")
        self.events.write_text("", encoding="utf-8")
        self.make_release("v0.2.0")
        result = self.run_installer(
            "--version",
            "v0.2.0",
            LATEST_TAG="v0.2.0",
            XATTR_HAS="1",
            XATTR_DELETE_FAIL="1",
            expect=1,
        )
        self.assertIn("quarantine", result.stderr)
        self.assertEqual(os.readlink(self.home / ".openjev/bin/openjev"), old_link)
        events = self.events.read_text().splitlines()
        self.assertFalse(any(event.startswith("version ") for event in events), events)


if __name__ == "__main__":
    unittest.main()
