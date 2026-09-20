#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_native_build.py"
SPEC = importlib.util.spec_from_file_location("openjev_check_native_build", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class NativeBuildConfigurationTests(unittest.TestCase):
    def cache(self, directory, platform, type_for_key=None, overrides=None, omitted=()):
        type_for_key = type_for_key or {}
        overrides = overrides or {}
        lines = ["# generated test cache"]
        for key, expected in CHECKER.required_values(platform).items():
            if key in omitted:
                continue
            cache_type = type_for_key.get(key, "BOOL")
            lines.append("%s:%s=%s" % (key, cache_type, overrides.get(key, expected)))
        path = directory / "CMakeCache.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_cache_type_markers_do_not_change_value_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            type_for_key = {
                key: ("BOOL", "STRING", "UNINITIALIZED")[index % 3]
                for index, key in enumerate(CHECKER.required_values("macOS"))
            }
            type_for_key["CMAKE_OSX_DEPLOYMENT_TARGET"] = "UNINITIALIZED"
            path = self.cache(root, "macOS", type_for_key=type_for_key)
            expected, actual = CHECKER.validate_cache(path, "macOS")
            self.assertEqual({key: value for key, (_, value) in actual.items()}, expected)
            self.assertEqual(actual["CMAKE_OSX_DEPLOYMENT_TARGET"], ("UNINITIALIZED", "14.0"))

    def test_wrong_and_missing_values_fail_with_complete_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.cache(
                root,
                "Linux",
                overrides={"GGML_AVX2": "ON"},
                omitted={"GGML_OPENMP"},
            )
            with self.assertRaises(CHECKER.CacheError) as caught:
                CHECKER.validate_cache(path, "Linux")
            message = str(caught.exception)
            self.assertIn("GGML_AVX2: expected 'OFF', actual 'ON'", message)
            self.assertIn("GGML_OPENMP: expected 'OFF', actual <missing>", message)
            self.assertIn("BUILD_SHARED_LIBS: expected 'OFF', actual 'OFF'", message)

    def test_platform_required_sets_preserve_all_release_gates(self):
        common = {
            "BUILD_SHARED_LIBS": "OFF",
            "GGML_BLAS": "OFF",
            "GGML_OPENMP": "OFF",
            "GGML_NATIVE": "OFF",
        }
        self.assertEqual(
            CHECKER.required_values("macOS"),
            {
                **common,
                "GGML_METAL": "ON",
                "GGML_METAL_EMBED_LIBRARY": "ON",
                "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
            },
        )
        self.assertEqual(
            CHECKER.required_values("Linux"),
            {
                **common,
                "GGML_SSE42": "OFF",
                "GGML_AVX": "OFF",
                "GGML_AVX_VNNI": "OFF",
                "GGML_AVX2": "OFF",
                "GGML_BMI2": "OFF",
                "GGML_AVX512": "OFF",
                "GGML_AVX512_VBMI": "OFF",
                "GGML_AVX512_VNNI": "OFF",
                "GGML_AVX512_BF16": "OFF",
                "GGML_FMA": "OFF",
                "GGML_F16C": "OFF",
            },
        )


if __name__ == "__main__":
    unittest.main()
