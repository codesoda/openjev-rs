#!/usr/bin/env python3
"""Validate release-relevant values in a generated CMakeCache.txt.

CMake cache type markers (for example BOOL, STRING, or UNINITIALIZED) are not
build semantics. This checker compares the parsed values while retaining the
type in diagnostics.
"""

import argparse
from pathlib import Path
import sys

COMMON_REQUIRED = {
    "BUILD_SHARED_LIBS": "OFF",
    "GGML_BLAS": "OFF",
    "GGML_OPENMP": "OFF",
    "GGML_NATIVE": "OFF",
}
MACOS_REQUIRED = {
    "GGML_METAL": "ON",
    "GGML_METAL_EMBED_LIBRARY": "ON",
    "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
}
LINUX_REQUIRED = {
    name: "OFF"
    for name in (
        "GGML_SSE42",
        "GGML_AVX",
        "GGML_AVX_VNNI",
        "GGML_AVX2",
        "GGML_BMI2",
        "GGML_AVX512",
        "GGML_AVX512_VBMI",
        "GGML_AVX512_VNNI",
        "GGML_AVX512_BF16",
        "GGML_FMA",
        "GGML_F16C",
    )
}


class CacheError(Exception):
    pass


def normalize_platform(value):
    normalized = value.lower()
    if normalized == "macos":
        return "macos"
    if normalized == "linux":
        return "linux"
    raise CacheError("platform must be macOS or Linux, got %r" % value)


def required_values(platform):
    platform = normalize_platform(platform)
    expected = dict(COMMON_REQUIRED)
    expected.update(MACOS_REQUIRED if platform == "macos" else LINUX_REQUIRED)
    return expected


def parse_cache(path):
    values = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        if "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        left, value = line.split("=", 1)
        key, cache_type = left.rsplit(":", 1)
        if not key or not cache_type:
            continue
        if key in values:
            raise CacheError("duplicate CMake cache key %s at line %d" % (key, line_number))
        values[key] = (cache_type, value)
    return values


def validate_cache(path, platform):
    expected = required_values(platform)
    actual = parse_cache(path)
    failures = []
    for key, expected_value in expected.items():
        entry = actual.get(key)
        if entry is None or entry[1] != expected_value:
            failures.append(key)
    if failures:
        lines = ["native CMake configuration mismatch in %s:" % path]
        for key, expected_value in expected.items():
            entry = actual.get(key)
            if entry is None:
                actual_text = "<missing>"
            else:
                actual_text = "%r (cache type %s)" % (entry[1], entry[0])
            status = "FAIL" if key in failures else "ok"
            lines.append("  %s: expected %r, actual %s [%s]" % (key, expected_value, actual_text, status))
        raise CacheError("\n".join(lines))
    return expected, actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, help="GitHub runner OS: macOS or Linux")
    parser.add_argument("cache", type=Path, help="path to llama-cpp-sys CMakeCache.txt")
    args = parser.parse_args(argv)
    try:
        platform = normalize_platform(args.platform)
        if not args.cache.is_file():
            raise CacheError("CMake cache is not a regular file: %s" % args.cache)
        expected, actual = validate_cache(args.cache, platform)
    except (CacheError, OSError, UnicodeError) as error:
        print("check_native_build.py: error: %s" % error, file=sys.stderr)
        return 1
    print("verified %s native CMake values in %s" % (len(expected), args.cache))
    for key, expected_value in expected.items():
        print("  %s=%s (cache type %s)" % (key, expected_value, actual[key][0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
