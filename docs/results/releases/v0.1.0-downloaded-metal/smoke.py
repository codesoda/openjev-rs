#!/usr/bin/env python3
"""Smoke-test an installed, downloaded OpenJev macOS release."""
import argparse, datetime as dt, hashlib, json, os, re, secrets, shutil
import signal, socket, subprocess, sys, tempfile, time, urllib.error, urllib.request
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_env():
    env = os.environ.copy()
    exact = {"LD_LIBRARY_PATH", "LD_PRELOAD", "LIBRARY_PATH", "GGML_METAL_PATH",
             "GGML_METAL_LIBRARY", "OPENJEV_API_KEY", "OPENJEV_RELEASE_SMOKE_KEY"}
    for name in list(env):
        if name in exact or name.startswith(("DYLD_", "GGML_METAL_")):
            env.pop(name, None)
    return env


def captured(command, cwd, env, timeout, out_path, err_path, secret=None):
    try:
        result = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        out, err = error.stdout or b"", error.stderr or b""
        if secret:
            out, err = out.replace(secret.encode(), b"<redacted>"), err.replace(secret.encode(), b"<redacted>")
        out_path.write_bytes(out); err_path.write_bytes(err)
        raise AssertionError(f"command timed out; see {out_path.name}, {err_path.name}")
    out, err = result.stdout, result.stderr
    if secret:
        out, err = out.replace(secret.encode(), b"<redacted>"), err.replace(secret.encode(), b"<redacted>")
    out_path.write_bytes(out); err_path.write_bytes(err)
    assert result.returncode == 0, f"command exited {result.returncode}; see {out_path.name}, {err_path.name}"
    return out, err


def http(base, method, path, body=None, token=None, timeout=120):
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    headers = {"Accept": "application/json"}
    if data is not None: headers["Content-Type"] = "application/json"
    if token is not None: headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    value = json.loads(response.read())
    safe = {k.lower(): v for k, v in response.headers.items()
            if k.lower() not in {"authorization", "proxy-authorization", "set-cookie"}}
    return response.status, safe, value


def probability(value):
    assert type(value) in (int, float) and 0 <= value <= 1
    assert abs(value * 100 - round(value * 100)) < 1e-9


def validate(value):
    assert value["model"] == "qwen3-0.6b"
    answers = value["answers"]
    assert list(answers) == ["route", "review", "urgency"]
    route = answers["route"]
    assert route["type"] == "choice" and route["choice"] in ("billing", "support")
    probability(route["confidence"])
    assert list(route["probabilities"]) == ["billing", "support"]
    for item in route["probabilities"].values(): probability(item)
    assert abs(sum(route["probabilities"].values()) - 1) <= .01
    review = answers["review"]
    assert review["type"] == "noul"; probability(review["noul"])
    urgency = answers["urgency"]
    assert urgency["type"] == "score" and 0 <= urgency["score"] <= 2
    assert abs(urgency["score"] * 100 - round(urgency["score"] * 100)) < 1e-9
    assert urgency["legend"] == {"0": "low", "1": "medium", "2": "high"}
    probability(urgency["confidence"])
    assert list(urgency["probabilities"]) == ["0", "1", "2"]
    for item in urgency["probabilities"].values(): probability(item)
    assert abs(sum(urgency["probabilities"].values()) - 1) <= .015
    usage = value["usage"]
    assert type(usage["input_tokens"]) is int and usage["input_tokens"] > 0
    assert usage["output_tokens"] == 0


def main():
    ap = argparse.ArgumentParser()
    for name in ("binary", "expected-sha256", "cache-dir", "sdk-dir", "evidence-dir"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    binary_arg = Path(args.binary)
    assert binary_arg.is_absolute(), "--binary must be absolute"
    binary = binary_arg.resolve(strict=True)
    assert binary.is_file() and os.access(binary, os.X_OK), "binary must be executable"
    expected = args.expected_sha256.lower()
    assert re.fullmatch(r"[0-9a-f]{64}", expected), "invalid expected SHA-256"
    actual = sha256(binary)
    assert actual == expected, "binary hash differs from verified BUILD-INFO payload"

    cache, sdk = Path(args.cache_dir).resolve(strict=True), Path(args.sdk_dir).resolve(strict=True)
    assert cache.is_dir() and sdk.is_dir()
    assert (sdk / "smoke.mjs").is_file() and (sdk / "package-lock.json").is_file()
    evidence = Path(args.evidence_dir)
    evidence.mkdir(mode=0o700, parents=False, exist_ok=False)
    info_path = binary.parent / "BUILD-INFO.json"
    assert info_path.is_file(), "BUILD-INFO.json must be beside resolved binary"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["files_sha256"]["openjev"].lower() == expected
    source_sha = info["source_sha"]
    assert re.fullmatch(r"[0-9a-fA-F]{40,64}", source_sha)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    key, env = secrets.token_urlsafe(32), clean_env()
    work = Path(tempfile.mkdtemp(prefix="openjev-release-smoke-"))
    assert sdk not in work.parents and work != sdk
    modules = sdk / "node_modules"
    assert not modules.is_symlink(), "refusing symlinked node_modules"
    remove_modules = not modules.exists()
    proc = server_out = server_err = None
    failure = None
    summary = {"schema": "openjev-downloaded-release-smoke-v1", "timestamp_utc": stamp,
               "status": "failed", "binary_sha256": actual, "source_sha": source_sha}
    manifest = {"timestamp_utc": stamp, "commands": [],
                "environment_removed": sorted(name for name in os.environ if name not in env)}
    responses = []
    try:
        for flag, schema in (("--version", "openjev-version-v1"), ("--help", "openjev-help-v1")):
            name, command = flag[2:], [str(binary), flag]
            manifest["commands"].append({"argv": command, "cwd": str(work)})
            out, err = captured(command, work, env, 30, evidence / f"{name}.stdout.json",
                                evidence / f"{name}.stderr.txt")
            assert not err and json.loads(out)["schema"] == schema

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [str(binary), "--serve", "--offline", "--model", "qwen3-0.6b", "--device", "metal",
                   "--cache-dir", str(cache), "--host", "127.0.0.1", "--port", str(port),
                   "--api-key-env", "OPENJEV_RELEASE_SMOKE_KEY"]
        manifest["commands"].append({"argv": command, "cwd": str(work),
                                     "env": {"OPENJEV_RELEASE_SMOKE_KEY": "<ephemeral-redacted>"}})
        server_env = env.copy(); server_env["OPENJEV_RELEASE_SMOKE_KEY"] = key
        server_out = (evidence / "server.stdout.txt").open("wb", buffering=0)
        server_err = (evidence / "server.stderr.txt").open("wb", buffering=0)
        proc = subprocess.Popen(command, cwd=work, env=server_env, stdin=subprocess.DEVNULL,
                                stdout=server_out, stderr=server_err)
        pid, deadline = proc.pid, time.monotonic() + 120
        while True:
            assert proc.poll() is None, "server exited before readiness"
            try:
                health, ready = http(base, "GET", "/healthz", timeout=2), http(base, "GET", "/readyz", timeout=2)
                if health[0] == 200 and health[2].get("status") == "ok" and ready[0] == 200 and ready[2].get("status") == "ready":
                    responses += [{"name": "health", "status": health[0], "headers": health[1], "body": health[2]},
                                  {"name": "ready", "status": ready[0], "headers": ready[1], "body": ready[2]}]
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            assert time.monotonic() < deadline, "readiness exceeded 120 seconds"
            time.sleep(.2)

        body = {"model": "jev-latest", "state": {"ticket": "duplicate charge", "severity": 3},
                "questions": {"route": {"type": "choice", "criteria": {"billing": "payments", "support": "general"}},
                              "review": {"type": "noul", "instructions": "Does a human need review?"},
                              "urgency": {"type": "score", "criteria": ["low", "medium", "high"]}}}
        values = []
        for index in range(3):
            status, headers, value = http(base, "POST", "/v1/systemone", body, key)
            assert status == 200 and "x-openjev-fallback" in headers
            validate(value); values.append(value)
            responses.append({"name": f"mixed-{index + 1}", "status": status, "headers": headers, "body": value})
            assert proc.poll() is None and proc.pid == pid
        assert values[0] == values[1] == values[2], "resident responses differ"
        assert (evidence / "server.stdout.txt").stat().st_size == 0

        checks = [("wrong-bearer", body, "definitely-wrong", 401),
                  ("unknown-model", dict(body, model="not-the-loaded-model"), key, 404),
                  ("unsupported-float", dict(body, state={"unsupportedFloat": 1.5}), key, 422)]
        for name, payload, token, wanted in checks:
            status, headers, value = http(base, "POST", "/v1/systemone", payload, token)
            assert status == wanted, f"{name}: expected {wanted}, got {status}"
            responses.append({"name": name, "status": status, "headers": headers, "body": value})
        assert proc.poll() is None and proc.pid == pid

        npm = shutil.which("npm", path=env.get("PATH")); assert npm, "npm not found"
        npm_ci = [npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"]
        manifest["commands"].append({"argv": npm_ci, "cwd": str(sdk)})
        captured(npm_ci, sdk, env, 300, evidence / "npm-ci.stdout.txt", evidence / "npm-ci.stderr.txt")
        sdk_env = env.copy(); sdk_env.update({"OPENJEV_BASE_URL": base, "OPENJEV_API_KEY": key})
        sdk_cmd = [npm, "run", "smoke"]
        manifest["commands"].append({"argv": sdk_cmd, "cwd": str(sdk),
                                     "env": {"OPENJEV_BASE_URL": base, "OPENJEV_API_KEY": "<ephemeral-redacted>"}})
        sdk_out, _ = captured(sdk_cmd, sdk, sdk_env, 180, evidence / "sdk.stdout.txt", evidence / "sdk.stderr.txt", key)
        sdk_result = next(json.loads(line) for line in reversed(sdk_out.decode().splitlines()) if line.startswith("{"))
        assert sdk_result["status"] == "passed" and sdk_result["model"] == "qwen3-0.6b"
        dump(evidence / "sdk-result.json", sdk_result)
        assert proc.poll() is None and proc.pid == pid
        assert (evidence / "server.stdout.txt").stat().st_size == 0

        proc.send_signal(signal.SIGTERM); proc.wait(timeout=45)
        assert proc.returncode == 0, f"SIGTERM exit was {proc.returncode}"
        server_out.close(); server_out = None
        server_err.close(); server_err = None
        assert (evidence / "server.stdout.txt").stat().st_size == 0
        log = (evidence / "server.stderr.txt").read_text(encoding="utf-8", errors="replace")
        assert key not in log, "server log contained bearer secret"
        assert log.count("loaded pinned GGUF model=qwen3-0.6b") == 1, "expected one pinned model load"
        assert "using embedded metal library" in log, "embedded Metal library log missing"
        assert re.search(r"GPU name:\s+MTL\d+ \(Apple[^)]*\)", log), "Apple Metal device log missing"
        summary.update({"status": "passed", "server_pid": pid, "resident_pid_unchanged": True,
                        "http_checks": len(responses), "sdk_status": sdk_result["status"]})
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"; summary["error"] = failure
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(); failure = failure or "server did not stop after SIGTERM"
        if server_out is not None: server_out.close()
        if server_err is not None: server_err.close()
        if remove_modules and modules.exists():
            if modules.is_dir() and not modules.is_symlink() and modules.parent.resolve() == sdk:
                shutil.rmtree(modules)
            else: failure = failure or "refused unsafe node_modules cleanup"
        shutil.rmtree(work)
        secret = key.encode()
        for path in evidence.iterdir():
            if path.is_file() and secret in (data := path.read_bytes()):
                path.write_bytes(data.replace(secret, b"<ephemeral-redacted>"))
        if failure: summary.update(status="failed", error=failure)
        dump(evidence / "http-responses.json", responses)
        dump(evidence / "artifact-identity.json", {"binary": str(binary_arg), "resolved_binary": str(binary),
             "binary_sha256": actual, "build_info": str(info_path), "build_info_sha256": sha256(info_path),
             "source_sha": source_sha})
        dump(evidence / "command-manifest.json", manifest); dump(evidence / "summary.json", summary)

    if failure:
        print(f"FAILED: {failure}; evidence: {evidence}", file=sys.stderr); return 1
    print(json.dumps(summary, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
