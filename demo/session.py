#!/usr/bin/env python3
"""Record real server startup, then live HTTP requests. Own/stop only our children."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request

from present import present


def command(args):
    print("\033[1;36m$ \033[0m", end="", flush=True)
    for char in shlex.join(args):
        print(char, end="", flush=True)
        time.sleep(0.025)
    print(flush=True)


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def interrupted(*_):
    raise KeyboardInterrupt


def main():
    logs = Path(os.environ["OPENJEV_RECORDING_DIR"])
    url = os.environ["DEMO_URL"]
    server_args = ["openjev", "serve", "--offline", "--model", "qwen3-0.6b",
                   "--device", os.environ["DEMO_DEVICE"], "--port", os.environ["DEMO_PORT"]]
    server = client = None
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    print("\033[2J\033[H\033[?25l\033[1;36mOPENJEV / start the resident server\033[0m\n")
    print("\033[2mReal model startup / cached weights / native logs saved to out/demo-recording\033[0m\n")
    try:
        command(server_args)
        with (logs/'server.stdout').open('w') as out, (logs/'server.stderr').open('w') as err:
            started = time.monotonic()
            server = subprocess.Popen(server_args, stdout=out, stderr=err)
            print("\nLoading and verifying Qwen3-0.6B; warming the model...", flush=True)
            deadline = started + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('Server exited; see out/demo-recording/server.stderr')
                try:
                    with urllib.request.urlopen(url+'/readyz', timeout=1) as response:
                        ready = json.load(response)
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError('Server not ready after 120 seconds')
            elapsed = time.monotonic() - started
            print(f"\nGET /readyz  ->  {json.dumps(ready)}")
            print(f"\033[1;32mReady after {elapsed:.1f}s / model stays loaded for all requests\033[0m", flush=True)
            time.sleep(5)
            print("\n\033[1;35mSECOND TERMINAL / run the demo\033[0m\n")
            client_args = ["openjev", "demo", "--base-url", url, "--quiet"]
            command(client_args)
            print("\nInputs first, then results. Display pauses are for reading.", flush=True)
            time.sleep(3)
            with (logs/'client.stderr').open('w') as client_err:
                client = subprocess.Popen(client_args, stdout=subprocess.PIPE, stderr=client_err, text=True)
                present(client.stdout)
                if client.wait(timeout=10) != 0:
                    raise RuntimeError('Demo failed; see out/demo-recording/client.stderr')
    finally:
        stop(client)
        stop(server)
        print("\033[?25h", end="", flush=True)
    if server.returncode != 0:
        raise RuntimeError(f'Server shutdown failed: {server.returncode}')
    (logs/'complete').write_text('eight responses; server stopped cleanly\n')
    print("\nRecording complete; demo server stopped.", flush=True)


if __name__ == '__main__':
    main()
