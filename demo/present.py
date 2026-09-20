#!/usr/bin/env python3
"""Read live `openjev demo --quiet` JSONL; show a labelled compact projection.

Only presentation is paced. No predictions are mocked, replaced, or preloaded.
The unmodified rows are retained in the recording directory.
"""
import json
import os
from pathlib import Path
import sys
import time


def compact(row):
    return {
        "example": row["example"],
        "model": row["response"]["model"],
        "answers": {
            key: {"type": answer["type"], answer["type"]: answer[answer["type"]]}
            for key, answer in row["response"]["answers"].items()
        },
        "http_ms": round(row["elapsed_ms"], 1),
    }


def main():
    destination = Path(os.environ["OPENJEV_RECORDING_DIR"]) / "responses.jsonl"
    count = 0
    print("\033[?25l", end="", flush=True)
    try:
        with destination.open("w", encoding="utf-8") as raw:
            for count, line in enumerate(sys.stdin, 1):
                row = json.loads(line)
                if row["response"]["usage"]["output_tokens"] != 0:
                    raise ValueError("unexpected generated tokens in demo response")
                raw.write(line)
                raw.flush()
                print("\033[2J\033[H\033[1;36mOPENJEV / local typed decisions\033[0m")
                print('\033[2m$ openjev demo --base-url "$DEMO_URL" --quiet | python3 demo/present.py\033[0m')
                print(f"\n\033[1;35m{count:02d} / 08   {row['example']}\033[0m\n")
                print(json.dumps(compact(row), indent=2, ensure_ascii=False))
                print("\n\033[2mLive HTTP / compact projection / display paced for reading")
                print("Probabilities are conditional and uncalibrated. Not an accuracy test.\033[0m")
                if reason := row.get("metadata", {}).get("x-openjev-fallback"):
                    print(f"\033[33mFallback: {reason}\033[0m")
                sys.stdout.flush()
                time.sleep(2.4 if count < 8 else 1)
        if count != 8:
            raise ValueError(f"expected eight successful responses, got {count}")
        print("\n\033[1;32m8 HTTP requests complete / one resident model / no generated tokens\033[0m", flush=True)
    finally:
        print("\033[?25h", end="", flush=True)


if __name__ == "__main__":
    main()
