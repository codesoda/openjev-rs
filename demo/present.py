#!/usr/bin/env python3
"""Paced presentation of the actual request and response in demo JSONL rows."""
import json
import os
from pathlib import Path
import sys
import textwrap
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


def header(count, name, stage):
    print("\033[2J\033[H\033[1;36mOPENJEV / local typed decisions\033[0m")
    print("\033[2mOne resident server / eight real HTTP examples / paced for reading\033[0m")
    print(f"\n\033[1;35m{count:02d} / 08   {name} / {stage}\033[0m\n")


def wrapped(value, indent="  "):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    print(textwrap.fill(text, width=94, initial_indent=indent, subsequent_indent=indent,
                        break_long_words=False, break_on_hyphens=False))


def show_inputs(row, count):
    request = row["request"]
    header(count, row["example"], "INPUT")
    print(f"POST /v1/systemone   model: {request['model']}\n")
    print("\033[1;36mState\033[0m")
    wrapped(request["state"])
    for key, question in request["questions"].items():
        print(f"\n\033[1;36m{key} / {question['type'].upper()}\033[0m")
        wrapped(question["instructions"])
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            for label, description in criteria.items():
                wrapped(f"{label}: {description}", "    ")
        elif isinstance(criteria, list):
            for index, description in enumerate(criteria):
                wrapped(f"{index}: {description}", "    ")
        else:
            wrapped("yes / no (fixed Noul options)", "    ")
    print("\n\033[2mExact request values / next: the model's response\033[0m", flush=True)


def show_result(row, count):
    header(count, row["example"], "RESULT")
    print(json.dumps(compact(row), indent=2, ensure_ascii=False))
    print("\n\033[2mLive response / compact view / not a speed or accuracy benchmark")
    print("Probabilities are conditional and uncalibrated.\033[0m")
    if reason := row.get("metadata", {}).get("x-openjev-fallback"):
        print(f"\033[33mFallback: {reason}\033[0m")
    sys.stdout.flush()


def present(lines):
    destination = Path(os.environ["OPENJEV_RECORDING_DIR"]) / "responses.jsonl"
    count = 0
    with destination.open("w", encoding="utf-8") as raw:
        for count, line in enumerate(lines, 1):
            row = json.loads(line)
            if row["response"]["usage"]["output_tokens"] != 0:
                raise ValueError("unexpected generated tokens in demo response")
            raw.write(line)
            raw.flush()
            mixed = len(row["request"]["questions"]) > 1
            show_inputs(row, count)
            time.sleep(14 if mixed else 8)
            show_result(row, count)
            time.sleep(10 if mixed else 6)
    if count != 8:
        raise ValueError(f"expected eight successful responses, got {count}")
    print("\n\033[1;32m8 HTTP requests complete / one resident model / no generated tokens\033[0m", flush=True)


if __name__ == "__main__":
    present(sys.stdin)
