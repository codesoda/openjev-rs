#!/usr/bin/env python3
"""Reproduce the narrowly scoped Qwen3 template-equivalence evidence.

Requires Python 3 and Jinja2 3.1.4. This script performs no network access and
never modifies reference material.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF_TEMPLATE = ROOT / "fixtures/templates/qwen3-gguf-57f1fd00.jinja"
NATIVE_TEMPLATE = ROOT / "fixtures/templates/qwen3-native-a55ee1b1.jinja"
EVIDENCE = ROOT / "fixtures/qwen-template-equivalence.json"
AUTHORED = ROOT / "reference/semif-py/benchmarks/data/authored144.jsonl"
PERTURBATIONS = ROOT / "reference/semif-py/benchmarks/data/perturbations108.jsonl"
PREDICTIONS = ROOT / "reference/semif-py/browser-ladder-qwen3-0.6b.predictions.jsonl"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOP"
EXPECTED_FILES = {
    GGUF_TEMPLATE: (4100, "57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361"),
    NATIVE_TEMPLATE: (4168, "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def messages_for(row: dict) -> list[dict]:
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
        },
    ]


def restricted_render(messages: list[dict]) -> str:
    return (
        "<|im_start|>system\n"
        + messages[0]["content"]
        + "<|im_end|>\n<|im_start|>user\n"
        + messages[1]["content"]
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def main() -> None:
    if importlib.metadata.version("Jinja2") != "3.1.4":
        raise SystemExit("this oracle requires Jinja2==3.1.4")
    from jinja2 import Environment

    sources: dict[Path, str] = {}
    for path, (expected_size, expected_sha) in EXPECTED_FILES.items():
        data = path.read_bytes()
        assert len(data) == expected_size, (path, len(data))
        assert digest(data) == expected_sha, path
        sources[path] = data.decode("utf-8")

    environment = Environment(autoescape=False)
    templates = {
        path: environment.from_string(source) for path, source in sources.items()
    }
    expected_prompts = {
        row["id"]: row["prompt_sha256"] for row in read_jsonl(PREDICTIONS)
    }
    authored = read_jsonl(AUTHORED)
    perturbations = read_jsonl(PERTURBATIONS)
    assert len(authored) == 144
    assert len(perturbations) == 108
    assert len(expected_prompts) == 252

    for row in authored + perturbations:
        messages = messages_for(row)
        restricted = restricted_render(messages)
        assert digest(restricted.encode("utf-8")) == expected_prompts[row["id"]]
        for path, template in templates.items():
            rendered = template.render(
                messages=messages,
                tools=None,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert rendered == restricted, (path, row["id"])

    edge_states = [
        "<tool_response>x</tool_response>",
        "<think>\n</think>",
        "Unicode π, quotes \"double\", apostrophe ' and backslash \\",
        {"second": [True, None, {"z": "末"}], "first": 0},
    ]
    for index, state in enumerate(edge_states):
        row = {
            "state": state,
            "question": f"Edge state {index}",
            "options": [
                {"id": "a", "description": "First"},
                {"id": "b", "description": "Second"},
            ],
        }
        messages = messages_for(row)
        restricted = restricted_render(messages)
        for path, template in templates.items():
            assert (
                template.render(
                    messages=messages,
                    tools=None,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                == restricted
            ), (path, f"edge-{index}")

    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["authored_rows"] == 144
    assert evidence["perturbation_rows"] == 108
    assert evidence["reference_prompt_hashes_matched"] == 252
    assert evidence["edge_states"] == 4
    assert evidence["jinja_version"] == "3.1.4"
    print(
        json.dumps(
            {
                "schema": "openjev-qwen-template-oracle-result-v1",
                "authored": 144,
                "perturbations": 108,
                "reference_prompt_hashes": 252,
                "edge_states": 4,
                "status": "restricted-profile-equivalent",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
