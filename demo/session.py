#!/usr/bin/env python3
"""Staged README walkthrough: print illustrative commands/results, execute neither."""
import json
from pathlib import Path
import time

SCENES = Path(__file__).with_name("scenes.json")


def curl_command(scene):
    return ("curl -s http://127.0.0.1:8080/v1/systemone \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            f"  -d '{scene['body']}'")


def type_command(text):
    print("\033[1;36m$ \033[0m", end="", flush=True)
    for char in text:
        print(char, end="", flush=True)
        time.sleep(0.012)
    print(flush=True)


def header(title):
    print("\033[2J\033[H\033[1;36mOPENJEV / " + title + "\033[0m")
    print("\033[2mScripted walkthrough / illustrative responses\033[0m\n", flush=True)


def formatted_response(scene):
    # Keep nested maps on one line so the request and response fit together.
    response = scene["response"]
    answers = response["answers"]
    lines = ["{", f'  "model": {json.dumps(response["model"])},', '  "answers": {']
    for index, (key, answer) in enumerate(answers.items()):
        lines.append(f'    {json.dumps(key)}: {{')
        for field_index, (field, value) in enumerate(answer.items()):
            comma = "," if field_index < len(answer) - 1 else ""
            lines.append(f'      {json.dumps(field)}: {json.dumps(value)}{comma}')
        lines.append('    }' + (',' if index < len(answers) - 1 else ''))
    lines.extend(['  },', f'  "usage": {json.dumps(response["usage"])}', '}'])
    return "\n".join(lines)


def main():
    scenes = json.loads(SCENES.read_text())
    print("\033[?25l", end="", flush=True)
    try:
        header("Start the server")
        type_command("openjev serve --model qwen3-0.6b")
        time.sleep(1)
        print("\nLoading the model...", flush=True)
        time.sleep(1.5)
        print("\033[32mReady at http://127.0.0.1:8080\033[0m")
        print("\nKeep this terminal open. Run curl in another terminal.", flush=True)
        time.sleep(5)
        for scene in scenes:
            header(scene["title"])
            type_command(curl_command(scene))
            time.sleep(8)
            print("\n\033[1;35mResponse\033[0m")
            print(formatted_response(scene), flush=True)
            time.sleep(10)
        print("\n\033[1;32mWALKTHROUGH COMPLETE\033[0m", flush=True)
    finally:
        print("\033[?25h", end="", flush=True)


if __name__ == "__main__":
    main()
