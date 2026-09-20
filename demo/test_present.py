import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import present


def row():
    return {
        "example": "Support routing",
        "request": {"model": "jev-latest", "state": {"ticket": "Charged twice."},
                    "questions": {"route": {"type": "choice", "instructions": "Which team?",
                                            "criteria": {"billing": "Payments", "technical": "Defects"}}}},
        "response": {"model": "qwen3-0.6b", "usage": {"output_tokens": 0},
                     "answers": {"route": {"type": "choice", "choice": "billing", "confidence": 0.8}}},
        "elapsed_ms": 123.456,
        "metadata": {},
    }


class PresentationTests(unittest.TestCase):
    def test_inputs_include_actual_state_question_and_options(self):
        value = row()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            present.show_inputs(value, 1)
        text = output.getvalue()
        for field in ["INPUT", "Charged twice.", "Which team?", "billing: Payments", "technical: Defects"]:
            self.assertIn(field, text)

    def test_compact_preserves_values_without_mutating_row(self):
        value = row()
        before = copy.deepcopy(value)
        projected = present.compact(value)
        self.assertEqual(projected["answers"]["route"], {"type": "choice", "choice": "billing"})
        self.assertEqual(projected["http_ms"], 123.5)
        self.assertEqual(value, before)

    def test_inputs_precede_results_with_separate_reading_pauses(self):
        rows = [row() for _ in range(8)]
        rows[-1]["request"]["questions"]["review"] = {"type": "noul", "instructions": "Human review?"}
        lines = [json.dumps(value) + "\n" for value in rows]
        events = []
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"OPENJEV_RECORDING_DIR": directory}), \
                patch.object(present, "show_inputs", side_effect=lambda *_: events.append("input")), \
                patch.object(present, "show_result", side_effect=lambda *_: events.append("result")), \
                patch.object(present.time, "sleep", side_effect=events.append), \
                contextlib.redirect_stdout(io.StringIO()):
            present.present(iter(lines))
            self.assertEqual((Path(directory)/"responses.jsonl").read_text(), "".join(lines))
        self.assertEqual(events, ["input", 8, "result", 6] * 7 + ["input", 14, "result", 10])


if __name__ == "__main__":
    unittest.main()
