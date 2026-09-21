import contextlib
import io
import json
from pathlib import Path
import shlex
import unittest
from unittest.mock import patch

import session


class WalkthroughTests(unittest.TestCase):
    def setUp(self):
        self.scenes = json.loads(session.SCENES.read_text())

    def test_curl_commands_contain_valid_requests_with_matching_answers(self):
        self.assertEqual(len(self.scenes), 2)
        for scene in self.scenes:
            args = shlex.split(session.curl_command(scene))
            request = json.loads(args[args.index('-d') + 1])
            self.assertEqual(request, json.loads(scene['body']))
            self.assertEqual(set(request['questions']), set(scene['response']['answers']))
            for key, question in request['questions'].items():
                answer = scene['response']['answers'][key]
                self.assertEqual(question['type'], answer['type'])
                if question['type'] == 'choice':
                    self.assertIn(answer['choice'], question['criteria'])
                else:
                    self.assertTrue(0 <= answer['noul'] <= 1)
            self.assertEqual(scene['response']['usage']['output_tokens'], 0)

    def test_displayed_json_is_the_complete_illustrative_response(self):
        for scene in self.scenes:
            self.assertEqual(json.loads(session.formatted_response(scene)), scene['response'])

    def test_docs_contain_the_same_copyable_commands(self):
        docs = (Path(__file__).resolve().parents[1] / 'docs/demo.md').read_text()
        for scene in self.scenes:
            self.assertIn(session.curl_command(scene), docs)

    def test_startup_then_two_scenes_have_separate_reading_pauses(self):
        with patch.object(session, 'type_command') as commands, \
                patch.object(session.time, 'sleep') as sleep, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            session.main()
        self.assertEqual(commands.call_args_list[0].args[0], 'openjev serve --model qwen3-0.6b')
        self.assertEqual(len(commands.call_args_list), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 1.5, 5, 8, 10, 8, 10])
        self.assertIn('illustrative responses', output.getvalue())
        self.assertIn('WALKTHROUGH COMPLETE', output.getvalue())


if __name__ == '__main__':
    unittest.main()
