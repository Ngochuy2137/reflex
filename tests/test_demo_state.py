"""Check demo inputs and trial boundaries without simulator dependencies."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/reflex'))
from reflex.eval.demo_state import TrialState, parse_settings


class DemoTests(unittest.TestCase):
    def test_fixed_and_empty_inputs(self):
        self.assertEqual(parse_settings('5', '9', (3, 3.5)), ((5, 5), 9))
        with patch('reflex.eval.demo_state.secrets.randbelow', return_value=123):
            self.assertEqual(parse_settings('', '', (3, 3.5)), ((3, 3.5), 123))

    def test_invalid_settings(self):
        for distance, seed in [('nan', '9'), ('inf', '9'), ('0', '9'), ('-1', '9'),
                               ('1', '-1'), ('1', '3.2'), ('1', str(2**32))]:
            with self.subTest(distance=distance, seed=seed), self.assertRaises(ValueError):
                parse_settings(distance, seed, (3, 3.5))

    def update(self, trial, **changes):
        values = dict(launched=True, held=True, fell=False, landed=False, finite=True,
                      limit=640, hold_frames=30)
        values.update(changes)
        trial.update(**values)

    def test_play_once_and_hold_then_freeze(self):
        trial = TrialState()
        self.update(trial)
        self.assertEqual(trial.steps, 0)
        trial.play()
        for _ in range(29):
            self.update(trial)
        self.assertEqual(trial.state, 'running')
        self.update(trial)
        self.assertEqual(trial.state, 'finished')
        trial.play()
        self.update(trial)
        self.assertEqual(trial.steps, 30)

    def test_hold_requires_consecutive_launched_frames(self):
        trial = TrialState()
        trial.play()
        self.update(trial, launched=False)
        self.assertEqual(trial.hold_steps, 0)
        for _ in range(20):
            self.update(trial)
        self.update(trial, held=False)
        self.assertEqual(trial.hold_steps, 0)

    def test_failures_and_timeout(self):
        for change in [dict(fell=True), dict(landed=True), dict(finite=False), dict(limit=1)]:
            trial = TrialState()
            trial.play()
            self.update(trial, **change)
            self.assertEqual(trial.state, 'finished')


if __name__ == '__main__':
    unittest.main()
