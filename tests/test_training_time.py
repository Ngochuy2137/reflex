"""Check progress estimates without a simulator or GPU."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/reflex'))
from reflex.utils.training_time import estimate_time, format_duration, make_timed_runner


class TimingTests(unittest.TestCase):
    def test_average_and_final_iteration(self):
        self.assertEqual(estimate_time(10, 2, 5), (5, 15, 25))
        self.assertEqual(estimate_time(25, 5, 5), (5, 0, 25))
        self.assertEqual(format_duration(90061), '25:01:01')

    def test_resume_uses_only_current_run_and_keeps_base_methods(self):
        class BaseRunner:
            def learn(self, **kwargs):
                self.log(dict(it=901, start_iter=900, num_learning_iterations=5))
                return 'original result'

            def log(self, locs, width=80, pad=35):
                print('Original runner log')

        runner = make_timed_runner(BaseRunner)()
        runner.writer = Mock()
        output = io.StringIO()
        with patch('reflex.utils.training_time.time.monotonic', side_effect=[100, 110]), \
                contextlib.redirect_stdout(output):
            result = runner.learn(num_learning_iterations=5)
        self.assertEqual(result, 'original result')
        self.assertIn('Original runner log', output.getvalue())
        self.assertIn('2/5 (this run)', output.getvalue())
        self.assertIn('00:00:15', output.getvalue())
        self.assertIn('00:00:25', output.getvalue())
        runner.writer.add_scalar.assert_any_call('Perf/remaining_seconds', 15, 901)

    def test_final_progress_without_writer(self):
        class BaseRunner:
            def learn(self):
                self.log(dict(it=4, start_iter=0, num_learning_iterations=5))

            def log(self, locs, width=80, pad=35):
                pass

        output = io.StringIO()
        with patch('reflex.utils.training_time.time.monotonic', side_effect=[100, 125]), \
                contextlib.redirect_stdout(output):
            make_timed_runner(BaseRunner)().learn()
        self.assertIn('5/5 (this run)', output.getvalue())
        self.assertIn('Estimated remaining time: 00:00:00', output.getvalue())


if __name__ == '__main__':
    unittest.main()
