"""Check optional logging without an account, GPU, or network connection."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/reflex'))
from reflex.utils.experiment_logging import ExperimentLogger, finish_ppo_logging, scalar_metrics
from reflex.eval import summarize


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.run = SimpleNamespace(
            id='test1234', name='test', project='reflex', group='experiment_01',
            settings=SimpleNamespace(mode='offline'), summary={}, log=Mock(), finish=Mock())
        self.sdk = SimpleNamespace(init=Mock(return_value=self.run), run=self.run, finish=Mock())
        self.env = dict(REFLEX_LOGGER='wandb', WANDB_MODE='offline', WANDB_PROJECT='reflex',
                        WANDB_RUN_GROUP='experiment_01', WANDB_NAME='stage2_bc',
                        WANDB_DIR=str(self.folder))

    def test_disabled_does_not_start_sdk(self):
        with patch.dict(os.environ, {'REFLEX_LOGGER': 'none'}), patch.dict(sys.modules, wandb=self.sdk):
            logger = ExperimentLogger('stage2_bc', {})
            logger.log({'loss': 1})
            logger.finish()
        self.sdk.init.assert_not_called()

    def test_finite_metrics_and_vectors(self):
        self.assertEqual(scalar_metrics({'loss': 1.2, 'bad': float('nan'), 'missing': None,
                                         'test': {'pos': [1, float('inf'), 3]}}),
                         {'loss': 1.2, 'test/pos/0': 1, 'test/pos/2': 3})

    def test_group_mode_metadata_and_finish(self):
        with patch.dict(os.environ, self.env), patch.dict(sys.modules, wandb=self.sdk):
            logger = ExperimentLogger('stage2_bc', {'seed': 11})
            logger.log({'train/loss': 0.2, 'undefined': float('nan')}, step=1)
            logger.finish({'best': {'loss': 0.2}})
        kwargs = self.sdk.init.call_args.kwargs
        self.assertEqual((kwargs['mode'], kwargs['group'], kwargs['job_type']),
                         ('offline', 'experiment_01', 'stage2_bc'))
        self.run.log.assert_called_once_with({'train/loss': 0.2}, step=1)
        self.assertEqual(self.run.summary, {'best/loss': 0.2})
        info = json.loads((self.folder / 'links/test1234.json').read_text())
        self.assertIsNone(info['url'])
        self.run.finish.assert_called_once()

    def test_native_ppo_failure_is_not_marked_successful(self):
        with patch.dict(sys.modules, wandb=self.sdk):
            finish_ppo_logging(self.folder, exit_code=1)
        self.sdk.finish.assert_called_once_with(exit_code=1)
        self.assertTrue((self.folder / 'wandb_run.json').is_file())

    def evaluate(self, throws):
        path = self.folder / 'eval.json'
        path.write_text(json.dumps({'throw_log': throws}))
        with patch.dict(os.environ, self.env), patch.dict(sys.modules, wandb=self.sdk), \
                patch.object(sys, 'argv', ['summarize', str(path), '--condition', 'regular']):
            summarize.main()
        return self.run.log.call_args.args[0]

    def test_evaluation_counts_only_first_throws(self):
        def throw(idx, caught, fell):
            return dict(idx=idx, caught_twin=caught, dropped_after=False,
                        fell=fell, end_v=0.2, intercepted=caught)
        metrics = self.evaluate([throw(1, True, False), throw(1, False, True), throw(2, True, False)])
        self.assertEqual(metrics['eval/throws'], 2)
        self.assertEqual(metrics['eval/catch_success_rate'], 0.5)
        self.assertEqual(metrics['eval/fall_rate'], 0.5)

    def test_no_throws_does_not_log_a_misleading_zero_rate(self):
        metrics = self.evaluate([])
        self.assertEqual(metrics['eval/throws'], 0)
        self.assertNotIn('eval/catch_success_rate', metrics)


if __name__ == '__main__':
    unittest.main()
