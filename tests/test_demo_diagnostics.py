"""Verify units and time ratios used in the demo's diagnostic log."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/reflex'))
from reflex.eval.demo_diagnostics import orientation_angles, summarize_frames


class DiagnosticTests(unittest.TestCase):
    def test_orientation_wxyz_and_backward_pitch(self):
        self.assertEqual(orientation_angles([1, 0, 0, 0]), (0, 0))
        angle = math.radians(-30)
        tilt, pitch = orientation_angles([math.cos(angle/2), 0, math.sin(angle/2), 0])
        self.assertAlmostEqual(tilt, 30)
        self.assertAlmostEqual(pitch, -30)

    def test_ratio_uses_simulated_time_and_real_elapsed_time(self):
        frame = dict(step=100, wall_seconds=6, policy_seconds=0.5, env_step_seconds=1,
                     root_height_m=0.7, root_tilt_deg=20, torso_tilt_deg=30,
                     torso_pitch_deg=-30, forward_displacement_m=-0.2,
                     root_speed_m_s=0.6, physics_steps=4)
        summary = summarize_frames([frame], 0.02)
        self.assertEqual(summary['simulation_seconds'], 2)
        self.assertAlmostEqual(summary['speed_ratio'], 1/3)
        self.assertEqual(summary['forward_displacement_final_m'], -0.2)
        self.assertEqual(summary['physics_steps_min'], 4)
        self.assertEqual(summarize_frames([], 0.02), {})


if __name__ == '__main__':
    unittest.main()
