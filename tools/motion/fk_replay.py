# SPDX-License-Identifier: MIT
"""Kinematic replay (Isaac Lab) that turns a root + joint trajectory into a
BeyondMimic motion npz (joint_pos/vel + world poses/velocities of all bodies).

This is a function-level port of BeyondMimic's ``scripts/csv_to_npz.py``
(whole_body_tracking, MIT). The arithmetic (float32 on the simulation device,
lerp / slerp resampling, finite-difference velocities, PhysX forward
kinematics without stepping physics) is kept identical so that the output is
bit-identical to files produced by that script.

Import this module only AFTER ``isaaclab.app.AppLauncher`` has started the app.
"""

from __future__ import annotations

import importlib

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

# Joint order of the input trajectory columns 7..35 (MuJoCo / MJCF order).
REPLAY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def load_robot_cfg(spec: str) -> ArticulationCfg:
    """``spec`` = "package.module:ATTRIBUTE", e.g. whole_body_tracking.robots.g1:G1_CYLINDER_CFG."""
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)


def make_scene(robot_cfg: ArticulationCfg, output_fps: int, device: str):
    """Same scene as csv_to_npz.py: one robot, ground plane, dome light.
    (The dome light texture is omitted; lighting does not affect kinematics.)"""

    @configclass
    class ReplaySceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = sim_utils.SimulationCfg(device=device)
    sim_cfg.dt = 1.0 / output_fps
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    return sim, scene


class _Resampler:
    """csv_to_npz.py MotionLoader, fed from an in-memory (T,36) float64 array."""

    def __init__(self, motion: np.ndarray, input_fps: int, output_fps: int, device, z_offset: float = 0.0):
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        m = torch.from_numpy(np.asarray(motion, dtype=np.float64)).to(torch.float32).to(device)
        self.base_pos_in = m[:, :3].clone()
        if z_offset != 0.0:
            self.base_pos_in[:, 2] += z_offset
        self.base_rot_in = m[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
        self.dof_pos_in = m[:, 7:]
        self.input_frames = m.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        self._interpolate()
        self._velocities()

    def _interpolate(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        phase = times / self.duration
        i0 = (phase * (self.input_frames - 1)).floor().long()
        i1 = torch.minimum(i0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - i0
        lerp = lambda a, b, w: a * (1 - w) + b * w  # noqa: E731
        self.base_pos = lerp(self.base_pos_in[i0], self.base_pos_in[i1], blend.unsqueeze(1))
        rots = torch.zeros_like(self.base_rot_in[i0])
        a, b = self.base_rot_in[i0], self.base_rot_in[i1]
        for k in range(a.shape[0]):
            rots[k] = quat_slerp(a[k], b[k], blend[k])
        self.base_rot = rots
        self.dof_pos = lerp(self.dof_pos_in[i0], self.dof_pos_in[i1], blend.unsqueeze(1))

    def _velocities(self):
        self.base_lin_vel = torch.gradient(self.base_pos, spacing=self.output_dt, dim=0)[0]
        self.dof_vel = torch.gradient(self.dof_pos, spacing=self.output_dt, dim=0)[0]
        q_prev, q_next = self.base_rot[:-2], self.base_rot[2:]
        omega = axis_angle_from_quat(quat_mul(q_next, quat_conjugate(q_prev))) / (2.0 * self.output_dt)
        self.base_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def replay_motion(sim, scene, motion: np.ndarray, input_fps: int, output_fps: int = 50,
                  z_offset: float = 0.0) -> dict:
    """Resample ``motion`` (T,36: pos xyz, quat xyzw, 29 joints in REPLAY_JOINT_NAMES order)
    to ``output_fps`` and record PhysX forward kinematics. Returns a npz-ready dict."""
    mot = _Resampler(motion, input_fps, output_fps, sim.device, z_offset)
    robot = scene["robot"]
    jidx = robot.find_joints(REPLAY_JOINT_NAMES, preserve_order=True)[0]
    log = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    for t in range(mot.output_frames):
        s = slice(t, t + 1)
        root = robot.data.default_root_state.clone()
        root[:, :3] = mot.base_pos[s]
        root[:, :2] += scene.env_origins[:, :2]
        root[:, 3:7] = mot.base_rot[s]
        root[:, 7:10] = mot.base_lin_vel[s]
        root[:, 10:] = mot.base_ang_vel[s]
        robot.write_root_state_to_sim(root)
        jp = robot.data.default_joint_pos.clone()
        jv = robot.data.default_joint_vel.clone()
        jp[:, jidx] = mot.dof_pos[s]
        jv[:, jidx] = mot.dof_vel[s]
        robot.write_joint_state_to_sim(jp, jv)
        sim.render()  # no physics step: kinematics only
        scene.update(sim.get_physics_dt())
        d = robot.data
        log["joint_pos"].append(d.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(d.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(d.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(d.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(d.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(d.body_ang_vel_w[0, :].cpu().numpy().copy())
    out = {"fps": np.asarray([output_fps], dtype=np.int64)}
    for k, v in log.items():
        out[k] = np.stack(v, axis=0)
    return out
