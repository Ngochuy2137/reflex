# SPDX-License-Identifier: MIT
"""NumPy-only helpers shared by the motion rebuild tools.

Nothing here needs Isaac Sim. Every function reproduces, operation for
operation, a step of the pipeline that produced the reference motions used
in the paper, so that rebuilt files match the originals bit for bit (or to
within float32 rounding where noted).

Pipeline overview (see assets/motions/README.md):

  BONES-SEED G1 CSV (120 fps, cm / deg)
    -> load_bones_g1_csv()            SONIC motion_lib conversion, stride-4 -> 30 fps
    -> build_walk_then_catch()        6 s walk + 1 s blend + 6 s hold of our catch pose
    -> round_trip_csv()               BeyondMimic CSV layout, "%.6f" text round trip
    -> fk_replay.replay_motion()      (Isaac Sim) 30 -> 50 fps + forward kinematics
    -> hold_extend()                  pad to 30 s by holding the last pose
"""

from __future__ import annotations

import csv
import io

import numpy as np

# ----------------------------------------------------------------------------
# G1 joint order used by the BONES-SEED CSVs, the SONIC motion_lib "dof" field
# and the BeyondMimic CSV layout (MuJoCo / MJCF actuator order).
# ----------------------------------------------------------------------------
G1_MJ_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Key order of the BeyondMimic motion npz files.
NPZ_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
NPZ_VEL_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")

# ----------------------------------------------------------------------------
# Constants of the reference motion "walk_then_catch_v17" (all at 30 fps).
# ----------------------------------------------------------------------------
BONES_SOURCE_FPS = 120          # BONES-SEED CSVs are sampled at 120 fps
BUILD_FPS = 30                  # motion_lib / build fps (stride 4 downsample)
WALK_START_FRAME = 40           # skip the start transient of the walk clip
WALK_FRAMES = 180               # 6.0 s of walking
TRANSITION_FRAMES = 30          # 1.0 s linear blend walk -> catch pose
CATCH_FRAMES = 180              # 6.0 s hold of the catch pose
OUTPUT_FPS = 50                 # BeyondMimic tracking fps
REF_TOTAL_FRAMES_50 = 1500      # 30 s reference used by the catch policy


# ----------------------------------------------------------------------------
# BONES-SEED CSV -> SONIC motion_lib fields (root_trans_offset, root_rot, dof)
# ----------------------------------------------------------------------------
def _quat_xyzw_from_euler_xyz_deg(euler_deg: np.ndarray) -> np.ndarray:
    """Same call as the SONIC converter: scipy Rotation.from_euler("xyz", deg)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()


def load_bones_g1_csv(csv_path: str, out_fps: int = BUILD_FPS) -> dict:
    """Load a BONES-SEED G1 CSV and convert it like SONIC's motion_lib converter.

    Mirrors gear_sonic/data_process/convert_soma_csv_to_motion_lib.py
    (load_bones_csv -> convert_sequence -> downsample_sequence with
    fps_source=120, fps=30). Returns float32 arrays at ``out_fps``:
      root_trans_offset (T,3) [m], root_rot (T,4) xyzw, dof (T,29) [rad].
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in r] for r in reader if r]
    data = np.asarray(rows, dtype=np.float64)
    col = {name: i for i, name in enumerate(header)}

    root_pos = (
        np.stack([data[:, col["root_translateX"]], data[:, col["root_translateY"]],
                  data[:, col["root_translateZ"]]], axis=1).astype(np.float32) / 100.0
    )  # cm -> m (division done in float32, as in the original)
    euler_deg = np.stack([data[:, col["root_rotateX"]], data[:, col["root_rotateY"]],
                          data[:, col["root_rotateZ"]]], axis=1).astype(np.float64)
    root_quat_xyzw = _quat_xyzw_from_euler_xyz_deg(euler_deg).astype(np.float32)
    joint_cols = [col[n + "_dof"] for n in G1_MJ_JOINT_NAMES]
    dof = np.deg2rad(data[:, joint_cols]).astype(np.float32)  # deg -> rad

    # The original converter goes xyzw -> wxyz -> xyzw; both are pure reindexing.
    jump = int(BONES_SOURCE_FPS / out_fps)
    return {
        "root_trans_offset": root_pos[::jump].astype(np.float32),
        "root_rot": root_quat_xyzw[::jump].astype(np.float32),
        "dof": dof[::jump].astype(np.float32),
        "fps": out_fps,
    }


# ----------------------------------------------------------------------------
# Walk -> blend -> catch-hold track at 30 fps
# ----------------------------------------------------------------------------
def build_walk_then_catch(walk: dict, catch_dof: np.ndarray) -> dict:
    """Concatenate 6 s walk + 1 s linear blend + 6 s hold of ``catch_dof``.

    Reproduces build_walk_then_catch.py (SONIC pkl "walk_then_catch__U001"):
      * walk frames [40, 220) of the 30 fps clip;
      * root x locked to 0, root y made relative to the first walk frame,
        root z and root rotation kept from the clip;
      * blend: alpha = (i+1)/30, dof only; root held at the last walk frame;
      * hold: catch_dof, root held at the last walk frame.
    Returns float32 arrays (390 frames): root_trans_offset, root_rot (xyzw), dof.
    """
    catch_dof = np.asarray(catch_dof, dtype=np.float32).reshape(29)
    a, b = WALK_START_FRAME, WALK_START_FRAME + WALK_FRAMES
    walk_dof = walk["dof"][a:b]
    walk_root_trans = walk["root_trans_offset"][a:b].copy()
    walk_root_rot = walk["root_rot"][a:b]

    walk_start_pos = walk_root_trans[0].copy()
    walk_root_trans_rel = walk_root_trans.copy()
    walk_root_trans_rel[:, 0] = 0.0
    walk_root_trans_rel[:, 1] -= walk_start_pos[1]

    walk_dof_last = walk_dof[-1].copy()
    root_trans_last = walk_root_trans_rel[-1].copy()
    root_rot_last = walk_root_rot[-1].copy()

    trans_dof = np.zeros((TRANSITION_FRAMES, 29), dtype=np.float32)
    trans_root_trans = np.zeros((TRANSITION_FRAMES, 3), dtype=np.float32)
    trans_root_rot = np.zeros((TRANSITION_FRAMES, 4), dtype=np.float32)
    for i in range(TRANSITION_FRAMES):
        alpha = (i + 1) / TRANSITION_FRAMES  # python float, as in the original
        trans_dof[i] = (1 - alpha) * walk_dof_last + alpha * catch_dof
        trans_root_trans[i] = root_trans_last
        trans_root_rot[i] = root_rot_last

    hold_dof = np.tile(catch_dof, (CATCH_FRAMES, 1))
    hold_root_trans = np.tile(root_trans_last[None, :], (CATCH_FRAMES, 1))
    hold_root_rot = np.tile(root_rot_last[None, :], (CATCH_FRAMES, 1))

    return {
        "root_trans_offset": np.concatenate([walk_root_trans_rel, trans_root_trans, hold_root_trans]).astype(np.float32),
        "root_rot": np.concatenate([walk_root_rot, trans_root_rot, hold_root_rot]).astype(np.float32),
        "dof": np.concatenate([walk_dof, trans_dof, hold_dof]).astype(np.float32),
        "fps": BUILD_FPS,
    }


def motion_to_bm_array(track: dict) -> np.ndarray:
    """(T,36) = [x, y, z, qx, qy, qz, qw, 29 dof] (BeyondMimic CSV layout)."""
    return np.concatenate([
        np.asarray(track["root_trans_offset"], dtype=np.float64),
        np.asarray(track["root_rot"], dtype=np.float64),
        np.asarray(track["dof"], dtype=np.float64),
    ], axis=1)


def round_trip_csv(arr: np.ndarray, fmt: str | None = "%.6f", path: str | None = None) -> np.ndarray:
    """Write ``arr`` as CSV text (``fmt``) and parse it back, like the original
    sonic_pkl_to_csv.py (np.savetxt fmt="%.6f") + csv_to_npz.py (np.loadtxt).
    ``fmt=None`` skips the text round trip (returns float64 copy).
    If ``path`` is given the CSV is also written there for inspection."""
    arr = np.asarray(arr, dtype=np.float64)
    if fmt is None:
        out = arr.copy()
        if path:
            np.savetxt(path, out, delimiter=",")
        return out
    buf = io.StringIO()
    np.savetxt(buf, arr, delimiter=",", fmt=fmt)
    text = buf.getvalue()
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return np.loadtxt(io.StringIO(text), delimiter=",")


# ----------------------------------------------------------------------------
# Post-processing of BeyondMimic npz dicts (50 fps)
# ----------------------------------------------------------------------------
def hold_extend(motion: dict, total_frames: int) -> dict:
    """Extend to ``total_frames`` by repeating the last pose with zero velocities."""
    T = motion["joint_pos"].shape[0]
    n = total_frames - T
    if n < 0:
        raise ValueError(f"motion already has {T} > {total_frames} frames")
    out = {"fps": np.asarray(motion["fps"]).reshape(1).astype(np.int64)}
    for k in NPZ_KEYS:
        v = np.asarray(motion[k])
        tail = np.zeros((n,) + v.shape[1:], v.dtype) if k in NPZ_VEL_KEYS else np.repeat(v[-1:], n, axis=0)
        out[k] = np.concatenate([v, tail], axis=0)
    return out


def segment_then_hold(motion: dict, start: int, stop: int, total_frames: int) -> dict:
    """Frames [start, stop) followed by a hold of frame stop-1 (zero velocities)."""
    seg = {"fps": motion["fps"]}
    for k in NPZ_KEYS:
        seg[k] = np.asarray(motion[k])[start:stop]
    return hold_extend(seg, total_frames)


def save_npz(path: str, motion: dict) -> None:
    np.savez(path, fps=np.asarray(motion["fps"]).reshape(1).astype(np.int64),
             **{k: np.asarray(motion[k]) for k in NPZ_KEYS})


def load_npz(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


# ----------------------------------------------------------------------------
# Own data loader
# ----------------------------------------------------------------------------
def load_catch_pose(path: str) -> np.ndarray:
    """Read the 29-dof catch pose CSV (columns: joint, value_rad) -> float32 (29,)."""
    vals = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0] == "joint":
                continue
            vals[row[0].strip()] = float(row[1])
    missing = [n for n in G1_MJ_JOINT_NAMES if n not in vals]
    if missing:
        raise ValueError(f"catch pose file is missing joints: {missing}")
    return np.asarray([vals[n] for n in G1_MJ_JOINT_NAMES], dtype=np.float32)
