"""Small, simulator-independent timing and posture measurements for the viewer."""

import math


def orientation_angles(quaternion):
    """Return tilt and signed pitch in degrees for a wxyz quaternion."""
    w, x, y, z = quaternion
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x*x + y*y)))))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w*y - z*x)))))
    return tilt, pitch


def summarize_frames(frames, control_dt):
    if not frames:
        return {}
    final = frames[-1]
    simulation = final['step'] * control_dt
    wall = final['wall_seconds']
    return dict(simulation_seconds=simulation, wall_seconds=wall,
                speed_ratio=simulation / wall if wall > 0 else 0,
                policy_seconds=sum(f['policy_seconds'] for f in frames),
                env_step_seconds=sum(f['env_step_seconds'] for f in frames),
                root_height_min_m=min(f['root_height_m'] for f in frames),
                root_tilt_max_deg=max(f['root_tilt_deg'] for f in frames),
                torso_tilt_max_deg=max(f['torso_tilt_deg'] for f in frames),
                torso_pitch_min_deg=min(f['torso_pitch_deg'] for f in frames),
                torso_pitch_max_deg=max(f['torso_pitch_deg'] for f in frames),
                forward_displacement_min_m=min(f['forward_displacement_m'] for f in frames),
                forward_displacement_final_m=final['forward_displacement_m'],
                root_speed_max_m_s=max(f['root_speed_m_s'] for f in frames),
                physics_steps_min=min(f['physics_steps'] for f in frames),
                physics_steps_max=max(f['physics_steps'] for f in frames))
