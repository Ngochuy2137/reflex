"""Interactive single-robot catching, using the evaluator's visual policy."""

import json
import os
from pathlib import Path
import time

from reflex.eval.demo_state import TrialState, parse_settings
from reflex.eval.demo_diagnostics import orientation_angles, summarize_frames


def run_demo(env, policy, app, args, reset_vision):
    import omni.ui as ui
    import torch
    from reflex.envs import throw

    raw = env.unwrapped
    robot, box = raw.scene['robot'], raw.scene['balloon']
    torso_index = robot.data.body_names.index('torso_link')
    runtime = dict(physics_dt=raw.sim.get_physics_dt(), control_dt=raw.step_dt,
                   decimation=raw.cfg.decimation, gravity=list(raw.cfg.sim.gravity),
                   fixed_base=bool(robot.is_fixed_base),
                   gpu_physx=bool(raw.sim.get_physics_context().is_gpu_dynamics_enabled()),
                   arm_gain=float(os.environ.get('EVAL_ARM_GAIN', '1')),
                   action_delay=int(os.environ.get('EVAL_ACT_DELAY', '0')))
    from pxr import UsdPhysics
    physics_scene = UsdPhysics.Scene.Get(raw.sim.stage, raw.cfg.sim.physics_prim_path)
    runtime['usd_gravity_direction'] = list(physics_scene.GetGravityDirectionAttr().Get())
    runtime['usd_gravity_magnitude'] = float(physics_scene.GetGravityMagnitudeAttr().Get())
    runtime['actuators'] = {
        name: dict(stiffness=actuator.cfg.stiffness, damping=actuator.cfg.damping,
                   effort_limit=actuator.cfg.effort_limit_sim)
        for name, actuator in robot.actuators.items()
    }
    print(f'[demo] Runtime: {json.dumps(runtime)}', flush=True)
    default_range = throw.DIST_RANGE
    default_speed = throw.LAUNCH_SPEED_RANGE
    original_sampler = throw.sample_throws
    cpu_device = torch.device(raw.device)
    cuda_devices = [cpu_device.index or 0] if cpu_device.type == 'cuda' else []
    settings = {'seed': 0, 'range': default_range}
    requested = {'reset': True, 'play': False}
    records = []
    sample = {}
    trial = TrialState()

    # Only object sampling uses this seed. Restore CPU/CUDA RNG state afterwards.
    def sample_object(*positional, **keywords):
        old_range = throw.DIST_RANGE
        try:
            throw.DIST_RANGE = settings['range']
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.default_generator.manual_seed(settings['seed'])
                if cuda_devices:
                    with torch.cuda.device(cuda_devices[0]):
                        torch.cuda.manual_seed(settings['seed'])
                result = original_sampler(*positional, **keywords)
            sample.clear()
            sample.update({key: value.detach().cpu().tolist() for key, value in result.items()
                           if torch.is_tensor(value)})
            return result
        finally:
            throw.DIST_RANGE = old_range

    throw.sample_throws = sample_object
    window = ui.Window('Reflex Catching Demo', width=470, height=310)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label('One robot | RGB-D policy | Play one throw, then Reset', word_wrap=True)
            with ui.HStack():
                ui.Label('Distance (m)', width=130)
                distance_field = ui.StringField()
                distance_field.model.set_value(args.demo_distance)
            with ui.HStack():
                ui.Label('Object seed', width=130)
                seed_field = ui.StringField()
                seed_field.model.set_value(args.demo_seed)
            ui.Label(f'Empty distance: random in {default_range[0]}–{default_range[1]} m.\n'
                     'Empty seed: a new seed on each Reset. Changes apply on Reset.',
                     word_wrap=True, height=45)
            with ui.HStack(spacing=8):
                play_button = ui.Button('Play', clicked_fn=lambda: requested.update(play=True))
                ui.Button('Reset', clicked_fn=lambda: requested.update(reset=True))
            status = ui.Label('Preparing...', word_wrap=True, height=55)
            ui.Label('Demo hold indicator only; use eval.sh for benchmark success rates.', word_wrap=True)

    obs = None
    first_signature = None
    smoke_stage = 0
    frozen_frames = 0
    smoke_checks = []
    camera_set = False
    frames = []
    if args.demo_smoke:
        seed_field.model.set_value('9')
    try:
        with torch.inference_mode():
            while app.is_running():
                started = time.monotonic()
                if requested['reset']:
                    requested.update(reset=False, play=False)
                    trial = TrialState()
                    try:
                        distances, seed = parse_settings(distance_field.model.as_string,
                                                        seed_field.model.as_string, default_range)
                        settings.update(seed=seed, range=distances)
                        # Match the existing 5 m/7 m evaluation speed cap for distant throws.
                        throw.LAUNCH_SPEED_RANGE = (default_speed[0], max(default_speed[1], 12.0)) if distances[1] > 4.2 else default_speed
                        # Use the existing reset distribution, without reseeding robot/noise RNGs.
                        obs_dict, _ = env.reset()
                        obs = obs_dict['policy']
                        reset_vision()
                        if not camera_set:
                            x, y = robot.data.root_pos_w[0, :2].tolist()
                            raw.sim.set_camera_view(eye=(x + 3, y - 6, 3), target=(x + 1.5, y, 1))
                            camera_set = True
                        raw.sim.render()
                        trial = TrialState()
                        status.text = f'Ready | object seed {seed} | distance {distances[0]:g}–{distances[1]:g} m'
                        print(f'[demo] {status.text}', flush=True)
                        if args.demo_smoke:
                            signature = {key: sample[key] for key in ['d', 'q0', 'omega', 'aim_err', 'catch_h', 't_mult', 'jitter']}
                            if smoke_stage == 0:
                                first_signature = signature
                            elif smoke_stage == 1:
                                assert signature == first_signature, 'Fixed object seed did not repeat its samples'
                                smoke_checks.append('fixed_seed_repeats_object_samples')
                            elif smoke_stage == 2:
                                assert seed != 9 and signature != first_signature
                                smoke_checks.append('empty_seed_creates_new_throw')
                                break
                            requested['play'] = True
                        if args.demo_audit:
                            requested['play'] = True
                    except ValueError as error:
                        trial.state = 'invalid'
                        status.text = str(error)
                if requested['play']:
                    requested['play'] = False
                    if trial.state == 'ready':
                        trial_started = time.monotonic()
                        initial_position = robot.data.root_pos_w[0].clone()
                        initial_quaternion = robot.data.root_quat_w[0].tolist()
                        w, x, y, z = initial_quaternion
                        forward = torch.tensor([1 - 2*(y*y + z*z), 2*(x*y + w*z)], device=raw.device)
                        forward = forward / forward.norm().clamp_min(1e-6)
                        frames = []
                    trial.play()
                play_button.enabled = trial.state == 'ready'
                if trial.state != 'running':
                    # Render only: physics and policy stay frozen while the user inspects the scene.
                    raw.sim.render()
                    if args.demo_smoke and trial.state == 'finished':
                        frozen_frames += 1
                        assert int(raw.episode_length_buf[0]) == frozen_step
                        assert torch.equal(box.data.root_state_w, frozen_box)
                        assert torch.equal(robot.data.root_state_w, frozen_robot)
                        if frozen_frames == 8:
                            smoke_checks.append('finished_trial_does_not_step_or_reset')
                            smoke_stage += 1
                            if smoke_stage == 2:
                                seed_field.model.set_value('')
                            requested['reset'] = True
                            frozen_frames = 0
                    time.sleep(1 / 60)
                    continue

                policy_started = time.monotonic()
                actions = policy(obs)
                torch.cuda.synchronize(cpu_device)
                policy_seconds = time.monotonic() - policy_started
                step_started = time.monotonic()
                before_physics = raw._sim_step_counter
                obs_dict, _, terminated, truncated, _ = env.step(actions)
                torch.cuda.synchronize(cpu_device)
                step_seconds = time.monotonic() - step_started
                obs = obs_dict['policy']
                if bool(terminated.any() or truncated.any()):
                    raise RuntimeError('Unexpected automatic reset in the demo')
                position = box.data.root_pos_w[0]
                base = robot.data.root_pos_w[0]
                speed = float(box.data.root_lin_vel_w[0].norm())
                launched = bool(raw._box_launched[0])
                ground = float(raw.scene.env_origins[0, 2])
                finite = bool(torch.isfinite(torch.cat([position, base, actions.flatten()])).all())
                held = float((position - base).norm()) < 0.65 and speed < 1.0 and float(position[2] - base[2]) > -0.25
                trial.update(launched=launched, held=held and not args.demo_audit,
                             fell=float(base[2]) - ground < 0.45 and not args.demo_audit,
                             landed=float(position[2]) - ground < 0.2 and not args.demo_audit, finite=finite,
                             limit=args.episode_steps, hold_frames=round(0.6 / raw.step_dt))
                root_tilt, _ = orientation_angles(robot.data.root_quat_w[0].tolist())
                torso_tilt, torso_pitch = orientation_angles(robot.data.body_quat_w[0, torso_index].tolist())
                elapsed = time.monotonic() - trial_started
                simulation = trial.steps * raw.step_dt
                frames.append(dict(step=trial.steps, wall_seconds=elapsed, policy_seconds=policy_seconds,
                                   env_step_seconds=step_seconds, physics_steps=raw._sim_step_counter - before_physics,
                                   root_height_m=float(base[2]) - ground, root_tilt_deg=root_tilt,
                                   torso_tilt_deg=torso_tilt, torso_pitch_deg=torso_pitch,
                                   root_speed_m_s=float(robot.data.root_lin_vel_w[0].norm()),
                                   forward_displacement_m=float(torch.dot((base - initial_position)[:2], forward)),
                                   object_speed_m_s=speed, hold_indicator=held))
                timing = f'Sim {simulation:.2f} s | Wall {elapsed:.2f} s | Speed {simulation / max(elapsed, 1e-9):.2f}x'
                status.text = f'{timing}\nObject seed {settings["seed"]}'
                if trial.steps % 50 == 0:
                    print(f'[demo] {timing} | torso pitch {torso_pitch:.1f} deg', flush=True)
                if trial.state == 'finished':
                    frozen_step = int(raw.episode_length_buf[0])
                    frozen_box = box.data.root_state_w.clone()
                    frozen_robot = robot.data.root_state_w.clone()
                    status.text = f'{trial.result}. Press Reset.\n{timing}'
                    record = dict(object_seed=settings['seed'], distance_range=settings['range'],
                                  steps=trial.steps, result=trial.result, object_sample=dict(sample),
                                  controller=args.bm_ckpt, detector=args.detector,
                                  runtime=runtime, diagnostics=summarize_frames(frames, raw.step_dt), frames=frames)
                    records.append(record)
                    path = Path(args.out)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(dict(trials=records, smoke_checks=smoke_checks), indent=2) + '\n')
                    print(f'[demo] {status.text}', flush=True)
                    if args.demo_audit:
                        break
                if not args.demo_smoke:
                    time.sleep(max(0, raw.step_dt - (time.monotonic() - started)))
    finally:
        throw.sample_throws = original_sampler
        throw.LAUNCH_SPEED_RANGE = default_speed
        window.destroy()
    if args.demo_smoke:
        assert len(records) == 2 and len(smoke_checks) == 4, (records, smoke_checks)
        Path(args.out).write_text(json.dumps(dict(trials=records, smoke_checks=smoke_checks), indent=2) + '\n')
        print('[demo] SMOKE PASS: visual policy, fixed/random seed, frozen outcomes, Reset', flush=True)
