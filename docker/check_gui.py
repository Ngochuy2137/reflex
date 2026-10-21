"""Open a short GUI scene and check a falling cube on GPU PhysX."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--steps', type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless or args.steps < 120:
    parser.error('Use GUI mode and at least 120 steps')
print('Starting the desktop GUI check...', flush=True)
launcher = AppLauncher(args)
print('GUI application initialized.', flush=True)
sim = None
completed = False
try:
    import omni.appwindow
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    # Use local geometry so the check needs no remote ground-plane asset.
    floor = sim_utils.CuboidCfg(
        size=(10, 10, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)))
    floor.func('/World/Ground', floor, translation=(0, 0, -0.05))
    light = sim_utils.DomeLightCfg(intensity=2500)
    light.func('/World/Light', light)
    cube = RigidObject(RigidObjectCfg(
        prim_path='/World/Cube',
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.9))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 1))))
    sim.set_camera_view((2, 2, 1.5), (0, 0, 0.4))
    sim.reset()
    print('Scene and physics initialized.', flush=True)
    gpu_physics = sim.get_physics_context().is_gpu_dynamics_enabled()
    if not gpu_physics:
        raise RuntimeError('GPU PhysX dynamics are not enabled')
    window = omni.appwindow.get_default_app_window()
    if window is None or not window.get_window():
        raise RuntimeError('No desktop window was created')
    for _ in range(args.steps):
        if not launcher.app.is_running():
            raise RuntimeError('Window closed before the check finished')
        sim.step()
        cube.update(sim.get_physics_dt())
        if not torch.isfinite(cube.data.root_state_w).all():
            raise RuntimeError('Non-finite cube state')
        time.sleep(0.02)
    height = float(cube.data.root_pos_w[0, 2])
    if cube.data.root_state_w.device.type != 'cuda' or not 0.08 < height < 0.15:
        raise RuntimeError(f'GPU falling-cube check failed: final height={height}')
    result = dict(passed=True, desktop_window=True, gpu_physx=bool(gpu_physics),
                  physics_device=str(cube.data.root_state_w.device),
                  gpu=torch.cuda.get_device_name(), steps=args.steps, final_cube_height_m=height)
    output = Path('outputs/gui_check.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: desktop GUI, rendering, and falling cube on GPU PhysX', flush=True)
    completed = True
finally:
    if completed:
        # Only this standalone check exits here, after saving and flushing its result.
        # Native GUI teardown can stall; the OS releases this process's GPU resources.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if sim is not None:
        sim.stop()
    launcher.app.close(wait_for_replicator=False)
