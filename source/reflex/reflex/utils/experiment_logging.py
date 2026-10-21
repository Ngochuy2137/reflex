"""Optional W&B logging shared by BC, visual training, and evaluation."""

import json
import math
from numbers import Real
import os
from pathlib import Path


def scalar_metrics(values, prefix=''):
    """Flatten numeric values; omit undefined metrics rather than logging zeros."""
    result = {}
    for key, value in values.items():
        name = f'{prefix}/{key}' if prefix else str(key)
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (list, tuple)):
            result.update(scalar_metrics(dict(enumerate(value)), name))
        elif isinstance(value, Real) and math.isfinite(value):
            result[name] = value
    return result


def save_run_info(run, path):
    """Save a link and identifiers without credentials or raw input data."""
    if run is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    online = run.settings.mode == 'online'
    info = dict(id=run.id, name=run.name, project=run.project, group=run.group,
                mode=run.settings.mode, url=run.url if online else None)
    path.write_text(json.dumps(info, indent=2) + '\n')
    print(f'[logging] Run details: {path}', flush=True)


class ExperimentLogger:
    """Create one optional run per training process; leave learning unchanged."""

    def __init__(self, job_type, config):
        self.run = None
        if os.environ.get('REFLEX_LOGGER') != 'wandb':
            return
        import wandb
        folder = Path(os.environ.get('WANDB_DIR', 'logs/wandb'))
        folder.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'reflex'),
            entity=os.environ.get('WANDB_ENTITY') or None,
            group=os.environ.get('WANDB_RUN_GROUP') or None,
            name=os.environ.get('WANDB_NAME', job_type), job_type=job_type,
            config=config, dir=str(folder),
            mode=os.environ.get('WANDB_MODE', 'online'), save_code=False)
        save_run_info(self.run, folder / 'links' / f'{self.run.id}.json')

    def log(self, values, step=None):
        if self.run is not None:
            self.run.log(scalar_metrics(values), step=step)

    def finish(self, summary=None):
        if self.run is not None:
            if summary:
                self.run.summary.update(scalar_metrics(summary))
            self.run.finish()


def finish_ppo_logging(log_dir, exit_code=0):
    """Finish the native RSL-RL run and save its dashboard link locally."""
    import wandb
    if wandb.run is not None:
        save_run_info(wandb.run, Path(log_dir) / 'wandb_run.json')
        wandb.finish(exit_code=exit_code)
