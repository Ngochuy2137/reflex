"""Add wall-clock time estimates to the runner's existing progress log."""

import time


def estimate_time(elapsed, completed, total):
    """Estimate this training call from its completed iterations."""
    if elapsed < 0 or not 0 < completed <= total:
        raise ValueError('Invalid elapsed time or iteration counts')
    average = elapsed / completed
    remaining = average * (total - completed)
    return average, remaining, elapsed + remaining


def format_duration(seconds):
    """Show hours without wrapping after one day."""
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def make_timed_runner(base_runner):
    """Extend logging while keeping the original learning and saving methods."""
    class TimedRunner(base_runner):
        def learn(self, *args, **kwargs):
            self._progress_start = time.monotonic()
            return super().learn(*args, **kwargs)

        def log(self, locs, width=80, pad=35):
            super().log(locs, width=width, pad=pad)
            if getattr(self, 'disable_logs', False):
                return
            completed = locs['it'] - locs['start_iter'] + 1
            total = locs['num_learning_iterations']
            elapsed = time.monotonic() - self._progress_start
            average, remaining, estimated_total = estimate_time(elapsed, completed, total)
            print(f"{'Iteration progress:':>{pad}} {completed}/{total} (this run)")
            print(f"{'Average iteration time:':>{pad}} {average:.2f}s")
            print(f"{'Estimated remaining time:':>{pad}} {format_duration(remaining)}")
            print(f"{'Estimated total time:':>{pad}} {format_duration(estimated_total)}", flush=True)
            writer = getattr(self, 'writer', None)
            if writer is not None:
                for name, value in (
                    ('completed_iterations', completed),
                    ('average_iteration_seconds', average),
                    ('remaining_seconds', remaining),
                    ('estimated_total_seconds', estimated_total),
                ):
                    writer.add_scalar(f'Perf/{name}', value, locs['it'])

    return TimedRunner
