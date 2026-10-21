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


class ProgressTimer:
    """Report time for one collection block or training phase, not a whole pipeline."""

    def __init__(self, label, total, unit='steps', every=1):
        if total <= 0 or every <= 0:
            raise ValueError('Total and report interval must be positive')
        self.label, self.total, self.unit, self.every = label, total, unit, every
        self.started = time.monotonic()
        self.last_report = 0
        print(f'[progress] {label}: 0/{total} {unit}; waiting for the first measurement', flush=True)

    def update(self, completed):
        """Print at the chosen interval and return numeric metrics for an optional logger."""
        if not 0 < completed <= self.total:
            raise ValueError('Completed count is outside this phase')
        if completed <= self.last_report:
            return {}
        if completed != self.total and completed % self.every:
            return {}
        self.last_report = completed
        elapsed = time.monotonic() - self.started
        average, remaining, total = estimate_time(elapsed, completed, self.total)
        print(f'[progress] {self.label}: {completed}/{self.total} {self.unit} | '
              f'average: {average:.3f}s/{self.unit} | elapsed: {format_duration(elapsed)} | '
              f'estimated remaining: {format_duration(remaining)} | '
              f'estimated phase total: {format_duration(total)}', flush=True)
        return {'Perf/completed': completed, 'Perf/total': self.total,
                'Perf/average_unit_seconds': average, 'Perf/elapsed_seconds': elapsed,
                'Perf/remaining_seconds': remaining, 'Perf/estimated_total_seconds': total}


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
