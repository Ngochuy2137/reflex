"""Input checks and trial state for the catching viewer; no simulator imports."""

from dataclasses import dataclass
import math
import secrets


def parse_settings(distance, seed, default_range):
    distance = distance.strip()
    try:
        value = float(distance) if distance else None
    except ValueError as error:
        raise ValueError('Distance must be a positive number, or empty for the default range') from error
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError('Distance must be a positive number, or empty for the default range')
    seed = seed.strip()
    try:
        number = int(seed) if seed else secrets.randbelow(2**32)
    except ValueError as error:
        raise ValueError('Seed must be an integer from 0 to 4294967295, or empty for random') from error
    if not 0 <= number < 2**32:
        raise ValueError('Seed must be an integer from 0 to 4294967295, or empty for random')
    return (value, value) if value is not None else default_range, number


@dataclass
class TrialState:
    state: str = 'ready'
    steps: int = 0
    hold_steps: int = 0
    launched: bool = False
    result: str = ''

    def play(self):
        if self.state == 'ready':
            self.state = 'running'

    def update(self, *, launched, held, fell, landed, finite, limit, hold_frames):
        if self.state != 'running':
            return
        self.steps += 1
        self.launched = self.launched or launched
        self.hold_steps = self.hold_steps + 1 if self.launched and held else 0
        if not finite:
            self.result = 'Invalid physics state'
        elif fell:
            self.result = 'Robot fell'
        elif self.launched and landed:
            self.result = 'Object landed'
        elif self.hold_steps >= hold_frames:
            self.result = 'Caught and held for 0.6 s'
        elif self.steps >= limit:
            self.result = 'Time limit reached'
        if self.result:
            self.state = 'finished'
