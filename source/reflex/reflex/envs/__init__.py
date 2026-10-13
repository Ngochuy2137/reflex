"""Reflex box-catch environments (Isaac Lab, Unitree G1).

Task configs for the Stage 1 teacher, the ballistic box thrower and catch reward
terms, the batched box Kalman filter, and the wrist-roll reflex action term.
"""
# Performance: older MDP term modules call Isaac Lab's deprecated
# quat_rotate_inverse, which logs a warning on EVERY call and is the slower
# implementation.  Alias it to quat_apply_inverse BEFORE those modules are
# imported (this package initialises before the env cfgs import their MDP terms).
import isaaclab.utils.math as _ilm  # noqa: E402

_ilm.quat_rotate_inverse = _ilm.quat_apply_inverse
