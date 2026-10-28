"""The Sock Heist's rules for training: the event's numbers and the Arena's arithmetic, copied.

The training job installs no Arena, so nothing here is imported from it;
tests/test_sock_heist_6m.py pins each copy against it.
"""
import math

import torch

CTRL_DT = 0.02                 # arena.constants
SETTLE_S = 0.5                 # [start] settle_s
FIRST_GREEN_S = (0.6, 1.0)     # [lights] first_green_s
GREEN_S = (1.6, 3.0)           # [lights] green_s
RED_S = (1.4, 2.4)             # [lights] red_s
TURN_S = 0.4                   # [lights] turn_s
TOLERANCE_M = 0.03             # [lights] tolerance_m
LOOKAHEAD_M = 0.5              # [steering] lookahead_m
GAIN = 2.0                     # [steering] gain
MAX_YAW_RATE = 1.0             # [steering] max_yaw_rate


def steps(seconds: float) -> int:
    return round(seconds / CTRL_DT)


def _draw(span: tuple[float, float], shape: tuple[int, ...], device) -> torch.Tensor:
    """Whole steps, both ends included: arena.lights.timetable's draw."""
    return torch.randint(steps(span[0]), steps(span[1]) + 1, shape, device=device)


def draw_reds(n: int, episode_steps: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Each env's reds as (onset, length), (n, R) int64, onsets counted from the episode's first step.
    R is enough for the last onset never to be reached."""
    settle = steps(SETTLE_S)
    r = math.ceil((episode_steps - settle - steps(FIRST_GREEN_S[0])) / (steps(GREEN_S[0]) + steps(RED_S[0]))) + 1
    length, green = _draw(RED_S, (n, r), device), _draw(GREEN_S, (n, r), device)
    gap = length + green
    onset = settle + _draw(FIRST_GREEN_S, (n, 1), device) + torch.cumsum(gap, dim=1) - gap
    return onset, length


def light_steps(onset: torch.Tensor, length: torch.Tensor, turn: int, k: torch.Tensor):
    """(red, mark, watch) at each env's step k, (n,) bool: arena.runner._LightSteps at one step."""
    k = k[:, None]
    red = (onset <= k) & (k < onset + length)
    mark = k == onset + turn - 1
    watch = (onset + turn <= k) & (k < onset + length)
    return red.any(1), mark.any(1), watch.any(1)


def hold_yaw_rate(y: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """The yaw rate that turns the duck toward the lane's centre line `LOOKAHEAD_M` ahead (arena.runner._steered).
    `y` is the offset from the lane (n,), `quat` the trunk's wxyz (n, 4)."""
    w, qx, qy, qz = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    error = torch.remainder(torch.atan2(-y, torch.full_like(y, LOOKAHEAD_M)) - yaw + math.pi, 2.0 * math.pi) - math.pi
    return torch.clamp(GAIN * error, -MAX_YAW_RATE, MAX_YAW_RATE)
