"""Sock Heist trains under the event's rules: every copy in lights.py is pinned against the Arena."""
from types import SimpleNamespace

import numpy as np, pytest, torch

from arena import runner as arena_runner
from arena.constants import CTRL_DT as ARENA_CTRL_DT
from arena.lights import timetable
from arena.manifest import load_event
from microduck_challenges.sock_heist_6m import lights

EVENT = load_event("sock-heist-6m")
SETTLE = lights.steps(lights.SETTLE_S)


def _quat(yaw, roll=0.0, pitch=0.0):
    """wxyz, z-y-x order."""
    cy, sy, cp, sp, cr, sr = (f(a / 2) for a in (yaw, pitch, roll) for f in (np.cos, np.sin))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dtype=np.float32)


def test_the_rules_are_the_events():
    l, s = EVENT.lights, EVENT.steering
    assert lights.CTRL_DT == ARENA_CTRL_DT and lights.SETTLE_S == EVENT.start.settle_s
    assert (lights.FIRST_GREEN_S, lights.GREEN_S, lights.RED_S) == (l.first_green_s, l.green_s, l.red_s)
    assert (lights.TURN_S, lights.TOLERANCE_M) == (l.turn_s, l.tolerance_m)
    assert (lights.LOOKAHEAD_M, lights.GAIN, lights.MAX_YAW_RATE) == (s.lookahead_m, s.gain, s.max_yaw_rate)


def test_every_phase_is_drawn_from_its_range_ends_included():
    torch.manual_seed(0)
    onset, length = lights.draw_reds(4096, 1000, "cpu")
    every = lambda r: set(range(lights.steps(r[0]), lights.steps(r[1]) + 1))
    assert set((onset[:, 0] - SETTLE).tolist()) == every(lights.FIRST_GREEN_S)     # 30..50
    assert set(length.flatten().tolist()) == every(lights.RED_S)                   # 70..120
    greens = onset[:, 1:] - onset[:, :-1] - length[:, :-1]
    assert set(greens.flatten().tolist()) == every(lights.GREEN_S)                 # 80..150
    assert (onset[:, -1] >= 1000).all()                                            # the table outlasts the episode


@pytest.mark.parametrize("turn", [lights.steps(lights.TURN_S), 0])
def test_the_light_steps_are_the_arenas(turn):
    steps = 3000
    for seed in EVENT.seeds:
        reds = timetable(EVENT.lights, EVENT.id, seed, steps)
        onset = torch.tensor([[r.onset + SETTLE for r in reds]]).expand(steps, -1)
        length = torch.tensor([[r.length for r in reds]]).expand(steps, -1)
        red, mark, watch = lights.light_steps(onset, length, turn, torch.arange(SETTLE, SETTLE + steps))
        expected = arena_runner._LightSteps.of(reds, turn, steps)
        assert (red.tolist(), mark.tolist(), watch.tolist()) == (
            expected.red.tolist(), expected.mark.tolist(), expected.watch.tolist())


def test_the_hold_is_the_arenas():
    command = np.asarray(EVENT.command, dtype=np.float64)
    for y in (-0.35, -0.05, 0.0, 0.12, 0.35):
        for yaw in (-3.1, -1.0, -0.1, 0.0, 0.2, 1.5, 3.1):
            q = _quat(yaw, roll=0.1, pitch=-0.08)                  # a tilted trunk: the yaw formula, not just qz
            data = SimpleNamespace(qpos=np.concatenate([[0.0, y, 0.12], q]))
            want = arena_runner._steered(EVENT.steering, data, SimpleNamespace(free=0), command)[2]
            got = lights.hold_yaw_rate(torch.tensor([y]), torch.tensor([q], dtype=torch.float32)).item()
            assert got == pytest.approx(want, abs=1e-5)
    rate = lambda y, yaw: lights.hold_yaw_rate(torch.tensor([y]), torch.tensor([_quat(yaw)])).item()
    assert rate(0.2, 0.0) < 0 < rate(-0.2, 0.0)                    # left of the line it turns right
    assert (rate(0.0, 3.0), rate(0.0, -3.0)) == (-1.0, 1.0)        # facing back: the short way round, capped
