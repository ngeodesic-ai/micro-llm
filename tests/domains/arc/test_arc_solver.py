import numpy as np
from micro_lm.domains.arc.solver import solve_arc_io

class DummyAudit:
    @staticmethod
    def out(T=20):
        # 1 rotate op suggested at t=5
        env = np.zeros((T,4), dtype=np.float32)
        env[2:8,1] = 1.0  # prefer k=1
        keep = [0,1,2,3]; order = [1,0,2,3]
        windows = {i: (0, T-1) for i in range(4)}
        labels = [f"rotate_{i}" for i in range(4)]
        return {"keep": keep, "order": order, "windows": windows, "env": env, "labels": labels}

def test_solver_lmax3_replay_hit():
    img = np.arange(16, dtype=np.int32).reshape(4,4)
    img_rot = np.rot90(img, 1)
    io_pair = {"input": img, "output": img_rot}
    out = solve_arc_io(io_pair, DummyAudit.out(), {"arc": {"L_MAX": 3}})
    assert out["verified"] is True


def test_solver_abstains_when_no_env():
    img = np.zeros((4,4), dtype=np.int32)
    io_pair = {"input": img, "output": img}
    out = solve_arc_io(io_pair, {"keep":[],"order":[],"windows":{}}, {"arc": {"L_MAX": 1}})
    assert out["verified"] is False and out["reason"] == "no_env"