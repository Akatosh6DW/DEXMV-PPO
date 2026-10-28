"""Isolated NumPy/MuJoCo worker; intentionally never imports JAX or CUDA.

Framed pickle messages travel only through pipes owned by our parent process.
No listener, remote input, or filesystem checkpoint unpickling is involved.
"""

from dataclasses import asdict
import pickle
import struct
import sys
import traceback

import mujoco
import numpy as np

from s12_soft import SoftConfig, SoftGrasp

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def read_message(stream):
    header = stream.read(8)
    if not header:
        raise EOFError
    if len(header)!=8:
        raise EOFError("Partial IPC header")
    length, = struct.unpack("!Q", header)
    data = stream.read(length)
    if len(data)!=length:
        raise EOFError("Partial IPC payload")
    return pickle.loads(data)


def write_message(stream, message):
    data = pickle.dumps(message, protocol=5)
    stream.write(struct.pack("!Q", len(data)))
    stream.write(data)
    stream.flush()


class SoftWorker:
    def __init__(self, cfg):
        self.sim = SoftGrasp(cfg)
        self.model = self.sim.model
        self.scratch = self.sim.data
        self.state_size = mujoco.mj_stateSize(self.model, STATE_SPEC)
        self.audit = np.zeros(14, np.float32)
        self.audit[12] = 1.

    def pack(self):
        d = self.sim.data
        words = np.empty(self.state_size, np.float64)
        mujoco.mj_getState(self.model, d, words, STATE_SPEC)
        return dict(integration_words=words.view(np.uint32).copy(), qpos=d.qpos.astype(np.float32),
            qvel=d.qvel.astype(np.float32), ctrl=d.ctrl.astype(np.float32),
            mocap_pos=d.mocap_pos.astype(np.float32), mocap_quat=d.mocap_quat.astype(np.float32),
            vertices=d.flexvert_xpos.astype(np.float32), geom_xpos=d.geom_xpos.astype(np.float32),
            actuator_force=d.actuator_force.astype(np.float32), audit=self.audit.copy())

    def single(self, mode, *arrays):
        sim = self.sim
        sim.data = self.scratch
        if mode == "s12_reset":
            offset, = arrays
            row = sim.reset(offset=offset)
            row.update(alive=True, valid=False)
        elif mode == "s12_step":
            words, desired, offset = arrays
            state = np.ascontiguousarray(words, dtype=np.uint32).view(np.float64)
            if state.shape != (self.state_size,):
                raise ValueError("Native integration state size mismatch")
            mujoco.mj_setState(self.model, sim.data, state, STATE_SPEC)
            baseline = np.array([0., 0., sim.cfg.hand_offset_z])
            sim.home = sim.original_pos+[0, -.20, 0]+offset+baseline
            sim.goal = sim.original_pos+offset+baseline
            sim.hold = sim.best = sim.max_strain = sim.max_depth = sim.peak_force = 0.
            sim.ever_invalid = False
            row = sim.step(desired)
        else:
            raise ValueError(mode)
        self.audit[:] = np.r_[row["loads"], row["penetration_m"], row["peak_force_n"],
            row["support_force_n"], row["forbidden_force_n"], row["max_principal_strain"],
            row["min_volume_ratio"], row["mean_volume_ratio"], row["alive"], row["valid"]]
        return self.pack()


def main():
    # MuJoCo warnings must not corrupt the binary protocol on stdout.
    mujoco.set_mju_user_warning(lambda message: print(message, file=sys.stderr, flush=True))
    cfg = SoftConfig(**read_message(sys.stdin.buffer))
    worker = SoftWorker(cfg)
    write_message(sys.stdout.buffer, dict(ready=True, nq=worker.model.nq, config=asdict(cfg)))
    while True:
        try:
            mode, batch = read_message(sys.stdin.buffer)
        except EOFError:
            return
        try:
            result = [worker.single(mode, *arrays) for arrays in batch]
            write_message(sys.stdout.buffer, dict(result=result))
        except Exception:
            write_message(sys.stdout.buffer, dict(error=traceback.format_exc()))


if __name__ == "__main__":
    main()
