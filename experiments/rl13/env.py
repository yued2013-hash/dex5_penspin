"""Reference-assisted, phase-conditioned residual control over the ENTIRE episode.

This simulation-only experiment does not consume the frozen Isaac O0/S0 banks.
The policy is privileged: it sees simulated pen pose and per-finger contacts.
"""
from pathlib import Path
import json
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from .runtime.model import build, ACTIVE
from .runtime.contact_metrics import axis

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())


class ResidualPenEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, training=True, rank=0, residual_limit=None):
        self.training = training
        self.residual_limit = CONFIG['residual_limit_rad'] if residual_limit is None else float(residual_limit)
        self.m = build()
        self.d = mujoco.MjData(self.m)
        ref = np.load(ROOT / "data/reference.npz", allow_pickle=False)
        self.states, self.commands = ref["states"], ref["ctrl"]
        self.normal = ref["normal"]
        self.n = len(self.commands)
        self.qids = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in ACTIVE])
        self.vids = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in ACTIVE])
        self.pq = int(self.m.jnt_qposadr[self.m.joint("pen_free").id])
        self.pv = int(self.m.jnt_dofadr[self.m.joint("pen_free").id])
        self.reference_q = self.states[:, 1:1 + self.m.nq]
        self.force_addr = [int(self.m.sensor(f"finger{i}_pen_contact").adr[0]) for i in range(1, 6)]
        self.palm_addr = int(self.m.sensor("palm_pen_contact").adr[0])
        self.action_space = spaces.Box(-1, 1, (16,), dtype=np.float32)
        self.observation_space = spaces.Box(-20, 20, (101,), dtype=np.float32)
        self.rng = np.random.default_rng(CONFIG["seed"] + rank)
        self.offset = np.zeros(16)
        self.last_action = np.zeros(16)
        self.last_command = self.commands[0].copy()
        self.k = self.start = self.steps = 0
        self.end = self.n
        self.strength = 1.0

    def forces(self):
        return np.array([np.linalg.norm(self.d.sensordata[a + 1:a + 4]) for a in self.force_addr])

    def pose_error(self, frame):
        target = self.reference_q[frame, self.pq:self.pq + 7]
        current = self.d.qpos[self.pq:self.pq + 7]
        phi = np.zeros(3)
        quat = target[3:].copy()
        if quat @ current[3:] < 0:
            quat = -quat
        mujoco.mju_subQuat(phi, quat, current[3:])
        return current[:3] - target[:3], phi

    def obs(self):
        p, phi = self.pose_error(self.k)
        phase = self.k / self.n
        command = self.commands[min(self.k, self.n - 1)]
        ranges = self.m.actuator_ctrlrange
        normalized_command = 2 * (command - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1
        value = np.r_[
            (self.d.qpos[self.qids] - self.reference_q[self.k, self.qids]) / .2,
            self.d.qvel[self.vids] * .1, p / .02, phi / .5,
            self.d.qvel[self.pv:self.pv + 3] / .1,
            self.d.qvel[self.pv + 3:self.pv + 6], normalized_command,
            self.offset / self.residual_limit, self.last_action,
            phase, np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase), 1 - phase,
            self.forces() / 5,
        ]
        return np.clip(value, -20, 20).astype(np.float32)

    def potential(self, frame):
        p, phi = self.pose_error(frame)
        return -float(min(20, 3 * np.sum((p / .015) ** 2) + np.sum((phi / .4) ** 2)))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        if "frame" in options:
            frame = int(options["frame"])
        elif self.training:
            gate = json.loads((ROOT / "results/reset_validation.json").read_text())
            assert gate["ready_for_bounded_training"], "Run validate_reference first."
            # Half of the windows start at the true initial state; half cover all stages.
            frame = 0 if self.rng.random() < .5 else int(self.rng.choice(gate["approved_frames"]))
        else:
            frame = 0
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_setState(self.m, self.d, self.states[frame], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.m, self.d)
        self.k = self.start = frame
        self.steps = 0
        self.end = self.n if options.get("full", not self.training) else min(self.n, frame + CONFIG["train_window_steps"])
        self.offset[:] = 0
        self.last_action[:] = 0
        self.last_command = self.commands[max(0, frame - 1)].copy()
        self.previous_phi = self.potential(frame)
        return self.obs(), {"source_id": CONFIG["reference_source_id"], "frame": frame}

    def step(self, action):
        action = np.clip(np.asarray(action), -1, 1)
        alpha = CONFIG["residual_filter_alpha"]
        self.offset = (1 - alpha) * self.offset + alpha * self.residual_limit * self.strength * action
        command = np.clip(self.commands[self.k] + self.offset, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
        self.d.ctrl[:] = command
        self.d.qfrc_applied[:] = 0
        self.d.xfrc_applied[:] = 0
        mujoco.mj_step(self.m, self.d, nstep=20)
        self.k += 1
        self.steps += 1
        p, phi = self.pose_error(self.k)
        f = self.forces()
        loaded = int(np.sum(f > .02))
        palm = float(np.linalg.norm(self.d.sensordata[self.palm_addr + 1:self.palm_addr + 4]))
        failed = bool(self.d.qpos[self.pq + 2] < .4 or np.linalg.norm(p) > .07 or palm > .02
                      or not np.isfinite(self.d.qpos).all() or any(w.number for w in self.d.warning))
        potential = self.potential(self.k)
        # Potential shaping + positive survival reward, with remaining-horizon failure cost.
        reward = (np.exp(-np.sum((p / .015) ** 2)) + .6 * np.exp(-np.sum((phi / .4) ** 2))
                  + .2 * (loaded >= 2) + CONFIG["gamma"] * potential - self.previous_phi
                  - .01 * np.mean((action - self.last_action) ** 2)
                  - .01 * np.mean(((command - self.last_command) / .04) ** 2))
        if failed:
            reward -= 2 * (self.end - self.k + 1)
        self.previous_phi = potential
        self.last_action = action.copy()
        self.last_command = command.copy()
        info = {"frame": self.k, "failure": failed, "reference_position_error_mm": float(np.linalg.norm(p) * 1000),
                "reference_orientation_error_deg": float(np.linalg.norm(phi) * 180 / np.pi), "loaded_fingers": loaded}
        return self.obs(), float(reward), failed, self.k >= self.end, info
