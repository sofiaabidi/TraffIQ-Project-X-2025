# models/env_sumo_single.py
import time
import random
import traci
import traci.constants as tc
import numpy as np
from collections import defaultdict


class SingleIntersectionSUMO:

    def __init__(self, sumo_cfg, env_cfg, reward_weights=None, seed=42):
        self.sumo_cfg = sumo_cfg
        self.tls_id = env_cfg["tls_id"]
        self.t_min = float(env_cfg.get("t_min", 10.0))
        self.t_max = float(env_cfg.get("t_max", 60.0))
        self.obs_dim = int(env_cfg.get("obs_dim", 64))
        self.act_dim = int(env_cfg.get("act_dim", 1))
        self.step_length = float(sumo_cfg.get("step_length", 1.0))

        self.yellow_time = 3.0
        self.current_phase = 0   # 0 = NS green, 2 = EW green

        self.reward_weights = reward_weights or {}
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.controlled_lanes = []
        self.episode_stats = defaultdict(list)

        self._start_traci(self.sumo_cfg)

    def _start_traci(self, sumo_cfg):
        if traci.isLoaded():
            try:
                traci.close(False)
            except Exception:
                pass
            time.sleep(0.05)

        sumo_binary = "sumo-gui" if sumo_cfg.get("use_gui", False) else "sumo"
        cmd = [
            sumo_binary,
            "-c", sumo_cfg["sumo_config_path"],
            "--step-length", str(self.step_length),
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--time-to-teleport", "-1",
        ]
        traci.start(cmd)
        print("[DEBUG] SUMO started.")

        try:
            self.controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
            print(f"[DEBUG] Controlled lanes: {len(self.controlled_lanes)}")
        except Exception:
            self.controlled_lanes = []

        traci.trafficlight.setPhase(self.tls_id, 0)
        traci.trafficlight.setPhaseDuration(self.tls_id, 10000)
        self.current_phase = 0

    def reset(self):
        if traci.isLoaded():
            try:
                traci.close(False)
            except Exception:
                pass
            time.sleep(0.05)

        self._start_traci(self.sumo_cfg)
        self.current_phase = 0
        self.episode_stats = defaultdict(list)
        return self._get_observation()

    def step(self, action):
        if not traci.isLoaded() or traci.simulation.getMinExpectedNumber() == 0:
            return self._get_observation(), 0.0, True, {}

        green_time = self._action_to_seconds(action)

        traci.trafficlight.setPhase(self.tls_id, self.current_phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, green_time)
        reward_sum = 0.0
        steps_green = max(1, int(round(green_time / self.step_length)))

        for _ in range(steps_green):
            if traci.simulation.getMinExpectedNumber() == 0:
                break
            traci.simulationStep()
            reward_sum += self._reward_step()

        if traci.simulation.getMinExpectedNumber() > 0:
            yphase = 1 if self.current_phase == 0 else 3
            traci.trafficlight.setPhase(self.tls_id, yphase)
            traci.trafficlight.setPhaseDuration(self.tls_id, self.yellow_time)

            for _ in range(int(self.yellow_time / self.step_length)):
                if traci.simulation.getMinExpectedNumber() == 0:
                    break
                traci.simulationStep()
                reward_sum += self._reward_step()

            self.current_phase = 2 if self.current_phase == 0 else 0
            traci.trafficlight.setPhase(self.tls_id, self.current_phase)
            traci.trafficlight.setPhaseDuration(self.tls_id, 10000)

        obs = self._get_observation()
        done = traci.simulation.getMinExpectedNumber() == 0
        info = {"phase": int(self.current_phase), "green_time": float(green_time)}
        return obs, reward_sum, done, info

    def _action_to_seconds(self, action):
        try:
            if hasattr(action, "__len__"):
                a = float(action[0])
            else:
                a = float(action)
        except Exception:
            a = 0.0

        if self.act_dim == 1:
            a = np.clip(a, 0.0, 1.0)
            secs = self.t_min + a * (self.t_max - self.t_min)
        else:
            bins = np.linspace(self.t_min, self.t_max, self.act_dim)
            idx = int(np.clip(round(a), 0, self.act_dim - 1))
            secs = bins[idx]
        return float(np.clip(secs, self.t_min, self.t_max))

    def _get_observation(self):
        vals = []
        for lane in self.controlled_lanes:
            try:
                vals.append(traci.lane.getLastStepVehicleNumber(lane))
            except Exception:
                vals.append(0)
        if len(vals) < self.obs_dim:
            vals += [0] * (self.obs_dim - len(vals))
        else:
            vals = vals[:self.obs_dim]
        return np.array(vals, dtype=np.float32)

    def _reward_step(self):
        halted = 0
        for lane in self.controlled_lanes:
            try:
                halted += traci.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass
        return -float(halted)

    def close(self):
        try:
            if traci.isLoaded():
                traci.close(False)
                time.sleep(0.05)
        except Exception:
            pass
