import time
import random
import traci
import numpy as np
from collections import defaultdict
import torch  # used only to test for tensor input in _action_to_seconds

class MultiAgentSUMO:

    def __init__(self, sumo_cfg, env_cfg, reward_weights=None, seed=42):
        self.sumo_cfg = sumo_cfg
        self.tls_ids = env_cfg["tls_ids"] 
        self.t_min = float(env_cfg.get("t_min", 10.0))
        self.t_max = float(env_cfg.get("t_max", 60.0))
        self.obs_dim = int(env_cfg.get("obs_dim", 64))
        self.act_dim = int(env_cfg.get("act_dim", 1))
        self.step_length = float(sumo_cfg.get("step_length", 1.0))

        self.yellow_time = 3.0
        # current_phase per tls (0 or 2 used in your earlier design)
        self.current_phase = {tls: 0 for tls in self.tls_ids}

        self.reward_weights = reward_weights or {}
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Expose list of agents
        self.agents = self.tls_ids
        self.controlled_lanes = {tls: [] for tls in self.tls_ids}
        self.episode_stats = defaultdict(list)

        self._start_traci(self.sumo_cfg)

    # -------------------- SUMO bootstrap --------------------
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

        # cache lanes per tls and freeze switching
        for tls in self.tls_ids:
            try:
                self.controlled_lanes[tls] = traci.trafficlight.getControlledLanes(tls)
                print(f"[DEBUG] {tls}: {len(self.controlled_lanes[tls])} controlled lanes")
            except Exception:
                self.controlled_lanes[tls] = []

            # freeze auto switching (set a very large phase duration)
            try:
                traci.trafficlight.setPhase(tls, 0)
                traci.trafficlight.setPhaseDuration(tls, 10000)
            except Exception:
                pass
            self.current_phase[tls] = 0

    # -------------------- RL API --------------------
    def reset(self):
        if traci.isLoaded():
            try:
                traci.close(False)
            except Exception:
                pass
            time.sleep(0.05)

        self._start_traci(self.sumo_cfg)
        self.current_phase = {tls: 0 for tls in self.tls_ids}
        self.episode_stats = defaultdict(list)

        obs = {tls: self._get_observation(tls) for tls in self.tls_ids}
        return obs

    def step(self, actions):
        """
        Synchronized multi-agent step: apply all agents' green times,
        then advance SUMO globally for the required number of internal steps.
        """
        if not traci.isLoaded() or traci.simulation.getMinExpectedNumber() == 0:
            obs = {tls: self._get_observation(tls) for tls in self.tls_ids}
            return obs, {tls: 0.0 for tls in self.tls_ids}, {tls: True for tls in self.tls_ids}, {}

        # 1) compute and apply green times for all tls
        green_times = {}
        for tls in self.tls_ids:
            action = actions.get(tls, 0)
            gtime = self._action_to_seconds(action)
            green_times[tls] = gtime
            try:
                traci.trafficlight.setPhase(tls, self.current_phase[tls])
                traci.trafficlight.setPhaseDuration(tls, gtime)
            except Exception:
                # continue even if a tls fails
                pass

        # 2) simulate for as many internal steps as needed (use the max green to cover all)
        steps_green = max(1, int(round(max(green_times.values()) / self.step_length)))
        rewards = {tls: 0.0 for tls in self.tls_ids}

        for _ in range(steps_green):
            if traci.simulation.getMinExpectedNumber() == 0:
                break
            traci.simulationStep()
            for tls in self.tls_ids:
                rewards[tls] += self._reward_step(tls)

        # 3) apply yellow and flip phases (we do it after green simulation)
        for tls in self.tls_ids:
            if traci.simulation.getMinExpectedNumber() == 0:
                # If sim finished, keep current_phase as is
                continue
            # yellow phase index depends on your tls definition; we keep same logic as before:
            yphase = 1 if self.current_phase[tls] == 0 else 3
            try:
                traci.trafficlight.setPhase(tls, yphase)
                traci.trafficlight.setPhaseDuration(tls, self.yellow_time)
            except Exception:
                pass

            # step through yellow duration
            for _ in range(int(self.yellow_time / self.step_length)):
                if traci.simulation.getMinExpectedNumber() == 0:
                    break
                traci.simulationStep()
                rewards[tls] += self._reward_step(tls)

            # flip phase (0 <-> 2)
            self.current_phase[tls] = 2 if self.current_phase[tls] == 0 else 0
            try:
                traci.trafficlight.setPhase(tls, self.current_phase[tls])
                traci.trafficlight.setPhaseDuration(tls, 10000)
            except Exception:
                pass

        # 4) observations, done, infos
        obs = {tls: self._get_observation(tls) for tls in self.tls_ids}
        done_flag = traci.simulation.getMinExpectedNumber() == 0
        dones = {tls: done_flag for tls in self.tls_ids}
        dones["__all__"] = done_flag
        infos = {tls: {"phase": int(self.current_phase[tls]), "green_time": float(green_times[tls])} for tls in self.tls_ids}

        return obs, rewards, dones, infos

    # -------------------- Helpers --------------------
    def _action_to_seconds(self, action):
        """
        Accept tensors, numpy arrays, python floats or 1-element iterables.
        Return a float number of seconds clipped to [t_min, t_max].
        """
        try:
            # torch tensor
            import torch
            if isinstance(action, torch.Tensor):
                if action.numel() == 1:
                    a = float(action.item())
                else:
                    a = float(action.reshape(-1)[0].item())
            elif hasattr(action, "numpy"):  # numpy-ish
                arr = np.array(action)
                if arr.size == 0:
                    a = 0.0
                else:
                    a = float(np.ravel(arr)[0])
            elif hasattr(action, "__len__") and not isinstance(action, (str, bytes)):
                # list-like or tuple-like
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

    def _get_observation(self, tls):
        vals = []
        for lane in self.controlled_lanes[tls]:
            try:
                vals.append(traci.lane.getLastStepVehicleNumber(lane))
            except Exception:
                vals.append(0)
        if len(vals) < self.obs_dim:
            vals += [0] * (self.obs_dim - len(vals))
        else:
            vals = vals[:self.obs_dim]
        return np.array(vals, dtype=np.float32)

    def _reward_step(self, tls):
        halted = 0
        for lane in self.controlled_lanes[tls]:
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
