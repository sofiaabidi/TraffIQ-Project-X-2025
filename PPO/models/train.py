import json, argparse, numpy as np, torch
from models.buffer import RolloutBuffer
from models.ppo import PPOAgent
from models.env_sumo_single import MultiAgentSUMO

def load_config(path):
    with open(path) as f:
        return json.load(f)

def to_scalar_action(action):
    # convert torch tensor / numpy / list to python float
    try:
        import torch
        if isinstance(action, torch.Tensor):
            return float(action.detach().cpu().item()) if action.numel() == 1 else float(action.reshape(-1)[0].detach().cpu().item())
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(action, np.ndarray):
            return float(np.ravel(action)[0])
    except Exception:
        pass
    # list/tuple
    try:
        if hasattr(action, "__len__") and not isinstance(action, (str, bytes)):
            return float(action[0])
    except Exception:
        pass
    return float(action)

def train(cfg_path):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiAgentSUMO(cfg["sumo"], cfg["env"], cfg.get("reward_weights", None), seed=cfg["train"]["seed"])
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    agent_cfg = cfg["agent"]
    agent_cfg["t_min"] = cfg["env"]["t_min"]
    agent_cfg["t_max"] = cfg["env"]["t_max"]

    ppo_agents = {agent_id: PPOAgent(obs_dim, act_dim, agent_cfg, device=device) for agent_id in env.agents}
    steps_per_update = int(cfg["agent"]["rollout_seconds"] / cfg["sumo"]["step_length"])
    buffers = {
        agent_id: RolloutBuffer(
            obs_dim,
            act_dim,
            size=steps_per_update,
            gamma=cfg["agent"]["gamma"],
            lam=cfg["agent"]["gae_lambda"],
            device=device,
        )
        for agent_id in env.agents
    }

    obs = env.reset()
    total_steps = 0
    while total_steps < cfg["train"]["total_timesteps"]:
        for _ in range(steps_per_update):
            actions, logps, values = {}, {}, {}
            for agent_id in env.agents:
                act, logp, v = ppo_agents[agent_id].act(obs[agent_id])
                # convert action to python scalar or numpy array as needed by env
                actions[agent_id] = to_scalar_action(act)
                logps[agent_id] = logp
                values[agent_id] = v

            next_obs, rews, dones, infos = env.step(actions)

            for agent_id in env.agents:
                buffers[agent_id].store(
                    obs[agent_id], actions[agent_id], rews[agent_id], values[agent_id], logps[agent_id]
                )
            obs = next_obs
            total_steps += 1

            # prefer explicit __all__ check
            if dones.get("__all__", False):
                for agent_id in env.agents:
                    buffers[agent_id].finish_path(last_val=0.0)
                obs = env.reset()
                break

        else:
            # bootstrap values for each agent
            for agent_id in env.agents:
                with torch.no_grad():
                    device_obs = torch.as_tensor(obs[agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                    last_val = ppo_agents[agent_id].value(device_obs).item()
                buffers[agent_id].finish_path(last_val)

        # apply updates
        for agent_id in env.agents:
            data = buffers[agent_id].get()
            ppo_agents[agent_id].update(data)
            buffers[agent_id].reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="constants/constants.json")
    args = parser.parse_args()
    train(args.config)
