# train.py
import json, argparse, numpy as np, torch
from models.buffer import RolloutBuffer
from models.ppo import PPOAgent
from models.env_sumo_single import SingleIntersectionSUMO

def load_config(path):
    with open(path) as f:
        return json.load(f)

def train(cfg_path):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SingleIntersectionSUMO(cfg["sumo"], cfg["env"], cfg["reward_weights"], seed=cfg["train"]["seed"])
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    agent_cfg = cfg["agent"]
    agent_cfg["t_min"] = cfg["env"]["t_min"]
    agent_cfg["t_max"] = cfg["env"]["t_max"]

    agent = PPOAgent(obs_dim, act_dim, agent_cfg, device=device)
    steps_per_update = int(cfg["agent"]["rollout_seconds"] / cfg["sumo"]["step_length"])
    buf = RolloutBuffer(obs_dim, act_dim, size=steps_per_update, gamma=cfg["agent"]["gamma"], lam=cfg["agent"]["gae_lambda"], device=device)

    obs = env.reset()
    total_steps = 0
    while total_steps < cfg["train"]["total_timesteps"]:
        for _ in range(steps_per_update):
            act, logp, v = agent.act(obs)
            a_val = float(act[0]) if hasattr(act, "__len__") else float(act)
            next_obs, rew, done, info = env.step(a_val)
            buf.store(obs, np.array([a_val], dtype=np.float32), rew, float(v), float(logp), float(done))
            obs = next_obs
            total_steps += 1
            if done:
                buf.finish_path(last_val=0.0)
                obs = env.reset()
                break
        else:
            with torch.no_grad():
                device_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                last_val = agent.value(device_obs).item()
            buf.finish_path(last_val)

        data = buf.get()
        agent.update(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="constants/constants.json")
    args = parser.parse_args()
    train(args.config)

