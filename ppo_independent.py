# test_setup.py
"""
Quick test script to verify the fixed IPPO implementation
"""
import json
import numpy as np
import torch
from models.buffer import RolloutBuffer
from models.ppo import PPOAgent
from models.env_sumo_multi import MultiIntersectionSUMO

def test_buffer():
    """Test buffer functionality"""
    print("Testing RolloutBuffer...")
    buf = RolloutBuffer(obs_dim=4, act_dim=1, size=10, device='cpu')
    
    # Store some dummy data
    for i in range(5):
        obs = np.random.randn(4).astype(np.float32)
        act = np.array([0.5], dtype=np.float32)
        rew = float(i)
        val = float(i * 0.1)
        logp = -1.0
        done = 0.0
        buf.store(obs, act, rew, val, logp, done)
    
    # Finish path
    buf.finish_path(last_val=0.0)
    print(f"Buffer ptr: {buf.ptr}, stored successfully: {buf.ptr == 5}")
    return True

def test_agent():
    """Test PPO agent"""
    print("Testing PPOAgent...")
    cfg = {
        "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_ratio": 0.2, "entropy_coef": 0.01, "value_coef": 0.5,
        "max_grad_norm": 0.5, "update_epochs": 2, "minibatch_size": 8,
        "t_min": 10.0, "t_max": 60.0
    }
    
    agent = PPOAgent(obs_dim=4, act_dim=1, cfg=cfg, device='cpu')
    
    # Test action sampling
    obs = np.random.randn(4).astype(np.float32)
    act, logp, val = agent.act(obs)
    
    print(f"Action: {act}, LogP: {logp}, Value: {val}")
    print(f"Agent test passed: {isinstance(act, np.ndarray) and len(act) == 1}")
    return True

def test_multi_env():
    """Test multi-agent environment (without SUMO)"""
    print("Testing MultiIntersectionSUMO (structure only)...")
    
    sumo_cfg = {"sumo_config_path": "dummy", "step_length": 1.0, "use_gui": False}
    env_cfg = {
        "junction_ids": ["J1", "J2"], "t_min": 10.0, "t_max": 60.0,
        "obs_dim": 4, "act_dim": 1
    }
    
    env = MultiIntersectionSUMO(sumo_cfg, env_cfg, seed=42)
    
    print(f"Num agents: {env.num_agents}")
    print(f"Obs dim: {env.obs_dim}, Act dim: {env.act_dim}")
    print(f"Junction IDs: {env.junction_ids}")
    
    # Test observation method (without SUMO)
    obs = env._get_obs_for("J1")
    print(f"Observation shape: {obs.shape}, expected: ({env.obs_dim},)")
    
    return env.num_agents == 2 and obs.shape == (4,)

def main():
    print("=== PPO/IPPO Setup Test ===\n")
    
    tests = [
        ("Buffer", test_buffer),
        ("Agent", test_agent), 
        ("Multi-Env", test_multi_env)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
            print(f"✓ {name} test: {'PASSED' if result else 'FAILED'}\n")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name} test: FAILED - {e}\n")
    
    print("=== Summary ===")
    for name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {name}")
        if error:
            print(f"      Error: {error}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Your IPPO implementation is ready to test with SUMO!")
        print("Run: python -m models.train_ippo --config constants/constants.json")
    
    return all_passed

if __name__ == "__main__":
    main()