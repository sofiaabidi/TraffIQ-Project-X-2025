# Quick Test Script for Optimized DQN
# Run this to test if everything works before the full training

import os
import sys
import traci

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    print("Please set SUMO_HOME environment variable")
    sys.exit(1)

def test_sumo_connection():
    """Test if SUMO can start properly"""
    sumo_config = [
        "sumo",
        "-c", "dqn.sumocfg", 
        "--step-length", "1.0",
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--waiting-time-memory", "10000",
        "--time-to-teleport", "-1"
    ]
    
    try:
        print("Testing SUMO connection...")
        traci.start(sumo_config)
        
        # Test basic functionality
        print("✓ SUMO started successfully")
        
        # Check if traffic light exists
        tls_ids = traci.trafficlight.getIDList()
        print(f"✓ Traffic lights found: {tls_ids}")
        
        # Run a few simulation steps
        for i in range(5):
            traci.simulationStep()
        print("✓ Simulation steps working")
        
        # Check lanes
        lanes = traci.lane.getIDList()
        print(f"✓ Total lanes found: {len(lanes)}")
        
        # Check if expected lanes exist
        expected_lanes = ["Node1_2_EB_0", "Node2_7_SB_0", "Node2_3_WB_0", "Node2_5_NB_0"]
        missing_lanes = []
        for lane in expected_lanes:
            if lane not in lanes:
                missing_lanes.append(lane)
        
        if missing_lanes:
            print(f"⚠ Warning: Missing expected lanes: {missing_lanes}")
        else:
            print("✓ All expected lanes found")
        
        traci.close()
        print("✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        if traci.isLoaded():
            traci.close()
        return False

def test_pytorch():
    """Test if PyTorch is available"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
        return True
    except ImportError:
        print("✗ PyTorch not found. Please install: pip install torch")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("Checking dependencies...")
    
    required_packages = ['numpy', 'matplotlib']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} found")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} missing")
    
    if missing:
        print(f"Please install missing packages: pip install {' '.join(missing)}")
        return False
    return True

if __name__ == "__main__":
    print("=== Optimized DQN Pre-flight Check ===\n")
    
    all_good = True
    
    # Test dependencies
    if not test_dependencies():
        all_good = False
    print()
    
    # Test PyTorch
    if not test_pytorch():
        all_good = False
    print()
    
    # Test SUMO
    if not test_sumo_connection():
        all_good = False
    print()
    
    if all_good:
        print("🎉 All tests passed! The optimized DQN is ready to run.")
        print("\nTo start training, run:")
        print("python optimized_dqn.py")
    else:
        print("❌ Some tests failed. Please fix the issues above before running.")