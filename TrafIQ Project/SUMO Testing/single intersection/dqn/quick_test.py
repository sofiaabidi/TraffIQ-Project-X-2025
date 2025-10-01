# Quick Test Script - SUMO-Optional Version
# This version will test dependencies and allow you to run DQN training even if SUMO setup is incomplete

import os
import sys

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

def test_sumo_optional():
    """Test SUMO setup but don't fail if it's not working"""
    print("Checking SUMO setup (optional)...")
    
    # Check SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        print("⚠ SUMO_HOME environment variable not set")
        return False
    
    sumo_home = os.environ['SUMO_HOME']
    print(f"SUMO_HOME set to: {sumo_home}")
    
    # Check if directory exists
    if not os.path.exists(sumo_home):
        print(f"⚠ SUMO_HOME directory does not exist: {sumo_home}")
        return False
    
    # Check if SUMO executable exists
    sumo_exe = os.path.join(sumo_home, 'bin', 'sumo.exe')
    if not os.path.exists(sumo_exe):
        print(f"⚠ SUMO executable not found: {sumo_exe}")
        return False
    
    # Try to run SUMO
    try:
        # Add SUMO tools to path
        tools = os.path.join(sumo_home, 'tools')
        sys.path.append(tools)
        
        import traci
        print("✓ SUMO TraCI module imported successfully")
        return True
    except Exception as e:
        print(f"⚠ SUMO TraCI import failed: {e}")
        return False

def check_project_files():
    """Check if required project files exist"""
    print("Checking project files...")
    
    required_files = [
        'dqn.sumocfg',
        'dqn.net.xml', 
        'dqn.rou.xml',
        'dqn.add.xml'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            missing_files.append(file)
            print(f"✗ {file} missing")
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    return True

def create_minimal_test():
    """Create a minimal test script that doesn't require SUMO"""
    test_code = '''
# Minimal DQN Test - No SUMO Required
import torch
import torch.nn as nn
import numpy as np

# Test the DQN network architecture
class TestDQN(nn.Module):
    def __init__(self, input_dim=10, output_dim=32):
        super(TestDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Test the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create network
net = TestDQN().to(device)
print("✓ DQN network created successfully")

# Test forward pass
test_input = torch.randn(1, 10).to(device)
output = net(test_input)
print(f"✓ Forward pass successful, output shape: {output.shape}")

# Test optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
print("✓ Optimizer created successfully")

print("\\n🎉 DQN components are working correctly!")
print("You can proceed with training once SUMO is properly installed.")
'''
    
    with open('minimal_dqn_test.py', 'w') as f:
        f.write(test_code)
    
    print("✓ Created minimal_dqn_test.py for testing DQN components")

if __name__ == "__main__":
    print("=== Enhanced DQN Setup Check ===\\n")
    
    # Critical dependencies (must pass)
    deps_ok = test_dependencies()
    pytorch_ok = test_pytorch()
    
    if not (deps_ok and pytorch_ok):
        print("\\n❌ Critical dependencies missing. Please install them first.")
        sys.exit(1)
    
    print()
    
    # Optional checks
    files_ok = check_project_files()
    sumo_ok = test_sumo_optional()
    
    print()
    
    # Create minimal test
    create_minimal_test()
    
    print("\\n=== Summary ===")
    print(f"✓ Python dependencies: {'OK' if deps_ok else 'FAILED'}")
    print(f"✓ PyTorch: {'OK' if pytorch_ok else 'FAILED'}")
    print(f"{'✓' if files_ok else '⚠'} Project files: {'OK' if files_ok else 'MISSING'}")
    print(f"{'✓' if sumo_ok else '⚠'} SUMO setup: {'OK' if sumo_ok else 'NEEDS ATTENTION'}")
    
    if deps_ok and pytorch_ok:
        print("\\n🎉 Core components are ready!")
        if not sumo_ok:
            print("\\n⚠  SUMO needs to be properly installed for full functionality.")
            print("   See SUMO_INSTALLATION_GUIDE.md for installation instructions.")
            print("   You can run 'python minimal_dqn_test.py' to test DQN components.")
        else:
            print("\\n✅ Everything is ready! You can run the optimized DQN training.")
    else:
        print("\\n❌ Please install missing dependencies before proceeding.")