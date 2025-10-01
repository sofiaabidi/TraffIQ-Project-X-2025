# SUMO Installation Guide for Windows

## Problem: SUMO is not properly installed or configured

Your error indicates that SUMO cannot be found. Here are the solutions:

## Option 1: Install SUMO (Recommended)

### Download and Install SUMO:
1. Go to: https://eclipse.dev/sumo/
2. Download the Windows installer (latest stable version)
3. Install SUMO to the default location: `C:\Program Files (x86)\Eclipse\Sumo`

### Set Environment Variables:
After installation, set the SUMO_HOME environment variable:

**Method 1 - Permanent (Recommended):**
1. Press `Windows + R`, type `sysdm.cpl`, press Enter
2. Click "Environment Variables"
3. Under "System Variables", click "New"
4. Variable name: `SUMO_HOME`
5. Variable value: `C:\Program Files (x86)\Eclipse\Sumo` (or wherever you installed SUMO)
6. Click OK
7. **Restart your PowerShell/Command Prompt**

**Method 2 - Temporary (for this session only):**
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH += ";$env:SUMO_HOME\bin"
```

## Option 2: Quick Fix for Testing

If you want to test without full installation, you can modify the test script to use a local SUMO installation or skip SUMO checks temporarily.

### Modify the test script to skip SUMO:
Run this to create a SUMO-free test:
```powershell
python -c "
import torch
import numpy as np
import matplotlib.pyplot as plt
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ NumPy version:', np.__version__)
print('✓ Matplotlib version:', matplotlib.__version__)
print('All Python dependencies are working!')
"
```

## Option 3: Use Conda/Pip Installation

You can also install SUMO via conda:
```bash
conda install -c conda-forge sumo
```

Or use the Windows Subsystem for Linux (WSL) if you have it:
```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

## Verify Installation

After installing SUMO, verify it works:
```powershell
sumo --version
```

You should see something like:
```
Eclipse SUMO sumo Version 1.19.0
Build features: Windows msvc14 x64 GDAL Proj GUI SWIG GL2PS Eigen3
Copyright (C) 2001-2023 German Aerospace Center (DLR) and others; https://sumo.dlr.de
License EPL-2.0: Eclipse Public License Version 2 <https://eclipse.org/legal/epl-v20.html>
```

## Test Your Setup Again

Once SUMO is installed:
1. **Restart your PowerShell** (important!)
2. Navigate to your project:
   ```powershell
   cd "c:\Ojas\X\TrafIQ Project\SUMO Testing\single intersection\dqn"
   ```
3. Run the test:
   ```powershell
   python test_dqn_setup.py
   ```

## Common Issues and Solutions

### "SUMO_HOME not set" error:
- Make sure you set the environment variable correctly
- Restart PowerShell after setting environment variables
- Check the path exists: `Test-Path $env:SUMO_HOME`

### "sumo command not found":
- Add SUMO's bin directory to PATH: `$env:PATH += ";$env:SUMO_HOME\bin"`
- Or use full path in scripts: `"$env:SUMO_HOME\bin\sumo.exe"`

### Permission errors:
- Run PowerShell as Administrator
- Check if antivirus is blocking SUMO

## Quick Test Commands

Run these one by one to diagnose:

```powershell
# Check SUMO_HOME
echo $env:SUMO_HOME

# Check if SUMO directory exists
Test-Path $env:SUMO_HOME

# Check if SUMO executable exists
Test-Path "$env:SUMO_HOME\bin\sumo.exe"

# Try to run SUMO
& "$env:SUMO_HOME\bin\sumo.exe" --version
```

## Next Steps

After fixing SUMO installation:
1. Run the test script again: `python test_dqn_setup.py`
2. If it passes, run the optimized DQN: `python ../optimized_dqn.py`

Let me know which solution works for you!