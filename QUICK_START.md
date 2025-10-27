# Quick Start Guide - Kernel Fix

## Problem
Your Jupyter kernel is failing to start in VS Code.

## Solution (3 minutes)

### Step 1: Run the Setup Script

Open Terminal and run:

```bash
cd "/Users/nadavyaniv/Projects/QML Project"
python3 setup_kernel.py
```

Wait for it to complete. You should see:
```
✅ Setup complete!
```

### Step 2: Restart VS Code

**Important**: Completely quit and restart VS Code (not just reload window)

### Step 3: Select the Kernel

1. Open `QML_Autoencoder.ipynb` in VS Code
2. Look at the **top right corner** of the notebook
3. Click where it says "Select Kernel" or shows the current kernel
4. From the dropdown, select: **"Python (QML Project)"**

### Step 4: Run Your Notebook

Click **"Run All"** or press `Shift + Enter` on each cell.

---

## What the Setup Script Does

1. ✅ Installs `ipykernel` (needed for VS Code)
2. ✅ Installs `torch`, `pennylane`, `medmnist` (needed for your code)
3. ✅ Registers a Jupyter kernel called "Python (QML Project)"
4. ✅ Makes it available to VS Code

---

## Still Not Working?

### Option A: Manual Installation

```bash
cd "/Users/nadavyaniv/Projects/QML Project"

# Activate virtual environment
source .venv/bin/activate

# Install packages
pip install ipykernel jupyter torch torchvision numpy pennylane medmnist

# Register kernel
python -m ipykernel install --user --name=qml-project --display-name="Python (QML Project)"

# Deactivate
deactivate
```

Then restart VS Code and select "Python (QML Project)" kernel.

### Option B: Select Python Interpreter Directly

1. In VS Code, press `Cmd + Shift + P`
2. Type: "Python: Select Interpreter"
3. Choose: `/Users/nadavyaniv/Projects/QML Project/.venv/bin/python`
4. Restart the kernel

---

## Verification

Once it's working, you should see:

```python
# Cell 1 output:
All imports successful!
PyTorch version: 2.9.0
PennyLane version: 0.43.0
```

If you see this, everything is working! 🎉

---

## Need More Help?

See the full [README.md](README.md) for detailed documentation.
