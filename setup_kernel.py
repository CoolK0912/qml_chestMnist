"""
Setup script to install all required packages and register the Jupyter kernel.
Run this with: python setup_kernel.py
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("QML Project Kernel Setup")
    print("=" * 60)

    # Get the venv python path
    venv_python = os.path.join('.venv', 'bin', 'python')
    venv_pip = os.path.join('.venv', 'bin', 'pip')

    if not os.path.exists(venv_python):
        print(f"Error: Virtual environment not found at {venv_python}")
        print("Please create a virtual environment first:")
        print("  python3 -m venv .venv")
        return False

    # Step 1: Upgrade pip
    if not run_command([venv_pip, 'install', '--upgrade', 'pip'],
                      "Step 1: Upgrading pip"):
        return False

    # Step 2: Install packages
    packages = [
        'ipykernel',
        'jupyter',
        'torch',
        'torchvision',
        'numpy',
        'pennylane',
        'medmnist'
    ]

    if not run_command([venv_pip, 'install'] + packages,
                      f"Step 2: Installing packages: {', '.join(packages)}"):
        return False

    # Step 3: Register kernel
    if not run_command(
        [venv_python, '-m', 'ipykernel', 'install', '--user',
         '--name=qml-project', '--display-name=Python (QML Project)'],
        "Step 3: Registering Jupyter kernel"
    ):
        return False

    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. In VS Code, open QML_Autoencoder.ipynb")
    print("2. Click the kernel selector in the top right")
    print("3. Select 'Python (QML Project)' from the list")
    print("4. Run your notebook!")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
