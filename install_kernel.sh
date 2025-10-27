#!/bin/bash

# Install packages in the virtual environment
echo "Installing packages in .venv..."

# Activate virtual environment
source .venv/bin/activate

# Install required packages
pip install ipykernel jupyter torch torchvision numpy pennylane medmnist

# Register the kernel with Jupyter
python -m ipykernel install --user --name=qml-project --display-name="Python (QML Project)"

echo "Done! Kernel 'Python (QML Project)' is now available."
echo "In VS Code, select this kernel from the kernel picker in the top right."

deactivate
