#!/bin/bash

# Activate virtual environment
source /opt/venv/bin/activate

# List of PyG modules to check
modules=(torch_scatter torch_sparse torch_cluster torch_spline_conv)

echo "🔍 Checking PyG module installations..."

for module in "${modules[@]}"; do
    if python -c "import $module; print('$module:', $module.__version__)" 2>/dev/null; then
        echo "$module is installed correctly."
    else
        echo "⚠️ WARNING: $module is missing or not installed correctly."
    fi
done

echo "✅ PyG module verification complete."
