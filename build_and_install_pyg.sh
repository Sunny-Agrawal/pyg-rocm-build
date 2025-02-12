#!/bin/bash

# Get the directory where the script is located
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: ${script_dir}"

# Check if the virtual environment exists and activate it
if [ ! -d "/opt/venv/bin" ]; then
    echo "Virtual environment does not exist. Creating..."
    python3 -m venv /opt/venv
    source /opt/venv/bin/activate
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install --no-cache-dir -r /tmp/requirements.txt
else
    echo "Virtual environment exists. Activating..."
    source /opt/venv/bin/activate
    echo "Checking for dependency updates..."
    pip install --no-cache-dir -r /tmp/requirements.txt
fi

python_version=$(python --version 2>&1)
echo "Using Python: $python_version"

# Change to the script directory
cd "${script_dir}"

for dir in pytorch_cluster-1.6.3 pytorch_scatter-2.1.2 pytorch_sparse-0.6.18 pytorch_spline_conv-1.2.2
do
  (
    cd "${script_dir}/$dir"
    echo "Building $dir..."
    python ./setup.py bdist_wheel
    
    # Find and install the newly created wheel
    wheel_file=$(ls dist/*.whl | head -n 1)
    if [ -f "$wheel_file" ]; then
      echo "Installing $wheel_file..."
      pip install "$wheel_file" --force-reinstall
    else
      echo "Error: No wheel file found for $dir"
      exit 1
    fi
  ) &
done

wait

echo "Build and installation complete. All PyG modules have been built and installed."


### THE FOLLOWING IS FOR ZIPPING BUILT WHEELS FOR DISTRIBUTION

# if [ -d ${current_path}/dist ]; then
#   rm -rf ${current_path}/dist
# fi
# mkdir ${current_path}/dist

# package_name="torch-2.4-rocm-6.1"

# versions=("py310")

# for version in "${versions[@]}"; do
#     mkdir -p ${current_path}/dist/${package_name}-${version}-linux_x86_64
    
#     for module in "pytorch_cluster-1.6.3" "pytorch_scatter-2.1.2" "pytorch_sparse-0.6.18" "pytorch_spline_conv-1.2.2"; do
#         mv ${current_path}/${module}/dist/*-cp${version: -2}-cp${version: -2}-linux_x86_64.whl ${current_path}/dist/${package_name}-${version}-linux_x86_64/
#     done

#     zip ${current_path}/dist/${package_name}-${version}-linux_x86_64.zip ${current_path}/dist/${package_name}-${version}-linux_x86_64/*.whl
# done

# for module in "pytorch_cluster-1.6.3" "pytorch_scatter-2.1.2" "pytorch_sparse-0.6.18" "pytorch_spline_conv-1.2.2"; do
#     rm -rf ${current_path}/${module}/dist
#     rm -rf ${current_path}/${module}/build
# done


# wget "https://files.pythonhosted.org/packages/03/9f/157e913626c1acfb3b19ce000b1a6e4e4fb177c0bc0ea0c67ca5bd714b5a/torch_geometric-2.6.1-py3-none-any.whl" -O "${current_path}/dist/torch_geometric-2.6.1-py3-none-any.whl"

# for version in "${versions[@]}"; do
#     cp ${current_path}/dist/torch_geometric-2.6.1-py3-none-any.whl ${current_path}/dist/${package_name}-${version}-linux_x86_64/
#     zip ${current_path}/dist/${package_name}-${version}-linux_x86_64.zip ${current_path}/dist/${package_name}-${version}-linux_x86_64/*.whl
# done

# rm ${current_path}/dist/torch_geometric-2.6.1-py3-none-any.whl
