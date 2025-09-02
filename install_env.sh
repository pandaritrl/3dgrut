# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# Exit on error
set -e

UV_ENV=${1:-"3dgrut"}
CUDA_VERSION=${2:-"12.8.1"}

# parse an optional second arg WITH_GCC11 to also manually use gcc-11 within the environment
WITH_GCC11=false
if [ $# -ge 2 ]; then
    if [ "$2" = "WITH_GCC11" ]; then
        WITH_GCC11=true
    fi
fi

CUDA_VERSION=${CUDA_VERSION:-"11.8.0"}

# Verify user arguments
echo "Arguments:"
echo "  UV_ENV: $UV_ENV"
echo "  WITH_GCC11: $WITH_GCC11"
echo "  CUDA_VERSION: $CUDA_VERSION"
echo ""

# Make sure TORCH_CUDA_ARCH_LIST matches the pytorch wheel setting.
# Reference: https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_cuda.sh#L54
#
# (cuda11) $ python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
# 11.8 ['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_37', 'sm_90', 'compute_37']
#
# (cuda12) $ python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
# 12.8 ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']
#
# Check if CUDA_VERSION is supported
if [ "$CUDA_VERSION" = "11.8.0" ]; then
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0";
elif [ "$CUDA_VERSION" = "12.8.1" ]; then
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0";
else
    echo "Unsupported CUDA version: $CUDA_VERSION, available options are 11.8.0 and 12.8.1"
    exit 1
fi
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# Test if we have GCC<=11, and early-out if not
if [ ! "$WITH_GCC11" = true ]; then
    # Make sure gcc is at most 11 for nvcc compatibility
    gcc_version=$(gcc -dumpversion)
    if [ "$gcc_version" -gt 11 ]; then
        echo "Default gcc version $gcc_version is higher than 11. CUDA requires GCC 11 or lower."
        echo "The script will automatically install gcc-11 and g++-11 for CUDA compatibility."
        echo "Alternatively, you can rerun with: ./install_env.sh 3dgrut WITH_GCC11"
    fi
fi

# If we're going to set gcc11, make sure it is available
if [ "$WITH_GCC11" = true ]; then
    # Ensure gcc-11 is on path
    if ! command -v gcc-11 2>&1 >/dev/null
    then
        echo "gcc-11 could not be found. Perhaps you need to run 'sudo apt-get install gcc-11 g++-11'?"
        exit 1
    fi
    if ! command -v g++-11 2>&1 >/dev/null
    then
        echo "g++-11 could not be found. Perhaps you need to run 'sudo apt-get install gcc-11 g++-11'?"
        exit 1
    fi

    GCC_11_PATH=$(which gcc-11)
    GXX_11_PATH=$(which g++-11)
fi

# Create and activate uv environment
UV_ENV_PATH="./.venv"

if [ ! -d "${UV_ENV_PATH}" ]; then
  echo "UV environment not found, creating it"
  uv venv --python 3.11
else
  echo "NOTE: UV environment already exists at ${UV_ENV_PATH}, skipping environment creation"
fi

# Activate the uv environment
source ${UV_ENV_PATH}/bin/activate

# Set CC and CXX variables to gcc11 in the environment
if [ "$WITH_GCC11" = true ]; then
    echo "Setting CC=$GCC_11_PATH and CXX=$GXX_11_PATH in environment"
    export CC=$GCC_11_PATH
    export CXX=$GXX_11_PATH

    # Make sure it worked
    gcc_version=$($CC -dumpversion | cut -d '.' -f 1)
    echo "gcc_version=$gcc_version"
    if [ "$gcc_version" -gt 11 ]; then
        echo "gcc version $gcc_version is still higher than 11, setting gcc-11 failed"
        exit 1
    fi
fi

# Set TORCH_CUDA_ARCH_LIST environment variable
export TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

# Install CUDA and PyTorch dependencies
# CUDA 11.8 supports until compute capability 9.0
if [ "$CUDA_VERSION" = "11.8.0" ]; then
    echo "Installing CUDA 11.8.0 ..."
    # Install system dependencies via apt (assuming Ubuntu/Debian)
    sudo apt-get update
    sudo apt-get install -y cmake ninja-build
    
    # Check if CUDA runtime is available from NVIDIA driver
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA driver detected. Checking CUDA runtime availability..."
        nvidia-smi | grep -q "CUDA Version" && echo "CUDA runtime available from driver" || echo "CUDA runtime not detected"
    fi
    
    # Install CUDA toolkit for compilation (already installed via apt)
    echo "Setting up CUDA environment..."
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    # Install PyTorch with CUDA 11.8 support using uv's torch-backend
    uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --torch-backend cu118
    uv pip install "numpy<2.0"
    uv pip install --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html kaolin==0.17.0

# CUDA 12.8 supports compute capability 10.0 and 12.0
elif [ "$CUDA_VERSION" = "12.8.1" ]; then

    # get the GCC version so we can use it in the install command below
    # the _PATHs might already be set
    if [ ! "$WITH_GCC11" = true ]; then
        GCC_11_PATH=$(which gcc)
        GXX_11_PATH=$(which g++)
    fi

    gcc_version=$($GCC_11_PATH -dumpversion | cut -d '.' -f 1)

    echo "Installing CUDA 12.8.1 ..."
    # Install system dependencies via apt (assuming Ubuntu/Debian)
    sudo apt-get update
    sudo apt-get install -y cmake ninja-build gcc-${gcc_version} g++-${gcc_version}
    
    # Set up CUDA environment (already installed via apt)
    echo "Setting up CUDA environment..."
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    # Install PyTorch with CUDA 12.8 support using uv's torch-backend
    uv pip install torch torchvision torchaudio --torch-backend cu128
    uv pip install --force-reinstall "numpy<2"

    # TODO move to using wheel once kaolin is available
    rm -fr thirdparty/kaolin
    git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git thirdparty/kaolin
    pushd thirdparty/kaolin
    git checkout c2da967b9e0d8e3ebdbd65d3e8464d7e39005203  # ping to a fixed commit for reproducibility
    sed -i 's!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type()!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type()!g' kaolin/csrc/render/spc/raytrace_cuda.cu
    uv pip install --upgrade pip
    uv pip install --no-cache-dir ninja imageio imageio-ffmpeg
    uv pip install --no-cache-dir        \
        -r tools/viz_requirements.txt \
        -r tools/requirements.txt     \
        -r tools/build_requirements.txt
    IGNORE_TORCH_VER=1 python setup.py install
    popd
    rm -fr thirdparty/kaolin

# Unsupported CUDA version
else
    echo "Unsupported CUDA version: $CUDA_VERSION, available options are 11.8.0 and 12.8.1"
    exit 1
fi

# Install OpenGL headers for the playground
sudo apt-get install -y libgl1-mesa-dev libglu1-mesa-dev

# Initialize git submodules and install Python requirements
git submodule update --init --recursive
uv pip install -r requirements.txt
uv pip install -e .

echo "Setup completed successfully!"
