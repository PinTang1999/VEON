#!/usr/bin/env bash
# Usage from any directory: bash /path/to/VEON/tools/install_env.sh [env_name]
# Requires Conda, NVIDIA driver, and CUDA Toolkit 10.2, 11.3 or 11.7 with nvcc.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' 'Usage: bash tools/install_env.sh [env_name=veon]' \
        'Target: Linux x86_64, NVIDIA GPU (compute capability <= 8.6).' \
        'Prerequisites: Conda, CUDA Toolkit 10.2, 11.3 or 11.7 (nvcc), GCC/G++.' \
        'GCC/G++ ranges: 5-8 for CUDA 10.2; 5-10 for 11.3; 5-11 for 11.7.' \
        'CUDA 10.2 targets compute capabilities up to 7.5 (including V100 7.0).' \
        'Set CUDA_HOME to select the toolkit. Its nvcc selects the package profile.' \
        'Example: CUDA_HOME=/usr/local/cuda-11.7 bash tools/install_env.sh veon-cu117' \
        'V100 example: CUDA_HOME=/usr/local/cuda-10.2 bash tools/install_env.sh veon-cu102' \
        'Installs software only; datasets and model weights are prepared separately.'
    exit 0
fi

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "Installation stopped at line %s. Fix the reported error and rerun.\n" "$LINENO" >&2' ERR

[[ $# -le 1 ]] || die 'Expected at most one environment name.'
ENV_NAME="${1:-veon}"
[[ "$ENV_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'Use letters, numbers, underscores or hyphens for the environment name.'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || \
    die 'This installer targets Linux x86_64 with NVIDIA CUDA; run it on the training server.'
command -v conda >/dev/null || die 'Conda is missing. Install Miniconda/Miniforge first.'
command -v nvidia-smi >/dev/null || die 'nvidia-smi is missing. Install the NVIDIA driver first.'
nvidia-smi -L || die 'No accessible NVIDIA GPU. Run on a GPU node.'
if [[ -n "${CUDA_HOME:-}" ]]; then
    CUDA_ROOT="$CUDA_HOME"
elif command -v nvcc >/dev/null; then
    CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
elif [[ -x /usr/local/cuda-11.7/bin/nvcc ]]; then
    CUDA_ROOT=/usr/local/cuda-11.7
elif [[ -x /usr/local/cuda-11.3/bin/nvcc ]]; then
    CUDA_ROOT=/usr/local/cuda-11.3
elif [[ -x /usr/local/cuda-10.2/bin/nvcc ]]; then
    CUDA_ROOT=/usr/local/cuda-10.2
else
    die 'CUDA Toolkit 10.2/11.3/11.7 (nvcc) is missing. nvidia-smi alone is not sufficient.'
fi
[[ -x "$CUDA_ROOT/bin/nvcc" ]] || die "nvcc was not found in $CUDA_ROOT/bin."
NVCC_INFO="$("$CUDA_ROOT/bin/nvcc" --version)"
case "$NVCC_INFO" in
    *'release 10.2,'*)
        CUDA_VERSION=10.2
        TORCH_VERSION=1.10.1
        VISION_VERSION=0.11.2
        MMCV_VERSION=1.6.0
        MMCV_INDEX=https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
        DETECTRON_INDEX=https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
        # Python dependency versions match the 11.3 profile; CUDA wheels differ.
        REQUIREMENTS=requirements/veon-cu113.txt
        CUDA_PACKAGES=(cudatoolkit=10.2)
        CHANNELS=(-c pytorch -c conda-forge)
        GCC_MAX=8
        ;;
    *'release 11.3,'*)
        CUDA_VERSION=11.3
        TORCH_VERSION=1.10.1
        VISION_VERSION=0.11.2
        MMCV_VERSION=1.6.0
        MMCV_INDEX=https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
        DETECTRON_INDEX=https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
        REQUIREMENTS=requirements/veon-cu113.txt
        CUDA_PACKAGES=(cudatoolkit=11.3)
        CHANNELS=(-c pytorch -c conda-forge)
        GCC_MAX=10
        ;;
    *'release 11.7,'*)
        CUDA_VERSION=11.7
        TORCH_VERSION=1.13.1
        VISION_VERSION=0.14.1
        MMCV_VERSION=1.7.0
        MMCV_INDEX=https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
        REQUIREMENTS=requirements/veon-cu117.txt
        CUDA_PACKAGES=(pytorch-cuda=11.7)
        CHANNELS=(-c pytorch -c nvidia -c conda-forge)
        GCC_MAX=11
        command -v git >/dev/null || die 'Git is required to build Detectron2 for CUDA 11.7.'
        ;;
    *) die "Expected CUDA Toolkit 10.2, 11.3 or 11.7. Set CUDA_HOME to its directory. Found: $NVCC_INFO" ;;
esac
command -v gcc >/dev/null || die 'gcc is missing; install a matching GCC/G++ toolchain.'
command -v g++ >/dev/null || die 'g++ is missing; install a matching GCC/G++ toolchain.'
GCC_MAJOR="$(gcc -dumpversion | cut -d. -f1)"
GXX_MAJOR="$(g++ -dumpversion | cut -d. -f1)"
[[ "$GCC_MAJOR" -ge 5 && "$GCC_MAJOR" -le "$GCC_MAX" && "$GXX_MAJOR" -eq "$GCC_MAJOR" ]] || \
    die "CUDA $CUDA_VERSION needs matching GCC/G++ 5-$GCC_MAX. Select compatible compilers in PATH."

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONDA_ROOT="$(conda info --base)"
ENV_EXISTS="$(conda env list --json | "$CONDA_ROOT/bin/python" -c \
    'import json, os, sys; print(int(any(os.path.basename(p) == sys.argv[1] for p in json.load(sys.stdin)["envs"])))' "$ENV_NAME")"
CONDA_ACTION=create
if [[ "$ENV_EXISTS" == 1 ]]; then
    # A Torch/CUDA switch also requires rebuilding every compiled dependency.
    # Require a new environment instead of mixing an existing profile's wheels.
    if ! conda run -n "$ENV_NAME" python -c \
        'import sys, torch; sys.exit(0 if torch.__version__.split("+")[0] == sys.argv[1] and torch.version.cuda == sys.argv[2] else 1)' \
        "$TORCH_VERSION" "$CUDA_VERSION"; then
        die "Environment $ENV_NAME has another or incomplete PyTorch/CUDA profile. Choose a new name, e.g. veon-cu${CUDA_VERSION//./}."
    fi
    CONDA_ACTION=install
    printf 'Updating the requested environment: %s\n' "$ENV_NAME"
fi

printf '[1/5] Preparing Python 3.8, PyTorch %s and CUDA runtime %s...\n' "$TORCH_VERSION" "$CUDA_VERSION"
conda "$CONDA_ACTION" -y -n "$ENV_NAME" --override-channels "${CHANNELS[@]}" \
    python=3.8 "pytorch=$TORCH_VERSION" "torchvision=$VISION_VERSION" "${CUDA_PACKAGES[@]}" \
    numpy=1.20.3 'mkl<2024.1' pip=24.0 setuptools=59.5.0 wheel=0.38.4
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export FORCE_CUDA=1
# CUDA 10.2 compiles native targets through Turing; 11.x also supports Ampere.
TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('PyTorch cannot access CUDA. Check the driver and GPU allocation.')
caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
max_cap = (7, 5) if torch.version.cuda == '10.2' else (8, 6)
if any(cap < (3, 7) or cap > max_cap for cap in caps):
    raise SystemExit('CUDA {} profile targets compute capabilities 3.7-{}.{}. '
                     'Use a matching CUDA profile for your GPU.'.format(torch.version.cuda, *max_cap))
print(';'.join('{}.{}'.format(*cap) for cap in sorted(caps)))
PY
)"
[[ -n "$TORCH_CUDA_ARCH_LIST" ]] || die 'Could not determine a supported GPU architecture.'
export TORCH_CUDA_ARCH_LIST

printf '[2/5] Installing pinned BEVDet, SAN and depth dependencies...\n'
# Keep later pip operations from replacing PyTorch or upgrading the old ABI.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
cat "$REQUIREMENTS" > "$CONSTRAINTS"
printf '\ntorch==%s\ntorchvision==%s\nmmcv-full==%s\ndetectron2==0.6\nsetuptools==59.5.0\n' \
    "$TORCH_VERSION" "$VISION_VERSION" "$MMCV_VERSION" >> "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"
python -m pip install -r "$REQUIREMENTS"

printf '[3/5] Installing matching MMCV and Detectron2 CUDA operators...\n'
python -m pip install --only-binary=mmcv-full "mmcv-full==$MMCV_VERSION" -f "$MMCV_INDEX"
if [[ "$CUDA_VERSION" == 11.7 ]]; then
    # The v0.6 prebuilt wheel table has no cu117/torch1.13 combination.
    python -m pip install --no-build-isolation --no-deps -v \
        'git+https://github.com/facebookresearch/detectron2.git@v0.6'
else
    python -m pip install --only-binary=detectron2 detectron2==0.6 -f "$DETECTRON_INDEX"
fi

printf '[4/5] Compiling VEON BEVPoolv2...\n'
python -m pip install --no-build-isolation --no-deps -v -e .

printf '[5/5] Checking package imports and executing CUDA operators...\n'
python -m pip check
python tools/check_veon_env.py
printf '\nInstallation complete. Activate with: conda activate %s\n' "$ENV_NAME"
printf 'Prepare the datasets and pretrained weights as described in README.md before training.\n'
