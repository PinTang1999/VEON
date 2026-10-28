# VEON 分步安装：已有 Python 3.8 / PyTorch 1.12.1 / CUDA 11.3

适用条件：Linux x86_64、V100、内网 pip 镜像可用。保留现有的 PyTorch
1.12.1 和 torchvision 0.13.1，按下列步骤增加 VEON 依赖。这是兼容配置，尚未在 V100 上实测完整
安装和训练；不是 VEON 原 Dockerfile 环境的逐版本复刻。

PyTorch 1.12.1 对应 torchvision 0.13.1，这两个包已经安装好，无需重装。
两者都应使用匹配 CUDA 11.3 的构建。不要安装为 PyTorch 1.10 构建的 MMCV
或 Detectron2 wheel。

本文件包包括：

- MMCV-full 1.6.0，CUDA 11.3 / PyTorch 1.12.0 / CPython 3.8 wheel。
  OpenMMLab 为 1.12.x 提供的该构建用于 1.12.1。
- Detectron2 v0.6 官方源码 ZIP，在服务器上用现有 PyTorch/CUDA 编译。
- 普通依赖清单、版本约束和更新后的环境自检工具。
- 原版 VEON 的模型/测试 API 循环导入修复补丁。

## 1. 解压并检查当前环境

先激活已经安装 PyTorch 1.12.1 的环境，在 VEON 根目录解压文件包：

```bash
unzip -o veon-torch112-cu113-manual.zip
git apply --reverse --check veon-import-cycle-fix.patch 2>/dev/null || git apply veon-import-cycle-fix.patch
```

补丁只将测试 API 的模型类导入移到 `single_gpu_test()` 函数内部；已经应用
过补丁的仓库会跳过。这使模型优先导入的环境检查可以正常完成初始化。

选择实际包含 `nvcc` 的 CUDA 11.3 Toolkit。若按之前的用户目录方式安装：

```bash
export CUDA_HOME="$HOME/cuda-11.3"
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=4

python -c "import sys, torch, torchvision; print(sys.version); print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc -V
gcc --version
g++ --version
```

如果 Toolkit 在 `/usr/local/cuda-11.3`，将 CUDA_HOME 改成这个目录。
应看到 Python 3.8、PyTorch 1.12.1、torchvision 0.13.1、`torch.version.cuda == '11.3'`、CUDA
可用为 True，且 nvcc 为 11.3。GCC/G++ 应为匹配的 5–10 版本。
只安装 Conda 的 cudatoolkit 不包含本项目编译扩展所需的 nvcc。

## 2. 固定版本与构建工具

以下 pip 命令使用服务器已经配置好的内网镜像，不额外指定外网源。
版本约束防止安装其他包时连带更换 PyTorch 或 torchvision。

```bash
export PIP_CONSTRAINT="$PWD/requirements/constraints-veon-torch112-cu113.txt"
python -m pip install pip==24.0 setuptools==59.5.0 wheel==0.38.4
```

这些命令应在同一终端中依次执行，以保留 CUDA_HOME 和 PIP_CONSTRAINT。

## 3. 从内网镜像安装普通依赖

```bash
python -m pip install -r requirements/veon-torch112-cu113.txt
```

核心版本：NumPy 1.20.3、Numba 0.53.0、SciPy 1.7.3、MMDet 2.25.1、
MMSeg 0.25.0、OpenCLIP 2.16.0、timm 0.6.12、einops 0.6.1、
fvcore 0.1.5.post20221221。清单也包含 MMCV/Detectron2 的全部直接 Python
依赖，故下一步可以关闭自动依赖下载。

共享文件叫 `veon-cu113.txt`，它不包含 PyTorch 的版本要求；此处的
`constraints-veon-torch112-cu113.txt` 将 PyTorch 固定为 1.12.1。

## 4. 安装本地 MMCV 并编译 Detectron2

```bash
python -m pip install --no-index --no-deps ./veon_external_torch112_cu113/mmcv_full-1.6.0-cp38-cp38-manylinux1_x86_64.whl

python -m pip install --no-index --no-cache-dir --no-build-isolation --no-deps -v ./veon_external_torch112_cu113/detectron2-0.6.zip
```

Detectron2 的 0.6 官方预编译表没有 PyTorch 1.12 的对应构建，因此此处
从本地源码编译；不会访问 GitHub。不使用旧 Torch 环境缓存的 Detectron2 wheel。

## 5. 编译安装当前 VEON

```bash
python setup.py build_ext --inplace --force
python -m pip install --no-index --no-build-isolation --no-deps -v -e .
```

强制重编译 BEVPoolv2，确保使用当前 PyTorch 1.12.1，而非之前环境生成的算子。

## 6. 验证

```bash
python -m pip check
python tools/check_veon_env.py
```

自检包括模型/数据集/训练入口导入，CUDA NMS，以及 BEVPoolv2 前向和反向。
最终应显示 `PASS`。检查通过仍不能代替真实数据上的训练验证。
数据集和预训练权重仍按 VEON README 准备。

## 官方版本依据

- [PyTorch 旧版安装组合](https://pytorch.org/get-started/previous-versions/)
- [MMCV CUDA 11.3 / Torch 1.12 wheel 索引](https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html)
- [MMCV 安装说明](https://mmcv.readthedocs.io/en/v1.6.0/get_started/installation.html)
- [Detectron2 v0.6 安装说明](https://github.com/facebookresearch/detectron2/blob/v0.6/INSTALL.md)
