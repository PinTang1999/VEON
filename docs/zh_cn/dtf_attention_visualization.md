# DTF 注意力可视化

## GPU 服务器

在服务器的 VEON 工作目录中运行。脚本使用 T2 配置及其原有测试 pipeline，加载主 checkpoint 后，按 `tools/test.py` 的顺序加载配置中的 ZoeDepth checkpoint。测试集需具备该 pipeline 使用的 nuScenes 信息文件、图像、点云和 Occ3D 标签。

```bash
cd /home/ma-user/work/code/VEON
conda activate veon

# 一键导出一个样本，再生成 PDF 和 300 dpi PNG
bash tools/run_dtf_visualization.sh

# Smoke test：仅导出一个 NPZ，并在同一批输入上开关采集各推理一次
bash tools/run_dtf_visualization.sh export --sample-index 1 --verify-output
```

Smoke test 成功时会打印 `Verified identical occupancy labels` 以及 NPZ 路径。默认输出为 `work_dirs/dtf_visualization/sample_000001.npz`（索引 1）。导出前会校验两个分支的张量形状、概率和及采样坐标范围。`--verify-output` 比较 VEON 最终返回的逐体素类别 ID。

默认 config 为 `configs/veon/veon-temporal-large-512x1408-zoe-withdepthcache-T2.py`。在 `work_dirs/ijcv_l_t2_true/` 中优先使用有效的 `latest.pth`，其次按数字取最大的普通 `epoch_N.pth`。目录中仅有 EMA 系列时按数字取最大的 EMA epoch。如果只有一个其他名称的 PTH，则使用该文件。也可以用 `--checkpoint FILE.pth` 明确指定普通或 EMA 权重。NPZ 保存解析后的 checkpoint 路径和配置路径。不同实验建议分别指定 `--out-dir`，同一目录内相同索引会更新同名 NPZ。

默认选择第一个具有独立历史帧的测试样本。若数据集的历史帧全部为重复帧，则使用索引 0。使用 `--sample-index` 或 `--sample-token` 可以复现具体样本。当 pipeline 在场景边界重用当前帧充当历史帧时，图中会标明。使用 `--gpu-id` 选择当前可见 GPU 的序号。

## Mac 独立绘图

将 NPZ 复制到本地。绘图仅需 NumPy 和 Matplotlib，可在已有的绘图环境中运行。如果需要新建本地环境，可在仓库根目录执行以下命令。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
```

以下示例假定 NPZ 保存在仓库的 `work_dirs/dtf_visualization/` 下。

```bash
bash tools/run_dtf_visualization.sh plot work_dirs/dtf_visualization/sample_000001.npz

# 选定第 2 个 head 和指定的 DTF 体素，生成另一组图片
bash tools/run_dtf_visualization.sh plot work_dirs/dtf_visualization/sample_000001.npz \
  --head 1 --query 50 45 2 --out-dir work_dirs/dtf_figures
```

`--query X Y Z` 和 `--head` 均从 0 开始计数。T2 的查询网格为 100 × 100 × 8，每个 DTF 体素边长为 0.8 m。省略查询坐标时，会在预测为占用的 DTF 体素中，选取当前分支、指定 head 的最大单点概率最高的位置。如果整个网格都预测为空闲，则在全网格中选择。省略 `--head` 时使用 head 0。输出文件名包含选用的 head 和坐标。`--dpi` 控制 PNG 分辨率，PDF 中的文字和采样点使用矢量图元。

## 读图和 NPZ 字段

每行代表一个 DTF 分支。`current` 的 Query 和 offset 来自当前帧特征，`history` 的 Query 和 offset 来自已对齐的历史特征。两行的 Key 和 Value 都从同一融合参考特征采样。采样位置均处于当前 LiDAR 时刻的自车坐标系。

左列显示所选高度切片中，每个查询体素在采样点维度上的最大 softmax 概率。白色轮廓表示这一高度的预测占用边界。中列、右列显示选中查询体素的实际采样点在 x–y 和 x–z 平面的投影，红叉是查询中心，坐标单位为米。两个分支采用相同的峰值概率色标，两组采样点投影也共用各自的概率色标。多个点可能投影到同一位置，可从 NPZ 逐点读取。

| 字段 | 含义 |
| --- | --- |
| `{current,history}_weights` | `[head, sample, Z, Y, X]`，进入加权求和的 softmax 权重，每个 head 和查询体素沿 sample 轴的和为 1 |
| `{current,history}_offsets_xyz` | `[head, sample, Z, Y, X, 3]`，Tanh 后的原始位移，按 x、y、z 排列，单位为 DTF 体素 |
| `{current,history}_grid_xyz` | 同形状，实际传给 `grid_sample` 的坐标，已按原模型的 offset scaling 缩放并裁剪至 `[-1,1]` |
| `origin_xyz`、`voxel_size_xyz` | 网格下边界和 DTF 体素尺寸，单位为米，均按 x、y、z 排列 |
| `occupied_zyx` | `[Z,Y,X]`，每个 DTF 体素中是否含有预测为占用的细体素 |
| `sample_index`、`sample_token`、`scene_token`、`history_tokens` | 测试集索引和当前、历史样本标识 |
| `schema_version`、`config`、`checkpoint`、`depth_checkpoint` | 文件格式版本 1 和推理所用配置、权重路径 |

T2 每个分支的权重形状为 `[4,8,8,100,100]`。使用 `np.load(path, allow_pickle=False)` 即可读取。各 head 独立对本分支的 8 个采样点归一化。

## 本地回归测试

```bash
python -m pytest -q tests/test_dtf_visualization.py tests/test_models/test_veon_sampling.py
bash -n tools/run_dtf_visualization.sh
```

测试使用 NumPy、Matplotlib、pytest 和 CPU 版 PyTorch。张量测试覆盖两分支、非立方网格、单层 Z、边界裁剪和采集开关的逐元素输出一致性。离线测试通过同一个 shell 入口从 NPZ 实际生成 PDF 和 PNG。
