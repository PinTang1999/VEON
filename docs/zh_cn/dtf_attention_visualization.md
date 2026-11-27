# DTF 注意力与语义占用可视化

默认加载 `work_dirs/ijcv_l_t2_true/epoch_10.pth` 和 `configs/veon/veon-temporal-large-512x1408-zoe-withdepthcache-T2.py`。所有命令都从仓库根目录运行。可视化的采集器只在 `run` 和 `export` 中启用。两种模式复用 `tools/test.py` 的模型构建、主权重与 ZoeDepth 权重加载顺序，以及测试集的图像、深度、点云和历史帧处理。导出器单独读取可选的当前与历史 Occ3D 标注。

## 截图中的灰线和真正的模型轮廓

原图 XY、XZ 中从红叉通往圆点的灰线由 query 和采样终点直接连接生成。旧版的预测占用信息只用于左侧全局峰值图。它先把最终类别中非 free 的细体素做布尔 `any` 池化，再在粗网格高度切片上画 0.5 等值线。旧 NPZ 只存储这个池化掩码。

新版直接读取同一次 `simple_test` 最终返回的 semantic occupancy。模型在二分类占用概率大于 0.5 且最大语义概率大于 0 的位置取语义类别，其余位置使用 free（17）。导出时将原始 XYZ 数组转为 `prediction_zyx`。在穿过 query 中心的细网格切片上，沿相邻已知体素的公共边绘制占用与 free 的边界，以及不同已占用类别的交界。unknown（255）和图幅外沿不产生物体边界。相接的同类实例在语义图上可能连通。

| 图例 | 数据和画法 |
| --- | --- |
| Predicted semantic boundary | 赭色实线，来自最终 DTF 预测标签；叠加在其他帧背景上时仍表示当前帧的最终 DTF 预测 |
| Sampling connections | 半透明灰色虚线，连接 query 中心与实际采样点的二维投影 |
| Query voxel / Sampling location | 红叉为 query 中心，圆点为实际采样位置，颜色和面积表示原始权重 |

在三维采样诊断图中，XY 固定 query 的 z，XZ 固定 y，YZ 固定 x。三维采样点投影到这三个平面，终点可以来自切片以外的体素。坐标轴均为当前 LiDAR 时刻自车坐标，单位为米，每个方向均使用同一物理比例。若中心恰落在两个细体素之间，使用左闭右开的网格约定。统一颜色的类内区域不会凭空出现轮廓。

## 服务器运行

服务器需具备配置所指向的权重、验证集索引、图像和点云。有 Occ3D 标注时自动保存语义和相机可见性。先进入项目和环境。

```bash
cd /home/ma-user/work/code/VEON
conda activate veon
```

**T2 单样本 smoke test**。复现截图中的 sample 1、head 0 和 query (50, 43, 0)，生成 NPZ、论文图和诊断图。`--verify-output` 对同一批输入分别启用和关闭采集，逐体素比较最终类别，成功时打印 `Verified identical occupancy labels`。

```bash
bash tools/run_dtf_visualization.sh run --sample-index 1 \
  --query 50 43 0 --head 0 --verify-output --out-dir work_dirs/dtf_ijcv_t2
```

**T2 批量生成论文图**。从不同场景选择至多 12 个具有独立历史帧的样本，每个样本选至多 2 个满足类别和边界条件的 query。`--head all` 导出每个 head 和跨 head 汇总，`--head aggregate` 只出汇总图。某个样本无匹配时在 JSON 记录零个 query，整批无匹配时命令返回错误。

```bash
bash tools/run_dtf_visualization.sh run --num-samples 12 --distinct-scenes \
  --classes car truck bus pedestrian --selection-source prediction --strategy boundary \
  --num-queries 2 --head aggregate --radius 4 --figure-mode publication \
  --out-dir work_dirs/dtf_ijcv_t2_batch
```

**T3 单样本 smoke test**。将环境变量 `T3_CKPT` 设为服务器上实际的 T3 权重文件。每个历史帧生成一张有对应语义背景的论文图，图中复用同一次递归历史分支的注意力。可用 `--sample-token TOKEN` 替代 `--sample-index` 指定样本。

```bash
bash tools/run_dtf_visualization.sh run \
  --config configs/veon/veon-temporal-large-512x1408-zoe-withdepthcache-T3.py \
  --checkpoint "${T3_CKPT:?set T3_CKPT to the trained T3 checkpoint}" \
  --num-samples 1 --distinct-scenes --head aggregate --verify-output \
  --out-dir work_dirs/dtf_ijcv_t3
```

**服务器只导出，Mac 独立绘图**。服务器端使用 `export`，将生成的 NPZ 复制到 Mac 仓库（示例中的相对路径相同）。Mac 只需要 NumPy 和 Matplotlib，无需导入 PyTorch、MMCV、nuScenes 或 CUDA。输入也可以是含 `sample_*.npz` 的目录。通过 `PYTHON=/path/to/python` 可以指定已有绘图环境。

```bash
# GPU
bash tools/run_dtf_visualization.sh export --sample-index 1 --verify-output \
  --out-dir work_dirs/dtf_ijcv_t2

# Mac
bash tools/run_dtf_visualization.sh plot work_dirs/dtf_ijcv_t2/sample_000001.npz \
  --query 50 43 0 --head 0 --radius 4 --out-dir work_dirs/dtf_paper
```

同一条 `plot` 命令也能读旧 v1 NPZ。当输入为 v1 时，诊断图中的轮廓标为 `Predicted occupancy boundary (pooled)`。真实细分辨率多类别轮廓、帧语义背景及 pre-Tanh 分布需要在服务器重新导出 v2。同一样本索引在同一导出目录会更新同名 NPZ，比较不同 checkpoint 时应使用各自的目录。

## Query 选择和对照模型

- `--query X Y Z` 使用从零开始的 DTF 网格索引。`--metric-query X Y Z` 使用当前自车米制坐标，再取包含该位置的 query 体素中心。手动模式固定一个 query。
- 自动模式的 `--classes` 接受 Occ3D 类别名，如 car、truck、bus、pedestrian、bicycle、motorcycle、barrier。默认候选为已占用的已知类别。`--selection-source prediction` 使用 DTF 标签，`gt` 使用当前 GT，并在元数据中记录。
- `auto` 优先考虑边界、可移动类别和低语义置信度。`boundary` 筛选边界，`dynamic` 筛选可移动类别（含停车车辆），`low-confidence` 按语义置信度从低到高选取。有独立 current-only 置信度时优先使用，否则记录和使用 DTF 最终语义置信度。`occluded` 要求当前 `mask_camera` 为假且至少一个不同历史帧对齐后为真。这是相机覆盖代理条件。运动速度和被遮挡物体的实例身份需要其他证据。
- `--num-queries` 控制每个样本的上限，邻近体素会被抑制以减少重复示例。`--sample-stride` 控制批量扫描步长。`--radius` 控制局部半径，默认 ±4 m。
- 全局和局部语义背景都从最高 Z 层向下取首个可见占用标签。遇到 free 或 unknown 继续向下。有观测但全为空闲的列显示白色，完全无观测的列显示灰色。当前 GT 和论文预测使用当前相机掩码；历史 GT 使用经过同一 SE(3) 最近邻重采样的历史掩码。缺少掩码时面板标记 Unavailable，同时输出警告。
- `semantic_context_raw` 保留完整 DTF 预测。样本级 `projection_diagnostics.json` 记录各源的类别数和比例、无观测列数、相机可见体素比例和首个占用高度分布。高度 −1 表示没有占用命中。查询的具体高度和该体素的原始类别、可见性另存于 `queries.json`。
- `*_attention_bev` 跨全部高度累积端点概率。`*_attention_query_height` 仅累积落入 query 所在细体素 Z 层的端点，沿用原始概率，质量和可以小于 1。论文叠加图中的语义颜色取每列首个占用体素，采样概率则跨高度累积。
- 可选的 current-only 与 ACC 面板显示独立训练模型的结果。缺失的对照标为 `Unavailable`。如有 GT，在 `queries.json` 记录当前 query 类别在局部俯视投影上的 intersection、union 和 IoU。各模型使用共同的有效区域，有相机掩码时只统计可见体素。该局部结果不能替代验证集上的整体评测。

提供服务器独立训练的 ACC 配置和权重时，在主模型完成导出后按相同 sample token 运行 ACC，并保存 `acc/{token}.npz`。

```bash
bash tools/run_dtf_visualization.sh run --sample-index 1 --head aggregate \
  --acc-config "${ACC_CONFIG:?set ACC_CONFIG}" --acc-checkpoint "${ACC_CKPT:?set ACC_CKPT}" \
  --out-dir work_dirs/dtf_with_acc
```

已有对照预测时，可以在 `plot` 或 `run` 中按需传入下列参数。模板中的历史 token 和 index 对应该历史帧自己的原始坐标系。独立文件需要 `prediction_zyx`、`sample_token`、`origin_xyz` 和 `fine_voxel_size_xyz`，可选 `semantic_confidence_zyx`。几何、标签 ID 和 token 在加载时检查。

```bash
bash tools/run_dtf_visualization.sh plot work_dirs/dtf_ijcv_t2/sample_000001.npz \
  --current-only 'work_dirs/current_only/{token}.npz' \
  --history-prediction 'work_dirs/current_only/{token}.npz' \
  --acc 'work_dirs/dtf_with_acc/acc/{token}.npz' --head aggregate
```

## DTF 提取、坐标与读图

共享模块先调用 `t_deform(mid_feat, current)`，再调用 `t_deform(mid_feat, history)`。两次均由第二个参数产生 Query 和 offsets。K、V 都从第一个参数的投影上采样，`mid_feat` 已融合当前与历史信息。T2 的历史 Query 来自对齐后的上一帧。T3 按从远到近的顺序融合历史特征，再生成一组历史 Query。T3 的多张背景图可以展示相同采样位置处不同帧的语义证据，但模型只提供一组历史分支权重。

| NPZ 字段 | 形状和含义 |
| --- | --- |
| `{current,history}_weights` | `[head,point,Z,Y,X]`，实际加权求和前的 softmax 概率。模型在 `[B,head,point,Z,Y,X]` 的 dim=2 对每个分支、head 和 query 独立归一化 |
| `*_pre_tanh_xyz`、`*_offsets_xyz` | 前者是 offset 卷积输出，后者是模型 Tanh 激活后的体素位移，形状均为 `[head,point,Z,Y,X,3]` |
| `*_grid_xyz` | 与 offsets 同形，末轴顺序为 x、y、z。读取实际传给 `grid_sample` 的裁剪后网格 |
| `origin_xyz`、`voxel_size_xyz`、`fine_voxel_size_xyz` | 网格下边界、DTF 间距和原始占用细体素间距，单位为米 |
| `prediction_zyx`、`occupied_zyx` | 最终语义细网格，以及向后兼容的粗网格布尔占用 |
| `semantic_confidence_zyx`、`occupancy_probability_zyx` | 语义 softmax 最大值，以及独立二分类头的占用概率 |
| `gt_zyx`、`visible_zyx`、`history_{i}_gt`、`history_{i}_visible` | 可选的原生细网格 GT 和相机掩码。历史数组保留其本帧原始坐标，i 从 0 开始 |
| `ego_to_global`、`feature_rms_{i}` | 当前帧和各历史帧的位姿，以及融合模块入口的通道 RMS。RMS 已处于当前帧坐标 |
| `config`、`checkpoint`、`depth_checkpoint`、`sample_token`、`history_tokens` 等 | 推理来源、样本索引以及历史帧的 token 与索引 |

每个轴取 DTF 大小 N、网格下边界 o、DTF 体素宽度 s、查询索引 i 和实际采样坐标 g。query 中心为 `o + (i + 0.5) * s`。模型按照 `2/(N-1)` 缩放激活后的 offset，与参考网格相加并裁剪至 [-1,1]。`align_corners=True` 的实际采样位置为 `o + (0.5 + (g+1)*(N-1)/2) * s`。单体素轴位移为零。边界使用细网格的 `o + j * fine_spacing`，因此轮廓落在格面、红叉落在粗网格中心、圆点可位于亚体素位置。

`*_attention_bev` 将选中 query 的各采样概率按双线性系数累积到细 BEV 网格，在全局与局部视图、两个分支间共用同一质量色标。多 head 汇总对各 head 的质量图等权取均值，所有散点仍保留各自真实位置和原始概率。诊断图左列的 peak 图给出每个 query 的最大单点概率，与选中 query 的空间质量具有不同含义。

`queries.json` 还记录每个三维采样终点所在的原生细体素、DTF 预测类别，以及可用的当前和对齐历史 GT、相机可见性与独立预测类别。可用这些记录核对投影到同一位置的采样点是否具有不同高度或语义。

### 截图中 ±0.8 m 的解释

T2 配置的细网格为 0.4 m，DTF 网格为 0.8 m。Tanh 将每个轴的位移限制在一个 DTF 体素以内，端点接近 ±0.8 m 与模型允许的范围吻合。截图中的 (50,43,0) 对应当前自车 (0.4,−5.2,−0.6) m，位于最低 DTF 层。向下越过边界的偏移会被裁剪为 Δz=0。二维重合也可能来自投影丢失了一个坐标。单张 JPG 无法测量三维重合率、Tanh 饱和比例或权重熵。

`diagnostics.json` 按分支和 head 遍历所有 query，包含原始和激活后的 offset 分布、各轴接近 Tanh 极限的比例（绝对值 ≥0.95 个体素）、触发网格裁剪的样点比例、归一化坐标回算误差、概率和的最大误差、峰值 ≥0.95 的比例、平均熵（nats），以及三维采样点对距离 ≤5 cm 和 ≤1 µm 的比例。直方图同时保留 bin 边与计数，可用多个样本按计数汇总。对于真实模型的注意力集中程度，应查看服务器导出的结果。

## 输出与验证

当使用 v2 和 `--figure-mode both` 时，默认结果目录如下。`publication` 生成论文和场景图，`diagnostic` 生成诊断图与直方图。两种模式均保存数值统计和所选 query 的空间质量图。论文 PNG 至少 300 DPI，同时输出含矢量文字、线段和采样点的 PDF。

```text
work_dirs/dtf_ijcv_t2/
  sample_000001.npz
  acc/<token>.npz                         # 若运行独立 ACC
  sample_000001/
    diagnostics.json
    histograms.{png,pdf}
    queries.json                          # 来源、类别、坐标、head、每个三维采样点和权重
    q50-43-0/
      semantic_context.{png,pdf}          # 全局与局部
      h0_diagnostic.{png,pdf}             # 峰值图与 XY/XZ/YZ 投影
      h0_attention_bev.{png,pdf}
      h0_attention_mass.npz
      h0_history1_paper.{png,pdf}
      h0_history2_paper.{png,pdf}         # T3
      aggregate_...                       # 选择汇总时
```

实现位于 `tools/dtf_visualization.py`（采集和批量导出）、`tools/analysis_tools/dtf_scene.py`（语义、几何与 query）、`dtf_plot.py`（投影和图例）、`dtf_figures.py`（多面板、对照和元数据）、`dtf_diagnostics.py`（统计）、`mmdet3d/models/detectors/veon_temporal.py`（可选置信度观察）和 `mmdet3d/utils/vis.py`（共享既有配色）。入口是 `tools/run_dtf_visualization.sh`。

本地可运行合成和 CPU 回归测试，覆盖非立方网格、XYZ 与 ZYX 转换、细体素边界、历史位姿与相机掩码、采样点权重和、原始 DTF 张量、开关采集的一致性、T2/T3 配置和离线图表输出。

```bash
bash -n tools/run_dtf_visualization.sh
bash tools/run_dtf_visualization.sh plot --help
python -m pytest -q tests/test_dtf_visualization.py tests/test_models/test_veon_sampling.py
```

真实 CUDA 推理、nuScenes 的场景对齐、T2/T3 训练权重、独立训练 baseline 以及跨样本的实际统计，分别使用上面的服务器命令验证。
