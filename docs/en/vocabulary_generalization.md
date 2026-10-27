# Vocabulary generalization ablations for the IJCV revision

本实现提供 Experiment A（固定 checkpoint 的动态词汇检索）和 Experiment B（移除训练 text prototypes 后重新训练）。不提供尚未测量的性能数字。原配置的训练词表、17 个 superclass priority、模型参数结构及原检索接口保持兼容。新消融通过独立入口启用。

## 代码审计与协议边界

1. `SANInVeonEntryTemporal.prepare_vocabulary()` 从 `nuscenes_vol.py` 展平 subclass 名称和描述，调用 SAN 的 `prepare_vocabulary()`。`SANInVeonTemporal` 用冻结 CLIP text encoder 生成 classifier weights，额外附加一个 SAN background embedding。该 background channel 与 Occ3D free slot 不是同一个训练文本类别。
2. SAN 用图像 mask embedding 与文本 prototypes 的相似度生成 mask logits；`semantic_inference_2d_w_embed()` 对类别 softmax 后移除 background，融合 mask probabilities 得到每个 subclass 的二维 pseudo-label score map。
3. `Proj2Dto3DLoss.forward()` 把 Occ3D occupied voxel centers 投影至图像，在二维 subclass score map 上采样。soft alignment 的目标来自最高分 subclass 的 text prototype，**不是**直接用 `sem_embed_2d` 做 dense feature regression；当前该 embedding 虽被传入和 detach，未进入最终 alignment 数值。
4. 原合并函数按相邻相同 `class_reflection` 分组，随后用合并顺序表示 superclass ID；原 `_onehot_restricted_class()` 也使用连续组编号。只删除部分 subclass 且每个 superclass 都仍有词时通常可以保持顺序，删除完整组后则会错位；任意重排也不安全。新代码直接按 Occ3D ID 0..16 max-reduce，不依赖排列或组是否存在。缺失组填 `-inf`，不能赢得 argmax；SAN background 保留为最终输出 slot 17。原完整词表的合并结果逐元素一致。
5. PCFA 使用 17 项 priority，按二维 pseudo label 和三维预测映射后的 **superclass ID** 索引。soft loss 先按每类样本数求逆权重，再乘 priority；epoch 达到 `stage2_start` 后，对三维预测高置信且预测 priority 更高的 soft 项忽略。`MEGVIIEMAHook.before_train_epoch()` 设置 epoch。`uniform` 将 priority 设为 1；仍保留按类逆频数均衡，**不应称为完全关闭 PCFA**。
6. `VeonTemporal.simple_test()` 生成 `feat_occ`。新检索分支调用原 SAN 的 `retrieval_embedding()`，从同一冻结文本编码器生成任意查询的 embedding，并对点所在 voxel feature 计算 cosine similarity。查询数量改变不需要改变 feature channels、occupancy head 或 checkpoint。只修改非持久化文本缓存，不修改模型参数和文件。
7. 原 `compute_single_retrieval()` 把预测与标注评测混在一起。它仍保留原行为。新 `score_retrieval_queries()` 将相同 voxel lookup / cosine 检索逻辑独立为无标注 API；入口在模型返回后才加载 retrieval labels 和 visible-point indices。默认测试 pipeline 中 `LoadOccGTFromFile`、`LoadRetrievalInfo` 被移除，`Collect3D` 只传预测所需字段，BEV 变换进入 `geometry_only=True`，不读取框或类别标注。

### 语义监督审计

`LoadOccGTFromFile` 原本确实加载 Occ3D `semantics`、`mask_lidar`、`mask_camera`。`forward_train()` 把 semantics 和 camera mask 交给 `OccLossFB`，后者将 camera 不可见 voxel 设为 ignore。具体语义 ID 在旧 projection loss 中参与 restricted prototype selection、soft/deterministic 分支选择，因此**不能对所有配置笼统宣称从不使用语义标签**。

但当前 DAV2/Zoe 主配置是 `out_channel=18, empty_idx=17, ov_class_number=17`：

- binary loss 只区分 ID 0..16 与 free=17，并忽略 255；
- 所有有效 occupied voxel 都进入 soft selection，具体 GT superclass 不再控制这一选择；
- deterministic loss 的系数为零，且 `OccLossFB.loss_voxel()` 不把它加入最终 loss dict；
- soft loss 的 targets、reweighting 和 priority ignoring 来自 SAN pseudo labels / 三维预测，未使用 GT semantic identity。

真实 CPU tensor 测试验证：在保持 occupied/free/ignore 和 camera mask 不变时，置换所有 occupied semantic IDs，默认有效 loss 完全一致，soft loss 对 feature 的梯度也完全一致。不能把“读取语义数组”误写成“默认配置已经发生语义 ID 泄漏”；也不能把这个结论推广至 `ov_class_number<17` 的设置。

新配对消融进一步把身份边界写死：`train_vocab.py` 在 loader 中将 ID 0..16 折叠为 occupied=0，loss 设置 `semantic_supervision='occupancy_only'`，该模式强制 `ov_class_number=17`。数据仍来源于 Occ3D occupancy 与 camera masks；这不是无几何监督，也不排除 CLIP、SAN、深度模型预训练中的相关知识。Stage-1 深度入口的有效训练 loss 来自 LiDAR 投影深度；检测框/类别没有在当前 Stage-2 有效 loss 中使用。

### Held-out 的准确含义

移除 subclass 的名称、对应描述/prototype、SAN pseudo-label channel；保留 siblings 和 broader superclass 文本。`car` 可以继续覆盖 `sedan` 物体，这属于 broader supervision，必须披露。基础模型预训练不是本实验 held out 的对象。

名称归一化处理大小写、空格、下划线和连字符，提供保守的显式 alias groups（如 road sign / street sign）。允许通过 `--aliases` 补充 JSON 列表。移除列表会按 alias closure 扩展，保留的描述直接提及被移除名称或常规复数时会拒绝运行。**自动化检查不能替代作者对同义词、broader concepts 和 query-to-annotation equivalence 的语义审核**。`nuscenes_brief` 是推荐起点；描述中包含广泛实例的 `nuscenes_default` 可能被检查拒绝。

完整 superclass 被删除时 manifest 会记录缺失的 superclass ID 并标记实际范围；subclass 结果不能表述为完整 superclass 的 zero-shot recognition。也不将任意新文本类别映射到 Occ3D 固定 17 类来报告 unseen-class IoU。

## 评测口径

参照 [POP-3D 官方 retrieval.py](https://github.com/vobecant/POP3D/blob/main/retrieval.py) 和当前 VEON 检索代码：每个 CSV 场景/查询计算一次 non-interpolated AP；主要 mAP 是有效场景/查询 AP 的等权均值。附加 `class_macro_map` 是每类场景/查询均值再按类平均，二者不能混写。per-class CSV 中的 AP 是该类有效场景/查询 AP 的均值，未把不同场景点混池。

- all / visible 分别计算；visible 来自 release 的 `matching_points`。
- 使用当前 VEON `RetrievalForPointsIndices` 的边界 clamp 规则，不引入 occupancy threshold 过滤。
- AP 使用二元 0/1 标注，明确忽略 -1/255。无点、无正样本、无负样本、非有限 scores 分别记录原因并返回 null，**不会给出虚假 AP**。相比 sklearn 对退化 target 的默认返回，这是有意的显式有效性过滤；每个 aggregate 都记录有效范围。
- `seen`：训练词表名称或声明的等价 alias；`heldout`：指定排除的名称或 alias；`novel`：其他查询。Experiment A 的新词默认属于 `novel`；不会无条件假装它们是 Experiment B 的 held-out subclasses。
- CSV prompt 只做准确归一化匹配。若 query 与 release prompt 不同，必须在 JSON 中明确声明 `annotation_prompt` 和 `class_name`。这是作者提供的 equivalence，不是程序发现的真实标签。不能把粗粒度 car 标注拿去验证 sedan 查询。
- original 模式默认查询原 subclass 名称；expanded / heldout 默认查询完整原词表与评测 CSV 的查询并集。指定额外查询或文件时，expanded / heldout 保留完整原词表，并附加指定查询；没有对应标注的查询仅保存定性 scores，AP 为 null。
- 同一 checkpoint 上原有 query 的 cosine score 不因其他查询加入而重新归一化；没有 query-set softmax。

所有结果保留实际参与均值的 query 和 instance keys。图表工具要求各模型在相同 group/scope 的有效实例完全一致，且 benchmark / annotations / info PKL 的 hash 一致；拒绝把 Original 与 Expanded 的不同总体 query sets 当成直接性能提升。

## 运行环境与依赖

以下命令在仓库根目录、README 的 Linux CUDA / MMDetection3D / Detectron2 / CLIP 环境执行。沿用仓库 DAV2 路径；Zoe checkpoint 应改用对应 `zoe-retrieval.py`，不要把不同模型配置与 checkpoint 混用。训练使用同一 SAN / depth 初始化、batch size、seed、24 epochs 和 PCFA schedule。

本机 2026-10-19 检查：没有 nuScenes infos、Occ3D labels、POP-3D benchmark、SAN/DAV2 checkpoint 或 CUDA/MMCV/Detectron2 环境。下面的模型 checkpoint 路径是 README 既有训练目录或新命令的明确输出路径，**本机尚不存在，不代表已经运行完训练**。没有生成真实 mAP。

```bash
cd /Users/tangpin/Documents/ChatGPT/VEON
# 在训练服务器上进入其 VEON checkout 后执行下方相同相对路径命令。
```

### Experiment A：只评测已有模型

使用 README 的 DAV2 训练产物 `epoch_10.pth`；original 和 expanded 两次评测读取同一文件。

```bash
python tools/ablation/eval_open_vocab.py \
  configs/veon/veon-temporal-base-512x1408-dav2-retrieval.py \
  work_dirs/veon-temporal-base-512x1408-dav2-nodepthcache/epoch_10.pth \
  --mode original --split test --seed 0 --out-dir work_dirs/vocab_A_original

python tools/ablation/eval_open_vocab.py \
  configs/veon/veon-temporal-base-512x1408-dav2-retrieval.py \
  work_dirs/veon-temporal-base-512x1408-dav2-nodepthcache/epoch_10.pth \
  --mode expanded --split test --seed 0 --out-dir work_dirs/vocab_A_expanded
```

第二条默认加入 release 中所有查询，只有实际标注支持的查询进入主要量化结果。任意文本的定性扩展示例（这些词的标注覆盖**尚未确认**）：

```bash
python tools/ablation/eval_open_vocab.py \
  configs/veon/veon-temporal-base-512x1408-dav2-retrieval.py \
  work_dirs/veon-temporal-base-512x1408-dav2-nodepthcache/epoch_10.pth \
  --mode expanded --queries 'delivery robot' 'yellow school bus' \
  --split test --seed 0 --save-score-samples 3 \
  --out-dir work_dirs/vocab_A_custom
```

`--eval-prompts` 支持每行一个查询的 TXT 或 JSON，例如 `["delivery robot"]`。需要作者审核的准确映射使用：

```json
[{"query": "police officer", "class_name": "police officer", "annotation_prompt": "police officer"}]
```

这里的示例不表示 benchmark 确有该查询；缺少完全匹配标注就输出 `no_matching_annotation`。

### Experiment B：先检查覆盖，再固定 split，配对训练和评测

先只用 POP-3D train split 的真实标注统计候选。下面选一个同时在 all / visible 有至少 10 个正样本、具有有效正负样本场景且仍保留 sibling 的 subclass。10 是可配置工程筛选阈值，**不是 POP-3D 的既定协议**。无合格候选会停止并留下 `coverage.json`，不会凭空选词；作者应查看 coverage 并确认语义/alias。

```bash
python tools/ablation/generate_vocab_split.py \
  --base nuscenes_brief --auto-count 1 --min-positive 10 \
  --benchmark-root data/nuscenes/retrieval_benchmark \
  --selection-split train --seed 0 --out-dir work_dirs/vocab_split
```

也可通过 `--held-out` / `--held-out-list` 明确指定类别，或 `--held-out-superclasses` 指定整组，仍默认检查标注覆盖。`--query-map` 为有明确等价关系的 benchmark prompts 提供 class_name；`--aliases` 为等价名称组的 JSON。`--allow-unannotated` 只适合结构验证/定性研究，会记录 `selection.verified=false`，不能作为已有定量证据。不要根据 test AP 选择 split。

query-map 会同时固化到 original / held-out manifests，默认配对评测自动复用；显式传入 `--eval-prompts` 可覆盖额外查询定义，应在两组使用同一文件。

配对训练命令：

```bash
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 \
  tools/ablation/train_vocab.py configs/veon/ablation/vocab_original.py \
  --train-vocabulary work_dirs/vocab_split/original_vocabulary.json \
  --out-dir work_dirs/vocab_original_seed0 --seed 0 --launcher pytorch --deterministic

PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=8 --master_port=29502 \
  tools/ablation/train_vocab.py configs/veon/ablation/vocab_heldout.py \
  --train-vocabulary work_dirs/vocab_split/train_vocabulary.json \
  --out-dir work_dirs/vocab_heldout_seed0 --seed 0 --launcher pytorch --deterministic
```

入口冻结完整 manifest 到 resolved config，并记录初始化 checkpoint hashes。标准 checkpoint 和本次消融的 EMA checkpoint 都保留 config / vocabulary 元数据。旧 EMA 没有 config 时，Original 只能标记为来源未验证的假设；新 held-out 评测会拒绝缺少内嵌词表的 checkpoint，或与 `--train-vocabulary` 不一致的声明。

训练结束后评测同一 epoch；两边都使用完整原词表 + 相同 POP-3D test queries，baseline 的 `--held-out-list` 只定义统计组，不改其训练词表：

```bash
python tools/ablation/eval_open_vocab.py \
  configs/veon/veon-temporal-base-512x1408-dav2-retrieval.py \
  work_dirs/vocab_original_seed0/epoch_24.pth \
  --mode expanded --train-vocabulary work_dirs/vocab_split/original_vocabulary.json \
  --held-out-list work_dirs/vocab_split/held_out.json \
  --split test --seed 0 --out-dir work_dirs/vocab_B_original_full

python tools/ablation/eval_open_vocab.py \
  configs/veon/veon-temporal-base-512x1408-dav2-retrieval.py \
  work_dirs/vocab_heldout_seed0/epoch_24.pth \
  --mode heldout --train-vocabulary work_dirs/vocab_split/train_vocabulary.json \
  --split test --seed 0 --out-dir work_dirs/vocab_B_heldout_full

python tools/ablation/plot_vocab_comparison.py \
  work_dirs/vocab_B_original_full/results.json work_dirs/vocab_B_heldout_full/results.json \
  --labels 'Original vocabulary' 'Held-out vocabulary' \
  --out-dir work_dirs/vocab_B_figure
```

`epoch_24_ema.pth` 也受支持，但比较时应两边一致选择 EMA。可选 `--priority-mode uniform` 训练应另设输出目录，明确命名为 uniform-priority ablation，而非 PCFA No。`--max-samples 2 --workers 0` 用于真实环境 smoke，输出标记 incomplete，绘图工具拒绝把它当最终结果。

默认 `--split test` 对应 `data/nuscenes/bevdetv2-nuscenes_infos_test.pkl`。train / val 同理；eval / all 必须提供覆盖所有 CSV token 的 `--ann-file`。info PKL 缺少 CSV 场景时会拒绝评测，不会静默减少样本。

## 输出

每次评测写出：

| 文件 | 内容 |
|---|---|
| `results.json` | 完整 metadata、per-instance、per-class、四个 group 的 all / visible aggregates |
| `metadata.json` | train/eval vocabulary、held-out、subclass mapping、配置、checkpoint/数据/代码 hashes、seed、PCFA/监督来源及精确协议 |
| `per_instance.csv` | 每个场景/查询/范围的 AP、正/负/忽略样本数、跳过原因 |
| `per_class.csv` | 类内 AP 均值、汇总正负样本数、有效/总 instance 数 |
| `aggregate.csv` | seen / heldout / novel / overall 的 scene-query mAP 与 class-macro mAP |
| `resolved_eval.py` | 实际使用的模型、预测 pipeline 和数据配置 |
| `scores_*.npz` | 前若干样本的查询、token、voxel indices、cosine scores，用于定性检查；不含伪造标注 |

AP/mAP 在机器文件中使用 0..1，论文图用百分比。绘图生成可编辑文字的 PDF、300 dpi PNG 和数据 CSV。无真实结果时所有论文表项应保留 TBD；无标注/无有效正负样本的组是 N/A，不能写 0。

## 已执行验证与限制

2026-10-19 本机执行：

```bash
/opt/anaconda3/envs/mayavi/bin/python -m unittest discover -s tests -p test_vocab_generalization.py -v
```

20 项通过，使用 Python 3.9.20 / PyTorch 2.4.1 CPU、真实 tensor operations、grid_sample、autograd、生产 loss 和 loader 方法。AST extraction 只去掉可选 CUDA/registry imports，不替代核心映射或监督逻辑。覆盖原词表和描述、非连续映射、subclass / superclass 删除、SAN 实际词表安装、prototype 数量、free/空组/无效索引、stage-1/2 PCFA loss 与梯度、empty/ignore volumes、语义 ID 置换、无标注 scorer、预测 pipeline 禁止标注访问、AP 退化场景、split CLI、aggregate 口径、EMA metadata。

旧 PCFA golden reference 来源于任务开始时 Git HEAD 的生产实现，保留了工作区已有的几何采样修复；seed=19，在 stage2 ignoring 前后保存数值和梯度。reference 文件记录 source hash / revision，**不是实验性能**。新增 AP 与 sklearn 在 341 个随机/同分有效查询上误差小于 1e-12。CLI help、原词表 manifest 生成、Python 编译、diff whitespace 检查已执行。绘图用明确标记为 synthetic 的测试数据做 smoke，不作为模型结果。

尚未验证：CUDA 完整模型加载/前向、真实 nuScenes/Occ3D 数据加载、POP-3D annotation 形状与覆盖、分布式完整训练、真实 AP/mAP、真实 checkpoint 的旧新数值一致性。固定 PCFA 数值和 CPU tests 不能代替这些目标环境验证。

作者在用于 rebuttal 前需要决定并冻结：最终 held-out 类别与 alias 审核、selection split/最小支持量、query-to-annotation equivalence、评测 split、基础 checkpoint/epoch/EMA 与训练 seed 数。对于没有细粒度标签的新增类别，只提供定性检索，不报告 unseen-class IoU 或伪造 AP。

## 文件范围

新文件：`mmdet3d/models/semantic_net/vocabulary/generalization.py`、`configs/veon/ablation/{vocab_original,vocab_heldout}.py`、`tools/ablation/{common,generate_vocab_split,train_vocab,eval_open_vocab,plot_vocab_comparison}.py`、`tests/test_vocab_generalization.py`、`tests/vocab_legacy_loss_reference.json`、本文档。

修改：SAN vocabulary entry、PCFA loss / OccLossFB、detector 的可选 label-free retrieval 分支、dataset 的可选 benchmark root 与 row ID、loader 的可选 binary/geometry-only 模式，以及仅为本消融保存元数据的 `tools/train.py` / EMA hook。工作区任务开始前已有的 geometry、resume 和 classifier-cache 修改已保留；没有重写已有 checkpoint。
