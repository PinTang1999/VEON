# 通过 upload_folder.py 同步到内网数据桶

在 VEON 根目录执行：

```bash
# 第一次可先检查待上传清单，不需要安装额外依赖或提供上传工具
bash sync_bucket.sh --dry-run --verbose

# 将你已有的 upload_folder.py 放在项目根目录，然后上传
bash sync_bucket.sh

# 上传工具在其他位置时，指定其路径
bash sync_bucket.sh --uploader /path/to/upload_folder.py
```

也可以直接运行 `python tools/sync_bucket.py`，功能相同。需要 Python 3.8+，
只有标准库依赖，不导入 Torch/MMCV。默认使用运行本脚本的 Python 调用上传工具；
如果上传工具依赖其他环境，可加 `--python /path/to/python`。

默认目标为 `s3://bucket-6425/tangpin/code/VEON/`。默认同步项目源代码、
配置、文档、资源，以及 `ckpts/`、`outputs/`、`output/`、`work_dirs/`、
`exps/` 等输出目录中的普通文件，包括权重、日志、图片和安装包。
脚本不使用 `.gitignore` 过滤，因此被 Git 忽略的 `.pth` 和训练输出也会同步。

默认排除：

- 项目根目录的 `data/`、`datasets/`。
- Git 元数据、Python 缓存、虚拟环境、构建目录、编辑器文件和 `.env` 文件。
- 根目录的 ZIP/TAR/WHL 等离线安装包；`outputs/` 等目录内的安装包仍然同步。
- 所有符号链接，包括指向数据集的目录链接。通过 `--verbose` 查看跳过的链接。

如果数据放在其他普通目录中，请使用 `--exclude` 额外排除。例如：

```bash
bash sync_bucket.sh --exclude custom_dataset/ --exclude 'resources/raw/*'
```

## 修改代码后的日常操作

```bash
bash sync_bucket.sh
```

首次上传全部选中文件，之后只上传新增或内容变化的文件。相同目标的成功同步记录
存放在本地 `.bucket_sync/`，不会上传，也不会提交到 Git。没有变化的文件通过大小、
修改时间和状态变更时间复用 SHA-256；元数据变化后重新计算内容哈希。
因此修改几行配置不会再次上传没有变化的大权重。

首次遇到的文件先只扫描元数据，在复制到暂存目录的同时计算 SHA-256，
省去提前读取全部大权重的开销。上传清单还会分别显示代码、`ckpts/`、`work_dirs/`
等目录的容量，方便确认大文件来自哪里。

脚本默认按约 **4 GiB 一批**暂存并上传，源码先于权重和输出。
每批仅复制本批文件到临时目录，保留原有相对路径，再执行：

```text
python /path/to/upload_folder.py /tmp/.../VEON/ s3://bucket-6425/tangpin/code/VEON/
```

每批成功后立即保存同步记录并清理临时目录，再处理下一批。即使全部文件有 162 GiB，
也不需要为全部文件同时准备 162 GiB 的暂存空间。**单文件不会拆分**：
若单个检查点超过 4 GiB，则该文件单独一批，临时磁盘仍须能容纳它。
脚本预览会显示最大一批需要的空间。可以调整批大小，或指定临时磁盘：

```bash
bash sync_bucket.sh --batch-size-mib 2048
# /path/to/large_disk/tmp 必须是已存在且可写的目录
TMPDIR=/path/to/large_disk/tmp bash sync_bucket.sh
```

训练还在写入的日志或检查点，在扫描或暂存期间发生变化时，会显示
`[跳过] ...；其他文件继续，下次同步重试。`，并移除不稳定的暂存副本。
变化文件不会阻止其他文件上传，也不会被记为本次上传成功；它若以前成功同步过，
仍保留上次成功版本的记录。写入完成后再次执行即可同步。
如果文件在本批暂存之前就变化，也会跳过，因此长时间上传时跳过活跃日志是正常情况。
脚本只读取开始时记录的字节数，避免持续增长的日志使哈希或复制无限延长。
存在跳过项时返回码仍为 0，结尾会明确统计其数量；需要完整同步时应在训练停止写入后重跑。

启动后立即显示 `[1/3] 开始扫描`。扫描大文件时会显示当前文件的哈希读取进度；
首次出现的大文件在暂存时才计算哈希。每批的 `[2/3]` 显示暂存文件数和字节进度，
`[3/3]` 开始调用上传工具。
进度默认每 5 秒更新一次，可用 `--progress-interval 2` 调整。
暂存时边复制边计算哈希，不再额外完整读取一次暂存权重。

`upload_folder.py` 中的 `mox.file.copy_parallel` 可能没有逐文件进度。
上传期间脚本定期显示已等待时间和子进程 PID，并禁用 Python 输出缓冲。
“上传工具仍在运行”仅表示子进程尚未退出，不表示已传输的字节数。
如果长期停留在上传阶段，可按 Ctrl+C 中断后，在相同 Python 环境用你原来的
小文件夹上传命令检查 moxing 的网络和认证。若停留在扫描或暂存阶段，则先看提示中的
文件名、读取进度，以及磁盘/挂载目录的读写情况。

上传工具需要符合你现有命令的行为：递归将源目录**内部的文件**按相对路径上传到目标
前缀，覆盖同名对象，保留目标中其他对象。源目录中的 `configs/a.py` 应变成
`s3://bucket-6425/tangpin/code/VEON/configs/a.py`。脚本不要求工具支持过滤参数。
仅在工具返回退出码 0 后确认本批文件的同步记录。工具失败或中断时，本批文件下次
重试；之前成功的批次保留记录，不会全部重传。
上传工具本身需要正确报告失败，脚本不额外调用桶 API 校验远端对象。

这是增量上传：本地删除或排除的文件不会从桶里删除，下载到训练环境也不会删除当地
旧文件。远端被人工修改、清空，或需要重新传输时，使用：

```bash
bash sync_bucket.sh --full
```

目标地址和上传工具也可通过环境变量设置，之后仍然只需运行同一条命令：

```bash
export VEON_UPLOAD_SCRIPT=/path/to/upload_folder.py
export VEON_BUCKET_URL=s3://bucket-6425/tangpin/code/VEON/
bash sync_bucket.sh
```

也支持 `--remote s3://bucket/another/VEON/`；不同目标分别保存增量记录。
`--dry-run` 不上传、不创建或更新同步记录。删除本地 `.bucket_sync/` 也会触发下次全量上传。

## 在训练环境下载

在新机器上 `git clone` 本仓库、准备 Python 环境并下载数据后，
激活可以使用 `moxing` 的环境，从 VEON 根目录执行：

```bash
bash download_bucket.sh
```

脚本默认只访问 `s3://bucket-6425/tangpin/code/VEON/` 下的以下目录：

| 桶中的子目录 | 本地目录 | 用途 |
| --- | --- | --- |
| `ckpts/` | `ckpts/` | SAN、深度模型和预训练权重 |
| `outputs/` | `outputs/` | 输出、离线安装包等 |
| `output/` | `output/` | 使用单数目录名保存的输出 |
| `work_dirs/` | `work_dirs/` | 实验检查点、日志、词表划分等 |
| `exps/` | `exps/` | 其他实验输出 |

不会整前缀下载，不访问桶中的根级代码或 `data/`、`datasets/`，因此保留 Git 拉下来的
代码和已经下载好的数据。权重按原始相对路径恢复，例如
`ckpts/clipsan/SAN_ViT-L.pth`，无需为了换机器修改该路径。

默认要求桶中有 `ckpts/`。桶中没有 `output/` 等可选输出目录时会显示跳过。
如果只需要做新实验的预训练权重，或只想拿回某次实验，指定目录即可，避免全量下载：

```bash
# 只恢复预训练权重
bash download_bucket.sh --paths ckpts

# 权重 + 指定的实验输出
bash download_bucket.sh --paths ckpts work_dirs/ijcv_l_t2

# 权重 + outputs（保留其他历史实验在桶里）
bash download_bucket.sh --paths ckpts outputs

# 不联网、不写入，只预览来源与目标路径
bash download_bucket.sh --dry-run
```

`--paths` 只接受上表中的目录及其子目录，禁止代码、数据目录和越界路径。
显式选择的目录若不存在会返回失败。同一父目录和子目录重复选择时只下载一次。
工具需要放在项目根目录；如果位于其他位置或依赖其他 Python 环境，可以指定：

```bash
bash download_bucket.sh --uploader /path/to/upload_folder.py
bash download_bucket.sh --python /path/to/python
bash download_bucket.sh --remote s3://bucket-6425/tangpin/code/VEON/
```

下载脚本与上传脚本共用 `VEON_UPLOAD_SCRIPT`、`VEON_BUCKET_URL`、
`VEON_SYNC_PYTHON` 环境变量，也可以运行 `python tools/download_bucket.py`。
它复用 `tools/sync_bucket.py` 中的工具查找和进程管理；两份 Python 文件都应来自当前仓库。

脚本先在所选 Python 中通过 `mox.file.exists`、`mox.file.is_directory` 检查远端目录，
再逐目录调用原有的 `upload_folder.py`。这些检查接口见
[MoXing 官方示例](https://support.huaweicloud.com/intl/en-us/devtool-modelarts/modelarts_11_0005.html)。
检查或传输长时间没有输出时，每 5 秒显示等待时间，可用 `--progress-interval 2` 调整。

文件直接下载到对应本地目录，不再额外暂存全部权重；磁盘仍需有容纳最终文件的空间。
同名文件由现有工具覆盖，本地多余文件保留，不做删除。为避免通过本地链接写入代码
或数据，下载目标及其内部若已有符号链接，脚本会停止并列出该链接。

某目录下载失败时会保留已下载文件、继续其他目录，最后返回非零退出码并打印失败项的
重试命令。重新执行会再次调用选中目录的下载，是否省略相同文件由现有工具决定，
不保证文件级增量或字节级断点续传。没有额外的远端哈希验证。

下载完成后，按原有训练命令指定配置开始实验。脚本只恢复桶里已有的文件，不会自动
启动训练或改变配置中的 `load_from` / `resume_from`；原上传时未成功同步的文件及被跳过
的符号链接（如某些 `latest.pth`）也不会凭空恢复，需要选择实际存在的检查点文件。

当前 `upload_folder.py` 使用 `mox.file.copy_parallel(src, dst)`。
已用替代上传工具及离线 MoXing 接口验证上传/下载目录结构、代码和数据保留、
增量上传、变化文件跳过、分批失败重试和进度输出，
尚未连接内网数据桶验证真实传输。
