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

脚本先将待上传文件复制到临时目录，保留原有相对路径，再执行一次：

```text
python /path/to/upload_folder.py /tmp/.../VEON/ s3://bucket-6425/tangpin/code/VEON/
```

临时空间约等于本次上传文件的总大小。上传结束会清理临时目录；暂存期间检测到文件
变化会停止并提示重试。同步正在训练的输出时，建议在检查点写入完成后执行。

上传工具需要符合你现有命令的行为：递归将源目录**内部的文件**按相对路径上传到目标
前缀，覆盖同名对象，保留目标中其他对象。源目录中的 `configs/a.py` 应变成
`s3://bucket-6425/tangpin/code/VEON/configs/a.py`。脚本不要求工具支持过滤参数。
仅在工具返回退出码 0 后更新同步记录；工具失败或中断时，下次执行会重试。
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

在训练环境的 VEON 根目录，继续使用你已有的下载命令：

```bash
python /path/to/upload_folder.py s3://bucket-6425/tangpin/code/VEON/ ./
```

数据集仍由训练环境本地的 `data/` 提供。上传脚本不会新增桶内的 `data/` 对象；如果该
前缀原来就存有数据，整前缀下载仍会获取那些旧对象，应选择干净的代码前缀。

当前仓库没有提供实际的 `upload_folder.py`，已用替代上传工具验证目录结构、增量上传
和失败重试，尚未连接内网数据桶验证真实传输。
