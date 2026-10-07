# MiniMax-H3 单卡推理加速实验

本仓库记录 MiniMax-H3 在单张 NVIDIA A100-SXM4-80GB 上的推理、监控、性能分析与加速实验。正式主线仅针对 **T2VA（文本生成视频+音频）**：1344×768、124 帧、24 fps、50 个 sigma 网格点（实际 49 次 Transformer 前向）；Turbo LoRA 的 8-evaluation 结果作为独立探索记录。

完整的架构分析、实验方法和质量结果见 [MiniMax-H3 推理指南](docs/minimax-h3-inference-guide.md)。

## 主要结论

统一环境：Python 3.12、PyTorch 2.13.0+cu130、CUDA 13.0、diffusers 0.40.0、SageAttention 2.2.0。

| 配置 | denoise（秒） | 每视频总耗时（秒） | 相对 raw |
|---|---:|---:|---:|
| raw cold（默认 SDPA） | 829.9 | 1269.0 | 1.00× |
| FBCache 0.25 | 151.5 | 581.7 | 2.18× |
| SageAttention | 749.9 | 1186.7 | 1.07× |
| SageAttention + FBCache 0.25 | 136.9 | 569.5 | 2.23× |
| 上述组合，阶段主序 batch=20 | 144.8 | 191.1/视频 | 6.64× |

- 冷启动加载约 397 秒；`HF_ENABLE_PARALLEL_LOADING` 现场 A/B 仅改善 0.13%，确认当前瓶颈仍是约 350 MiB/s 的存储读取。
- FBCache 减少完整 Transformer 前向次数；阈值越高，速度越快，但质量损失通常越大。
- SageAttention 缩短单次注意力计算，与 FBCache 的收益近似乘法叠加。
- 阶段主序批处理让多个不同 prompt 共享组件加载，不是 tensor batch。
- Turbo LoRA v4 EMA 将 denoise 从 829.9 秒降至 130.2 秒（6.37×），统一 cold 总时长 568.0 秒；单 prompt 无黑帧，但与 raw 的 audio SNR 为 −8.58 dB，尚不能宣称等质。
- `torch.compile` 与 SageAttention、FBCache 组合时出现 latent NaN 和黑帧，暂不推荐。

## 仓库结构

```text
scripts/    推理入口、监控、报告、阶段表和实验脚本
runs/       每次运行的参数、结构化指标、原始日志和 timeline
outputs/    生成的 MP4 等输出
logs/       扫描状态、质量指标和辅助日志
docs/       主报告及归档文档
inputs/     FL2VA 功能验证所用输入
```

模型权重目录 `MiniMax-H3/` 和第三方源码目录 `explore/` 不纳入 Git。

## 准备模型

安装 `modelscope` 后，可只下载约 134 GiB 的 T2VA 组件：

```bash
python3 scripts/download_h3.py
```

默认模型目录是 `/mnt/workspace/MiniMax-H3`。使用其他路径时设置 `H3_MODEL_DIR`，或向 `scripts/run_h3.py` 传入 `--model-dir`。

## 快速开始

以下命令应从仓库根目录执行。`scripts/bench_768p.sh` 固定 1344×768、124 帧、50 步和 seed 0，以保证实验可比。

### raw cold 基线

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  scripts/bench_768p.sh --drop-page-cache --tag raw-cold
```

### SageAttention + FBCache 0.25

```bash
DIFFUSERS_ATTN_BACKEND=sage \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  scripts/bench_768p.sh --drop-page-cache \
  --cache-dit --cache-dit-threshold 0.25 \
  --tag sage-fbcache025-cold
```

### 20 个不同 prompt 的阶段主序批处理

```bash
DIFFUSERS_ATTN_BACKEND=sage \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/run_h3.py \
  --prompt-file scripts/batch_prompts_20.json --batch-limit 20 \
  --steps 50 --cache-dit --cache-dit-threshold 0.25 \
  --tag batch20-sage-fbc025
```

## 运行产物与命名

run ID 由 `scripts/run_naming.py` 自动生成，格式为：

```text
<method>-<YYYYMMDD>-<HHMMSS>
```

例如 `batch20_sage_fbcache025-20260928-114735`。`--drop-page-cache` 是测量条件，不进入名称，是否成功执行记录在 `run.json.page_cache.dropped`。

每次运行通常生成：

```text
runs/<run-id>/cmd.txt
runs/<run-id>/run.json
runs/<run-id>/run.log
runs/<run-id>/metrics.csv
runs/<run-id>/timeline.png
outputs/<run-id>/video.mp4
```

其中 `run.log`、`metrics.csv`、profile trace 和媒体文件属于原始实验凭证，应保持不可变。

## 报告与质量检查

生成多个 run 的阶段耗时表：

```bash
python3 scripts/stage_table.py \
  raw-20260929-103055 \
  fbcache025-20260928-203358 \
  sage-20260928-165335 \
  sage_fbcache025-20260928-204352
```

汇总运行结果：

```bash
python3 scripts/h3_report.py --sort pervideo
```

计算同 prompt/seed 成片的视频 PSNR、SSIM 和 decoded-audio SNR：

```bash
python3 scripts/h3_report.py \
  --quality-reference raw-20260929-103055 \
  --quality-target sage-20260928-165335 \
  --quality-target fbcache025-20260928-203358 \
  --quality-target sage_fbcache025-20260928-204352
```

可通过 `--min-psnr`、`--min-ssim`、`--min-audio-snr` 设置业务阈值；未提供阈值时脚本只报告测量值。

## 已知限制

- 性能与质量数据来自单张 A100-80GB；正式主线覆盖 T2VA 768p/124 帧/50 步，Turbo LoRA 另按 9 个 sigma 点（8 次前向）独立探索。
- FBCache 质量评估目前只覆盖一个主 prompt，阈值选择仍需业务样本集验证。
- FL2VA 仅做过功能验证；Ref2VA、多卡、TF32 和仅 compile 的独立消融未完成。Turbo LoRA 已完成单 prompt 独立探索，但尚未在 SGLang runtime 内复核，也没有多 prompt 等质结论。
- 非 cold run 的加载时间受 page cache 影响；跨配置比较应采用统一 cold 口径。
