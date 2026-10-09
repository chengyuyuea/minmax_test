# MiniMax-H3 单卡推理加速实验

本仓库记录 MiniMax-H3 在单张 NVIDIA A100-SXM4-80GB 上的 T2VA 推理加速实验。所有正式输出均为 **1344×768、124 帧、24 fps、seed 0**。标准 diffusers 与 SGLang 使用 **50 个 sigma 点（49 次 DiT 前向）**；diffusers Turbo LoRA 使用 **9 个 sigma 点（8 次前向）**，单独标注。加速手段分为 **diffusers 原生加速**和 **SGLang 加速**两类。

完整的架构分析、实验方法和质量结果见 [MiniMax-H3 推理指南](docs/minimax-h3-inference-guide.md)。

## 主要结论

统一环境：Python 3.12、PyTorch 2.13.0+cu130、CUDA 13.0、diffusers 0.40.0、SageAttention 2.2.0。

### 单视频 cold latency

所有行均为 1344×768、124 帧、24 fps；batch 吞吐不混入本表。

| 框架 | 配置 | sigma 点 / DiT 前向 | denoise | cold 总时长 | 质量说明 |
|---|---|---:|---:|---:|---|
| diffusers | raw SDPA | 50 / 49 | 829.9 s | 1269.0 s | 参考 |
| diffusers | SageAttention | 50 / 49 | 749.9 s | 1186.7 s | 轻微数值漂移 |
| diffusers | FBCache 0.25 | 50 / 约 8 次完整前向 | 151.5 s | 581.7 s | 有损 |
| diffusers | Sage + FBCache 0.25 | 50 / 约 8 次完整前向 | 136.9 s | 569.5 s | 有损 |
| diffusers | Turbo LoRA v4 | 9 / 8 | 130.2 s | 568.0 s | 与 50-sigma raw 不同采样口径 |
| diffusers | Sage + Turbo LoRA v4 | 9 / 8 | 118.2 s | 557.7 s | 与 50-sigma raw 不同采样口径 |
| SGLang | 流式 offload raw | 50 / 49 | 4824.6 s | 5147.7 s | SGLang 参考 |
| SGLang | 52/52 block 常驻 raw | 50 / 49 | 833.3 s | 1148.8 s | 与 SGLang raw 逐字节一致 |
| SGLang | 常驻 + Cache-DiT 0.24 | 50 / 49 调度 step | 335.7 s | 651.5 s | 有损 |

### 吞吐结果

阶段主序 batch=20 使用 diffusers Sage+FBCache 0.25，统一 cold 口径为 **191.1 s/视频**。这是 20 个不同 prompt 共用组件加载的吞吐结果，不是单请求延迟，也不是 tensor batch。

### 结论

- diffusers：Sage 缩短单次前向；FBCache 减少完整前向；Turbo LoRA 使用 9-sigma/8-forward；阶段主序批处理摊薄加载。
- SGLang：52/52 block 常驻消除逐 step 权重读取；Cache-DiT 进一步加速但有损。
- 两类框架封装不同，加速实现不能直接互换。具体边界见推理指南 §9.3。
- `torch.compile` 组合测试出现 2/5 黑帧，不采用。

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

### diffusers Turbo LoRA v4（8 次前向）

Turbo LoRA 必须把指定 adapter、9 点 schedule 和 scale 1.0 配套使用；当前正式结果采用 BF16 预融合路径：

```bash
python3 scripts/run_h3.py --steps 9 --seed 0 \
  --lora explore/turbo-lora-larry/minimax_h3_turbo_v4_step600_ema.safetensors \
  --lora-scale 1.0 \
  --run-id "turbolora_v4_8eval-$(date +%Y%m%d-%H%M%S)" \
  --runs-dir runs --outputs-dir outputs --tag turbo-lora-v4-8eval
```

### SageAttention + Turbo LoRA v4

```bash
DIFFUSERS_ATTN_BACKEND=sage \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/run_h3.py --steps 9 --seed 0 \
  --lora explore/turbo-lora-larry/minimax_h3_turbo_v4_step600_ema.safetensors \
  --lora-scale 1.0 --drop-page-cache --tag sage-turbo-v4-8eval
```

### SGLang 全 DiT 常驻

SGLang 命令固定为 1344×768、124 帧、24 fps、50 个 sigma 点（49 次 DiT 前向）、seed 0；完整结果见指南 §15.3。

```bash
python3 scripts/run_sglang_h3.py --drop-page-cache \
  --tag sglang_resident50 --dit-resident-layers 50

python3 scripts/run_sglang_h3.py --drop-page-cache \
  --tag sglang_resident50_cachedit024 --dit-resident-layers 50 \
  --cache-dit --cache-dit-threshold 0.24
```

## 运行产物与命名

run ID 由 `scripts/run_naming.py` 自动生成，格式为：

```text
<method>-<YYYYMMDD>-<HHMMSS>
```

例如 `batch20_sage_fbcache025-20260928-114735`。两类框架通过名称区分：diffusers 使用方法名（Turbo 为 `turbolora_v4_8eval-*`），SGLang 使用 `sglang_*`；两套缓存分别命名为 diffusers `fbcache` 和 SGLang `cachedit`。`--drop-page-cache` 是测量条件，不进入名称，是否成功执行记录在 `run.json.page_cache.dropped`。

每次运行通常生成：

```text
runs/<run-id>/cmd.txt
runs/<run-id>/run.json
runs/<run-id>/run.log
runs/<run-id>/metrics.csv
runs/<run-id>/timeline.png
outputs/<run-id>/video.mp4
```

其中 `run.log`、`metrics.csv`、profile trace 和媒体文件属于原始实验凭证，应保持不可变。`runs/` 与 `outputs/` 根目录按 diffusers、SGLang 两类框架保留可交付主线并保持同名配对；smoke、短程探针、失败和旧环境结果移入各自的 `bak/`。

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

- 性能与质量数据来自单张 A100-80GB；正式主线均覆盖 T2VA 768p/124 帧，其中标准 diffusers 与 SGLang 为 50 个 sigma 点，Turbo LoRA 独立主线为 9 个 sigma 点（8 次前向）。
- FBCache/Cache-DiT 质量评估目前只覆盖一个主 prompt，阈值选择仍需业务样本集验证。
- FL2VA 仅做过功能验证；Ref2VA、多卡、TF32 和仅 compile 的独立消融未完成。Turbo LoRA 尚未在 SGLang runtime 内复核，SGLang warm-server 多请求吞吐也未测试。
- diffusers 使用统一 cold-load 口径，SGLang 使用实测 wrapper cold wall；跨 backend 可比较端到端耗时，不应强行对齐内部阶段。
