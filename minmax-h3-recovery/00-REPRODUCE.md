# minmax-h3 工作复现指南（新 DSW 实例 onboarding）

> 本文件是恢复包入口，写给**新实例上的新 Qoder 会话**：先读这份，再按 §2 顺序加载知识。
> 原实例：阿里云 PAI DSW（8.130.13.7:1024，root，`/mnt/workspace`，A100-80GB 单卡），2026-09-14 后被释放。
> 活跃期：2026-09-01 18:13 ~ 09-14 09:39，共 17 次成功连接（见 `05-connection-log/`）。

---

## 1. 恢复包目录导航

| 目录 | 内容 | 完整度 |
|---|---|---|
| `00-REPRODUCE.md` | 本文件 | — |
| `01-workspace-files/` | 按远程路径还原的 12 个文件（`_INDEX.md` 有 hash 对照） | AI 编辑过的文件全量恢复 |
| `01-workspace-files/_all-snapshots/` | 编辑过程中全部 35 个历史版本 | 完整 |
| `02-chat-history/timeline.md` | 39 条 AI 对话年表（探索决策脉络的唯一本地记录） | 只有 title+时间，正文未留存 |
| `03-editing-sessions/` | 两个编辑会话 state.json 原文（801KB/29KB） | 完整 |
| `04-artifacts-inventory/` | 打开过的 38 个文件清单 + outputs/runs 目录名 | 仅清单，产物本体不可恢复 |
| `05-connection-log/timeline.md` | 23 次连接尝试时间线 | 完整 |

## 2. 知识加载顺序（新会话开工必读）

1. **`01-workspace-files/docs/expert-handoff.md`**（289 行）—— 13 个正式 runs 实验总表、瓶颈转移、10 条陷阱速查、torch.compile / SageAttention / 批处理三轮增量探索。**这是工作现状的权威记录。**
2. **`01-workspace-files/docs/minimax-h3-inference-guide.md`**（591 行）—— Part I 架构原理（Context-IR/Base/Regenerate-2K 三模块、33B Omni-Transformer、Rectified Flow）、Part II 加速实践（§7–13：方法论、baseline 剖析、device_map / FBCache / Turbo LoRA）。已同步钉钉，含环境指纹附录。
3. **`02-chat-history/timeline.md`** —— 用户 39 条原始提问，还原每次方向转折的动机（如「Turbo LoRA 非官方不应是主流方向」「变 prompt 批处理优先级很高」）。
4. 本文件 §4–§7。

## 3. 环境重建清单

| 项 | 值 | 备注 |
|---|---|---|
| 实例 | PAI DSW，**A100-SXM4-80GB 单卡**（HBM 79.32 GiB） | 硬要求：transformer 单组件 62 GiB；FL2VA 半 checkpoint 134 GiB bf16，磁盘与 host RAM 需预留（原机 host 122 GB） |
| 模型 | `/mnt/workspace/MiniMax-H3` | run_h3.py 默认路径，可用 `H3_MODEL_DIR` 覆盖 |
| venv | `venv-h3`（Python 3.12） | torch 2.10.0+cu128、diffusers 0.40.0、CUDA 12.8 / cuDNN 9.10.02 |
| SageAttention | 2.2.0，**源码编译** `pip install . --no-build-isolation` | PyPI 只有 V1.0.6，不满足 diffusers ≥ 2.1.1；激活靠 `export DIFFUSERS_ATTN_BACKEND=sage`，零代码改动 |
| Turbo LoRA | `explore/turbo-lora-A/minimax_h3_turbo_4step.safetensors`（744 MiB） | 用已恢复的 `explore/download_turbo_lora.py` 重下；来源 larryvrh/MiniMax-H3-Turbo-Lora（Apache 2.0，非官方） |
| SSH | 更新 `~/.ssh/config` 的 minmax-h3 段为新公网 IP:端口 | 私钥 `id_ed25519_dsw` 本地未丢，直接复用 |

## 4. 需重建的缺失文件（按依赖顺序）

以下文件在原实例上存在但未进入 AI 编辑快照（未恢复），按此规格重建：

### 4.1 `scripts/h3_monitor.py`（P0 —— 不重建则 run_h3.py 直接 import 失败）

从 `run_h3.py` 反推的接口契约：

```python
from h3_monitor import RunMonitor
mon = RunMonitor(run_dir, run_id, hz=args.monitor_hz).start()   # hz 默认 2.0
# ... 各阶段推进时 run_h3.py 以 phase 名标记（如 "load:denoise#0"、迭代号后缀区分冷/热）
# 结束时产出（runs/<run_id>/ 下）：run.json / run.log / timeline.png / metrics.csv
```

功能规格（依据 inference-guide §0 与 runs/ 产物反推）：
- **2 Hz 采样时序**：显存（HBM allocated/peak）+ host 内存，按阶段分段记录 —— `timeline.png` 即此可视化
- **`env_fingerprint()`**（原文件 83–119 行）：采集 GPU/驱动/torch/diffusers/CUDA 版本写入 `run.json` 的 `env` 段
- **run.json 先写后跑**：raw argv + parsed args 落盘要发生在推理前（首个 768p 视频因 prompt 无记录而丢失的教训，见 run_h3.py:528-535 注释）
- 阶段名由 run_h3.py 侧拼好传入，monitor 只负责分段计时与采样

### 4.2 `scripts/bench_768p.sh`（P1 —— 基准跑批入口）

已知行为：固定 768×1344 / 124 帧 / `--steps 50` 在前、`"$@"` 追加在后（argparse 后者胜）。**两个已知坑必须规避**（见 expert-handoff §4）：tag 是位置参数（`--tag foo` 会让 TAG 变成字面量）；--steps 重复出现靠后者覆盖。

### 4.3 `scripts/h3_report.py`（P2 —— 汇总报告）

已知行为：聚合 runs/*/run.json 生成对比表。已知缺陷：对 `--repeat` 的 run 拿 N 视频总 wall 与 baseline 单视频比，需手动拆迭代（Run 11 教训）。

### 4.4 `docs/2026-09-02-handoff.md`（P2 —— 主交接文档，未恢复）

功能已被 `expert-handoff.md`（增量）+ `inference-guide.md` §13（全量对比表）覆盖大半；「待探索项清单」按本文件 §6 重建即可，无需逐字复原。

## 5. 工作流约定（原实例 9/2 定下的策略，建议延续）

- **目录策略**：`outputs/<run_id>/video-N.mp4`、`runs/<run_id>/{run.json, run.log, timeline.png, metrics.csv}`、脚本进 `scripts/`、过期产物进对应 `bak/` 子目录（不留散文件）。`run_id = YYYYMMDD-HHMMSS`；实验变量写在 `--tag`（one lever per run），不编码进目录名
- **监控策略**：RunMonitor 2Hz 遥测 + 阶段命名带迭代号（`load:denoise#0` 冷盘读 vs `#1` page cache 命中 —— 这就是冷/热分离的测量手段）；kernel 级归因用 `--profile`（torch profiler，仅限 denoise 阶段）
- **验收纪律**：**PSNR 不可做自动门禁**（12–17 dB 量级与人眼不对应，14.49 的 E 组反而胜出）—— 每轮有效配置产出视频表，人工验片定质量
- **多会话接力**：用户的工作模式是 handoff 文档驱动（「把最近的成果回写到 handoff，我开新 branch 继续」）—— 新探索前先更新 expert-handoff.md，保持它的权威性
- **报告纪律**：对照口径固定 —— 后续加速改动 vs **交付基线 188s**（FBCache+reuse 热态），不再 vs 原始 baseline 1212.5s；2.55× 的口径是 vs 含 device_map 的平摊单跑 479s

## 6. 加速探索现状（从哪里继续）

**基线演化**：1212.5s 原始 → 1151.6s device_map → 457.7s FBCache t=0.25 → **188s 交付基线**（+reuse-text-embeds, repeat≥2 热态）→ **178s 推荐新基线**（+SageAttention，零代码改动）→ 167s 三重叠加（Sage+compile+FBCache，因 graph break 暂不默认）。

**推荐命令模板**（expert-handoff §9.4 验证过的写法）：

```bash
# 单视频（推荐新基线）
DIFFUSERS_ATTN_BACKEND=sage python scripts/run_h3.py \
    --model-dir MiniMax-H3 --prompt "..." \
    --height 768 --width 1344 --num-frames 124 --steps 50 \
    --cache-dit --cache-dit-threshold 0.25 \
    --reuse-text-embeds --repeat 2 --tag <变量名>

# 变 prompt 批处理（阶段×视频：组件各 load 1 次逐视频复用）
DIFFUSERS_ATTN_BACKEND=sage python scripts/run_h3.py \
    --model-dir MiniMax-H3 --prompt-file prompts.json \
    --height 768 --width 1344 --num-frames 124 \
    --steps 50 --cache-dit --cache-dit-threshold 0.25 --tag my-batch-run
# 注意：--prompt-file 与 --repeat/--reuse-text-embeds/--image 互斥
```

**已排除项**（不要重试）：TF32（bf16 主体无效）、fullgraph=True（+1.3% 噪声）、Turbo LoRA 与 FBCache 叠加（无效）、compile 单独使用（-6.3%，被 Sage 的 -8.5% 覆盖且预热 70s）。

**下一步候选**（按原探索脉络的优先级）：
1. **变 prompt 批处理深化** —— 刚跑通 batch×3（冷启动省 20.5%），热启动收益、更多 prompt 数量级、排队服务化（用户 9/8 20:50 提的业务形态）未探
2. **SGLang 4 卡路线** —— 官方标注 1.85–1.95×（手写融合 kernel），与单卡路线正交；用户多次提起（9/2、9/4、9/8、9/11），单卡天花板已到顶
3. 三重叠加的 graph break 稳定性（低优先级）

## 7. 陷阱速查（完整 10 条见 expert-handoff.md §4）

| 陷阱 | 一句话 |
|---|---|
| FBCache hooks 循环引用 | 生成后不 `remove_hook(recurse=True)` 则 62 GiB transformer 在 decode 阶段驻留 → OOM |
| kwargs_type 必须随 embeds 回放 | 漏掉则静默失去文本条件，无报错但画面全错 |
| `_STICKY_COMPONENTS` | image_processor/processor/video_processor/tokenizer 不能 free，否则 repeat 第二轮崩 |
| FBCache 阈值非单调 | t=0.10 的 PSNR 低于 t=0.15，不是越小越保真 |
| PSNR 不可自动门禁 | 人工验片，见 §5 |
| page cache 命中率 | safetensors 走 mmap，rchar 恒为 0，用「组件体积 − read_gib」推算 |

## 8. 第一天 checklist

- [ ] 新 DSW 实例启动（A100-80GB），更新本地 `~/.ssh/config` 的 minmax-h3 段
- [ ] 上传本恢复包（建议整包 tar 后 scp），`01-workspace-files/` 内容按原路径铺回 `/mnt/workspace`
- [ ] 重建 `venv-h3`（torch 2.10.0+cu128 / diffusers 0.40.0 / SageAttention 2.2.0 源码编译）
- [ ] 按 §4.1 规格**先重建 `scripts/h3_monitor.py`**（run_h3.py 的硬依赖）
- [ ] 模型权重就位（OSS 备份或重新下载；FL2VA 半 134 GiB）
- [ ] 冒烟：`--steps 2` 单视频 + `--monitor-hz 2` 跑通，检查 runs/<id>/ 四件套齐全
- [ ] 复现交付基线：Sage + FBCache + reuse，repeat=2 热态 ≈178s（±10% 视新机器状态）
- [ ] 人工验片一条，确认画质在可接受线
- [ ] 之后按 §6 候选清单继续，探索成果持续回写 `docs/expert-handoff.md`
