<div align="right"><sub>[English](./README.en.md)&nbsp;&nbsp;⇄&nbsp;&nbsp;<b>简体中文</b></sub></div>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg">
    <img src="./assets/hero-light.svg" width="880" alt="MTP-Align — 把 agent 工具调用批对齐到 DeepSeek-V4 的 MTP 解码窗口">
  </picture>
</p>

<p align="center"><sub>把 agent 的工具调用批对齐到 DeepSeek-V4 的 MTP 解码窗口，让本地 tps 重新跑满。</sub></p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/github/license/SuperMarioYL/mtp-align?style=flat&color=blue" alt="license"></a>
  <a href="https://github.com/SuperMarioYL/mtp-align/releases"><img src="https://img.shields.io/github/v/release/SuperMarioYL/mtp-align?style=flat&color=blue&label=release" alt="release"></a>
  <img src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/mtp-align/ci.yml?branch=main&label=CI&style=flat" alt="CI">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat&logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/MTP--aware-window%20scheduler-5E5CE6?style=flat" alt="MTP-aware">
  <img src="https://img.shields.io/badge/DeepSeek--V4-10A37F?style=flat" alt="DeepSeek-V4">
</p>

> **DeepSeek-V4 的 MTP 多 token 预测能在一次前向传播里解码多个 token——但 agent 每次工具调用都打断解码流，MTP 增益归零。MTP-Align 把工具调用批对齐到 MTP 解码窗口边界，让本地 tps 重新跑满。**

---

## 目录

- [架构](#架构)
- [为什么需要 MTP-Align](#为什么需要-mtp-align)
- [安装](#安装)
- [快速开始](#快速开始)
- [用法](#用法)
- [Demo](#demo)
- [配置](#配置)
- [付费与商业版](#付费与商业版)
- [路线图](#路线图)
- [License](#license)

<h2><img src="https://api.iconify.design/tabler:topology-star-3.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 架构</h2>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/atlas-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/atlas-light.svg">
    <img src="./assets/atlas-light.svg" width="880" alt="架构：Agent Framework → MTP-Align Proxy → llama.cpp (DeepSeek-V4 + MTP)">
  </picture>
</p>

三个组件，我们交付中间那个。没有微服务、容器或 Kubernetes——代理是一个跑在 uvicorn 后面的单 ASGI 进程。

- **Agent Framework**（OpenCode / Hermes）通过 OpenAI 兼容的 `/v1/chat/completions` 把请求打到代理。
- **MTP-Align Proxy**（`:8090`）拦截流式响应，用 `MTPScheduler` 跟踪当前 MTP 解码窗口，把工具调用触发缓冲到窗口边界再 flush。
- **llama.cpp server**（`:8080`）跑 DeepSeek-V4 + MTP/DSpark，提供真实解码。

核心原语是 **MTP 解码窗口**——一个调度器层面对 DeepSeek-V4 多 token 预测批的建模：一次前向传播吐出 `window_size` 个 token（DSV4 为 4）。如果 agent 在窗口中途打断流去触发工具调用，窗口里剩下的 token 就被浪费。调度器跟踪窗口边界，把工具调用 flush 推迟到边界，让窗口跑满。这是 DeepSeek 特有的解码结构——GPT/Claude 单 token 流根本没有窗口可对齐。

## 为什么需要 MTP-Align

DeepSeek-V4 的 MTP 让单次前向传播吐多个 token，**但只在解码流连续时成立**。Agent 的工具调用循环恰恰打破这种连续性：每次工具调用都在窗口中途打断流，把多 token 吞吐压回单 token 解码。今天跑 DSV4F 0731 + Hermes / OpenCode 的用户拿到的是"干活主力"的吞吐，却在每个工具调用边界默默丢掉 MTP 增益，而且没有任何现有工具测量或回收这个损失。MTP-Align 就是那个调度器——把工具调用对齐到解码窗口边界，把没被用上的模型原语变成实在的本地提速。

> ⚠️ 说明：MTP 增益被打断这个核心论点目前**未经实测**——现有证据只有 r/LocalLLaMA 的两条帖子。`mtp-align bench` 就是用来证伪的：合成模型上界显示 +9.2% delta（见下），但真正的 kill-gate 是 `mtp-align bench --upstream <你的 llama.cpp>`。若 delta < 5%，项目到此为止。

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 安装</h2>

```bash
pip install mtp-align        # 或：uv tool install mtp-align
```

需要 Python ≥ 3.12。代理另需 `fastapi` / `uvicorn`，已包含在依赖里。

<h2><img src="https://api.iconify.design/tabler:bolt.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 快速开始</h2>

三条命令，从冷克隆到看到结果：

```bash
mtp-align bench                                   # 合成基准，无需 GPU，~1 秒
mtp-align serve --upstream http://localhost:8080  # 启动 MTP 感知代理 :8090
OPENAI_BASE_URL=http://localhost:8090/v1 your-agent  # agent 指向代理即可
```

<details><summary>合成基准输出（示例）</summary>

```
       MTP-Align benchmark — naive vs. MTP-aligned decode       
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ mode    ┃ tokens ┃ wall (s) ┃   tps ┃ continuity % ┃ delta % ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╩
│ naive   │    512 │   11.344 │ 45.13 │         91.1 │       — │
│ aligned │    512 │   10.384 │ 49.31 │        100.0 │    +9.2 │
└─────────┴────────┴──────────┴───────┴──────────────┴─────────┘
✓ kill-gate PASSED: tps delta +9.2% (gate ≥ 5%)
```

> 上述为**合成模型上界**（每个工具调用按浪费整窗口计），不是 GPU 实测。真实 kill-gate 用 `mtp-align bench --upstream <url>`。
</details>

<h2><img src="https://api.iconify.design/tabler:terminal-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 用法</h2>

五个最常用的工作流：

```bash
# 1. 合成基准（CI / 无 GPU 环境，永远可跑）
mtp-align bench

# 2. 真实 kill-gate：打到你跑着的 llama.cpp（DSV4 + MTP/DSpark 已开）
mtp-align bench --upstream http://localhost:8080 --tokens 512 --tool-calls 12

# 3. 启动 MTP 感知代理（m2 里程碑）
mtp-align serve --upstream http://localhost:8080 --port 8090 --window 4

# 4. 让 agent 走代理（改一个环境变量，不改 agent 代码）
OPENAI_BASE_URL=http://localhost:8090/v1 opencode chat

# 5. 编程式调用：见 examples/benchmark.py
python examples/benchmark.py
```

CLI 全部选项见 `mtp-align --help`。

<h2><img src="https://api.iconify.design/tabler:photo.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Demo</h2>

![demo](assets/demo.gif)

10 分钟快乐路径：安装 → 合成基准 → 读 tps delta。重渲染用 `docs/demo.tape`（vhs）。

<h2><img src="https://api.iconify.design/tabler:adjustments.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 配置</h2>

`MTPConfig` 的关键字段（`mtp_align/config.py`）：

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `upstream` | str | `http://localhost:8080` | 上游 llama.cpp 基址 |
| `listen_port` | int | `8090` | 代理监听端口 |
| `window_size` | int | `4` | 每次 MTP 前向传播的 token 数（DSV4=4） |
| `flush_policy` | str | `window_boundary` | flush 策略：`window_boundary` 或 `eager` |
| `max_batch` | int | `8` | 触发提前 flush 的批次上限 |
| `benchmark_tokens` | int | `512` | 基准工作负载的总 token 数 |
| `benchmark_tool_calls` | int | `12` | 交错工具调用次数 |
| `window_latency_s` | float | `0.08` | 合成模型：一次前向传播墙钟（仅 mock 用） |
| `tool_overhead_s` | float | `0.012` | 合成模型：单次工具调用往返开销（仅 mock 用） |

<h2><img src="https://api.iconify.design/tabler:cash.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 付费与商业版</h2>

v0.1 的代理与基准 **完全开源、免费**（MIT）。商业路线针对在自持 GPU 上跑 DeepSeek-V4、且不能用云 API 的 **信创 / air-gapped 推理集群**——把 OSS 代理证明的 tps 论点，变现为舰队级遥测与 SLA。

**商业版交付（on-prem 可部署，Helm / systemd，air-gapped 友好）：**

- 舰队级 tps 仪表盘：跨节点聚合 window-continuity %
- 按团队 / 按集群的连续性 % 与 SLA 看板
- tps 回归告警（超过阈值即报警）
- 优先调度器修复与支持

**定价（按集群，非按席位）**：¥30k–80k / 集群 / 年，按 GPU 数分档。低于 ¥30k 覆盖不了 on-prem 销售开销；高于 ¥80k OSS 代理 + 两天内部脚本即可替代。前 3 个设计伙伴集群享首发折扣。

**最小成单路径**：① 在一个节点跑免费 OSS 代理，看到 bench delta → ② 30 分钟远程 demo：2 节点集群装遥测 bundle，看舰队仪表盘聚合 continuity + SLA 告警 → ③ 设计伙伴协议，¥30k PO，90 天 SLA 试点。结算走企业微信 / 对公转账 + 发票（fapiao）流，不要 Stripe。

v0.1 不做：云托管（产品就是因为解码在本地）、多用户鉴权、Web UI。这些是 v2 路线。

<h2><img src="https://api.iconify.design/tabler:map-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> 路线图</h2>

- [x] **m1 — 测量 tps delta**：`mtp-align bench` 端到端跑通，打印可证伪的 delta 表（kill-gate < 5% 即停）
- [ ] **m2 — 构建 MTP 代理**：FastAPI 代理拦截 `/v1/chat/completions`，跟踪 MTP 窗口，工具调用在边界 flush；point OpenCode at the proxy 看真实 tps 提升
- [ ] **m3 — 集成与发布**：OpenCode/Hermes 集成、60 秒 before/after-tps demo、双语 README、r/LocalLLaMA + Show HN 发布
- [ ] 未来：Qwen-MTP / Kimi-MTP 等非 DeepSeek MTP 模型；舰队遥测商业版

<h2><img src="https://api.iconify.design/tabler:license.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> License</h2>

MIT，见 [LICENSE](./LICENSE)。欢迎在 [Issues](https://github.com/SuperMarioYL/mtp-align/issues) 报 bug 或在 [Discussions](https://github.com/SuperMarioYL/mtp-align/discussions) 提建议。如果要把 MTP-Align 作为你 agent 的后端选项，欢迎开 PR。

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
