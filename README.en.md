[简体中文](./README.md) · [Website](https://mtp-align.lei6393.com) · [GitHub](https://github.com/SuperMarioYL/mtp-align)

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/hero-dark.svg">
  <img src="./assets/presentation/hero-light.svg" width="960" alt="Hero diagram">
</picture>

# mtp-align

**Explore tool-call buffering at explicit window boundaries.**

MTP-Align provides a small scheduling state machine that queues tool calls and flushes them at configured token-count boundaries.

## Why use it

A buffering policy is easier to evaluate when its state and flush decisions are explicit. This project separates that deterministic mechanism from the inference performance hypothesis it is intended to explore.

- **Explicit state transitions** — Window progress and pending calls can be inspected.
- **Deterministic experiment** — The scheduler runs without an inference server.
- **Configurable timing policy** — Compare boundary and eager flush behavior.

## Architecture

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-dark.svg">
  <img src="./assets/presentation/architecture-light.svg" width="960" alt="Architecture diagram">
</picture>

MTPScheduler maintains a configured window and pending ToolCallBatch. on_token advances the counter and returns WAIT or FLUSH. The proxy connects this mechanism to a streamed HTTP path; the benchmark also drives it with a synthetic timing model.

| Component | Responsibility |
| --- | --- |
| `Token events` | Explicit scheduler input |
| `MTP window` | mtp_align/scheduler.py |
| `Queued tool batch` | Pending calls and horizon |
| `Flush decision` | WAIT or FLUSH |

## Install and quickstart

Build with the version declared in the repository manifest. Run the example from the repository root.

```bash
git clone https://github.com/SuperMarioYL/mtp-align.git
cd mtp-align
uv venv .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

Queue one lookup call and feed four explicit token events to a window-size-four scheduler.

```bash
.venv/bin/python examples/presentation-demo.py
```

## Recorded demo

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/process-dark.svg">
  <img src="./assets/presentation/process-light.svg" width="960" alt="Process diagram">
</picture>

The tool call remains buffered for three events and flushes on the fourth.

```text
token 0: wait
token 1: wait
token 2: wait
token 3: flush
flushed calls: ['lookup']
windows completed: 1
```

The complete command and output are recorded in [docs/demo-results.json](./docs/demo-results.json). Inputs and reproduction code are included in the repository.

![Existing terminal recording](./assets/demo.gif)

The existing recording is retained for context; the text example above documents the reproducible scenario.

## Usage

The CLI exposes the following operations. Commands after the example use your own paths or identifiers.

```bash
mtp-align bench
# Experimental proxy, with your own compatible upstream:
mtp-align serve --upstream http://localhost:8080 --port 8090 --window 4
```

## Configuration

MTPConfig defines upstream, listen_port, window_size, flush_policy and max_batch. window_latency_s and tool_overhead_s are synthetic model assumptions. Choose window_boundary or eager to compare scheduling behavior; neither config alone proves a server optimization.

## Integrations and responsibilities

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-dark.svg">
  <img src="./assets/presentation/integrations-light.svg" width="960" alt="Integrations diagram">
</picture>

The following routes are implemented in the source. Choose the input that matches your task and keep the resulting artifact with your project.

| Route | Implemented role |
| --- | --- |
| Python state machine | Direct scheduling API |
| SSE proxy scaffold | Chat-completions forwarding |
| Config | Window, batch and flush policy |
| Synthetic benchmark | Timing-model exploration |

## Limits and next steps

- An SSE content chunk is not necessarily one model token, and token counts do not establish actual MTP forward-pass boundaries.
- No GPU throughput benefit is established. The synthetic benchmark uses assumed timing and interruption costs.
- The proxy is experimental; request isolation, partial tool-call reconstruction and production backpressure require further validation.

The next requirement is matching scheduler events to observed server decode boundaries and measuring end-to-end behavior with real workloads.

## License and contributions

See [LICENSE](./LICENSE). When reporting an issue, include a minimal input, the command, and the observed output.
