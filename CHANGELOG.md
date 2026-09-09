# Changelog

All notable changes to MTP-Align are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.2.0] - 2026-09-10

### Fixed

- **Proxy: tool calls now actually flush at MTP decode-window boundaries.**
  The proxy never registered tool-call deltas with the scheduler, so the
  boundary flush could never fire and tool-call chunks were withheld until
  end-of-stream and emitted *after* `data: [DONE]` — where conforming OpenAI
  clients stop parsing and never see them. Flush decisions were also dropped
  for content-only lines crossing a boundary. Buffered chunks are now queued
  with the scheduler, released at the next window boundary, and drained
  before `[DONE]` when the stream ends first.
- **Proxy: the re-framed SSE stream is spec-conformant.** Every `data:` event
  is now terminated by a blank line as the SSE spec requires; the previous
  single-newline framing broke strict parsers such as the OpenAI SDK, which
  dispatch events only on a blank line.
- **Proxy: per-request state and live metrics.** Scheduler window state now
  resets per request (each completion is an independent decode stream, so
  boundary alignment no longer drifts between requests), and the metrics
  collector is driven by decoded tokens so live stderr metrics log during
  streaming.
- **CLI: invalid arguments produce clean errors.** `bench --tokens 128
  --warmup 128`, `serve --flush bogus`, and `serve --window 0` now print a
  one-line error and exit with code 2 instead of dumping a raw `ValueError`
  traceback. The missing-dependency hint no longer points at the nonexistent
  `mtp-align[serve]` extra.

### Changed

- Version surfaces bumped in lockstep to 0.2.0: `VERSION`, `pyproject.toml`,
  `mtp_align.__version__`, the FastAPI app version, and the `X-MTP-Align`
  response header (the latter two now derive from `__version__` instead of
  hardcoded literals), plus `web/site.json` `content_version` to `v0.2.0`.

### Added

- `CHANGELOG.md` (this file).
- Test suites for the proxy streaming path (`tests/test_proxy_stream.py`),
  CLI error handling (`tests/test_cli_errors.py`), and version lockstep
  (`tests/test_version.py`).

## [0.1.0] - 2026-08-09

Initial release.

- `mtp-align bench` — before/after decode-tps benchmark (deterministic
  synthetic mock plus a live-upstream engine) with the 5% kill-gate.
- `mtp-align serve` — OpenAI-compatible streaming proxy with the MTP
  decode-window scheduler (`MTPWindow`, `MTPScheduler`).
- Live metrics collector (`tokens / tps / continuity`), bilingual README
  (zh primary + English sibling), 60s demo recording.
