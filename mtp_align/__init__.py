"""MTP-Align — DeepSeek-V4 MTP-aware agent-loop scheduler.

MTP-Align is a drop-in proxy that batches agent tool calls to DeepSeek-V4's
multi-token-prediction (MTP) decode windows instead of interrupting the decode
stream mid-window. The core primitive is the :class:`~mtp_align.scheduler.MTPWindow`,
a scheduler-level model of the model's MTP batch.

v0.1 ships the benchmark harness (``mtp-align bench``) that measures the
decode-tps delta between naive (tool-call-interrupted) and MTP-aligned decode.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
