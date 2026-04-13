"""
base.py — Shared base class for all pipeline steps.

Every step:
  - Has a name and a verbose flag.
  - Receives a shared `context` dict and returns an updated copy of it.
  - Logs every step with a timestamp.
  - Reports its status (idle → running → done / error).
"""

from datetime import datetime


class PipelineStep:
    """Base class every pipeline step inherits from."""

    def __init__(self, name: str, verbose: bool = True):
        self.name    = name
        self.verbose = verbose
        self.status  = "idle"
        self._logs: list[str] = []

    # ── Logging ────────────────────────────────────────────────────────────────

    def log(self, message: str, level: str = "INFO") -> None:
        ts    = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] [{self.name}] [{level}] {message}"
        self._logs.append(entry)
        if self.verbose:
            print(entry)

    def warn(self, message: str) -> None:
        self.log(message, level="WARN")

    def error(self, message: str) -> None:
        self.log(message, level="ERROR")

    def get_logs(self) -> list[str]:
        return list(self._logs)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        """
        Override in every subclass.
        Takes the shared context dict, does work, returns the updated context.
        Raise an exception on unrecoverable errors; the orchestrator will catch it.
        """
        raise NotImplementedError(f"{self.name}.run() is not implemented.")

    def _start(self) -> None:
        self.status = "running"
        self.log(f"Starting ...")

    def _finish(self) -> None:
        self.status = "done"
        self.log(f"Done.")

    def _fail(self, exc: Exception) -> None:
        self.status = "error"
        self.error(f"Failed: {exc}")
