"""In-memory registry for background ingestion tasks.

Ingestion is CPU-heavy (reading files, chunking, embedding), so the API kicks
it off as a background task and returns immediately. This module tracks each
task's lifecycle:

    pending -> processing -> completed | failed

The registry is guarded by a lock: the FastAPI endpoints read it from the
event loop thread while the worker updates it from a threadpool thread. It
also enforces that only one ingestion runs at a time, since ingestion rewrites
the shared index files on disk.
"""

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

_ACTIVE_STATUSES = {"pending", "processing"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class IngestionTask:
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "pending"  # pending | processing | completed | failed
    detail: str | None = None  # error message when failed
    chunks: int | None = None  # chunk count when completed
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class TaskRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[str, IngestionTask] = {}

    def create(self) -> IngestionTask:
        """Register a new pending task, or raise if one is already active."""
        with self._lock:
            active = self._active_locked()
            if active is not None:
                raise IngestionInProgressError(active.task_id)
            task = IngestionTask()
            self._tasks[task.task_id] = task
            return task

    def get(self, task_id: str) -> IngestionTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def mark_processing(self, task_id: str) -> None:
        with self._lock:
            self._tasks[task_id].status = "processing"

    def mark_completed(self, task_id: str, chunks: int) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "completed"
            task.chunks = chunks
            task.finished_at = _now()

    def mark_failed(self, task_id: str, detail: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "failed"
            task.detail = detail
            task.finished_at = _now()

    def _active_locked(self) -> IngestionTask | None:
        return next(
            (t for t in self._tasks.values() if t.status in _ACTIVE_STATUSES), None
        )


class IngestionInProgressError(RuntimeError):
    def __init__(self, task_id: str):
        super().__init__(f"An ingestion is already running (task {task_id}).")
        self.task_id = task_id
