"""Serialize capacity checks and submissions from interactive and watcher processes."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def campaign_submission_lock(root):
    path=Path(root)/'results/train04dev/submission.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
