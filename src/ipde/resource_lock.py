"""Process locks outside immutable datasets, shared across project apps."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile


@contextmanager
def resource_lock(path: Path | str, *, shared: bool = False):
    """Fail promptly if cleanup conflicts with a reader or another writer.

    POSIX supports concurrent readers. Windows uses an exclusive byte lock for
    both modes, trading concurrency for the same preservation guarantee.
    The OS releases locks when a cancelled/killed task exits.
    """
    resource = Path(path).expanduser().resolve()
    digest = hashlib.sha256(os.fsencode(resource)).hexdigest()
    directory = Path(tempfile.gettempdir()) / f"ipde-resource-locks-{getattr(os, 'getuid', lambda: 'user')()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / digest).open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("This dataset or run is in use by another task; wait for it to finish.") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("This dataset or run is in use by another task; wait for it to finish.") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
