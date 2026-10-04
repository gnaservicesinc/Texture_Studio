"""Exercise the packaged subapps' real project registration and exit path."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    app_root = repo / "build/IPDE Studio.app/Contents/Applications"
    with tempfile.TemporaryDirectory(prefix="ipde-session-smoke-") as folder:
        project = Path(folder).resolve()
        (project / "project.ini").write_text("[General]\nname=Session smoke\ngoal=manual\nschema=ipde-project-v1\n")
        digest = hashlib.sha256(str(project).encode()).hexdigest()[:32]
        address = str(Path(tempfile.gettempdir()) / ("ipde-" + digest))
        roles = [("IPDE", "extractor"), ("Dataset Studio", "datasets"), ("RAFT Studio", "trainer"),
                 ("Photo Studio", "photo"), ("Raw Studio", "raw")]
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(address); listener.listen(5); listener.settimeout(20)
        registered: list[str] = []
        errors: list[Exception] = []

        def serve() -> None:
            try:
                for _, expected in roles:
                    connection, _ = listener.accept()
                    with connection:
                        stream = connection.makefile("rb")
                        message = json.loads(stream.readline())
                        if message.get("role") != expected or message.get("token") != "smoke-session-token":
                            raise RuntimeError("Wrong packaged app/session identity")
                        registered.append(expected)
                        # Registration plus change event in one packet also
                        # exercises real startup event buffering.
                        connection.sendall(b'{"accepted":true}\n{"event":"project_changed"}\n')
                        while stream.read(1):
                            pass
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=serve, daemon=True); worker.start()
        try:
            environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", IPDE_STUDIO_TOKEN="smoke-session-token")
            for name, role in roles:
                executable = app_root / f"{name}.app/Contents/MacOS/{name}"
                result = subprocess.run([str(executable), "--project", str(project), "--mode", role, "--smoke-test"],
                    env=environment, text=True, capture_output=True, timeout=20)
                if result.returncode:
                    raise RuntimeError(f"{name} exited {result.returncode}: {result.stderr}")
                if list((project / ".studio").glob(role + ".lock")):
                    raise RuntimeError(f"{name} left a session lock behind")
            worker.join(5)
            if worker.is_alive() or errors or registered != [role for _, role in roles]:
                raise RuntimeError(f"Hub registration failed: {registered}, {errors}")
            print("Packaged extractor, datasets, trainer, photo and RAW: project registration and clean exit passed")
        finally:
            listener.close()
            Path(address).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
