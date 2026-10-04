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
import sys


def backend_smoke(app_root: Path, project: Path) -> None:
    """Exercise embedded Python modules, rather than just Qt startup/session IPC."""
    from test_collection_storage import _dataset
    workspace = project / "workspace"
    owner = workspace / "datasets"
    owner.mkdir(parents=True)
    fixture, _ = _dataset(project)
    source = owner / "source"
    fixture.rename(source)
    prepared = owner / "prepared"

    def invoke(name: str, arguments: list[str]) -> dict:
        script = app_root / f"{name}.app/Contents/Resources/raft_studio.py"
        # -I prevents PYTHONPATH from hiding omitted bundle modules. The bundle
        # entrypoint explicitly selects its own Resources/src package.
        result = subprocess.run([sys.executable, "-I", str(script), "--json", *arguments],
                                cwd=project, text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f"{name} embedded backend failed: {result.stdout} {result.stderr}")
        return json.loads(result.stdout)

    invoke("Dataset Studio", ["inspect-dataset", str(source), "--workers", "2"])
    invoke("Dataset Studio", ["compose-datasets", str(source), "--output-dir", str(prepared), "--workers", "2"])
    edited = owner / "edited"
    edits_file = project / "edits.json"
    source_samples = json.loads((source / "dataset.json").read_text())["samples"]
    edits_file.write_text(json.dumps({"keep": [s["id"] for s in source_samples[1:]],
                                      "splits": {source_samples[1]["id"]: "validation"}}))
    invoke("Dataset Studio", ["edit-dataset", str(source), "--edits-json", str(edits_file),
                              "--output-dir", str(edited), "--workers", "2"])
    review = invoke("Dataset Studio", ["review-dataset", str(edited)])
    if len(review["samples"]) != 2 or not all(s.get("management_group_id") for s in review["samples"]):
        raise RuntimeError("Embedded editor/review omitted membership edits or split component IDs")
    payloads = {file: file.read_bytes() for file in source.rglob("*")
                if file.is_file() and file.suffix in {".npy", ".npz"}}
    source_hash = hashlib.sha256((source / "dataset.json").read_bytes()).hexdigest()
    edits_file.write_text(json.dumps({"keep": [s["id"] for s in source_samples[1:]],
                                      "validation_fraction": 1.0, "seed": 42}))
    update_arguments = ["update-dataset", str(source), "--edits-json", str(edits_file),
                        "--expected-manifest-sha256", source_hash, "--operation-id", "packaged-smoke-remove"]
    updated = invoke("Dataset Studio", update_arguments)
    replayed = invoke("Dataset Studio", update_arguments)
    if (not replayed["already_applied"] or replayed["manifest_sha256"] != updated["manifest_sha256"]
            or replayed["edit_revision"] != updated["edit_revision"]):
        raise RuntimeError("Embedded editor failed to acknowledge an already committed save without rewriting it")
    restored_review = invoke("Dataset Studio", ["review-dataset", str(source)])
    if (Path(updated["dataset_path"]) != source or updated["summary"]["train_samples"] != 0
            or updated["training_eligibility"]["trainable"]
            or len(restored_review["samples"]) != len(source_samples)
            or restored_review["samples"][0]["included"]):
        raise RuntimeError("Embedded in-place editor lost membership or ignored 100 percent validation")
    edits_file.write_text(json.dumps({"keep": [s["id"] for s in source_samples],
                                      "validation_fraction": .05, "seed": 42}))
    invoke("Dataset Studio", ["update-dataset", str(source), "--edits-json", str(edits_file),
                               "--expected-manifest-sha256", updated["manifest_sha256"]])
    if any(file.read_bytes() != content for file, content in payloads.items()):
        raise RuntimeError("Embedded metadata editor rewrote scientific array bytes")
    report = invoke("RAFT Studio", ["workspace", str(workspace)])
    if len(report["datasets"]) != 3 or report["warnings"]:
        raise RuntimeError("Embedded Trainer did not recognize the prepared fixture")
    invoke("Dataset Studio", ["cleanup-dataset", str(edited), "--workspace", str(workspace), "--confirm"])
    invoke("Dataset Studio", ["cleanup-dataset", str(prepared), "--workspace", str(workspace), "--confirm"])
    if prepared.exists() or not source.is_dir():
        raise RuntimeError("Embedded cleanup affected the wrong fixture")
    archived = invoke("Dataset Studio", ["archive-dataset", str(source), "--workspace", str(workspace)])
    if source.exists() or not Path(archived["archived_dataset"]).is_dir():
        raise RuntimeError("Embedded archive did not preserve the source fixture")
    print("Packaged dataset/trainer backends: in-place membership/restore, committed-save recovery and 5/100 percent splits, unchanged array bytes, composition, review, workspace, cleanup and archive passed")


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
            backend_smoke(app_root, project)
        finally:
            listener.close()
            Path(address).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
