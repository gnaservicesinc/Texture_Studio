"""Create an isolated Texture Studio venv; never modify the selected base Python."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import venv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        raise RuntimeError('Use Python 3.11 or later on Apple Silicon.')
    target = Path(args.destination).absolute()
    if target.exists():
        raise RuntimeError('Runtime destination already exists; remove the app-managed runtime before reinstalling.')
    active_child = None
    cancelled = False
    cancellation_deadline = 0

    def kill_group(sig):
        if active_child is not None:
            try:
                os.killpg(active_child.pid, sig)
            except ProcessLookupError:
                pass

    def cancel(signum, frame):
        nonlocal cancelled, cancellation_deadline
        if cancelled:
            return
        cancelled = True
        cancellation_deadline = time.monotonic() + 1
        # Never wait/poll a subprocess from a signal handler: that can reenter
        # subprocess's waitpid lock. The main loop kills and reaps the child.
        kill_group(signal.SIGTERM)

    signal.signal(signal.SIGTERM, cancel)

    def run_child(arguments, *, stdout=sys.stderr, stderr=sys.stderr):
        nonlocal active_child
        if cancelled:
            raise SystemExit(130)
        active_child = subprocess.Popen(arguments, stdout=stdout, stderr=stderr, start_new_session=True)
        while True:
            status = active_child.poll()
            if cancelled:
                # Even if the direct child exits early, kill any descendants
                # remaining in its owned process group before cleanup begins.
                if time.monotonic() >= cancellation_deadline:
                    kill_group(signal.SIGKILL)
                    active_child.wait()
                    active_child = None
                    raise SystemExit(130)
            elif status is not None:
                active_child.wait()
                active_child = None
                if status != 0:
                    raise RuntimeError(f'Python setup child exited with status {status}; the earlier diagnostic identifies the cause.')
                return
            time.sleep(0.05)

    print(json.dumps({'progress': 0.05, 'message': 'Creating the app-managed Python environment'}), flush=True)
    target.mkdir(parents=True)
    (target / '.texture-studio-runtime').write_text('Texture Studio DA3 runtime\n')
    # ensurepip is run explicitly so it shares the same cancellation/reaping
    # contract as pip and the final runtime probe.
    venv.EnvBuilder(with_pip=False, symlinks=True).create(target)
    python = target / 'bin/python'
    run_child([str(python), '-I', '-m', 'ensurepip', '--upgrade'])
    print(json.dumps({'progress': 0.15, 'message': 'Installing pinned PyTorch and inference dependencies'}), flush=True)
    run_child([str(python), '-I', '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
        '--prefer-binary', '--only-binary=torch,torchvision,numpy,Pillow,safetensors,opencv-python-headless',
        '-r', str(Path(__file__).with_name('requirements.txt'))])
    print(json.dumps({'progress': 0.9, 'message': 'Checking the Metal inference backend'}), flush=True)
    run_child([str(python), '-I', '-B', str(Path(__file__).with_name('worker.py')), '--probe'], stdout=sys.stdout)
    print(json.dumps({'progress': 1, 'message': 'Python runtime ready'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error)}), flush=True)
        raise SystemExit(1)
