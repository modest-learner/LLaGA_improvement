"""Allow disjoint stages alongside an already running, older LP runner."""
import argparse
import os
from pathlib import Path


def check(lock_path, requested):
    stat = lock_path.stat()
    owners = []
    # flock(1) exits after acquiring a lock on the shell's inherited descriptor.
    # The kernel may report the exited flock PID; inspect the shell's fdinfo instead.
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            held = (proc / 'fd/9').stat()
            if (held.st_dev, held.st_ino) != (stat.st_dev, stat.st_ino):
                continue
            info = (proc / 'fdinfo/9').read_text().splitlines()
            if not any('FLOCK ADVISORY WRITE' in ' '.join(line.split()) for line in info):
                continue
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            if (len(argv) < 2 or Path(os.fsdecode(argv[0])).name != 'bash'
                    or Path(os.fsdecode(argv[1])).name != 'run_lp_next_round.sh'):
                continue
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        pid = int(proc.name)
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            # Recognize only this project's legacy runner, with an explicit stage.
            if (len(argv) < 2 or Path(os.fsdecode(argv[0])).name != 'bash'
                    or Path(os.fsdecode(argv[1])).name != 'run_lp_next_round.sh'
                    or (proc / 'cwd').resolve() != Path.cwd().resolve()):
                raise ValueError('unrecognized lock owner')
            env = dict(item.split(b'=', 1) for item in
                       (proc / 'environ').read_bytes().split(b'\0') if b'=' in item)
            stage = env.get(b'STAGE', b'all').decode()
            if stage not in ('source', 'baselines', 'seeds'):
                raise ValueError('legacy runner may execute multiple stages')
            if requested == 'all' or requested == stage:
                raise ValueError(f'legacy runner already owns stage {stage}')
            owners.append((pid, stage))
        except FileNotFoundError:
            continue  # The owner finished during inspection.
        except (PermissionError, ValueError) as exc:
            raise SystemExit(f'Cannot start {requested}: lock owner PID {pid}: {exc}')
    for pid, stage in owners:
        print(f'Allow independent {requested} stage alongside legacy {stage} runner PID {pid}.', flush=True)
    if not owners:
        # A released lock is fine; an owner we could not identify is not.
        import fcntl
        with lock_path.open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('Cannot identify the legacy lock owner; retry after it finishes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', 'source', 'baselines', 'seeds', 'summary'], required=True)
    args = parser.parse_args()
    check(args.lock, args.stage)
