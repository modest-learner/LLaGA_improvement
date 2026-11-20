"""CPU-only checks of stage exclusion, legacy compatibility and partial JSONL reads."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.summarize_lp_next_round import read


def main():
    repo = Path(__file__).resolve().parents[1]
    prefix = (repo / 'scripts/run_lp_next_round.sh').read_text().split(
        'if [[ "$stage" != summary ]]; then', 1)[0]
    processes = []
    with tempfile.TemporaryDirectory(prefix='lp_stage_lock_') as temporary:
        root = Path(temporary)
        probe = root / 'probe.sh'
        probe.write_text(prefix + '\nprintf "LOCK_READY\\n"\nread -r release\n')

        def start(stage, script=probe):
            env = dict(os.environ, STAGE=stage, ROUND_ROOT=str(root))
            proc = subprocess.Popen(['bash', str(script)], cwd=repo, env=env,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            processes.append(proc)
            # Rejected processes exit; accepted probes print a marker and wait on stdin.
            output = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    return proc, ''.join(output)
                output.append(line)
                if line.strip() == 'LOCK_READY':
                    return proc, ''.join(output)

        def release(proc):
            proc.communicate('done\n', timeout=10)

        try:
            source, _ = start('source')
            seeds, _ = start('seeds')
            assert source.poll() is None and seeds.poll() is None
            duplicate, msg = start('seeds')
            assert duplicate.wait(timeout=10) != 0 and 'already running' in msg
            all_stage, _ = start('all')
            assert all_stage.wait(timeout=10) != 0
            release(source);release(seeds)
            all_stage, _ = start('all')
            assert all_stage.poll() is None
            source, _ = start('source')
            assert source.wait(timeout=10) != 0
            release(all_stage)

            legacy_file = root / 'run_lp_next_round.sh'
            legacy_file.write_text('set -eu\nexec 9>>"$ROUND_ROOT/.runner.lock"\n'
                                   'flock -xn 9\nprintf "LOCK_READY\\n"\nread -r release\n')
            legacy, _ = start('source', legacy_file)
            seeds, msg = start('seeds')
            assert seeds.poll() is None and 'legacy source' in msg
            source, _ = start('source')
            assert source.wait(timeout=10) != 0
            all_stage, _ = start('all')
            assert all_stage.wait(timeout=10) != 0
            release(seeds);release(legacy)

            legacy, _ = start('all', legacy_file)
            seeds, _ = start('seeds')
            assert seeds.wait(timeout=10) != 0
            release(legacy)

            partial = root / 'partial.jsonl'
            partial.write_text('{"id": 1}\n{"id":')
            assert read(partial, allow_partial_tail=True) == [{'id': 1}]
            partial.write_text('{"id": 1}\n{"id": 2}\n')
            assert len(read(partial, allow_partial_tail=True)) == 2
            print('PASS: disjoint stages, duplicate/all exclusion, legacy migration, partial JSONL.')
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
                    proc.communicate(timeout=10)


if __name__ == '__main__':
    main()
