"""Launch velocity sweeps, keeping a persistent manifest and logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import threading
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['reference'], default='reference')
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--robots', nargs='+', choices=['ant', 'humanoid', 'swimmer', 'hopper', 'cheetah', 'walker'])
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error('--parallel must be positive')
    if (args.robots is None) != (args.seeds is None):
        parser.error('--robots and --seeds must be supplied together')
    if args.seeds is not None and any(seed < 0 for seed in args.seeds):
        parser.error('seeds must be nonnegative')
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_dir = root / 'logs' / f'rcrl_batch_{stamp}'
    tasks = [('humanoid', 0), ('ant', 0), ('swimmer', 0), ('hopper', 0)]
    tasks += [(robot, seed) for seed in range(1, 5) for robot in ('humanoid', 'ant')]
    if args.robots is not None:
        tasks = [(robot, seed) for seed in dict.fromkeys(args.seeds)
                 for robot in dict.fromkeys(args.robots)]
    runs = []
    for robot, seed in tasks:
        command = ['bash', str(root / 'scripts/train_rcrl_reference.sh'), robot,
                   f'--seed={seed}', '--max_steps=3000000', '--save_interval=25000',
                   f'--run_name={robot}_rcrl_{args.protocol}_3m_{stamp}', '--notqdm']
        runs.append(dict(robot=robot, seed=seed, command=command, status='queued'))
    if args.dry_run:
        for run in runs:
            print(shlex.join(run['command']))
        return
    log_dir.mkdir(parents=True, exist_ok=False)
    run_root = root
    if args.protocol == 'reference':
        run_root = log_dir / 'source'
        run_root.mkdir()
        for folder in ('jaxrl5', 'examples', 'scripts'):
            shutil.copytree(root / folder, run_root / folder,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (root / 'results').mkdir(exist_ok=True)
        (run_root / 'results').symlink_to(root / 'results', target_is_directory=True)
        for run in runs:
            run['command'][1] = str(run_root / 'scripts/train_rcrl_reference.sh')
    (log_dir / 'manifest.json').write_text(json.dumps(runs, indent=2))
    (log_dir / 'source_hashes.json').write_text(json.dumps({
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('jaxrl5', 'examples/states', 'scripts')
        for p in (root / folder).rglob('*') if p.suffix in ('.py', '.sh')
    }, indent=2))
    with (log_dir / 'pip-freeze.txt').open('w') as output:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=output, check=True)
    env = dict(os.environ, RCRL_PYTHON=sys.executable, JAX_PLATFORMS='cuda',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env.pop('LD_LIBRARY_PATH', None)
    print(f'Batch logs: {log_dir}', flush=True)

    report_lock = threading.Lock()

    def report():
        with report_lock:
            with (log_dir / 'report.log').open('a') as output:
                subprocess.run([sys.executable, str(run_root / 'scripts/summarize_rcrl_batch.py'),
                                str(log_dir), '--results-root', str(root / 'results')],
                               stdout=output, stderr=subprocess.STDOUT, check=True)

    report()

    def train(run):
        label = f"{run['robot']}_seed{run['seed']}"
        status_path = log_dir / f'{label}.json'
        def save_status():
            temporary = status_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(run, indent=2))
            temporary.replace(status_path)
        run['status'] = 'running'
        run['started_at'] = datetime.now(timezone.utc).isoformat()
        with (log_dir / f'{label}.log').open('w') as output:
            process = subprocess.Popen(run['command'], cwd=run_root, env=env,
                                       stdout=output, stderr=subprocess.STDOUT)
            run['pid'] = process.pid
            save_status()
            print(f'{label}: started PID {process.pid}', flush=True)
            run['exit_code'] = process.wait()
        run['status'] = 'completed' if run['exit_code'] == 0 else 'failed'
        run['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_status()
        print(f"{label}: {run['status']} (exit {run['exit_code']})", flush=True)
        report()
        return run['exit_code']

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        codes = list(pool.map(train, runs))
    (log_dir / 'manifest.json').write_text(json.dumps(runs, indent=2))
    sys.exit(int(any(codes)))


if __name__ == '__main__':
    main()
