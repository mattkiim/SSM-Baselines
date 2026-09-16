"""Write per-seed status and final-checkpoint aggregates for a batch manifest."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


def summarize(batch_dir, results_root):
    rows = []
    for run in json.loads((batch_dir/'manifest.json').read_text()):
        status = batch_dir/f"{run['robot']}_seed{run['seed']}.json"
        record = json.loads(status.read_text()) if status.exists() else run
        name = next(x.split('=', 1)[1] for x in run['command'] if x.startswith('--run_name='))
        histories = list(results_root.glob(f'*/{name}/*seed{run["seed"]:04d}/history.csv'))
        metrics = None
        if histories:
            with histories[0].open() as f:
                valid = []
                for row in csv.DictReader(f):
                    try:
                        int(row['step'])
                        float(row['eval/cost_mean'])
                        float(row['eval/return_mean'])
                    except (KeyError, TypeError, ValueError):
                        continue  # A running trainer may be partway through appending a row.
                    valid.append(row)
            if valid:
                metrics = valid[-1]
        rows.append((run, record['status'], metrics))
    lines = ['# RCRL batch results', '',
             f'Updated {datetime.now(timezone.utc).isoformat()}.', '',
             'Final aggregates use only successful runs evaluated at 3,000,000 steps. '
             'Each evaluation averages 50 deterministic episodes. ± is the sample '
             'standard deviation across training-seed means; it is not an error bar across episodes.', '',
             '| Task | Completed seeds | Return mean ± SD | Cost mean ± SD |',
             '|---|---:|---:|---:|']
    for robot in dict.fromkeys(r['robot'] for r, _, _ in rows):
        finished = [m for r,s,m in rows if r['robot']==robot and s=='completed' and m and int(m['step'])==3000000]
        if not finished:
            lines.append(f'| {robot.title()} | 0 | — | — |')
            continue
        def aggregate(key):
            vals = [float(m[key]) for m in finished]
            return f'{statistics.mean(vals):,.2f} ± {statistics.stdev(vals):,.2f}' if len(vals)>1 else f'{vals[0]:,.2f} (n=1)'
        lines.append(f"| {robot.title()} | {len(finished)} | {aggregate('eval/return_mean')} | {aggregate('eval/cost_mean')} |")
    lines += ['', '| Task | Seed | Status | Latest evaluation step | Return | Cost |', '|---|---:|---|---:|---:|---:|']
    for r,s,m in rows:
        tail = f"{int(m['step']):,} | {float(m['eval/return_mean']):,.2f} | {float(m['eval/cost_mean']):,.2f}" if m else '— | — | —'
        lines.append(f"| {r['robot'].title()} | {r['seed']} | {s} | {tail} |")
    output = batch_dir/'results.md'
    temp = batch_dir/'results.md.tmp'
    temp.write_text('\n'.join(lines)+'\n')
    temp.replace(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch_dir', type=Path)
    parser.add_argument('--results-root', type=Path, default=Path('results'))
    args = parser.parse_args()
    print(summarize(args.batch_dir, args.results_root))
