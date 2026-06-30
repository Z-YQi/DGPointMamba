import argparse
import csv
import json
from pathlib import Path


LOSS_FIELDS = ['loss_total', 'loss_rec_clean', 'loss_rec_aug_src']
METRIC_FIELDS = ['F-Score', 'CDL1', 'CDL2', 'EMDistance', 'UCD', 'UHD']


def _read_summary(exp_dir):
    summary_path = exp_dir / 'summary.csv'
    if not summary_path.exists():
        return {}
    with summary_path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _read_metrics(exp_dir):
    metrics_path = exp_dir / 'metrics.jsonl'
    if not metrics_path.exists():
        return []
    records = []
    with metrics_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _loss_trend(records):
    train_records = [r for r in records if r.get('event') == 'train_step']
    trend = {}
    for field in LOSS_FIELDS:
        values = [float(r[field]) for r in train_records if field in r and r[field] != '']
        if values:
            trend[f'{field}_first'] = values[0]
            trend[f'{field}_last'] = values[-1]
            trend[f'{field}_min'] = min(values)
    validation_records = [r for r in records if r.get('event') == 'validation']
    for field in METRIC_FIELDS:
        values = [float(r[field]) for r in validation_records if field in r and r[field] != '']
        if values:
            trend[f'{field}_last'] = values[-1]
            trend[f'{field}_best_min'] = min(values)
            trend[f'{field}_best_max'] = max(values)
    return trend


def collect_experiment(exp_dir):
    exp_dir = Path(exp_dir)
    row = _read_summary(exp_dir)
    if not row:
        run_meta_path = exp_dir / 'run_meta.json'
        row = json.loads(run_meta_path.read_text(encoding='utf-8')) if run_meta_path.exists() else {}
    row = dict(row)
    row['experiment_dir'] = str(exp_dir)
    row.update(_loss_trend(_read_metrics(exp_dir)))
    return row


def write_rows(rows, output_path=None):
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    if output_path:
        with Path(output_path).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return
    writer = csv.DictWriter(__import__('sys').stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Summarize DGPointMamba experiment directories.')
    parser.add_argument('experiment_dirs', nargs='+', help='Experiment output directories containing summary.csv or run_meta.json.')
    parser.add_argument('--output', type=str, default=None, help='Optional CSV path for the comparison table.')
    args = parser.parse_args()

    rows = [collect_experiment(path) for path in args.experiment_dirs]
    write_rows(rows, args.output)


if __name__ == '__main__':
    main()
