import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


METRIC_FIELDS = ['F-Score', 'CDL1', 'CDL2', 'EMDistance', 'UCD', 'UHD']
SUMMARY_FIELDS = [
    'experiment_name',
    'commit_hash',
    'config_path',
    'seed',
    'source_dataset',
    'target_dataset',
    'category',
    'method',
    'status',
    'best_epoch',
    'best_metric_name',
    'best_metric_value',
    *METRIC_FIELDS,
    'checkpoint_path',
    'output_dir',
]


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _git_commit_hash(repo_root):
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except Exception:
        return ''
    return result.stdout.strip()


def metric_dict(metric_names, metric_values):
    return {name: float(value) for name, value in zip(metric_names, metric_values)}


class ExperimentLogger:
    """Writes per-run DGPointMamba metadata, metrics, and one-row summaries."""

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.output_dir = Path(getattr(args, 'experiment_path', '.'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_meta_path = self.output_dir / 'run_meta.json'
        self.metrics_path = self.output_dir / 'metrics.jsonl'
        self.summary_path = self.output_dir / 'summary.csv'
        self.repo_root = Path(__file__).resolve().parents[1]
        self.start_time = _now_iso()
        self.base_meta = self._build_base_meta()
        self.best_epoch = ''
        self.best_metrics = {}
        self.best_metric_name = self._default_best_metric_name()
        self.checkpoint_path = ''

    def _build_base_meta(self):
        dataset = _cfg_get(self.config, 'dataset', None)
        train_cfg = _cfg_get(dataset, 'train', None)
        train_base = _cfg_get(train_cfg, '_base_', None)
        model_cfg = _cfg_get(self.config, 'model', None)
        generator_cfg = _cfg_get(self.config, 'domain_generator', None)
        generator_type = _cfg_get(generator_cfg, 'type', 'none')
        generator_enabled = bool(_cfg_get(generator_cfg, 'enable', generator_type not in (None, 'none')))
        method = generator_type if generator_enabled else 'source_only'

        return {
            'experiment_name': getattr(self.args, 'exp_name', ''),
            'commit_hash': _git_commit_hash(self.repo_root),
            'config_path': getattr(self.args, 'config', ''),
            'seed': getattr(self.args, 'seed', ''),
            'source_dataset': _cfg_get(train_cfg, 'virtual_dataset', _cfg_get(train_base, 'NAME', '')),
            'target_dataset': _cfg_get(train_cfg, 'real_dataset', ''),
            'category': _cfg_get(train_base, 'CLASS_CHOICE', ''),
            'method': method,
            'model_name': _cfg_get(model_cfg, 'NAME', ''),
            'output_dir': str(self.output_dir),
            'checkpoint_path': '',
            'start_time': self.start_time,
            'status': 'running',
        }

    def _default_best_metric_name(self):
        target_dataset = self.base_meta.get('target_dataset', '')
        if target_dataset in ['KITTI', 'ScanNet', 'MatterPort']:
            return _cfg_get(self.config, 'consider_metric_3', 'UCD')
        return _cfg_get(self.config, 'consider_metric_2', 'CDL2')

    def write_run_meta(self, status='running', **extra):
        meta = dict(self.base_meta)
        meta.update({
            'status': status,
            'checkpoint_path': self.checkpoint_path,
            'best_epoch': self.best_epoch,
            'best_metric_name': self.best_metric_name,
        })
        if self.best_metrics:
            meta['best_metrics'] = dict(self.best_metrics)
            meta['best_metric_value'] = self.best_metrics.get(self.best_metric_name, '')
        if status in ['completed', 'failed']:
            meta['end_time'] = _now_iso()
        meta.update(extra)
        self.run_meta_path.write_text(json.dumps(_jsonable(meta), indent=2, sort_keys=True) + '\n', encoding='utf-8')

    def log_metrics(self, record):
        payload = {'time': _now_iso(), **record}
        with self.metrics_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(_jsonable(payload), sort_keys=True) + '\n')

    def update_best(self, epoch, metrics, checkpoint_path):
        self.best_epoch = int(epoch) if epoch not in ['', None] else ''
        self.best_metrics = dict(metrics)
        self.checkpoint_path = checkpoint_path

    def write_summary(self, status='running', metrics=None, epoch=None, checkpoint_path=None):
        metrics = dict(metrics or self.best_metrics or {})
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        row = {
            'experiment_name': self.base_meta.get('experiment_name', ''),
            'commit_hash': self.base_meta.get('commit_hash', ''),
            'config_path': self.base_meta.get('config_path', ''),
            'seed': self.base_meta.get('seed', ''),
            'source_dataset': self.base_meta.get('source_dataset', ''),
            'target_dataset': self.base_meta.get('target_dataset', ''),
            'category': self.base_meta.get('category', ''),
            'method': self.base_meta.get('method', ''),
            'status': status,
            'best_epoch': epoch if epoch is not None else self.best_epoch,
            'best_metric_name': self.best_metric_name,
            'best_metric_value': metrics.get(self.best_metric_name, ''),
            'checkpoint_path': self.checkpoint_path,
            'output_dir': str(self.output_dir),
        }
        for metric_name in METRIC_FIELDS:
            row[metric_name] = metrics.get(metric_name, '')

        with self.summary_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow({field: _jsonable(row.get(field, '')) for field in SUMMARY_FIELDS})

    def mark_failed(self, stage, error, last_step=None):
        error_summary = str(error).splitlines()[0][:500]
        self.log_metrics({
            'event': 'failure',
            'status': 'failed',
            'stage': stage,
            'error_summary': error_summary,
            'last_step': last_step,
        })
        self.write_run_meta(status='failed', failed_stage=stage, error_summary=error_summary, last_step=last_step)
        self.write_summary(status='failed')
