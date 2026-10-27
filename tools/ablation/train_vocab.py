#!/usr/bin/env python3
"""Snapshot the ablation policy, then run VEON's standard training entrypoint."""
import argparse
import os
import runpy
import sys
from pathlib import Path

from common import ROOT, file_hash, vocab, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--train-vocabulary', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--priority-mode', choices=['original', 'uniform'], default='original')
    p.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    p.add_argument('--deterministic', action='store_true')
    a = p.parse_args()
    manifest = vocab.load_manifest(a.train_vocabulary)
    sys.path.insert(0, str(ROOT))
    from mmcv import Config
    cfg = Config.fromfile(a.config)
    if cfg.model.type != 'VeonTemporal' or cfg.model.loss_occ.ov_class_number != 17:
        raise ValueError('This protocol requires VeonTemporal with all 17 superclasses open')
    cfg.model.semantic_model.vocabulary_file = None
    cfg.model.semantic_model.vocabulary_spec = manifest
    cfg.model.loss_occ.semantic_supervision = 'occupancy_only'
    cfg.model.loss_occ.priority_mode = a.priority_mode
    # Erase semantic identities before any training pipeline branch sees them.
    loaders = [x for x in cfg.data.train.pipeline if x['type'] == 'LoadOccGTFromFile']
    if len(loaders) != 1:
        raise ValueError('Expected one LoadOccGTFromFile in the VEON training pipeline')
    loaders[0]['occupancy_only'] = True
    out = Path(a.out_dir).resolve()
    cfg.work_dir = str(out)
    metadata = dict(config=str(Path(a.config).resolve()), config_sha256=file_hash(a.config),
                    train_vocabulary=manifest, train_vocabulary_file=str(Path(a.train_vocabulary).resolve()),
                    seed=a.seed, priority_mode=a.priority_mode, semantic_supervision='occupancy_only',
                    occupancy_labels='Occ3D IDs 0..16 collapse to occupied=0; free=17, ignore=255; camera mask retained',
                    initialization={k: dict(path=cfg[k], sha256=file_hash(cfg[k]))
                                    for k in ('load_from', 'depth_load_from') if cfg.get(k)})
    source_paths = ['tools/ablation/train_vocab.py', 'tools/ablation/common.py', 'tools/train.py',
                    'mmdet3d/models/detectors/veon_temporal.py',
                    'mmdet3d/models/semantic_net/vocabulary/generalization.py',
                    'mmdet3d/models/semantic_net/san_in_veon_entry_temporal.py',
                    'mmdet3d/models/semantic_net/san_in_veon_temporal.py',
                    'mmdet3d/models/semantic_net/loss/occ_loss.py',
                    'mmdet3d/models/semantic_net/loss/occ_loss_utils/occ3d_nuscenes.py',
                    'mmdet3d/datasets/pipelines/loading.py', 'mmdet3d/core/hook/ema.py']
    metadata['source_sha256'] = {path: file_hash(ROOT / path) for path in source_paths}
    cfg.vocabulary_generalization = metadata
    # One rank writes the exact same resolved config before launching normal DDP.
    # Other ranks wait through the process group initialized by tools/train.py;
    # each uses its own immutable resolved config to avoid a filesystem race.
    rank = int(os.environ.get('RANK', '0'))
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / ('resolved_train_rank{}.py'.format(rank))
    cfg.dump(str(resolved))
    if rank == 0:
        write_json(out / 'training_metadata.json', metadata)
        write_json(out / 'train_vocabulary.json', manifest)
    sys.argv = [str(ROOT / 'tools/train.py'), str(resolved), '--work-dir', str(out),
                '--seed', str(a.seed), '--launcher', a.launcher, '--local_rank', str(a.local_rank)]
    if a.deterministic:
        sys.argv.append('--deterministic')
    runpy.run_path(str(ROOT / 'tools/train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
