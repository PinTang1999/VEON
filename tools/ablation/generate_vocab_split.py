#!/usr/bin/env python3
"""Freeze a train vocabulary after auditing POP-3D annotation coverage."""
import argparse
import json
import random
from pathlib import Path

from common import coverage, file_hash, query_specs, read_benchmark, read_words, vocab, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', choices=['nuscenes_brief', 'nuscenes_default'], default='nuscenes_brief')
    p.add_argument('--held-out', nargs='*', default=[])
    p.add_argument('--held-out-list')
    p.add_argument('--held-out-superclasses', nargs='*', default=[])
    p.add_argument('--aliases', help='JSON list of explicitly equivalent name groups')
    p.add_argument('--query-map', help='JSON evaluation queries with explicit class_name mappings')
    p.add_argument('--auto-count', type=int, default=0, help='Select subclass candidates using train/val annotations')
    p.add_argument('--benchmark-root', default='data/nuscenes/retrieval_benchmark')
    p.add_argument('--selection-split', choices=['train', 'val'], default='train')
    p.add_argument('--min-positive', type=int, default=10)
    p.add_argument('--allow-unannotated', action='store_true', help='Structural split only; not approved for quantitative evaluation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', required=True)
    a = p.parse_args()
    aliases = json.loads(Path(a.aliases).read_text()) if a.aliases else []
    held = a.held_out + (read_words(a.held_out_list) if a.held_out_list else [])
    if a.auto_count < 0 or a.min_positive < 1:
        p.error('auto-count must be nonnegative and min-positive must be positive')
    if a.auto_count and (held or a.held_out_superclasses):
        p.error('auto-count and explicit exclusions are mutually exclusive')
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {}
    selection = dict(verified=False)
    if a.auto_count or ((held or a.held_out_superclasses) and not a.allow_unannotated):
        rows = read_benchmark(a.benchmark_root, a.selection_split)
        stats = coverage(a.benchmark_root, rows)
        write_json(out / 'coverage.json', stats)
        mapping = {q['annotation_prompt']: q['class_name'] for q in query_specs(a.query_map)}
        by_name = {}
        for prompt, counts in stats.items():
            names = vocab.alias_closure([mapping.get(prompt, prompt)], aliases)
            for name in names:
                item = by_name.setdefault(name, dict(positive=0, visible_positive=0, valid_instances=0))
                for key in item:
                    item[key] += counts[key]
        def covered(name):
            c = by_name.get(vocab.normalize(name), {})
            return c.get('positive', 0) >= a.min_positive and c.get('visible_positive', 0) >= a.min_positive and c.get('valid_instances', 0) > 0
        if a.auto_count:
            candidates = [e['name'] for e in vocab.base_entries(a.base) if covered(e['name'])]
            random.Random(a.seed).shuffle(candidates)
            for name in candidates:
                candidate = vocab.make_split(held + [name], base=a.base, alias_groups=aliases)
                # Subclass study must retain a sibling in every superclass.
                if not candidate['removed_superclass_ids'] and len(candidate['held_out']) > len(vocab.make_split(held, base=a.base, alias_groups=aliases)['held_out']):
                    held.append(name)
                if len(held) == a.auto_count:
                    break
            if len(held) != a.auto_count:
                raise ValueError('Only {} eligible subclass exclusions; inspect coverage.json or supply an explicit query-map'.format(len(held)))
        preliminary = vocab.make_split(held, a.held_out_superclasses, a.base, aliases)
        unsupported = [name for name in preliminary['held_out'] if not covered(name)]
        if unsupported:
            raise ValueError('Insufficient annotated positive/negative coverage for: {}. '
                             'Use --allow-unannotated only for structural/qualitative studies.'.format(unsupported))
        csv_path = Path(a.benchmark_root) / ('retrieval_anns_{}.csv'.format(a.selection_split))
        selection = dict(verified=True, split=a.selection_split, csv=str(csv_path.resolve()),
                         csv_sha256=file_hash(csv_path), min_positive=a.min_positive,
                         annotation_sha256={path: file_hash(Path(a.benchmark_root) / path)
                                            for path in sorted({r[k] for r in rows for k in ('anno', 'matching_points')})},
                         query_map=query_specs(a.query_map))
    manifest = vocab.make_split(held, a.held_out_superclasses, a.base, aliases)
    manifest.update(seed=a.seed, selection=selection)
    original = vocab.make_split(base=a.base, alias_groups=aliases)
    manifest['evaluation_query_map'] = query_specs(a.query_map)
    original['evaluation_query_map'] = query_specs(a.query_map)
    write_json(out / 'train_vocabulary.json', manifest)
    write_json(out / 'original_vocabulary.json', original)
    write_json(out / 'held_out.json', manifest['held_out'])
    write_json(out / 'evaluation_prompts.json', [dict(query=e['name'], class_name=e['name'], annotation_prompt=e['name'])
                                              for e in manifest['original_entries']])
    print('{} retained / {} original; excluded: {}'.format(len(manifest['entries']), len(manifest['original_entries']), manifest['held_out']))
    print('Manifest: {}'.format((out / 'train_vocabulary.json').resolve()))


if __name__ == '__main__':
    main()
