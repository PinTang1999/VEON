"""Dependency-light vocabulary manifests shared by SAN and ablation tools.

Superclass IDs are Occ3D IDs, never positions in a filtered vocabulary.
"""
import json
import re
from pathlib import Path

from .nuscenes_vol import NUSCENES_CLASSES, NUSCENES_CLASSES_BRIEF

ORIGINAL_PRIORITY = [2, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 1, 1, 1, 1, 1, 1]
SUPERCLASSES = [x['category'] for x in NUSCENES_CLASSES_BRIEF]
# These are explicit equivalences, not inferred substring matches. Broader
# concepts (e.g. car vs sedan) deliberately remain distinct.
ALIAS_GROUPS = [
    ['road sign', 'street sign', 'traffic sign'],
    ['trash can', 'garbage bin', 'garbage-bin with wheels', 'rubbish bin'],
    ['wheel barrow', 'wheelbarrow'], ['mini-van', 'minivan'],
    ['SUV', 'sport utility vehicle'], ['sedan', 'saloon'],
    ['police officer', 'policeman', 'policewoman'],
]


def normalize(text):
    return re.sub(r'\s+', ' ', re.sub(r'[_-]+', ' ', text.strip().lower()))


def base_entries(name='nuscenes_brief'):
    sources = {'nuscenes_brief': NUSCENES_CLASSES_BRIEF,
               'nuscenes_default': NUSCENES_CLASSES}
    if name not in sources:
        raise ValueError('Vocabulary ablations require a nuScenes vocabulary')
    return [dict(name=item[0], prompt=(item[0] if len(item) == 1 else
                 item[0] + ", in detail '" + item[1] + "'"), superclass_id=i)
            for i, group in enumerate(sources[name]) for item in group['detailed_items']]


def alias_closure(words, extra_groups=()):
    result = {normalize(w) for w in words}
    groups = [{normalize(x) for x in g} for g in ALIAS_GROUPS + list(extra_groups)]
    while True:
        expanded = result | set().union(*(g for g in groups if g & result))
        if expanded == result:
            return result
        result = expanded


def validate_entries(entries):
    if not entries:
        raise ValueError('Training vocabulary cannot be empty')
    seen = set()
    for entry in entries:
        name = normalize(entry['name'])
        sid = entry['superclass_id']
        if not name or name in seen or not str(entry['prompt']).strip():
            raise ValueError('Empty or duplicate vocabulary entry: {}'.format(entry))
        if isinstance(sid, bool) or not isinstance(sid, int) or not 0 <= sid < 17:
            raise ValueError('Superclass ID must be an integer in [0,16]; free=17 is not a text class')
        seen.add(name)
    return entries


def make_split(held_out=(), held_out_superclasses=(), base='nuscenes_brief',
               alias_groups=(), entries=None):
    original = validate_entries(base_entries(base) if entries is None else entries)
    known = {normalize(e['name']) for e in original}
    requested = {normalize(w) for w in held_out}
    if requested - known:
        raise ValueError('Unknown held-out subclasses: {}'.format(sorted(requested - known)))
    superclass_ids = []
    for superclass in held_out_superclasses:
        matches = [i for i, name in enumerate(SUPERCLASSES) if normalize(name) == normalize(superclass)]
        if not matches:
            raise ValueError('Unknown superclass: {}'.format(superclass))
        superclass_ids.extend(matches)
    requested |= {normalize(e['name']) for e in original if e['superclass_id'] in superclass_ids}
    blocked = alias_closure(requested, alias_groups)
    removed = [e for e in original if normalize(e['name']) in blocked]
    retained = [dict(e) for e in original if normalize(e['name']) not in blocked]
    # Descriptive prompts can themselves directly name excluded subclasses.
    for e in retained:
        prompt = ' ' + normalize(re.sub(r'[^\w\s-]', ' ', e['prompt'])) + ' '
        forms = {form for word in blocked for form in
                 (word, word + 's', word + 'es', word[:-1] + 'ies' if word.endswith('y') else word)}
        if any(' ' + word + ' ' in prompt for word in forms):
            raise ValueError('Retained description names a held-out alias: {}. '
                             'Use nuscenes_brief or explicitly edit the prompt.'.format(e['name']))
    validate_entries(retained)
    ids = {e['superclass_id'] for e in retained}
    removed_ids = {e['superclass_id'] for e in original} - ids
    partial_ids = {e['superclass_id'] for e in removed} & ids
    return dict(schema_version=1, base=base, entries=retained, original_entries=original,
                held_out=[e['name'] for e in removed], requested_held_out=list(held_out),
                held_out_superclasses=list(held_out_superclasses),
                removed_superclass_ids=sorted(set(range(17)) - ids),
                excluded_aliases=sorted(blocked), alias_groups=list(alias_groups),
                superclass_names=SUPERCLASSES, free_index=17, ignore_index=255,
                scope=('original vocabulary' if not removed else
                       'mixed subclass and superclass holdouts' if removed_ids and partial_ids else
                       'held-out superclasses' if removed_ids else 'held-out subclasses'),
                supervision_notes=[
                    'Only direct text prototypes and corresponding SAN pseudo-label channels are removed.',
                    'Broader retained concepts can still supervise the same objects.',
                    'Frozen CLIP/SAN/depth foundation pretraining is not vocabulary held out.',
                    'Alias groups require author review; automatic linguistic equivalence is not claimed.',
                    'Occupied/free targets and camera masks remain supervised by Occ3D.'])


def load_manifest(value):
    if isinstance(value, (str, Path)):
        with open(value) as f:
            value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError('Training vocabulary must be a manifest generated by generate_vocab_split.py')
    validate_entries(value['entries'])
    if value.get('schema_version') != 1:
        raise ValueError('Unsupported vocabulary manifest schema')
    # Reconstruct the exclusion instead of trusting an unchecked manifest.
    expected = make_split(value.get('requested_held_out', value.get('held_out', [])),
                          value.get('held_out_superclasses', []), value['base'],
                          value.get('alias_groups', []), value['original_entries'])
    if expected['entries'] != value['entries'] or expected['held_out'] != value['held_out']:
        raise ValueError('Manifest entries do not match its exclusion policy')
    return value


def merge_superclasses(tensor, dim, reflection, include_background=False):
    """Max-reduce arbitrary subclass order into stable Occ3D IDs 0..16.

    Absent superclasses have -inf logits and cannot win argmax. SAN's learned
    background is a separate final channel; it is not a train vocabulary item.
    """
    import torch
    if not reflection or any(isinstance(i, bool) or not isinstance(i, int)
                             or i < 0 or i >= 17 for i in reflection):
        raise ValueError('class_reflection must contain semantic IDs in [0,16]')
    if tensor.shape[dim] != len(reflection) + int(include_background):
        raise ValueError('Vocabulary channels and class_reflection disagree')
    shape = list(tensor.shape)
    shape[dim] = 1
    merged = []
    for sid in range(17):
        indices = [j for j, value in enumerate(reflection) if value == sid]
        if indices:
            selected = torch.tensor(indices, device=tensor.device)
            merged.append(tensor.index_select(dim, selected).max(dim, keepdim=True).values)
        else:
            merged.append(tensor.new_full(shape, float('-inf')))
    if include_background:
        merged.append(tensor.narrow(dim, len(reflection), 1))
    return torch.cat(merged, dim=dim)
