"""Operational selection for training without changing collection provenance."""
import json
from pathlib import Path


def all_accepted_for_training(root):
    path = Path(root) / 'training_selection.json'
    if not path.exists():
        return False
    config = json.loads(path.read_text(encoding='utf-8'))
    if config != {'mode': 'all_accepted_for_training'}:
        raise ValueError('Invalid training selection configuration')
    return True


def selected(rows, split, *, all_for_training):
    if split == 'accepted':
        return list(rows)
    if all_for_training:
        return list(rows) if split == 'train' else []
    return [row for row in rows if row['split'] == split]


def accepted_supplements(root):
    """Resolve accepted supplemental episodes inside this batch only."""
    root = Path(root).resolve()
    manifest = root/'shore_clear_10'/'manifest.json'
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or data.get('kind') != 'shore_clear_10':
        raise ValueError('Invalid supplemental manifest')
    result = []
    for entry in data['entries']:
        if entry.get('status') != 'accepted':
            continue
        relative = Path(entry['accepted_episode'])
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != entry['episode_id'] or not relative.parts[1].startswith('attempt_'):
            raise ValueError('Invalid supplemental episode path')
        path = manifest.parent/relative
        if path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(manifest.parent) or not path.is_dir():
            raise ValueError('Missing or unsafe supplemental episode')
        result.append((Path('shore_clear_10')/relative).as_posix())
    if len(result) != len(set(result)):
        raise ValueError('Duplicate supplemental episode')
    return result
