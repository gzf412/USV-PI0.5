#!/usr/bin/env python3
"""Replace short red-target direct runs with diverse nonblocking buoy scenes."""
import math

from usv_vla import batch_collect as bc
from usv_vla.red_multi_scene import generate_scene
from usv_vla.scene_plan import validate_scene
import replace_red_multi_buoy as base

VERSION = 'red_multi_buoy_direct_v2'


def choose_entries(root, manifest):
    protected = {x['episode_id'] for x in base.repl.read(root/'instruction_overrides.json')['entries']}
    result = []
    for category in ('approach_visible', 'approach_offscreen'):
        for split, quota in (('train', 8), ('validation', 2)):
            candidates = []
            for entry in manifest['entries']:
                if entry['status'] != 'accepted' or entry['category'] != category or entry['split'] != split or entry['episode_id'] in protected:
                    continue
                scene = base.repl.read(root/entry['scene'])
                if scene.get('dataset_type') in ('red_multi_v1', 'green_multi_v1', 'blue_multi_v1'):
                    continue
                distance = math.dist(scene['initial_pose'][:2], scene['target_xy'])
                candidates.append((distance, entry['episode_id'], entry))
            if len(candidates) < quota:
                raise ValueError('Insufficient single-buoy scenes')
            for distance, _, entry in sorted(candidates)[:quota]:
                result.append((entry, {'initial_target_distance_m': distance,
                                       'ranking': 'shortest single-buoy route within category and historical split'}))
    return result


def make_scene(entry, variant, nonce=0):
    mode = 'direct_visible' if entry['category'] == 'approach_visible' else 'direct_offscreen'
    # Index 1..7 generates 2..8 buoys including the target.
    scene_index = 1 + variant % 7
    for offset in range(50):
        seed = bc._scene_seed(entry['seed'], f'{VERSION}-{nonce}-{mode}', offset)
        try:
            scene = generate_scene(mode, scene_index, seed)
            scene['seed'] = entry['seed']
            scene['index'] = entry['category_index']
            scene['replacement_geometry'] = {'geometry_seed': seed, 'mode': mode, 'nonce': nonce}
            scene['collection'] = {'episode_id':entry['episode_id'], 'category':entry['category'],
                                   'category_index':entry['category_index'], 'seed':entry['seed'],
                                   'split':entry['split']}
            if scene['category'] != entry['category'] or len(scene['buoys']) < 2:
                raise ValueError('Category or buoy count mismatch')
            scene['initial_visibility'] = validate_scene(scene)
            return scene
        except ValueError:
            continue
    raise ValueError('Unable to generate safe multi-buoy direct scene')


if __name__ == '__main__':
    base.VERSION = VERSION
    base.choose_entries = choose_entries
    base.make_scene = make_scene
    base.main()
