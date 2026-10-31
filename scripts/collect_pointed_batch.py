#!/usr/bin/env python3
"""Collect one pointed-buoy target batch with deterministic checked geometry."""
import argparse
import copy
import sys

import collect_green_batch as collector
from usv_vla.green_routes import generate_scene as green_route
from usv_vla.green_shore import generate_green_shore
from usv_vla.shore_clear_scene import generate as generate_clear_shore
from usv_vla.scene_plan import validate_scene

SLOTS = {'red': 1, 'yellow': 3, 'green': 5, 'blue': 7}


def pointed_scene(scene, color, seed):
    """Swap green-round target with the requested pointed slot and revalidate."""
    target_slot = SLOTS[color]
    scene = copy.deepcopy(scene)
    scene['dataset_type'], scene['target_slot'] = f'{color}_pointed_multi_v1', target_slot
    scene['source_generation_seed'], scene['seed'] = scene['seed'], int(seed)
    for buoy in scene['buoys']:
        buoy['slot'] = {4: target_slot, target_slot: 4}.get(buoy['slot'], buoy['slot'])
    swap = {'v2_buoy_5': f'v2_buoy_{target_slot+1}', f'v2_buoy_{target_slot+1}': 'v2_buoy_5'}
    route = scene['route']
    for key in ('required_obstacle_models', 'cylinder_models'):
        route[key] = [swap.get(name, name) for name in route.get(key, [])]
    route['instruction'] = (route['instruction'].replace(f'{color} pointed buoy', '__POINTED_TARGET_SWAP__')
                            .replace('green round buoy', f'{color} pointed buoy')
                            .replace('__POINTED_TARGET_SWAP__', 'green round buoy'))
    if 'shore_obstacle' in scene:
        info = scene['shore_obstacle']
        info['obstacle_model'] = swap.get(info['obstacle_model'], info['obstacle_model'])
    scene.pop('initial_visibility', None)
    scene['initial_visibility'] = validate_scene(scene)
    return scene


def make_pointed(color, mode, index, seed):
    errors = []
    for attempt in range(100):
        generation_seed = seed if attempt == 0 else collector.bc._scene_seed(seed, f'{color}_pointed_geometry', attempt)
        try:
            if mode == 'shore' and index >= 30:
                base = generate_clear_shore(4, index, generation_seed)
            elif mode == 'shore':
                base = generate_green_shore(index, generation_seed)
            else:
                base = green_route(mode, index, generation_seed)
            return pointed_scene(base, color, seed)
        except ValueError as error:
            errors.append(str(error))
    raise ValueError(f'Cannot generate safe {color} pointed {mode} scene: {errors[-3:]}')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--color', required=True, choices=tuple(SLOTS))
    options, remainder = parser.parse_known_args()
    color = options.color
    collector.ROOT = collector.bc.OUTPUTS / f'episodes/{color}_pointed_200_20261031'
    collector.ROOT_PREFIX = f'{color}_pointed_'
    collector.DATASET_TYPE = f'{color}_pointed_multi_v1'
    collector.TARGET_SLOT = SLOTS[color]
    collector.TARGET_NAME = f'{color}-pointed'
    collector.SEED = 20261031 + SLOTS[color]
    collector.generate_route = lambda mode, index, seed: make_pointed(color, mode, index, seed)
    collector.generate_green_shore = lambda index, seed: make_pointed(color, 'shore', index, seed)
    sys.argv = [sys.argv[0], *remainder]
    if '--root' not in sys.argv:
        sys.argv.extend(['--root', str(collector.ROOT)])
    collector.main()


if __name__ == '__main__':
    main()
