#!/usr/bin/env python3
"""Collect a yellow-round target batch with the existing multi-buoy checks."""
import copy
import sys

import collect_green_batch as collector
from usv_vla.green_routes import generate_scene as green_route
from usv_vla.green_shore import generate_green_shore
from usv_vla.shore_clear_scene import generate as generate_clear_shore
from usv_vla.scene_plan import validate_scene


def yellow_scene(scene):
    scene = copy.deepcopy(scene)
    scene['dataset_type'], scene['target_slot'] = 'yellow_multi_v1', 2
    for buoy in scene['buoys']:
        buoy['slot'] = {4: 2, 2: 4}.get(buoy['slot'], buoy['slot'])
    swap = {'v2_buoy_5': 'v2_buoy_3', 'v2_buoy_3': 'v2_buoy_5'}
    route = scene['route']
    for key in ('required_obstacle_models', 'cylinder_models'):
        route[key] = [swap.get(name, name) for name in route.get(key, [])]
    route['instruction'] = (route['instruction'].replace('yellow round buoy', '__ROUND_BUOY_SWAP__')
                            .replace('green round buoy', 'yellow round buoy')
                            .replace('__ROUND_BUOY_SWAP__', 'green round buoy'))
    if 'shore_obstacle' in scene:
        info = scene['shore_obstacle']
        info['obstacle_model'] = swap.get(info['obstacle_model'], info['obstacle_model'])
    scene.pop('initial_visibility', None)
    scene['initial_visibility'] = validate_scene(scene)
    return scene


def main():
    collector.ROOT = collector.bc.OUTPUTS / 'episodes/yellow_round_200_20261030'
    collector.ROOT_PREFIX = 'yellow_'
    collector.DATASET_TYPE = 'yellow_multi_v1'
    collector.TARGET_SLOT = 2
    collector.TARGET_NAME = 'yellow-round'
    collector.SEED = 20261030
    collector.generate_route = lambda mode, index, seed: yellow_scene(green_route(mode, index, seed))
    collector.generate_green_shore = lambda index, seed: yellow_scene(
        generate_clear_shore(4, index, seed) if index >= 30 else generate_green_shore(index, seed))
    if '--root' not in sys.argv:
        sys.argv.extend(['--root', str(collector.ROOT)])
    collector.main()


if __name__ == '__main__':
    main()
