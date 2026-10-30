#!/usr/bin/env python3
"""Blue-round target batch using the green collector's unchanged safety checks."""
import copy
import sys

import collect_green_batch as collector
from usv_vla.green_routes import generate_scene as green_route
from usv_vla.green_shore import generate_green_shore
from usv_vla.scene_plan import validate_scene


def blue_scene(scene):
    scene = copy.deepcopy(scene)
    scene['dataset_type'], scene['target_slot'] = 'blue_multi_v1', 6
    for buoy in scene['buoys']:
        buoy['slot'] = {4: 6, 6: 4}.get(buoy['slot'], buoy['slot'])
    swap = {'v2_buoy_5': 'v2_buoy_7', 'v2_buoy_7': 'v2_buoy_5'}
    route = scene['route']
    for key in ('required_obstacle_models', 'cylinder_models'):
        route[key] = [swap.get(name, name) for name in route.get(key, [])]
    instruction = route['instruction']
    route['instruction'] = (instruction.replace('blue round buoy', '__ROUND_BUOY_SWAP__')
                            .replace('green round buoy', 'blue round buoy')
                            .replace('__ROUND_BUOY_SWAP__', 'green round buoy'))
    if 'shore_obstacle' in scene:
        info = scene['shore_obstacle']
        info['obstacle_model'] = swap.get(info['obstacle_model'], info['obstacle_model'])
    scene.pop('initial_visibility', None)
    scene['initial_visibility'] = validate_scene(scene)
    return scene


def main():
    collector.ROOT = collector.bc.OUTPUTS / 'episodes/blue_round_200_20261028'
    collector.ROOT_PREFIX = 'blue_'
    collector.DATASET_TYPE = 'blue_multi_v1'
    collector.TARGET_SLOT = 6
    collector.TARGET_NAME = 'blue-round'
    collector.SEED = 20261029
    collector.generate_route = lambda mode, index, seed: blue_scene(green_route(mode, index, seed))
    collector.generate_green_shore = lambda index, seed: blue_scene(generate_green_shore(index, seed))
    if '--root' not in sys.argv:
        sys.argv.extend(['--root', str(collector.ROOT)])
    collector.main()


if __name__ == '__main__':
    main()
