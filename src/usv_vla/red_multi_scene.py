"""Red-target multi-buoy plans derived from the audited green-route generator."""
import copy

from usv_vla.green_routes import generate_scene as generate_green_scene
from usv_vla.scene_plan import validate_scene


def generate_scene(mode, index, seed):
    scene = copy.deepcopy(generate_green_scene(mode, index, seed))
    scene['dataset_type'] = 'red_multi_v1'
    scene['target_slot'] = 0
    for buoy in scene['buoys']:
        buoy['slot'] = {0: 4, 4: 0}.get(buoy['slot'], buoy['slot'])
    route = scene['route']
    for key in ('required_obstacle_models', 'cylinder_models'):
        route[key] = [{'v2_buoy_1': 'v2_buoy_5', 'v2_buoy_5': 'v2_buoy_1'}.get(v, v)
                      for v in route.get(key, [])]
    route['instruction'] = (route['instruction'].replace('red round buoy', '__SWAPPED_ROUND_BUOY__')
                            .replace('green round buoy', 'red round buoy')
                            .replace('__SWAPPED_ROUND_BUOY__', 'green round buoy'))
    scene.pop('initial_visibility', None)
    scene['initial_visibility'] = validate_scene(scene)
    return scene
