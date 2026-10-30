"""Export audited episodes to local LeRobot v2.1. Never uploads data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from usv_vla.audit import audit_episode


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_sha256(scene):
    return hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def instruction_for_export(directory, metadata, rows):
    """Audit the recorded language, then apply a bound, post-hoc language label.

    A batch may provide instruction_overrides.json at its root. Original route,
    recorded prompts and actions remain untouched and are still independently
    audited. The mapping uses paths relative to the batch root, so moving the
    batch does not change any binding.
    """
    original = training_instruction(metadata, rows)
    directory = Path(directory).resolve()
    root = directory.parent.parent
    mapping_path = root / 'instruction_overrides.json'
    if not mapping_path.exists():
        return original, {'source': 'recorded_route', 'task': original}
    mapping_bytes = mapping_path.read_bytes()
    mapping = json.loads(mapping_bytes)
    if (not isinstance(mapping, dict) or mapping.get('schema_version') != 1
            or not isinstance(mapping.get('version'), str) or not mapping['version'].strip()
            or not isinstance(mapping.get('entries'), list)):
        raise ValueError('Invalid training instruction override mapping.')
    index = {}
    ids = set()
    for item in mapping['entries']:
        if not isinstance(item, dict):
            raise ValueError('Invalid training instruction override entry.')
        path = item.get('episode')
        episode_id = item.get('episode_id')
        if not isinstance(path, str) or not isinstance(episode_id, str):
            raise ValueError('Override requires a relative episode path and ID.')
        parts = Path(path).parts
        if (len(parts) != 2 or Path(path).is_absolute() or parts[0] != episode_id
                or not parts[1].startswith('attempt_') or path != '/'.join(parts)
                or path in index or episode_id in ids):
            raise ValueError('Duplicate or invalid override episode path/ID.')
        index[path] = item
        ids.add(episode_id)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    entries = manifest.get('entries')
    if not isinstance(entries, list):
        raise ValueError('Invalid batch manifest for training override.')
    by_id = {}
    for entry in entries:
        episode_id = entry.get('episode_id')
        if episode_id in by_id:
            raise ValueError('Duplicate episode ID in batch manifest.')
        by_id[episode_id] = entry
    for path, item in index.items():
        entry = by_id.get(item['episode_id'])
        if (entry is None or entry.get('status') != 'accepted'
                or entry.get('accepted_episode') != path
                or entry.get('split') != item.get('split')
                or item.get('split') not in ('train', 'validation')
                or entry.get('scene_sha256') != item.get('scene_sha256')):
            raise ValueError('Training override does not match the batch manifest.')
    relative = directory.relative_to(root).as_posix()
    if relative not in index:
        if directory.parent.name in ids:
            raise ValueError('Training override points to a different attempt of this episode.')
        return original, {'source': 'recorded_route', 'task': original}
    item = index[relative]
    instruction = item.get('instruction')
    if (not isinstance(instruction, str) or not instruction.strip() or instruction != instruction.strip()
            or 'Current phase:' in instruction or 'frozen initial route heading:' in instruction):
        raise ValueError('Invalid training override instruction.')
    if _scene_sha256(metadata.get('scene_plan')) != item.get('scene_sha256'):
        raise ValueError('Training override does not match accepted episode, split, or scene hash.')
    for key, filename in (('steps_sha256', 'steps.jsonl'), ('episode_metadata_sha256', 'episode.json')):
        if item.get(key) != _file_sha256(directory / filename):
            raise ValueError(f'Training override {key} does not match recorded file.')
    return instruction, {'source': 'posthoc_training_override', 'task': instruction,
                         'recorded_task': original, 'version': mapping['version'],
                         'mapping_sha256': hashlib.sha256(mapping_bytes).hexdigest(),
                         'episode_id': item['episode_id'], 'episode': relative,
                         'split': item['split'], 'scene_sha256': item['scene_sha256'],
                         'steps_sha256': item['steps_sha256'],
                         'episode_metadata_sha256': item['episode_metadata_sha256']}


def training_instruction(metadata, rows):
    """Return the fixed user task after checking every recorded task prompt.

    Historical recordings include evaluator-only progress and a frozen route
    reference in their prompts. Keep the original recording untouched for its
    independent replay audit; neither field belongs in the policy input.
    """
    route = metadata.get('route')
    if not isinstance(route, dict):
        raise ValueError('Export requires an explicit route and task instruction.')
    scene = metadata.get('scene_plan')
    if scene is not None and (not isinstance(scene, dict) or scene.get('route') != route):
        raise ValueError('Recorded task route differs from the recorded scene plan.')
    instruction = route.get('instruction')
    if not isinstance(instruction, str) or not instruction.strip() or instruction != instruction.strip():
        raise ValueError('Export requires a nonempty, explicit route instruction.')
    if 'Current phase:' in instruction or 'frozen initial route heading:' in instruction:
        raise ValueError('Route instruction contains evaluator-only context.')
    if not rows:
        raise ValueError('Export requires at least one observation.')
    include_phase = route.get('prompt_include_phase', True)
    if not isinstance(include_phase, bool):
        raise ValueError('Invalid prompt_include_phase setting.')
    if include_phase:
        try:
            initial_yaw = float(rows[0]['pose_xyzrpy'][5])
            reference_yaw = float(route['reference_yaw_rad'])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError('Legacy prompt requires recorded initial and route headings.') from error
        if not math.isfinite(initial_yaw) or not math.isfinite(reference_yaw):
            raise ValueError('Legacy prompt headings must be finite.')
        relative = math.atan2(math.sin(reference_yaw-initial_yaw), math.cos(reference_yaw-initial_yaw))
        suffix = (f'Left and right use the frozen initial route heading: '
                  f'sin {math.sin(relative):.3f}, cos {math.cos(relative):.3f} '
                  f'relative to the initial boat heading.')
    for row in rows:
        if include_phase:
            phase = row.get('phase')
            if not isinstance(phase, str) or not phase:
                raise ValueError('Legacy prompt requires a recorded phase.')
            expected = f'{instruction} Current phase: {phase}. {suffix}'
        else:
            expected = instruction
        if row.get('prompt') != expected:
            raise ValueError('Recorded prompt differs from its route instruction or evaluator context.')
    return instruction


def export(episodes,repo_id,output,*,image_format='original-jpeg'):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    if image_format not in ('png', 'original-jpeg'):
        raise ValueError('Image format must be png or original-jpeg')
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    audits=[]
    instructions=[]
    instruction_provenance=[]
    for directory in episodes:
        result=audit_episode(directory)
        if not result['success']:
            raise ValueError('Initial exporter accepts successful demonstrations only.')
        gaps=result['decision_gap_s']
        if gaps and (gaps[0]<.08 or gaps[1]>.12):
            raise ValueError('Recording is not near 10Hz; resampling must be explicit before export.')
        source_rows=[json.loads(line) for line in (Path(directory)/'steps.jsonl').read_text().splitlines()]
        metadata=json.loads((Path(directory)/'episode.json').read_text())
        instruction, provenance = instruction_for_export(directory, metadata, source_rows)
        instructions.append(instruction)
        instruction_provenance.append(provenance)
        origin=source_rows[0]['decision_sim_time_ns']
        if any(abs((row['decision_sim_time_ns']-origin)/1e9-index/10) > 1e-4
               for index,row in enumerate(source_rows)):
            raise ValueError('Episode is not on the 10Hz grid; refusing implicit timestamp resampling.')
        audits.append(result)
    dataset=LeRobotDataset.create(repo_id=repo_id,root=output,robot_type='usv_differential_thrust',fps=10,
        use_videos=False,image_writer_threads=2 if image_format == 'png' else 0,tolerance_s=.025,features={
            'observation.images.front':{'dtype':'image','shape':(720,1280,3),'names':['height','width','channel']},
            'observation.state':{'dtype':'float32','shape':(33,),'names':['state']},
            'action':{'dtype':'float32','shape':(2,),'names':['left_thrust_N','right_thrust_N']},
        })
    if image_format == 'original-jpeg':
        # LeRobot normally stages each image as PNG and embeds those bytes in
        # parquet. Stage the verified source JPEG bytes instead; the standard
        # datasets.Image decoder still produces the same RGB observation.
        original_image_path = dataset._get_image_file_path
        dataset._get_image_file_path = lambda episode_index, image_key, frame_index: original_image_path(
            episode_index, image_key, frame_index).with_suffix('.jpg')
        jpeg_source = {'path': None}
        def copy_source_image(image, destination):
            if jpeg_source['path'] is None:
                raise ValueError('Missing source JPEG for frame')
            shutil.copyfile(jpeg_source['path'], destination)
        dataset._save_image = copy_source_image
    try:
        for directory, instruction in zip(episodes, instructions):
            directory=Path(directory)
            rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
            for row in rows:
                source_image = directory/row['image']
                with Image.open(source_image) as image:
                    if image_format == 'original-jpeg' and image.format != 'JPEG':
                        raise ValueError(f'Expected original JPEG image: {source_image}')
                    rgb=np.asarray(image.convert('RGB')).copy()
                if image_format == 'original-jpeg':
                    jpeg_source['path'] = source_image
                dataset.add_frame({
                    'observation.images.front':rgb,
                    'observation.state':np.asarray(row['state'],np.float32),
                    'action':np.asarray(row['action'],np.float32),
                    'task':instruction,
                })
            dataset.save_episode()
    finally:
        dataset.stop_image_writer()
    report=dict(repo_id=repo_id,root=str(output),episodes=audits,frames=sum(x['steps'] for x in audits),
                published=False,units='raw state, raw thrust N; normalized once by UsvInputs',
                data_split='none; caller must split whole episodes/layouts before formal training',
                image_storage='original JPEG bytes embedded in parquet' if image_format == 'original-jpeg' else 'PNG encoded in parquet',
                instruction_source='recorded fixed task or validated post-hoc training instruction; evaluator phase and frozen heading excluded',
                training_instructions=instruction_provenance,
                purpose='pipeline verification; small scripted smoke episodes are not a trained VLA')
    (output/'usv_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',nargs='+')
    parser.add_argument('--repo-id',default='local/usv_v1')
    parser.add_argument('--output',required=True)
    parser.add_argument('--image-format',choices=['png','original-jpeg'],default='original-jpeg')
    args=parser.parse_args()
    print(json.dumps(export(args.episodes,args.repo_id,args.output,image_format=args.image_format),indent=2))
