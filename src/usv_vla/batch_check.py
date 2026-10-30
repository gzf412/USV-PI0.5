"""Read a local LeRobot dataset through this project's model adapter."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

import numpy as np
from usv_vla.paths import OUTPUTS, PROJECT_ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id',default='local/usv_v1')
    parser.add_argument('--data-home',type=Path)
    parser.add_argument('--report',type=Path,default=OUTPUTS/'checks/batch.json')
    args=parser.parse_args()
    if args.data_home:
        os.environ['HF_LEROBOT_HOME']=str(args.data_home.expanduser().resolve())
    from openpi.training.config import get_config
    from openpi.training.data_loader import create_data_loader,create_torch_dataset,transform_dataset
    config=dataclasses.replace(get_config('pi05_usv'),batch_size=2,num_workers=0)
    config=dataclasses.replace(config,data=dataclasses.replace(config.data,repo_id=args.repo_id))
    data=config.data.create(config.assets_dirs,config.model)
    raw=create_torch_dataset(data,config.model.action_horizon,config.model)
    transformed=transform_dataset(raw,data)
    lengths=[]
    metadata=raw._dataset.meta
    indices={0,len(raw)-1}
    base=raw._dataset
    for start,end in zip(base.episode_data_index['from'],base.episode_data_index['to']):
        indices.update([int(start),int(end)-1])
    for index in sorted(indices):
        row=transformed[index]
        assert row['state'].shape==(33,)
        assert row['actions'].shape==(10,2)
        lengths.append(int(row['tokenized_prompt_mask'].sum()))
    observation,actions=next(iter(create_data_loader(config,num_batches=1)))
    source_root=(PROJECT_ROOT/'src').resolve()
    module_sources={}
    for name,module in sys.modules.items():
        if name=='openpi' or name.startswith(('openpi.','openpi_client.')) or name=='openpi_client':
            filename=getattr(module,'__file__',None)
            if filename:
                path=Path(filename).resolve()
                if not path.is_relative_to(source_root):
                    raise RuntimeError(f'Unexpected external model code: {name} from {path}')
                module_sources[name]=str(path.relative_to(PROJECT_ROOT))
    report=dict(repo_id=args.repo_id,dataset_frames=len(raw),dataset_episodes=metadata.total_episodes,
        state_shape=list(observation.state.shape),action_shape=list(actions.shape),
        image_shapes={key:list(value.shape) for key,value in observation.images.items()},
        active_views={key:np.asarray(value).tolist() for key,value in observation.image_masks.items()},
        checked_token_lengths=lengths,token_limit=config.model.max_token_len,
        action_range=[float(np.asarray(actions).min()),float(np.asarray(actions).max())],
        large_weights_loaded=False,all_model_code_is_local=True,module_sources=module_sources)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    compact={key:value for key,value in report.items() if key!='module_sources'}
    print(json.dumps(compact,indent=2))


if __name__=='__main__': main()
