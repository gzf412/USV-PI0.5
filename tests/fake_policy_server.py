"""TEST ONLY: lightweight protocol stub; no model, checkpoint, or VLA behavior.

Run under the model Python with project/src in PYTHONPATH. It deliberately has
no production launcher. --mode and --delay-s exercise failure/timeout paths.
"""
import argparse
import time

import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed

from openpi_client import msgpack_numpy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=18086)
    parser.add_argument('--delay-s',type=float,default=0.)
    parser.add_argument('--action',type=float,nargs=2,default=[0.,0.])
    parser.add_argument('--mode',choices=['valid','wrong_shape','nonfinite','wrong_units','disconnect'],default='valid')
    args=parser.parse_args()
    metadata=dict(state_dim=33,action_dim=2,action_units='N' if args.mode!='wrong_units' else 'normalized',
                  policy_kind='test_stub_not_vla',checkpoint_source=None)
    def handle(connection):
        packer=msgpack_numpy.Packer()
        connection.send(packer.pack(metadata))
        try:
            for message in connection:
                observation=msgpack_numpy.unpackb(message)
                assert observation['state'].shape==(33,)
                assert observation['images']['front'].ndim==3
                assert isinstance(observation['prompt'],str)
                if args.mode=='disconnect':
                    connection.close()
                    return
                time.sleep(args.delay_s)
                actions=np.tile(np.asarray(args.action,np.float32),(10,1))
                if args.mode=='wrong_shape': actions=actions[:,0]
                if args.mode=='nonfinite': actions[0,0]=np.nan
                connection.send(packer.pack(dict(actions=actions)))
        except ConnectionClosed:
            # The bridge deliberately terminates a blocked worker on timeout.
            return
    print(f'TEST STUB, NOT VLA: ws://{args.host}:{args.port}',flush=True)
    # Real RGB arrives as msgpack ndarray (1280*720*3 > the 1 MiB default).
    # Match the production WebsocketPolicyServer rather than truncating frames.
    with serve(handle,args.host,args.port,max_size=None) as server:
        server.serve_forever()


if __name__=='__main__': main()
