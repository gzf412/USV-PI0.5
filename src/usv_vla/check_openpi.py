"""Offline adapter and abstract model checks. Does not load large weights."""
import os
from pathlib import Path
import tempfile
import unittest

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0_config import Pi0Config
from openpi.policies.usv_policy import UsvInputs, UsvOutputs, normalize_state, parse_image
from openpi.transforms import PadStatesAndActions


class AdapterChecks(unittest.TestCase):
    def test_offline_paligemma_initialization_preserves_new_action_weights(self):
        from openpi.training.weight_loaders import PaliGemmaWeightLoader
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pt_224.npz'
            np.savez(path, **{'params/llm/embedding':np.full((2,3),7.,np.float32)})
            reference = {'PaliGemma':{'llm':{'embedding':np.zeros((2,3),np.float32)}},
                         'action_in_proj':{'kernel':np.full((2,4),5.,np.float32)}}
            loaded = PaliGemmaWeightLoader(weights_path=str(path)).load(reference)
            np.testing.assert_array_equal(loaded['PaliGemma']['llm']['embedding'],7.)
            np.testing.assert_array_equal(loaded['action_in_proj']['kernel'],5.)
            with self.assertRaises(FileNotFoundError):
                PaliGemmaWeightLoader(weights_path=str(path.with_name('missing.npz'))).load(reference)

    def test_real_tokenizer_rejects_truncation_and_range_errors(self):
        from usv_vla.paths import ASSETS
        cache=Path(os.environ.get('USV_TOKENIZER_PATH',str(ASSETS/'tokenizer/paligemma_tokenizer.model')))
        self.assertTrue(cache.is_file(),'Bundled tokenizer must exist; model checks do not silently skip it.')
        from openpi.models.tokenizer import PaligemmaTokenizer
        state=np.zeros(33,np.float32)
        state[-3:]=[.5,-.5,1]
        tokenizer=PaligemmaTokenizer(384,strict=True)
        tokens,mask=tokenizer.tokenize('Pass the red round buoy on its left side.',state)
        text=tokenizer._tokenizer.decode(tokens[mask].tolist())
        encoded=text.split('State: ',1)[1].split(';',1)[0].split()
        self.assertEqual(len(encoded),33)
        self.assertEqual(encoded[-1],'255')
        with self.assertRaises(ValueError): PaligemmaTokenizer(16,strict=True).tokenize('test',state)
        state[0]=1.1
        with self.assertRaises(ValueError): tokenizer.tokenize('test',state)

    def test_lerobot_chw_image_conversion(self):
        image=np.full((3,72,128),.5,np.float32)
        parsed=parse_image(image)
        self.assertEqual(parsed.shape,(72,128,3))
        self.assertEqual(parsed.dtype,np.uint8)
        np.testing.assert_array_equal(parsed,128)
        with self.assertRaises(ValueError): parse_image(np.full((3,72,128),2.,np.float32))

    def test_original_defaults_and_separate_shapes(self):
        self.assertEqual(Pi0Config().state_dim,32)
        config=Pi0Config(pi05=True,state_dim=33,action_dim=2,action_horizon=10,max_token_len=384)
        observation,actions=config.inputs_spec()
        self.assertEqual(observation.state.shape,(1,33))
        self.assertEqual(actions.shape,(1,10,2))
        padded=PadStatesAndActions(2,33)({'state':np.zeros(33),'actions':np.zeros((10,2))})
        self.assertEqual(padded['state'].shape,(33,))
        self.assertEqual(padded['actions'].shape,(10,2))
        with self.assertRaises(ValueError):
            PadStatesAndActions(2,33)({'state':np.zeros(33),'actions':np.zeros((10,3))})

    def test_presence_images_and_motor_units(self):
        state=np.zeros(33,np.float32)
        state[6]=1
        state[9:12]=[-10,5,1]
        state[7:9]=[.2,-.4]
        result=UsvInputs()({'state':state,'images':{'front':np.zeros((72,128,3),np.uint8)},
                            'actions':np.array([[3,-6]],np.float32),'prompt':'test'})
        np.testing.assert_allclose(result['state'][9:12],[-10/32,5/32,1])
        np.testing.assert_array_equal(result['state'][12:],0)
        np.testing.assert_array_equal(result['state'][7:9],state[7:9])
        self.assertEqual(list(result['image_mask'].values()),[True,False,False])
        np.testing.assert_allclose(result['actions'],[[.2,-.4]])
        np.testing.assert_allclose(UsvOutputs()({'actions':result['actions']})['actions'],[[3,-6]])
        np.testing.assert_allclose(UsvOutputs()({'actions':np.array([[2,-2]])})['actions'],[[15,-15]])
        state[12]=1
        with self.assertRaises(ValueError): normalize_state(state)
        with self.assertRaises(ValueError): UsvOutputs()({'actions':np.array([[float('nan'),0]])})

    def test_full_pi05_abstract_loss_and_sampling(self):
        config=Pi0Config(pi05=True,state_dim=33,action_dim=2,action_horizon=10,max_token_len=384)
        # Abstract execution traces the real full-sized architecture without allocating weights.
        model=nnx.eval_shape(config.create,jax.random.key(0))
        self.assertFalse(hasattr(model,'state_proj'))
        self.assertEqual(model.action_in_proj.kernel.shape,(2,1024))
        self.assertEqual(model.action_out_proj.kernel.shape,(1024,2))
        observation,actions=config.inputs_spec()
        def shapes():
            net=config.create(jax.random.key(1))
            obs=jax.tree.map(lambda x:jnp.zeros(x.shape,x.dtype),observation)
            act=jnp.zeros(actions.shape,actions.dtype)
            return net.compute_loss(jax.random.key(2),obs,act),net.sample_actions(jax.random.key(3),obs,num_steps=2)
        loss,samples=jax.eval_shape(shapes)
        self.assertEqual(loss.shape,(1,10))
        self.assertEqual(samples.shape,(1,10,2))

    def test_pi0_compatibility_projection(self):
        model=nnx.eval_shape(Pi0Config(state_dim=33,action_dim=2).create,jax.random.key(0))
        self.assertEqual(model.state_proj.kernel.shape,(33,1024))


if __name__=='__main__': unittest.main(verbosity=2)
