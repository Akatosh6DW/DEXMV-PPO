import unittest

import jax
import jax.numpy as jp
import numpy as np

from probe_s09_temporal_noise import noise_at,noise_key


class TemporalNoiseTests(unittest.TestCase):
    def test_repeat_and_seed_reproducibility(self):
        key=noise_key(250000,0)
        samples=np.asarray(jax.vmap(lambda i:noise_at(key,i,4))(jp.arange(8)))
        np.testing.assert_array_equal(samples[:4],np.broadcast_to(samples[0],(4,17)))
        np.testing.assert_array_equal(samples[4:],np.broadcast_to(samples[4],(4,17)))
        self.assertFalse(np.array_equal(samples[0],samples[4]))
        self.assertFalse(np.array_equal(noise_at(key,0,1),noise_at(noise_key(250000,1),0,1)))
        np.testing.assert_array_equal(noise_at(key,4,4),noise_at(key,1,1))

    def test_equal_marginal_amplitude(self):
        keys=jax.random.split(jax.random.PRNGKey(7),8192)
        independent=np.asarray(jax.vmap(lambda k:noise_at(k,0,1))(keys))*.03
        held=np.asarray(jax.vmap(lambda k:noise_at(k,0,4))(keys))*.03
        np.testing.assert_array_equal(independent,held)
        np.testing.assert_allclose(independent.std(axis=0),.03,atol=.001)


if __name__=="__main__":
    unittest.main()
