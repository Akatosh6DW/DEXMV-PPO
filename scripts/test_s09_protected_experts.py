import unittest

import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.training.agents.ppo.losses import PPONetworkParams

from run_s09_constrained_ppo import split_adam
from run_s09_protected_experts import protect_gradients, audit_frozen


class ProtectedExpertTests(unittest.TestCase):
    def test_adam_scope_and_critic(self):
        params = PPONetworkParams(policy=dict(experts=tuple(dict(w=jp.ones(3)) for _ in range(6))),value=dict(w=jp.ones(4)))
        a = split_adam(3e-6,3e-4)
        b = optax.chain(protect_gradients((0,1,3)),split_adam(3e-6,3e-4))
        sa,sb = a.init(params),b.init(params)
        for scale in (1.,-2.,.5):
            grad = jax.tree_util.tree_map(lambda x:x*scale,params)
            da,sa = a.update(grad,sa)
            db,sb = b.update(grad,sb)
            np.testing.assert_array_equal(da.value["w"],db.value["w"])
            for i in range(6):
                np.testing.assert_array_equal(db.policy["experts"][i]["w"],
                    0 if i in (0,1,3) else da.policy["experts"][i]["w"])

    def test_invalid_indices(self):
        for indices in ([],[6],[0,0]):
            with self.assertRaises(ValueError):
                protect_gradients(indices)

    def test_frozen_audit_rejects_changes(self):
        before = (None,dict(experts=tuple(dict(w=np.ones(3)) for _ in range(6))))
        after = (None,dict(experts=tuple(dict(w=np.ones(3)+(i==2)) for i in range(6))))
        self.assertEqual(audit_frozen(before,after,(0,1,3))[2]["changed_parameters"],3)
        after[1]["experts"][0]["w"][0] += 1
        with self.assertRaises(AssertionError):
            audit_frozen(before,after,(0,1,3))


if __name__ == "__main__":
    unittest.main()
