"""Fixed BC-head actuator correlations, with likelihood-consistent PPO.

Only exploration covariance changes. Every step draws fresh noise. This borrows
the cross-actuator motivation of Lattice, not its full latent/time algorithm.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jp
import jax.scipy as jsp
import numpy as np
from brax.training.distribution import NormalTanhDistribution

import run_s09_constrained_ppo as constrained
from retarget_s09_reference import digest


def head_correlation(kernel, strength):
    w = np.asarray(kernel, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 34 or not np.isfinite(w).all():
        raise ValueError("Expected finite full-target 34-output head")
    if not np.isfinite(strength) or not 0 <= strength <= .9:
        raise ValueError("Correlation strength must lie in [0,.9]")
    gram = w[:, :17].T @ w[:, :17]
    scale = np.sqrt(np.diag(gram))
    valid = scale > 1e-10
    denom = np.outer(np.where(valid,scale,1.), np.where(valid,scale,1.))
    corr = np.where(valid[:,None] & valid[None,:], gram/denom, 0.)
    corr = (corr + corr.T) * .5
    np.fill_diagonal(corr, 1.)
    result = (1-strength)*np.eye(17) + strength*corr
    assert np.linalg.eigvalsh(result).min() >= 1-strength-1e-10
    return result


class FullNormal:
    def __init__(self, loc, scale, correlation, factor):
        self.loc, self.scale = loc, scale
        self.cov = scale[..., :, None] * correlation * scale[..., None, :]
        self.tril = scale[..., :, None] * factor

    def sample(self, seed):
        noise = jax.random.normal(seed, shape=self.loc.shape)
        return self.loc + jp.einsum("...ij,...j->...i", self.tril, noise)

    def mode(self):
        return self.loc

    def log_prob(self, x):
        return jsp.stats.multivariate_normal.logpdf(x, self.loc, self.cov)

    def entropy(self):
        return .5*self.loc.shape[-1]*(1+jp.log(2*jp.pi)) + jp.log(
            jp.diagonal(self.tril, axis1=-2, axis2=-1)).sum(-1)

    def kl_divergence(self, old):
        # Brax calls NEW.kl_divergence(OLD), meaning KL(OLD || NEW).
        transform = jsp.linalg.solve_triangular(self.tril, old.tril, lower=True)
        shift = jsp.linalg.solve_triangular(
            self.tril, (old.loc-self.loc)[..., None], lower=True)[..., 0]
        logdet = (jp.log(jp.diagonal(self.tril, axis1=-2, axis2=-1)) -
                  jp.log(jp.diagonal(old.tril, axis1=-2, axis2=-1))).sum(-1)
        return logdet + .5*((transform**2).sum((-2,-1)) + (shift**2).sum(-1)-self.loc.shape[-1])


class CorrelatedTanh(NormalTanhDistribution):
    def __init__(self, correlation):
        matrix = np.asarray(correlation, dtype=np.float64)
        if matrix.shape != (17,17) or not np.isfinite(matrix).all():
            raise ValueError("Expected a finite 17x17 correlation matrix")
        if not np.allclose(matrix, matrix.T, atol=1e-10) or not np.allclose(np.diag(matrix), 1.):
            raise ValueError("Require a symmetric unit-diagonal correlation matrix")
        factor = np.linalg.cholesky(matrix)
        super().__init__(17)
        self.correlation, self.factor = jp.asarray(matrix, jp.float32), jp.asarray(factor, jp.float32)

    def create_dist(self, parameters):
        loc, raw = jp.split(parameters, 2, axis=-1)
        return FullNormal(loc, jax.nn.softplus(raw)+self._min_std, self.correlation, self.factor)

    def log_prob(self, parameters, actions):
        # Multivariate log_prob already sums the event; sum only the tanh Jacobian.
        return self.create_dist(parameters).log_prob(actions) - self._postprocessor.forward_log_det_jacobian(actions).sum(-1)

    def entropy(self, parameters, seed):
        dist = self.create_dist(parameters)
        return dist.entropy() + self._postprocessor.forward_log_det_jacobian(dist.sample(seed)).sum(-1)


def wrap_network(network, matrix):
    original = network.parametric_action_distribution
    if type(original) is not NormalTanhDistribution or original._min_std != .001 or original._var_scale != 1:
        raise ValueError("Require the original NormalTanh scale semantics")
    return network.replace(parametric_action_distribution=CorrelatedTanh(matrix))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--correlation-strength", type=float, required=True)
    args, rest = parser.parse_known_args()
    if rest[rest.index("--method")+1] != "guided" or "--checkpoint" not in rest:
        parser.error("This isolated group starts from an explicit full-target BC")
    if any(flag in rest for flag in ("--teacher-kl", "--dataset", "--ppo-initial-std")):
        parser.error("Do not add KL, demonstrations, or an exploration-scale override")
    source = Path(rest[rest.index("--checkpoint")+1])
    cfg = json.loads((source.parent.parent/"run_config.json").read_text())
    if cfg["steps"] != 0:
        parser.error("Use a BC-only source, not a selected PPO checkpoint")
    source_hash = digest(source)
    params = pickle.loads(source.read_bytes())
    matrix = head_correlation(params[1]["params"]["hidden_2"]["kernel"], args.correlation_strength)
    output = Path(rest[rest.index("--output")+1])
    runner = constrained.improvement.runner
    original_train = runner.train.train

    def train(*values, **kwargs):
        factory = kwargs["network_factory"]
        kwargs["network_factory"] = lambda *a, **kw: wrap_network(factory(*a, **kw), matrix)
        record = dict(correlation_strength=args.correlation_strength, correlation_matrix=matrix.tolist(),
            independent_zero_head_columns=np.flatnonzero(np.linalg.norm(np.asarray(
                params[1]["params"]["hidden_2"]["kernel"],dtype=float)[:,:17],axis=0)<=1e-10).tolist(),
            eigenvalues=np.linalg.eigvalsh(matrix).tolist(), source_checkpoint=str(source.resolve()),
            source_checkpoint_sha256=source_hash, covariance="diag(softplus(raw_std)+.001) @ R @ diag(softplus(raw_std)+.001)",
            matrix_origin="normalized Gram matrix of frozen BC mean-output head, shrunk toward identity",
            correlation_fixed_during_training=True, independently_resampled_every_step=True,
            initial_marginal_std_unchanged=True, sampling_and_loss_same_distribution=True,
            mean_network_and_critic_unchanged=True, deterministic_deployment_needs_wrapper=False,
            reward="S09-R0 unchanged", source_sha256=digest(output/"sources"/Path(__file__).name),
            caveat="Not full Lattice; no time correlation, no latent-feature-dependent covariance; full-target BC+PPO, not residual")
        (output/"correlated_exploration_config.json").write_text(json.dumps(record, indent=2)+"\n")
        return original_train(*values, **kwargs)

    runner.train.train = train
    sys.argv = [sys.argv[0], *rest]
    try:
        constrained.main()
    finally:
        runner.train.train = original_train
        assert digest(source) == source_hash


if __name__ == "__main__":
    main()
