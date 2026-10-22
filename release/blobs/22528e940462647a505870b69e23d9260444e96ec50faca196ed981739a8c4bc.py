"""S03 shared task: explicit training-only shaping and held-out push settings."""

import math

from orca_suite_env import OrcaSuiteGrasp


class OrcaS03Grasp(OrcaSuiteGrasp):
    def __init__(self, model, arm, reference, *, shaping=1., push_force=None, **kwargs):
        if not math.isfinite(shaping) or not 0 <= shaping <= 3:
            raise ValueError("Invalid acquisition shaping multiplier")
        if push_force is not None and (not math.isfinite(push_force) or not 0 <= push_force <= 2):
            raise ValueError("Invalid horizontal force")
        super().__init__(model, arm, reference, **kwargs)
        self.shaping = float(shaping)
        if push_force is not None:
            self.settings["push_force"] = float(push_force)
