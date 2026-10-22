"""Replay expert inference with the evaluator's frame/batch shape.

The physical trace and original action tolerance are unchanged. This separate
entry leaves frozen evaluation and audit scripts untouched.
"""

import json
from pathlib import Path
import sys

import audit_s09_bc_repair
import audit_s09_wide_actor
from run_s09_mode_experts import ARCHITECTURE, expert_networks, runner
from retarget_s09_reference import digest


if __name__ == "__main__":
    run = Path(sys.argv[sys.argv.index("--run")+1])
    cfg = json.loads((run/"run_config.json").read_text())
    if cfg.get("policy_architecture") != ARCHITECTURE:
        raise ValueError("Require six full-target experts")
    with expert_networks():
        original = runner.initialize_matched
        def initialize(*args,**kwargs):
            net,params = original(*args,**kwargs)
            actor = audit_s09_wide_actor.matched_actor(net.policy_network,cfg["eval_batch"])
            return net.replace(policy_network=actor),params
        runner.initialize_matched = initialize
        try:
            audit_s09_bc_repair.main()
        finally:
            runner.initialize_matched = original
    step = int(sys.argv[sys.argv.index("--step")+1]) if "--step" in sys.argv else 0
    path = run/f"step_{step:09d}"/"native_replay_audit.json"
    record = json.loads(path.read_text())
    record["inference_shape"] = f"scan(vmap(actor)) at evaluation batch {cfg['eval_batch']}; unchanged tolerance"
    record["supplementary_audit_sources"] = {str(p.resolve().relative_to(runner.ROOT)):digest(p)
        for p in (Path(__file__),Path(audit_s09_wide_actor.__file__))}
    path.write_text(json.dumps(record,indent=2)+"\n")
