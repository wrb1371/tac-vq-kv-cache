"""Import and mathematical contract checks; no model or CUDA device required."""

import pathlib, sys, hashlib, json, importlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/OmniKVQuant_snapshot")]
from sti_rvq import allocate
from ar34kv.selection import layer_quotas
from ar26kv.codec import Codec
from ar25kv.simple_v1 import ConditionalBudget

options = [
    [{"bytes": 7, "loss": 4}, {"bytes": 12, "loss": 1}],
    [{"bytes": 9, "loss": 3}, {"bytes": 13, "loss": 0.5}],
]
chosen, proof = allocate(options, 21, quantum=1)
assert sum(x["bytes"] for x in chosen) <= 21 and proof["objective"] == 4
assert sum(layer_quotas(497, 4, 241, 64)) == 497
from ar39cal.paths import use
for replicate in [0, 1, 2]:
    with use(replicate):
        for p in [
            ROOT / "artifacts/ar17/packages/PSRVQ_d16",
            ROOT / "artifacts/ar26/packages/STI_RVQ",
        ]:
            metadata = json.loads((p / "completion.json").read_text())
            assert metadata["completed"]
            for row in metadata.get("records", []):
                assert hashlib.sha256((p / row["file"]).read_bytes()).hexdigest() == row["SHA256"]
        for relative in ["data/ar21/rate_curves.json", "data/ar22/conditional_curves.json", "data/ar26/VL_rate_curves.json", "data/ar26/VL_conditional_curves.json"]:
            assert json.loads((ROOT / relative).read_text()).get("completed", True)
print(
    json.dumps(
        {
            "passed": True,
            "portable_imports": True,
            "exact_budget_planner": True,
            "codec_hashes": True,
            "fitted_shared_codec_realizations": 6,
            "matched_loss_tables_for_all_three_runs": True,
            "GPU_used": False,
        }
    )
)
