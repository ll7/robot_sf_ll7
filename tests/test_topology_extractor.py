import json
from scripts.analysis.extract_topology_scores import extract_topology

def test_topology_extractor_fields():
    with open("tests/fixtures/topology_sample.jsonl") as f:
        rec = json.loads(f.readline())
    out = extract_topology(rec)
    keys = ["per_frame_hypothesis_count","alternative_hypothesis_count","selected_hypothesis","rejection_reason","score_margin_to_primary_route","switch_opportunity_count"]
    for k in keys:
        assert k in out
    assert out["selected_hypothesis"] == "primary_route"
