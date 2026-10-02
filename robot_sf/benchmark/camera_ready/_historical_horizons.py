"""Exact immutable historical campaign content admitted by D-064.

Versions identify the historical protocol, not the date of a later rerun.
Names are documentation only; admission uses the complete source byte digest.
runtime_smoke_v0_4 belongs to the 0.0.8 cycle and is deliberately absent.
"""

from types import MappingProxyType

HISTORICAL_CAMPAIGN_REGISTRY = MappingProxyType(
    {
        # configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_roster.yaml
        "d4425b0e381160bc750c5341a7d4ed475eb9a202dd78aaaca7ae876343292132": (
            "0.0.2",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml
        "82eee7a1dca241df9bec65e18072f2dd3c46c86489b3b55816d2504af88896b6": (
            "0.0.2",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/paper_experiment_matrix_v1_h600_trace_capable_rerun.yaml
        "280bff07464103b4a2448702c5eaca036f8a01f320639df521d2350f0e6287c4": (
            "0.0.2",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended_post1.yaml
        "c43d7bc24a182dcc56082f4da11d76e09a66d8b1cc0d0dc84b29536726022701": (
            "0.0.3.post1",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml
        "aa3057faeeefbd2ced41e3e093da32d1705270330cb1c124bc0b3558f8f88afd": (
            "0.0.7",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml
        "c3671790d0beb12511223efa86e2cf26245692566b2c849dba956b6e36bdf64a": (
            "0.0.7",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_3.yaml
        "fbd900243f5a004cc07f7d10c672126f46ec583eb6f108ec7a0e8fce9daa7ad4": (
            "0.0.7",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/paper_experiment_matrix_v2_h600_hybrid_stress_smoke.yaml
        # Old digest: 1201cc1c5d374659e045071d035e87cb0a3684d8dcfd172f149e127d9102d455
        # retired seed 116 -> dev 1001, #10080 commit 4d9dd1ec0
        "c088e62e7fb7758dc2670d81600d54c7de903ebf2d6e9310cf6e672c53400184": (
            "0.0.7",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/issue_5756_trace90_ppo.yaml
        "466dbae50b811e9993dce9a2fa0ec67f64f50d82de3c316e2cad6d043767926e": (
            "0.0.3.post1",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/issue_5756_trace90_ppo_canary.yaml
        "4f65904020b4468a8e34bb24fb07aad4e61317345a052ce9a69b98cc969920f6": (
            "0.0.3.post1",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/issue_5756_trace90_goal.yaml
        "68a1ab34a693de92d195e28b7539ddc291e27d17f0fed80a40750b108651a034": (
            "0.0.3.post1",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/issue_6642_radius_sweep_arm_1p0m.yaml
        "b4855e6ec51729a35ee87af487ad8c3154ccb32a0691b0e1f9d92d5186321a5b": (
            "0.0.3.post1",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/issue_6642_radius_sweep_arm_0p8m.yaml
        "ac4006522e77a02306cb34064bc11a151e56fcd2f1f9409bcdb50440c436dda3": (
            "0.0.3.post1",
            "legacy_runner_cap",
        ),
        # configs/benchmarks/issue_6642_radius_sweep_arm_0p5m.yaml
        "bbf309868951cff5f4bb299acbcc3b01f2e37bd54ef72fb138e5b7d99073883f": (
            "0.0.3.post1",
            "legacy_runner_cap",
        ),
    }
)
