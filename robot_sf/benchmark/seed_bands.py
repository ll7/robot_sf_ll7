"""Author-approved release seed bands. Importable without the simulation stack.

Both evaluation bands remain sealed for development, tuning and calibration.
The YAML schedule is a static transport copy, checked against this module.
"""

EVAL_SEEDS_0_0_8 = (
    50036,
    50140,
    50331,
    50403,
    50813,
    51339,
    51709,
    51767,
    52094,
    52175,
    52257,
    52671,
    52850,
    52971,
    53020,
    53198,
    53239,
    53636,
    53671,
    53779,
    55022,
    55379,
    55568,
    56170,
    56966,
    57077,
    57113,
    57494,
    57943,
    59019,
)
RETIRED_EVAL_SEEDS_0_0_7 = tuple(range(111, 141))
DEV_SEEDS = tuple(range(1001, 1031))
EVAL_SEEDS_DERIVATION_LABEL = "robot_sf_ll7 release 0.0.8 evaluation seeds v1 (sealed 2026-09-30)"
EVAL_SEEDS_DERIVATION_SHA256 = "166597da1e0e813d8a9cdc810f4b85db1407e9286c3113821423e17c50908dc0"
HELD_OUT_SEEDS = frozenset(RETIRED_EVAL_SEEDS_0_0_7 + EVAL_SEEDS_0_0_8)
EVAL_SEED_SET_0_0_8 = "release_eval_0_0_8"
