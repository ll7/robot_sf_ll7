"""Static sealed 0.0.8 evaluation commitment; never a simulation entrypoint."""

import hashlib
import json
from collections.abc import Sequence

# Static companion until seed_bands from PR #10039 is present. Equality is tested
# whenever that module is importable; no development run may use these seeds.
SEALED_EVALUATION_SEEDS = (
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


def evaluation_seeds_sha256(seeds: Sequence[int]) -> str:
    """Hash the sorted integer list as compact JSON, retaining duplicates.

    Returns:
        SHA256 commitment to the exact seed list.
    """
    if any(type(seed) is not int for seed in seeds):
        raise ValueError("Evaluation seeds must be integers")
    return hashlib.sha256(json.dumps(sorted(seeds), separators=(",", ":")).encode()).hexdigest()


SEALED_EVALUATION_SEEDS_SHA256 = evaluation_seeds_sha256(SEALED_EVALUATION_SEEDS)
