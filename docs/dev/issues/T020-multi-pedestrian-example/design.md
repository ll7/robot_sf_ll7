# T020: Multi-Pedestrian Example – Design Doc

## Problem Statement
Demonstrate and validate the ability to spawn and simulate multiple single pedestrians (goal-based, trajectory-based, static) in a single scenario, using the new data model and API.

## Solution Overview
- Add `examples/example_multi_pedestrian.py` to show multiple single pedestrians in one map.
- Add `tests/test_multi_pedestrian.py` to validate correct loading and simulation.
- Link example in `docs/README.md`.

## Implementation Details
- Uses `SinglePedestrianDefinition` for each pedestrian.
- Mixes goal, trajectory, and static types.
- MapDefinition and MapDefinitionPool used for scenario setup.
- Example runs a short simulation and prints pedestrian info.

## Impact Analysis
- Demonstrates new API for multi-pedestrian support.
- Validates integration with environment and config.
- No breaking changes; only new example and test.

## Testing Strategy
- Unit test: checks correct loading and types of all pedestrians.
- Smoke test: runs env for 10 steps, asserts no crash.

## Future Considerations
- Extend to support group/crowd definitions.
- Add visualization or video output for demo.

## Related Links
- [example_multi_pedestrian.py](../../../examples/example_multi_pedestrian.py)
- [test_multi_pedestrian.py](../../../tests/test_multi_pedestrian.py)
- [docs/README.md](../../../docs/README.md)
