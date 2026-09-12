Public API Facade (``robot_sf.api``)
====================================

The ``robot_sf.api`` module implements the high-level, lightweight public facade for Robot SF.
It provides ergonomic entry points for scenario loading, environment construction,
and headless episode rollouts with :class:`~robot_sf.benchmark.types.EpisodeRecord` persistence.

For top-level export inventory and stability guarantees, see :doc:`Robot SF Public API </public_api>`.

Runnable Examples
-----------------

The following examples demonstrate the core lifecycle and execution workflows using only
the supported top-level public API exports.

Example 1: Environment Creation and Lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a simulation environment with a deterministic seed and ensure its resources
are released using a ``try/finally`` block:

.. testcode::

   import robot_sf

   # Create a default simulation environment with a fixed seed
   env = robot_sf.make_env(seed=42)
   try:
       obs, info = env.reset(seed=42)
       assert obs is not None
       assert isinstance(info, dict)
   finally:
       env.close()

Example 2: Safe Scenario Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load and inspect a scenario specification from the repository scenario tree:

.. testcode::

   import robot_sf

   # Load a standard circular crossing scenario by name
   scenario = robot_sf.load_scenario("francis2023_circular_crossing")
   assert scenario["name"] == "francis2023_circular_crossing"
   assert "map" in scenario or "map_file" in scenario

Example 3: Episode Rollout and Persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute a short headless episode, serialize the resulting :class:`~robot_sf.benchmark.types.EpisodeRecord`
to disk, and reload it to verify fidelity:

.. testcode::

   import tempfile
   from pathlib import Path
   import robot_sf

   with tempfile.TemporaryDirectory() as tmp_dir:
       out_path = Path(tmp_dir) / "demo_episode.json"

       # Initialize environment with the loaded scenario
       env = robot_sf.make_env(scenario="francis2023_circular_crossing", seed=42)
       try:
           # Run a short 3-step deterministic episode
           record = robot_sf.run_episode(env, max_steps=3, seed=42)
           assert record.seed == 42
           assert record.horizon == 3
           assert record.metrics.get("steps") == 3.0

           # Persist to disk and reload
           record.save(out_path)
           assert out_path.is_file()

           loaded = robot_sf.EpisodeRecord.load(out_path)
           assert loaded.episode_id == record.episode_id
           assert loaded.seed == record.seed
           assert loaded.metrics.get("steps") == record.metrics.get("steps")
           assert loaded.metrics.to_dict() == record.metrics.to_dict()
       finally:
           env.close()

Error Handling Contracts
------------------------

The public API fails closed with descriptive exceptions under invalid configurations or missing assets.

Missing Scenario Assets
~~~~~~~~~~~~~~~~~~~~~~~

Source checkouts bundle the ``configs/scenarios/`` hierarchy. In deployment modes where
that asset tree is not installed or when an unknown scenario is requested, ``load_scenario``
fails closed with an actionable :exc:`FileNotFoundError`:

.. testcode::

   import robot_sf

   try:
       robot_sf.load_scenario("non_existent_scenario_name_123")
       assert False, "Expected FileNotFoundError"
   except FileNotFoundError as exc:
       assert "could not be resolved" in str(exc)

Invalid Planner Objects
~~~~~~~~~~~~~~~~~~~~~~~

``run_episode`` validates the planner argument. If an invalid planner object is passed
(neither callable nor implementing a callable ``step()`` method), it raises :exc:`TypeError`
rather than failing silently or substituting unverified behavior:

.. testcode::

   import robot_sf

   env = robot_sf.make_env(seed=42)
   try:
       try:
           robot_sf.run_episode(env, planner=object(), max_steps=1)
           assert False, "Expected TypeError"
       except TypeError as exc:
           assert "must provide a callable step() method or be callable" in str(exc)
   finally:
       env.close()

Invalid Step Limits
~~~~~~~~~~~~~~~~~~~

``max_steps`` must be a positive integer when provided; non-integer or non-positive bounds
are rejected fail-closed:

.. testcode::

   import robot_sf

   env = robot_sf.make_env(seed=42)
   try:
       try:
           robot_sf.run_episode(env, max_steps=-5)
           assert False, "Expected ValueError"
       except ValueError as exc:
           assert "max_steps must be a positive integer" in str(exc)
   finally:
       env.close()

Module Reference
----------------

.. automodule:: robot_sf.api
   :members:
   :undoc-members:
   :show-inheritance:
