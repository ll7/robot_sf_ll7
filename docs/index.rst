Robot SF Documentation
======================

This is a lightweight Sphinx navigation layer over the existing Robot SF Markdown docs. It keeps the
repository docs as the source of truth and provides a local browsable site for quick orientation,
planner and scenario discovery, benchmark surfaces, evidence policy, agent workflow, and developer
references.

.. toctree::
   :caption: Overview
   :maxdepth: 1

   Docs Index <README>
   Repository Overview <ai/repo_overview>
   Maintainer Values <maintainer_values>

.. toctree::
   :caption: Quickstart
   :maxdepth: 1

   Quickstart Map <quickstart>
   Development Guide <dev_guide>
   Runtime Requirements <dev_runtime_requirements>
   External Data Setup <external_data_setup>
   ETH/UCY External Trajectory Data <datasets/eth-ucy>

.. toctree::
   :caption: Scenario Zoo
   :maxdepth: 1

   Scenario Zoo Index <scenario_zoo/index>
   Scenario Contracts <scenario_contracts>
   Scenario Certification <scenario_certification>
   Scenario Perturbation Manifest <scenario_perturbation_manifest>

.. toctree::
   :caption: Planner Zoo
   :maxdepth: 1

   Planner Zoo Index <planner_zoo/index>
   Planner Contribution Guide <contributing_planner>
   Policy Search Portfolio <context/policy_search/portfolio_overview_2026-05-05>
   Candidate Registry Summary <context/policy_search/candidate_registry_summary>

.. toctree::
   :caption: Benchmark Suites
   :maxdepth: 1

   Benchmark Suites Map <benchmark_suites>
   Benchmark Runner And Metrics <benchmark>
   Benchmark Spec <benchmark_spec>
   Camera-Ready Benchmark Workflow <benchmark_camera_ready>
   Static Leaderboards <leaderboards/README>

.. toctree::
   :caption: Learned Policy Integration
   :maxdepth: 1

   Learned Policy Registry <context/policy_search/learned_policy_registry>
   Learned Policy Eligibility <context/policy_search/contracts/learned_local_policy_eligibility>
   External Policy Intake <context/policy_search/contracts/external_policy_intake>
   Policy Cards <policy_cards/README>

.. toctree::
   :caption: Evidence And Artifacts
   :maxdepth: 1

   Evidence Bundles <context/evidence/README>
   Artifact Publication <benchmark_artifact_publication>
   Model Registry Publication <model_registry_publication>
   Benchmark Release Protocol <benchmark_release_protocol>

.. toctree::
   :caption: Agent Workflow
   :maxdepth: 1

   AI Coding Workflow <ai/ai-workflow>
   Agent Index <AGENT_INDEX>
   Context Notes Workflow <context/README>
   Context Retrieval Index <context/INDEX>

.. toctree::
   :caption: Developer/API Reference
   :maxdepth: 1

   Developer Guide <dev_guide>
   API Reference <api/index>
   Environment API <ENVIRONMENT>
   Reward Profiles Reference <training/reward_profiles>
   Observation Contract <dev/observation_contract>
   Helper Catalog <dev/helper_catalog>
   Code Review Guide <code_review>
