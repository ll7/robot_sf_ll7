<!-- AI-GENERATED/NEEDS-REVIEW -->
Normal merge of origin/main c979e0337da4ad053d59a76225fbb9154140ee73 into f16a53f5a924e0d50a749ea8f74e30fa4ac2db4a.

Two conflict blocks in hybrid_rule_local_planner.py:
- Realized-rollout initialization (former line 935): retain main’s _min_linear_speed(max_speed) floor and HYBDIAG’s supplied current_angular, with the prior estimate only as fallback.
- Wall stopping sweep / reverse bound (former line 1033): retain both the lane’s wall stopping method and main’s _min_linear_speed method. The reverse stopping defect is addressed separately, with a counterproof on the merged tree after platform removal.

No rebase and no whole-file ours/theirs resolution. tests/conftest.py merged automatically.
