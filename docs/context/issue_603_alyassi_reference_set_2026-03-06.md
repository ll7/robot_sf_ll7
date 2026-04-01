# Issue 603 / Alyassi Planner Reference Set

Date: 2026-03-06
Related issues:
* `robot_sf_ll7#603` Alyassi-family coverage matrix
* `robot_sf_ll7#601` CrowdNav feasibility spike
* `robot_sf_ll7#599` Go-MPC feasibility spike
* `robot_sf_ll7#602` guarded-PPO safety-aware formalization
* `robot_sf_ll7#600` DSRNN stretch follow-up

## Why this note matters

Treat this page as the canonical paper-side planner reference set for the Alyassi-style external
family anchors. Downstream roadmap notes should point back here when they need the citekeys, 
upstream repositories, or local clone locations.

## Confirmed external planner anchors

### CrowdNav

* Paper citekey: `chenCrowdRobotInteractionCrowdaware2019`
* Public repo: <https://github.com/vita-epfl/CrowdNav>
* Local clone: `/Users/lennart/git/robot_sf_ll7/output/repos/CrowdNav`
* Intended role: first attention-based family anchor.

### CrowdNav DSRNN

* Paper citekey: `liuDecentralizedStructuralRNNRobot2025`
* Public repo: <https://github.com/Shuijing725/CrowdNav_DSRNN.git>
* Local clone: `/Users/lennart/git/robot_sf_ll7/output/repos/CrowdNav_DSRNN`
* Intended role: stretch graph-/attention follow-up.

### Go-MPC

* Paper citekey: `britoWhereGoNext2021`
* Public repo: <https://github.com/tud-amr/go-mpc>
* Local clone: `/Users/lennart/git/robot_sf_ll7/output/repos/go-mpc`
* Intended role: first external prediction-based family anchor.

### Pred2Nav

* Paper citekey: `poddarCrowdMotionPrediction2023`
* Public repo: <https://github.com/sriyash421/Pred2Nav.git>
* Local clone: `/Users/lennart/git/robot_sf_ll7/output/repos/Pred2Nav`
* Intended role: optional stretch follow-up if prediction-family coverage is broadened.

## Safety-aware anchor note

### SoNIC

* Paper citekey: `yaoSoNICSafeSocial2025`
* Public repo: <https://github.com/tasl-lab/SoNIC-Social-Nav.git>
* Local clone: `/Users/lennart/git/robot_sf_ll7/output/repos/SoNIC-Social-Nav`
* Intended role: external safety-aware comparison anchor for guarded-PPO formalization work.

Current provenance caveats:

* Keep `yaoSoNICSafeSocial2025` as the current SoNIC paper anchor.
* The verified project/repo ladder is still missing.
* Reported `arXiv:2511.07820` should not be silently substituted for the current SoNIC anchor
	without verification.
