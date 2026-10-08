<!-- AI-GENERATED (robot_sf#10104) - NEEDS-REVIEW -->

Historical diagnostic summary; raw JSON and driver bytes are preserved in
[external custody](context/evidence/issue_10101_contact_review/custody.json).
JSON links below resolve to that checksum manifest. Historical source results
are separate from the current [populated probe](context/evidence/issue_10101_contact_review/populated_summary.json); neither establishes acceptance.

<!-- AI-GENERATED (#10101) - NEEDS-REVIEW -->
Historical Round 2 evidence. Superseded by [Round 3 corrections and current results](pedcontact_10101.md); the old two-sided V6 ranking and whole-crowd-fallback results are withdrawn.
AI-GENERATED / NEEDS-REVIEW

# Full dev CALFIT comparison — Round 2

Producer ffdd8618b01c9bcba3272771aa97c4b8d8724a1d. Thirty paired dev seeds 1001–1030. Intervals are two-sided 95% Student intervals over dev episodes; V6 spread reflects tiny initial-geometry perturbations, and does not quantify human-reference or threshold uncertainty. PASS concerns the declared mean range, separately from physical validity. MISSING is a failing required check with insufficient observed samples; censored V3 trials are not passes.

## off

Overlap pair-steps 91665533; wall-penetration ped-steps 0; fallback 0; unresolved 0; over-cap samples 0; max speed 2 m/s. Cold-inclusive integration mean 7.4 ms/step; includes all five V6 baselines.

| Item | Value | 95% interval | n | Status | Accepted range |
|---|---|---|---|---|---|
| V1/native | 1.25510791993694 | [1.1825575631024041, 1.3276582767714757] | 30 | PASS | [1.1, 1.48] |
| V2/0.61 | None | None | 0 | MISSING | [0.06, 0.24] |
| V2/0.788 | None | None | 0 | MISSING | [0.06, 0.24] |
| V2/0.966 | None | None | 0 | MISSING | [0.06, 0.24] |
| V3/0.8 | None | None | 0 | MISSING | [1.2880000000000003, 1.932] |
| V3/0.9 | None | None | 0 | MISSING | [1.4880000000000002, 2.232] |
| V3/1.0 | None | None | 0 | MISSING | [1.52, 2.28] |
| V3/1.1 | 0.8486352139661273 | None | 1 | MISSING | [1.544, 2.316] |
| V3/1.2 | 0.821863649808436 | None | 1 | MISSING | [1.576, 2.364] |
| V5/diagnostic | 0.8678063631088293 | [0.8634744250734203, 0.8721383011442383] | 30 | FAIL | [0.4, 0.6] |
| V6/1.15 | 3.158747087331176 | [3.1584385741716794, 3.1590556004906727] | 30 | FAIL | [1.6800000000000002, 2.52] |
| V6/1.42 | 3.5998591829471547 | [3.5995649968495464, 3.600153369044763] | 30 | FAIL | [1.92, 2.88] |
| V6/1.78 | 4.274831502270387 | [4.2745483781629625, 4.275114626377811] | 30 | FAIL | [2.16, 3.24] |
| V4/all_data_flow_persons_s width slope | 1.732367091175001 | [1.7155175391295445, 1.7492166432204577] | 30 | FAIL | [1.8399999999999999, 2.76] |
| V4/steady_flow_persons_s width slope | 2.6448266509117504 | [2.619368086784022, 2.670285215039479] | 30 | PASS | [2.0, 3.0] |

Constituent protocol measurements (V4 widths contribute to the slope checks; V2 passed_n measures physical traversal):

| Case | Mean | 95% interval | Observed n | Traversal | Integration ms/step | Overlap pair-steps | Wall ped-steps |
|---|---|---|---|---|---|---|---|
| V1/native | 1.25510791993694 | [1.1825575631024041, 1.3276582767714757] | 30 | n/a | 32.8229292180622 | 0 | 0 |
| V2/0.61 | None | None | 0 | 0/30 | 0.09483268077795705 | 0 | 0 |
| V2/0.788 | None | None | 0 | 0/30 | 0.09402089043675611 | 0 | 0 |
| V2/0.966 | None | None | 0 | 0/30 | 0.09369742168734471 | 0 | 0 |
| V3/0.8 | None | None | 0 | n/a | 2.0066262161756945 | 5433802 | 0 |
| V3/0.9 | None | None | 0 | n/a | 0.9803744772846888 | 5553529 | 0 |
| V3/1.0 | None | None | 0 | n/a | 0.9555557230632985 | 19993292 | 0 |
| V3/1.1 | 0.8486352139661273 | None | 1 | n/a | 0.8974203625693917 | 1456959 | 0 |
| V3/1.2 | 0.821863649808436 | None | 1 | n/a | 0.9057347391644726 | 1443779 | 0 |
| V4/2.4 | 1.9722426300070977 | [1.9553746591270995, 1.9891106008870958] | 30 | n/a | 23.040767046399804 | 12050605 | 0 |
| V4/3.0 | 1.9042479152217298 | [1.885169704392187, 1.9233261260512726] | 30 | n/a | 23.514853456348646 | 11720767 | 0 |
| V4/3.6 | 1.8155223350820922 | [1.798956582153975, 1.8320880880102093] | 30 | n/a | 23.706118323874154 | 11549155 | 0 |
| V4/4.4 | 1.7006680075837217 | [1.678071090908476, 1.7232649242589675] | 30 | n/a | 23.837822208641835 | 11334089 | 0 |
| V4/5.0 | 1.5966627622629146 | [1.5750176334101362, 1.618307891115693] | 30 | n/a | 23.909646771697833 | 11129532 | 0 |
| V4/all_data_flow_persons_s width slope | 1.732367091175001 | [1.7155175391295445, 1.7492166432204577] | 30 | n/a | None | n/a | n/a |
| V4/steady_flow_persons_s width slope | 2.6448266509117504 | [2.619368086784022, 2.670285215039479] | 30 | n/a | None | n/a | n/a |
| V5/diagnostic | 0.8678063631088293 | [0.8634744250734203, 0.8721383011442383] | 30 | n/a | 0.903261164505966 | 0 | 0 |
| V6/1.15 | 3.158747087331176 | [3.1584385741716794, 3.1590556004906727] | 30 | n/a | 0.08879275647578416 | 0 | 0 |
| V6/1.42 | 3.5998591829471547 | [3.5995649968495464, 3.600153369044763] | 30 | n/a | 0.08932835809161849 | 0 | 0 |
| V6/1.78 | 4.274831502270387 | [4.2745483781629625, 4.275114626377811] | 30 | n/a | 0.0891954293142377 | 24 | 0 |

## on

Overlap pair-steps 0; wall-penetration ped-steps 0; fallback 3461; unresolved 0; over-cap samples 0; max speed 2 m/s. Cold-inclusive integration mean 15.776 ms/step; includes all five V6 baselines.

| Item | Value | 95% interval | n | Status | Accepted range |
|---|---|---|---|---|---|
| V1/native | 1.25510791993694 | [1.1825575631024041, 1.3276582767714757] | 30 | PASS | [1.1, 1.48] |
| V2/0.61 | 0.04673000573467526 | [0.043648556528135755, 0.04981145494121477] | 30 | FAIL | [0.06, 0.24] |
| V2/0.788 | 0.0003749862198652217 | [-5.2683677099879566e-05, 0.0008026561168303229] | 30 | FAIL | [0.06, 0.24] |
| V2/0.966 | 0.0 | [0.0, 0.0] | 30 | FAIL | [0.06, 0.24] |
| V3/0.8 | None | None | 0 | MISSING | [1.2880000000000003, 1.932] |
| V3/0.9 | None | None | 0 | MISSING | [1.4880000000000002, 2.232] |
| V3/1.0 | None | None | 0 | MISSING | [1.52, 2.28] |
| V3/1.1 | None | None | 0 | MISSING | [1.544, 2.316] |
| V3/1.2 | None | None | 0 | MISSING | [1.576, 2.364] |
| V5/diagnostic | 0.3858362688055085 | [0.3847874149863435, 0.38688512262467345] | 30 | FAIL | [0.4, 0.6] |
| V6/1.15 | 3.158747087331176 | [3.1584385741716794, 3.1590556004906727] | 30 | FAIL | [1.6800000000000002, 2.52] |
| V6/1.42 | 3.5998591829471547 | [3.5995649968495464, 3.600153369044763] | 30 | FAIL | [1.92, 2.88] |
| V6/1.78 | 4.263621808499972 | [4.2630791867889055, 4.264164430211039] | 30 | FAIL | [2.16, 3.24] |
| V4/all_data_flow_persons_s width slope | 1.6152285491998153 | [1.6029152112622622, 1.6275418871373684] | 30 | FAIL | [1.8399999999999999, 2.76] |
| V4/steady_flow_persons_s width slope | 3.0437565816471563 | [3.006178273543322, 3.0813348897509907] | 30 | FAIL | [2.0, 3.0] |

Constituent protocol measurements (V4 widths contribute to the slope checks; V2 passed_n measures physical traversal):

| Case | Mean | 95% interval | Observed n | Traversal | Integration ms/step | Overlap pair-steps | Wall ped-steps |
|---|---|---|---|---|---|---|---|
| V1/native | 1.25510791993694 | [1.1825575631024041, 1.3276582767714757] | 30 | n/a | 32.12586508652506 | 0 | 0 |
| V2/0.61 | 0.04673000573467526 | [0.043648556528135755, 0.04981145494121477] | 30 | 30/30 | 0.17588619357214372 | 0 | 0 |
| V2/0.788 | 0.0003749862198652217 | [-5.2683677099879566e-05, 0.0008026561168303229] | 30 | 30/30 | 0.16621842641660348 | 0 | 0 |
| V2/0.966 | 0.0 | [0.0, 0.0] | 30 | 30/30 | 0.16621983140300647 | 0 | 0 |
| V3/0.8 | None | None | 0 | n/a | 4.936910459383701 | 0 | 0 |
| V3/0.9 | None | None | 0 | n/a | 3.3812231489622113 | 0 | 0 |
| V3/1.0 | None | None | 0 | n/a | 2.8266790135530755 | 0 | 0 |
| V3/1.1 | None | None | 0 | n/a | 2.5703618542853897 | 0 | 0 |
| V3/1.2 | None | None | 0 | n/a | 2.370214892202057 | 0 | 0 |
| V4/2.4 | 1.987172270164877 | [1.9634816041035899, 2.0108629362261645] | 30 | n/a | 45.90010937791764 | 0 | 0 |
| V4/3.0 | 1.86390946264159 | [1.8421575025247299, 1.88566142275845] | 30 | n/a | 45.44310379594875 | 0 | 0 |
| V4/3.6 | 1.7384290400084632 | [1.7208192240357143, 1.7560388559812121] | 30 | n/a | 45.88829255021604 | 0 | 0 |
| V4/4.4 | 1.544043975621478 | [1.5282825651540821, 1.559805386088874] | 30 | n/a | 44.85693220682151 | 0 | 0 |
| V4/5.0 | 1.431265786394288 | [1.410496176197439, 1.4520353965911368] | 30 | n/a | 44.17515250226521 | 0 | 0 |
| V4/all_data_flow_persons_s width slope | 1.6152285491998153 | [1.6029152112622622, 1.6275418871373684] | 30 | n/a | None | n/a | n/a |
| V4/steady_flow_persons_s width slope | 3.0437565816471563 | [3.006178273543322, 3.0813348897509907] | 30 | n/a | None | n/a | n/a |
| V5/diagnostic | 0.3858362688055085 | [0.3847874149863435, 0.38688512262467345] | 30 | n/a | 0.29761368428977825 | 0 | 0 |
| V6/1.15 | 3.158747087331176 | [3.1584385741716794, 3.1590556004906727] | 30 | n/a | 0.18954822066713403 | 0 | 0 |
| V6/1.42 | 3.5998591829471547 | [3.5995649968495464, 3.600153369044763] | 30 | n/a | 0.13067054461420172 | 0 | 0 |
| V6/1.78 | 4.263621808499972 | [4.2630791867889055, 4.264164430211039] | 30 | n/a | 0.1305375187383765 | 0 | 0 |
