# Issue #3544 Signalized Runtime Metrics Report

The trace-recording run produced four runtime rows from
`configs/scenarios/single/issue_2799_signalized_runtime_smoke.yaml`.

| Episode ID | Row Type | Eligible | Denominator | Exclusion Reason | Crossed Red | Min Dist m | Ped Conflicts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `issue_2799_red_required_stop_observable--2799--60c725f2f37c0307` | `red_required_stop` | true | 1 | - | true | 0.00 | 0 |
| `issue_2799_green_proceed_observable--2800--80d40eb20c230d85` | `green_proceed` | true | 1 | - | false | 8.63 | 0 |
| `issue_2799_unavailable_no_claim--2801--ce49d0bf33f66d62` | `unavailable_no_claim` | false | 0 | `signal_state_metadata_absent` | false | N/A | 0 |
| `issue_2799_proxy_only_denominator_excluded--2802--b57258021f1094d1` | `proxy_only_denominator_excluded` | false | 0 | `signal_state_not_benchmark_evidence` | false | N/A | 0 |

Summary:

- Total rows: 4.
- Observable denominator rows: 2.
- Fail-closed excluded rows: 2.
- Runtime evidence is planner-observable for the two denominator rows.

The red required-stop row has `crossed_red=true`. The issue #3544 builder run did not classify that
signal metric as a failure-pack case; issue #3546 updates the builder contract so future reruns can
classify planner-observable red-phase and stop-line signal metrics as signalized failure-pack
predicates.
