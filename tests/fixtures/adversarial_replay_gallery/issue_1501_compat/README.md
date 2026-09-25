# Historical replay gallery compatibility fixture

This one-candidate input lets the #9647 canonical CLI exercise strict candidate selection, a real
replay, and the existing renderer when the #9645 bounded search produces no eligible failures. It
is adapted from tracked #1501 `failure_0002` evidence and the #9645 current-code replay. It is not a
persisted search result or a new discovery.

The archived source revision is `4bd5fb412d2bf023d26f674833dd379aedf8c17e`; the replay episode
revision is `58e516aa4f69ff3098bf518199f483006589758c`. The manifest carries a complete
`scenario_cert.v1` certificate generated post-hoc by the canonical certifier against the tracked
scenario and current map/route inputs. Its generation command, code revision, input digests, and
certificate digest are recorded in `scenario_certification_provenance.json`. This is static route
classification only; dynamic task feasibility remains unknown. The certificate is not from the
original search run and does not establish planner performance or real-world safety.
