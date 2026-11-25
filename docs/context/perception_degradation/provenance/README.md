# Acquisition and custody

The main acquisition and frozen contract are both at commit
`4e16d46908789c37651a4b851a23f10bf1540e7b`, reachable on the retained
`xperc-perception-degradation` branch. The starting release commit is
`373dbfde4f39667cf9e8732dabe7df5118bdeab1`. The packet-only branch is based
on later main and does not itself install the experimental runtime hook.

The execution commit's Git author and committer timestamps are 2026-10-28
16:09:11 UTC. The archived main `environment.txt` records a start at
16:09:36 UTC; the earliest producer episode's `timestamps.start` is
16:09:45.092746 UTC. On this host, the earliest remote-tracking reflog entry
observing the execution SHA is a fetch at 16:10:22 UTC; the original PR was
opened at 16:10:32 UTC. The Git timestamp supports a pre-episode commit but
does not independently establish when it was pushed. No earlier push receipt
or independent remote observation was found in the available xperc lane and
archive records. This is an unresolved proof gap, not evidence of a late push.

[`runtime.patch`](runtime.patch) is the zero-context release-to-acquisition diff for all
five changed runtime files and the two associated diagnostic tests. It
contains both the guard-switch ablation and the pedestrian-perception hook.
It can be applied to the release commit with `git apply --unidiff-zero`; its reverse applies
cleanly at the acquisition commit. The original preregistration is preserved
unchanged as [`../contract.yaml`](../contract.yaml).

The raw main, pilot and smoke stages have separately verified SHA-256 manifests
in the lab host archive. The pilot compact qualification is tracked in this
packet. The main-stage W&B URI, digest and file-manifest readback are recorded
in [`preservation.json`](preservation.json).
