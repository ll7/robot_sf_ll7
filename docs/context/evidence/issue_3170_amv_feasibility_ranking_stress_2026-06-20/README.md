# Issue #3170 AMV Feasibility Ranking Stress Synthesis

This directory contains the compact tracked evidence summary for issue #3170.
It synthesizes existing durable AMV/AMMV artifacts rather than adding a new
benchmark run.

The result is diagnostic-only:

- the broadest existing AMMV/default slice has 15 paired rows across five
  scenario families and three seeds per family;
- every paired row is frame-identical and episode-metric-identical;
- the prior actuation-aware feasibility ordering remains one scenario and one
  seed only;
- no general AMV feasibility ranking claim is justified from the available
  evidence.

Follow-up issue #3181 tracks the missing direct actuation-aware multi-scenario,
multi-seed execution slice.

