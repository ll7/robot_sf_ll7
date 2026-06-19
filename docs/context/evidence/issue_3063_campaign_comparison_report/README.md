# Issue #3063 Campaign Comparison Report Evidence

This directory contains compact, tracked evidence for the analysis-only campaign comparison report
introduced for issue #3063.

Source fixture:

- `tests/fixtures/campaign_result_store/issue_3063_episode_rows.json`

Generated artifacts:

- `report.json`: machine-readable `campaign-comparison-report.v1` payload.
- `report.md`: Markdown report with row-status caveats, metric summaries, visual summaries, and
  descriptive statistical hooks.

Regeneration outline:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from scripts.tools.campaign_result_store import write_result_store

rows = json.loads(
    Path("tests/fixtures/campaign_result_store/issue_3063_episode_rows.json").read_text(
        encoding="utf-8"
    )
)
write_result_store(
    Path("output/issue_3063_campaign_comparison_fixture/result-store"),
    rows,
    study_id="issue-3063-fixture",
    command="uv run python scripts/tools/build_campaign_comparison_report.py --result-store output/issue_3063_campaign_comparison_fixture/result-store --input-label tests/fixtures/campaign_result_store/issue_3063_episode_rows.json --output-json docs/context/evidence/issue_3063_campaign_comparison_report/report.json --output-md docs/context/evidence/issue_3063_campaign_comparison_report/report.md --min-sample 1",
    source_commit="8e65f902fb5ed1825f7ae49fcb8f5391e223ed93",
)
PY

uv run python scripts/tools/build_campaign_comparison_report.py \
  --result-store output/issue_3063_campaign_comparison_fixture/result-store \
  --input-label tests/fixtures/campaign_result_store/issue_3063_episode_rows.json \
  --output-json docs/context/evidence/issue_3063_campaign_comparison_report/report.json \
  --output-md docs/context/evidence/issue_3063_campaign_comparison_report/report.md \
  --min-sample 1
```

The `output/` result store is disposable local state. The durable evidence is this compact report
bundle plus the tracked fixture input.
