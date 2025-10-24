from __future__ import annotations

from robot_sf.benchmark.ranking import compute_ranking, format_csv, format_markdown


def _rec(g, **m):
    return {
        "scenario_params": {"algo": g},
        "algo": g,
        "scenario_id": g,
        "metrics": m,
    }


def test_compute_ranking_basic():
    records = [
        _rec("a", collisions=1),
        _rec("a", collisions=3),
        _rec("b", collisions=0),
        _rec("b", collisions=2),
        _rec("c", collisions=5),
    ]
    rows = compute_ranking(records, metric="collisions", ascending=True)
    # Means: a=2, b=1, c=5 -> ascending gives b, a, c
    assert [r.group for r in rows] == ["b", "a", "c"]
    assert [r.count for r in rows] == [2, 2, 1]


def test_compute_ranking_top_and_desc():
    records = [
        _rec("a", comfort_exposure=0.1),
        _rec("a", comfort_exposure=0.4),
        _rec("b", comfort_exposure=0.3),
        _rec("b", comfort_exposure=0.2),
        _rec("c", comfort_exposure=0.5),
    ]
    # Descending (higher is better for this synthetic case), limit top 2
    rows = compute_ranking(records, metric="comfort_exposure", ascending=False, top=2)
    # Means: a=0.25, b=0.25, c=0.5 -> descending gives c first, then a/b
    assert rows[0].group == "c"
    assert rows[0].count == 1
    assert len(rows) == 2


def test_formatters_return_strings():
    records = [_rec("a", collisions=1), _rec("a", collisions=3)]
    rows = compute_ranking(records, metric="collisions")
    md = format_markdown(rows, "collisions")
    csv = format_csv(rows, "collisions")
    assert "| Rank |" in md
    assert md.endswith("\n")
    assert csv.splitlines()[0].startswith("rank,group,mean_")
    assert csv.endswith("\n")
