# Skill Content Contracts

Declarative, versioned content policies for individual skills. Each
`<skill>.content-contract.v1.yaml` file lists literal text requirements that the
skill's `SKILL.md` must satisfy. `scripts/dev/check_skills.py` evaluates them;
the schema and evaluator live in `scripts/dev/skill_content_contracts.py`.

## Authoring

Add or edit a contract without touching Python:

```yaml
version: 1
skill: <skill name>            # must match the file stem exactly
requirements:
  - id: <stable-semantic-id>   # used in failure messages
    description: <remediation hint shown on failure>
    scope: raw                 # raw | lowercase | normalized
    operator: all_of           # all_of | any_of
    values:
      - "<literal that must appear>"
```

- `scope: raw` matches case-sensitively; `lowercase` ignores case; `normalized`
  also collapses whitespace runs (use for phrases broken across lines).
- `all_of` requires every value; `any_of` requires at least one.
- Unknown fields, unknown scopes/operators, duplicate ids, and a `skill` field
  that disagrees with the filename are rejected (fail closed).

## Testing

```bash
uv run python scripts/dev/check_skills.py
uv run python -m pytest tests/dev/test_skill_content_contracts.py -q
```

Parity tests in `tests/dev/test_skill_content_contracts.py` verify every migrated
skill's real `SKILL.md` against its contract.
