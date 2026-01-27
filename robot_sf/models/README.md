# Model Registry Helpers

This module wraps the `model/registry.yaml` file and provides convenience
functions for loading models by `model_id`.

## Quick usage

```python
from robot_sf.models import resolve_model_path

path = resolve_model_path("my_model_id", allow_download=True)
```

## Auto-population

Use `upsert_registry_entry` to insert/update entries:

```python
from robot_sf.models import upsert_registry_entry

upsert_registry_entry(
    {
        "model_id": "my_model_id",
        "display_name": "Short human-readable title",
        "local_path": "model/my_model_id/model.zip",
        "config_path": "configs/training/...",
        "commit": "abcdef123456",
        "wandb_run_path": "entity/project/run_id",
        "wandb_file": "model.zip",
        "tags": ["ppo", "socnav"],
        "notes": ["Optional notes."],
    }
)
```
