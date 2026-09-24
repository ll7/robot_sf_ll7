"""Load one preserved synthetic Proximal Policy Optimization (PPO) checkpoint for a loader-contract smoke; source/config lineage is unavailable.

This diagnostic-only example stages its checkpoint and normalizer under a fresh temporary root,
disables model downloads, uses the canonical registry resolver, and performs one deterministic
prediction. It does not claim full checkpoint reconstruction or model provenance, and does not run
navigation, training, a benchmark, or scientific evaluation.

Real learned-policy provenance (see ``docs/context/artifact_evidence_vocabulary.md``) requires a tracked config and training/data-generation commit; this fixture records their absence rather than inferring them.

Run from the repository root with ``uv run python examples/advanced/42_offline_checkpoint_cold_load.py --isolated-cache --json``. ``--isolated-cache`` explicitly asserts the cache-disabled execution contract; isolation remains enforced by default. ``--fixture`` accepts ``synthetic_ppo_v1`` or a local manifest path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_sf.common.seed import _configure_torch_213_runtime  # noqa: E402
from robot_sf.models.registry import resolve_model_path, sha256_of_file  # noqa: E402

DEFAULT_FIXTURE = REPO_ROOT / "examples/fixtures/offline_cold_load_v1/manifest.json"
FIXTURE_ID = "synthetic_ppo_v1"
MANIFEST_SCHEMA = "offline_cold_load_manifest.v1"
REPORT_SCHEMA = "offline_cold_load_report.v1"
SOURCE_CONFIG_REASON = "No source/config receipt was retained with this synthetic checkpoint; no commit or training lineage is inferred."  # fmt: skip
CLAIM_BOUNDARY = "loader-contract smoke-only; source/config lineage unavailable; not benchmark or scientific/model evidence"  # fmt: skip
STALE_ALIASES = frozenset({"latest", "best-success"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEPENDENCIES = (("stable-baselines3", "stable_baselines3", "2.9.0"), ("torch", "torch", "2.13.0"), ("gymnasium", "gymnasium", "1.1.1"))  # fmt: skip
OWNER_MARKER = '{"schema":"offline_cold_load_owner.v1","owner":"issue-8901"}\n'
MANIFEST_KEYS = set("artifact claim_boundary dependencies fixture_id fixture_rights inference loader manifest_version model normalizer observation action registry source_config aliases schema".split())  # fmt: skip


class ColdLoadError(ValueError):
    """Fail-closed example error with a stable, path-free reason code."""

    def __init__(self, code: str) -> None:
        """Store the stable failure code."""
        super().__init__(code)
        self.code = code

    def to_dict(self) -> dict[str, str]:
        """Return a machine-readable error without local paths or host details."""
        return {"code": self.code}


def _fail(code: str) -> NoReturn:
    raise ColdLoadError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _keys(value: Any, expected: set[str], code: str = "manifest_schema_invalid") -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code)
    return value


def _read_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _shape(value: Any, code: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        _fail(code)
    return tuple(value)


def _array(value: Any, shape: tuple[int, ...], code: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        _fail(code)
    if result.shape != shape or not np.all(np.isfinite(result)):
        _fail(code)
    return result


def _relative(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name or Path(name).is_absolute():
        _fail("manifest_path_invalid")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        _fail("fixture_path_escape")
    if any(part in {"", "."} for part in path.parts):
        _fail("manifest_path_invalid")
    return path.as_posix()


def _artifact(value: Any) -> dict[str, Any]:
    item = _keys(value, {"path", "sha256", "size_bytes"})
    digest, size = item["sha256"], item["size_bytes"]
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, "digest_not_pinned")  # fmt: skip
    _require(not isinstance(size, bool) and isinstance(size, int) and size > 0, "manifest_size_invalid")  # fmt: skip
    return {"path": _relative(item["path"]), "sha256": digest, "size_bytes": size}


def _contract(value: Any, shape_code: str) -> dict[str, Any]:
    item = _keys(value, {"dtype", "high", "schema", "shape", "low"})
    _require(item["dtype"] == "float32" and isinstance(item["schema"], str) and item["schema"], "contract_schema_invalid")  # fmt: skip
    shape = _shape(item["shape"], shape_code)
    low, high = (
        _array(item["low"], shape, "contract_bounds_invalid"),
        _array(item["high"], shape, "contract_bounds_invalid"),
    )
    _require(not np.any(low > high), "contract_bounds_invalid")  # fmt: skip
    return {"shape": shape, "low": low, "high": high}


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _keys(dict(manifest), MANIFEST_KEYS)
    _require(manifest["schema"] == MANIFEST_SCHEMA and manifest["manifest_version"] == 1, "unsupported_schema")  # fmt: skip
    _require(manifest["fixture_id"] == FIXTURE_ID and manifest["claim_boundary"] == CLAIM_BOUNDARY, "manifest_identity_invalid")  # fmt: skip

    artifact = _keys(manifest["artifact"], {"checkpoint", "normalizer"})
    artifacts = {name: _artifact(artifact[name]) for name in artifact}
    _require(artifacts["checkpoint"]["path"] != artifacts["normalizer"]["path"], "manifest_companion_invalid")  # fmt: skip
    normalizer = _keys(manifest["normalizer"], {"dtype", "path", "schema", "sha256", "shape"})
    normalizer_shape = _shape(normalizer["shape"], "normalizer_shape_invalid")
    _require(not (normalizer["dtype"] != "float32" or normalizer["schema"] != "normalizer_state.v1" or _relative(normalizer["path"]) != artifacts["normalizer"]["path"] or normalizer["sha256"] != artifacts["normalizer"]["sha256"]), "normalizer_manifest_invalid")  # fmt: skip

    observation = _contract(manifest["observation"], "observation_shape_invalid")
    action = _contract(manifest["action"], "action_shape_invalid")
    _require(normalizer_shape == observation["shape"], "normalizer_shape_mismatch")
    inference = _keys(manifest["inference"], {"deterministic", "observation"})
    _require(inference["deterministic"] is True, "determinism_required")
    sample = _array(inference["observation"], observation["shape"], "inference_observation_invalid")
    _require(not (np.any(sample < observation["low"]) or np.any(sample > observation["high"])), "inference_observation_out_of_bounds")  # fmt: skip

    expected_model = {"algorithm_class": "stable_baselines3.ppo.ppo.PPO", "feature_extractor_class": "stable_baselines3.common.torch_layers.FlattenExtractor", "model_id": "offline_synthetic_ppo_v1", "policy_alias": "MlpPolicy", "policy_class": "stable_baselines3.common.policies.ActorCriticPolicy"}  # fmt: skip
    model = _keys(manifest["model"], set(expected_model))
    _require(dict(model) == expected_model, "model_identity_invalid")
    expected_loader = {"allow_download": False, "load_mode": "direct_sb3", "qualified_name": "stable_baselines3.ppo.ppo.PPO.load"}  # fmt: skip
    loader = _keys(manifest["loader"], set(expected_loader), "loader_identity_invalid")
    _require(loader["allow_download"] is False, "cache_policy_invalid")
    _require(loader["load_mode"] == "direct_sb3", "loader_mode_invalid")
    _require(dict(loader) == expected_loader, "loader_identity_invalid")
    registry = _keys(manifest["registry"], {"local_only", "model_id", "tags"})
    _require(registry["local_only"] is True, "cache_policy_invalid")
    _require(registry["model_id"] == model["model_id"] and isinstance(registry["tags"], list), "cache_policy_invalid")  # fmt: skip
    _require({"synthetic", "smoke-only", "not-for-benchmark"}.issubset(registry["tags"]), "fixture_boundary_invalid")  # fmt: skip

    rights = _keys(manifest["fixture_rights"], {"basis", "license", "status", "evidence"})
    _require(rights["status"] == "project-authored", "fixture_rights_invalid")
    _require(not (rights["license"] != "repository" or rights["basis"] != "project-authored synthetic test fixture; no external weights or data" or not isinstance(rights["evidence"], list) or not {"scripts/validation/asset_rights_inventory.v1.yaml", "LICENSE"}.issubset(rights["evidence"])), "fixture_rights_invalid")  # fmt: skip
    dependencies = manifest["dependencies"]
    expected_dependencies = [{"distribution": d, "import": i, "version": v} for d, i, v in DEPENDENCIES]  # fmt: skip
    _require(dependencies == expected_dependencies, "dependency_manifest_invalid")

    source = _keys(manifest["source_config"], {"reason", "status"}, "source_identity_invalid")
    _require(source == {"status": "unavailable", "reason": SOURCE_CONFIG_REASON}, "source_identity_invalid")  # fmt: skip
    aliases = _keys(manifest["aliases"], {"accepted", "canonical", "stale"})
    _require(not (aliases["accepted"] != [FIXTURE_ID] or aliases["canonical"] != FIXTURE_ID or set(aliases["stale"]) != STALE_ALIASES), "alias_manifest_invalid")  # fmt: skip
    return {"artifact": artifacts, "normalizer": {"path": artifacts["normalizer"]["path"], "shape": normalizer_shape}, "observation": {**observation, "sample": sample}, "action": action, "model": dict(model), "loader": dict(loader), "source_config": dict(source), "dependencies": [{"distribution": d, "version": v} for d, _, v in DEPENDENCIES]}  # fmt: skip


def resolve_fixture(identifier: str | Path = FIXTURE_ID) -> Path:
    """Resolve the canonical fixture id or an explicit local manifest path."""
    raw = str(identifier)
    if raw in STALE_ALIASES:
        _fail("stale_alias")
    if raw == FIXTURE_ID:
        return DEFAULT_FIXTURE
    if "://" in raw:
        _fail("fixture_path_invalid")
    path = Path(identifier).expanduser()
    if path.is_dir():
        path /= "manifest.json"
    try:
        path = path.resolve(strict=False)
    except OSError:
        _fail("fixture_path_invalid")
    if not path.is_file() or path.is_symlink():
        _fail("fixture_missing")
    return path


def _source_member(root: Path, name: str) -> Path:
    base = root.resolve()
    path = (base / name).resolve(strict=False)
    if not path.is_relative_to(base):
        _fail("fixture_path_escape")
    if path.is_symlink() or not path.is_file():
        _fail("missing_companion_state")
    return path


def _stage(root: Path, manifest_path: Path, data: Mapping[str, Any]) -> tuple[Path, Path]:
    artifact_root = root / "artifact"
    artifact_root.mkdir()
    for identity in data["artifact"].values():
        source = _source_member(manifest_path.parent, identity["path"])
        target = artifact_root / identity["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(source, target)
            valid = target.stat().st_size == identity["size_bytes"] and sha256_of_file(target) == identity["sha256"]  # fmt: skip
        except OSError:
            _fail("missing_companion_state")
        if not valid:
            _fail("digest_mismatch")
    try:
        registry = {"version": 1, "models": [{"model_id": data["model"]["model_id"], "local_path": str((artifact_root / data["artifact"]["checkpoint"]["path"]).resolve()), "local_only": True}]}  # fmt: skip
        (root / "registry.yaml").write_text(json.dumps(registry), encoding="utf-8")
    except OSError:
        _fail("cache_registry_unavailable")
    return artifact_root, root / "registry.yaml"


@contextmanager
def _isolated_environment(root: Path, *, assert_cache: bool = False) -> Iterator[None]:
    """Point model/cache homes at the task-owned root and disable downloads."""
    values = {"ROBOT_SF_DISABLE_MODEL_DOWNLOADS": "1", "HOME": root / "home", "XDG_CACHE_HOME": root / "cache", "TORCH_HOME": root / "torch"}  # fmt: skip
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        os.environ[key] = str(value)
        if key != "ROBOT_SF_DISABLE_MODEL_DOWNLOADS":
            Path(value).mkdir(parents=True, exist_ok=True)
    try:
        if assert_cache and (
            os.environ.get("ROBOT_SF_DISABLE_MODEL_DOWNLOADS") != "1"
            or any(
                Path(os.environ.get(key, "")).resolve() != Path(value).resolve()
                for key, value in values.items()
                if key != "ROBOT_SF_DISABLE_MODEL_DOWNLOADS"
            )
        ):
            _fail("cache_policy_invalid")
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _check_dependencies(data: Mapping[str, Any]) -> None:
    for dependency in data["dependencies"]:
        try:
            observed = str(package_version(dependency["distribution"])).split("+", 1)[0]
        except PackageNotFoundError:
            _fail("dependency_unavailable")
        if observed != dependency["version"]:
            _fail("dependency_version_mismatch")


def _qualified(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _check_space(space: object, expected: Mapping[str, Any], code: str) -> None:
    try:
        actual = type(space)
        low, high = (
            np.asarray(space.low, dtype=np.float64),
            np.asarray(space.high, dtype=np.float64),
        )
        dtype, shape = str(np.dtype(space.dtype)), tuple(space.shape)
    except (AttributeError, TypeError, ValueError):
        _fail(code)
    _require(not (actual.__name__ != "Box" or actual.__module__ != "gymnasium.spaces.box" or dtype != "float32" or shape != expected["shape"] or not np.array_equal(low, expected["low"]) or not np.array_equal(high, expected["high"])), code)  # fmt: skip


def _check_normalizer(path: Path, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, float]:
    state = _read_json(path, "normalizer_invalid")
    if set(state) != {"clip", "count", "mean", "schema", "shape", "variance"}:
        _fail("normalizer_invalid")
    if state["schema"] != "normalizer_state.v1" or state["shape"] != list(shape):
        _fail("normalizer_shape_mismatch")
    try:
        count, clip = float(state["count"]), float(state["clip"])
    except (TypeError, ValueError):
        _fail("normalizer_invalid")
    mean, variance = _array(state["mean"], shape, "normalizer_invalid"), _array(state["variance"], shape, "normalizer_invalid")  # fmt: skip
    _require(not (not math.isfinite(count) or count <= 0 or not math.isfinite(clip) or clip <= 0 or np.any(variance <= 0)), "normalizer_invalid")  # fmt: skip
    return mean, variance, clip


def _check_parameters(model: object) -> None:
    try:
        arrays = [parameter.detach().cpu().numpy() for parameter in model.policy.parameters()]
    except Exception:
        _fail("parameters_unreadable")
    if not arrays or any(not np.all(np.isfinite(array)) for array in arrays):
        _fail("non_finite_parameters")


def _predict(model: object, observation: np.ndarray, expected: Mapping[str, Any]) -> np.ndarray:
    try:
        first, _ = model.predict(observation.astype(np.float32), deterministic=True)
        second, _ = model.predict(observation.astype(np.float32), deterministic=True)
        first, second = np.asarray(first), np.asarray(second)
    except Exception:
        _fail("output_contract_mismatch")
    if first.shape != expected["shape"] or second.shape != expected["shape"]:
        _fail("output_shape_mismatch")
    if str(first.dtype) != "float32" or str(second.dtype) != "float32":
        _fail("output_dtype_mismatch")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        _fail("output_non_finite")
    if any(
        (np.any(value < expected["low"]) or np.any(value > expected["high"]))
        for value in (first, second)
    ):
        _fail("output_out_of_bounds")
    if not np.array_equal(first, second):
        _fail("output_nondeterministic")
    return first


def _new_root(requested: str | Path | None) -> Path:
    if requested is None:
        root = Path(tempfile.mkdtemp(prefix="robot_sf_offline_cold_load_"))
    else:
        root = Path(requested).expanduser()
        if root.exists() or root.is_symlink():
            _fail("isolated_root_not_fresh")
        try:
            root.parent.mkdir(parents=True, exist_ok=True)
            root.mkdir()
        except OSError:
            _fail("isolated_root_unusable")
    try:
        (root / ".robot_sf_cold_load_owner").write_text(OWNER_MARKER, encoding="utf-8")
    except OSError:
        _fail("isolated_root_unusable")
    return root


def _cleanup(root: Path) -> None:
    try:
        owned = (root / ".robot_sf_cold_load_owner").read_text(encoding="utf-8") == OWNER_MARKER
    except OSError:
        owned = False
    if not owned or root == Path("/") or root.is_symlink():
        _fail("cleanup_ownership")
    try:
        shutil.rmtree(root)
    except OSError:
        _fail("cleanup_failed")


def run_cold_load(manifest_path: str | Path = FIXTURE_ID, *, isolated_root: str | Path | None = None, assert_isolated_cache: bool = False) -> dict[str, Any]:  # fmt: skip
    """Cold-load one local fixture and return smoke-only validation evidence."""
    path = resolve_fixture(manifest_path)
    data = _validate_manifest(_read_json(path, "manifest_unreadable"))
    root = _new_root(isolated_root)
    try:
        with _isolated_environment(root, assert_cache=assert_isolated_cache):
            artifact_root, registry_path = _stage(root, path, data)
            _check_dependencies(data)
            _configure_torch_213_runtime()
            try:
                importlib.import_module("torch")
                importlib.import_module("gymnasium")
                from stable_baselines3.ppo.ppo import PPO
            except (ImportError, OSError):
                _fail("dependency_unavailable")
            checkpoint = artifact_root / data["artifact"]["checkpoint"]["path"]
            try:
                resolved = resolve_model_path(
                    data["model"]["model_id"],
                    registry_path=registry_path,
                    allow_download=False,
                    cache_dir=root / "cache",
                )
            except (FileNotFoundError, KeyError, OSError, ValueError):
                _fail("cache_resolution_failed")
            if resolved.resolve() != checkpoint.resolve() or not resolved.is_relative_to(
                root.resolve()
            ):
                _fail("cache_resolution_failed")
            try:
                model = PPO.load(str(resolved), device="cpu", print_system_info=False)
            except Exception:
                _fail("checkpoint_load_failed")
            if _qualified(model) != data["model"]["algorithm_class"]:
                _fail("algorithm_class_mismatch")
            if _qualified(model.policy) != data["model"]["policy_class"]:
                _fail("policy_class_mismatch")
            _require(_qualified(getattr(model.policy, "features_extractor", None)) == data["model"]["feature_extractor_class"], "feature_extractor_class_mismatch")  # fmt: skip
            _check_parameters(model)
            _check_space(
                model.observation_space, data["observation"], "observation_contract_mismatch"
            )
            _check_space(model.action_space, data["action"], "action_contract_mismatch")
            mean, variance, clip = _check_normalizer(
                artifact_root / data["normalizer"]["path"], data["observation"]["shape"]
            )
            normalized = np.clip(
                (data["observation"]["sample"] - mean) / np.sqrt(variance), -clip, clip
            )
            if not np.all(np.isfinite(normalized)):
                _fail("normalized_observation_non_finite")
            _require(not np.any((normalized < data["observation"]["low"]) | (normalized > data["observation"]["high"])), "normalized_observation_out_of_bounds")  # fmt: skip
            output = _predict(model, normalized, data["action"])
            return {"schema": REPORT_SCHEMA, "status": "pass", "evidence_status": "smoke_only", "claim_boundary": CLAIM_BOUNDARY, "fixture_id": FIXTURE_ID, "model_id": data["model"]["model_id"], "source_config": data["source_config"], "source_config_status": data["source_config"]["status"], "load_mode": data["loader"]["load_mode"], "loader": data["loader"]["qualified_name"], "downloads_allowed": False, "cache": "isolated_temporary_root", "checkpoint": data["artifact"]["checkpoint"], "normalizer": {"sha256": data["artifact"]["normalizer"]["sha256"], "shape": list(data["normalizer"]["shape"]), "state": "validated"}, "observation": {"shape": list(data["observation"]["shape"]), "normalized_finite": True}, "action": {"shape": list(data["action"]["shape"]), "output": output.tolist(), "within_bounds": True}, "validation": {"algorithm_class": data["model"]["algorithm_class"], "policy_class": data["model"]["policy_class"], "feature_extractor_class": data["model"]["feature_extractor_class"], "parameters_finite": True, "digest_verified": True, "deterministic_output": True}}  # fmt: skip
    finally:
        _cleanup(root)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline example and print a compact pass/fail result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default=FIXTURE_ID, help="fixture id or local manifest path")
    parser.add_argument(
        "--isolated-cache", action="store_true", help="assert the enforced cache policy"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable evidence")
    args = parser.parse_args(argv)
    try:
        report = run_cold_load(args.fixture, assert_isolated_cache=args.isolated_cache)
    except ColdLoadError as error:
        print(json.dumps({"schema": REPORT_SCHEMA, "status": "failed", "evidence_status": "smoke_only", "claim_boundary": CLAIM_BOUNDARY, "error": error.to_dict()}, sort_keys=True))  # fmt: skip
        return 2
    print(
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else "status=pass fixture=synthetic_ppo_v1 load_mode=direct_sb3 evidence=smoke_only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
