"""Tests for binding direct Zenodo operations to the benchmark release manifest."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest
from robot_sf.benchmark.zenodo_publisher import (
    ZENODO_STATE_SCHEMA,
    ZENODO_VERIFICATION_SCHEMA,
    ZenodoPublisherError,
    _seal_state,
    _verify_integrity,
    build_release_binding,
    load_dataset_metadata,
    load_state,
    publish,
    recover,
    repair_draft_metadata,
    reserve,
    upload,
    verify,
    write_state,
)

_MANIFEST_PATH = Path("configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml")
_OLD_SOURCE_TAG = "https://github.com/ll7/robot_sf_ll7/releases/tag/previous-candidate"


class _Response:
    """Small requests-like response fixture."""

    def __init__(
        self,
        payload: Any,
        *,
        content: bytes | None = None,
        status_code: int = 200,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.content = content if content is not None else json.dumps(payload).encode()

    def json(self) -> Any:
        """Return the configured JSON payload."""
        return self.payload

    def raise_for_status(self) -> None:
        """Implement the response protocol for successful fixtures."""

    def iter_content(self, *, chunk_size: int) -> Any:
        """Yield the configured body as one streamed chunk."""
        del chunk_size
        yield self.content


class _Session:
    """Queue-backed session fixture for all four publisher modes."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.posts: list[_Response] = []
        self.gets: list[_Response] = []
        self.puts: list[_Response] = []
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        """Consume a queued POST response."""
        self.calls.append(("POST", url, kwargs))
        return self.posts.pop(0)

    def get(self, url: str, **kwargs: Any) -> _Response:
        """Consume a queued GET response."""
        self.calls.append(("GET", url, kwargs))
        return self.gets.pop(0)

    def put(self, url: str, **kwargs: Any) -> _Response:
        """Consume a queued PUT response."""
        self.calls.append(("PUT", url, kwargs))
        return self.puts.pop(0)


def _binding_and_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the checked-in benchmark metadata and its release binding."""
    manifest = load_release_manifest(_MANIFEST_PATH)
    binding = build_release_binding(manifest)
    metadata = load_dataset_metadata(
        binding["metadata_path"],
        expected_source_tag=binding["release_tag"],
        expected_metadata_sha256=binding["metadata_sha256"],
    )
    return binding, metadata


def _deposition_payload(
    binding: dict[str, Any],
    *,
    submitted: bool = False,
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a reserved or published deposition response."""
    version_record_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    concept_record_id = binding["concept_doi"].rsplit(".", 1)[-1]
    return {
        "id": version_record_id,
        "record_id": version_record_id,
        "conceptrecid": concept_record_id,
        "doi": binding["version_doi"] if submitted else None,
        "state": "done" if submitted else "unsubmitted",
        "submitted": submitted,
        "metadata": {"prereserve_doi": {"doi": binding["version_doi"]}},
        "links": {"bucket": "https://zenodo.org/api/files/bucket"},
        "files": list(files or []),
    }


def _draft_file(binding: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one legacy draft-file identity bound to the manifest deposition."""
    deposition_id = binding["version_doi"].rsplit(".", 1)[-1]
    return {
        "id": "uploaded-file",
        "filename": name,
        "links": {
            "self": (
                f"https://zenodo.org/api/deposit/depositions/{deposition_id}/files/uploaded-file"
            )
        },
    }


def _unbound_state(
    binding: dict[str, Any], *, files: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build a valid state that lets a bound operation adopt its binding."""
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    concept_record_id = binding["concept_doi"].rsplit(".", 1)[-1]
    return _seal_state(
        {
            "schema_version": ZENODO_STATE_SCHEMA,
            "deposition_id": deposition_id,
            "record_id": deposition_id,
            "concept_record_id": concept_record_id,
            "doi": binding["version_doi"],
            "submitted": False,
            "state": "unsubmitted",
            "files": files or [],
        }
    )


def _bound_uploaded_state(
    binding: dict[str, Any], metadata: dict[str, Any], bundle: Path
) -> dict[str, Any]:
    """Build a sealed manifest-bound state with one uploaded bundle."""
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    concept_record_id = binding["concept_doi"].rsplit(".", 1)[-1]
    metadata_contract = {key: value for key, value in metadata.items() if key != "prereserve_doi"}
    metadata_contract_sha256 = hashlib.sha256(
        json.dumps(
            metadata_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return _seal_state(
        {
            "schema_version": ZENODO_STATE_SCHEMA,
            "deposition_id": deposition_id,
            "record_id": deposition_id,
            "concept_record_id": concept_record_id,
            "doi": binding["version_doi"],
            "submitted": False,
            "state": "unsubmitted",
            "files": [
                {
                    "name": bundle.name,
                    "size": bundle.stat().st_size,
                    "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                }
            ],
            "release_binding": {
                "metadata_sha256": binding["metadata_sha256"],
                "metadata_contract_sha256": metadata_contract_sha256,
                "release_tag": binding["release_tag"],
                "concept_doi": binding["concept_doi"],
                "version_doi": binding["version_doi"],
            },
        }
    )


def _operational_draft_payload(
    binding: dict[str, Any],
    metadata: dict[str, Any],
    bundle: Path,
    *,
    submitted: bool = False,
) -> dict[str, Any]:
    """Return a remote draft with the release-bound metadata and bundle file."""
    payload = _deposition_payload(binding, submitted=submitted)
    payload["metadata"] = {
        **metadata,
        "prereserve_doi": {"doi": binding["version_doi"]},
        "version": "0.0.7",
        "publication_date": "2026-09-24",
    }
    payload["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {
                "download": (
                    f"https://zenodo.org/api/records/{payload['record_id']}"
                    f"/files/{bundle.name}/content"
                )
            },
        }
    ]
    return payload


def test_benchmark_manifest_loads_exact_zenodo_metadata_binding() -> None:
    """The v0.2 benchmark manifest exposes and validates metadata bytes."""
    manifest = load_release_manifest(_MANIFEST_PATH)

    assert manifest.metadata_path is not None
    assert manifest.metadata_path.is_file()
    assert (
        manifest.metadata_sha256 == hashlib.sha256(manifest.metadata_path.read_bytes()).hexdigest()
    )


def test_benchmark_manifest_rejects_metadata_checksum_drift(tmp_path: Path) -> None:
    """A changed or stale metadata digest blocks manifest loading."""
    manifest = load_release_manifest(_MANIFEST_PATH)
    payload = yaml.safe_load(_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["canonical_campaign_config"] = str(manifest.canonical_campaign_config_path)
    payload["scenario"]["matrix_path"] = str(manifest.scenario_matrix_path)
    payload["scenario"]["suite_policy_path"] = str(manifest.suite_policy_path)
    payload["scenario"]["route_certification_path"] = str(manifest.route_certification_path)
    payload["seed_policy"]["seed_sets_path"] = str(
        Path("configs/benchmarks/seed_sets_v1.yaml").resolve()
    )
    payload["metrics"]["snqi_weights_path"] = str(manifest.snqi_weights_path)
    payload["metrics"]["snqi_baseline_path"] = str(manifest.snqi_baseline_path)
    payload["citation_path"] = str(manifest.citation_path)
    payload["release_checklist_path"] = str(manifest.release_checklist_path)
    payload["publication"]["metadata_path"] = str(manifest.metadata_path)
    payload["publication"]["metadata_sha256"] = "0" * 64
    path = tmp_path / "release.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata_sha256 does not match"):
        load_release_manifest(path)


def test_release_validation_rechecks_metadata_digest() -> None:
    """Post-load manifest objects cannot bypass the metadata checksum gate."""
    manifest = load_release_manifest(_MANIFEST_PATH)
    drifted = replace(manifest, metadata_sha256="0" * 64)

    report = validate_release_manifest(drifted)

    assert report["status"] == "invalid"
    assert (
        "publication.metadata_sha256 does not match publication.metadata_path" in report["problems"]
    )


def test_all_zenodo_modes_preserve_manifest_binding(tmp_path: Path) -> None:
    """Reserve, upload, verify, and publish carry the same release identity."""
    binding, metadata = _binding_and_metadata()
    version_record_id = binding["version_doi"].rsplit(".", 1)[-1]
    session = _Session()
    session.posts = [
        _Response(_deposition_payload(binding)),
        _Response(_deposition_payload(binding, submitted=True)),
    ]

    state = reserve(session, metadata, release_binding=binding)
    assert state["release_binding"]["metadata_sha256"] == binding["metadata_sha256"]
    assert state["release_binding"]["concept_doi"] == binding["concept_doi"]
    assert state["release_binding"]["version_doi"] == binding["version_doi"]

    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"manifest-bound bundle")
    uploaded = _draft_file(binding, bundle.name)
    session.gets = [
        _Response(_deposition_payload(binding)),
        _Response(_deposition_payload(binding, files=[uploaded])),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"})]
    state = upload(session, state, [bundle], release_binding=binding)

    remote_draft = _deposition_payload(binding)
    remote_draft["metadata"] = {
        **metadata,
        "prereserve_doi": {"doi": binding["version_doi"]},
    }
    remote_draft["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {
                "download": (
                    f"https://zenodo.org/api/records/{version_record_id}/files/bundle.tar.gz/content"
                )
            },
        }
    ]
    session.gets = [
        _Response(remote_draft),
        _Response({}, content=bundle.read_bytes()),
    ]
    report = verify(session, state, metadata, release_binding=binding)
    assert report["status"] == "pass", report
    assert state["verification_receipt"]["release_binding"] == state["release_binding"]
    assert state["verification_receipt"]["manifest_metadata_sha256"] == binding["metadata_sha256"]

    session.gets = [
        _Response(remote_draft),
        _Response({}, content=bundle.read_bytes()),
    ]
    state = publish(session, state, metadata, release_binding=binding)
    assert state["submitted"] is True
    assert state["release_binding"]["version_doi"] == binding["version_doi"]


def test_recover_restores_manifest_bound_state_for_upload_and_verify(tmp_path: Path) -> None:
    """A read-only draft lookup restores the same state contract as reserve."""
    binding, metadata = _binding_and_metadata()
    version_record_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    draft = _deposition_payload(binding)
    draft["metadata"] = {**metadata, "prereserve_doi": {"doi": binding["version_doi"]}}
    session = _Session()
    session.gets = [_Response(draft)]

    state = recover(
        session,
        version_record_id,
        metadata,
        release_binding=binding,
    )

    assert session.posts == []
    assert state["deposition_id"] == version_record_id
    assert state["submitted"] is False
    assert state["files"] == []
    assert state["release_binding"]["metadata_sha256"] == binding["metadata_sha256"]
    state_path = tmp_path / "recovered-state.json"
    write_state(state_path, state)
    assert state_path.stat().st_mode & 0o777 == 0o600
    state = load_state(state_path)

    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"recovered draft bundle")
    uploaded = _draft_file(binding, bundle.name)
    session.gets = [
        _Response(draft),
        _Response(_deposition_payload(binding, files=[uploaded])),
    ]
    session.puts = [_Response({"checksum": "md5:fixture"})]
    state = upload(session, state, [bundle], release_binding=binding)

    remote_draft = dict(draft)
    remote_draft["files"] = [
        {
            "filename": bundle.name,
            "size": bundle.stat().st_size,
            "links": {
                "download": (
                    f"https://zenodo.org/api/records/{version_record_id}/files/bundle.tar.gz/content"
                )
            },
        }
    ]
    session.gets = [
        _Response(remote_draft),
        _Response({}, content=bundle.read_bytes()),
    ]
    report = verify(session, state, metadata, release_binding=binding)
    assert report["status"] == "pass", report


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("deposition", "requested deposition ID"),
        ("concept", "concept DOI"),
        ("version", "version DOI"),
        ("source", "metadata.related_identifiers"),
        ("metadata", "metadata.title"),
        ("missing_submitted", "submitted state"),
        ("invalid_submitted", "submitted state"),
        ("published", "unpublished draft"),
    ],
)
def test_recover_rejects_draft_identity_metadata_and_state_drift(drift: str, error: str) -> None:
    """Recovery fails closed before writing state when the remote draft drifts."""
    binding, metadata = _binding_and_metadata()
    version_record_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    draft = _deposition_payload(binding)
    draft["metadata"] = {**metadata, "prereserve_doi": {"doi": binding["version_doi"]}}
    if drift == "deposition":
        draft["id"] = version_record_id + 1
    elif drift == "concept":
        draft["conceptrecid"] = "999999"
    elif drift == "version":
        draft["metadata"]["prereserve_doi"] = {"doi": "10.5281/zenodo.999999"}
    elif drift == "source":
        draft["metadata"]["related_identifiers"] = [
            {
                "identifier": "https://github.com/ll7/robot_sf_ll7/releases/tag/other",
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ]
    elif drift == "metadata":
        draft["metadata"]["title"] = "Different release"
    elif drift == "missing_submitted":
        draft.pop("submitted")
    elif drift == "invalid_submitted":
        draft["submitted"] = "false"
    else:
        draft["submitted"] = True
        draft["state"] = "done"
        draft["doi"] = binding["version_doi"]
    session = _Session()
    session.gets = [_Response(draft)]

    with pytest.raises(ZenodoPublisherError, match=error):
        recover(session, version_record_id, metadata, release_binding=binding)
    assert session.posts == []


def _repair_remote(
    binding: dict[str, Any], metadata: dict[str, Any], *, aliases: bool = False
) -> dict[str, Any]:
    """Build an empty draft with only the two intended frozen-field changes."""
    remote = _deposition_payload(binding)
    remote_metadata = {
        **metadata,
        "prereserve_doi": {"doi": binding["version_doi"]},
        "version": "0.0.6",
        "publication_date": "2026-09-23",
        "related_identifiers": [
            {
                "identifier": _OLD_SOURCE_TAG,
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ],
    }
    if aliases:
        remote_metadata["license"] = "gpl-3.0"
        remote_metadata["creators"] = [
            {**creator, "affiliation": None} for creator in metadata["creators"]
        ]
    remote["metadata"] = remote_metadata
    return remote


def _provenance_repair_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create a realistic frozen metadata contract with tag and commit links."""
    binding, metadata = _binding_and_metadata()
    target_source_sha = "a" * 40
    target_base_sha = "b" * 40
    target_tag = f"reviewed-candidate-{target_source_sha}"
    resolved = deepcopy({key: value for key, value in metadata.items() if key != "prereserve_doi"})
    resolved["description"] = (
        "Benchmark-data snapshot for Robot SF covering 14 planner arms, 48 social-navigation "
        "scenarios, and 30 seeds at horizon 600 with dt=0.1 and differential-drive kinematics "
        "(20,160 expected episode identities). The archive preserves raw episode and "
        "component-metric evidence with exact source provenance: source commit "
        f"{target_source_sha}, mainline base {target_base_sha}, release tag {target_tag}, "
        f"concept DOI {binding['concept_doi']}, and version DOI {binding['version_doi']}. "
        "The Social Navigation Quality Index (SNQI) is advisory only because calibration failed; "
        "this release makes no SNQI ranking claim."
    )
    resolved["related_identifiers"] = [
        {
            "identifier": f"https://github.com/ll7/robot_sf_ll7/releases/tag/{target_tag}",
            "relation": "isSupplementTo",
            "scheme": "url",
        },
        {
            "identifier": f"https://github.com/ll7/robot_sf_ll7/commit/{target_source_sha}",
            "relation": "isDerivedFrom",
            "scheme": "url",
        },
    ]
    path = tmp_path / "zenodo_metadata.resolved.json"
    serialized = json.dumps({"metadata": resolved}, ensure_ascii=False, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")
    binding = {
        **binding,
        "release_tag": target_tag,
        "metadata_path": path,
        "metadata_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }
    target_metadata = {**resolved, "prereserve_doi": True}
    old_source_sha = "c" * 40
    old_base_sha = "d" * 40
    old_tag = f"previous-candidate-{old_source_sha}"
    old_description = (
        target_metadata["description"]
        .replace(target_tag, old_tag)
        .replace(target_source_sha, old_source_sha)
        .replace(target_base_sha, old_base_sha)
    )
    remote = _deposition_payload(binding)
    remote["metadata"] = {
        **target_metadata,
        "prereserve_doi": {"doi": binding["version_doi"]},
        "version": "0.0.6",
        "publication_date": "2026-09-23",
        "description": old_description,
        "related_identifiers": [
            {
                "identifier": f"https://github.com/ll7/robot_sf_ll7/releases/tag/{old_tag}",
                "relation": "isSupplementTo",
                "scheme": "url",
            },
            {
                "identifier": f"https://github.com/ll7/robot_sf_ll7/commit/{old_source_sha}",
                "relation": "isDerivedFrom",
                "scheme": "url",
            },
        ],
    }
    return binding, target_metadata, remote


def test_repair_draft_metadata_previews_get_first_then_puts_and_verifies_exact_readback() -> None:
    """Repair previews are read-only and writes are followed by exact remote readback."""
    binding, metadata = _binding_and_metadata()
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    remote = _repair_remote(binding, metadata, aliases=True)
    target_metadata = {
        **{key: value for key, value in metadata.items() if key != "prereserve_doi"},
        "version": "0.0.7",
        "publication_date": "2026-09-24",
    }

    preview_session = _Session()
    preview_session.gets = [_Response(remote)]
    preview = repair_draft_metadata(
        preview_session,
        deposition_id,
        metadata,
        version="0.0.7",
        publication_date="2026-09-24",
        release_binding=binding,
        apply=False,
    )

    assert preview["status"] == "ready"
    assert preview["changed_fields"] == [
        "publication_date",
        "related_identifiers",
        "version",
    ]
    assert preview_session.calls[0][0] == "GET"
    assert len(preview_session.calls) == 1
    assert preview_session.puts == []
    expected_digest = preview["remote_metadata_sha256_before"]

    repaired_remote = _deposition_payload(binding)
    repaired_remote["metadata"] = {
        **target_metadata,
        "prereserve_doi": {"doi": binding["version_doi"]},
    }
    apply_session = _Session()
    apply_session.gets = [_Response(remote), _Response(repaired_remote)]
    apply_session.puts = [_Response(repaired_remote)]

    result = repair_draft_metadata(
        apply_session,
        deposition_id,
        metadata,
        version="0.0.7",
        publication_date="2026-09-24",
        release_binding=binding,
        expected_remote_metadata_sha256=expected_digest,
        expected_remote_source_tag=_OLD_SOURCE_TAG,
        apply=True,
    )

    assert result["status"] == "repaired"
    assert result["changed_fields"] == [
        "publication_date",
        "related_identifiers",
        "version",
    ]
    assert result["remote_metadata_sha256_after"]
    assert [call[0] for call in apply_session.calls] == ["GET", "PUT", "GET"]
    assert apply_session.calls[1][2]["json"] == {"metadata": target_metadata}
    assert apply_session.gets == []


def test_repair_draft_metadata_reviews_exact_two_link_provenance_before_apply(
    tmp_path: Path,
) -> None:
    """A real two-link source correction is explicit in preview and apply bindings."""
    binding, metadata, remote = _provenance_repair_fixture(tmp_path)
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    preview_session = _Session()
    preview_session.gets = [_Response(remote)]

    preview = repair_draft_metadata(
        preview_session,
        deposition_id,
        metadata,
        version="0.0.7",
        publication_date="2026-09-24",
        release_binding=binding,
    )

    assert preview["status"] == "ready"
    assert preview["remote_source_sha_before"] == "c" * 40
    assert preview["remote_base_sha_before"] == "d" * 40
    assert preview["source_sha_after"] == "a" * 40
    assert preview["base_sha_after"] == "b" * 40
    assert preview["metadata_diff"]["description"]["before"] == remote["metadata"]["description"]
    assert preview["metadata_diff"]["description"]["after"] == metadata["description"]
    assert (
        preview["metadata_diff"]["related_identifiers"]["before"]
        == remote["metadata"]["related_identifiers"]
    )
    assert preview_session.puts == []

    repaired = _deposition_payload(binding)
    repaired["metadata"] = {
        **{key: value for key, value in metadata.items() if key != "prereserve_doi"},
        "version": "0.0.7",
        "publication_date": "2026-09-24",
        "prereserve_doi": {"doi": binding["version_doi"]},
    }
    apply_session = _Session()
    apply_session.gets = [_Response(remote), _Response(repaired)]
    apply_session.puts = [_Response(repaired)]

    result = repair_draft_metadata(
        apply_session,
        deposition_id,
        metadata,
        version="0.0.7",
        publication_date="2026-09-24",
        release_binding=binding,
        expected_remote_metadata_sha256=preview["remote_metadata_sha256_before"],
        expected_remote_source_tag=preview["remote_source_tag_before"],
        expected_remote_source_sha=preview["remote_source_sha_before"],
        expected_remote_base_sha=preview["remote_base_sha_before"],
        apply=True,
    )

    assert result["status"] == "repaired"
    assert [call[0] for call in apply_session.calls] == ["GET", "PUT", "GET"]


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("extra_relation", "inventory drift"),
        ("missing_commit_relation", "inventory drift"),
        ("modified_commit_relation", "source commit relation"),
        ("tag_sha_mismatch", "disagrees with its source commit"),
        ("unstructured_description", "outside source provenance"),
        ("scenario_count", "outside source provenance"),
        ("scientific_prose", "outside source provenance"),
        ("description_concept_doi", "source-provenance DOI"),
        ("description_version_doi", "source-provenance DOI"),
    ],
)
def test_repair_draft_metadata_rejects_unreviewed_source_drift_without_put(
    tmp_path: Path, drift: str, error: str
) -> None:
    """Unrelated prose, DOI, and source-inventory changes fail before any remote write."""
    binding, metadata, remote = _provenance_repair_fixture(tmp_path)
    remote = deepcopy(remote)
    identifiers = remote["metadata"]["related_identifiers"]
    description = remote["metadata"]["description"]
    if drift == "extra_relation":
        identifiers.append(
            {
                "identifier": "https://example.org/unreviewed",
                "relation": "isReferencedBy",
                "scheme": "url",
            }
        )
    elif drift == "missing_commit_relation":
        identifiers.pop()
    elif drift == "modified_commit_relation":
        identifiers[1]["identifier"] = "https://example.org/unreviewed"
    elif drift == "tag_sha_mismatch":
        bad_tag = f"previous-candidate-{'e' * 40}"
        remote["metadata"]["related_identifiers"][0]["identifier"] = (
            f"https://github.com/ll7/robot_sf_ll7/releases/tag/{bad_tag}"
        )
        remote["metadata"]["description"] = description.replace(
            "previous-candidate-" + "c" * 40, bad_tag
        )
    elif drift == "unstructured_description":
        remote["metadata"]["description"] = "A changed description with no reviewed provenance."
    elif drift == "scenario_count":
        remote["metadata"]["description"] = description.replace(
            "14 planner arms", "15 planner arms"
        )
    elif drift == "scientific_prose":
        remote["metadata"]["description"] = description.replace(
            "SNQI) is advisory only", "SNQI) establishes the definitive ranking"
        )
    elif drift == "description_concept_doi":
        remote["metadata"]["description"] = description.replace(
            binding["concept_doi"], "10.5281/zenodo.999999"
        )
    else:
        remote["metadata"]["description"] = description.replace(
            binding["version_doi"], "10.5281/zenodo.999998"
        )
    session = _Session()
    session.gets = [_Response(remote)]

    with pytest.raises(ZenodoPublisherError, match=error):
        repair_draft_metadata(
            session,
            int(binding["version_doi"].rsplit(".", 1)[-1]),
            metadata,
            version="0.0.7",
            publication_date="2026-09-24",
            release_binding=binding,
        )

    assert [call[0] for call in session.calls] == ["GET"]
    assert session.puts == []


@pytest.mark.parametrize(
    ("reviewed_value", "expected_error"),
    [
        ("source_sha", "reviewed remote source SHA"),
        ("base_sha", "reviewed remote mainline base SHA"),
    ],
)
def test_repair_draft_metadata_apply_requires_review_of_each_changed_provenance_value(
    tmp_path: Path, reviewed_value: str, expected_error: str
) -> None:
    """A source-value diff cannot be applied based on the tag and digest alone."""
    binding, metadata, remote = _provenance_repair_fixture(tmp_path)
    preview_session = _Session()
    preview_session.gets = [_Response(remote)]
    preview = repair_draft_metadata(
        preview_session,
        int(binding["version_doi"].rsplit(".", 1)[-1]),
        metadata,
        version="0.0.7",
        publication_date="2026-09-24",
        release_binding=binding,
    )
    apply_session = _Session()
    apply_session.gets = [_Response(remote)]
    expected_source_sha = (
        preview["remote_source_sha_before"] if reviewed_value == "base_sha" else None
    )
    expected_base_sha = (
        preview["remote_base_sha_before"] if reviewed_value == "source_sha" else None
    )

    with pytest.raises(ZenodoPublisherError, match=expected_error):
        repair_draft_metadata(
            apply_session,
            int(binding["version_doi"].rsplit(".", 1)[-1]),
            metadata,
            version="0.0.7",
            publication_date="2026-09-24",
            release_binding=binding,
            expected_remote_metadata_sha256=preview["remote_metadata_sha256_before"],
            expected_remote_source_tag=preview["remote_source_tag_before"],
            expected_remote_source_sha=expected_source_sha,
            expected_remote_base_sha=expected_base_sha,
            apply=True,
        )

    assert [call[0] for call in apply_session.calls] == ["GET"]
    assert apply_session.puts == []


def test_repair_draft_metadata_stale_remote_digest_blocks_put() -> None:
    """An apply attempt with a stale preview digest cannot issue a PUT."""
    binding, metadata = _binding_and_metadata()
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    session = _Session()
    session.gets = [_Response(_repair_remote(binding, metadata))]

    with pytest.raises(ZenodoPublisherError, match="changed since the repair preview"):
        repair_draft_metadata(
            session,
            deposition_id,
            metadata,
            version="0.0.7",
            publication_date="2026-09-24",
            release_binding=binding,
            expected_remote_metadata_sha256="0" * 64,
            expected_remote_source_tag=_OLD_SOURCE_TAG,
            apply=True,
        )

    assert [call[0] for call in session.calls] == ["GET"]
    assert session.puts == []


def test_repair_draft_metadata_apply_requires_and_checks_reviewed_source_tag() -> None:
    """An apply needs the previewed source identity and rejects a different reviewed tag."""
    binding, metadata = _binding_and_metadata()
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    remote = _repair_remote(binding, metadata)

    missing_tag_session = _Session()
    with pytest.raises(ZenodoPublisherError, match="reviewed remote source tag"):
        repair_draft_metadata(
            missing_tag_session,
            deposition_id,
            metadata,
            version="0.0.7",
            publication_date="2026-09-24",
            release_binding=binding,
            expected_remote_metadata_sha256="a" * 64,
            apply=True,
        )
    assert missing_tag_session.calls == []

    preview_session = _Session()
    preview_session.gets = [_Response(remote)]
    preview = repair_draft_metadata(
        preview_session,
        deposition_id,
        metadata,
        version="0.0.7",
        publication_date="2026-09-24",
        release_binding=binding,
    )
    apply_session = _Session()
    apply_session.gets = [_Response(remote)]

    with pytest.raises(ZenodoPublisherError, match="source tag differs from the reviewed preview"):
        repair_draft_metadata(
            apply_session,
            deposition_id,
            metadata,
            version="0.0.7",
            publication_date="2026-09-24",
            release_binding=binding,
            expected_remote_metadata_sha256=preview["remote_metadata_sha256_before"],
            expected_remote_source_tag="https://github.com/ll7/robot_sf_ll7/releases/tag/other",
            apply=True,
        )

    assert [call[0] for call in apply_session.calls] == ["GET"]
    assert apply_session.puts == []


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("deposition", "deposition ID"),
        ("concept", "concept DOI"),
        ("version_doi", "version DOI"),
        ("published", "unpublished draft"),
        ("files", "empty draft file inventory"),
        ("title", "unrelated drift"),
    ],
)
def test_repair_draft_metadata_rejects_identity_lifecycle_inventory_and_unrelated_drift(
    drift: str, error: str
) -> None:
    """The repair gate accepts only a matching empty unpublished release draft."""
    binding, metadata = _binding_and_metadata()
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    remote = _repair_remote(binding, metadata)
    if drift == "deposition":
        remote["id"] = deposition_id + 1
    elif drift == "concept":
        remote["conceptrecid"] = "999999"
    elif drift == "version_doi":
        remote["metadata"]["prereserve_doi"] = {"doi": "10.5281/zenodo.999999"}
    elif drift == "published":
        remote["submitted"] = True
        remote["state"] = "done"
        remote["doi"] = binding["version_doi"]
    elif drift == "files":
        remote["files"] = [{"filename": "unexpected.tar.gz"}]
    else:
        remote["metadata"]["title"] = "Different release"
    session = _Session()
    session.gets = [_Response(remote)]

    with pytest.raises(ZenodoPublisherError, match=error):
        repair_draft_metadata(
            session,
            deposition_id,
            metadata,
            version="0.0.7",
            publication_date="2026-09-24",
            release_binding=binding,
        )

    assert [call[0] for call in session.calls] == ["GET"]
    assert session.puts == []


def test_verify_and_publish_bind_operational_metadata_into_verification_receipt(
    tmp_path: Path,
) -> None:
    """Version/date are checked during verification and sealed into publish admission."""
    binding, metadata = _binding_and_metadata()
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"manifest-bound bundle")
    state = _bound_uploaded_state(binding, metadata, bundle)
    remote = _operational_draft_payload(binding, metadata, bundle)
    operational_metadata = {"version": "0.0.7", "publication_date": "2026-09-24"}
    session = _Session()
    session.gets = [
        _Response(remote),
        _Response({}, content=bundle.read_bytes()),
    ]

    report = verify(
        session,
        state,
        metadata,
        release_binding=binding,
        expected_operational_metadata=operational_metadata,
    )

    assert report["status"] == "pass", report
    receipt = report["receipt"]
    assert receipt["operational_metadata"] == operational_metadata
    _verify_integrity(receipt, key="integrity", schema=ZENODO_VERIFICATION_SCHEMA)
    assert state["verification_receipt"] == receipt

    published = _deposition_payload(binding, submitted=True)
    session.gets = [
        _Response(remote),
        _Response({}, content=bundle.read_bytes()),
    ]
    session.posts = [_Response(published)]
    published_state = publish(
        session,
        state,
        metadata,
        release_binding=binding,
        expected_operational_metadata=operational_metadata,
    )

    assert published_state["submitted"] is True
    assert published_state["verification_receipt"]["operational_metadata"] == operational_metadata
    _verify_integrity(
        published_state["verification_receipt"],
        key="integrity",
        schema=ZENODO_VERIFICATION_SCHEMA,
    )


@pytest.mark.parametrize("field", ["version", "publication_date"])
def test_verify_rejects_operational_metadata_drift(field: str, tmp_path: Path) -> None:
    """A passing receipt requires both expected Zenodo-only publication fields."""
    binding, metadata = _binding_and_metadata()
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"manifest-bound bundle")
    state = _bound_uploaded_state(binding, metadata, bundle)
    remote = _operational_draft_payload(binding, metadata, bundle)
    remote["metadata"][field] = "0.0.6" if field == "version" else "2026-09-23"
    state_before = json.dumps(state, sort_keys=True)
    session = _Session()
    session.gets = [
        _Response(remote),
        _Response({}, content=bundle.read_bytes()),
    ]

    report = verify(
        session,
        state,
        metadata,
        release_binding=binding,
        expected_operational_metadata={
            "version": "0.0.7",
            "publication_date": "2026-09-24",
        },
    )

    assert report["status"] == "fail"
    assert any(f"metadata.{field}" in problem for problem in report["problems"])
    assert json.dumps(state, sort_keys=True) == state_before
    assert "verification_receipt" not in state


def test_bound_zenodo_operation_rejects_metadata_checksum_mismatch() -> None:
    """A binding with a stale checksum cannot reach the Zenodo API."""
    binding, metadata = _binding_and_metadata()
    version_record_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    binding["metadata_sha256"] = "0" * 64
    session = _Session()

    with pytest.raises(ZenodoPublisherError, match="metadata file SHA-256"):
        reserve(session, metadata, release_binding=binding)
    with pytest.raises(ZenodoPublisherError, match="metadata file SHA-256"):
        recover(session, version_record_id, metadata, release_binding=binding)
    assert session.posts == []
    assert session.gets == []


def test_bound_reserve_rejects_concept_or_version_identity_drift() -> None:
    """Reserved DOI identity must match both manifest DOI fields exactly."""
    binding, metadata = _binding_and_metadata()
    session = _Session()
    response = _deposition_payload(binding)
    response["conceptrecid"] = "999999"
    session.posts = [_Response(response)]

    with pytest.raises(ZenodoPublisherError, match="concept DOI"):
        reserve(session, metadata, release_binding=binding)


@pytest.mark.parametrize("receipt_kind", ["missing", "stale"])
def test_publish_failure_preserves_bound_caller_state(
    receipt_kind: str,
) -> None:
    """Receipt admission failures cannot mutate or invalidate caller state."""
    binding, metadata = _binding_and_metadata()
    deposition_id = int(binding["version_doi"].rsplit(".", 1)[-1])
    concept_record_id = binding["concept_doi"].rsplit(".", 1)[-1]
    state_payload: dict[str, Any] = {
        "schema_version": ZENODO_STATE_SCHEMA,
        "deposition_id": deposition_id,
        "record_id": deposition_id,
        "concept_record_id": concept_record_id,
        "doi": binding["version_doi"],
        "submitted": False,
        "state": "unsubmitted",
        "files": [{"name": "bundle.tar.gz", "size": 1, "sha256": "0" * 64}],
    }
    if receipt_kind == "stale":
        state_payload["verification_receipt"] = {
            "status": "pass",
            "publication_state": "draft",
        }
    state = _seal_state(state_payload)
    state_before = json.loads(json.dumps(state, sort_keys=True))
    session = _Session()

    expected_error = "verification receipt" if receipt_kind == "missing" else "integrity"
    with pytest.raises(ZenodoPublisherError, match=expected_error):
        publish(session, state, metadata, release_binding=binding)

    assert state == state_before
    assert session.gets == []
    assert session.posts == []
    _verify_integrity(state, key="integrity", schema=ZENODO_STATE_SCHEMA)


def test_bound_upload_failure_preserves_unbound_caller_state(tmp_path: Path) -> None:
    """Upload binding adoption is discarded when the remote bucket is rejected."""
    binding, _ = _binding_and_metadata()
    state = _unbound_state(binding)
    state_before = json.dumps(state, sort_keys=True)
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    draft = _deposition_payload(binding)
    draft["links"]["bucket"] = "http://zenodo.org/api/files/bucket"
    session = _Session()
    session.gets = [_Response(draft)]

    with pytest.raises(ZenodoPublisherError, match="secure upload bucket"):
        upload(session, state, [bundle], release_binding=binding)

    assert json.dumps(state, sort_keys=True) == state_before
    assert "release_binding" not in state
    _verify_integrity(state, key="integrity", schema=ZENODO_STATE_SCHEMA)


def test_bound_verify_failure_preserves_unbound_caller_state() -> None:
    """Verify binding adoption is discarded when the remote lookup fails."""
    binding, metadata = _binding_and_metadata()
    state = _unbound_state(binding)
    state_before = json.dumps(state, sort_keys=True)
    session = _Session()
    session.gets = [_Response({}, status_code=503)]

    with pytest.raises(ZenodoPublisherError, match="verify request failed"):
        verify(session, state, metadata, release_binding=binding)

    assert json.dumps(state, sort_keys=True) == state_before
    assert "release_binding" not in state
    _verify_integrity(state, key="integrity", schema=ZENODO_STATE_SCHEMA)


def test_state_shape_for_binding_remains_credential_free() -> None:
    """Manifest binding state contains no token-shaped field or value."""
    state = _seal_state(
        {
            "schema_version": ZENODO_STATE_SCHEMA,
            "deposition_id": 1,
            "record_id": 2,
            "concept_record_id": "3",
            "doi": "10.5281/zenodo.4",
            "submitted": False,
            "state": "unsubmitted",
            "files": [],
        }
    )
    assert "token" not in json.dumps(state).casefold()
