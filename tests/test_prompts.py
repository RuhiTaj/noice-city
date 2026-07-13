from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from voice_trainer.prompts import DEFAULT_PROMPTS_PATH, PromptCatalog


V1_ASSIGNMENT_SHA256 = "5b7415e7b7c097f52b0df0a4c97a393bb765dcd531ac85d7f964b3f944fc81e5"


def _payload() -> dict:
    return json.loads(DEFAULT_PROMPTS_PATH.read_text(encoding="utf-8"))


def _v1_payload() -> dict:
    payload = _payload()
    payload["version"] = 1
    payload["prompts"] = [
        {key: value for key, value in item.items() if key != "split"}
        for item in payload["prompts"][:100]
    ]
    return payload


def _write_catalog(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _assignment_hash(catalog: PromptCatalog) -> str:
    assignments = [(prompt.id, prompt.split) for prompt in catalog.prompts]
    encoded = json.dumps(assignments, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_v1_implicit_catalog_keeps_exact_legacy_hash_assignment(tmp_path: Path) -> None:
    v1_path = _write_catalog(tmp_path, _v1_payload())
    catalog = PromptCatalog.load(v1_path)
    assert _assignment_hash(catalog) == V1_ASSIGNMENT_SHA256

    payload = _v1_payload()
    payload["version"] = 2
    second_path = tmp_path / "v2-prompts.json"
    second_path.write_text(json.dumps(payload), encoding="utf-8")
    v2_implicit = PromptCatalog.load(second_path)
    assert _assignment_hash(v2_implicit) == V1_ASSIGNMENT_SHA256

    legacy_ids = {item["id"] for item in payload["prompts"]}
    deployed_v2 = PromptCatalog.load()
    deployed_legacy = [
        (prompt.id, prompt.split)
        for prompt in deployed_v2.prompts
        if prompt.id in legacy_ids
    ]
    encoded = json.dumps(deployed_legacy, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == V1_ASSIGNMENT_SHA256


def test_v2_accepts_fully_explicit_catalog(tmp_path: Path) -> None:
    payload = _payload()
    assigned = {prompt.id: prompt.split for prompt in PromptCatalog.load().prompts}
    payload["version"] = 2
    for item in payload["prompts"]:
        item["split"] = assigned[item["id"]]

    catalog = PromptCatalog.load(_write_catalog(tmp_path, payload))

    assert {prompt.id: prompt.split for prompt in catalog.prompts} == assigned
    expected = Counter(assigned.values())
    assert catalog.summary()["splits"] == dict(expected)


def test_v2_mixed_catalog_preserves_explicit_and_stratifies_implicit(
    tmp_path: Path,
) -> None:
    payload = _v1_payload()
    payload["version"] = 2
    guided = [item for item in payload["prompts"] if item["kind"] == "guided"]
    explicit = {
        **{item["id"]: "train" for item in guided[:8]},
        guided[8]["id"]: "validation",
        guided[9]["id"]: "test",
    }
    for item in payload["prompts"]:
        if item["id"] in explicit:
            item["split"] = explicit[item["id"]]

    catalog = PromptCatalog.load(_write_catalog(tmp_path, payload))

    for prompt_id, split in explicit.items():
        assert catalog.get(prompt_id).split == split
    assert Counter(prompt.split for prompt in catalog.prompts) == {
        "train": 80,
        "validation": 10,
        "test": 10,
    }
    for kind, expected in (
        ("guided", {"train": 64, "validation": 8, "test": 8}),
        ("interview", {"train": 16, "validation": 2, "test": 2}),
    ):
        assert Counter(
            prompt.split for prompt in catalog.prompts if prompt.kind == kind
        ) == expected


def test_v1_rejects_explicit_splits(tmp_path: Path) -> None:
    payload = _v1_payload()
    payload["prompts"][0]["split"] = "train"
    with pytest.raises(ValueError, match="version 2"):
        PromptCatalog.load(_write_catalog(tmp_path, payload))


@pytest.mark.parametrize("invalid", ["", "val", None, 1, [], {}])
def test_catalog_rejects_invalid_explicit_split_values(
    tmp_path: Path, invalid
) -> None:
    payload = _v1_payload()
    payload["version"] = 2
    payload["prompts"][0]["split"] = invalid
    with pytest.raises(ValueError, match="Prompt split"):
        PromptCatalog.load(_write_catalog(tmp_path, payload))


def test_mixed_catalog_rejects_implicit_kind_not_divisible_by_ten(
    tmp_path: Path,
) -> None:
    payload = _v1_payload()
    payload["version"] = 2
    payload["prompts"][0]["split"] = "train"
    with pytest.raises(ValueError, match="Implicit guided prompts"):
        PromptCatalog.load(_write_catalog(tmp_path, payload))


def test_mixed_catalog_rejects_malformed_explicit_distribution(
    tmp_path: Path,
) -> None:
    payload = _v1_payload()
    payload["version"] = 2
    guided = [item for item in payload["prompts"] if item["kind"] == "guided"]
    for index, item in enumerate(guided[:10]):
        item["split"] = "train" if index < 7 else "validation" if index < 9 else "test"

    with pytest.raises(ValueError, match="exactly 80% train"):
        PromptCatalog.load(_write_catalog(tmp_path, payload))
