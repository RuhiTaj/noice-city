"""Versioned prompt catalog with stable pre-recording dataset splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "assets" / "prompts.json"
VALID_KINDS = frozenset({"guided", "interview"})
VALID_STYLES = frozenset({"neutral", "expressive"})
VALID_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class Prompt:
    id: str
    kind: str
    style: str
    text: str
    split: str = ""
    ordinal: int = 0

    @property
    def locked(self) -> bool:
        return self.split == "test"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["locked"] = self.locked
        return value


class PromptCatalog:
    """Load, validate, and deterministically split the bundled prompt corpus."""

    def __init__(self, prompts: Iterable[Prompt], *, seed: str, version: int = 1) -> None:
        raw = list(prompts)
        self.seed = seed
        self.version = version
        self._validate(raw)
        assigned = self._assign_splits(raw)
        self._validate_final_splits(raw, assigned)
        self.prompts = tuple(assigned)
        self._by_id = {prompt.id: prompt for prompt in self.prompts}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROMPTS_PATH) -> "PromptCatalog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload["version"]
        for item in payload["prompts"]:
            if "split" in item and (
                not isinstance(item["split"], str)
                or item["split"] not in VALID_SPLITS
            ):
                raise ValueError(
                    "Prompt split must be train, validation, or test when provided"
                )
        prompts = [
            Prompt(
                id=item["id"],
                kind=item["kind"],
                style=item["style"],
                text=item["text"],
                split=item.get("split", ""),
                ordinal=index,
            )
            for index, item in enumerate(payload["prompts"])
        ]
        return cls(prompts, seed=payload["seed"], version=version)

    def _validate(self, prompts: list[Prompt]) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("Prompt catalog version must be a positive integer")
        if not prompts:
            raise ValueError("The prompt catalog cannot be empty")
        ids = [prompt.id for prompt in prompts]
        if len(ids) != len(set(ids)):
            raise ValueError("Prompt ids must be unique")
        if any(prompt.kind not in VALID_KINDS for prompt in prompts):
            raise ValueError("Prompt kind must be guided or interview")
        if any(prompt.style not in VALID_STYLES for prompt in prompts):
            raise ValueError("Prompt style must be neutral or expressive")
        if any(not prompt.text.strip() for prompt in prompts):
            raise ValueError("Prompt text cannot be blank")
        invalid_splits = sorted({
            repr(prompt.split)
            for prompt in prompts
            if not isinstance(prompt.split, str)
            or (prompt.split != "" and prompt.split not in VALID_SPLITS)
        })
        if invalid_splits:
            raise ValueError(
                "Prompt split must be train, validation, or test when provided: "
                + ", ".join(invalid_splits)
            )
        explicit = [prompt for prompt in prompts if prompt.split in VALID_SPLITS]
        if explicit and self.version < 2:
            raise ValueError("Explicit prompt splits require catalog version 2 or newer")
        kind_counts = Counter(prompt.kind for prompt in prompts)
        style_counts = Counter(prompt.style for prompt in prompts)
        total = len(prompts)
        if kind_counts["guided"] * 5 != total * 4:
            raise ValueError("The catalog must contain exactly 80% guided prompts")
        if style_counts["neutral"] * 10 != total * 9:
            raise ValueError("The catalog must contain exactly 90% neutral prompts")
        for count in kind_counts.values():
            if count % 10:
                raise ValueError("Each prompt kind must be divisible into an 80/10/10 split")

        implicit = [prompt for prompt in prompts if prompt.split == ""]
        for kind in VALID_KINDS:
            implicit_count = sum(prompt.kind == kind for prompt in implicit)
            if implicit_count and implicit_count % 10:
                raise ValueError(
                    f"Implicit {kind} prompts must be divisible into an 80/10/10 split"
                )

        self._validate_distribution(explicit, "Explicit prompts")
        for kind in VALID_KINDS:
            self._validate_distribution(
                [prompt for prompt in explicit if prompt.kind == kind],
                f"Explicit {kind} prompts",
            )

    @staticmethod
    def _expected_split_counts(total: int) -> dict[str, int]:
        return {
            "train": total * 8 // 10,
            "validation": total // 10,
            "test": total // 10,
        }

    @classmethod
    def _validate_distribution(cls, prompts: list[Prompt], label: str) -> None:
        if not prompts:
            return
        if len(prompts) % 10:
            raise ValueError(f"{label} must be divisible into an 80/10/10 split")
        actual = Counter(prompt.split for prompt in prompts)
        expected = cls._expected_split_counts(len(prompts))
        if any(actual[split] != count for split, count in expected.items()):
            raise ValueError(
                f"{label} must contain exactly 80% train, "
                "10% validation, and 10% test"
            )

    def _assign_splits(self, prompts: list[Prompt]) -> list[Prompt]:
        by_kind: dict[str, list[Prompt]] = defaultdict(list)
        for prompt in prompts:
            if prompt.split == "":
                by_kind[prompt.kind].append(prompt)
        assigned: dict[str, str] = {
            prompt.id: prompt.split for prompt in prompts if prompt.split in VALID_SPLITS
        }
        for kind, group in by_kind.items():
            ordered = sorted(
                group,
                key=lambda prompt: hashlib.sha256(
                    f"{self.seed}:{kind}:{prompt.id}".encode("utf-8")
                ).digest(),
            )
            train_end = len(ordered) * 8 // 10
            validation_end = len(ordered) * 9 // 10
            for index, prompt in enumerate(ordered):
                assigned[prompt.id] = (
                    "train" if index < train_end else "validation" if index < validation_end else "test"
                )
        return [replace(prompt, split=assigned[prompt.id]) for prompt in prompts]

    def _validate_final_splits(
        self, raw: list[Prompt], assigned: list[Prompt]
    ) -> None:
        self._validate_distribution(assigned, "Final prompt catalog")
        for kind in VALID_KINDS:
            self._validate_distribution(
                [prompt for prompt in assigned if prompt.kind == kind],
                f"Final {kind} prompts",
            )

        implicit_ids = {prompt.id for prompt in raw if prompt.split == ""}
        implicit_assigned = [prompt for prompt in assigned if prompt.id in implicit_ids]
        self._validate_distribution(implicit_assigned, "Implicit prompts")
        for kind in VALID_KINDS:
            self._validate_distribution(
                [prompt for prompt in implicit_assigned if prompt.kind == kind],
                f"Implicit {kind} prompts",
            )

    def get(self, prompt_id: str) -> Prompt:
        try:
            return self._by_id[prompt_id]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt: {prompt_id}") from exc

    def select(
        self,
        *,
        split: str | None = None,
        kind: str | None = None,
        style: str | None = None,
        exclude: Iterable[str] = (),
        chooser_seed: str = "default",
    ) -> Prompt | None:
        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if kind is not None and kind not in VALID_KINDS:
            raise ValueError(f"Unknown kind: {kind}")
        if style is not None and style not in VALID_STYLES:
            raise ValueError(f"Unknown style: {style}")
        excluded = set(exclude)
        eligible = [
            prompt
            for prompt in self.prompts
            if prompt.id not in excluded
            and (split is None or prompt.split == split)
            and (kind is None or prompt.kind == kind)
            and (style is None or prompt.style == style)
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda prompt: hashlib.sha256(
                f"{self.seed}:{chooser_seed}:{prompt.id}".encode("utf-8")
            ).digest(),
        )

    def summary(self) -> dict[str, object]:
        return {
            "version": self.version,
            "total": len(self.prompts),
            "kinds": dict(Counter(prompt.kind for prompt in self.prompts)),
            "styles": dict(Counter(prompt.style for prompt in self.prompts)),
            "splits": dict(Counter(prompt.split for prompt in self.prompts)),
        }
