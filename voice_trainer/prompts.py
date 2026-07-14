"""Versioned prompt catalog with stable pre-recording dataset splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "assets" / "prompts.json"
US_ENGLISH_PACK = "en-US"
INDIAN_ENGLISH_PACK = "en-IN"
DEFAULT_PROMPT_PACK = US_ENGLISH_PACK
VALID_PROMPT_PACKS = frozenset({US_ENGLISH_PACK, INDIAN_ENGLISH_PACK})
VALID_KINDS = frozenset({"guided", "interview"})
VALID_STYLES = frozenset({"neutral", "expressive"})
VALID_SPLITS = frozenset({"train", "validation", "test"})


# The first 100 deployed prompts are neutral international-English coverage and
# are reused verbatim in the US pack under new, pack-scoped IDs. These 50
# replacements turn the deployed India-specific extension into equally sized
# US vocabulary, names, places, numbers, and natural-answer coverage. Keeping
# every deployed ID untouched preserves old recordings, frozen datasets, and
# locked evaluations exactly as they were.
US_PROMPT_TEXT = {
    "guided-081": "Victor wore a white vest while Vivian waited near the west gate.",
    "guided-082": "Three thin threads were tied tightly to the wooden frame.",
    "guided-083": "Those red doors stood beside two dusty roads.",
    "guided-084": "The ship carried six sheep past the green beach.",
    "guided-085": "A full bowl of soup cooled beside the cook's blue stool.",
    "guided-086": "Sam packed the black bag beside Ben's narrow bed.",
    "guided-087": "The cot near the court was bought for a short stay.",
    "guided-088": "The young clerk parked the cart under a large arch.",
    "guided-089": "Fresh vision and careful judgment shaped the final decision.",
    "guided-090": "The judge chose a cheerful jacket for the village journey.",
    "guided-091": "Zoe zipped the silver case and closed the noisy door.",
    "guided-092": "The sixth guest asked for crisp snacks and warm drinks.",
    "guided-093": "She packed maps, saved notes, and sorted labeled pages.",
    "guided-094": "The cats watched as three dogs crossed two bridges.",
    "guided-095": "A faraway train rolled over the rural railway bridge.",
    "guided-096": "Riley recorded a record before presenting the final report.",
    "guided-097": "Please present the present after the project meeting ends.",
    "guided-098": "This minute detail may take a minute to explain.",
    "guided-099": "The bass player noticed a fresh bass beside the boat.",
    "guided-100": "The annual budget rose from two million to three million dollars.",
    "guided-101": "Maya paid two hundred thirty thousand dollars by bank transfer.",
    "guided-102": "The invoice total was one thousand two hundred forty dollars and fifty cents.",
    "guided-103": "The online payment needs a six-digit verification code before noon.",
    "guided-104": "Identity verification finished at eight forty-five on Tuesday morning.",
    "guided-105": "The sales tax rate changed from five percent to twelve percent.",
    "guided-106": "Send the file to demo dot voice at example dot com.",
    "guided-107": "Open version two point five slash beta in the mobile browser.",
    "guided-108": "The Seattle office called the Austin team after lunch.",
    "guided-109": "Avery traveled from Boston to Miami through Atlanta.",
    "guided-110": "The Albuquerque office sent its report to Sacramento, California.",
    "guided-111": "Philadelphia, Baltimore, and Charleston appeared on the coastal route map.",
    "guided-112": "A morning train carried visitors from Portland toward Eugene, Oregon.",
    "guided-113": "Worcester, Massachusetts, and Louisville, Kentucky, require careful pronunciation.",
    "guided-114": "The Minneapolis team welcomed a new colleague from Boise, Idaho.",
    "guided-115": "We shared bagels, barbecue, tacos, cornbread, clam chowder, and apple pie.",
    "guided-116": "I can't join at six, but I'll call you when I'm free.",
    "guided-117": "\"Wait, did you say fifteen or fifty?\" she asked.",
    "guided-118": "Wow! That tiny camera captured the entire night sky.",
    "guided-119": "First call Mom, then message Riley, and please don't forget the tickets.",
    "guided-120": "\"I never expected that result,\" Jordan said with genuine surprise.",
    "interview-021": "Tell me about a city or town in the United States that feels familiar to you.",
    "interview-022": "Describe how your family celebrates a birthday, holiday, or important milestone.",
    "interview-023": "Explain how you use cash, a bank card, or a digital wallet during an ordinary week.",
    "interview-024": "Talk about a meal from home and pronounce each dish exactly as you normally would.",
    "interview-025": "Describe a journey by car, train, bus, subway, or airplane.",
    "interview-026": "Explain a work idea using one dollar amount, one date, and one percentage.",
    "interview-027": "Tell a coworker why you'll be late and suggest a new meeting time.",
    "interview-028": "Describe a technology product, app, or market trend you follow closely.",
    "interview-029": "Talk about a word or place name people sometimes mispronounce and say it naturally.",
    "interview-030": "Give an enthusiastic two-sentence welcome to friends arriving for a celebration.",
}


@dataclass(frozen=True)
class Prompt:
    id: str
    kind: str
    style: str
    text: str
    pack: str = INDIAN_ENGLISH_PACK
    split: str = ""
    ordinal: int = 0

    @property
    def locked(self) -> bool:
        return self.split == "test"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["prompt_pack"] = self.pack
        value["locked"] = self.locked
        return value


class PromptCatalog:
    """Load, validate, and deterministically split the bundled prompt corpus."""

    def __init__(
        self,
        prompts: Iterable[Prompt],
        *,
        seed: str,
        version: int = 1,
        include_bundled_packs: bool = False,
    ) -> None:
        raw = list(prompts)
        self.seed = seed
        self.version = version
        self._validate(raw)
        assigned = self._assign_splits(raw)
        self._validate_final_splits(raw, assigned)
        if include_bundled_packs:
            us_prompts = [
                replace(
                    prompt,
                    id=f"us-{prompt.id}",
                    text=US_PROMPT_TEXT.get(prompt.id, prompt.text),
                    pack=US_ENGLISH_PACK,
                    ordinal=len(assigned) + index,
                )
                for index, prompt in enumerate(assigned)
            ]
            assigned = [*assigned, *us_prompts]
            self._validate_pack_distributions(assigned)
        self.prompts = tuple(assigned)
        self._by_id = {prompt.id: prompt for prompt in self.prompts}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROMPTS_PATH) -> "PromptCatalog":
        resolved_path = Path(path).resolve()
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
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
                pack=item.get("pack", INDIAN_ENGLISH_PACK),
                split=item.get("split", ""),
                ordinal=index,
            )
            for index, item in enumerate(payload["prompts"])
        ]
        return cls(
            prompts,
            seed=payload["seed"],
            version=version,
            include_bundled_packs=resolved_path == DEFAULT_PROMPTS_PATH.resolve(),
        )

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
        if any(prompt.pack not in VALID_PROMPT_PACKS for prompt in prompts):
            raise ValueError("Prompt pack must be en-US or en-IN")
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

    @classmethod
    def _validate_pack_distributions(cls, prompts: list[Prompt]) -> None:
        for pack in VALID_PROMPT_PACKS:
            packed = [prompt for prompt in prompts if prompt.pack == pack]
            if not packed:
                raise ValueError(f"Prompt pack is empty: {pack}")
            kinds = Counter(prompt.kind for prompt in packed)
            styles = Counter(prompt.style for prompt in packed)
            if kinds["guided"] * 5 != len(packed) * 4:
                raise ValueError(f"Prompt pack {pack} must contain exactly 80% guided prompts")
            if styles["neutral"] * 10 != len(packed) * 9:
                raise ValueError(f"Prompt pack {pack} must contain exactly 90% neutral prompts")
            cls._validate_distribution(packed, f"Prompt pack {pack}")
            for kind in VALID_KINDS:
                cls._validate_distribution(
                    [prompt for prompt in packed if prompt.kind == kind],
                    f"Prompt pack {pack} {kind}",
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
        pack: str | None = None,
        exclude: Iterable[str] = (),
        chooser_seed: str = "default",
    ) -> Prompt | None:
        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if kind is not None and kind not in VALID_KINDS:
            raise ValueError(f"Unknown kind: {kind}")
        if style is not None and style not in VALID_STYLES:
            raise ValueError(f"Unknown style: {style}")
        if pack is not None and pack not in VALID_PROMPT_PACKS:
            raise ValueError(f"Unknown prompt pack: {pack}")
        excluded = set(exclude)
        eligible = [
            prompt
            for prompt in self.prompts
            if prompt.id not in excluded
            and (split is None or prompt.split == split)
            and (kind is None or prompt.kind == kind)
            and (style is None or prompt.style == style)
            and (pack is None or prompt.pack == pack)
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
            "packs": dict(Counter(prompt.pack for prompt in self.prompts)),
            "default_pack": DEFAULT_PROMPT_PACK,
        }
