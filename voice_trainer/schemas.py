"""Request schemas that can be imported without initializing the local app."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .model_runtime import MAX_GENERATION_SEED


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationRequest(StrictRequest):
    variant_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=2_000)
    delivery: Literal["natural", "warm", "clear", "measured"] = "natural"
    checkpoint_version: str = Field(min_length=1, max_length=180)
    seed: int | None = Field(
        default=None,
        ge=0,
        le=MAX_GENERATION_SEED,
        strict=True,
    )
