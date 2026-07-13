from __future__ import annotations

import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear

from voice_trainer.training import TrainingConfig, _apply_lora, _target_module_path


class _ProjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = {
            "q_proj": nn.Linear(4, 4, bias=False),
            "k_proj": nn.Linear(4, 4, bias=False),
            "v_proj": nn.Linear(4, 4, bias=False),
            "o_proj": nn.Linear(4, 4, bias=False),
        }
        self.mlp = {
            "gate_proj": nn.Linear(4, 8, bias=False),
            "up_proj": nn.Linear(4, 8, bias=False),
            "down_proj": nn.Linear(8, 4, bias=False),
        }


class _TwoStackTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.talker = {
            "model": {"layers": [_ProjectionBlock()]},
            "code_predictor": {"model": {"layers": [_ProjectionBlock()]}},
        }


def test_lora_replaces_main_and_code_predictor_projections() -> None:
    model = _TwoStackTalker()
    targets = _apply_lora(model, TrainingConfig(stage="pilot", epochs=1))

    assert len(targets) == 14
    assert sum("code_predictor" in path for path in targets) == 7
    assert sum("code_predictor" not in path for path in targets) == 7
    assert all(
        isinstance(module, LoRALinear)
        for path, module in model.named_modules()
        if _target_module_path(path, TrainingConfig().lora_targets)
    )


def test_target_filter_excludes_non_transformer_linear_layers() -> None:
    targets = TrainingConfig().lora_targets
    assert _target_module_path("talker.model.layers.0.self_attn.q_proj", targets)
    assert _target_module_path(
        "talker.code_predictor.model.layers.4.mlp.down_proj", targets
    )
    assert not _target_module_path("speaker_encoder.projection", targets)
    assert not _target_module_path("talker.codec_head", targets)
