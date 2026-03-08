from typing import Optional

import torch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from gptwm import _make_green_list_mask_numpy, _get_english_token_ids

_WATERMARK_CONFIG = None


def set_watermark_config(
    fraction: float = 0.5,
    strength: float = 2.0,
    vocab_size: int = None,
    model_emb_length: int = None,
    only_English: bool = False,
    tokenizer: Optional[object] = None,
    default_watermark_key: Optional[int] = None,
):
    global _WATERMARK_CONFIG
    _WATERMARK_CONFIG = {
        "fraction": fraction,
        "strength": strength,
        "vocab_size": vocab_size,
        "model_emb_length": model_emb_length,
        "only_English": only_English,
        "tokenizer": tokenizer,
        "default_watermark_key": default_watermark_key,
    }
    if only_English:
        _WATERMARK_CONFIG["english_token_ids"] = _get_english_token_ids(
            tokenizer, vocab_size
        )


class GPTWatermarkAdapterLogitsProcessor(AdapterLogitsProcessor):
    """Per-request watermark logits processor for vLLM.

    Each request can specify its own watermark seed via
    ``SamplingParams(extra_args={"watermark_key": <int>})``.
    If no per-request seed is given, falls back to ``default_watermark_key``
    set in :func:`set_watermark_config`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        cfg = _WATERMARK_CONFIG
        assert cfg is not None, "call set_watermark_config() before building the LLM"

        self.fraction = cfg["fraction"]
        self.strength = cfg["strength"]
        self.vocab_size = cfg["vocab_size"]
        self.model_emb_length = cfg["model_emb_length"]
        self.only_English = cfg["only_English"]
        self.tokenizer = cfg["tokenizer"]
        self.english_token_ids = cfg.get("english_token_ids")
        self.default_watermark_key = cfg["default_watermark_key"]
        self._mask_cache: dict[int, torch.Tensor] = {}

    def _get_mask(self, watermark_key: int) -> torch.Tensor:
        if watermark_key not in self._mask_cache:
            self._mask_cache[watermark_key] = torch.tensor(
                _make_green_list_mask_numpy(
                    watermark_key,
                    self.fraction,
                    self.vocab_size,
                    self.model_emb_length,
                    self.only_English,
                    self.tokenizer,
                    self.english_token_ids,
                ),
                dtype=torch.float32,
            )
        return self._mask_cache[watermark_key]

    def new_req_logits_processor(self, params: SamplingParams):
        watermark_key = self.default_watermark_key
        if params.extra_args and "watermark_key" in params.extra_args:
            watermark_key = params.extra_args["watermark_key"]
        if watermark_key is None:
            return None

        mask = self._get_mask(watermark_key)
        strength = self.strength

        def _apply_watermark(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            return logits + strength * mask.to(logits.device)

        return _apply_watermark

    def is_argmax_invariant(self) -> bool:
        return False
