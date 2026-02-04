from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.config import VllmConfig
import torch

_WATERMARK_BASE = None

def set_watermark_base(wm_base):
    global _WATERMARK_BASE
    _WATERMARK_BASE = wm_base


class vLLMGPTWatermarkLogitsWarper(LogitsProcessor):
    """
    vLLM logits processor
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ):
        # super().__init__(vllm_config, device, is_pin_memory)
        self.watermark = _WATERMARK_BASE
        assert self.watermark.green_list_mask.numel() == vllm_config.model_config.get_vocab_size(), "green_list_mask length mismatch with vocab size"

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits shape: [num_active_tokens, vocab_size]
        """
        logits = logits + (
            self.watermark.strength
            * self.watermark.green_list_mask
            .to(logits.device)
        )
        return logits

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update):
        pass
