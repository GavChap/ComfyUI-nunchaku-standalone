import logging
from typing import Callable, List, Tuple, Union
from pathlib import Path

import torch
from comfy.ldm.common_dit import pad_to_patch_size
from einops import rearrange, repeat
from torch import nn

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.caching.fbcache import cache_context, create_cache_context
from ..nunchaku_code.lora_qwen import compose_loras_v2, reset_lora_v2


class ComfyQwenImageWrapper(nn.Module):
    """
    Wrapper for NunchakuQwenImageTransformer2DModel to support ComfyUI workflows.

    This wrapper separates LoRA composition from the forward pass for maximum efficiency.
    It detects changes to its `loras` attribute and recomposes the underlying model
    lazily when the forward pass is executed.
    """

    def __init__(
        self,
        model: NunchakuQwenImageTransformer2DModel,
        config,
        customized_forward: Callable = None,
        forward_kwargs: dict | None = None,
    ):
        super().__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        # This list is the authoritative state, modified by LoRA loader nodes
        self.loras: List[Tuple[Union[str, Path, dict], float]] = []
        # This tracks the LoRAs currently composed into the model to detect changes
        self._applied_loras: List[Tuple[Union[str, Path, dict], float]] = None

        self.customized_forward = customized_forward
        self.forward_kwargs = forward_kwargs or {}

        self._prev_timestep = None
        self._cache_context = None

    def to_safely(self, device):
        """Safely move the model to the specified device."""
        if hasattr(self.model, "to_safely"):
            self.model.to_safely(device)
        else:
            self.model.to(device)
        return self

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        """Preprocess an input image tensor for the model."""
        bs, c, h, w = x.shape
        patch_size = self.config.get("patch_size", 2)
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size

        h_offset = (h_offset + (patch_size // 2)) // patch_size
        w_offset = (w_offset + (patch_size // 2)) // patch_size

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(
        self,
        x,
        timestep,
        context=None,
        y=None,
        guidance=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        """
        Forward pass for the wrapped model.

        Detects changes to the `self.loras` list and recomposes the model
        on-the-fly before inference.
        """
        logging.info(f"Timestep {timestep}")

        if isinstance(timestep, torch.Tensor):
            timestep_float = timestep.item() if timestep.numel() == 1 else timestep.flatten()[0].item()
        else:
            timestep_float = float(timestep)

        logging.info(f"Are loras changed? {self._applied_loras == self.loras}")
        # Check if the LoRA stack has been changed by a loader node
        if self._applied_loras != self.loras or timestep_float == 1:
            # The compose function handles resetting before applying the new stack
            reset_lora_v2(self.model)
            self._applied_loras = self.loras.copy()
            compose_loras_v2(self.model, self.loras)

        # Caching logic
        use_caching = getattr(self.model, "residual_diff_threshold_multi", 0) != 0 or getattr(self.model, "_is_cached", False)
        if use_caching:
            cache_invalid = self._prev_timestep is None or self._prev_timestep < timestep_float + 1e-5
            if cache_invalid:
                self._cache_context = create_cache_context()
            self._prev_timestep = timestep_float

            with cache_context(self._cache_context):
                out = self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)
        else:
            out = self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)

        if isinstance(out, tuple):
            out = out[0]

        if x.ndim == 5 and out.ndim == 4:
            out = out.unsqueeze(2)

        return out

    def _execute_model(self, x, timestep, context, guidance, control, transformer_options, **kwargs):
        """Helper function to run the model's forward pass."""
        model_device = next(self.model.parameters()).device

        # Move input tensors to the model's device
        x = x.to(model_device)
        if context is not None:
            context = context.to(model_device)

        # Keep original input shape check
        input_is_5d = x.ndim == 5
        if input_is_5d:
            x = x.squeeze(2)

        if self.customized_forward:
            return self.customized_forward(
                self.model,
                hidden_states=x,
                encoder_hidden_states=context,
                timestep=timestep,
                guidance=guidance if self.config.get("guidance_embed", False) else None,
                control=control,
                transformer_options=transformer_options,
                **self.forward_kwargs,
                **kwargs,
            )
        else:
            return self.model(
                hidden_states=x,
                encoder_hidden_states=context,
                timestep=timestep,
                guidance=guidance if self.config.get("guidance_embed", False) else None,
                control=control,
                transformer_options=transformer_options,
                **kwargs,
            )