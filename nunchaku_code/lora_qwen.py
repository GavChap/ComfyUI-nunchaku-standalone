import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict

import torch
import torch.nn as nn
from safetensors import safe_open

from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight, reorder_adanorm_lora_up, unpack_lowrank_weight

logger = logging.getLogger(__name__)

# --- Regular Expression Definitions (unchanged) ---
_RE_QKV_DBL_DECOMP = re.compile(r"^(transformer_blocks\.\d+)\.attn\.to_(q|k|v)(?=\.|$)")
_RE_QKV_DBL_FUSED = re.compile(r"^(transformer_blocks\.\d+)\.attn\.to_qkv(?=\.|$)")
_RE_ADDQKV_DBL_DECOMP = re.compile(r"^(transformer_blocks\.\d+)\.attn\.add_(q|k|v)_proj(?=\.|$)")
_RE_ADDQKV_DBL_FUSED = re.compile(r"^(transformer_blocks\.\d+)\.attn\.add_qkv_proj(?=\.|$)")
_RE_QKV_SGL_DECOMP = re.compile(r"^(single_transformer_blocks\.\d+)\.attn\.to_(q|k|v)(?=\.|$)")
_RE_QKV_SGL_FUSED = re.compile(r"^(single_transformer_blocks\.\d+)\.attn\.to_qkv(?=\.|$)")
_RE_OUTPROJ_DBL = re.compile(r"^(transformer_blocks\.\d+)\.out_proj(?=\.|$)")
_RE_OUTPROJ_CTX_DBL = re.compile(r"^(transformer_blocks\.\d+)\.out_proj_context(?=\.|$)")
_RE_TOOUT_DBL = re.compile(r"^(transformer_blocks\.\d+)\.attn\.to_out(?=\.|$)")
_RE_TOADDOUT_DBL = re.compile(r"^(transformer_blocks\.\d+)\.attn\.to_add_out(?=\.|$)")
_RE_PROJOUT_SGL = re.compile(r"^(single_transformer_blocks\.\d+)\.proj_out(?=\.|$)")
_RE_PROJMLP_SGL = re.compile(r"^(single_transformer_blocks\.\d+)\.proj_mlp(?=\.|$)")
_RE_TOOUT_SGL = re.compile(r"^(single_transformer_blocks\.\d+)\.attn\.to_out(?=\.|$)")
_RE_NORM_SGL = re.compile(r"^(single_transformer_blocks\.\d+)\.norm\.linear(?=\.|$)")
_RE_NORM1_DBL = re.compile(r"^(transformer_blocks\.\d+)\.norm1\.linear(?=\.|$)")
_RE_NORM1CTX_DBL = re.compile(r"^(transformer_blocks\.\d+)\.norm1_context\.linear(?=\.|$)")
_RE_FF_DBL_FC1 = re.compile(r"^(transformer_blocks\.\d+)\.ff\.net\.0(?:\.proj)?(?=\.|$)")
_RE_FF_DBL_FC2 = re.compile(r"^(transformer_blocks\.\d+)\.ff\.net\.2(?=\.|$)")
_RE_FFCTX_DBL_FC1 = re.compile(r"^(transformer_blocks\.\d+)\.ff_context\.net\.0(?:\.proj)?(?=\.|$)")
_RE_FFCTX_DBL_FC2 = re.compile(r"^(transformer_blocks\.\d+)\.ff_context\.net\.2(?=\.|$)")
_RE_MLP_IMG_FC1 = re.compile(r"^(transformer_blocks\.\d+\.img_mlp\.net\.0(?:\.proj)?)(?=\.|$)")
_RE_MLP_IMG_FC2 = re.compile(r"^(transformer_blocks\.\d+\.img_mlp\.net\.2)(?=\.|$)")
_RE_MLP_TXT_FC1 = re.compile(r"^(transformer_blocks\.\d+\.txt_mlp\.net\.0(?:\.proj)?)(?=\.|$)")
_RE_MLP_TXT_FC2 = re.compile(r"^(transformer_blocks\.\d+\.txt_mlp\.net\.2)(?=\.|$)")
_RE_LORA_SUFFIX = re.compile(r"\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$")
_RE_ALPHA_SUFFIX = re.compile(r"\.(?:alpha|lora_alpha)(?:\.[^.]+)*$")
_RE_QKV_DBL_DECOMP_ALT = re.compile(r"^(transformer_blocks\.\d+)\.attn\.(q|k|v)_proj(?=\.|$)")
_RE_IMGMOD_LINEAR = re.compile(r"^(transformer_blocks\.\d+)\.img_mod\.1(?=\.|$)")
_RE_TXTMOD_LINEAR = re.compile(r"^(transformer_blocks\.\d+)\.txt_mod\.1(?=\.|$)")


# --- Helper Functions (mostly unchanged) ---
def _classify_and_map_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """
    key -> (group, base_key, comp, ab)
    """
    k = key
    if k.startswith("transformer."):
        k = k[len("transformer."):]

    if k.startswith("diffusion_model."):
        k = k[len("diffusion_model."):]

    base = None
    ab = None

    m = _RE_LORA_SUFFIX.search(k)
    if m:
        tag = m.group("tag")  # lora_A / lora_B / lora.down / lora.up
        base = k[: m.start()]
        if "lora_A" in tag or tag.endswith(".A"):
            ab = "A"
        elif "lora_B" in tag or tag.endswith(".B"):
            ab = "B"
        elif "down" in tag:
            ab = "A"
        elif "up" in tag:
            ab = "B"
        else:
            return None
    else:
        m = _RE_ALPHA_SUFFIX.search(k)
        if m:
            ab = "alpha"
            base = k[: m.start()]
        else:
            # Fallback for unconventional naming
            if ".lora_A" in k:
                ab = "A"
                base = k.replace(".lora_A.weight", "").replace(".lora_A", "")
            elif ".lora_B" in k:
                ab = "B"
                base = k.replace(".lora_B.weight", "").replace(".lora_B", "")
            elif ".lora_down" in k:
                ab = "A"
                base = k.replace(".lora_down.weight", "").replace(".lora_down", "")
            elif ".lora_up" in k:
                ab = "B"
                base = k.replace(".lora_up.weight", "").replace(".lora_up", "")
            elif ".alpha" in k:
                ab = "alpha"
                base = k.replace(".alpha", "")
            else:
                return None

    # QKV (double)
    m = _RE_QKV_DBL_FUSED.match(base)
    if m:
        return ("qkv", f"{m.group(1)}.attn.to_qkv", None, ab)

    m = _RE_QKV_DBL_DECOMP.match(base)
    if m:
        return ("qkv", f"{m.group(1)}.attn.to_qkv", m.group(2).upper(), ab)

    # ADD_QKV (double)
    m = _RE_ADDQKV_DBL_FUSED.match(base)
    if m:
        return ("add_qkv", f"{m.group(1)}.attn.add_qkv_proj", None, ab)

    m = _RE_ADDQKV_DBL_DECOMP.match(base)
    if m:
        return ("add_qkv", f"{m.group(1)}.attn.add_qkv_proj", m.group(2).upper(), ab)

    m = _RE_QKV_DBL_DECOMP_ALT.match(base)
    if m:
        return ("qkv", f"{m.group(1)}.attn.to_qkv", m.group(2).upper(), ab)

    # QKV (single)
    m = _RE_QKV_SGL_FUSED.match(base)
    if m:
        return ("qkv", f"{m.group(1)}.attn.to_qkv", None, ab)

    m = _RE_QKV_SGL_DECOMP.match(base)
    if m:
        return ("qkv", f"{m.group(1)}.attn.to_qkv", m.group(2).upper(), ab)

    # out/ff (double)
    m = _RE_OUTPROJ_CTX_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_add_out", None, ab)

    m = _RE_TOADDOUT_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_add_out", None, ab)

    m = _RE_OUTPROJ_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_out.0", None, ab)

    m = _RE_TOOUT_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_out.0", None, ab)

    m = _RE_FF_DBL_FC1.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_fc1", None, ab)

    m = _RE_FF_DBL_FC2.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_fc2", None, ab)

    m = _RE_FFCTX_DBL_FC1.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_context_fc1", None, ab)

    m = _RE_FFCTX_DBL_FC2.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_context_fc2", None, ab)

    # single
    m = _RE_PROJOUT_SGL.match(base)
    if m:
        return ("single_proj_out", f"{m.group(1)}.proj_out", None, ab)

    m = _RE_PROJMLP_SGL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.mlp_fc1", None, ab)

    m = _RE_TOOUT_SGL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.attn.to_out", None, ab)

    # norm.linear
    m = _RE_NORM_SGL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.norm.linear", None, ab)

    m = _RE_NORM1_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.norm1.linear", None, ab)

    m = _RE_NORM1CTX_DBL.match(base)
    if m:
        return ("regular", f"{m.group(1)}.norm1_context.linear", None, ab)

    m = _RE_MLP_IMG_FC1.match(base) or _RE_MLP_TXT_FC1.match(base)
    if m:
        return ("regular", m.group(1), None, ab)

    m = _RE_MLP_IMG_FC2.match(base) or _RE_MLP_TXT_FC2.match(base)
    if m:
        return ("regular", m.group(1), None, ab)

    m = _RE_IMGMOD_LINEAR.match(base)
    if m:
        return ("regular", f"{m.group(1)}.img_mod.1", None, ab)

    m = _RE_TXTMOD_LINEAR.match(base)
    if m:
        return ("regular", f"{m.group(1)}.txt_mod.1", None, ab)

    return None


def _resolve_module_name(model: nn.Module, name: str) -> Tuple[str, Optional[nn.Module]]:
    """Resolve a name string path to a module, attempting fallback paths."""
    # First try as-is
    m = _get_module_by_name(model, name)
    if m is not None:
        return name, m

    # Correction paths
    if name.endswith(".attn.to_out.0"):
        alt = name[:-2]
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m
    elif name.endswith(".attn.to_out"):
        alt = name + ".0"
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m

    mapping = {
        ".ff.net.0.proj": ".mlp_fc1", ".ff.net.2": ".mlp_fc2",
        ".ff_context.net.0.proj": ".mlp_context_fc1", ".ff_context.net.2": ".mlp_context_fc2",
    }
    for src, dst in mapping.items():
        if src in name:
            alt = name.replace(src, dst)
            m = _get_module_by_name(model, alt)
            if m is not None: return alt, m

    logger.debug(f"[MISS] Module not found: {name}")
    return name, None


def _is_indexable_module(m):
    return isinstance(m, (nn.ModuleList, nn.Sequential, list, tuple))


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Traverse a path like 'a.b.3.c' to find and return a module."""
    if not name: return model
    module = model
    for part in name.split("."):
        if not part: continue
        if part.isdigit():
            if _is_indexable_module(module):
                try:
                    module = module[int(part)]
                except IndexError:
                    return None
            else:
                return None
        elif hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module


def _load_lora_state_dict(lora_state_dict_or_path: Union[str, Path, Dict[str, torch.Tensor]]) -> Dict[
    str, torch.Tensor]:
    """Load LoRA state dict from path or return existing dict."""
    if isinstance(lora_state_dict_or_path, (str, Path)):
        path = Path(lora_state_dict_or_path)
        if path.suffix == ".safetensors":
            state_dict = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            return state_dict
        else:
            return torch.load(path, map_location="cpu")
    return lora_state_dict_or_path


def _fuse_qkv_lora(qkv_weights: Dict[str, torch.Tensor]) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse Q/K/V LoRA weights into a single QKV tensor."""
    required_keys = ["Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B"]
    if not all(k in qkv_weights for k in required_keys):
        return None, None, None

    A_q, A_k, A_v = qkv_weights["Q_A"], qkv_weights["K_A"], qkv_weights["V_A"]
    B_q, B_k, B_v = qkv_weights["Q_B"], qkv_weights["K_B"], qkv_weights["V_B"]

    if not (A_q.shape == A_k.shape == A_v.shape):
        logger.warning(f"Q/K/V LoRA A dimensions mismatch: {A_q.shape}, {A_k.shape}, {A_v.shape}")
        return None, None, None

    alpha_q, alpha_k, alpha_v = qkv_weights.get("Q_alpha"), qkv_weights.get("K_alpha"), qkv_weights.get("V_alpha")
    alpha_fused = None
    if all(a is not None and a.item() == alpha_q.item() for a in [alpha_k, alpha_v]):
        alpha_fused = alpha_q

    A_fused = torch.cat([A_q, A_k, A_v], dim=0)

    r = B_q.shape[1]
    out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
    B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
    B_fused[:out_q, :r] = B_q
    B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
    B_fused[out_q + out_k:, 2 * r:] = B_v

    return A_fused, B_fused, alpha_fused


def _handle_proj_out_split(
        lora_dict: Dict[str, Dict[str, torch.Tensor]], base_key: str, model: nn.Module
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]], List[str]]:
    """Split single-block proj_out LoRA into two branches."""
    result, consumed = {}, []
    m = re.search(r"single_transformer_blocks\.(\d+)", base_key)
    if not m or base_key not in lora_dict:
        return result, consumed

    block_idx = m.group(1)
    block = _get_module_by_name(model, f"single_transformer_blocks.{block_idx}")
    if block is None: return result, consumed

    A_full, B_full, alpha = lora_dict[base_key].get("A"), lora_dict[base_key].get("B"), lora_dict[base_key].get("alpha")
    if A_full is None or B_full is None: return result, consumed

    attn_to_out = getattr(getattr(block, "attn", None), "to_out", None)
    mlp_fc2 = getattr(block, "mlp_fc2", None)
    if attn_to_out is None or mlp_fc2 is None: return result, consumed

    attn_in, mlp_in = attn_to_out.in_features, mlp_fc2.in_features
    if A_full.shape[1] != attn_in + mlp_in:
        logger.warning(f"{base_key}: A_full shape mismatch {A_full.shape} vs expected in_features {attn_in + mlp_in}")
        return result, consumed

    A_attn, A_mlp = A_full[:, :attn_in], A_full[:, attn_in:]
    result[f"single_transformer_blocks.{block_idx}.attn.to_out"] = (A_attn, B_full, alpha)
    result[f"single_transformer_blocks.{block_idx}.mlp_fc2"] = (A_mlp, B_full, alpha)
    consumed.append(base_key)
    return result, consumed


# --- NEW: Core Composition Function ---

def compose_loras_v2(
        model: torch.nn.Module,
        lora_configs: List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]],
) -> None:
    """
    Resets and composes multiple LoRAs into the model with individual strengths.

    This function first aggregates weights from all provided LoRAs, then applies them
    to the corresponding model modules in a single pass.

    Parameters
    ----------
    model : torch.nn.Module
        The model to update (e.g., FLUX).
    lora_configs : List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]]
        A list of tuples, where each tuple contains:
        - A LoRA file path or a pre-loaded state dictionary.
        - The strength (float) to apply to that LoRA.
    """
    logger.info(f"Composing {len(lora_configs)} LoRAs...")
    reset_lora_v2(model)

    aggregated_weights: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unused_keys: List[str] = []

    # 1. Aggregate weights from all LoRAs
    for lora_path_or_dict, strength in lora_configs:
        lora_name = lora_path_or_dict if isinstance(lora_path_or_dict, str) else "dict"
        lora_state_dict = _load_lora_state_dict(lora_path_or_dict)

        lora_grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        for key, value in lora_state_dict.items():
            parsed = _classify_and_map_key(key)
            if parsed is None:
                unused_keys.append(key)
                continue

            group, base_key, comp, ab = parsed
            if group in ("qkv", "add_qkv") and comp is not None:
                lora_grouped[base_key][f"{comp}_{ab}"] = value
            else:
                lora_grouped[base_key][ab] = value

        # Process grouped weights for this LoRA
        processed_groups = {}
        special_handled = []
        for base_key, lw in lora_grouped.items():
            if base_key in special_handled: continue

            if ".to_qkv" in base_key or ".add_qkv_proj" in base_key:
                A, B, alpha = (lw["A"], lw["B"], lw.get("alpha")) if "A" in lw else _fuse_qkv_lora(lw)
                if A is not None and B is not None:
                    processed_groups[base_key] = (A, B, alpha)
            elif ".proj_out" in base_key and "single_transformer_blocks" in base_key:
                split_map, consumed = _handle_proj_out_split(lora_grouped, base_key, model)
                processed_groups.update(split_map)
                special_handled.extend(consumed)
            else:
                if "A" in lw and "B" in lw:
                    processed_groups[base_key] = (lw["A"], lw["B"], lw.get("alpha"))

        for module_key, (A, B, alpha) in processed_groups.items():
            aggregated_weights[module_key].append({
                "A": A, "B": B, "alpha": alpha, "strength": strength, "source": lora_name
            })

    # 2. Apply aggregated weights to the model
    applied_modules_count = 0
    for module_name, parts in aggregated_weights.items():
        resolved_name, module = _resolve_module_name(model, module_name)
        if module is None or not (hasattr(module, "proj_down") and hasattr(module, "proj_up")):
            logger.warning(f"Module '{module_name}' not found or not a valid LoRA target. Skipping.")
            continue

        all_A = []
        all_B_scaled = []

        for part in parts:
            A, B, alpha, strength = part["A"], part["B"], part["alpha"], part["strength"]
            r_lora = A.shape[0]
            scale_alpha = alpha.item() if alpha is not None else float(r_lora)
            scale = strength * (scale_alpha / max(1.0, float(r_lora)))

            # Adanorm reordering for B matrix
            if ".norm1.linear" in resolved_name or ".norm1_context.linear" in resolved_name:
                B = reorder_adanorm_lora_up(B, splits=6)
            elif ".single_transformer_blocks." in resolved_name and ".norm.linear" in resolved_name:
                B = reorder_adanorm_lora_up(B, splits=3)

            all_A.append(A.to(dtype=module.proj_down.dtype, device=module.proj_down.device))
            all_B_scaled.append((B * scale).to(dtype=module.proj_up.dtype, device=module.proj_up.device))

        if not all_A: continue

        final_A = torch.cat(all_A, dim=0)
        final_B = torch.cat(all_B_scaled, dim=1)

        _apply_lora_to_module(module, final_A, final_B, resolved_name, model)
        applied_modules_count += 1

    logger.info(f"Applied LoRA compositions to {applied_modules_count} modules.")
    if unused_keys:
        logger.warning(f"Unused keys ({len(unused_keys)}): {unused_keys[:5]}...")


# --- Main Functions ---

def update_lora_params_v2(
        model: torch.nn.Module,
        lora_state_dict_or_path: Union[str, Path, Dict[str, torch.Tensor]],
        strength: float = 1.0,
) -> None:
    """
    Loads and applies a single LoRA to the model.
    This is a convenience wrapper around `compose_loras_v2`.
    """
    logger.info(f"Loading single LoRA with strength {strength}.")
    compose_loras_v2(model, [(lora_state_dict_or_path, strength)])


def _apply_lora_to_module(
        module: nn.Module,
        A: torch.Tensor,  # [total_r_lora, in_features]
        B: torch.Tensor,  # [out_features, total_r_lora]
        module_name: str,
        model: nn.Module,
) -> None:
    """Helper to append combined LoRA weights to a module."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{module_name}: A/B must be 2D, got {A.shape}, {B.shape}")
    if A.shape[1] != module.in_features:
        raise ValueError(f"{module_name}: A shape {A.shape} mismatch with in_features={module.in_features}")
    if B.shape[0] != module.out_features:
        raise ValueError(f"{module_name}: B shape {B.shape} mismatch with out_features={module.out_features}")

    pd, pu = module.proj_down.data, module.proj_up.data
    pd = unpack_lowrank_weight(pd, down=True)
    pu = unpack_lowrank_weight(pu, down=False)

    base_rank = pd.shape[0] if pd.shape[1] == module.in_features else pd.shape[1]

    if pd.shape[1] == module.in_features:  # [rank, in]
        new_proj_down = torch.cat([pd, A], dim=0)
        axis_down = 0
    else:  # [in, rank]
        new_proj_down = torch.cat([pd, A.T], dim=1)
        axis_down = 1

    new_proj_up = torch.cat([pu, B], dim=1)

    module.proj_down.data = pack_lowrank_weight(new_proj_down, down=True)
    module.proj_up.data = pack_lowrank_weight(new_proj_up, down=False)
    module.rank = base_rank + A.shape[0]

    # Track applied lora
    if not hasattr(model, "_lora_slots"):
        model._lora_slots = {}

    slot = model._lora_slots.setdefault(module_name, {"base_rank": base_rank, "appended": 0, "axis_down": axis_down})
    slot["appended"] += A.shape[0]
    model._lora_slots[module_name] = slot


def set_lora_strength_v2(model: nn.Module, strength: float) -> None:
    """
    Adjusts the overall strength of all applied LoRAs.
    This acts as a global multiplier on top of the individual strengths
    set during composition.
    """
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        logger.warning("No LoRA weights loaded, cannot set strength.")
        return

    old_strength = getattr(model, "_lora_strength", 1.0)
    # Avoid division by zero if old strength was 0
    s = strength / old_strength if old_strength != 0 else 0

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None: continue

        base, appended = info["base_rank"], info["appended"]
        if appended <= 0: continue

        with torch.no_grad():
            # The appended weights are always the last columns of proj_up
            module.proj_up.data[:, base: base + appended] *= s

    model._lora_strength = strength
    logger.info(f"LoRA global strength updated to {strength}.")


def reset_lora_v2(model: nn.Module) -> None:
    """Removes all appended LoRA weights from the model."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        return

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None: continue

        base_rank = info["base_rank"]
        pd = unpack_lowrank_weight(module.proj_down.data, down=True)
        pu = unpack_lowrank_weight(module.proj_up.data, down=False)

        with torch.no_grad():
            if info.get("axis_down", 0) == 0:  # [rank, in]
                pd_reset = pd[:base_rank, :].clone()
            else:  # [in, rank]
                pd_reset = pd[:, :base_rank].clone()

            pu_reset = pu[:, :base_rank].clone()

            module.proj_down.data = pack_lowrank_weight(pd_reset, down=True)
            module.proj_up.data = pack_lowrank_weight(pu_reset, down=False)
            module.rank = base_rank

    model._lora_slots.clear()
    model._lora_strength = 1.0
    logger.info("All LoRA weights have been reset from the model.")