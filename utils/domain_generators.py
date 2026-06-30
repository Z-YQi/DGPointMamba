import math
from types import SimpleNamespace

import torch

from utils.sinpoint_augmentation import SinPoint


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class FixedSinPointGenerator:
    """Fixed SinPoint pseudo-domain generator for source-only DG training."""

    def __init__(self, config=None):
        self.type = "fixed_sinpoint"
        self.A = float(_cfg_get(config, "A", 0.8))
        self.w = float(_cfg_get(config, "w", 3.0))
        self.rand_center_num = int(_cfg_get(config, "rand_center_num", 4))
        self.sample = str(_cfg_get(config, "sample", "RPS"))
        self.isCat = bool(_cfg_get(config, "isCat", False))
        self.shuffle = bool(_cfg_get(config, "shuffle", False))

        if self.isCat:
            raise ValueError("Fixed SinPoint baseline requires isCat=False to keep P_aug.shape == P_s.shape.")

        self.augmenter = SinPoint(
            SimpleNamespace(
                A=self.A,
                w=self.w,
                rand_center_num=self.rand_center_num,
                sample=self.sample,
                isCat=self.isCat,
                shuffle=self.shuffle,
            )
        )

    def __call__(self, partial):
        if not torch.is_tensor(partial):
            raise TypeError(f"partial must be a torch.Tensor, got {type(partial)}")
        if partial.dim() != 3 or partial.size(-1) != 3:
            raise ValueError(f"partial must have shape [B, N, 3], got {tuple(partial.shape)}")

        input_shape = partial.shape
        input_device = partial.device

        partial_aug, _ = self.augmenter.Sin(partial)
        if partial_aug.device != input_device:
            partial_aug = partial_aug.to(input_device)
        if partial_aug.shape != input_shape:
            raise RuntimeError(f"Fixed SinPoint changed shape from {tuple(input_shape)} to {tuple(partial_aug.shape)}")

        delta = (partial_aug - partial).detach()
        delta_norm = torch.linalg.norm(delta, dim=-1)
        mean_delta_p = float(delta_norm.mean().item())
        max_delta_p = float(delta_norm.max().item())
        mean_abs_delta = float(delta.abs().mean().item())

        stats = {
            "domain_generator_type": self.type,
            "sinpoint_A": self.A,
            "sinpoint_w": self.w,
            "sinpoint_rand_center_num": self.rand_center_num,
            "sinpoint_sample": self.sample,
            "sinpoint_isCat": self.isCat,
            "sinpoint_shuffle": self.shuffle,
            "mean_delta_p": mean_delta_p,
            "max_delta_p": max_delta_p,
            "mean_abs_delta": mean_abs_delta,
            "stats_finite": math.isfinite(mean_delta_p) and math.isfinite(max_delta_p) and math.isfinite(mean_abs_delta),
        }
        return partial_aug, stats


def build_domain_generator(config):
    generator_config = _cfg_get(config, "domain_generator", None)
    generator_type = _cfg_get(generator_config, "type", "none")
    enabled = bool(_cfg_get(generator_config, "enable", generator_type not in (None, "none")))

    if not enabled or generator_type in (None, "none"):
        return None
    if generator_type == "fixed_sinpoint":
        return FixedSinPointGenerator(generator_config)

    raise NotImplementedError(f"Unsupported domain_generator.type: {generator_type}")
