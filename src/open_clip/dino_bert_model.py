# open_clip/dino_bert_model.py
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Vision: DINOv2 -----
def _load_dinov2_from_ckpt(ckpt_path: str):
    """
    Tries to load a DINOv2 model from a checkpoint using the official API.
    Requires 'dinov2' to be installed (facebookresearch/dinov2).
    """
    try:
        # Most convenient entrypoint for user-supplied checkpoints:
        from dinov2.models import build_model_from_checkpoint  # type: ignore
        model = build_model_from_checkpoint(ckpt_path)
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DINOv2 from checkpoint {ckpt_path}. "
            f"Make sure the 'dinov2' package is installed and the checkpoint is compatible.\n"
            f"Original error: {e}"
        )

def _extract_global_from_dino_out(out: torch.Tensor) -> torch.Tensor:
    """
    Normalize DINOv2 forward outputs to a single [B, D] embedding.
    Supports common return shapes/patterns.
    """
    # If it's already [B, D]
    if out.ndim == 2:
        return out
    # If it's tokens [B, N, D], take CLS token (assumed index 0)
    if out.ndim == 3:
        return out[:, 0]
    # Some DINOv2 forward() variants return dict-like outputs
    if isinstance(out, dict):
        for k in ("x_norm_clstoken", "x_norm_clstoken_mlp", "cls_token", "pooled"):
            if k in out and isinstance(out[k], torch.Tensor):
                t = out[k]
                return t if t.ndim == 2 else t[:, 0]
        # Fallback: take first tensor value in dict
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v if v.ndim == 2 else v[:, 0]
    raise ValueError("Unsupported DINOv2 output format; cannot extract a global embedding.")

class DinoVisionWrapper(nn.Module):
    """
    Wraps a DINOv2 backbone and projects to CLIP embedding space.
    """
    def __init__(
        self,
        checkpoint_path: str,
        embed_dim: int,
        freeze_backbone: bool = True,
        proj_bias: bool = False,
        proj_bn: bool = False,
    ):
        super().__init__()
        self.backbone = _load_dinov2_from_ckpt(checkpoint_path)
        # Infer backbone output dim
        d_backbone = getattr(self.backbone, "embed_dim", None)
        if d_backbone is None:
            # Try a dry run with a dummy tensor to infer dim
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.backbone(dummy)
                d_backbone = _extract_global_from_dino_out(out).shape[-1]
        # Projection head
        proj = [nn.Linear(d_backbone, embed_dim, bias=proj_bias)]
        if proj_bn:
            proj.append(nn.BatchNorm1d(embed_dim))
        self.proj = nn.Sequential(*proj)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _encode_no_grad(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        g = _extract_global_from_dino_out(out)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return unnormalized CLIP-space features
        if any(p.requires_grad for p in self.backbone.parameters()):
            feats = _extract_global_from_dino_out(self.backbone(x))
        else:
            feats = self._encode_no_grad(x)
        return self.proj(feats)

    def encode_image(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        z = self.forward(x)
        return F.normalize(z, dim=-1) if normalize else z


# ----- Text: BiomedBERT -----
class BiomedBERTWrapper(nn.Module):
    """
    Wraps a HF BERT (BiomedBERT) and projects to CLIP embedding space.
    """
    def __init__(
        self,
        model_name: str,
        embed_dim: int,
        freeze_backbone: bool = True,
        pool: str = "cls",  # 'cls' or 'mean'
        proj_bias: bool = False,
        proj_bn: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel  # lazy import
        self.text_model = AutoModel.from_pretrained(model_name)
        hidden = self.text_model.config.hidden_size
        proj = [nn.Linear(hidden, embed_dim, bias=proj_bias)]
        if proj_bn:
            proj.append(nn.BatchNorm1d(embed_dim))
        self.proj = nn.Sequential(*proj)
        self.pool = pool

        if freeze_backbone:
            for p in self.text_model.parameters():
                p.requires_grad = False

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # last_hidden_state: [B, T, H]
        if self.pool == "mean":
            if attention_mask is None:
                return last_hidden_state.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        # CLS
        return last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.proj(pooled)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        z = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(z, dim=-1) if normalize else z


# ----- CLIP-style container -----
class DinoBertCLIP(nn.Module):
    """
    Minimal CLIP-like container exposing encode_image/encode_text and forward() that returns logits.
    Drop-in where OpenCLIP expects a CLIP-ish module.
    """
    def __init__(self, vision: nn.Module, text: nn.Module, logit_scale_init: float = math.log(1/0.07)):
        super().__init__()
        self.visual = vision
        self.text = text
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))

    def encode_image(self, image: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.visual.encode_image(image, normalize=normalize) if hasattr(self.visual, "encode_image") else \
               F.normalize(self.visual(image), dim=-1) if normalize else self.visual(image)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        if hasattr(self.text, "encode_text"):
            return self.text.encode_text(input_ids=input_ids, attention_mask=attention_mask, normalize=normalize)
        z = self.text(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(z, dim=-1) if normalize else z

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits_per_image, logits_per_text)
        """
        img = self.encode_image(image, normalize=True)          # [B, D]
        txt = self.encode_text(input_ids, attention_mask, True) # [B, D]
        scale = self.logit_scale.exp()
        logits_per_image = scale * img @ txt.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# ----- Builder -----
def build_model_dino_bert(
    vision_checkpoint: str,
    text_model_name: str = "microsoft/BiomedBERT-base-uncased-abstract",
    embed_dim: int = 512,
    freeze_vision: bool = True,
    freeze_text: bool = True,
    text_pool: str = "cls",          # 'cls' or 'mean'
    proj_bias: bool = False,
    proj_bn: bool = False,
) -> DinoBertCLIP:
    """
    Create a CLIP-like model using DINOv2 (vision) + BiomedBERT (text).
    """
    if not os.path.isfile(vision_checkpoint):
        raise FileNotFoundError(f"DINOv2 checkpoint not found: {vision_checkpoint}")

    vision = DinoVisionWrapper(
        checkpoint_path=vision_checkpoint,
        embed_dim=embed_dim,
        freeze_backbone=freeze_vision,
        proj_bias=proj_bias,
        proj_bn=proj_bn,
    )
    text = BiomedBERTWrapper(
        model_name=text_model_name,
        embed_dim=embed_dim,
        freeze_backbone=freeze_text,
        pool=text_pool,
        proj_bias=proj_bias,
        proj_bn=proj_bn,
    )
    return DinoBertCLIP(vision=vision, text=text)

