"""Early-exit models shared by training (train_exits.py), trace logging and real-GPU inference (gpu_runner.py).

Both backbones are driven layer by layer through one interface:
    h = m.embed(batch)            # [n, T, H]
    h = m.layer(i, h, mask)       # one transformer layer on the n active rows
    logits = m.head(e, h)         # exit head e on the CLS token
`mask` is the additive attention mask for text and None for vision.
"""
from __future__ import annotations

import torch
from torch import nn


class ExitHead(nn.Module):
    """RobertaClassificationHead on the CLS state: dropout -> dense -> tanh -> dropout -> out_proj."""

    def __init__(self, hidden: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, num_labels)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.drop(torch.tanh(self.dense(self.drop(cls)))))


class VitExitHead(nn.Module):
    """ViT classifier head on the CLS token: LayerNorm -> Linear (same as ViTForImageClassification)."""

    def __init__(self, hidden: int, num_labels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, eps=1e-12)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden, num_labels)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.drop(self.norm(cls)))


class EarlyExitModel(nn.Module):
    """Backbone (RobertaModel or ViTModel) + one exit head per exit layer."""

    def __init__(self, backbone: nn.Module, kind: str, exit_layers: list[int], num_labels: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone, self.kind, self.exit_layers = backbone, kind, list(exit_layers)
        H = backbone.config.hidden_size
        Head = ExitHead if kind == "text" else VitExitHead
        self.heads = nn.ModuleList(Head(H, num_labels, dropout) for _ in exit_layers)

    # ---- layer-by-layer interface (inference) ----
    def embed(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kind == "text":
            h = self.backbone.embeddings(input_ids=batch["input_ids"])
            m = batch["attention_mask"][:, None, None, :].to(h.dtype)
            return h, (1.0 - m) * torch.finfo(h.dtype).min
        return self.backbone.embeddings(batch["pixel_values"].to(self.dtype)), None

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def blocks(self) -> nn.ModuleList:
        bb = self.backbone          # RobertaModel: encoder.layer; ViTModel (transformers v5): layers
        return bb.encoder.layer if hasattr(bb, "encoder") else bb.layers

    def layer(self, i: int, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.blocks[i](h, attention_mask=mask)

    def head(self, e: int, h: torch.Tensor) -> torch.Tensor:
        return self.heads[e](h[:, 0])

    # ---- full forward (training / trace logging): logits at every exit, [B, E, C] ----
    def forward(self, batch: dict) -> torch.Tensor:
        h, mask = self.embed(batch)
        out, layer = [], 0
        for e, stop in enumerate(self.exit_layers):
            while layer < stop:
                h = self.layer(layer, h, mask)
                layer += 1
            out.append(self.head(e, h))
        return torch.stack(out, 1)


def build(cfg: dict, load_backbone_weights: bool = True) -> EarlyExitModel:
    """Build the early-exit model described by a config (see configs/train_*.yaml)."""
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
    if cfg["kind"] == "text":
        if load_backbone_weights:
            clf = AutoModelForSequenceClassification.from_pretrained(cfg["backbone"], num_labels=cfg["num_labels"],
                                                                     ignore_mismatched_sizes=True)
            bb = clf.roberta
        else:
            clf, bb = None, AutoModel.from_config(AutoConfig.from_pretrained(cfg["backbone"]), add_pooling_layer=False)
        m = EarlyExitModel(bb, "text", cfg["exit_layers"], cfg["num_labels"], cfg.get("dropout", 0.1))
        # A fine-tuned classifier has the same shape as ExitHead: use it to initialise every exit head.
        if clf is not None and cfg.get("init_heads_from_classifier", False):
            sd = {k: v.clone() for k, v in clf.classifier.state_dict().items()}
            perm = cfg.get("classifier_label_perm")  # checkpoint class k == dataset label perm[k]
            if perm is not None:
                inv = torch.argsort(torch.tensor(perm))   # dataset label j <- checkpoint class inv[j]
                sd["out_proj.weight"], sd["out_proj.bias"] = sd["out_proj.weight"][inv], sd["out_proj.bias"][inv]
            for h in m.heads:
                h.load_state_dict(sd)
        return m
    if load_backbone_weights:
        bb = AutoModel.from_pretrained(cfg["backbone"], add_pooling_layer=False)
    else:
        bb = AutoModel.from_config(AutoConfig.from_pretrained(cfg["backbone"]), add_pooling_layer=False)
    return EarlyExitModel(bb, "vision", cfg["exit_layers"], cfg["num_labels"], cfg.get("dropout", 0.0))
