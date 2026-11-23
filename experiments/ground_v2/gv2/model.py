"""Ground v2 model.

One Qwen3-VL forward. A forward hook on text decoder layer L (1-based: the output
of layer L == ``hidden_states[L]`` in HF's output_hidden_states convention):

  1. reads V_L (image tokens), E_L (user text before the first frame) and q0
     (the token right before the first <TGT>, which has seen the sentence and
     all frames);
  2. single query decoder -> heatmap logits over all image tokens;
  3. p = sigmoid(logits).detach(); two summaries of V_L (p-weighted mean, top-k
     mean) go through zero-initialised MLPs and are ADDED to the two <TGT>
     hidden states; with prob ``summary_drop`` a learned null vector is used
     instead (training only);
  4. layers L+1..N continue unchanged and the LM head produces {"category": ...}.

Losses: weighted BCE + Dice on the heatmap (labels/weights from the collator),
plus LM cross-entropy on the answer. LoRA on the language model is applied by
the caller (see ``attach_lora``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

HEADS_FILE = "gv2_heads.pt"
CONFIG_FILE = "gv2_config.json"
LORA_DIR = "lora"


def base_model_of(qwen: nn.Module) -> nn.Module:
    return qwen.get_base_model() if hasattr(qwen, "get_base_model") else qwen


def text_layers(qwen: nn.Module) -> nn.ModuleList:
    return base_model_of(qwen).model.language_model.layers


class QueryLayer(nn.Module):
    """One query: read the sentence (E_L), then look in the frames (V_L)."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.text_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.visual_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, q: torch.Tensor, text: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        q = q + self.text_attn(self.n1(q), text, text, need_weights=False)[0]
        q = q + self.visual_attn(self.n2(q), visual, visual, need_weights=False)[0]
        return q + self.ffn(self.n3(q))


class SummaryMLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 4), nn.SiLU(), nn.Linear(hidden // 4, hidden))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroundV2(nn.Module):
    def __init__(
        self,
        qwen: nn.Module,
        *,
        tgt_token_id: int,
        im_start_id: Optional[int],
        feature_layer: int = 12,
        query_dim: int = 1024,
        decoder_layers: int = 2,
        num_heads: int = 8,
        mask_dim: int = 256,
        logit_scale: float = 10.0,
        topk: int = 16,
        summary_drop: float = 0.3,
        lambda_mask: float = 1.0,
        lambda_cat: float = 0.5,
        dropout: float = 0.05,
        use_summary: bool = True,
    ) -> None:
        super().__init__()
        self.qwen = qwen
        base = base_model_of(qwen)
        self.config = base.config
        cfg = base.config
        hidden = int(cfg.text_config.hidden_size)
        n_layers = len(text_layers(qwen))
        n_deepstack = len(getattr(cfg.vision_config, "deepstack_visual_indexes", []) or [])
        # deepstack features are added to the outputs of the first n_deepstack layers; read after them
        if not (n_deepstack + 1 <= feature_layer < n_layers):
            raise ValueError(f"feature_layer must be in [{n_deepstack + 1}, {n_layers - 1}], got {feature_layer}")
        self.hparams = dict(feature_layer=feature_layer, query_dim=query_dim, decoder_layers=decoder_layers,
                            num_heads=num_heads, mask_dim=mask_dim, logit_scale=logit_scale, topk=topk,
                            summary_drop=summary_drop, lambda_mask=lambda_mask, lambda_cat=lambda_cat,
                            dropout=dropout, tgt_token_id=tgt_token_id, im_start_id=im_start_id,
                            use_summary=use_summary)
        self.use_summary = use_summary  # False: <TGT> gets the null vector (ablation without summary tokens)
        self.image_token_id = int(cfg.image_token_id)
        self.vision_start_id = int(cfg.vision_start_token_id)
        self.tgt_token_id = int(tgt_token_id)
        self.im_start_id = im_start_id
        self.feature_layer = feature_layer
        self.topk = topk
        self.summary_drop = summary_drop
        self.logit_scale = logit_scale
        self.lambda_mask = lambda_mask
        self.lambda_cat = lambda_cat

        # heads (all trainable; the LLM itself is trained only through LoRA)
        self.text_proj = nn.Linear(hidden, query_dim)
        self.visual_proj = nn.Linear(hidden, query_dim)
        self.query_proj = nn.Linear(hidden, query_dim)
        self.query_bias = nn.Parameter(torch.zeros(query_dim))
        self.decoder = nn.ModuleList([QueryLayer(query_dim, num_heads, dropout) for _ in range(decoder_layers)])
        self.mask_q = nn.Linear(query_dim, mask_dim)
        self.mask_v = nn.Linear(query_dim, mask_dim)
        self.summary_mean = SummaryMLP(hidden)
        self.summary_topk = SummaryMLP(hidden)
        self.null_summary = nn.Parameter(torch.zeros(2, hidden))

        self._ctx: Dict[str, Any] = {"active": False}
        self.last_metrics: Dict[str, float] = {}
        self._hook = text_layers(qwen)[feature_layer - 1].register_forward_hook(self._layer_hook)

    # ------------------------------------------------------------------ positions
    def head_modules(self) -> List[nn.Module]:
        return [self.text_proj, self.visual_proj, self.query_proj, self.decoder, self.mask_q, self.mask_v,
                self.summary_mean, self.summary_topk]

    def head_parameters(self):
        for m in self.head_modules():
            yield from m.parameters()
        yield self.query_bias
        yield self.null_summary

    def positions(self, ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ids: (L,) one sample. Returns index tensors for V, E, q0 and the two <TGT>."""
        vis = torch.nonzero(ids == self.image_token_id).flatten()
        tgt = torch.nonzero(ids == self.tgt_token_id).flatten()
        if vis.numel() == 0:
            raise ValueError("no image tokens in the sequence")
        if tgt.numel() != 2:
            raise ValueError(f"expected 2 <TGT> tokens, found {tgt.numel()} (is <TGT> added to the tokenizer?)")
        first_vs = int(torch.nonzero(ids == self.vision_start_id).flatten()[0])
        start = 0
        if self.im_start_id is not None:
            starts = torch.nonzero(ids[:first_vs] == self.im_start_id).flatten()
            if starts.numel():
                start = int(starts[-1]) + 1  # user turn (\"user\\n\" + expression text)
        txt = torch.arange(start, first_vs, device=ids.device)
        return {"vis": vis, "txt": txt, "q": tgt[:1] - 1, "tgt": tgt}

    def set_context(self, input_ids: torch.Tensor, drop: Optional[torch.Tensor] = None) -> None:
        pos = [self.positions(input_ids[b]) for b in range(input_ids.shape[0])]
        if drop is None:
            drop = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        self._ctx = {"active": True, "seq_len": int(input_ids.shape[1]), "pos": pos, "drop": drop.tolist(),
                     "logits": [None] * input_ids.shape[0]}

    def clear_context(self) -> None:
        self._ctx = {"active": False}

    # ------------------------------------------------------------------ heatmap
    def heat_logits(self, v: torch.Tensor, t: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
        """v (N,H) t (T,H) q0 (1,H) layer-L states -> (N,) logits."""
        dt = self.visual_proj.weight.dtype
        vq = self.visual_proj(v.to(dt)).unsqueeze(0)
        tq = self.text_proj(t.to(dt)).unsqueeze(0)
        q = (self.query_proj(q0.to(dt)) + self.query_bias).unsqueeze(0)  # (1,1,D)
        for layer in self.decoder:
            q = layer(q, tq, vq)
        qn = F.normalize(self.mask_q(q[0, 0]), dim=-1)
        vn = F.normalize(self.mask_v(vq[0]), dim=-1)
        return (vn @ qn) * self.logit_scale

    def _layer_hook(self, module, args, output):
        ctx = self._ctx
        h = output[0] if isinstance(output, tuple) else output
        if not ctx.get("active") or h.dim() != 3 or h.shape[1] != ctx["seq_len"]:
            return None  # decode steps with KV cache, or no context
        new_h = h.clone()
        for b, pos in enumerate(ctx["pos"]):
            pos = {k: x.to(h.device) for k, x in pos.items()}
            v, t, q0 = h[b, pos["vis"]], h[b, pos["txt"]], h[b, pos["q"]]
            logits = self.heat_logits(v, t, q0)
            ctx["logits"][b] = logits
            p = torch.sigmoid(logits.float()).detach()
            dt = self.summary_mean.net[1].weight.dtype
            if ctx["drop"][b] or not self.use_summary:
                delta = self.null_summary.to(dt)
            else:
                vf = v.to(dt)
                s_mean = (p.to(dt)[:, None] * vf).sum(0) / (p.sum().to(dt) + 1e-6)
                k = min(self.topk, v.shape[0])
                s_top = vf[torch.topk(p, k).indices].mean(0)
                delta = torch.stack([self.summary_mean(s_mean), self.summary_topk(s_top)], 0)
            new_h[b, pos["tgt"]] = h[b, pos["tgt"]] + delta.to(h.dtype)
        if isinstance(output, tuple):
            return (new_h,) + tuple(output[1:])
        return new_h

    # ------------------------------------------------------------------ training forward
    @staticmethod
    def mask_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor):
        """(B,N) each. Weighted BCE (mean over weight) + Dice over tokens with weight > 0."""
        logits = logits.float()
        w = weights.float() * valid.float()
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
        bce = (bce * w).sum(-1) / w.sum(-1).clamp_min(1e-6)
        m = (w > 0).float()
        p = torch.sigmoid(logits) * m
        t = labels.float() * m
        dice = 1 - (2 * (p * t).sum(-1) + 1) / (p.sum(-1) + t.sum(-1) + 1)
        return (bce + dice).mean(), bce.mean(), dice.mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        token_valid: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        kwargs.pop("num_items_in_batch", None)
        drop = None
        if self.training and self.summary_drop > 0:
            drop = torch.rand(input_ids.shape[0]) < self.summary_drop
        self.set_context(input_ids, drop)
        # Non-reentrant gradient checkpointing replays decoder layers during
        # backward.  The layer-L hook therefore needs the same positions and
        # summary-drop decision to remain available until that replay finishes.
        # The next training forward overwrites this context after the previous
        # backward, while eval/inference can release it immediately.
        keep_context_for_checkpoint = self.training and bool(
            getattr(base_model_of(self.qwen), "is_gradient_checkpointing", False)
        )
        extra = {}
        if mm_token_type_ids is not None:
            extra["mm_token_type_ids"] = mm_token_type_ids
        # Only the answer positions need vocabulary logits: computing them for the whole
        # ~3000-token sequence costs seq x 151936 floats (several GB) for nothing.
        want_cat = labels is not None and self.lambda_cat > 0
        keep = 1
        if want_cat:
            has = labels != -100
            first = int(torch.where(has.any(1), has.int().argmax(1), torch.full_like(has[:, 0], labels.shape[1], dtype=torch.long)).min())
            keep = max(2, labels.shape[1] - first + 1)
        try:
            out = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
                logits_to_keep=keep,
                **extra,
            )
            heat = self._ctx["logits"]
        finally:
            if not keep_context_for_checkpoint:
                self.clear_context()
        if any(x is None for x in heat):
            raise RuntimeError("heatmap hook did not run; check feature_layer")
        heat = pad_sequence(heat, batch_first=True, padding_value=-30.0)
        loss = heat.new_zeros((), dtype=torch.float32)
        metrics: Dict[str, float] = {}
        if token_labels is not None:
            n = heat.shape[1]
            m_loss, bce, dice = self.mask_loss(heat, token_labels[:, :n].to(heat.device),
                                               token_weights[:, :n].to(heat.device), token_valid[:, :n].to(heat.device))
            loss = loss + self.lambda_mask * m_loss
            metrics.update(loss_mask=float(m_loss.detach()), loss_bce=float(bce.detach()), loss_dice=float(dice.detach()))
        if want_cat:
            lg = out.logits[:, :-1].float()                     # position t predicts token t+1
            tgt = labels[:, labels.shape[1] - keep + 1:].to(lg.device)
            cat = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            loss = loss + self.lambda_cat * cat
            metrics["loss_cat"] = float(cat.detach())
        metrics["loss_total"] = float(loss.detach())
        self.last_metrics = metrics
        return {"loss": loss, "heat_logits": heat}

    # ------------------------------------------------------------------ save / load
    def save(self, out_dir: str | Path, extra: Optional[Dict[str, Any]] = None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        heads = {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("qwen.")}
        torch.save(heads, out / HEADS_FILE)
        if hasattr(self.qwen, "peft_config"):
            self.qwen.save_pretrained(str(out / LORA_DIR))
        (out / CONFIG_FILE).write_text(json.dumps({"hparams": self.hparams, **(extra or {})}, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    def load_heads(self, ckpt_dir: str | Path) -> None:
        state = torch.load(Path(ckpt_dir) / HEADS_FILE, map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("qwen.")]
        if missing or unexpected:
            raise RuntimeError(f"head checkpoint mismatch: missing={missing} unexpected={unexpected}")


def attach_lora(qwen: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                layers: Optional[List[int]] = None) -> nn.Module:
    """LoRA on the *language model* linear layers only (vision tower untouched).

    ``layers``: optional 0-based decoder-layer indices to restrict LoRA to (VRAM fallback).
    """
    from peft import LoraConfig, get_peft_model

    names = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    layer_pat = r"\d+" if not layers else "(" + "|".join(str(i) for i in layers) + ")"
    target = rf".*language_model\.layers\.{layer_pat}\.(self_attn|mlp)\.({names})"
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target, bias="none")
    return get_peft_model(qwen, cfg)


def load_for_inference(model_path: str, ckpt_dir: str, dtype=torch.bfloat16, device_map: Optional[str] = "auto",
                       device: str = "cuda", drop_summary: bool = False):
    """Base Qwen3-VL + this checkpoint's LoRA + heads. Returns (processor, GroundV2)."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    ckpt = Path(ckpt_dir)
    hp = json.loads((ckpt / CONFIG_FILE).read_text(encoding="utf-8"))["hparams"]
    processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=dtype, device_map=device_map or None)
    if not device_map:
        qwen.to(device)
    if len(processor.tokenizer) > qwen.get_input_embeddings().weight.shape[0]:
        qwen.resize_token_embeddings(len(processor.tokenizer))
    if (ckpt / LORA_DIR).is_dir():
        from peft import PeftModel

        qwen = PeftModel.from_pretrained(qwen, str(ckpt / LORA_DIR))
    model = GroundV2(qwen, tgt_token_id=hp["tgt_token_id"], im_start_id=hp["im_start_id"],
                     feature_layer=hp["feature_layer"], query_dim=hp["query_dim"], decoder_layers=hp["decoder_layers"],
                     num_heads=hp["num_heads"], mask_dim=hp["mask_dim"], logit_scale=hp["logit_scale"], topk=hp["topk"],
                     summary_drop=0.0, lambda_mask=0.0, lambda_cat=0.0, dropout=hp["dropout"],
                     use_summary=hp.get("use_summary", True) and not drop_summary)
    model.load_heads(ckpt)
    dev = next(text_layers(qwen)[hp["feature_layer"] - 1].parameters()).device
    for m in model.head_modules():
        m.to(device=dev, dtype=dtype)
    model.query_bias.data = model.query_bias.data.to(dev, dtype)
    model.null_summary.data = model.null_summary.data.to(dev, dtype)
    model.eval()
    return processor, model
