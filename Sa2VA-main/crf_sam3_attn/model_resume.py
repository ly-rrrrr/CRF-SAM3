from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


class Qwen3VLCandidateMatcher(nn.Module):
    def __init__(
        self,
        model_path: str,
        attn_loss_weight: float = 0.3,
        attn_loss_layer: int = -1,
        attn_loss_eps: float = 1e-6,
        freeze_visual_encoder: bool = True,
        lora_cfg: Dict = None,
        lora_resume_path: str = None,
        attn_implementation: str = "sdpa",
        loss_type: str = "attn",
        align_tau: float = 1.0,
        align_similarity: str = "cosine",
        score_type: str = "mask_align",
        mask_context_alpha: float = 0.8,
        context_mode: str = "full",
        listwise_tau: float = 1.0,
        listwise_neg_policy: str = "all",
        listwise_force_pos: bool = True,
        hard_neg_weight: float = 0.0,
        hard_neg_margin: float = 0.2,
        hard_neg_pos_strategy: str = "max",
        debug_logits: bool = False,
        debug_logits_topk: int = 5,
        device_map=None,
        max_memory=None,
        offload_folder: str = None,
        offload_state_dict: bool = True,
        low_cpu_mem_usage: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.attn_loss_weight = float(attn_loss_weight)
        self.attn_loss_layer = int(attn_loss_layer)
        self.attn_loss_eps = float(attn_loss_eps)
        self.loss_type = (loss_type or "attn").lower()
        self.align_tau = float(align_tau)
        self.align_similarity = (align_similarity or "cosine").lower()
        self.score_type = (score_type or "mask_align").lower()
        self.mask_context_alpha = max(0.0, min(1.0, float(mask_context_alpha)))
        self.context_mode = (context_mode or "full").lower()
        self.listwise_tau = max(float(listwise_tau), 1e-6)
        self.listwise_neg_policy = (listwise_neg_policy or "all").lower()
        self.listwise_force_pos = bool(listwise_force_pos)
        self.hard_neg_weight = max(float(hard_neg_weight), 0.0)
        self.hard_neg_margin = float(hard_neg_margin)
        self.hard_neg_pos_strategy = (hard_neg_pos_strategy or "max").lower()
        self.debug_logits = bool(debug_logits)
        self.debug_logits_topk = int(debug_logits_topk)
        self.device_map = device_map

        use_device_map = device_map not in (None, "", "none", "null", False)
        model_kwargs = {
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
        }
        if use_device_map:
            model_kwargs["device_map"] = device_map
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory
            if offload_folder:
                model_kwargs["offload_folder"] = offload_folder
                model_kwargs["offload_state_dict"] = bool(offload_state_dict)
            model_kwargs["low_cpu_mem_usage"] = bool(low_cpu_mem_usage)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        )
        self.model.gradient_checkpointing_enable()

        if freeze_visual_encoder:
            self.model.visual.requires_grad_(False)

        if lora_resume_path:
            # Load an existing adapter as trainable weights. The caller decides
            # whether this is a fresh-optimizer warm start or a true resume.
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_resume_path,
                is_trainable=True,
            )
            self.model.print_trainable_parameters()
        elif lora_cfg:
            lora = LoraConfig(
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_cfg.get("target_modules"),
            )
            self.model = get_peft_model(self.model, lora)
            self.model.print_trainable_parameters()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        self.merge_size = self.processor.image_processor.merge_size

        if isinstance(device, str) and device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if use_device_map:
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(device)
            self.model.to(self.device)
        self._loss_hook_handle = None
        self._loss_capture = None
        self._register_loss_hook()

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        if self.score_type == "mask_align" and self.loss_type == "attn":
            print("[warn] score_type=mask_align expects loss_type=feature to capture hidden states.")

    def _get_backbone_model(self):
        model = self.model
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model") and not hasattr(model, "language_model"):
            model = model.model
        return model

    def _register_loss_hook(self):
        if self._loss_hook_handle is not None:
            self._loss_hook_handle.remove()
            self._loss_hook_handle = None
        backbone = self._get_backbone_model()
        if not hasattr(backbone, "language_model"):
            return
        layers = backbone.language_model.layers
        if not layers:
            return
        layer_idx = self.attn_loss_layer
        if layer_idx < 0:
            layer_idx = len(layers) + layer_idx
        layer_idx = max(0, min(layer_idx, len(layers) - 1))

        if self.loss_type == "attn":
            target_module = layers[layer_idx].self_attn

            def _capture(_, __, output):
                if isinstance(output, tuple) and len(output) > 1:
                    self._loss_capture = output[1]
                else:
                    self._loss_capture = None

        else:
            target_module = layers[layer_idx]

            def _capture(_, __, output):
                if torch.is_tensor(output):
                    self._loss_capture = output
                elif isinstance(output, (tuple, list)) and output:
                    self._loss_capture = output[0] if torch.is_tensor(output[0]) else None
                else:
                    self._loss_capture = None

        self._loss_hook_handle = target_module.register_forward_hook(_capture)

    def _build_prompt(self, expression: str) -> str:
        return (
            "Treat the masked / highlighted object as the anchor subject of the referring expression.\n"
            "If the expression becomes true when grounded on this object in the frame sequence, answer Yes; otherwise answer No.\n"
            "Answer with a single word only: Yes or No.\n"
            f"Expression: {expression}"
        )

    def _prepare_inputs(self, frames: List, expression: str, answer: str = None):
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": self._build_prompt(expression)})
        messages = [{"role": "user", "content": content}]

        prompt_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            do_sample_frames=False,
        )

        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        input_ids = prompt_inputs["input_ids"]
        attention_mask = prompt_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

        if answer is not None:
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            answer_ids = torch.tensor(answer_ids, dtype=input_ids.dtype).unsqueeze(0)
            input_ids = torch.cat([input_ids, answer_ids], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(answer_ids)], dim=1)
            labels = torch.full_like(input_ids, -100)
            labels[:, prompt_len:] = answer_ids
            inputs["input_ids"] = input_ids.to(self.device)
            inputs["attention_mask"] = attention_mask.to(self.device)
            inputs["labels"] = labels.to(self.device)

        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in prompt_inputs:
                val = prompt_inputs[key]
                if torch.is_tensor(val):
                    inputs[key] = val.to(self.device)
                else:
                    inputs[key] = val

        return inputs, prompt_len

    def _compute_attn_loss(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        attn_weights: torch.Tensor,
        image_grid_thw: torch.Tensor,
        masks: List,
    ) -> torch.Tensor:
        if attn_weights is None or image_grid_thw is None:
            return None

        attn = attn_weights
        if attn.dim() == 4:
            attn = attn.mean(dim=1)[0]
        elif attn.dim() == 3:
            attn = attn.mean(dim=0)
        else:
            return None

        prompt_ids = input_ids.to(attn.device)[0, :prompt_len]
        text_positions = (prompt_ids != self.image_pad_id).nonzero(as_tuple=False).squeeze(1)
        image_positions = (prompt_ids == self.image_pad_id).nonzero(as_tuple=False).squeeze(1)

        if text_positions.numel() == 0 or image_positions.numel() == 0:
            return None

        merge_length = self.merge_size ** 2
        image_grid_thw = image_grid_thw.detach().cpu().tolist()

        offset = 0
        losses = []
        for idx, grid in enumerate(image_grid_thw):
            if idx >= len(masks):
                break
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            num_tokens = (t * h * w) // merge_length
            frame_positions = image_positions[offset: offset + num_tokens]
            offset += num_tokens
            if frame_positions.numel() == 0:
                continue

            grid_h = h // self.merge_size
            grid_w = w // self.merge_size
            if grid_h <= 0 or grid_w <= 0:
                continue

            mask = masks[idx]
            if not torch.is_tensor(mask):
                mask_t = torch.from_numpy(mask).float()
            else:
                mask_t = mask.float()
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=(grid_h, grid_w), mode="nearest")
            mask_flat = (mask_t.squeeze(0).squeeze(0) > 0).float().view(-1).to(attn.device)
            if mask_flat.numel() != frame_positions.numel():
                continue

            attn_frame = attn[text_positions][:, frame_positions].mean(dim=0)
            denom = attn_frame.sum().clamp_min(self.attn_loss_eps)
            inside = (attn_frame * mask_flat).sum()
            losses.append(1.0 - inside / denom)

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _compute_align_loss(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
        masks: List,
    ) -> torch.Tensor:
        if hidden_states is None or image_grid_thw is None:
            return None
        if hidden_states.dim() != 3:
            return None

        prompt_ids = input_ids.to(hidden_states.device)[0, :prompt_len]
        text_positions = (prompt_ids != self.image_pad_id).nonzero(as_tuple=False).squeeze(1)
        image_positions = (prompt_ids == self.image_pad_id).nonzero(as_tuple=False).squeeze(1)

        if text_positions.numel() == 0 or image_positions.numel() == 0:
            return None

        hidden = hidden_states[0]
        text_vec = hidden[text_positions].mean(dim=0, keepdim=True)
        image_feats = hidden[image_positions]

        if self.align_similarity == "cosine":
            text_vec = F.normalize(text_vec, dim=-1)
            image_feats = F.normalize(image_feats, dim=-1)
            sims = (image_feats @ text_vec.t()).squeeze(1)
        else:
            sims = (image_feats * text_vec).sum(dim=-1)

        merge_length = self.merge_size ** 2
        image_grid_thw = image_grid_thw.detach().cpu().tolist()

        offset = 0
        losses = []
        tau = max(self.align_tau, 1e-6)
        for idx, grid in enumerate(image_grid_thw):
            if idx >= len(masks):
                break
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            num_tokens = (t * h * w) // merge_length
            frame_sims = sims[offset: offset + num_tokens]
            offset += num_tokens
            if frame_sims.numel() == 0:
                continue

            grid_h = h // self.merge_size
            grid_w = w // self.merge_size
            if grid_h <= 0 or grid_w <= 0:
                continue

            mask = masks[idx]
            if not torch.is_tensor(mask):
                mask_t = torch.from_numpy(mask).float()
            else:
                mask_t = mask.float()
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=(grid_h, grid_w), mode="nearest")
            mask_flat = (mask_t.squeeze(0).squeeze(0) > 0).float().view(-1).to(frame_sims.device)
            if mask_flat.numel() != frame_sims.numel():
                continue

            probs = F.softmax(frame_sims / tau, dim=0)
            inside = (probs * mask_flat).sum()
            losses.append(1.0 - inside)

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _compute_masked_similarity_score(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
        masks: List,
    ) -> torch.Tensor:
        if hidden_states is None or image_grid_thw is None:
            return None
        if hidden_states.dim() != 3:
            return None

        prompt_ids = input_ids.to(hidden_states.device)[0, :prompt_len]
        text_positions = (prompt_ids != self.image_pad_id).nonzero(as_tuple=False).squeeze(1)
        image_positions = (prompt_ids == self.image_pad_id).nonzero(as_tuple=False).squeeze(1)

        if text_positions.numel() == 0 or image_positions.numel() == 0:
            return None

        hidden = hidden_states[0]
        text_vec = hidden[text_positions].mean(dim=0)
        image_feats = hidden[image_positions]

        if self.align_similarity == "cosine":
            text_vec = F.normalize(text_vec, dim=0)

        merge_length = self.merge_size ** 2
        image_grid_thw = image_grid_thw.detach().cpu().tolist()

        offset = 0
        scores_inside = []
        scores_context = []
        for idx, grid in enumerate(image_grid_thw):
            if idx >= len(masks):
                break
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            num_tokens = (t * h * w) // merge_length
            frame_feats = image_feats[offset: offset + num_tokens]
            offset += num_tokens
            if frame_feats.numel() == 0:
                continue

            grid_h = h // self.merge_size
            grid_w = w // self.merge_size
            if grid_h <= 0 or grid_w <= 0:
                continue

            mask = masks[idx]
            if not torch.is_tensor(mask):
                mask_t = torch.from_numpy(mask).float()
            else:
                mask_t = mask.float()
            mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=(grid_h, grid_w), mode="nearest")
            mask_flat = (mask_t.squeeze(0).squeeze(0) > 0).view(-1).to(frame_feats.device)
            if mask_flat.numel() != frame_feats.shape[0]:
                continue

            inside_mask = mask_flat > 0
            if not inside_mask.any():
                continue

            inside_feat = frame_feats[inside_mask].mean(dim=0)
            if self.context_mode == "outside":
                outside_mask = ~inside_mask
                if outside_mask.any():
                    context_feat = frame_feats[outside_mask].mean(dim=0)
                else:
                    context_feat = frame_feats.mean(dim=0)
            else:
                context_feat = frame_feats.mean(dim=0)

            if self.align_similarity == "cosine":
                inside_feat = F.normalize(inside_feat, dim=0)
                context_feat = F.normalize(context_feat, dim=0)

            sim_inside = (inside_feat * text_vec).sum()
            sim_context = (context_feat * text_vec).sum()
            scores_inside.append(sim_inside)
            scores_context.append(sim_context)

        if not scores_inside:
            return None

        score_inside = torch.stack(scores_inside).mean()
        score_context = torch.stack(scores_context).mean()
        return self.mask_context_alpha * score_inside + (1.0 - self.mask_context_alpha) * score_context

    def _compute_yesno_score(self, outputs, prompt_len: int, score_type: str = None) -> torch.Tensor:
        if not hasattr(outputs, "logits") or outputs.logits is None:
            return None
        idx = outputs.logits.shape[1] - 1
        logits = outputs.logits[0, idx]
        yes_logit = logits[self.yes_token_id]
        no_logit = logits[self.no_token_id]
        score_type = (score_type or self.score_type).lower()
        if score_type == "prob":
            yes_no_logits = torch.stack([yes_logit, no_logit], dim=0)
            yes_no_probs = F.softmax(yes_no_logits.float(), dim=0)
            score = yes_no_probs[0]
        else:
            score = yes_logit - no_logit

        if self.debug_logits:
            topk = max(1, self.debug_logits_topk)
            top_vals, top_ids = torch.topk(logits.float(), k=topk)
            top_tokens = self.tokenizer.convert_ids_to_tokens(top_ids.tolist())
            yes_no_probs = F.softmax(torch.stack([yes_logit, no_logit]).float(), dim=0)
            print(
                f"[logits] shape={tuple(outputs.logits.shape)} idx={idx} "
                f"p_yes={float(yes_no_probs[0].detach()):.4f} "
                f"yes_logit={float(yes_logit.detach()):.4f} no_logit={float(no_logit.detach()):.4f}"
            )
            print("[logits] topk:", list(zip(top_tokens, top_vals.tolist())))

        return score

    def _reduce_positive_scores(self, pos_scores: torch.Tensor, strategy: str = None) -> torch.Tensor:
        strategy = (strategy or self.hard_neg_pos_strategy or "max").lower()
        if strategy == "mean":
            return pos_scores.mean()
        if strategy == "logsumexp":
            return torch.logsumexp(pos_scores, dim=0)
        return pos_scores.max()

    def _compute_hard_negative_loss(
        self,
        rank_scores: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        weight: float = None,
        margin: float = None,
        pos_strategy: str = None,
    ) -> torch.Tensor:
        weight = self.hard_neg_weight if weight is None else float(weight)
        margin = self.hard_neg_margin if margin is None else float(margin)
        pos_strategy = pos_strategy or self.hard_neg_pos_strategy

        if weight <= 0:
            return None
        if pos_mask is None or neg_mask is None:
            return None
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return None

        pos_scores = rank_scores[pos_mask]
        neg_scores = rank_scores[neg_mask]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return None

        pos_ref = self._reduce_positive_scores(pos_scores, strategy=pos_strategy)
        # Scheme A with a compatible margin: penalize every negative whose
        # score still violates the desired positive safety gap.
        violating_negs = neg_scores[neg_scores > (pos_ref - margin)]
        if violating_negs.numel() == 0:
            return None
        return F.softplus(violating_negs - pos_ref + margin).mean()

    def _forward_listwise(self, batch: Dict) -> Dict:
        expression = batch["expression"]
        candidates = batch.get("candidates", [])
        pos_obj_ids = set(batch.get("pos_obj_ids", []))
        neg_obj_ids = set(batch.get("neg_obj_ids", []))
        iou_map = batch.get("iou_map", {})
        effective_neg_policy = (batch.get("listwise_neg_policy") or self.listwise_neg_policy or "all").lower()
        effective_hard_neg_weight = float(
            self.hard_neg_weight if batch.get("hard_neg_weight") is None else batch.get("hard_neg_weight")
        )
        effective_hard_neg_margin = float(
            self.hard_neg_margin if batch.get("hard_neg_margin") is None else batch.get("hard_neg_margin")
        )
        effective_hard_neg_pos_strategy = (
            batch.get("hard_neg_pos_strategy") or self.hard_neg_pos_strategy or "max"
        ).lower()

        scores = []
        rank_scores = []
        score_ids = []
        attn_losses = []

        for cand in candidates:
            frames = cand["frames"]
            masks = cand["masks"]
            obj_id = int(cand["obj_id"])

            inputs, prompt_len = self._prepare_inputs(frames, expression, answer=None)

            self._loss_capture = None
            outputs = self.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                logits_to_keep=1,
                use_cache=False,
                return_dict=True,
            )

            attn_loss = None
            if "image_grid_thw" in inputs and self._loss_capture is not None:
                if self.loss_type == "attn":
                    attn_loss = self._compute_attn_loss(
                        inputs["input_ids"],
                        prompt_len,
                        self._loss_capture,
                        inputs["image_grid_thw"],
                        masks,
                    )
                else:
                    attn_loss = self._compute_align_loss(
                        inputs["input_ids"],
                        prompt_len,
                        self._loss_capture,
                        inputs["image_grid_thw"],
                        masks,
                    )

            if self.score_type == "mask_align":
                score = self._compute_masked_similarity_score(
                    inputs["input_ids"],
                    prompt_len,
                    self._loss_capture,
                    inputs.get("image_grid_thw"),
                    masks,
                )
                rank_score = score
            else:
                score = self._compute_yesno_score(outputs, prompt_len)
                rank_score = self._compute_yesno_score(outputs, prompt_len, score_type="logit")

            if score is None:
                continue

            scores.append(score)
            rank_scores.append(rank_score if rank_score is not None else score)
            score_ids.append(obj_id)
            if attn_loss is not None:
                attn_losses.append(attn_loss)

        if not scores:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss": zero,
                "loss_cls": zero,
                "loss_attn": None,
                "scores": {},
                "pos_obj_ids": list(pos_obj_ids),
                "neg_obj_ids": list(neg_obj_ids),
            }

        score_tensor = torch.stack(scores)
        rank_score_tensor = torch.stack(rank_scores)
        pos_mask = torch.tensor(
            [obj_id in pos_obj_ids for obj_id in score_ids],
            device=score_tensor.device,
            dtype=torch.bool,
        )
        neg_rank_mask = None

        if effective_neg_policy == "threshold":
            neg_mask = torch.tensor(
                [obj_id in neg_obj_ids for obj_id in score_ids],
                device=score_tensor.device,
                dtype=torch.bool,
            )
            valid_mask = pos_mask | neg_mask
            if valid_mask.any():
                score_tensor = score_tensor[valid_mask]
                rank_score_tensor = rank_score_tensor[valid_mask]
                pos_mask = pos_mask[valid_mask]
                neg_rank_mask = neg_mask[valid_mask]
                score_ids = [obj_id for obj_id, keep in zip(score_ids, valid_mask.tolist()) if keep]
        else:
            neg_rank_mask = ~pos_mask

        if pos_mask.sum() == 0 and self.listwise_force_pos and score_ids:
            best_id = None
            best_iou = -1.0
            for obj_id in score_ids:
                iou = iou_map.get(obj_id)
                if iou is None:
                    iou = iou_map.get(str(obj_id), 0.0)
                if iou > best_iou:
                    best_iou = iou
                    best_id = obj_id
            if best_id is not None:
                pos_mask = torch.tensor(
                    [obj_id == best_id for obj_id in score_ids],
                    device=score_tensor.device,
                    dtype=torch.bool,
                )
                if effective_neg_policy == "all":
                    neg_rank_mask = ~pos_mask
                elif neg_rank_mask is not None:
                    neg_rank_mask = neg_rank_mask & (~pos_mask)

        if pos_mask.sum() == 0:
            loss_cls = torch.tensor(0.0, device=score_tensor.device)
        else:
            scores_scaled = score_tensor / self.listwise_tau
            loss_cls = torch.logsumexp(scores_scaled, dim=0) - torch.logsumexp(scores_scaled[pos_mask], dim=0)

        loss_rank = self._compute_hard_negative_loss(
            rank_score_tensor,
            pos_mask,
            neg_rank_mask,
            weight=effective_hard_neg_weight,
            margin=effective_hard_neg_margin,
            pos_strategy=effective_hard_neg_pos_strategy,
        )
        loss_attn = torch.stack(attn_losses).mean() if attn_losses else None
        loss_total = loss_cls
        if loss_rank is not None and effective_hard_neg_weight > 0:
            loss_rank = loss_rank.to(loss_cls.device)
            loss_total = loss_total + loss_rank * effective_hard_neg_weight
        if loss_attn is not None and self.attn_loss_weight > 0:
            loss_attn = loss_attn.to(loss_cls.device)
            loss_total = loss_total + loss_attn * self.attn_loss_weight

        score_map = {obj_id: float(score.detach()) for obj_id, score in zip(score_ids, score_tensor)}

        return {
            "loss": loss_total,
            "loss_cls": loss_cls.detach(),
            "loss_rank": loss_rank.detach() if loss_rank is not None else None,
            "loss_attn": loss_attn.detach() if loss_attn is not None else None,
            "scores": score_map,
            "pos_obj_ids": list(pos_obj_ids),
            "neg_obj_ids": list(neg_obj_ids),
        }

    def forward(self, batch: Dict) -> Dict:
        if "candidates" in batch:
            return self._forward_listwise(batch)

        frames = batch["frames"]
        masks = batch["masks"]
        expression = batch["expression"]
        label = int(batch["label"])

        answer = "Yes" if label == 1 else "No"
        inputs, prompt_len = self._prepare_inputs(frames, expression, answer)

        self._loss_capture = None
        outputs = self.model(
            **inputs,
            output_hidden_states=False,
            output_attentions=False,
            logits_to_keep=max(int(inputs["input_ids"].shape[1] - prompt_len), 1) if "labels" in inputs else 1,
            use_cache=False,
            return_dict=True,
        )

        loss_cls = outputs.loss
        loss_rank = None
        attn_loss = None
        if "image_grid_thw" in inputs and self._loss_capture is not None:
            if self.loss_type == "attn":
                attn_loss = self._compute_attn_loss(
                    inputs["input_ids"],
                    prompt_len,
                    self._loss_capture,
                    inputs["image_grid_thw"],
                    masks,
                )
            else:
                attn_loss = self._compute_align_loss(
                    inputs["input_ids"],
                    prompt_len,
                    self._loss_capture,
                    inputs["image_grid_thw"],
                    masks,
                )
        loss_total = loss_cls
        if attn_loss is not None and self.attn_loss_weight > 0:
            attn_loss = attn_loss.to(loss_cls.device)
            loss_total = loss_total + attn_loss * self.attn_loss_weight

        prob_yes = None
        prob_no = None
        pred = None
        if hasattr(outputs, "logits") and outputs.logits is not None:
            idx = max(prompt_len - 1, 0)
            logits = outputs.logits[0, idx]
            yes_no_logits = torch.stack([logits[self.yes_token_id], logits[self.no_token_id]], dim=0)
            yes_no_probs = F.softmax(yes_no_logits.float(), dim=0)
            prob_yes = yes_no_probs[0].detach()
            prob_no = yes_no_probs[1].detach()
            pred = "Yes" if prob_yes >= prob_no else "No"

        return {
            "loss": loss_total,
            "loss_cls": loss_cls.detach(),
            "loss_rank": loss_rank,
            "loss_attn": attn_loss.detach() if attn_loss is not None else None,
            "prob_yes": prob_yes,
            "prob_no": prob_no,
            "pred": pred,
        }

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
