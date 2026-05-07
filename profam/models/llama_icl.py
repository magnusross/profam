"""Llama-based supervised in-context-learning module.

The architecture matches ``LlamaLitModule`` for the AA branch and adds:

* ``value_in_proj``  - lifts the scalar fitness value into the model's hidden
  dimension. The output replaces the ``[VAL_SLOT]`` input embedding at runtime.
* ``value_out_head`` - a linear regression head applied to the hidden state at
  ``[VAL]`` positions to predict standardised fitness.

The fitness loss is mean-squared error in z-score space; the original CE loss
is kept (with ``[VAL]``/``[VAL_SLOT]`` already masked by the collator). The
total objective is ``alpha * ce + beta * mse``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.optimization import get_scheduler

from profam.data.icl_constants import VAL_SLOT_TOKEN_ID, VAL_TOKEN_ID
from profam.models.base import BaseFamilyLitModule
from profam.models.utils import log_likelihood_from_outputs
from profam.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class FourierValueFeaturiser(nn.Module):
    """Sin/cos featurisation followed by a learned linear lift to ``hidden_size``.

    Frequencies span ``[10**log_freq_min, 10**log_freq_max]`` log-uniformly.
    Used when ``value_featurisation == "fourier"``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_frequencies: int = 64,
        log_freq_min: float = -2.0,
        log_freq_max: float = 2.0,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        ks = torch.linspace(log_freq_min, log_freq_max, num_frequencies)
        # store frequencies as a buffer so they move with the module's device.
        self.register_buffer("omegas", 2 * torch.pi * (10.0 ** ks))
        self.proj = nn.Linear(2 * num_frequencies, hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: (..., 1) -> (..., num_frequencies)
        v = values.squeeze(-1).unsqueeze(-1) * self.omegas
        feats = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        return self.proj(feats)


class LinearValueFeaturiser(nn.Module):
    """Affine lift ``y -> W @ y + b`` (the v1 default)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.proj(values)


def _build_value_featuriser(name: str, hidden_size: int, **kwargs) -> nn.Module:
    if name == "linear":
        return LinearValueFeaturiser(hidden_size)
    if name == "fourier":
        return FourierValueFeaturiser(hidden_size, **kwargs)
    raise ValueError(f"Unknown value_featurisation {name!r}; expected linear|fourier")


class LlamaICLLitModule(BaseFamilyLitModule):
    """Lightning module for supervised in-context fitness regression."""

    def __init__(
        self,
        config: LlamaConfig,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        num_decay_steps: Optional[int] = None,
        scoring_max_tokens: int = 10240,
        use_kv_cache_for_scoring: bool = True,
        pass_res_pos_in_doc_as_position_ids: bool = True,
        optimizer: str = "adamw",
        override_optimizer_on_load: bool = False,
        ce_loss_weight: float = 1.0,
        mse_loss_weight: float = 1.0,
        value_featurisation: str = "linear",
        value_featuriser_kwargs: Optional[Dict[str, Any]] = None,
        likelihood_featurisation: str = "linear",
        likelihood_featuriser_kwargs: Optional[Dict[str, Any]] = None,
        likelihood_score_grad: bool = False,
        likelihood_sigma_floor: float = 1e-3,
        backbone_lr_scale: float = 0.1,
        reinit_value_token_embeddings: bool = True,
        pretrained_ckpt_path: Optional[str] = None,
        pretrained_strict: bool = False,
    ) -> None:
        model = LlamaForCausalLM(config)
        super().__init__(
            model,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps,
            scoring_max_tokens=scoring_max_tokens,
            use_kv_cache_for_scoring=use_kv_cache_for_scoring,
            override_optimizer_on_load=override_optimizer_on_load,
            pass_res_pos_in_doc_as_position_ids=pass_res_pos_in_doc_as_position_ids,
        )
        self.ce_loss_weight = float(ce_loss_weight)
        self.mse_loss_weight = float(mse_loss_weight)
        self.value_featurisation = value_featurisation
        self.backbone_lr_scale = float(backbone_lr_scale)

        hidden_size = config.hidden_size
        self.value_in_proj = _build_value_featuriser(
            value_featurisation, hidden_size, **(value_featuriser_kwargs or {})
        )
        # Separate featuriser for the zero-shot likelihood z-score injected at
        # the [VAL] (SP1) input slot. Same featuriser interface as ``value_in_proj``.
        self.likelihood_in_proj = _build_value_featuriser(
            likelihood_featurisation,
            hidden_size,
            **(likelihood_featuriser_kwargs or {}),
        )
        self.likelihood_score_grad = bool(likelihood_score_grad)
        self.likelihood_sigma_floor = float(likelihood_sigma_floor)
        self.value_out_head = nn.Linear(hidden_size, 1)
        # Save extra hparams (the base class already saved its own).
        self.hparams["ce_loss_weight"] = self.ce_loss_weight
        self.hparams["mse_loss_weight"] = self.mse_loss_weight
        self.hparams["value_featurisation"] = self.value_featurisation
        self.hparams["likelihood_featurisation"] = likelihood_featurisation
        self.hparams["likelihood_score_grad"] = self.likelihood_score_grad
        self.hparams["likelihood_sigma_floor"] = self.likelihood_sigma_floor
        self.hparams["backbone_lr_scale"] = self.backbone_lr_scale
        self.hparams["optimizer"] = optimizer

        if pretrained_ckpt_path is not None:
            self._load_pretrained_state_dict(
                pretrained_ckpt_path, strict=pretrained_strict
            )

        if reinit_value_token_embeddings:
            self._reinit_value_token_embeddings()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _load_pretrained_state_dict(self, ckpt_path: str, strict: bool = False) -> None:
        """Load weights from a ProFam checkpoint as a fine-tune init.

        Uses ``torch.load(weights_only=False)`` to mirror
        :func:`profam.models.base.load_checkpoint` - the shipped ProFam
        checkpoint pickles ``ProFamTokenizer`` / ``LlamaConfig`` into
        ``hyper_parameters``, which the PyTorch 2.6+ default rejects. We only
        copy ``state_dict`` across; optimizer / scheduler / global_step are
        not restored (this is a fine-tune init, not a Lightning resume).
        """
        import sys
        import types

        from profam.constants import resolve_runtime_path
        from profam.data import tokenizers as _profam_tokenizers

        # The published ProFam-1 ckpt was pickled when the package lived under
        # `src.*`. Alias the legacy module path so unpickling resolves.
        sys.modules.setdefault("src", types.ModuleType("src"))
        sys.modules.setdefault("src.data", types.ModuleType("src.data"))
        sys.modules.setdefault("src.data.tokenizers", _profam_tokenizers)

        resolved = resolve_runtime_path(ckpt_path)
        log.info(
            "Loading pretrained weights from %s (state_dict only; optimizer "
            "and scheduler are not restored)",
            resolved,
        )
        ckpt_blob = torch.load(str(resolved), map_location="cpu", weights_only=False)
        state_dict = ckpt_blob["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            log.info(
                "%d new params not present in pretrained ckpt (expected: ICL heads): %s",
                len(missing),
                missing[:5] + (["..."] if len(missing) > 5 else []),
            )
        if unexpected:
            log.info(
                "%d pretrained params unused by current model: %s",
                len(unexpected),
                unexpected[:5] + (["..."] if len(unexpected) > 5 else []),
            )

    def _reinit_value_token_embeddings(self) -> None:
        """Re-initialise the embedding rows for the two repurposed tokens.

        In pretraining ``[SP1]``/``[SP2]`` are unused, so their rows are random
        but baked-in. We re-randomise them at fine-tune start so the optimiser
        is not nudged by stale init noise.
        """
        embed = self.model.get_input_embeddings()
        with torch.no_grad():
            std = float(getattr(self.model.config, "initializer_range", 0.02))
            for tok_id in (VAL_TOKEN_ID, VAL_SLOT_TOKEN_ID):
                embed.weight[tok_id].normal_(mean=0.0, std=std)

    def _icl_position_ids(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """ICL documents start with one BOS at position 0; position_ids = arange(L).

        We compute them explicitly rather than letting the base class assert
        ``batch_size == 1``, because the ICL fine-tune is happy with batched
        unpacked documents.
        """
        if not self.pass_res_pos_in_doc_as_position_ids:
            return None
        L = input_ids.shape[1]
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        return position_ids.expand(input_ids.shape[0], -1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        value_slot_mask: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        val_marker_mask: Optional[torch.Tensor] = None,
        predict_mask: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        variant_token_ids: Optional[torch.Tensor] = None,
        variant_attn_mask: Optional[torch.Tensor] = None,
        variant_valid_mask: Optional[torch.Tensor] = None,
        ll_z: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not (input_ids[:, 0] == self.tokenizer.bos_token_id).all():
            raise ValueError("ICL documents must start with [start-of-document]")
        if labels is not None:
            labels = labels.clone()
            labels[labels == self.tokenizer.bos_token_id] = self.ignore_index

        position_ids = self._icl_position_ids(input_ids)

        embeds = self.model.get_input_embeddings()(input_ids)
        if value_slot_mask is not None and bool(value_slot_mask.any()):
            value_inputs = values.unsqueeze(-1).to(embeds.dtype)
            embedded_y = self.value_in_proj(value_inputs).to(embeds.dtype)
            embeds = torch.where(
                value_slot_mask.unsqueeze(-1), embedded_y, embeds
            )

        # Inject the zero-shot likelihood z-score into [VAL] input slots.
        # ``ll_z`` may be precomputed by the caller (so the raw LL can also be
        # used for diagnostic correlations); otherwise we compute it here.
        if (
            ll_z is None
            and variant_token_ids is not None
            and val_marker_mask is not None
            and bool(val_marker_mask.any())
        ):
            raw_ll, vmask = self._compute_zero_shot_ll_raw(
                variant_token_ids=variant_token_ids,
                variant_attn_mask=variant_attn_mask,
                variant_valid_mask=variant_valid_mask,
            )
            ll_z = self._zscore_ll_per_doc(raw_ll, vmask)
        if ll_z is not None and val_marker_mask is not None and bool(val_marker_mask.any()):
            embedded_ll = self.likelihood_in_proj(ll_z.unsqueeze(-1).to(embeds.dtype)).to(
                embeds.dtype
            )  # (B, K_max, hidden)

            # Map each [VAL] position in the document to its variant index
            # (cumulative count of [VAL] markers seen so far - 1). Positions
            # that are not [VAL] get a dummy index 0; we only use the gather
            # output where val_marker_mask is True.
            var_idx = (
                val_marker_mask.long().cumsum(dim=1) - 1
            ).clamp(min=0)  # (B, L)
            gather_idx = var_idx.unsqueeze(-1).expand(-1, -1, embeds.shape[-1])
            gathered = torch.gather(embedded_ll, dim=1, index=gather_idx)
            embeds = torch.where(
                val_marker_mask.unsqueeze(-1), gathered, embeds
            )

        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            labels=labels,
        )
        return outputs

    # ------------------------------------------------------------------
    # Zero-shot likelihood scoring sub-pass
    # ------------------------------------------------------------------

    def _compute_zero_shot_ll_raw(
        self,
        variant_token_ids: torch.Tensor,
        variant_attn_mask: torch.Tensor,
        variant_valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the model in no-context mode on each variant; return raw mean LL.

        ``variant_token_ids``  shape (B, K_max, L_max). Each row is
        ``[BOS] [DOC_TYPE] aa_1 ... aa_n [SEP]`` padded to L_max with pad_token_id.
        ``variant_attn_mask``  same shape, 1 on real tokens, 0 on padding.
        ``variant_valid_mask`` shape (B, K_max). False marks padded variants
        added by the collator to align doc-level K's across the batch.

        Returns ``(raw_ll, valid_mask)`` both shaped ``(B, K_max)``. Entries
        where ``valid_mask`` is False have ``raw_ll == 0``; do not read them.
        """
        B, K_max, L_max = variant_token_ids.shape
        device = variant_token_ids.device
        pad_id = self.tokenizer.pad_token_id

        # Flatten to (B*K_max, L) and select only valid rows for the forward.
        flat_ids = variant_token_ids.view(B * K_max, L_max)
        flat_attn = variant_attn_mask.view(B * K_max, L_max)
        if variant_valid_mask is None:
            flat_valid = torch.ones(B * K_max, dtype=torch.bool, device=device)
        else:
            flat_valid = variant_valid_mask.view(B * K_max).to(torch.bool)

        valid_ids = flat_ids[flat_valid]
        valid_attn = flat_attn[flat_valid]
        if valid_ids.numel() == 0:
            empty = torch.zeros(B, K_max, device=device, dtype=torch.float32)
            return empty, flat_valid.view(B, K_max)

        labels = torch.where(
            valid_ids == pad_id,
            torch.full_like(valid_ids, -100),
            valid_ids,
        )

        with torch.set_grad_enabled(self.likelihood_score_grad):
            outputs = self.model(
                input_ids=valid_ids,
                attention_mask=valid_attn,
                use_cache=False,
            )
            # start_ix=1 mirrors profam.models.base._score_seqs_no_context:
            # skip BOS and predict the actual sequence (positions 2..L-1).
            log_likelihood = log_likelihood_from_outputs(
                outputs, labels, start_ix=1
            )  # (n_valid, L-2)
            shift_labels = labels[..., 2:]
            mask = (shift_labels != -100).to(log_likelihood.dtype)
            denom = mask.sum(dim=-1).clamp(min=1.0)
            ll_mean_valid = (log_likelihood.float() * mask).sum(dim=-1) / denom

        if not self.likelihood_score_grad:
            ll_mean_valid = ll_mean_valid.detach()

        ll_full = torch.zeros(B * K_max, device=device, dtype=ll_mean_valid.dtype)
        ll_full[flat_valid] = ll_mean_valid
        return ll_full.view(B, K_max), flat_valid.view(B, K_max)

    def _zscore_ll_per_doc(
        self, raw_ll: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-document z-score over *valid* variants; padded entries -> 0."""
        valid_f = valid_mask.to(raw_ll.dtype)
        n_valid = valid_f.sum(dim=1).clamp(min=1.0)
        mu = (raw_ll * valid_f).sum(dim=1) / n_valid
        diffs = (raw_ll - mu.unsqueeze(1)) * valid_f
        var = (diffs * diffs).sum(dim=1) / n_valid
        sigma = var.clamp_min(self.likelihood_sigma_floor**2).sqrt()
        z = (raw_ll - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        return z * valid_f

    def _compute_ll_fitness_spearman(
        self,
        raw_ll: torch.Tensor,
        valid_mask: torch.Tensor,
        target_values: torch.Tensor,
        val_marker_mask: torch.Tensor,
    ) -> Optional[float]:
        """Mean per-document Spearman(zero-shot LL, true fitness z-score).

        Per document, lines up the K+1 raw LL values against the ``target_values``
        read off at ``val_marker_mask`` positions (variant order). Returns the
        mean across documents, or ``None`` if no document has >=2 valid pairs.
        """
        B, K_max = raw_ll.shape
        rhos = []
        ll_cpu = raw_ll.detach().float().cpu().numpy()
        valid_cpu = valid_mask.detach().cpu().numpy()
        for b in range(B):
            n_valid = int(valid_cpu[b].sum())
            if n_valid < 2:
                continue
            ll_b = ll_cpu[b, :n_valid]
            marker_pos = torch.nonzero(val_marker_mask[b], as_tuple=False).flatten()
            if int(marker_pos.numel()) != n_valid:
                continue  # shape mismatch -> skip this doc
            tgt_b = target_values[b, marker_pos].detach().float().cpu().numpy()
            if np.std(ll_b) == 0 or np.std(tgt_b) == 0:
                continue
            rho, _ = spearmanr(ll_b, tgt_b)
            if rho is not None and not np.isnan(rho):
                rhos.append(float(rho))
        if not rhos:
            return None
        return float(np.mean(rhos))

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_position_mask(predict_mask: torch.Tensor) -> torch.Tensor:
        """Return a (B, L) bool mask selecting the *last* True per row of ``predict_mask``.

        Rows with no True positions contribute no entries.
        """
        query_mask = torch.zeros_like(predict_mask)
        if predict_mask.numel() == 0:
            return query_mask
        # last-True index per row; rows with all-False get 0 but are filtered next.
        L = predict_mask.shape[1]
        idx = torch.arange(L, device=predict_mask.device)
        masked_idx = torch.where(
            predict_mask, idx.unsqueeze(0), torch.full_like(predict_mask, -1, dtype=torch.long)
        )
        last_idx = masked_idx.max(dim=1).values  # (B,), -1 where row is empty
        has_any = last_idx >= 0
        rows = torch.nonzero(has_any, as_tuple=False).squeeze(-1)
        if rows.numel() > 0:
            query_mask[rows, last_idx[rows]] = True
        return query_mask

    def _icl_loss(
        self,
        outputs,
        predict_mask: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        last_hidden = outputs.hidden_states[-1]  # (B, L, d)
        preds = self.value_out_head(last_hidden).squeeze(-1).float()  # (B, L)
        targets = target_values.to(preds.dtype)
        if predict_mask.any():
            mse_all = F.mse_loss(preds[predict_mask], targets[predict_mask])
        else:
            mse_all = preds.new_zeros(())

        query_mask = self._query_position_mask(predict_mask)
        if query_mask.any():
            mse_query = F.mse_loss(preds[query_mask], targets[query_mask])
        else:
            mse_query = preds.new_zeros(())

        ce = outputs.loss if outputs.loss is not None else preds.new_zeros(())
        # Replace any NaN CE (which can happen if every label was ignored) with
        # zero so the joint loss stays finite.
        if torch.isnan(ce):
            ce = preds.new_zeros(())

        # Training/optimisation objective uses the all-values MSE.
        total = self.ce_loss_weight * ce + self.mse_loss_weight * mse_all
        return {
            "loss": total,
            "ce_loss": ce,
            "mse_loss": mse_all,
            "mse_loss_query": mse_query,
            "preds": preds,
            "query_mask": query_mask,
        }

    def _shared_step(self, batch: Dict[str, torch.Tensor]):
        # Compute the no-context LL once so we can reuse the *raw* values for
        # the diagnostic Spearman vs. true fitness, while passing the
        # per-document z-score on to ``forward`` for the input-slot injection.
        raw_ll: Optional[torch.Tensor] = None
        ll_valid: Optional[torch.Tensor] = None
        ll_z: Optional[torch.Tensor] = None
        if (
            batch.get("variant_token_ids") is not None
            and batch.get("val_marker_mask") is not None
            and bool(batch["val_marker_mask"].any())
        ):
            raw_ll, ll_valid = self._compute_zero_shot_ll_raw(
                variant_token_ids=batch["variant_token_ids"],
                variant_attn_mask=batch["variant_attn_mask"],
                variant_valid_mask=batch.get("variant_valid_mask"),
            )
            ll_z = self._zscore_ll_per_doc(raw_ll, ll_valid)

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            value_slot_mask=batch["value_slot_mask"],
            values=batch["values"],
            val_marker_mask=batch.get("val_marker_mask"),
            predict_mask=batch["predict_mask"],
            target_values=batch["target_values"],
            labels=batch.get("labels"),
            variant_token_ids=batch.get("variant_token_ids"),
            variant_attn_mask=batch.get("variant_attn_mask"),
            variant_valid_mask=batch.get("variant_valid_mask"),
            ll_z=ll_z,
        )
        loss_dict = self._icl_loss(
            outputs,
            predict_mask=batch["predict_mask"],
            target_values=batch["target_values"],
        )
        if raw_ll is not None and ll_valid is not None:
            rho = self._compute_ll_fitness_spearman(
                raw_ll=raw_ll,
                valid_mask=ll_valid,
                target_values=batch["target_values"],
                val_marker_mask=batch["val_marker_mask"],
            )
            loss_dict["zero_shot_ll_spearman"] = rho  # may be None if undefined
        return outputs, loss_dict

    def _diagnostic_stats(
        self,
        batch: Dict[str, torch.Tensor],
        loss_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Cheap scalars that help spot pipeline issues.

        Includes counts (predict positions, value-slot positions, batch shape)
        and the first/second moments of preds and targets at predict positions.
        Returned tensors live on the model device so ``log_dict`` handles them.
        """
        preds = loss_dict["preds"]
        device = preds.device
        predict_mask = batch["predict_mask"]
        value_slot_mask = batch.get("value_slot_mask")
        target_values = batch["target_values"]
        B, L = predict_mask.shape

        n_predict = predict_mask.sum().to(torch.float32)
        n_value_slots = (
            value_slot_mask.sum().to(torch.float32)
            if value_slot_mask is not None
            else torch.tensor(0.0, device=device)
        )

        if predict_mask.any():
            preds_at = preds[predict_mask].float()
            targets_at = target_values[predict_mask].float()
            preds_mean = preds_at.mean()
            preds_std = preds_at.std(unbiased=False) if preds_at.numel() > 1 else preds_at.new_zeros(())
            targets_mean = targets_at.mean()
            targets_std = (
                targets_at.std(unbiased=False) if targets_at.numel() > 1 else targets_at.new_zeros(())
            )
        else:
            zero = preds.new_zeros(())
            preds_mean = preds_std = targets_mean = targets_std = zero

        return {
            "num_predict_positions": n_predict,
            "num_value_slot_positions": n_value_slots,
            "preds_mean": preds_mean,
            "preds_std": preds_std,
            "targets_mean": targets_mean,
            "targets_std": targets_std,
            "batch_size": torch.tensor(float(B), device=device),
            "seq_len": torch.tensor(float(L), device=device),
        }

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _outputs, loss_dict = self._shared_step(batch)
        stats = self._diagnostic_stats(batch, loss_dict)
        self.log_dict(
            {
                "train/loss": loss_dict["loss"],
                "train/ce_loss": loss_dict["ce_loss"],
                "train/mse_loss": loss_dict["mse_loss"],
                "train/mse_loss_query": loss_dict["mse_loss_query"],
                **{f"train/{k}": v for k, v in stats.items()},
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
        )

        # Step-wise diagnostics: Spearman/Pearson of preds vs targets at every
        # [VAL] position in the batch. Mirrors ``val/icl_all_*``.
        corr_logs: Dict[str, float] = {}
        rho_ll = loss_dict.get("zero_shot_ll_spearman")
        if rho_ll is not None:
            corr_logs["train/zero_shot_ll_spearman"] = float(rho_ll)
        try:
            preds = loss_dict["preds"]
            predict_mask = batch["predict_mask"]
            target_values = batch["target_values"]
            if predict_mask.sum() > 1:
                ap = preds[predict_mask].detach().float().cpu().numpy()
                at = target_values[predict_mask].detach().float().cpu().numpy()
                if np.std(ap) > 0 and np.std(at) > 0:
                    rho_all, _ = spearmanr(ap, at)
                    r_all, _ = pearsonr(ap, at)
                    if not np.isnan(rho_all):
                        corr_logs["train/icl_all_spearman"] = float(rho_all)
                    if not np.isnan(r_all):
                        corr_logs["train/icl_all_pearson"] = float(r_all)
        except Exception as e:  # noqa: BLE001
            log.warning("ICL train correlation failed: %s", e)
        if corr_logs:
            self.log_dict(
                corr_logs,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=False,
            )
        return loss_dict["loss"]

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        # Standard ProteinGym scoring path is preserved for the zero-shot
        # validation stream that the base class handles.
        if "DMS_scores" in batch:
            return self.validation_step_proteingym(batch)

        outputs, loss_dict = self._shared_step(batch)
        stats = self._diagnostic_stats(batch, loss_dict)
        # ``val/loss`` is the all-values MSE-weighted loss (matches train/loss).
        # ``val/loss_query`` is the same loss but using only the final/query
        # value per row — comparable to the previous behaviour.
        val_loss_query = (
            self.ce_loss_weight * loss_dict["ce_loss"]
            + self.mse_loss_weight * loss_dict["mse_loss_query"]
        )
        self.log_dict(
            {
                "val/loss": loss_dict["loss"],
                "val/loss_query": val_loss_query,
                "val/ce_loss": loss_dict["ce_loss"],
                "val/mse_loss": loss_dict["mse_loss"],
                "val/mse_loss_query": loss_dict["mse_loss_query"],
                **{f"val/{k}": v for k, v in stats.items()},
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Two correlation flavours:
        # 1. ``icl_query_*``  - one prediction per row at the query position
        #    (the last [VAL]). Requires a multi-row batch to be meaningful.
        # 2. ``icl_all_*``    - over *every* predict-mask position in the batch
        #    (multiple values per row). Requires >= 2 predict positions.
        try:
            preds = loss_dict["preds"]
            predict_mask = batch["predict_mask"]
            target_values = batch["target_values"]

            query_mask = loss_dict["query_mask"]
            corr_logs: Dict[str, float] = {}
            if query_mask.sum() > 1:
                qp = preds[query_mask].detach().float().cpu().numpy()
                qt = target_values[query_mask].detach().float().cpu().numpy()
                rho, _ = spearmanr(qp, qt)
                r, _ = pearsonr(qp, qt)
                corr_logs["val/icl_query_spearman"] = float(rho)
                corr_logs["val/icl_query_pearson"] = float(r)
            if predict_mask.sum() > 1:
                ap = preds[predict_mask].detach().float().cpu().numpy()
                at = target_values[predict_mask].detach().float().cpu().numpy()
                rho_all, _ = spearmanr(ap, at)
                r_all, _ = pearsonr(ap, at)
                corr_logs["val/icl_all_spearman"] = float(rho_all)
                corr_logs["val/icl_all_pearson"] = float(r_all)
            rho_ll = loss_dict.get("zero_shot_ll_spearman")
            if rho_ll is not None:
                # Per-doc Spearman of the zero-shot LL feature vs. true fitness.
                # Averaged across docs at on_epoch reduction.
                corr_logs["val/zero_shot_ll_spearman"] = float(rho_ll)
            if corr_logs:
                self.log_dict(
                    corr_logs,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                    sync_dist=True,
                )
        except Exception as e:  # noqa: BLE001
            log.warning("ICL val correlation failed: %s", e)

        return loss_dict["loss"]

    # ------------------------------------------------------------------
    # Optimizer with backbone lr split
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        backbone_params = list(self.model.parameters())
        head_params = (
            list(self.value_in_proj.parameters())
            + list(self.likelihood_in_proj.parameters())
            + list(self.value_out_head.parameters())
        )
        backbone_ids = {id(p) for p in backbone_params}
        head_ids = {id(p) for p in head_params}
        # Sanity check: backbone and head sets are disjoint and cover all parameters.
        all_params = list(self.parameters())
        all_ids = {id(p) for p in all_params}
        assert backbone_ids & head_ids == set(), "Backbone/head parameter overlap"
        unaccounted = all_ids - backbone_ids - head_ids
        assert not unaccounted, f"{len(unaccounted)} parameters missing from optimizer groups"

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr * self.backbone_lr_scale},
                {"params": head_params, "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=self.eps,
        )

        optim_dict: Dict[str, Any] = {"optimizer": optimizer}
        if self.scheduler_name is not None:
            scheduler_kwargs = {}
            if self.scheduler_name == "cosine_with_min_lr":
                scheduler_kwargs["scheduler_specific_kwargs"] = {"min_lr": self.lr * 0.1}
                
            scheduler = get_scheduler(
                self.scheduler_name,
                optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
                **scheduler_kwargs
            )
            optim_dict["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}
        return optim_dict
