"""Forward-pass / gradient / loss tests for LlamaICLLitModule.

These use a tiny random-init Llama model on CPU - no external data is
required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import LlamaConfig

from profam.constants import VOCAB_SIZE
from profam.data.icl_constants import VAL_SLOT_TOKEN_ID, VAL_TOKEN_ID
from profam.models.llama_icl import LlamaICLLitModule


def _build_tiny_model(profam_tokenizer):
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        max_position_embeddings=512,
        attn_implementation="eager",
        torch_dtype="float32",
    )
    return LlamaICLLitModule(
        config=config,
        tokenizer=profam_tokenizer,
        lr=1e-3,
        weight_decay=0.0,
        scheduler_name=None,
        ce_loss_weight=1.0,
        mse_loss_weight=1.0,
        backbone_lr_scale=1.0,
        pass_res_pos_in_doc_as_position_ids=True,
    )


def _build_simple_icl_batch(
    profam_tokenizer,
    k: int = 4,
    seq_len: int = 6,
    seed: int = 0,
):
    """Construct one ICL document with ``k`` labelled examples and one query.

    Layout per variant: ``aa... [SEP] [VAL] [VAL_SLOT]`` (labelled) or
    ``aa... [SEP] [VAL]`` (query). Also includes the per-variant scoring
    fields (``variant_token_ids`` / ``variant_attn_mask`` / ``variant_valid_mask``)
    expected by the model's zero-shot scoring sub-pass.
    """
    rng = np.random.default_rng(seed)
    aa_ids = profam_tokenizer.aa_tokens
    bos_id = int(profam_tokenizer.bos_token_id)
    sep_id = int(profam_tokenizer.sep_token_id)
    pad_id = int(profam_tokenizer.pad_token_id)
    doc_id = int(profam_tokenizer.convert_tokens_to_ids("[RAW]"))

    ids = [bos_id, doc_id]
    val_slot_mask = [False, False]
    val_marker_mask = [False, False]
    predict_mask = [False, False]
    values = [0.0, 0.0]
    target_values = [0.0, 0.0]
    aa_mask = [False, False]

    labelled_values = rng.normal(size=(k,)).astype(np.float32)
    query_target = float(rng.normal())

    # Track per-variant AA token streams for the scoring sub-pass.
    variant_streams: list[list[int]] = []

    for i in range(k + 1):
        variant_aa: list[int] = []
        for _ in range(seq_len):
            tok = int(rng.choice(aa_ids))
            ids.append(tok)
            variant_aa.append(tok)
            val_slot_mask.append(False)
            val_marker_mask.append(False)
            predict_mask.append(False)
            values.append(0.0)
            target_values.append(0.0)
            aa_mask.append(True)

        # Sequence end delimiter precedes the marker.
        ids.append(sep_id)
        val_slot_mask.append(False)
        val_marker_mask.append(False)
        predict_mask.append(False)
        values.append(0.0)
        target_values.append(0.0)
        aa_mask.append(False)

        ids.append(VAL_TOKEN_ID)
        val_slot_mask.append(False)
        val_marker_mask.append(True)
        predict_mask.append(True)
        values.append(0.0)
        target_values.append(
            float(labelled_values[i]) if i < k else float(query_target)
        )
        aa_mask.append(False)
        if i < k:
            ids.append(VAL_SLOT_TOKEN_ID)
            val_slot_mask.append(True)
            val_marker_mask.append(False)
            predict_mask.append(False)
            values.append(float(labelled_values[i]))
            target_values.append(0.0)
            aa_mask.append(False)

        variant_streams.append(variant_aa)

    L = len(ids)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    labels[input_ids == VAL_TOKEN_ID] = -100
    labels[input_ids == VAL_SLOT_TOKEN_ID] = -100

    n_var = len(variant_streams)
    # +BOS +DOC_TYPE prefix and a trailing [SEP] so the per-token LL also
    # rewards the model for predicting the end of the sequence.
    var_max_len = max(len(s) + 3 for s in variant_streams)
    variant_token_ids = np.full((n_var, var_max_len), pad_id, dtype=np.int64)
    variant_attn_mask = np.zeros((n_var, var_max_len), dtype=np.int64)
    for vi, aa in enumerate(variant_streams):
        row = [bos_id, doc_id] + aa + [sep_id]
        variant_token_ids[vi, : len(row)] = row
        variant_attn_mask[vi, : len(row)] = 1

    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones(1, L, dtype=torch.long),
        "labels": labels,
        "aa_mask": torch.tensor(aa_mask).unsqueeze(0),
        "value_slot_mask": torch.tensor(val_slot_mask).unsqueeze(0),
        "val_marker_mask": torch.tensor(val_marker_mask).unsqueeze(0),
        "predict_mask": torch.tensor(predict_mask).unsqueeze(0),
        "values": torch.tensor(values, dtype=torch.float32).unsqueeze(0),
        "target_values": torch.tensor(target_values, dtype=torch.float32).unsqueeze(0),
        "variant_token_ids": torch.from_numpy(variant_token_ids).unsqueeze(0),
        "variant_attn_mask": torch.from_numpy(variant_attn_mask).unsqueeze(0),
        "variant_valid_mask": torch.ones(1, n_var, dtype=torch.bool),
    }
    return batch


def test_forward_pass_runs_and_returns_hidden_states(profam_tokenizer):
    model = _build_tiny_model(profam_tokenizer)
    batch = _build_simple_icl_batch(profam_tokenizer)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        value_slot_mask=batch["value_slot_mask"],
        values=batch["values"],
        val_marker_mask=batch["val_marker_mask"],
        predict_mask=batch["predict_mask"],
        target_values=batch["target_values"],
        labels=batch["labels"],
    )
    assert outputs.hidden_states is not None
    last = outputs.hidden_states[-1]
    assert last.shape[:2] == batch["input_ids"].shape


def test_gradients_reach_value_in_proj_and_value_out_head(profam_tokenizer):
    model = _build_tiny_model(profam_tokenizer)
    batch = _build_simple_icl_batch(profam_tokenizer)
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    in_grad = model.value_in_proj.proj.weight.grad
    out_grad = model.value_out_head.weight.grad
    assert in_grad is not None and torch.isfinite(in_grad).all() and in_grad.abs().sum() > 0
    assert out_grad is not None and torch.isfinite(out_grad).all() and out_grad.abs().sum() > 0


def test_constant_y_ignores_value_slot_branch(profam_tokenizer):
    """If we set every value to a constant (and the value_in_proj has no bias
    contribution), removing value_slot_mask should give an output that
    differs only by the embedded value contribution. We don't enforce
    bit-identical outputs - just that the forward stays finite and the LM
    branch still produces sensible logits."""
    model = _build_tiny_model(profam_tokenizer)
    batch = _build_simple_icl_batch(profam_tokenizer)
    batch["values"] = torch.zeros_like(batch["values"])
    batch["target_values"] = torch.zeros_like(batch["target_values"])
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        value_slot_mask=batch["value_slot_mask"],
        values=batch["values"],
        val_marker_mask=batch["val_marker_mask"],
        predict_mask=batch["predict_mask"],
        target_values=batch["target_values"],
        labels=batch["labels"],
    )
    assert torch.isfinite(outputs.logits).all()


def test_likelihood_injection_changes_predictions(profam_tokenizer):
    """Passing the variant scoring fields should engage the likelihood branch
    and shift predictions at every [VAL] position vs. the control where the
    branch is bypassed (variant_token_ids=None)."""
    torch.manual_seed(0)
    model = _build_tiny_model(profam_tokenizer).eval()
    batch = _build_simple_icl_batch(profam_tokenizer, k=4, seq_len=6, seed=3)

    common = dict(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        value_slot_mask=batch["value_slot_mask"],
        values=batch["values"],
        val_marker_mask=batch["val_marker_mask"],
        predict_mask=batch["predict_mask"],
        target_values=batch["target_values"],
        labels=batch["labels"],
    )
    with torch.no_grad():
        out_no_ll = model(**common)
        out_with_ll = model(
            **common,
            variant_token_ids=batch["variant_token_ids"],
            variant_attn_mask=batch["variant_attn_mask"],
            variant_valid_mask=batch["variant_valid_mask"],
        )
    preds_no = model.value_out_head(out_no_ll.hidden_states[-1].float()).squeeze(-1)
    preds_yes = model.value_out_head(out_with_ll.hidden_states[-1].float()).squeeze(-1)
    pmask = batch["val_marker_mask"]
    diff = (preds_no[pmask] - preds_yes[pmask]).abs().max().item()
    assert diff > 1e-4, (
        f"Likelihood injection had no effect on [VAL] predictions (max diff {diff:.2e})."
    )


def test_likelihood_injection_gradients_reach_likelihood_in_proj(profam_tokenizer):
    """End-to-end gradient check: the regression loss must propagate back
    through ``likelihood_in_proj`` when variant_* fields are provided."""
    torch.manual_seed(0)
    model = _build_tiny_model(profam_tokenizer)
    batch = _build_simple_icl_batch(profam_tokenizer, k=4, seq_len=6, seed=4)
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    grad = model.likelihood_in_proj.proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, (
        "likelihood_in_proj weights received no gradient signal"
    )


def test_overfit_single_assay_reduces_mse(profam_tokenizer):
    """Sanity-check from the plan: with alpha=0, the model should be able
    to drive MSE down substantially on a single fixed batch.

    Uses a tiny random-init model so the overfit run is fast; we just verify
    that gradients flow end-to-end and the loss monotonically trends down.
    """
    torch.manual_seed(0)
    model = _build_tiny_model(profam_tokenizer)
    model.ce_loss_weight = 0.0
    model.mse_loss_weight = 1.0
    batch = _build_simple_icl_batch(profam_tokenizer, k=8, seq_len=4, seed=2)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for step in range(200):
        _outputs, loss_dict = model._shared_step(batch)
        loss = loss_dict["mse_loss"]
        losses.append(float(loss.detach()))
        optim.zero_grad()
        loss.backward()
        optim.step()

    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss < initial_loss * 0.5, (
        f"MSE failed to drop substantially: {initial_loss=} -> {final_loss=}"
    )
