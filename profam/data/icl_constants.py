"""Token-id constants for the ICL fine-tune.

The shipped tokenizer reserves ``[SP1]``..``[SP10]`` (ids 52..61) as unused special tokens.
For the in-context-learning supervised fine-tune we repurpose the first two:

* ``[VAL]``      -> ``[SP1]`` (id 52): dual-role token. Its **input embedding** is
  replaced at runtime by ``W_likelihood @ phi(ll_z)`` where ``ll_z`` is the
  per-context z-scored zero-shot mean per-token log-likelihood of the variant
  sequence under the current model. Its **output hidden state** is then read by
  the regression head to predict the fitness z-score.
* ``[VAL_SLOT]`` -> ``[SP2]`` (id 53): placeholder token whose input embedding is
  replaced at runtime by ``W_in @ phi(y)`` for the corresponding labelled example.
  Only present after labelled examples; never carries a prediction.

Document layout used by :class:`profam.data.builders.proteingym_icl.ProteinGymICLDataset`:

    [BOS] [DOC_TYPE] x_1 [SEP] [VAL] [VAL_SLOT] x_2 [SEP] [VAL] [VAL_SLOT] ... x_q [SEP] [VAL]

The query trails with ``[SEP] [VAL]`` (no ``[VAL_SLOT]``). ``[SEP]`` always
precedes ``[VAL]`` so the model has a stable signal that "the sequence ended,
the next token's input embedding carries the likelihood, and a fitness
prediction is expected from this position."
"""

VAL_TOKEN = "[SP1]"
VAL_SLOT_TOKEN = "[SP2]"

VAL_TOKEN_ID = 52
VAL_SLOT_TOKEN_ID = 53

ICL_AUX_TENSOR_KEYS = (
    "value_slot_mask",
    "values",
    "val_marker_mask",
    "predict_mask",
    "target_values",
    "variant_token_ids",
    "variant_attn_mask",
    "variant_valid_mask",
)
