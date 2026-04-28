"""Pytest invariants for V3 SpanMasking (Team 11 extension, Section III.V3).

Mirrors the universal + V3-specific assertions in scripts/validate_masks.py
in pytest form so CI / local `pytest tests/` covers them. All tests run on
synthetic random data, no SCITAS, no GPU, no CLEVR mount.

Run:
    cd nano4M && python -m pytest tests/test_span_masking.py -v
"""
import random

import pytest
import torch

from nanofm.data.multimodal.span_masking import SpanMasking


MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"]
VOCAB_SIZES = [64000, 64000, 64000, 50304]
MAX_SEQ_LENS = [256, 256, 256, 256]
INPUT_ALPHAS = [1.0, 1.0, 1.0, 1.0]
TARGET_ALPHAS = [1.0, 1.0, 1.0, 1.0]
INPUT_TOKENS_RANGE = (1, 128)
TARGET_TOKENS_RANGE = (1, 128)
TEXT_MODALITIES = ["scene_desc"]
IMAGE_MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256"]
MEAN_SPAN_LENGTH = 3.0

REPS = 10  # Each test runs the transform this many times.


def make_masking(**overrides):
    kwargs = dict(
        modalities=MODALITIES,
        vocab_sizes=VOCAB_SIZES,
        max_seq_lens=MAX_SEQ_LENS,
        input_alphas=INPUT_ALPHAS,
        target_alphas=TARGET_ALPHAS,
        input_tokens_range=INPUT_TOKENS_RANGE,
        target_tokens_range=TARGET_TOKENS_RANGE,
        overlap_vocab=True,
        overlap_posembs=True,
        include_unmasked_data_dict=False,
        text_modalities=TEXT_MODALITIES,
        image_modalities=IMAGE_MODALITIES,
        mean_span_length=MEAN_SPAN_LENGTH,
    )
    kwargs.update(overrides)
    return SpanMasking(**kwargs)


def make_data_dict(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        mod: torch.randint(0, vsize, (slen,), generator=g, dtype=torch.long)
        for mod, vsize, slen in zip(MODALITIES, VOCAB_SIZES, MAX_SEQ_LENS)
    }


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    torch.manual_seed(0)


def _runs(positions):
    """Group sorted positions into consecutive runs. Returns list of (start, length)."""
    if not positions:
        return []
    positions = sorted(positions)
    runs = []
    run_start = positions[0]
    run_len = 1
    for p in positions[1:]:
        if p == run_start + run_len:
            run_len += 1
        else:
            runs.append((run_start, run_len))
            run_start = p
            run_len = 1
    runs.append((run_start, run_len))
    return runs


# -------------------------- universal invariants --------------------------

def test_output_keys_and_shapes():
    masking = make_masking()
    for i in range(REPS):
        out = masking(make_data_dict(seed=i))
        for k in (
            "enc_tokens", "enc_positions", "enc_modalities", "enc_pad_mask",
            "dec_tokens", "dec_positions", "dec_modalities", "dec_pad_mask",
        ):
            assert k in out
        assert out["enc_tokens"].shape == (INPUT_TOKENS_RANGE[1],)
        assert out["dec_tokens"].shape == (TARGET_TOKENS_RANGE[1],)
        assert out["enc_positions"].shape == (INPUT_TOKENS_RANGE[1],)
        assert out["dec_positions"].shape == (TARGET_TOKENS_RANGE[1],)


def test_dtypes_and_padding_values():
    masking = make_masking()
    for i in range(REPS):
        out = masking(make_data_dict(seed=i))
        for k in ("enc_tokens", "enc_positions", "enc_modalities",
                  "dec_tokens", "dec_positions", "dec_modalities"):
            assert out[k].dtype == torch.long, f"{k} dtype={out[k].dtype}"
        assert out["enc_pad_mask"].dtype == torch.bool
        assert out["dec_pad_mask"].dtype == torch.bool

        dec_pad = ~out["dec_pad_mask"]
        if dec_pad.any():
            assert (out["dec_tokens"][dec_pad] == -100).all(), \
                "dec_tokens at padded positions must be -100"


def test_pad_masks_are_contiguous_prefix():
    masking = make_masking()
    for i in range(REPS):
        out = masking(make_data_dict(seed=i))
        for k in ("enc_pad_mask", "dec_pad_mask"):
            m = out[k].tolist()
            n_true = sum(m)
            assert m[:n_true] == [True] * n_true
            assert m[n_true:] == [False] * (len(m) - n_true)


def test_enc_dec_positions_disjoint_per_modality():
    masking = make_masking()
    for i in range(REPS):
        out = masking(make_data_dict(seed=i))
        for mod_idx in range(len(MODALITIES)):
            enc_sel = (out["enc_modalities"] == mod_idx) & out["enc_pad_mask"]
            dec_sel = (out["dec_modalities"] == mod_idx) & out["dec_pad_mask"]
            enc_pos = set(out["enc_positions"][enc_sel].tolist())
            dec_pos = set(out["dec_positions"][dec_sel].tolist())
            assert not (enc_pos & dec_pos), \
                f"modality_idx={mod_idx} enc/dec overlap: {sorted(enc_pos & dec_pos)[:5]}"


# ----------------------------- V3-specific -----------------------------

def test_scene_desc_targets_form_contiguous_spans():
    """scene_desc decoder positions group into runs of length >= 1."""
    masking = make_masking()
    text_idx = MODALITIES.index("scene_desc")
    saw_multi_token_run = False
    for i in range(REPS):
        out = masking(make_data_dict(seed=i))
        dec_sel = (out["dec_modalities"] == text_idx) & out["dec_pad_mask"]
        dec_pos = out["dec_positions"][dec_sel].tolist()
        if not dec_pos:
            continue
        runs = _runs(dec_pos)
        for rs, rl in runs:
            assert rl >= 1
            assert 0 <= rs and rs + rl <= MAX_SEQ_LENS[text_idx]
        if any(rl >= 2 for _, rl in runs):
            saw_multi_token_run = True
    assert saw_multi_token_run, \
        "expected at least one span of length >= 2 across REPS calls"


def test_image_modality_uses_random_masking():
    """Image decoder positions should not be predominantly contiguous."""
    masking = make_masking()
    img_idx = MODALITIES.index("tok_rgb@256")
    contiguous_runs = 0
    for i in range(REPS):
        out = masking(make_data_dict(seed=i + 100))
        dec_sel = (out["dec_modalities"] == img_idx) & out["dec_pad_mask"]
        dec_pos = sorted(out["dec_positions"][dec_sel].tolist())
        if len(dec_pos) >= 4:
            is_contig = all(dec_pos[j + 1] == dec_pos[j] + 1 for j in range(len(dec_pos) - 1))
            if is_contig:
                contiguous_runs += 1
    assert contiguous_runs == 0, \
        f"image modality looks contiguous in {contiguous_runs}/{REPS} calls"


def test_geometric_mean_span_length_in_band():
    """Empirical mean span length on scene_desc should be near mu=3.

    Trim-to-fit (proposal Section III.V3) biases the empirical mean slightly
    below mu, so we accept a wide band [2.0, 4.0].
    """
    masking = make_masking()
    text_idx = MODALITIES.index("scene_desc")
    all_lengths = []
    for i in range(50):
        out = masking(make_data_dict(seed=i + 200))
        dec_sel = (out["dec_modalities"] == text_idx) & out["dec_pad_mask"]
        dec_pos = out["dec_positions"][dec_sel].tolist()
        all_lengths.extend(rl for _, rl in _runs(dec_pos))
    if not all_lengths:
        pytest.skip("no spans observed across calls")
    empirical_mean = sum(all_lengths) / len(all_lengths)
    assert 2.0 <= empirical_mean <= 4.0, \
        f"empirical mean span length {empirical_mean:.2f} far from mu=3"


def test_scene_desc_target_count_respects_budget():
    """Total scene_desc target tokens must not exceed TARGET_TOKENS_RANGE[1]."""
    masking = make_masking()
    text_idx = MODALITIES.index("scene_desc")
    for i in range(REPS):
        out = masking(make_data_dict(seed=i + 300))
        dec_sel = (out["dec_modalities"] == text_idx) & out["dec_pad_mask"]
        n = int(dec_sel.sum().item())
        assert n <= MAX_SEQ_LENS[text_idx]
        assert n <= TARGET_TOKENS_RANGE[1]


# ------------------------ shifts / passthrough ------------------------

def test_handles_overlap_posembs_false():
    masking = make_masking(overlap_posembs=False)
    for i in range(REPS):
        out = masking(make_data_dict(seed=i + 400))
        for mod_idx in range(len(MODALITIES)):
            shift = mod_idx * 256
            enc_sel = (out["enc_modalities"] == mod_idx) & out["enc_pad_mask"]
            dec_sel = (out["dec_modalities"] == mod_idx) & out["dec_pad_mask"]
            for p in out["enc_positions"][enc_sel].tolist():
                assert shift <= p < shift + 256, \
                    f"modality_idx={mod_idx} (shift={shift}) enc pos {p} out of [{shift},{shift+256})"
            for p in out["dec_positions"][dec_sel].tolist():
                assert shift <= p < shift + 256, \
                    f"modality_idx={mod_idx} (shift={shift}) dec pos {p} out of [{shift},{shift+256})"


def test_include_unmasked_data_dict_passthrough():
    masking = make_masking(include_unmasked_data_dict=True)
    out = masking(make_data_dict(seed=42))
    assert "unmasked_data_dict" in out
    for mod in MODALITIES:
        assert mod in out["unmasked_data_dict"]


def test_init_rejects_modality_dispatch_mismatch():
    with pytest.raises(ValueError, match="image_modalities .* text_modalities"):
        make_masking(
            image_modalities=["tok_rgb@256"],
            text_modalities=["scene_desc"],
        )


def test_init_rejects_invalid_mean_span_length():
    with pytest.raises(ValueError, match="mean_span_length"):
        make_masking(mean_span_length=0.5)


def test_init_rejects_overlapping_dispatch():
    with pytest.raises(ValueError, match="overlap"):
        make_masking(
            image_modalities=["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"],
            text_modalities=["scene_desc"],
        )
