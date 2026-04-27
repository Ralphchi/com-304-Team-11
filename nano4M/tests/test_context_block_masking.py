"""Pytest invariants for V2 ContextBlockMasking (Team 11 extension, Section III.V2).

Mirrors the universal + V2-specific assertions in scripts/validate_masks.py but
in pytest form so CI / local `pytest tests/` covers them. All tests run on
synthetic random data — no SCITAS, no GPU, no CLEVR mount.

Run:
    cd nano4M && python -m pytest tests/test_context_block_masking.py -v
"""
import random

import pytest
import torch

from nanofm.data.multimodal.context_block_masking import ContextBlockMasking


MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"]
VOCAB_SIZES = [64000, 64000, 64000, 50304]
MAX_SEQ_LENS = [256, 256, 256, 256]
INPUT_ALPHAS = [1.0, 1.0, 1.0, 1.0]
TARGET_ALPHAS = [1.0, 1.0, 1.0, 1.0]
INPUT_TOKENS_RANGE = (1, 128)
TARGET_TOKENS_RANGE = (1, 128)
IMAGE_MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256"]
TEXT_MODALITIES = ["scene_desc"]
GRID_SIZE = 16
CONTEXT_BLOCK_SIZES = [4, 5, 6]

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
        image_modalities=IMAGE_MODALITIES,
        text_modalities=TEXT_MODALITIES,
        grid_size=GRID_SIZE,
        context_block_sizes=CONTEXT_BLOCK_SIZES,
    )
    kwargs.update(overrides)
    return ContextBlockMasking(**kwargs)


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


# ----------------------------- V2-specific -----------------------------

def _image_visible_block(out, mod_idx):
    enc_sel = (out["enc_modalities"] == mod_idx) & out["enc_pad_mask"]
    enc_pos = sorted(out["enc_positions"][enc_sel].tolist())
    rows = [p // GRID_SIZE for p in enc_pos]
    cols = [p % GRID_SIZE for p in enc_pos]
    return enc_pos, rows, cols


def test_image_visible_block_is_contiguous_s_by_s():
    masking = make_masking()
    for i in range(REPS):
        out = masking(make_data_dict(seed=i))
        for mod_idx, mod in enumerate(MODALITIES):
            if mod not in IMAGE_MODALITIES:
                continue
            enc_pos, rows, cols = _image_visible_block(out, mod_idx)
            assert enc_pos, f"{mod}: must have at least one visible position"
            s_rows = max(rows) - min(rows) + 1
            s_cols = max(cols) - min(cols) + 1
            assert s_rows == s_cols, f"{mod}: bbox is {s_rows}x{s_cols}, not square"
            s = s_rows
            assert s in CONTEXT_BLOCK_SIZES, f"{mod}: derived s={s} not in {CONTEXT_BLOCK_SIZES}"
            # Regime 1: visible == s^2; Regime 2: visible <= s^2 (subset).
            assert len(enc_pos) <= s * s


def test_image_target_positions_outside_visible_block():
    """Regime 1 (the only regime under baseline budget) — image dec is outside the block."""
    masking = make_masking()
    for i in range(REPS):
        out = masking(make_data_dict(seed=i))
        for mod_idx, mod in enumerate(MODALITIES):
            if mod not in IMAGE_MODALITIES:
                continue
            enc_pos, rows, cols = _image_visible_block(out, mod_idx)
            s = max(rows) - min(rows) + 1
            block_set = {
                r * GRID_SIZE + c
                for r in range(min(rows), min(rows) + s)
                for c in range(min(cols), min(cols) + s)
            }
            dec_sel = (out["dec_modalities"] == mod_idx) & out["dec_pad_mask"]
            dec_pos = set(out["dec_positions"][dec_sel].tolist())
            if len(enc_pos) == s * s:
                assert not (dec_pos & block_set), \
                    f"{mod} regime 1: dec inside block: {sorted(dec_pos & block_set)[:5]}"


def test_same_s_across_image_modalities_in_one_call():
    masking = make_masking()
    for i in range(REPS):
        out = masking(make_data_dict(seed=i))
        s_seen = []
        for mod_idx, mod in enumerate(MODALITIES):
            if mod not in IMAGE_MODALITIES:
                continue
            _, rows, cols = _image_visible_block(out, mod_idx)
            s_seen.append(max(rows) - min(rows) + 1)
        assert len(set(s_seen)) == 1, f"image modalities used different s: {s_seen}"


def test_text_modality_uses_random_masking():
    """scene_desc visible positions should NOT be a contiguous run (random masking)."""
    masking = make_masking()
    text_idx = MODALITIES.index("scene_desc")
    contiguous_runs = 0
    for i in range(REPS):
        out = masking(make_data_dict(seed=i + 100))
        enc_sel = (out["enc_modalities"] == text_idx) & out["enc_pad_mask"]
        enc_pos = sorted(out["enc_positions"][enc_sel].tolist())
        if len(enc_pos) >= 2:
            is_contig = all(enc_pos[j + 1] == enc_pos[j] + 1 for j in range(len(enc_pos) - 1))
            if is_contig:
                contiguous_runs += 1
    # With Dirichlet alphas 1.0 and N>=2 text positions, contiguous runs are
    # vanishingly rare. Allow at most 1 of 10 by chance.
    assert contiguous_runs <= 1, \
        f"text modality looks contiguous in {contiguous_runs}/{REPS} calls — should be random"


# ------------------------ shifts / passthrough ------------------------

def test_handles_overlap_posembs_false():
    masking = make_masking(overlap_posembs=False)
    for i in range(REPS):
        out = masking(make_data_dict(seed=i + 200))
        # max_seq_len_shifts cumulative offset = [0, 256, 512, 768]
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
    data_dict = make_data_dict(seed=42)
    out = masking(data_dict)
    assert "unmasked_data_dict" in out
    for mod in MODALITIES:
        assert mod in out["unmasked_data_dict"]


def test_init_rejects_modality_dispatch_mismatch():
    with pytest.raises(ValueError, match="image_modalities .* text_modalities"):
        make_masking(image_modalities=["tok_rgb@256"], text_modalities=["scene_desc"])


def test_init_rejects_invalid_block_size():
    with pytest.raises(ValueError, match="context_block_size"):
        make_masking(context_block_sizes=[0])
    with pytest.raises(ValueError, match="context_block_size"):
        make_masking(context_block_sizes=[20])
