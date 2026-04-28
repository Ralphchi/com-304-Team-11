"""Tests for V1 BlockMasking (proposal Section III.V1)."""
from __future__ import annotations

import random
from typing import Dict, List

import pytest
import torch

from nanofm.data.multimodal.block_masking import BlockMasking
from nanofm.data.multimodal.masking import SimpleMultimodalMasking


MODALITIES = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"]
VOCAB_SIZES = [64000, 64000, 64000, 50304]
MAX_SEQ_LENS = [256, 256, 256, 256]
IMAGE_MODS = MODALITIES[:3]
TEXT_MODS = MODALITIES[3:]
GRID_SIZE = 16


def _make_data_dict() -> Dict[str, torch.Tensor]:
    return {
        m: torch.randint(0, V, (S,), dtype=torch.long)
        for m, V, S in zip(MODALITIES, VOCAB_SIZES, MAX_SEQ_LENS)
    }


def _make_masking(
    block_sizes=(2, 3, 4),
    input_alphas=(1.0, 1.0, 1.0, 1.0),
    target_alphas=(1.0, 1.0, 1.0, 1.0),
    input_tokens_range=(1, 128),
    target_tokens_range=(1, 128),
) -> BlockMasking:
    return BlockMasking(
        modalities=MODALITIES,
        vocab_sizes=VOCAB_SIZES,
        max_seq_lens=MAX_SEQ_LENS,
        input_alphas=list(input_alphas),
        target_alphas=list(target_alphas),
        input_tokens_range=input_tokens_range,
        target_tokens_range=target_tokens_range,
        image_modalities=IMAGE_MODS,
        text_modalities=TEXT_MODS,
        block_sizes=block_sizes,
        grid_size=GRID_SIZE,
    )


def _split_by_modality(
    indices: torch.Tensor,
    modality_ids: torch.Tensor,
    pad_mask: torch.Tensor,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {i: [] for i in range(len(MODALITIES))}
    for pos, mod, valid in zip(indices.tolist(), modality_ids.tolist(), pad_mask.tolist()):
        if valid:
            out[int(mod)].append(int(pos))
    return out


def _coverable_by_s_blocks(positions: List[int], grid_size: int, s: int) -> bool:
    """Every position lies inside at least one valid in-grid s x s window."""
    if not positions:
        return True
    for p in positions:
        r, c = p // grid_size, p % grid_size
        top_lo, top_hi = max(0, r - s + 1), min(grid_size - s, r)
        left_lo, left_hi = max(0, c - s + 1), min(grid_size - s, c)
        if top_lo > top_hi or left_lo > left_hi:
            return False
    return True


def test_constructor_validates_grid_size():
    with pytest.raises(ValueError):
        BlockMasking(
            modalities=["tok_rgb@128"],
            vocab_sizes=[64000],
            max_seq_lens=[128],
            input_alphas=[1.0],
            target_alphas=[1.0],
            input_tokens_range=(1, 64),
            target_tokens_range=(1, 64),
            image_modalities=["tok_rgb@128"],
            grid_size=16,
        )


def test_constructor_rejects_unknown_image_modality():
    with pytest.raises(ValueError):
        BlockMasking(
            modalities=MODALITIES,
            vocab_sizes=VOCAB_SIZES,
            max_seq_lens=MAX_SEQ_LENS,
            input_alphas=[1.0] * 4,
            target_alphas=[1.0] * 4,
            input_tokens_range=(1, 128),
            target_tokens_range=(1, 128),
            image_modalities=["does_not_exist"],
        )


def test_constructor_rejects_oversized_block():
    with pytest.raises(ValueError):
        BlockMasking(
            modalities=MODALITIES,
            vocab_sizes=VOCAB_SIZES,
            max_seq_lens=MAX_SEQ_LENS,
            input_alphas=[1.0] * 4,
            target_alphas=[1.0] * 4,
            input_tokens_range=(1, 128),
            target_tokens_range=(1, 128),
            image_modalities=IMAGE_MODS,
            block_sizes=[20],
            grid_size=16,
        )


def test_output_shapes_match_simple():
    random.seed(0)
    torch.manual_seed(0)
    simple = SimpleMultimodalMasking(
        modalities=MODALITIES,
        vocab_sizes=VOCAB_SIZES,
        max_seq_lens=MAX_SEQ_LENS,
        input_alphas=[1.0] * 4,
        target_alphas=[1.0] * 4,
        input_tokens_range=(1, 128),
        target_tokens_range=(1, 128),
    )
    block = _make_masking()
    data = _make_data_dict()
    out_simple = simple(dict(data))
    out_block = block(dict(data))
    for k, v in out_simple.items():
        assert k in out_block
        assert out_block[k].shape == v.shape
        assert out_block[k].dtype == v.dtype


def test_per_modality_counts_within_budget():
    random.seed(1)
    torch.manual_seed(1)
    out = _make_masking()(dict(_make_data_dict()))
    assert 1 <= int(out["enc_pad_mask"].sum().item()) <= 128
    assert 1 <= int(out["dec_pad_mask"].sum().item()) <= 128


def test_block_positions_are_within_an_sxs_window_for_each_image_modality():
    random.seed(42)
    torch.manual_seed(42)
    block = _make_masking()
    data = _make_data_dict()

    for trial in range(20):
        out = block(dict(data))
        enc_by_mod = _split_by_modality(
            out["enc_positions"], out["enc_modalities"], out["enc_pad_mask"]
        )
        dec_by_mod = _split_by_modality(
            out["dec_positions"], out["dec_modalities"], out["dec_pad_mask"]
        )
        for mod_idx, mod in enumerate(MODALITIES):
            if mod not in IMAGE_MODS:
                continue
            union = sorted(set(enc_by_mod[mod_idx]) | set(dec_by_mod[mod_idx]))
            assert len(union) == len(enc_by_mod[mod_idx]) + len(dec_by_mod[mod_idx])
            assert all(0 <= p < GRID_SIZE * GRID_SIZE for p in union)
            assert any(
                _coverable_by_s_blocks(union, GRID_SIZE, s)
                for s in block.block_sizes
            ), f"trial={trial} mod={mod} positions={union}"


def test_scene_desc_uses_random_masking():
    random.seed(123)
    torch.manual_seed(123)
    block = _make_masking()
    data = _make_data_dict()

    runs: List[int] = []
    for _ in range(50):
        out = block(dict(data))
        text_idx = MODALITIES.index("scene_desc")
        enc_by_mod = _split_by_modality(
            out["enc_positions"], out["enc_modalities"], out["enc_pad_mask"]
        )
        dec_by_mod = _split_by_modality(
            out["dec_positions"], out["dec_modalities"], out["dec_pad_mask"]
        )
        positions = sorted(set(enc_by_mod[text_idx]) | set(dec_by_mod[text_idx]))
        if not positions:
            continue
        longest = cur = 1
        for a, b in zip(positions[:-1], positions[1:]):
            cur = cur + 1 if b == a + 1 else 1
            longest = max(longest, cur)
        runs.append(longest)

    if runs:
        avg_longest = sum(runs) / len(runs)
        assert avg_longest < 30, f"scene_desc looks clustered (avg run = {avg_longest:.1f})"


def test_determinism_with_seed():
    block = _make_masking()
    data = _make_data_dict()

    random.seed(2024)
    torch.manual_seed(2024)
    out1 = block(dict(data))

    random.seed(2024)
    torch.manual_seed(2024)
    out2 = block(dict(data))

    for k in out1:
        if k == "unmasked_data_dict":
            continue
        assert torch.equal(out1[k], out2[k]), f"non-deterministic field: {k}"
