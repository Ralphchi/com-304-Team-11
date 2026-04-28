# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""V1 block masking on image modalities (proposal Section III.V1)."""
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import random

import torch
import torch.nn.functional as F

from .masking import SimpleMultimodalMasking


class BlockMasking(SimpleMultimodalMasking):
    """Place s x s blocks on a grid_size x grid_size token grid for each
    image modality; fall back to baseline random masking for the rest.

    Args:
        modalities, vocab_sizes, max_seq_lens, input_alphas, target_alphas,
        input_tokens_range, target_tokens_range, overlap_vocab,
        overlap_posembs, include_unmasked_data_dict: forwarded to the parent.
        image_modalities: subset of modalities masked with blocks.
        block_sizes: candidate s; one is drawn per (sample, modality).
        grid_size: side of the 2D token grid (16 for CLEVR @ 256 tokens).
        text_modalities: optional, validated to not overlap image_modalities.
    """

    def __init__(
        self,
        modalities: List[str],
        vocab_sizes: List[int],
        max_seq_lens: List[int],
        input_alphas: List[float],
        target_alphas: List[float],
        input_tokens_range: Union[int, Tuple[int, int]],
        target_tokens_range: Union[int, Tuple[int, int]],
        image_modalities: Sequence[str],
        block_sizes: Sequence[int] = (2, 3, 4),
        grid_size: int = 16,
        text_modalities: Optional[Sequence[str]] = None,
        overlap_vocab: bool = True,
        overlap_posembs: bool = True,
        include_unmasked_data_dict: bool = False,
    ) -> None:
        super().__init__(
            modalities=modalities,
            vocab_sizes=vocab_sizes,
            max_seq_lens=max_seq_lens,
            input_alphas=input_alphas,
            target_alphas=target_alphas,
            input_tokens_range=input_tokens_range,
            target_tokens_range=target_tokens_range,
            overlap_vocab=overlap_vocab,
            overlap_posembs=overlap_posembs,
            include_unmasked_data_dict=include_unmasked_data_dict,
        )

        self.image_modalities: Set[str] = set(image_modalities)
        self.block_sizes: Tuple[int, ...] = tuple(block_sizes)
        self.grid_size: int = int(grid_size)

        unknown = self.image_modalities - set(modalities)
        if unknown:
            raise ValueError(
                f"image_modalities not in modalities: {sorted(unknown)}"
            )
        if not self.block_sizes:
            raise ValueError("block_sizes must be non-empty.")
        for s in self.block_sizes:
            if not (1 <= s <= self.grid_size):
                raise ValueError(
                    f"Block size {s} out of [1, {self.grid_size}]."
                )

        expected_seq_len = self.grid_size * self.grid_size
        for mod, max_len in zip(modalities, max_seq_lens):
            if mod in self.image_modalities and max_len != expected_seq_len:
                raise ValueError(
                    f"Image modality '{mod}' has max_seq_len={max_len}, "
                    f"expected {expected_seq_len} for grid_size={self.grid_size}."
                )

        if text_modalities is not None:
            overlap = set(text_modalities) & self.image_modalities
            if overlap:
                raise ValueError(
                    f"Modalities listed as both image and text: {sorted(overlap)}"
                )

    def _sample_block_positions(self, n_select: int) -> torch.Tensor:
        if n_select <= 0:
            return torch.empty(0, dtype=torch.long)

        max_positions = self.grid_size * self.grid_size
        if n_select > max_positions:
            raise ValueError(
                f"n_select={n_select} exceeds grid capacity {max_positions}."
            )

        s = random.choice(self.block_sizes)
        max_top = self.grid_size - s
        selected: Set[int] = set()

        while len(selected) < n_select:
            top = random.randint(0, max_top)
            left = random.randint(0, max_top)
            block = [
                (top + i) * self.grid_size + (left + j)
                for i in range(s)
                for j in range(s)
            ]
            new = [p for p in block if p not in selected]
            remaining = n_select - len(selected)
            if len(new) > remaining:
                # Trim the final block so the realised count equals n_select.
                random.shuffle(new)
                new = new[:remaining]
            selected.update(new)

        return torch.tensor(sorted(selected), dtype=torch.long)

    def perform_random_masking(
        self,
        data_dict: Dict[str, Any],
        input_token_budget: List[int],
        target_token_budget: List[int],
    ) -> Dict[str, Any]:
        enc_tokens, enc_positions, enc_modalities = [], [], []
        dec_tokens, dec_positions, dec_modalities = [], [], []

        for mod_idx, mod in enumerate(self.modalities):
            num_tokens = data_dict[mod].shape[0]
            n_input = input_token_budget[mod_idx]
            n_target = target_token_budget[mod_idx]
            n_select = n_input + n_target

            if mod in self.image_modalities:
                if num_tokens != self.grid_size * self.grid_size:
                    # Truncated sample: fall back to random selection.
                    selected = torch.randperm(num_tokens)[:n_select].sort()[0]
                else:
                    selected = self._sample_block_positions(n_select)
                if n_select > 0:
                    perm = torch.randperm(n_select)
                    input_pos = selected[perm[:n_input]].sort()[0]
                    target_pos = selected[perm[n_input:n_input + n_target]].sort()[0]
                else:
                    input_pos = torch.empty(0, dtype=torch.long)
                    target_pos = torch.empty(0, dtype=torch.long)
            else:
                noise = torch.rand(num_tokens)
                ids_shuffle = torch.argsort(noise, dim=0)
                input_pos = ids_shuffle[:n_input].sort()[0]
                target_pos = ids_shuffle[n_input:n_input + n_target].sort()[0]

            pos_shift = 0 if self.overlap_posembs else self.max_seq_len_shifts[mod_idx]
            enc_positions.append(input_pos + pos_shift)
            dec_positions.append(target_pos + pos_shift)

            enc_tokens.append(data_dict[mod][input_pos])
            dec_tokens.append(data_dict[mod][target_pos])

            enc_modalities.append(
                mod_idx * torch.ones(input_pos.shape[0], dtype=torch.long)
            )
            dec_modalities.append(
                mod_idx * torch.ones(target_pos.shape[0], dtype=torch.long)
            )

        enc_tokens = torch.cat(enc_tokens) if enc_tokens else torch.empty(0, dtype=torch.long)
        dec_tokens = torch.cat(dec_tokens) if dec_tokens else torch.empty(0, dtype=torch.long)
        enc_positions = torch.cat(enc_positions) if enc_positions else torch.empty(0, dtype=torch.long)
        dec_positions = torch.cat(dec_positions) if dec_positions else torch.empty(0, dtype=torch.long)
        enc_modalities = torch.cat(enc_modalities) if enc_modalities else torch.empty(0, dtype=torch.long)
        dec_modalities = torch.cat(dec_modalities) if dec_modalities else torch.empty(0, dtype=torch.long)

        max_input = self.input_tokens_range[1]
        max_target = self.target_tokens_range[1]
        enc_pad = max_input - enc_tokens.shape[0]
        dec_pad = max_target - dec_tokens.shape[0]

        enc_tokens = F.pad(enc_tokens, (0, enc_pad), value=0)
        enc_positions = F.pad(enc_positions, (0, enc_pad), value=0)
        enc_modalities = F.pad(enc_modalities, (0, enc_pad), value=0)
        dec_tokens = F.pad(dec_tokens, (0, dec_pad), value=-100)
        dec_positions = F.pad(dec_positions, (0, dec_pad), value=0)
        dec_modalities = F.pad(dec_modalities, (0, dec_pad), value=0)

        enc_pad_mask = torch.ones(max_input, dtype=torch.bool)
        if enc_pad > 0:
            enc_pad_mask[-enc_pad:] = False
        dec_pad_mask = torch.ones(max_target, dtype=torch.bool)
        if dec_pad > 0:
            dec_pad_mask[-dec_pad:] = False

        return {
            "enc_tokens": enc_tokens,
            "enc_positions": enc_positions,
            "enc_modalities": enc_modalities,
            "enc_pad_mask": enc_pad_mask,
            "dec_tokens": dec_tokens,
            "dec_positions": dec_positions,
            "dec_modalities": dec_modalities,
            "dec_pad_mask": dec_pad_mask,
        }
