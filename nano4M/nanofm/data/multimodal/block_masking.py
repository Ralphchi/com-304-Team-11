# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""V1 (Nacer): 2D block masking for image-like modalities of nano4M.

This is the Week-1 deliverable for the "Alternate Masking Strategies" extension
(Team 11, COM-304). It implements ``BlockMasking``, a drop-in replacement for
``SimpleMultimodalMasking`` that selects per-modality token positions by
placing contiguous s x s blocks on the 16 x 16 token grid of each image-like
modality (RGB, depth, normals). Non-image modalities (e.g. ``scene_desc``)
keep the baseline uniform-random behaviour, so V1 differs from the V0 baseline
by exactly one change, as required by the proposal (Section III).

Algorithm
---------
For each image-like modality, given a per-modality token budget
``N = n_input + n_target`` sampled by the inherited Dirichlet machinery:

    1. Draw a block size ``s`` uniformly from ``block_sizes`` (default
       ``{2, 3, 4}``). Drawn once per (sample, modality), matching the
       proposal's "Block size s in {2, 3, 4} is sampled uniformly".
    2. Repeatedly place an ``s x s`` block at a uniformly random valid
       grid position (top-left in ``[0, G - s] x [0, G - s]`` so the block
       stays inside the ``G x G`` grid, default ``G = 16``).
    3. Track the union of covered flat positions. Stop as soon as the union
       is at least ``N``. If the last block overshoots, randomly drop a
       subset of its newly-added positions to bring the count to exactly
       ``N`` ("excess from the final block is trimmed to match N exactly,
       eliminating the mask-count confound of naive implementations").
    4. Randomly assign ``n_input`` of the ``N`` selected positions as the
       encoder ("input"/visible context) tokens, the remaining ``n_target``
       as the decoder ("target"/predict) tokens. The unselected positions
       are dropped, exactly as in ``SimpleMultimodalMasking``.

Non-image modalities take the parent class' uniform-random selection.

Why this preserves a "single change"
------------------------------------
The total per-modality token budget, the encoder/decoder split, the
unified-vocab handling and the padding/attention-mask plumbing are all
inherited unchanged from ``SimpleMultimodalMasking``. Only the *positions*
are different: contiguous patches instead of scattered tokens.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import random

import torch
import torch.nn.functional as F

from .masking import SimpleMultimodalMasking


class BlockMasking(SimpleMultimodalMasking):
    """2D block masking on image-like modalities, random masking elsewhere.

    Args:
        modalities: List of modality names (in the same order as in the
            baseline config). Position 0 is taken as the canonical reference.
        vocab_sizes: Vocabulary size per modality.
        max_seq_lens: Maximum sequence length per modality. For each modality
            in ``image_modalities``, this MUST equal ``grid_size ** 2`` since
            the 1D token sequence is treated as a flattened ``grid_size`` x
            ``grid_size`` grid.
        input_alphas, target_alphas: Dirichlet alphas for input / target
            per-modality budgets (forwarded to the parent class).
        input_tokens_range, target_tokens_range: Total token-budget ranges
            (forwarded to the parent class).
        image_modalities: Subset of ``modalities`` that should receive block
            masking. Anything not in this set falls back to random masking.
        block_sizes: Candidate block sizes ``s``. One ``s`` is drawn uniformly
            from this list per (sample, modality) at ``__call__`` time. For
            the V1 ablation, set to a singleton, e.g. ``[3]``. Default is
            ``[2, 3, 4]`` (proposal Section III).
        grid_size: Side length of the 2D token grid. Default 16 (CLEVR @ 256
            tokens). Each ``s`` in ``block_sizes`` must satisfy
            ``1 <= s <= grid_size``.
        text_modalities: Optional, informational. Modalities listed here are
            asserted not to be in ``image_modalities``. Anything else (i.e.
            modalities present in ``modalities`` but not in
            ``image_modalities``) is automatically random-masked.
        overlap_vocab, overlap_posembs, include_unmasked_data_dict: Forwarded
            to the parent class unchanged.
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
                f"`image_modalities` contains entries not present in `modalities`: "
                f"{sorted(unknown)}"
            )
        if not self.block_sizes:
            raise ValueError("`block_sizes` must contain at least one block size.")
        for s in self.block_sizes:
            if not (1 <= s <= self.grid_size):
                raise ValueError(
                    f"Block size {s} is out of range [1, grid_size={self.grid_size}]."
                )

        # Validate that every image-like modality has a square-grid sequence length.
        expected_seq_len = self.grid_size * self.grid_size
        for mod, max_len in zip(modalities, max_seq_lens):
            if mod in self.image_modalities and max_len != expected_seq_len:
                raise ValueError(
                    f"Image modality '{mod}' has max_seq_len={max_len} but "
                    f"grid_size={self.grid_size} expects {expected_seq_len}. "
                    "Either change grid_size or remove this modality from "
                    "`image_modalities`."
                )

        if text_modalities is not None:
            text_set = set(text_modalities)
            overlap = text_set & self.image_modalities
            if overlap:
                raise ValueError(
                    f"Modalities listed both as image and text: {sorted(overlap)}"
                )

    # -------------------------------------------------------------- helpers

    def _sample_block_positions(self, n_select: int) -> torch.Tensor:
        """Return a 1d tensor of ``n_select`` unique flat positions chosen by
        placing s x s blocks on the ``grid_size x grid_size`` grid.

        The returned tensor is sorted in ascending order so its elements can
        be indexed directly into the modality's flat token sequence.
        """
        if n_select <= 0:
            return torch.empty(0, dtype=torch.long)

        max_positions = self.grid_size * self.grid_size
        if n_select > max_positions:
            raise ValueError(
                f"n_select={n_select} exceeds grid capacity "
                f"grid_size**2={max_positions}."
            )

        s = random.choice(self.block_sizes)
        max_top = self.grid_size - s
        selected: Set[int] = set()

        while len(selected) < n_select:
            top = random.randint(0, max_top)
            left = random.randint(0, max_top)
            block_positions: List[int] = [
                (top + i) * self.grid_size + (left + j)
                for i in range(s)
                for j in range(s)
            ]
            new_positions = [p for p in block_positions if p not in selected]
            remaining = n_select - len(selected)
            if len(new_positions) > remaining:
                # Trim the final block: randomly drop excess so the realised
                # count matches `n_select` exactly.
                random.shuffle(new_positions)
                new_positions = new_positions[:remaining]
            selected.update(new_positions)

        return torch.tensor(sorted(selected), dtype=torch.long)

    # -------------------------------------------------- masking entry point

    def perform_random_masking(
        self,
        data_dict: Dict[str, Any],
        input_token_budget: List[int],
        target_token_budget: List[int],
    ) -> Dict[str, Any]:
        """Block masking on image modalities, random masking on others.

        The structure mirrors ``SimpleMultimodalMasking.perform_random_masking``
        so that the encoder/decoder padding, attention masks and modality
        indices are produced identically. Only the per-modality position
        sampling differs for image-like modalities.
        """
        enc_tokens, enc_positions, enc_modalities = [], [], []
        dec_tokens, dec_positions, dec_modalities = [], [], []

        for mod_idx, mod in enumerate(self.modalities):
            num_tokens = data_dict[mod].shape[0]
            n_input_tokens = input_token_budget[mod_idx]
            n_target_tokens = target_token_budget[mod_idx]
            n_select = n_input_tokens + n_target_tokens

            if mod in self.image_modalities:
                # Make sure block sampling is consistent with the actual
                # length of this modality's sequence. We assert in __init__
                # that ``max_seq_lens[mod] == grid_size**2``; if a sample
                # ever has a different ``num_tokens`` (e.g. truncated), fall
                # back to random sampling.
                if num_tokens != self.grid_size * self.grid_size:
                    selected = torch.randperm(num_tokens)[:n_select].sort()[0]
                else:
                    selected = self._sample_block_positions(n_select)
                # Random encoder/decoder split inside the block-selected set.
                # Trick: shuffle the indices of `selected`, then take the
                # first n_input as input and the next n_target as target.
                if n_select > 0:
                    perm = torch.randperm(n_select)
                    input_idx = perm[:n_input_tokens]
                    target_idx = perm[n_input_tokens:n_input_tokens + n_target_tokens]
                    input_pos = selected[input_idx].sort()[0]
                    target_pos = selected[target_idx].sort()[0]
                else:
                    input_pos = torch.empty(0, dtype=torch.long)
                    target_pos = torch.empty(0, dtype=torch.long)
            else:
                # Baseline random masking for non-image modalities (scene_desc).
                noise = torch.rand(num_tokens)
                ids_shuffle = torch.argsort(noise, dim=0)
                input_pos = ids_shuffle[:n_input_tokens].sort()[0]
                target_pos = ids_shuffle[
                    n_input_tokens:n_input_tokens + n_target_tokens
                ].sort()[0]

            pos_idx_shift = (
                0 if self.overlap_posembs else self.max_seq_len_shifts[mod_idx]
            )
            enc_positions.append(input_pos + pos_idx_shift)
            dec_positions.append(target_pos + pos_idx_shift)

            input_tokens, target_tokens = (
                data_dict[mod][input_pos],
                data_dict[mod][target_pos],
            )
            enc_tokens.append(input_tokens)
            dec_tokens.append(target_tokens)

            n_input_actual = input_pos.shape[0]
            n_target_actual = target_pos.shape[0]
            enc_modalities.append(
                mod_idx * torch.ones(n_input_actual, dtype=torch.long)
            )
            dec_modalities.append(
                mod_idx * torch.ones(n_target_actual, dtype=torch.long)
            )

        enc_tokens = torch.cat(enc_tokens) if enc_tokens else torch.empty(0, dtype=torch.long)
        dec_tokens = torch.cat(dec_tokens) if dec_tokens else torch.empty(0, dtype=torch.long)
        enc_positions = torch.cat(enc_positions) if enc_positions else torch.empty(0, dtype=torch.long)
        dec_positions = torch.cat(dec_positions) if dec_positions else torch.empty(0, dtype=torch.long)
        enc_modalities = torch.cat(enc_modalities) if enc_modalities else torch.empty(0, dtype=torch.long)
        dec_modalities = torch.cat(dec_modalities) if dec_modalities else torch.empty(0, dtype=torch.long)

        max_input_tokens = self.input_tokens_range[1]
        max_target_tokens = self.target_tokens_range[1]
        enc_pad_length = max_input_tokens - enc_tokens.shape[0]
        dec_pad_length = max_target_tokens - dec_tokens.shape[0]
        enc_tokens = F.pad(enc_tokens, (0, enc_pad_length), mode="constant", value=0)
        enc_positions = F.pad(enc_positions, (0, enc_pad_length), mode="constant", value=0)
        enc_modalities = F.pad(enc_modalities, (0, enc_pad_length), mode="constant", value=0)
        dec_positions = F.pad(dec_positions, (0, dec_pad_length), mode="constant", value=0)
        dec_tokens = F.pad(dec_tokens, (0, dec_pad_length), mode="constant", value=-100)
        dec_modalities = F.pad(dec_modalities, (0, dec_pad_length), mode="constant", value=0)

        enc_pad_mask = torch.ones(max_input_tokens, dtype=torch.bool)
        if enc_pad_length > 0:
            enc_pad_mask[-enc_pad_length:] = False
        dec_pad_mask = torch.ones(max_target_tokens, dtype=torch.bool)
        if dec_pad_length > 0:
            dec_pad_mask[-dec_pad_length:] = False

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


__all__ = ["BlockMasking"]
