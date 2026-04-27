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

"""V2 Context-Block Masking (Team 11 extension, proposal Section III.V2).

Image-like modalities (RGB / depth / normals tokens) get a single contiguous
s x s visible "context block" in the 16x16 grid; everything else in that
modality is a candidate target. Text modalities (scene_desc) keep baseline
random masking. The block size s is sampled per call from {4, 5, 6}; the
block position is sampled per image modality independently.

Budget matching (proposal Section III.V2):
  Let N = per-image-modality target count drawn from the parent's Dirichlet
  helper.
  - Regime 1 (N <= 256 - s^2): pick N random positions from outside the block.
  - Regime 2 (N >  256 - s^2): mask everything outside the block PLUS
    N - (256 - s^2) random positions from inside the block, shrinking the
    visible region accordingly.
  With the baseline target_tokens_range = (1, 128) and s in {4, 5, 6},
  256 - s^2 >= 220, so regime 2 never triggers under baseline budget;
  it is implemented for correctness only.
"""

from typing import List, Tuple, Dict, Any, Union
import random

import torch
import torch.nn.functional as F

from .masking import SimpleMultimodalMasking
from .utils import to_unified_multimodal_vocab


class ContextBlockMasking(SimpleMultimodalMasking):
    def __init__(
            self,
            modalities: List[str],
            vocab_sizes: List[int],
            max_seq_lens: List[int],
            input_alphas: List[float],
            target_alphas: List[float],
            input_tokens_range: Union[int, Tuple[int, int]],
            target_tokens_range: Union[int, Tuple[int, int]],
            overlap_vocab: bool = True,
            overlap_posembs: bool = True,
            include_unmasked_data_dict: bool = False,
            image_modalities: List[str] = ("tok_rgb@256", "tok_depth@256", "tok_normal@256"),
            text_modalities: List[str] = ("scene_desc",),
            grid_size: int = 16,
            context_block_sizes: List[int] = (4, 5, 6),
        ):
        """V2 context-block masking transform.

        Inherits __init__ from SimpleMultimodalMasking (sets up Dirichlet helpers,
        max_seq_len_shifts, etc.) and adds image/text dispatch + block-size config.

        Args:
            image_modalities: Modalities to apply context-block masking to. Each
                must have a 256-token sequence reshapeable to grid_size x grid_size.
            text_modalities: Modalities to apply baseline random masking to.
            grid_size: Side length of the image-token grid (16 for 256 tokens).
            context_block_sizes: Allowed visible-block side lengths; s sampled
                uniformly from this list per __call__.
        """
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

        self.image_modalities = list(image_modalities)
        self.text_modalities = list(text_modalities)
        self.grid_size = int(grid_size)
        self.context_block_sizes = list(context_block_sizes)

        # Validate that image + text modalities cover the full modality list.
        covered = set(self.image_modalities) | set(self.text_modalities)
        declared = set(modalities)
        if covered != declared:
            missing = declared - covered
            extra = covered - declared
            raise ValueError(
                f"image_modalities + text_modalities must equal modalities. "
                f"Missing from dispatch: {missing}. Extra: {extra}."
            )

        # Cache modality indices for fast lookup in __call__.
        self.image_idxs = [modalities.index(m) for m in self.image_modalities]
        self.text_idxs = [modalities.index(m) for m in self.text_modalities]

        # Per-image-modality position grid (256 tokens => 16x16).
        for m in self.image_modalities:
            mod_idx = modalities.index(m)
            if max_seq_lens[mod_idx] != self.grid_size * self.grid_size:
                raise ValueError(
                    f"Image modality {m!r} has max_seq_len={max_seq_lens[mod_idx]} "
                    f"but grid_size={self.grid_size} requires {self.grid_size**2}."
                )

        for s in self.context_block_sizes:
            if s < 1 or s > self.grid_size:
                raise ValueError(
                    f"context_block_size {s} out of range [1, {self.grid_size}]."
                )

    def _sample_block_positions(self, s: int) -> torch.Tensor:
        """Sample a random s x s block of grid positions; return as sorted long tensor."""
        r0 = random.randint(0, self.grid_size - s)
        c0 = random.randint(0, self.grid_size - s)
        rows = torch.arange(r0, r0 + s)
        cols = torch.arange(c0, c0 + s)
        # Cartesian product of rows x cols -> flat grid indices.
        grid = rows.unsqueeze(1) * self.grid_size + cols.unsqueeze(0)
        return grid.reshape(-1).sort()[0].long()

    def _image_modality_positions(
            self,
            s: int,
            n_target_tokens: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick (input_pos, target_pos) for a single image modality.

        Returns sorted long tensors. Implements both regimes from proposal V2.
        """
        block = self._sample_block_positions(s)
        block_set = set(block.tolist())
        all_positions = set(range(self.grid_size * self.grid_size))
        outside = sorted(all_positions - block_set)

        if n_target_tokens <= len(outside):
            # Regime 1 — N targets sampled from outside the block; encoder = full block.
            target_pos = sorted(random.sample(outside, n_target_tokens))
            input_pos = block.tolist()
        else:
            # Regime 2 (defensive; never triggers at baseline budget) — spill into block.
            extra = n_target_tokens - len(outside)
            spilled = random.sample(sorted(block_set), extra)
            target_pos = sorted(outside + spilled)
            input_pos = sorted(block_set - set(spilled))

        input_pos_t = torch.tensor(input_pos, dtype=torch.long)
        target_pos_t = torch.tensor(target_pos, dtype=torch.long)
        return input_pos_t, target_pos_t

    def _input_token_budget_v2(
            self,
            num_input_tokens: int,
            max_tokens: torch.Tensor,
            s: int,
        ) -> List[int]:
        """Per-modality input budget for V2.

        Image modalities are pinned to s^2 (one full visible block). Text
        modalities share the remaining encoder capacity via the parent's
        Dirichlet, but we cap the total so that 3*s^2 + text_budget never
        exceeds max_input_tokens (the encoder padding cap). With
        max_input_tokens=128 and s=6, text_budget in [0, 20]; with s=4,
        text_budget in [0, 80].
        """
        max_input_tokens = self.input_tokens_range[1]
        image_total = sum(s * s for _ in self.image_idxs)
        text_capacity = max(0, max_input_tokens - image_total)

        # If there are no text modalities, all budget goes to images.
        if not self.text_idxs:
            budget = [0] * self.num_modalities
            for mi in self.image_idxs:
                budget[mi] = s * s
            return budget

        # Cap text request at remaining capacity AND at num_input_tokens (so we
        # don't oversample text relative to the per-call random size).
        text_request = min(num_input_tokens, text_capacity)

        # Use the parent's Dirichlet helper to allocate text_request across text
        # modalities. We feed it max_tokens for those modalities only.
        # Build a mini-Dirichlet over text modalities. Easiest: run the parent
        # helper on the full set with text_request as the total, then zero out
        # image modalities (their share is overridden anyway).
        if text_request > 0 and len(self.text_idxs) > 0:
            text_alloc_full = self.input_token_budget(text_request, max_tokens)
        else:
            text_alloc_full = [0] * self.num_modalities

        budget = [0] * self.num_modalities
        for mi in self.image_idxs:
            budget[mi] = s * s
        for mi in self.text_idxs:
            budget[mi] = int(text_alloc_full[mi])

        # If Dirichlet redirected some text_request to image modalities (because
        # the alphas allow it), reclaim those tokens onto the first text modality
        # (best-effort, deterministic). Doesn't affect the alphas-1.0 default
        # case since Dirichlet then gives image modalities ~text_request/4 each
        # which we want to discard.
        leftover = text_request - sum(budget[mi] for mi in self.text_idxs)
        if leftover > 0 and self.text_idxs:
            # Spread leftover onto text modalities, capped by per-modality max.
            for mi in self.text_idxs:
                room = int(max_tokens[mi].item()) - budget[mi]
                add = min(leftover, max(0, room))
                budget[mi] += add
                leftover -= add
                if leftover <= 0:
                    break
        return budget

    def _perform_context_block_masking(
            self,
            data_dict: Dict[str, Any],
            input_token_budget: List[int],
            target_token_budget: List[int],
            s: int,
        ) -> Dict[str, Any]:
        """Build the masked_data_dict for one sample under V2 context-block masking.

        Mirrors the tail of SimpleMultimodalMasking.perform_random_masking (concat,
        pad, build pad masks) so the output contract matches FourM's expectations
        exactly.
        """
        enc_tokens, enc_positions, enc_modalities = [], [], []
        dec_tokens, dec_positions, dec_modalities = [], [], []

        for mod_idx, mod in enumerate(self.modalities):
            n_input_tokens = input_token_budget[mod_idx]
            n_target_tokens = target_token_budget[mod_idx]

            if mod in self.image_modalities:
                # Image modality: context-block selection.
                # n_input_tokens from _input_token_budget_v2 is s^2 by construction;
                # we use n_target_tokens as N (regime 1/2 inside the helper).
                input_pos, target_pos = self._image_modality_positions(s, n_target_tokens)
            else:
                # Text modality: baseline random masking.
                num_tokens = data_dict[mod].shape[0]
                noise = torch.rand(num_tokens)
                ids_shuffle = torch.argsort(noise, dim=0)
                input_pos = ids_shuffle[:n_input_tokens].sort()[0].long()
                target_pos = ids_shuffle[n_input_tokens:n_input_tokens + n_target_tokens].sort()[0].long()

            pos_idx_shift = 0 if self.overlap_posembs else int(self.max_seq_len_shifts[mod_idx].item())
            enc_positions.append(input_pos + pos_idx_shift)
            dec_positions.append(target_pos + pos_idx_shift)

            enc_tokens.append(data_dict[mod][input_pos])
            dec_tokens.append(data_dict[mod][target_pos])

            n_input_actual = input_pos.shape[0]
            n_target_actual = target_pos.shape[0]
            enc_modalities.append(mod_idx * torch.ones(n_input_actual, dtype=torch.long))
            dec_modalities.append(mod_idx * torch.ones(n_target_actual, dtype=torch.long))

        enc_tokens = torch.cat(enc_tokens) if enc_tokens else torch.empty(0, dtype=torch.long)
        dec_tokens = torch.cat(dec_tokens) if dec_tokens else torch.empty(0, dtype=torch.long)
        enc_positions = torch.cat(enc_positions) if enc_positions else torch.empty(0, dtype=torch.long)
        dec_positions = torch.cat(dec_positions) if dec_positions else torch.empty(0, dtype=torch.long)
        enc_modalities = torch.cat(enc_modalities) if enc_modalities else torch.empty(0, dtype=torch.long)
        dec_modalities = torch.cat(dec_modalities) if dec_modalities else torch.empty(0, dtype=torch.long)

        max_input_tokens = self.input_tokens_range[1]
        max_target_tokens = self.target_tokens_range[1]

        # Defensive: the encoder pad logic below assumes we don't overflow.
        # _input_token_budget_v2 enforces 3*s^2 + text_budget <= max_input_tokens,
        # so this should never fire. Surface clearly if it ever does.
        if enc_tokens.shape[0] > max_input_tokens:
            raise RuntimeError(
                f"V2 encoder fill {enc_tokens.shape[0]} exceeds "
                f"max_input_tokens={max_input_tokens} (s={s}, "
                f"input_token_budget={input_token_budget}). Bug in "
                f"_input_token_budget_v2."
            )
        if dec_tokens.shape[0] > max_target_tokens:
            raise RuntimeError(
                f"V2 decoder fill {dec_tokens.shape[0]} exceeds "
                f"max_target_tokens={max_target_tokens} "
                f"(target_token_budget={target_token_budget})."
            )

        enc_pad_length = max_input_tokens - enc_tokens.shape[0]
        dec_pad_length = max_target_tokens - dec_tokens.shape[0]
        enc_tokens = F.pad(enc_tokens, (0, enc_pad_length), mode='constant', value=0)
        enc_positions = F.pad(enc_positions, (0, enc_pad_length), mode='constant', value=0)
        enc_modalities = F.pad(enc_modalities, (0, enc_pad_length), mode='constant', value=0)
        dec_positions = F.pad(dec_positions, (0, dec_pad_length), mode='constant', value=0)
        dec_tokens = F.pad(dec_tokens, (0, dec_pad_length), mode='constant', value=-100)
        dec_modalities = F.pad(dec_modalities, (0, dec_pad_length), mode='constant', value=0)

        enc_pad_mask = torch.ones(max_input_tokens, dtype=torch.bool)
        if enc_pad_length > 0:
            enc_pad_mask[-enc_pad_length:] = False
        dec_pad_mask = torch.ones(max_target_tokens, dtype=torch.bool)
        if dec_pad_length > 0:
            dec_pad_mask[-dec_pad_length:] = False

        return {
            'enc_tokens': enc_tokens,
            'enc_positions': enc_positions,
            'enc_modalities': enc_modalities,
            'enc_pad_mask': enc_pad_mask,
            'dec_tokens': dec_tokens,
            'dec_positions': dec_positions,
            'dec_modalities': dec_modalities,
            'dec_pad_mask': dec_pad_mask,
        }

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.overlap_vocab:
            data_dict = to_unified_multimodal_vocab(data_dict, self.modalities, self.vocab_sizes)

        max_tokens = torch.tensor(self.max_seq_lens)

        num_input_tokens = random.randint(*self.input_tokens_range)
        num_target_tokens = random.randint(*self.target_tokens_range)

        # Sample one s for this call; all 3 image modalities share it. The
        # block POSITION is then sampled independently per image modality
        # inside _image_modality_positions.
        s = random.choice(self.context_block_sizes)

        input_token_budget = self._input_token_budget_v2(num_input_tokens, max_tokens, s)
        target_token_budget = self.target_token_budget(input_token_budget, num_target_tokens, max_tokens)

        # Cap image target budgets so regime 2 only triggers when truly forced
        # (it cannot under baseline budget; this is a no-op there).
        outside_capacity = self.grid_size * self.grid_size  # = 256, the total grid
        for mi in self.image_idxs:
            target_token_budget[mi] = min(target_token_budget[mi], outside_capacity)

        masked_data_dict = self._perform_context_block_masking(
            data_dict, input_token_budget, target_token_budget, s
        )

        if self.include_unmasked_data_dict:
            masked_data_dict['unmasked_data_dict'] = data_dict

        return masked_data_dict
