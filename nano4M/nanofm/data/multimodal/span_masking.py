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

"""V3 Span Masking (Team 11 extension, proposal Section III.V3).

Text modalities (scene_desc) get contiguous span masks: span lengths drawn
from a geometric distribution with mean mu=3, placed at random starts
without overlap until the per-modality target count N is reached. If a
sampled length would exceed N remaining, the span is trimmed to fit. Image
modalities (RGB, depth, normals) keep baseline random masking.

mu calibration (verified on 500 CLEVR val captions, 2804 objects, GPT-2):
  Position: 8 tokens (constant). Shape/Color/Material: 3 tokens each (constant).
  Median Field:value length = 3, so mu=3 matches the dominant phrase length.
Re-run via `python -m nanofm.data.multimodal.span_masking` (see __main__).
"""

from typing import List, Tuple, Dict, Any, Union
import random

import torch
import torch.nn.functional as F

from .masking import SimpleMultimodalMasking
from .utils import to_unified_multimodal_vocab


class SpanMasking(SimpleMultimodalMasking):
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
            text_modalities: List[str] = ("scene_desc",),
            image_modalities: List[str] = ("tok_rgb@256", "tok_depth@256", "tok_normal@256"),
            mean_span_length: float = 3.0,
            max_retries: int = 100,
        ):
        """V3 span masking transform.

        Inherits __init__ from SimpleMultimodalMasking (Dirichlet helpers,
        max_seq_len_shifts, etc.) and adds text/image dispatch + span config.

        Args:
            text_modalities: Modalities to apply span masking to.
            image_modalities: Modalities to apply baseline random masking to.
            mean_span_length: Mean of the geometric distribution span lengths
                are drawn from. Span lengths fall in {1, 2, ...} with this mean.
            max_retries: Max attempts to find a non-overlapping start position
                before falling back to random fill of remaining free positions.
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

        self.text_modalities = list(text_modalities)
        self.image_modalities = list(image_modalities)
        self.mean_span_length = float(mean_span_length)
        self.max_retries = int(max_retries)

        covered = set(self.image_modalities) | set(self.text_modalities)
        declared = set(modalities)
        if covered != declared:
            missing = declared - covered
            extra = covered - declared
            raise ValueError(
                f"image_modalities + text_modalities must equal modalities. "
                f"Missing from dispatch: {missing}. Extra: {extra}."
            )
        if set(self.image_modalities) & set(self.text_modalities):
            raise ValueError(
                f"image_modalities and text_modalities overlap: "
                f"{set(self.image_modalities) & set(self.text_modalities)}"
            )

        self.text_idxs = [modalities.index(m) for m in self.text_modalities]
        self.image_idxs = [modalities.index(m) for m in self.image_modalities]

        if self.mean_span_length < 1.0:
            raise ValueError(
                f"mean_span_length must be >= 1, got {self.mean_span_length}"
            )

        # torch.distributions.Geometric(probs=p) is supported on {0, 1, ...}
        # with mean (1-p)/p. Sampling X ~ Geometric(p=1/mu) and returning X+1
        # gives span lengths in {1, 2, ...} with mean exactly mu.
        self._geom_dist = torch.distributions.Geometric(
            probs=torch.tensor(1.0 / self.mean_span_length, dtype=torch.float32)
        )

    def _sample_span_length(self) -> int:
        """One geometric draw shifted to {1, 2, ...}."""
        return int(self._geom_dist.sample().item()) + 1

    def _text_modality_positions(
            self,
            n_input: int,
            n_target: int,
            max_seq_len: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick (input_pos, target_pos) for one text modality.

        Targets form contiguous spans of geometric length, no overlap, summing
        to exactly n_target. Inputs are sampled uniformly from the remaining
        free positions, capped at n_input.
        """
        target_set = set()
        remaining = n_target
        while remaining > 0:
            # Trim sampled length to remaining budget (proposal Section III.V3)
            # and to max_seq_len so start position is always valid.
            length = min(self._sample_span_length(), remaining, max_seq_len)
            placed = False
            for _ in range(self.max_retries):
                start = random.randint(0, max_seq_len - length)
                positions = set(range(start, start + length))
                if not (positions & target_set):
                    target_set.update(positions)
                    remaining -= length
                    placed = True
                    break
            if not placed:
                # Retry exhaustion: fill remaining budget from any free
                # positions. With baseline density (N <= 128, max_seq_len = 256,
                # mu = 3) this branch is essentially unreachable. Decrement
                # remaining by what we actually placed so we don't silently
                # undershoot if free < remaining (only possible if upstream
                # budget already exceeds max_seq_len, but be defensive).
                free = sorted(set(range(max_seq_len)) - target_set)
                if not free:
                    break
                take = min(remaining, len(free))
                extra = random.sample(free, take)
                target_set.update(extra)
                remaining -= take

        free = sorted(set(range(max_seq_len)) - target_set)
        n_input_actual = min(n_input, len(free))
        input_set = set(random.sample(free, n_input_actual)) if n_input_actual > 0 else set()

        input_pos = torch.tensor(sorted(input_set), dtype=torch.long)
        target_pos = torch.tensor(sorted(target_set), dtype=torch.long)
        return input_pos, target_pos

    def _perform_span_masking(
            self,
            data_dict: Dict[str, Any],
            input_token_budget: List[int],
            target_token_budget: List[int],
        ) -> Dict[str, Any]:
        """Build the masked_data_dict for one sample under V3 span masking.

        Mirrors the tail of SimpleMultimodalMasking.perform_random_masking
        (concat, pad, build pad masks) so the output contract matches FourM.
        """
        enc_tokens, enc_positions, enc_modalities = [], [], []
        dec_tokens, dec_positions, dec_modalities = [], [], []

        for mod_idx, mod in enumerate(self.modalities):
            n_input_tokens = input_token_budget[mod_idx]
            n_target_tokens = target_token_budget[mod_idx]

            if mod in self.text_modalities:
                input_pos, target_pos = self._text_modality_positions(
                    n_input_tokens, n_target_tokens, data_dict[mod].shape[0]
                )
            else:
                # Image modality: baseline random masking.
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

        if enc_tokens.shape[0] > max_input_tokens:
            raise RuntimeError(
                f"V3 encoder fill {enc_tokens.shape[0]} exceeds "
                f"max_input_tokens={max_input_tokens} (input_token_budget={input_token_budget})."
            )
        if dec_tokens.shape[0] > max_target_tokens:
            raise RuntimeError(
                f"V3 decoder fill {dec_tokens.shape[0]} exceeds "
                f"max_target_tokens={max_target_tokens} (target_token_budget={target_token_budget})."
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

        input_token_budget = self.input_token_budget(num_input_tokens, max_tokens)
        target_token_budget = self.target_token_budget(input_token_budget, num_target_tokens, max_tokens)

        masked_data_dict = self._perform_span_masking(
            data_dict, input_token_budget, target_token_budget
        )

        if self.include_unmasked_data_dict:
            masked_data_dict['unmasked_data_dict'] = data_dict

        return masked_data_dict


if __name__ == "__main__":
    # mu calibration: median tokenised length of CLEVR Field:value phrases.
    # Run on KUMA inside the nanofm env to reproduce the numbers in the docstring.
    import json, glob, re, os, statistics
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    from tokenizers.processors import TemplateProcessing

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({"pad_token": "[PAD]"})
    tok.add_special_tokens({"bos_token": "[SOS]", "eos_token": "[EOS]"})
    tok._tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[("[EOS]", tok.eos_token_id), ("[SOS]", tok.bos_token_id)],
    )

    DATA_GLOB = "/work/com-304/datasets/clevr_com_304/val/scene_desc/*.json"
    files = sorted(glob.glob(DATA_GLOB))[:500]
    if not files:
        raise SystemExit(f"no scene_desc files found at {DATA_GLOB}")
    captions = [json.load(open(fp))[0] for fp in files]

    obj_re = re.compile(
        r"Object\s+\d+\s*-\s*"
        r"(?P<position>Position:\s*x=-?\d+\s+y=-?\d+)\s+"
        r"(?P<shape>Shape:\s*\w+)\s+"
        r"(?P<color>Color:\s*\w+)\s+"
        r"(?P<material>Material:\s*\w+)"
    )

    field_lens = {"position": [], "shape": [], "color": [], "material": []}
    n_objects = 0
    for cap in captions:
        for m in obj_re.finditer(cap):
            n_objects += 1
            for f in ("position", "shape", "color", "material"):
                ids = tok(" " + m.group(f), add_special_tokens=False)["input_ids"]
                field_lens[f].append(len(ids))

    def stats(xs):
        return (
            f"n={len(xs):4d}  median={statistics.median(xs):.1f}  "
            f"mean={statistics.fmean(xs):.2f}  min={min(xs)}  max={max(xs)}"
        )

    print(f"\nn_captions={len(captions)}  n_objects={n_objects}\n")
    for f, xs in field_lens.items():
        print(f"  {f:9s}: {stats(xs)}")
    all_lens = sum(field_lens.values(), [])
    print(f"\n  ALL combined         : {stats(all_lens)}")
    non_pos = field_lens["shape"] + field_lens["color"] + field_lens["material"]
    print(f"  Shape+Color+Material : {stats(non_pos)}")
