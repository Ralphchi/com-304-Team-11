"""Qwen-based LLM judge for caption alignment.

Loads `Qwen/Qwen3-8B-Instruct` (or a smaller Qwen3 variant) once, then scores
each (original, generated) caption pair by asking the model whether the two
captions describe the same set of CLEVR objects.

The judge is deterministic (`do_sample=False`, `temperature=0`) so the same
caption pair always scores the same. JSON-parse failures fall back to a
regex-friendly plain-text prompt; any remaining failure is recorded as a
parse error and excluded from the alignment mean.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import torch

from .model_revisions import QWEN_REPO, QWEN_REVISION


PRIMARY_PROMPT = """You are evaluating two CLEVR scene captions. Each describes a small set of \
objects, where each object has attributes: position (x, y), shape, color, material.

ORIGINAL: {original}
GENERATED: {generated}

Determine whether GENERATED describes the same set of objects as ORIGINAL. \
Order does not matter; an object in GENERATED matches one in ORIGINAL if their \
shape, color, material, and approximate position all agree.

Return strictly this JSON, no extra text:
{{
  "alignment": <float in [0.0, 1.0]>,
  "missing_objects": [<short descriptions of objects in ORIGINAL but not GENERATED>],
  "extra_objects": [<short descriptions of objects in GENERATED but not ORIGINAL>],
  "wrong_attributes": [<short descriptions of attribute mismatches on otherwise-aligned objects>]
}}

alignment must be 1.0 iff missing_objects, extra_objects, and wrong_attributes are all empty.
"""

BACKUP_PROMPT = """ORIGINAL: {original}
GENERATED: {generated}
Do these captions describe the same scene? Reply with one line:
SCORE: <float 0-1>
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_SCORE_RE = re.compile(r"SCORE:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MAX_GENERATED_TOKENS_FOR_PROMPT = 300


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " […]"


class LLMJudge:
    """Wraps a Qwen3 instruction-tuned model for caption-pair alignment scoring."""

    def __init__(
        self,
        model_name: str = QWEN_REPO,
        model_revision: Optional[str] = QWEN_REVISION,
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 200,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision
        )
        load_kwargs: Dict[str, Any] = {"revision": model_revision}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not load_in_4bit:
            self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """One greedy generation pass; returns the new tokens decoded."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def score(self, original: str, generated: str) -> Dict[str, Any]:
        """Score a single (original, generated) pair.

        Returns a dict with at least:
            alignment: float in [0, 1] (0.0 on parse error)
            parse_error: bool (True if both prompts failed to yield a number)
            missing_objects, extra_objects, wrong_attributes (when JSON works)
        """
        # Cheap edge cases — no LLM call.
        if not (original or "").strip() or not (generated or "").strip():
            return {
                "alignment": 0.0,
                "parse_error": False,
                "missing_objects": [],
                "extra_objects": [],
                "wrong_attributes": [],
            }

        original = _truncate(original, _MAX_GENERATED_TOKENS_FOR_PROMPT * 4)
        generated = _truncate(generated, _MAX_GENERATED_TOKENS_FOR_PROMPT * 4)

        # Primary prompt → JSON.
        out = self._generate(PRIMARY_PROMPT.format(original=original, generated=generated))
        m = _JSON_BLOCK_RE.search(out)
        if m is not None:
            try:
                obj = json.loads(m.group(0))
                return {
                    "alignment": float(obj.get("alignment", 0.0)),
                    "parse_error": False,
                    "missing_objects": list(obj.get("missing_objects", [])),
                    "extra_objects": list(obj.get("extra_objects", [])),
                    "wrong_attributes": list(obj.get("wrong_attributes", [])),
                }
            except (json.JSONDecodeError, ValueError):
                pass

        # Backup prompt → SCORE: <float>.
        out = self._generate(BACKUP_PROMPT.format(original=original, generated=generated))
        m = _SCORE_RE.search(out)
        if m is not None:
            try:
                return {
                    "alignment": max(0.0, min(1.0, float(m.group(1)))),
                    "parse_error": False,
                    "missing_objects": [],
                    "extra_objects": [],
                    "wrong_attributes": [],
                }
            except ValueError:
                pass

        return {
            "alignment": 0.0,
            "parse_error": True,
            "missing_objects": [],
            "extra_objects": [],
            "wrong_attributes": [],
        }

    def score_batch(
        self, originals: List[str], generateds: List[str]
    ) -> List[Dict[str, Any]]:
        return [self.score(o, g) for o, g in zip(originals, generateds)]
