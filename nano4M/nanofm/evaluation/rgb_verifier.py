"""GroundingDINO-based object-detection verifier for cross-modal text → RGB.

Builds text prompts from the GT scene_desc (e.g. "a red metallic cube"), runs
zero-shot detection on the generated RGB image, and scores precision/recall
against the expected object list.

If GroundingDINO can't see CLEVR's stylized renders (recall < 0.8 on GT
images), swap in `clevr_rules_segmenter` (~50 lines) — defined here as a
stub for the team to fill in if/when needed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .scene_parser import SceneObject
from .model_revisions import GROUNDINGDINO_REPO, GROUNDINGDINO_REVISION


def _prompt_for(obj: SceneObject) -> str:
    parts = [p for p in (obj.color, obj.material, obj.shape) if p]
    return "a " + " ".join(parts) if parts else "an object"


class RGBVerifier:
    """Zero-shot object-detection verifier built on GroundingDINO."""

    def __init__(
        self,
        model_name: str = GROUNDINGDINO_REPO,
        model_revision: Optional[str] = GROUNDINGDINO_REVISION,
        device: str = "cuda",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.processor = AutoProcessor.from_pretrained(model_name, revision=model_revision)
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_name, revision=model_revision)
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def score(
        self, image: torch.Tensor, expected_objects: Sequence[SceneObject]
    ) -> Dict[str, float]:
        """Score one (image, expected) pair.

        image: (3, H, W) tensor in [0, 1].
        expected_objects: list of SceneObject from the GT caption.
        """
        if not expected_objects:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "n_expected": 0,
                "n_detected": 0,
            }

        prompts = [_prompt_for(obj) for obj in expected_objects]
        # GroundingDINO expects prompts joined with " . " separators, lowercase.
        text_prompt = " . ".join(prompts).lower() + " ."

        # Convert (3, H, W) tensor in [0, 1] to PIL for the processor.
        from torchvision.transforms.functional import to_pil_image
        pil_image = to_pil_image(image.clamp(0.0, 1.0))

        inputs = self.processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)

        post = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],  # (H, W)
        )[0]

        detected_labels = [str(lbl).strip() for lbl in post.get("labels", [])]
        n_detected = len(detected_labels)
        n_expected = len(expected_objects)

        # Match: a detected label matches an expected prompt if any expected
        # prompt is a substring of the label or vice versa.
        used_expected = [False] * n_expected
        true_positives = 0
        for lbl in detected_labels:
            for i, prompt in enumerate(prompts):
                if used_expected[i]:
                    continue
                p = prompt.replace("a ", "").strip().lower()
                if p in lbl.lower() or lbl.lower() in p:
                    used_expected[i] = True
                    true_positives += 1
                    break

        precision = true_positives / n_detected if n_detected > 0 else float("nan")
        recall = true_positives / n_expected if n_expected > 0 else float("nan")
        if precision != precision or recall != recall or (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_expected": n_expected,
            "n_detected": n_detected,
        }


class CLEVRRulesSegmenter:
    """CLEVR-specific fallback verifier (~50-line color+shape segmenter).

    To be implemented if and only if GroundingDINO fails the GT-recall
    sanity gate (mean recall < 0.8 on 20 GT val pairs). Until then, this is
    a stub that raises a clear error so callers know it's not wired up.
    """

    CLEVR_COLORS = (
        "gray", "red", "blue", "green", "brown",
        "purple", "cyan", "yellow",
    )

    def __init__(self) -> None:
        raise NotImplementedError(
            "CLEVRRulesSegmenter is the GroundingDINO fallback. Only fill "
            "this in if `RGBVerifier` recall on GT images < 0.8."
        )

    def score(
        self, image: torch.Tensor, expected_objects: Sequence[SceneObject]
    ) -> Dict[str, float]:
        raise NotImplementedError
