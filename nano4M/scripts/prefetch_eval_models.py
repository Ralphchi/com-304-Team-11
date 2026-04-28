#!/usr/bin/env python
"""Cache HuggingFace models on a SCITAS login node before submitting eval jobs.

Run once before the first eval submission:

    HF_HOME=/work/com-304/hf_cache python scripts/prefetch_eval_models.py

Eval SLURM jobs then export the same HF_HOME and TRANSFORMERS_OFFLINE=1 so
they don't try to download from compute nodes (which sometimes lack internet).
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    from huggingface_hub import snapshot_download

    from nanofm.evaluation.model_revisions import (
        QWEN_REPO, QWEN_REVISION,
        GROUNDINGDINO_REPO, GROUNDINGDINO_REVISION,
        GPT2_REPO, GPT2_REVISION,
        COSMOS_REPO, COSMOS_REVISION,
    )

    targets = [
        ("Qwen LLM judge", QWEN_REPO, QWEN_REVISION),
        ("GroundingDINO detector", GROUNDINGDINO_REPO, GROUNDINGDINO_REVISION),
        ("GPT-2 detokenizer", GPT2_REPO, GPT2_REVISION),
        ("Cosmos-DI16x16 tokenizer", COSMOS_REPO, COSMOS_REVISION),
    ]

    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    print(f"HF_HOME = {cache}")

    for label, repo, revision in targets:
        print(f"-> {label}: {repo} (revision={revision or 'latest'})")
        snapshot_download(repo_id=repo, revision=revision)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
