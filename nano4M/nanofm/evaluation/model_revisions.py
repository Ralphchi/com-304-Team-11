"""Pinned HuggingFace revisions for every external model used in eval.

Filled in once the team commits to a specific snapshot. Until then, leave
as None to fetch the latest, but pin before running the headline eval so
results are reproducible.
"""

# Qwen LLM judge
QWEN_REPO = "Qwen/Qwen3-8B-Instruct"
QWEN_REVISION: str | None = None  # pin a commit SHA before the headline eval

# GroundingDINO object-detection verifier
GROUNDINGDINO_REPO = "IDEA-Research/grounding-dino-tiny"
GROUNDINGDINO_REVISION: str | None = None

# GPT-2 detokenizer (used by SimpleMultimodalDataset; mirrored here for clarity)
GPT2_REPO = "gpt2"
GPT2_REVISION: str | None = None

# Cosmos-DI16x16 image tokenizer/decoder
COSMOS_REPO = "nvidia/Cosmos-0.1-Tokenizer-DI16x16"
COSMOS_REVISION: str | None = None
