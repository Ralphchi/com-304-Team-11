"""CLEVR scene-description parser.

Parses the natural-language scene descriptions used in the CLEVR dataset into
structured tuples. Each scene contains K objects, each described by its
pixel-space position (x, y) and categorical attributes (shape, color, material).

Canonical ground-truth format (confirmed on 2026-04-21 against 30 samples from
/work/com-304/datasets/clevr_com_304/val/scene_desc/*.json — see
nano4M/tests/fixtures/scene_desc_samples.txt):

    "Object 1 - Position: x=35 y=37 Shape: sphere Color: blue Material: metal. "
    "Object 2 - Position: x=79 y=47 Shape: cylinder Color: cyan Material: metal."

Key properties of the real format (drive the regex design):
- Per-object prefix `Object <N> - ` with N starting at 1, up to 10.
- Coordinates use `x=<int> y=<int>` syntax (no parentheses, no comma).
- x may be negative (observed min: -2); y observed in [22, 79]; both may be 1+ digits.
- Attributes are space-separated `Shape: <word> Color: <word> Material: <word>`
  with NO commas between fields.
- GPT-2 tokenizer round-trips this format byte-for-byte (verified on all 30
  val samples), so the parser does not need to tolerate tokenizer drift.

CLEVR vocabulary (enforced by the integration test, not by the parser):
- shapes   : {cube, sphere, cylinder}
- colors   : {blue, cyan, yellow, purple, red, gray, green, brown}
- materials: {metal, rubber}

Usage
-----
    from nanofm.evaluation.scene_parser import parse_scene_description
    text = "Object 1 - Position: x=35 y=37 Shape: sphere Color: blue Material: metal."
    parse_scene_description(text)
    # -> [SceneObject(x=35, y=37, shape='sphere', color='blue', material='metal')]
"""

from dataclasses import dataclass
from typing import List
import re


@dataclass(frozen=True)
class SceneObject:
    """One object in a CLEVR scene."""
    x: int
    y: int
    shape: str
    color: str
    material: str


_OBJECT_PATTERN = re.compile(
    r"Object\s+\d+\s*-\s*"
    r"Position\s*:\s*x\s*=\s*(?P<x>-?\d+)\s+y\s*=\s*(?P<y>-?\d+)\s+"
    r"Shape\s*:\s*(?P<shape>\w+)\s+"
    r"Color\s*:\s*(?P<color>\w+)\s+"
    r"Material\s*:\s*(?P<material>\w+)",
    re.IGNORECASE,
)


def parse_scene_description(text: str) -> List[SceneObject]:
    """Parse a decoded scene-description string into a list of SceneObjects.

    Parameters
    ----------
    text : str
        Decoded scene_desc token sequence as a plain string.

    Returns
    -------
    List[SceneObject]
        One entry per object successfully matched. Malformed objects are
        silently skipped (they count as missed predictions during evaluation).

    Notes
    -----
    - Values are lowercased so comparison against ground truth is canonical.
    - Returns an empty list if the string is empty or contains no objects.
    """
    if not text or not text.strip():
        return []

    objects: List[SceneObject] = []
    for match in _OBJECT_PATTERN.finditer(text):
        objects.append(
            SceneObject(
                x=int(match.group("x")),
                y=int(match.group("y")),
                shape=match.group("shape").lower().strip(),
                color=match.group("color").lower().strip(),
                material=match.group("material").lower().strip(),
            )
        )
    return objects


def format_scene_description(objects: List[SceneObject]) -> str:
    """Inverse of `parse_scene_description` — useful for tests and debugging.

    Emits the canonical CLEVR scene-description format confirmed against real
    val-split JSON on 2026-04-21:

        "Object 1 - Position: x=35 y=37 Shape: sphere Color: blue Material: metal. "
        "Object 2 - Position: x=79 y=47 Shape: cylinder Color: cyan Material: metal."

    Parameters
    ----------
    objects : list of SceneObject

    Returns
    -------
    str
        A canonical-format scene description that, when fed back into
        `parse_scene_description`, produces an equivalent object list.
    """
    parts = [
        f"Object {i} - Position: x={obj.x} y={obj.y} "
        f"Shape: {obj.shape} Color: {obj.color} Material: {obj.material}."
        for i, obj in enumerate(objects, start=1)
    ]
    return " ".join(parts)
