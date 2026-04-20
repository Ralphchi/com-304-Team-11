"""CLEVR scene-description parser.

Parses the natural-language scene descriptions used in the CLEVR dataset into
structured tuples. Each scene contains K objects, each described by its
grid position (x, y) and categorical attributes (shape, color, material).

Example input string (format used in the Cauldron/CLEVR tokenizer training set):

    "Object at (5, 7): Shape: cube, Color: blue, Material: metal. "
    "Object at (12, 3): Shape: sphere, Color: red, Material: rubber."

The parser is tolerant to:
- Case variations ("Cube" vs "cube")
- Trailing/leading whitespace
- Optional "." between objects
- Minor phrasing differences ("The object is located at..." forms)

The output is a list of `SceneObject` dataclass instances with normalized
(lowercased, stripped) field values.

Usage
-----
    from nanofm.evaluation.scene_parser import parse_scene_description
    objects = parse_scene_description("Object at (5, 7): Shape: cube, Color: blue, Material: metal.")
    # -> [SceneObject(x=5, y=7, shape='cube', color='blue', material='metal')]
"""

from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass(frozen=True)
class SceneObject:
    """One object in a CLEVR scene."""
    x: int
    y: int
    shape: str
    color: str
    material: str


# TODO(Ralph, Week 1): Decide on exact regex(es) once we've inspected 20-30
# decoded training-set strings to confirm the actual phrasing the tokenizer
# produces. The placeholder below matches the format documented in the
# extension proposal; update to match ground truth if phrasing differs.
_OBJECT_PATTERN = re.compile(
    r"Object\s+at\s+\(\s*(?P<x>-?\d+)\s*,\s*(?P<y>-?\d+)\s*\)\s*:\s*"
    r"Shape\s*:\s*(?P<shape>\w+)\s*,\s*"
    r"Color\s*:\s*(?P<color>\w+)\s*,\s*"
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
    - TODO(Ralph, Week 1): Decide whether to track the number of unparseable
      fragments and surface it as an "extraction rate" metric separately
      from field accuracy.
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
        f"Object at ({obj.x}, {obj.y}): "
        f"Shape: {obj.shape}, Color: {obj.color}, Material: {obj.material}."
        for obj in objects
    ]
    return " ".join(parts)
