"""Unit tests for the CLEVR scene-description parser.

Week-1 gate (proposal Section V): these tests must pass before any training
launches. They guard against silent parsing regressions caused by tokenizer
phrasing drift.

Run:
    cd nano4M && python -m pytest tests/test_scene_parser.py -v

Fixtures here are inline synthetic strings that follow the canonical format
confirmed against real val-split data on 2026-04-21
(see nano4M/tests/fixtures/scene_desc_samples.txt). The integration test
(test_scene_parser_integration.py) separately exercises the parser on 30
real samples.
"""
from nanofm.evaluation.scene_parser import (
    parse_scene_description,
    format_scene_description,
    SceneObject,
)


# ----------------------------------------------------- single-object cases

def test_single_object_canonical():
    text = "Object 1 - Position: x=35 y=37 Shape: sphere Color: blue Material: metal."
    objs = parse_scene_description(text)
    assert len(objs) == 1
    assert objs[0] == SceneObject(x=35, y=37, shape="sphere", color="blue", material="metal")


def test_single_object_case_insensitive():
    text = "OBJECT 1 - POSITION: X=3 Y=2 SHAPE: Sphere COLOR: Red MATERIAL: Rubber"
    objs = parse_scene_description(text)
    assert len(objs) == 1
    # Fields should be lowercased.
    assert objs[0].shape == "sphere"
    assert objs[0].color == "red"
    assert objs[0].material == "rubber"


def test_single_object_extra_whitespace():
    text = "Object  2  -  Position :  x = 10  y = 20   Shape :  cylinder   Color :  green   Material :  rubber ."
    objs = parse_scene_description(text)
    assert len(objs) == 1
    assert objs[0] == SceneObject(x=10, y=20, shape="cylinder", color="green", material="rubber")


def test_single_digit_coordinates():
    # Sample 18 in the real fixture has x=9; parser must accept 1+ digits.
    text = "Object 1 - Position: x=9 y=4 Shape: cube Color: blue Material: metal."
    objs = parse_scene_description(text)
    assert len(objs) == 1
    assert objs[0].x == 9 and objs[0].y == 4


# ------------------------------------------------------ multi-object cases

def test_two_objects():
    text = (
        "Object 1 - Position: x=35 y=37 Shape: sphere Color: blue Material: metal. "
        "Object 2 - Position: x=79 y=47 Shape: cylinder Color: cyan Material: metal."
    )
    objs = parse_scene_description(text)
    assert len(objs) == 2
    assert objs[0].shape == "sphere"
    assert objs[1].shape == "cylinder"


def test_ten_objects_double_digit_index():
    # Real val data contains scenes with up to 10 objects (samples 09, 16, 18, 27).
    # `Object 10` must be matched correctly (two-digit index).
    parts = [
        f"Object {i} - Position: x={10*i} y={20} Shape: cube Color: red Material: metal."
        for i in range(1, 11)
    ]
    text = " ".join(parts)
    objs = parse_scene_description(text)
    assert len(objs) == 10
    assert objs[9].x == 100  # the 10th object


# --------------------------------------------------------- edge cases

def test_empty_string():
    assert parse_scene_description("") == []


def test_whitespace_only():
    assert parse_scene_description("   \n\t  ") == []


def test_malformed_not_matched():
    # Missing the Material field -> parser should skip this object, not crash.
    text = "Object 1 - Position: x=5 y=7 Shape: cube Color: blue."
    objs = parse_scene_description(text)
    assert objs == []


def test_negative_coordinates():
    # Real data edge case: sample 00 in the fixture has x=-2.
    text = "Object 1 - Position: x=-2 y=50 Shape: cube Color: yellow Material: rubber."
    objs = parse_scene_description(text)
    assert len(objs) == 1
    assert objs[0].x == -2 and objs[0].y == 50


def test_rejects_old_scaffold_format():
    # The scaffold's original regex matched "Object at (5, 7): Shape: cube, ..."
    # which doesn't exist in real CLEVR data. A parser that still accepts it
    # would mask tokenizer drift. Guard against that regression.
    text = "Object at (5, 7): Shape: cube, Color: blue, Material: metal."
    assert parse_scene_description(text) == []


# ------------------------------------- roundtrip via format_scene_description

def test_roundtrip_preserves_content():
    original = [
        SceneObject(x=35, y=37, shape="sphere", color="blue", material="metal"),
        SceneObject(x=79, y=47, shape="cylinder", color="cyan", material="metal"),
    ]
    text = format_scene_description(original)
    parsed = parse_scene_description(text)
    assert parsed == original


def test_format_uses_canonical_form():
    # Format must emit the exact on-disk syntax so downstream string-matching
    # tools (e.g. diff reports, wandb logs) render the same as training data.
    obj = SceneObject(x=35, y=37, shape="sphere", color="blue", material="metal")
    expected = "Object 1 - Position: x=35 y=37 Shape: sphere Color: blue Material: metal."
    assert format_scene_description([obj]) == expected
