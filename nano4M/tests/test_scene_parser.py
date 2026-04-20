"""Unit tests for the CLEVR scene-description parser.

Week-1 gate (proposal Section V): these tests must pass before any training
launches. They guard against silent parsing regressions caused by tokenizer
phrasing drift.

Run:
    cd nano4M && python -m pytest tests/test_scene_parser.py -v

All tests use inline string fixtures; no external data dependency.
"""
import pytest

from nanofm.evaluation.scene_parser import (
    parse_scene_description,
    format_scene_description,
    SceneObject,
)


# ----------------------------------------------------- single-object cases

def test_single_object_canonical():
    text = "Object at (5, 7): Shape: cube, Color: blue, Material: metal."
    objs = parse_scene_description(text)
    assert len(objs) == 1
    assert objs[0] == SceneObject(x=5, y=7, shape="cube", color="blue", material="metal")


def test_single_object_case_insensitive():
    text = "OBJECT AT (3, 2): SHAPE: Sphere, COLOR: Red, MATERIAL: Rubber"
    objs = parse_scene_description(text)
    assert len(objs) == 1
    # Fields should be lowercased.
    assert objs[0].shape == "sphere"
    assert objs[0].color == "red"
    assert objs[0].material == "rubber"


def test_single_object_extra_whitespace():
    text = "Object at  ( 10 ,  20 ) :  Shape :  cylinder ,  Color :  green ,  Material :  rubber ."
    objs = parse_scene_description(text)
    assert len(objs) == 1
    assert objs[0] == SceneObject(x=10, y=20, shape="cylinder", color="green", material="rubber")


# ------------------------------------------------------ multi-object cases

def test_two_objects():
    text = (
        "Object at (5, 7): Shape: cube, Color: blue, Material: metal. "
        "Object at (12, 3): Shape: sphere, Color: red, Material: rubber."
    )
    objs = parse_scene_description(text)
    assert len(objs) == 2
    assert objs[0].shape == "cube"
    assert objs[1].shape == "sphere"


def test_three_objects_varied_format():
    text = (
        "Object at (1, 1): Shape: cube, Color: blue, Material: metal. "
        "Object at (2, 2): Shape: sphere, Color: red, Material: rubber "
        "Object at (3, 3): Shape: cylinder, Color: green, Material: metal."
    )
    objs = parse_scene_description(text)
    assert len(objs) == 3


# --------------------------------------------------------- edge cases

def test_empty_string():
    assert parse_scene_description("") == []


def test_whitespace_only():
    assert parse_scene_description("   \n\t  ") == []


def test_malformed_not_matched():
    # Missing one field -> parser should skip this object, not crash.
    text = "Object at (5, 7): Shape: cube, Color: blue."
    objs = parse_scene_description(text)
    assert objs == []


def test_negative_coordinates():
    text = "Object at (-5, -7): Shape: cube, Color: blue, Material: metal."
    objs = parse_scene_description(text)
    assert len(objs) == 1
    assert objs[0].x == -5 and objs[0].y == -7


# ------------------------------------- roundtrip via format_scene_description

def test_roundtrip_preserves_content():
    original = [
        SceneObject(x=1, y=2, shape="cube", color="blue", material="metal"),
        SceneObject(x=3, y=4, shape="sphere", color="red", material="rubber"),
    ]
    text = format_scene_description(original)
    parsed = parse_scene_description(text)
    assert parsed == original
