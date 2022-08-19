"""Utilities for use throughout the package."""

from collections import defaultdict
import os
from pathlib import Path
import math
from typing import List, Union
import colorutils
from colorutils import Color as Colour

ST_OUTPUT = {
    "Merged Situations": [],
    "Loglikelihood": float,
    "Mean Posterior Probabilities": [],
}


def check_list_contains_strings(str_list) -> bool:
    """Ensure that a list only contains strings"""
    for tup in str_list:
        if not isinstance(tup, str):
            return False
    return True


def check_tuple_contains_strings(tup) -> bool:
    """Check each element of the tuple to ensure it is a string"""
    if isinstance(tup, tuple):
        for elem in tup:
            if not isinstance(elem, str):
                return False
        return True
    else:
        return False


def generate_colours(number) -> list:
    """Generates a list of hex colour strings that are evenly spaced
    around the colour spectrum"""
    hex_colours = []
    low, start, high = 100, 150, 255
    starts = [
        Colour((start, low, low)),
        Colour((low, start, low)),
        Colour((low, low, start)),
    ]
    ends = [
        Colour((high, low, low)),
        Colour((low, high, low)),
        Colour((low, low, high)),
    ]
    prim_colours = math.ceil(number / 2)
    hex_colours += generate_colour_run(prim_colours, starts, ends)
    starts = [
        Colour((start, start, low)),
        Colour((low, start, start)),
        Colour((start, low, start)),
    ]
    ends = [
        Colour((high, high, low)),
        Colour((low, high, high)),
        Colour((high, low, high)),
    ]
    sec_colours = math.floor(number / 2)
    hex_colours += generate_colour_run(sec_colours, starts, ends)

    return hex_colours


def generate_colour_run(number, starts, ends) -> list:
    """Generates multiple colour runs with uniform distance between colours."""
    split = [0, 0, 0]
    for i in range(number):
        split[i % len(split)] += 1

    jumps = []
    for val in split:
        jumps.append(max(val - 1, 0))

    colours: List[Colour] = []

    for idx, end in enumerate(ends):
        if jumps[idx] > 0:
            colours += colorutils.color_run(starts[idx], end, jumps[idx], to_color=True)
        else:
            colours.append(end)
    hex_colours = []
    for colour in colours:
        hex_colours.append(colour.hex)

    return hex_colours


def generate_filename_and_mkdir(filename: Union[str, Path]) -> str:
    """Creates a filename."""
    if filename is not Path:
        filename = Path(filename)

    filetype = filename.suffix.strip(".")
    if filename.suffix == filetype:
        filename.joinpath(".png")
        filetype = "png"

    os.makedirs(filename.parent, exist_ok=True)
    return filename, filetype


def create_relative_path(relative_path) -> Path:
    """Creates pathlib object from a relative path such as:
    '/output/image.png'"""
    return Path(__file__).resolve().parent.joinpath(relative_path)


def create_sampling_zeros(sampling_zero_paths, path_dict) -> defaultdict:
    """The list of paths to zero must only contain tuples.
    Each tuple is a sampling zero path that needs to be added.
    If multiple edges along a path need to be added, they must be
    added in order. i.e. path[:-1] should already be a key in self.paths
    Eg suppose edge 'eat' already exists as an edge emanating from the
    root.
    To add paths ('eat', 'sleep'), ('eat', 'sleep', 'repeat'),
    the sampling zero parameter should be added as:
    et = ceg({'dataframe': df, 'sampling_zero_paths':
    [('eat', 'sleep',),('eat', 'sleep','repeat',)]})
    """
    for path in sampling_zero_paths:
        if (path[:-1] in list(path_dict.keys())) or len(path) == 1:
            path_dict[path] = 0
        else:
            raise ValueError(
                "The path up to it's last edge should be \
                added first. Ensure the tuple ends with a comma."
            )

    return path_dict
