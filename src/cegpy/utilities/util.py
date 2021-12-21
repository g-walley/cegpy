import os
import colorutils
import math
from collections import defaultdict
from pathlib import Path
from colorutils import Color as Colour
from datetime import datetime

ST_OUTPUT = {
    "Merged Situations": [],
    "Loglikelihood": float,
    "Mean Posterior Probabilities": [],
}


class Util:
    @staticmethod
    def check_list_contains_strings(str_list) -> bool:
        """Ensure that a list only contains strings"""
        for tup in str_list:
            if not isinstance(tup, str):
                return False
        return True

    @staticmethod
    def check_tuple_contains_strings(tup) -> bool:
        """Check each element of the tuple to ensure it is a string"""
        if isinstance(tup, tuple):
            for elem in tup:
                if not isinstance(elem, str):
                    return False
            return True
        else:
            return False

    @staticmethod
    def generate_colours(number) -> list:
        """Generates a list of hex colour strings that are evenly spaced
        around the colour spectrum"""
        hex_colours = []
        low, start, high = 100, 150, 255
        starts = [
            Colour((start, low, low)),
            Colour((low, start, low)),
            Colour((low, low, start))
        ]
        ends = [
            Colour((high, low, low)),
            Colour((low, high, low)),
            Colour((low, low, high))
        ]
        prim_colours = math.ceil(number / 2)
        hex_colours += Util.generate_colour_run(prim_colours, starts, ends)
        starts = [
            Colour((start, start, low)),
            Colour((low, start, start)),
            Colour((start, low, start))
        ]
        ends = [
            Colour((high, high, low)),
            Colour((low, high, high)),
            Colour((high, low, high))
        ]
        sec_colours = math.floor(number / 2)
        hex_colours += Util.generate_colour_run(sec_colours, starts, ends)

        return hex_colours

    @staticmethod
    def generate_colour_run(number, starts, ends) -> list:
        split = [0, 0, 0]
        for i in range(number):
            split[i % len(split)] += 1

        jumps = []
        for val in split:
            jumps.append(max(val - 1, 0))

        colours = []

        for idx in range(len(ends)):
            if jumps[idx] > 0:
                colours += colorutils.color_run(
                    starts[idx], ends[idx], jumps[idx], to_color=True)
            else:
                colours.append(ends[idx])
        hex_colours = []
        for colour in colours:
            hex_colours.append(colour.hex)

        return hex_colours

    @staticmethod
    def generate_filename_and_mkdir(filename) -> str:
        if filename is not Path:
            filename = Path(filename)

        filetype = filename.suffix.strip('.')
        if filename.suffix == filetype:
            filename.joinpath('.png')
            filetype = 'png'

        os.makedirs(filename.parent, exist_ok=True)
        return filename, filetype

    @staticmethod
    def create_relative_path(relative_path) -> Path:
        """Creates pathlib object from a relative path such as:
        '/output/image.png'"""
        return Path(__file__).resolve().parent.joinpath(relative_path)

    @staticmethod
    def create_sampling_zeros(sampling_zero_paths, path_dict) -> defaultdict:
        '''The list of paths to zero must only contain tuples.
        Each tuple is a sampling zero path that needs to be added.
        If multiple edges along a path need to be added, they must be
        added in order. i.e. path[:-1] should already be a key in self.paths
        Eg suppose edge 'eat' already exists as an edge emanating from the
        root.
        To add paths ('eat', 'sleep'), ('eat', 'sleep', 'repeat'),
        the sampling zero parameter should be added as:
        et = ceg({'dataframe': df, 'sampling_zero_paths':
        [('eat', 'sleep',),('eat', 'sleep','repeat',)]})
        '''
        for path in sampling_zero_paths:
            if (path[:-1] in list(path_dict.keys())) or len(path) == 1:
                path_dict[path] = 0
            else:
                raise ValueError("The path up to it's last edge should be \
                    added first. Ensure the tuple ends with a comma.")

        return path_dict

    @staticmethod
    def create_path(filename, add_time=False, filetype='png'):
        dt_string = ''
        if add_time:
            now = datetime.now()
            dt_string = now.strftime("__%d-%m-%Y_%H-%M-%S")
        fig_path = Path(__file__).resolve().parent\
            .parent.parent.parent.joinpath(
                filename + dt_string + '.' + filetype)
        return fig_path

    @staticmethod
    def flatten_list_of_lists(list_of_lists) -> list:
        flat_list = []
        for sublist in list_of_lists:
            flat_list = flat_list + sublist
        return flat_list
