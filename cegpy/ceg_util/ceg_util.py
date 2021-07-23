import os
from collections import defaultdict


class CegUtil:
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

    def get_package_root() -> str:
        """Finds the root directory of the package."""
        cwd = os.getcwd()
        return cwd[:cwd.index('pyceg') + len('pyceg')].replace("\\", "/")

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
