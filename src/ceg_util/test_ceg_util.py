from collections import defaultdict
from ceg_util import CegUtil as util

sampling_zero_list = [('path', 'to'), ('road', 'from', 'nowhere')]
sample_path_dict = defaultdict(int)
sample_path_dict[('path',)] += 3
sample_path_dict[('path', 'away')] += 2
sample_path_dict[('road',)] += 3
sample_path_dict[('road', 'from')] += 1
sample_path_dict[('road', 'to')] += 2
sample_path_dict[('road', 'from', 'somewhere')] += 1
sample_path_dict[('road', 'to', 'somewhere')] += 1
sample_path_dict[('road', 'to', 'nowhere')] += 1


def test_check_list_contains_strings() -> None:
    """Tests function util.check_list_contains_strings().
    Passes two lists. One that only contains strings, and another
    that contains a multitude of types."""

    str_list = ['blah', 'path/to/node', '123', '~$%^']
    non_str_list = [123, 'path/to/node', True]

    assert util.check_list_contains_strings(str_list) is True
    assert util.check_list_contains_strings(non_str_list) is False


def test_create_sampling_zeros() -> None:
    """Tests function util.create_sampling_zeros().
    Passes the new sampling zero list of tree path tuples, and a sample
    dictionary that does not contain those paths. It then checks that the
    keys exist, and that their values are zero."""
    test_path_dict = util.create_sampling_zeros(
        sampling_zero_list,
        sample_path_dict
        )

    for key in sampling_zero_list:
        assert key in list(test_path_dict.keys())
        assert test_path_dict[key] == 0


def test_check_tuple_contains_strings() -> None:
    """Tests function util.check_tuple_contains_strings().
    Passes various tuples that contain different types, and verifies output"""
    tup = ('string thing',)
    assert util.check_tuple_contains_strings(tup) is True
    tup = ('string thing')  # Not a tuple, missing comma
    assert util.check_tuple_contains_strings(tup) is False
    tup = ('string one', 'string two')
    assert util.check_tuple_contains_strings(tup) is True
    tup = (1, '2')
    assert util.check_tuple_contains_strings(tup) is False
