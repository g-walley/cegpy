from collections import defaultdict
from ..src.cegpy.utilities.util import Util


class TestCegUtil(object):

    def setup(self):
        self.sampling_zero_list = [('path', 'to'), ('road', 'from', 'nowhere')]
        self.sample_path_dict = defaultdict(int)
        self.sample_path_dict[('path',)] += 3
        self.sample_path_dict[('path', 'away')] += 2
        self.sample_path_dict[('road',)] += 3
        self.sample_path_dict[('road', 'from')] += 1
        self.sample_path_dict[('road', 'to')] += 2
        self.sample_path_dict[('road', 'from', 'somewhere')] += 1
        self.sample_path_dict[('road', 'to', 'somewhere')] += 1
        self.sample_path_dict[('road', 'to', 'nowhere')] += 1

    def test_check_list_contains_strings(self) -> None:
        """Tests function util.check_list_contains_strings().
        Passes two lists. One that only contains strings, and another
        that contains a multitude of types."""

        str_list = ['blah', 'path/to/node', '123', '~$%^']
        non_str_list = [123, 'path/to/node', True]

        assert Util.check_list_contains_strings(str_list) is True
        assert Util.check_list_contains_strings(non_str_list) is False

    def test_create_sampling_zeros(self) -> None:
        """Tests function Util.create_sampling_zeros().
        Passes the new sampling zero list of tree path tuples, and a sample
        dictionary that does not contain those paths. It then checks that the
        keys exist, and that their values are zero."""
        test_path_dict = Util.create_sampling_zeros(
            self.sampling_zero_list,
            self.sample_path_dict
            )

        for key in self.sampling_zero_list:
            assert key in list(test_path_dict.keys())
            assert test_path_dict[key] == 0

    def test_check_tuple_contains_strings(self) -> None:
        """Tests function Util.check_tuple_contains_strings().
        Passes various tuples that contain different types,
        and verifies output"""
        tup = ('string thing',)
        assert Util.check_tuple_contains_strings(tup) is True
        tup = ('string thing')  # Not a tuple, missing comma
        assert Util.check_tuple_contains_strings(tup) is False
        tup = ('string one', 'string two')
        assert Util.check_tuple_contains_strings(tup) is True
        tup = (1, '2')
        assert Util.check_tuple_contains_strings(tup) is False

    # def test_generate_colours(self) -> None:
    #     high = 6
    #     for cnt in range(1, high+1):
    #         print("%d:" % cnt)
    #         print(Util.generate_colours(cnt))
    #     assert 1==2
