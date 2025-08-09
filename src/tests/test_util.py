"""Tests cegpy.utilities"""
from collections import defaultdict
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from cegpy.utilities import (
    check_list_contains_strings,
    check_tuple_contains_strings,
    create_sampling_zeros,
    generate_filename_and_mkdir,
)


class TestCegUtil(unittest.TestCase):
    """Tests util functions"""

    def setUp(self):
        self.sampling_zero_list = [("path", "to"), ("road", "from", "nowhere")]
        self.sample_path_dict = defaultdict(int)
        self.sample_path_dict[("path",)] += 3
        self.sample_path_dict[("path", "away")] += 2
        self.sample_path_dict[("road",)] += 3
        self.sample_path_dict[("road", "from")] += 1
        self.sample_path_dict[("road", "to")] += 2
        self.sample_path_dict[("road", "from", "somewhere")] += 1
        self.sample_path_dict[("road", "to", "somewhere")] += 1
        self.sample_path_dict[("road", "to", "nowhere")] += 1

    def test_check_list_contains_strings(self) -> None:
        """Tests function check_list_contains_strings().
        Passes two lists. One that only contains strings, and another
        that contains a multitude of types."""

        str_list = ["blah", "path/to/node", "123", "~$%^"]
        non_str_list = [123, "path/to/node", True]

        self.assertTrue(check_list_contains_strings(str_list))
        self.assertFalse(check_list_contains_strings(non_str_list))

    def test_create_sampling_zeros(self) -> None:
        """Tests function create_sampling_zeros().
        Passes the new sampling zero list of tree path tuples, and a sample
        dictionary that does not contain those paths. It then checks that the
        keys exist, and that their values are zero."""
        test_path_dict = create_sampling_zeros(
            self.sampling_zero_list, self.sample_path_dict
        )

        for key in self.sampling_zero_list:
            self.assertIn(key, list(test_path_dict.keys()))
            self.assertEqual(test_path_dict[key], 0)

    def test_check_tuple_contains_strings(self) -> None:
        """Tests function check_tuple_contains_strings().
        Passes various tuples that contain different types,
        and verifies output"""
        self.assertTrue(check_tuple_contains_strings(("string thing",)))
        self.assertFalse(check_tuple_contains_strings(("string thing")))
        self.assertTrue(check_tuple_contains_strings(("string one", "string two")))
        self.assertFalse(check_tuple_contains_strings((1, "2")))

    def test_generate_filename_and_mkdir(self) -> None:
        """Ensure filenames are normalised and directories created."""
        with TemporaryDirectory() as tmpdir:
            path_no_ext = Path(tmpdir) / "out" / "fig"
            filename, filetype = generate_filename_and_mkdir(str(path_no_ext))
            self.assertTrue(filename.parent.exists())  # directory exists
            self.assertEqual(filename.suffix, ".png")
            self.assertEqual(filetype, "png")

            path_with_ext = Path(tmpdir) / "out2" / "fig.svg"
            filename2, filetype2 = generate_filename_and_mkdir(path_with_ext)
            self.assertTrue(filename2.parent.exists())
            self.assertEqual(filename2.suffix, ".svg")
            self.assertEqual(filetype2, "svg")
