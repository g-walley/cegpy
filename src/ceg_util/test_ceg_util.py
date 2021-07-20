from ceg_util import CegUtil as util

def test_check_list_contains_strings() -> None:
	str_list = ['blah', 'path/to/node', '123', '~$%^']
	non_str_list = [123, 'path/to/node', True]

	assert util.check_list_contains_strings(str_list) == True
	assert util.check_list_contains_strings(non_str_list) == False

def test_check_tuple_contains_strings() -> None:
	"""Tests the function that is checking that a tuple only contains strings"""
	tup = ('string thing',)
	assert util.check_tuple_contains_strings(tup) == True
	tup = ('string one', 'string two')
	assert util.check_tuple_contains_strings(tup) == True
	tup = (1, '2')
	assert util.check_tuple_contains_strings(tup) == False