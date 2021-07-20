import os


class CegUtil:

	def check_list_contains_strings(str_list) -> bool:
		"""Ensure that a list only contains strings"""
		for tup in str_list:
			if not isinstance(tup, str):
				return False
		return True

	def check_tuple_contains_strings(tup) -> bool:
		"""Check each element of the tuple to ensure it is a string""" 
		for elem in tup:
			if not isinstance(elem, str):
				return False
		return True

	def get_package_root() -> str: 
		cwd = os.getcwd()
		return cwd[:cwd.index('pyceg') + len('pyceg')].replace("\\", "/")