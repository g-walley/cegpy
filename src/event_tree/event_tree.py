from collections import defaultdict
import pandas as pd
import numpy as np

def create_edges(edge_information) -> defaultdict(int):
	# _edge_creation
	return defaultdict(int)

def check_list_contains_strings(str_list) -> bool:
	for tup in str_list:
		if not isinstance(tup, str):
			return False
	
	return True

def check_sampling_zero_paths_param(sampling_zero_paths) -> list[tuple]:
	"""Check param 'sampling_zero_paths' is in the correct format"""
	for tup in sampling_zero_paths:
		if not isinstance(tup, tuple):
			return None
		else:
			if not check_tuple_contains_strings(tup):
				return None
	
	return sampling_zero_paths

def check_tuple_contains_strings(tup) -> bool:
	"""Check each element of the tuple to ensure it is a string""" 
	for elem in tup:
		if not isinstance(elem, str):
			return False
	return True

class EventTree(object):
	"""Creates event trees from pandas dataframe."""
	def __init__(self, params) -> None:
		# pandas dataframe passed via parameters 
		self.dataframe = params.get("dataframe")
		self.variables = list(self.dataframe.columns)
		# sampling zeros paths added manually via parameters 
		self.sampling_zero_paths = self.get_sampling_zero_paths(params)

		# print(self._create_initial_nodes())

	def get_sampling_zero_paths(self, params) -> list[tuple]:
		sz_paths = params.get('sampling_zero_paths')
		if sz_paths == None:
			return None
		else:
			sz_paths = check_sampling_zero_paths_param(sz_paths)
			if sz_paths is None:
				error_str = "Parameter 'sampling_zero_paths' should be a list of tuples like so:\n \
				[('edge_1', 'edge_2', 'edge_3'), ('edge_4',), ...]"
				raise ValueError(error_str)
			else:
				return sz_paths
			

	def _create_initial_nodes(self) -> list[str]:
		"""Creates list of all nodes: includes root, situations, leaves"""
		# _nodes()
		
		# Where dictionary of paths does not contain any data,
		# create it with _create_path_dict_entries()
		if len(list(self.paths.keys())) == 0:
			self._create_path_dict_entries()
		
		node_list = ['s0'] # root node
		vertex_number = 1
		for path in list(self.paths.keys()):
			node_list.append('s%d' % vertex_number)
			vertex_number += 1

		return node_list

	def _create_path_dict_entries(self) -> defaultdict(int):
		'''Create path dict entries for each path, including the sampling zero paths if any.
		Each path is an ordered sequence of edge labels starting from the root.
		The keys in the dict are ordered alphabetically.
		Also calls the method self.sampling zeros to ensure manually added path format is correct.
		Added functionality to remove NaN/null edge labels assuming they are structural zeroes'''
		return defaultdict(int)



if __name__ == "__main__":
	df = pd.read_excel('../data/CHDS.latentexample1.xlsx')
	et = EventTree({'dataframe' : df})
