from collections import defaultdict
import pandas as pd
import numpy as np
from ceg_util import CegUtil as util


class EventTree(object):
	"""Creates event trees from pandas dataframe."""
	def __init__(self, params) -> None:
		# pandas dataframe passed via parameters 
		self.dataframe = params.get("dataframe")
		try:
			self.variables = list(self.dataframe.columns)
		except:
			print("No Dataframe provided")
			

		self.sampling_zero_paths = None
		# sampling zeros paths added manually via parameters 
		self.set_sampling_zero_paths(params.get('sampling_zero_paths'))

		# paths = self._create_path_dict_entries()
		# node_list = self._create_node_list_from_paths(paths)

		# Format of event_tree dict:
		# 
		# self.event_tree = self._construct_event_tree(paths, node_list)
		
		# print(self._create_initial_nodes())

	def get_sampling_zero_paths(self):
		return self.sampling_zero_paths

	def set_sampling_zero_paths(self, sz_paths):
		if sz_paths == None:
			self.sampling_zero_paths = None
		else:
			sz_paths = self._check_sampling_zero_paths_param(sz_paths)
			if sz_paths is None:
				error_str = "Parameter 'sampling_zero_paths' should be a list of tuples like so:\n \
				[('edge_1', 'edge_2', 'edge_3'), ('edge_4',), ...]"
				raise ValueError(error_str)
			else:
				self.sampling_zero_paths = None

	def _create_path_dict_entries(self) -> defaultdict(int):
		'''Create path dict entries for each path, including the sampling zero paths if any.
		Each path is an ordered sequence of edge labels starting from the root.
		The keys in the dict are ordered alphabetically.
		Also calls the method self.sampling zeros to ensure manually added path format is correct.
		Added functionality to remove NaN/null edge labels assuming they are structural zeroes'''

		self._dummy_paths = defaultdict(int)

		for variable_number in range(0, len(self.variables)):
			dataframe_upto_variable = self.dataframe.loc[:, self.variables[0:variable_number+1]]
			for row in dataframe_upto_variable.itertuples():
				row = row[1:]
				new_row = [edge_label for edge_label in row if edge_label != np.nan and
				str(edge_label) != 'NaN' and str(edge_label) != 'nan' and edge_label != '']
				new_row = tuple(new_row)

				#checking if the last edge label in row was nan. That would result in double counting
				#nan must be identified as string
				if  (row[-1] != np.nan and str(row[-1]) != 'NaN' and str(row[-1]) != 'nan' and row[-1] != ''):
					self._dummy_paths[new_row] += 1

		if self.sampling_zero_paths != None:
			self.sampling_zeros(self.sampling_zero_paths)	

		depth = len(max(list(self._dummy_paths.keys()), key=len))
		keys_of_list = list(self._dummy_paths.keys())
		sorted_keys = []
		for deep in range(0,depth+1):
			unsorted_mini_list = [key for key in keys_of_list if len(key) == deep]
			sorted_keys = sorted_keys + sorted(unsorted_mini_list)

		for key in sorted_keys:
			self.paths[key] = self._dummy_paths[key]

		return self.paths

		return defaultdict(int)

	def create_edges(self, edge_information) -> defaultdict(int):
		# _edge_creation
		return defaultdict(int)

	def _check_sampling_zero_paths_param(self, sampling_zero_paths) -> list[tuple]:
		"""Check param 'sampling_zero_paths' is in the correct format"""
		for tup in sampling_zero_paths:
			if not isinstance(tup, tuple):
				return None
			else:
				if not util.check_tuple_contains_strings(tup):
					return None
		
		return sampling_zero_paths

	def _create_node_list_from_paths(self, paths) -> list[str]:
		"""Creates list of all nodes: includes root, situations, leaves"""
		node_list = ['s0'] # root node

		for vertex_number,path in enumerate(list(paths.keys()), start=1):
			node_list.append('s%d' % vertex_number)

		return node_list

	def _construct_event_tree(self, paths, node_list) -> defaultdict:
		event_tree = defaultdict(int)
		return event_tree


if __name__ == "__main__":
	df = pd.read_excel('../data/CHDS.latentexample1.xlsx')
	et = EventTree({'dataframe' : df})
