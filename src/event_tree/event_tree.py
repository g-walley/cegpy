from collections import defaultdict
from typing import OrderedDict
import pandas as pd
import numpy as np
import pydotplus as pdp
from ceg_util import CegUtil as util
from IPython.display import Image
from pathlib import Path


class EventTree(object):
	"""Creates event trees from pandas dataframe."""
	def __init__(self, params) -> None:
		self.params = params
		self.sampling_zero_paths = None
		self.node_list = []
		# Paths taken from dataframe in order of occurance
		self.unsorted_paths = defaultdict(int)
		# Paths sorted alphabetically in order of length 
		self.sorted_paths = defaultdict(int)

		# pandas dataframe passed via parameters 
		self.dataframe = params.get("dataframe")
		try:
			self.variables = list(self.dataframe.columns)
		except:
			print("No Dataframe provided")

		# Format of event_tree dict:
		# 
		self.event_tree = self._construct_event_tree()

	def get_sampling_zero_paths(self):
		if not self.sampling_zero_paths:
			print("EventTree.get_sampling_zero_paths() called but no paths have been set.")
		
		return self.sampling_zero_paths

	def _set_sampling_zero_paths(self, sz_paths):
		"""Use this function to set the sampling zero paths.
		If different to previous value, will re-generate the event tree."""
		old_sz_paths = self.sampling_zero_paths

		if sz_paths == None:
			self.sampling_zero_paths = None
		else:
			# checkes if the user has inputted sz paths correctly
			sz_paths = self._check_sampling_zero_paths_param(sz_paths)

			if sz_paths:
				self.sampling_zero_paths = sz_paths
			else:
				error_str = "Parameter 'sampling_zero_paths' should be a list of tuples like so:\n \
				[('edge_1', 'edge_2'), ('edge_1',), ...]"
				raise ValueError(error_str)

	def _create_unsorted_paths_dict(self) -> defaultdict:
		"""Creates and populates a dictionary of all paths provided in the dataframe,
		in the order in which they are given."""
		unsorted_paths = defaultdict(int)

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
					unsorted_paths[new_row] += 1
		
		return unsorted_paths

	def _create_path_dict_entries(self):
		'''Create path dict entries for each path, including the sampling zero paths if any.
		Each path is an ordered sequence of edge labels starting from the root.
		The keys in the dict are ordered alphabetically.
		Also calls the method self.sampling zeros to ensure manu
		ally added path format is correct.
		Added functionality to remove NaN/null edge labels assuming they are structural zeroes'''

		self.unsorted_paths = self._create_unsorted_paths_dict()

		if self.sampling_zero_paths != None:
			self.unsorted_paths = util.create_sampling_zeros(self.sampling_zero_paths, self.unsorted_paths)	

		depth = len(max(list(self.unsorted_paths.keys()), key=len))
		keys_of_list = list(self.unsorted_paths.keys())
		sorted_keys = []
		for deep in range(0,depth+1):
			unsorted_mini_list = [key for key in keys_of_list if len(key) == deep]
			sorted_keys = sorted_keys + sorted(unsorted_mini_list)

		for key in sorted_keys:
			self.sorted_paths[key] = self.unsorted_paths[key]

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
	
	def get_node_list(self) -> list:
		"""Returns node list"""
		return self.node_list

	def _construct_event_tree(self) -> defaultdict:
		"""Constructs event_tree dictionary.
		Format of the dictionary is: 
		{ 			
			key:	(('path','to','leaf'), ('s4','s5')):
			value:	<Edge counts>
		}
		
		A key constructed from a tuple containing a tuple(path to leaf),
		and a tuple(eminating node, terminating node)."""

		self._set_sampling_zero_paths(self.params.get('sampling_zero_paths'))
		self._create_path_dict_entries()
		node_list = self._create_node_list_from_paths(self.sorted_paths)
		self.node_list = node_list

		# sampling zeros paths added manually via parameters 
		event_tree = defaultdict(int)

		# Work through the sorted paths list to build the event tree.
		edge_labels_list = ['root']
		edges_list = []
		for path in list(self.sorted_paths.keys()):
			path = list(path)
			edge_labels_list.append(path)
			if path[:-1] in edge_labels_list:
				path_edge_comes_from = edge_labels_list.index(path[:-1])
				edges_list.append([node_list[path_edge_comes_from], node_list[edge_labels_list.index(path)]])
			else:
				edges_list.append([node_list[0], node_list[edge_labels_list.index(path)]])
			event_tree[((*path,),(*edges_list[-1],))] = self.sorted_paths[tuple(path)]
		
		return event_tree

	def get_edge_labels(self) -> list:
		"""Once event tree dict has been populated, a list of all 
		edge labels can be obtained with this function"""
		if self.event_tree:
			return [key[0] for key in list(self.event_tree.keys())] 
		else:
			return []
			
	def get_edges(self) -> list:
		"""Once event tree dict has been populated, a list of all 
		edges can be obtained with this function"""
		if self.event_tree:
			return [key[1] for key in list(self.event_tree.keys())]
		else:
			return [] 

	def create_figure(self, filename):
		"""Draws the event tree for the process described by the dataset,
		and saves it to <filename>.png"""
		event_tree_graph = pdp.Dot(graph_type = 'digraph', rankdir = 'LR')

		for key, count in self.event_tree.items(): 
			# edge_index = self.edges.index(edge)
			path = key[0]
			edge = key[1]
			edge_details = str(path[-1]) + '\n' + str(count)

			event_tree_graph.add_edge(
				pdp.Edge(
					edge[0], 
					edge[1], 
					label = edge_details, 
					labelfontcolor="#009933", 
					fontsize="10.0", 
					color="black" 
				)
			)

		for node in self.node_list:
			event_tree_graph.add_node(pdp.Node(name = node, label = node, style = "filled"))

		event_tree_graph.write_png(str(filename) + '.png')
		
		return Image(event_tree_graph.create_png())
