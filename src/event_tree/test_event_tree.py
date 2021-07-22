from event_tree import EventTree
from collections import defaultdict
import pandas as pd
from ceg_util import CegUtil as util
import event_tree as et
import os

class TestEventTree():
	def setup(self):
		df_path = "%s/data/CHDS.latentexample1.xlsx" % util.get_package_root()

		self.df = pd.read_excel(df_path)
		self.et = EventTree({'dataframe' : self.df})
	
	def test_check_sampling_zero_paths_param(self) -> None:
		"""Tests the function that is checking the sampling zero paths param"""
		szp = [('Medium',),('Medium', 'High')]
		assert self.et._check_sampling_zero_paths_param(szp) == szp
		
		szp = [1,2,3,4]
		assert self.et._check_sampling_zero_paths_param(szp) == None

		szp = [('path', 'to'), (123, 'something'), 'path/to']
		assert self.et._check_sampling_zero_paths_param(szp) == None

	def test_check_sampling_zero_get_and_set(self) -> None:
		assert self.et.get_sampling_zero_paths() == None

		szp = [('Medium',),('Medium', 'High')]
		self.et._set_sampling_zero_paths(szp)
		assert self.et.get_sampling_zero_paths() == szp

	
	def test_create_node_list_from_paths(self) -> None:
		paths = defaultdict(int)
		paths[('path',)] += 1
		paths[('path','to')] += 1
		paths[('path','away')] += 1
		paths[('road',)] += 1
		paths[('road', 'to')] += 1
		paths[('road', 'away')] += 1

		# code being tested:
		node_list = self.et._create_node_list_from_paths(paths)

		print(node_list)
		assert len(list(paths.keys())) + 1 == len(node_list)
		assert node_list[0] == 's0'
		assert node_list[-1] == 's%d' % (len(node_list) - 1)

	def test_construct_event_tree(self) -> None:
		"""Tests the construction of an event tree from a set of paths, nodes, and ..."""
		event_tree = self.et._construct_event_tree()
		assert isinstance(event_tree, defaultdict)
		for key, value in event_tree.items():
			assert isinstance(key, tuple)
			for elem in key:
				assert isinstance(elem, tuple)
				
				for sub_elem in elem[0]:
					assert isinstance(sub_elem, str)
			
			# Check the edge data in the key is exactly 2 long.
			assert len(key[1]) == 2 

				
			
