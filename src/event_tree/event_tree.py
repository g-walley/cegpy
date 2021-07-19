from collections import defaultdict
import numpy as np

def create_edges(edge_information) -> defaultdict(int):
	# _edge_creation
	return defaultdict(int)


class EventTree(object):

	"""Creates event trees from panda dataframes."""
	def __init__(self, params) -> None:
		pass

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
	pass
