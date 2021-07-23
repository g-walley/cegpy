import logging
# import sys
# from pathlib import Path
# import event_tree
# import ceg_util
# import staged_tree
# from ceg_util import CegUtil
# from event_tree import EventTree
# from staged_tree import StagedTree
# from chain_event_graph import ChainEventGraph
# from ct_chain_event_graph import CTChainEventGraph

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('pyceg')
# list_of_modules = [
#     'ceg_util',
#     'chain_event_graph',
#     'ct_chain_event_graph',
#     'staged_tree',
#     'event_tree'
# ]

# for mod in list_of_modules:
#     path = str(Path(__file__).resolve().parent.joinpath(mod))
#     sys.path.append(path)

# sys_path = sys.path
# logger.debug("sys path: %s" % sys_path)
