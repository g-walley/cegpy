"""
CEGpy
=====

CEGpy is a python package for the creation, manipulation, and study of
Chain Event Graphs.

See documentation of more information.

"""
import logging

from cegpy.graphs._ceg import ChainEventGraph
from cegpy.graphs._reduced_ceg import ReducedChainEventGraph
from cegpy.trees._event import EventTree
from cegpy.trees._staged import StagedTree

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("cegpy")
