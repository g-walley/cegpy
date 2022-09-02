"""
CEGpy
=====

CEGpy is a python package for the creation, manipulation, and study of
Chain Event Graphs.

See documentation of more information.

"""
import logging

from cegpy.graphs._ceg import ChainEventGraph
from cegpy.graphs._ceg_reducer import ChainEventGraphReducer
from cegpy.trees._event import EventTree
from cegpy.trees._staged import StagedTree

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("cegpy")

__version__ = "1.0.3"
