"""
CEGpy
=====

CEGpy is a python package for the creation, manipulation, and study of
Chain Event Graphs.

See documentation of more information.

"""
from .utilities.util import *
from .trees.event import *
from .trees.staged import *
from .graphs.ceg import *
import logging
# from graphs import ctceg


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('cegpy')
