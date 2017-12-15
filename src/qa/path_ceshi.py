import os, sys
parent_dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dirpath)
from amq.sim import BenebotSim