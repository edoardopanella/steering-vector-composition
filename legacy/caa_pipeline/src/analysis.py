import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import pandas as pd

from steer_vec_loader import SteerVecLoader
from gram_matrix import compute_gram_matrix
from pair_strat import get_offdiag, get_all_pairs, stratify_pairs

PATH = "/Users/federicoscaffidimuta/Desktop/Third year/ML project/steering-vector-composition/results/vectors"

loader = SteerVecLoader(PATH, layer=13)
loader.load_steer_vecs()
behavior_vecs, behaviors = loader.to_matrix()