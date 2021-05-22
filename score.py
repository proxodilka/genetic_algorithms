from utils import TravellingSalesmanDataSets, calc_path_length
import numpy as np

weights = TravellingSalesmanDataSets.bier127

with open("test_v.txt", "r") as file:
    vec = np.array(list(map(int, file.read().splitlines()))) - 1

print(calc_path_length(weights, vec, is_pure_perm=True))

breakpoint()
