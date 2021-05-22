import numpy as np
from .utils import SEED

# np.random.seed(SEED.value)


class BasePopulation(np.ndarray):
    def __init__(self, individuals):
        self._individuals = individuals
