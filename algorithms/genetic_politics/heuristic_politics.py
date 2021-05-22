from ..heuristic_methods import MonteCarlo, NearestCity, DFS
from ..utils import Picker, rest

from .utils import politics_template, set_methods

methods_dict = {
    "MonteCarlo": MonteCarlo,
    "NearestCity": NearestCity,
    "DFS": DFS,
}
greedy_names = ["NearestCity"]

__heuristic_politics = set_methods(politics_template, methods_dict, greedy_names)


class heuristic_politics(metaclass=__heuristic_politics):
    pass
