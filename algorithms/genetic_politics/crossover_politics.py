from ..crossover import pmx_crossover, cycle_crossover
from ..utils import Picker

from .utils import politics_template, set_methods

methods_dict = {"pmx_crossover": pmx_crossover, "cycle_crossover": cycle_crossover}

__crossover_politics = set_methods(politics_template, methods_dict)


class crossover_politics(metaclass=__crossover_politics):
    pass


crossover_pc_politics = {
    "basic": {"methods": Picker([(pmx_crossover, 0.95), (cycle_crossover, 0.05)])},
    "kicker": {"methods": Picker([(pmx_crossover, 0.8), (cycle_crossover, 0.2)])}
}
