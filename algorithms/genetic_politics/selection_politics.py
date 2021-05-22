from ..population_updaters import wheel, tournament
from ..utils import Picker

from .utils import politics_template, set_methods


methods_dict = {
    "wheel": wheel,
    "tournament": tournament,
}

__selection_politics = set_methods(politics_template, methods_dict)


class selection_politics(metaclass=__selection_politics):
    pass


selection_politics_dict = {
    "light": {
        "methods": Picker([(tournament, 1)]),
    },
    "kicker": {
        "methods": Picker([(tournament, 0.5), (wheel, 0.5)]),
        "consider_weights": False,
        "include_parents": False,
        "nfighters": 8
    },
}
