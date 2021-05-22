from ..pairs_generators import tournament, inbriding, outbriding

from ..utils import Picker

methods_dict = {
    "tournament": tournament,
    "inbriding": inbriding,
    "outbriding": outbriding,
}

pairs_generation_politics = {
    "light": {
        "methods": Picker([(tournament, 0.8), (inbriding, 0.1), (outbriding, 0.1)]),
        "nbest": 5,
    },
    "kicker": {
        "methods": Picker([(tournament, 0.1), (outbriding, 0.9)]),
        "npairs": lambda self: len(self.population),
        "nbest": 0,
    },
}
