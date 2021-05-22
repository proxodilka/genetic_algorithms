from ..utils import Picker, rest


def set_methods(cls, methods_dict, greedy_names=None):
    if greedy_names is None:
        greedy_names = []
    cls = type("CLSCopy", cls.__bases__, dict(cls.__dict__))
    cls.methods_dict = methods_dict
    cls.greedy_names = greedy_names
    return cls


class politics_template(type):
    def __getitem__(cls, keys):
        if not isinstance(keys, list):
            keys = [keys]
        is_greedy = any(name in keys for name in cls.greedy_names)
        return {
            "methods": Picker(
                [
                    (cls.methods_dict[key], rest if key not in cls.greedy_names else 1)
                    for key in keys
                ],
                absolute=is_greedy,
            )
        }
