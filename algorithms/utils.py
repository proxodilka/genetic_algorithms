# from algorithms.evolution.algorithm import BaseEvolutionAlgo
import os


class SEED:
    value = 0


# SEED = os.environ.get("GA_SEED", 0)

import numpy as np

# np.random.seed(SEED)
import datetime

from pathlib import Path
from dataclasses import dataclass


def np_array(value):
    if isinstance(value, np.ndarray):
        return value
    a = np.empty(len(value), dtype=object)
    a[:] = value
    return a


def normalize(arr, reverse=False):
    """
    Normalize values of passed array

    Parameters
    ----------
    arr: Iterable,
        Array to normalize.

    Returns
    -------
    Normalized np.ndarray
    """
    if len(arr) == 0:
        return arr
    if reverse:
        arr = np.divide(1, arr)
    arr_sum = np.sum(arr)
    if arr_sum == 0:
        return np.repeat(1 / len(arr), len(arr))
    return np.divide(arr, arr_sum)


def static_vars(**kwargs):
    def wrapper(func):
        for key, value in kwargs.items():
            setattr(func, key, value)
        return func

    return wrapper


class DummyLogger:
    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass


class Logger:
    DEFAULT_LOG_DIR = Path.home()

    def __init__(self, filename=None, columns=None):
        self.storage = []
        self.filename = self.get_filename(filename)
        self.columns = columns

    def log(self, *args, **kwargs):
        kwargs.pop("population")
        if self.columns is None:
            self.columns = list(kwargs.keys())
        self.storage.append(list(args) + list(kwargs.values()))

    def values(self):
        return self.storage.__iter__()

    def write(self, *args, **kwargs):
        try:
            with open(self.filename, "w") as file:
                if self.columns is not None:
                    file.write(",".join(self.columns) + "\n")
                for item in self.values():
                    to_write = ",".join(map(str, item)) + "\n"
                    file.write(to_write)
        except Exception as e:
            # We could also crash while printing the error message
            try:
                print(
                    f"\n\tSome terrible error occured: I can't write my logs to the file because of that exception:\n\t\t{e}\n\n"
                    + f"\tTo prevent data losage entering into a debug mode, where you can extract data manualy.\n"
                    + f"\tHint: data located at `self.storage`"
                )
            except:
                pass
            finally:
                breakpoint()

    @staticmethod
    def default_log_dir():
        log_dir = Path.home().joinpath("Documents").joinpath("EGA_LOGS")
        log_dir.mkdir(exist_ok=True)
        return log_dir

    @classmethod
    def get_filename(cls, filename=None):
        log_dir = Path(os.environ.get("EGA_LOG_DIR", cls.default_log_dir()))
        if filename is None:
            filename = (
                str(datetime.datetime.now()).replace(":", "")
                + f"_id{np.random.randint(99999)}"
                + ".txt"
            )
        return log_dir.joinpath(filename)


class StatisticLogger(Logger):
    def __init__(self, *args, **kwargs):
        Logger.__init__(self, *args, **kwargs)
        self.unique = np.array([])
        self.storage = []

    def log(self, **kwargs):
        population = kwargs.pop("population")

        if self.columns is None:
            self.columns = list(kwargs.keys()) + ["diversity", "exploration"]

        unique_individs = np.unique([individ.gens for individ in population], axis=0)

        if self.unique.shape[0] != 0:
            try:
                all_unique = np.concatenate([self.unique, unique_individs], axis=0)
            except:
                breakpoint()
        else:
            all_unique = unique_individs

        self.unique = np.unique(all_unique, axis=0)
        diversity = self.get_diversity(population)

        self.storage.append(
            {
                **kwargs,
                "diversity": diversity,
                "exploration": self.unique.shape[0],
            }
        )

    def values(self):
        values_iter = self.storage.__iter__()

        def iterator():
            i = 0
            while True:
                yield list(next(values_iter).values())
                i += 1
                if i == len(self.storage):
                    break

        return iterator()

    def get_diversity(self, population):
        nunique = np.unique([individ.gens for individ in population], axis=0).shape[0]
        return nunique / population.shape[0]


class Picker:
    def __init__(self, variants, absolute=False):
        self.variants = []
        # breakpoint()
        self.total_sum = int(np.sum([o[1] for o in variants]))
        self.rest_count = len([o for o in variants if o[1] is rest])
        self.variants = [Variant(*variant) for variant in variants]
        self.absolute = absolute

    def compute_rates(self, n=None):
        assert (
            not self.absolute or n is not None
        ), "Must specify `n` in case of absolute probabilities."
        # breakpoint()
        if self.rest_count > 0:
            full_probability = n if self.absolute else 1
            rest_sum = full_probability - self.total_sum
            rest_probability = rest_sum / self.rest_count
            for i in range(len(self.variants)):
                if self.variants[i].probability is rest:
                    self.variants[i] = Variant(
                        self.variants[i].variant, rest_probability
                    )
        self.normalize_probabilities()

    def normalize_probabilities(self):
        if not self.absolute:
            return
        self.total_sum = np.sum([o.probability for o in self.variants])
        for i in range(len(self.variants)):
            self.variants[i] = Variant(
                self.variants[i].variant, self.variants[i].probability / self.total_sum
            )

    def items(self, n=None):
        self.compute_rates(n)
        return self.__iter__()

    def __iter__(self):
        return iter(self.variants)

    def __getitem__(self, key):
        return self.variants[key]


@dataclass
class Variant:
    variant: object
    probability: float


class __empty_int(type):
    def __add__(cls, other):
        return other

    def __radd__(cls, other):
        return other

    def __int__(cls):
        return 0

    def __float__(cls):
        return 0.0


class rest(metaclass=__empty_int):
    pass
