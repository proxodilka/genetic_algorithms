import numpy as np
import os
import string
import random

DATASETS_FOLDER = os.environ.get("DATASETS_PATH", os.path.abspath("datasets"))


def path(path, ext):
    return f"{DATASETS_FOLDER}/{path}.{ext}"


class ReaderMeta(type):
    def __getattr__(cls, key):
        if key in dir(cls):
            return object.__getattribute__(cls, key)
        if os.path.isfile(path(key, "txt")):
            return cls.read(path(key, "txt"))
        elif os.path.isfile(path(key, "csv")):
            return cls.read(path(key, "csv"), sep=",")
        else:
            return object.__getattribute__(cls, key)


class BaseDataSet(metaclass=ReaderMeta):
    @staticmethod
    def read(filepath, sep=" "):
        raise NotImplementedError()

    @staticmethod
    def random_dataset(*args, **kwargs):
        raise NotImplementedError()


class TravellingSalesmanDataSets(BaseDataSet):
    @classmethod
    def read(cls, filepath, sep=" "):
        with open(filepath, "r") as file:
            params = file.readline().split()
            if params[0] == "TYPE":
                parser = params[1]
            else:
                parser = "MATRIX"
                file.seek(0)
            return cls.parse(parser, file)

    @staticmethod
    def matrix_parser(file):
        return np.vstack(
            [list(map(float, line.split())) for line in file if line.strip() != ""]
        )

    @staticmethod
    def coords_parser(file):
        def get_distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (0.5)

        coords = [
            tuple(map(float, line.split()[1:])) for line in file if line.strip() != ""
        ]
        weights = np.empty((len(coords), len(coords)))
        for i in range(len(coords)):
            for j in range(len(coords)):
                weights[i, j] = get_distance(coords[i], coords[j])
        return weights

    @classmethod
    def parse(cls, parser, file):
        return getattr(cls, f"{parser.lower()}_parser")(file)

    simple_5x5 = np.array(
        [
            [0, 4, 3, 2, 5],
            [4, 0, 6, 5, 7],
            [3, 6, 0, 6, 7],
            [2, 5, 6, 0, 5],
            [5, 7, 7, 5, 0],
        ]
    )


class KnackPackDataSets(BaseDataSet):
    simple_5d = (14, np.array([4, 5, 6, 11, 5]), np.array([4, 5, 6, 11, 5]))

    @staticmethod
    def read(filepath, sep=" "):
        values, weights = [], []
        max_weight = 0
        with open(filepath, "r") as file:
            for line in file:
                if line.strip() == "":
                    continue
                items_info = list(map(float, line.split(sep)[1:]))
                if len(items_info) > 0:
                    values.append(items_info[0])
                    weights.append(items_info[1])
                else:
                    # reached last line
                    max_weight = float(line)
        all_items_weight = np.sum(weights)
        if max_weight >= all_items_weight:
            raise RuntimeError(
                f"Data is inconsistent: max_weight >= all_items_weight ({max_weight} >= {all_items_weight})"
            )
        return (max_weight, np.array(values), np.array(weights))


def get_nearest_city(weights, city, visited_mask, return_weight=False):
    masked_min_idx = weights[city][visited_mask].argmin()
    nearest_city = np.where(visited_mask)[0][masked_min_idx]
    if return_weight:
        return nearest_city, weights[city, nearest_city]
    else:
        return nearest_city


def calc_path_length(weights, path, is_pure_perm=False):
    res = 0
    for i in range(1, len(path)):
        res += weights[path[i - 1], path[i]]
    if is_pure_perm:
        res += weights[path[-1], path[0]]
    return res


def static_vars(**kwargs):
    def wrapper(func):
        for key, value in kwargs.items():
            setattr(func, key, value)
        return func

    return wrapper


def np_array(value):
    if isinstance(value, np.ndarray):
        return value
    a = np.empty(len(value), dtype=object)
    a[:] = value
    return a


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def get_population(name: str):
    pass


class Enumerator(list):
    pass
