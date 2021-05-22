from utils.utils import TravellingSalesmanDataSets
import numpy as np
from .utils import SEED
import ray
import re

# np.random.seed(SEED.value)
import abc


class BaseIdividual:
    def __init__(self, gens, score_fn=None):
        self.validate_gens(gens)
        self._gens = gens
        if not hasattr(self, "_score_fn"):
            assert score_fn is not None, "Must pass score_fn"
            self._score_fn = score_fn
        self._score_cache = None

    gens = property(lambda self: self._gens)

    @property
    def score(self):
        if self._score_cache is None:
            self._recompute_score()
        return self._score_cache

    def _recompute_score(self, *args, **kwargs):
        self._score_cache = self._score_fn(self._gens, *args, **kwargs)

    @abc.abstractclassmethod
    def build_random(cls):
        pass

    def distance(self, other):
        return np.count_nonzero(self._gens ^ other._gens)

    @classmethod
    def validate_gens(cls, gens):
        pass

    @classmethod
    def none(cls):
        res = cls([0], lambda x: x)
        res._score_cache = float("inf")
        return res

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return np.all(self.gens == other.gens)
        return self.gens == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __repr__(self):
        class_name = self._get_my_name()
        gens_str = repr(self._gens)
        self.score
        return f"{class_name} | Score: {self.score} | Gens:\n{gens_str}"

    def __str__(self):
        class_name = self._get_my_name()
        gens_str = str(self._gens)
        self.score
        return f"{class_name} | Score: {self.score} | Gens:\n{gens_str}"

    def __len__(self):
        return len(self._gens)

    def _get_my_name(self):
        return re.findall(r"([^\.']*)'>", str(type(self)))[0]

    @classmethod
    def shape(cls):
        return cls._size

    def copy(self):
        if hasattr(self, "_score_fn"):
            return type(self)(gens=self._gens.copy(), score_fn=self._score_fn)
        else:
            return type(self)(gens=self._gens.copy())

    def __setitem__(self, key, value):
        self._gens.__setitem__(key, value)
        self._score_cache = None

    def __getitem__(self, key):
        return self._gens.__getitem__(key)

    @property
    def __constructor__(self):
        return type(self)

    def __call__(self, *args, **kwargs):
        return self.__constructor__(*args, **kwargs)

    # def __getattr__(self, key):
    #     if key.startswith("__array"):
    #         raise AttributeError
    #     return getattr(self._gens, key)


class FloatIndividual(BaseIdividual):
    @classmethod
    def build_random(cls, size, score_fn, low=None, high=None):
        low = low if low else np.iinfo(np.int32).min
        high = high if high else np.iinfo(np.int32).max

        return cls(gens=np.random.randint(low, high, size), score_fn=score_fn)


class SerializationPrototype(type):
    def __reduce__(cls):
        def deserializer(size, weights):
            PermutationIndividual._size = size
            PermutationIndividual._weights = weights
            return PermutationIndividual

        return deserializer, cls._size, cls._weights


class PermutationIndividual(BaseIdividual):
    @classmethod
    def build_random(cls, size=None, score_fn=None):
        size, score_fn = getattr(cls, "_size", size), getattr(
            cls, "_score_fn", score_fn
        )
        return cls(gens=np.random.permutation(size), score_fn=score_fn)

    def hood(self):
        s_len = len(self)
        result = []
        for i in range(s_len):
            # skip generation of the same individual
            if i == s_len - i - 1:
                continue
            to_insert = self.copy()
            to_insert[i], to_insert[s_len - i - 1] = (
                to_insert[s_len - i - 1],
                to_insert[i],
            )
            result.append(to_insert)
        return result

    @classmethod
    def is_permutation(cls, array):
        return (
            np.min(array) == 0
            and np.max(array) == (len(array) - 1)
            and len(array) == len(np.unique(array))
        )

    @classmethod
    def validate_gens(cls, gens):
        if not cls.is_permutation(gens):
            raise RuntimeError(f"Passed sequence is not permutation: {gens}")

    def _recompute_score(self, *args, **kwargs):
        if not self.is_permutation(self._gens):
            self._score_cache = float("inf")
        else:
            # self._score = self.sukapidor228()
            super()._recompute_score()

    def _score_fn(self, *args, **kwargs):
        path = self._gens
        weights = self._weights
        res = 0
        for i in range(1, len(path)):
            res += weights[path[i - 1], path[i]]
        res += weights[path[-1], path[0]]
        return res

    def __reduce__(self):
        def deserealizer(gens, weights, size):
            PermutationIndividual._size = size
            PermutationIndividual._weights = weights
            res = PermutationIndividual(gens)
            return res

        return deserealizer, (self._gens, self._weights, self._size)


def init_individ(cls, weights):
    setattr(cls, "_size", weights.shape[0])
    setattr(cls, "_weights", weights)
    return cls
