import numpy as np
from .utils import normalize, np_array, SEED

# np.random.seed(SEED.value)
import random


def wheel(population, parents, n=None, repeat=False, include_parents=True, consider_weights=True, **kwargs):
    if include_parents:
        population = np.concatenate([population, parents])

    if consider_weights:
        weights = normalize([o.score for o in population], reverse=True)
    else:
        weights = normalize(np.ones(len(population)))
    result = np.random.choice(population, n, replace=not repeat, p=weights)
    return result


def tournament(
    population,
    parents,
    n=None,
    nfighters=4,
    repeat=False,
    consider_weights=True,
    include_parents=True,
    **kwargs,
):
    assert nfighters > 1, f"Number of figthers must be at least 2, got {nfighters}."
    if include_parents:
        population = np.concatenate([population, parents])

    result = [None] * n
    weights = None
    if consider_weights:
        weights = [o.score for o in population]
    else:
        weights = np.ones(len(population))

    for i in range(n):
        weights = normalize(weights, reverse=True)
        # breakpoint()
        fighters_idx = np.random.choice(
            len(population), nfighters, p=weights, replace=not repeat
        )
        fighters = population[fighters_idx]
        winners_idx = np.argsort([o.score for o in fighters])[0]
        result[i] = fighters[winners_idx]
    return np_array(result)


def generate_population(population, parents, method, n=None, **kwargs):
    population, parents = map(np_array, [population, parents])

    if n is None:
        n = len(parents)

    return method(population=population, parents=parents, n=n, **kwargs)


def execute_selection(population, parents, methods, n=None, **kwargs):
    if n is None:
        n = len(parents)
    elif n == "all":
        n = len(population) + len(parents)

    result = []
    for method in methods.items(n):
        n_method = int(n * method.probability)
        if n_method < 1:
            continue
        population_partition = generate_population(
            population, parents, method=method.variant, n=n_method, **kwargs
        )
        result.extend(population_partition)
    return np_array(result)
