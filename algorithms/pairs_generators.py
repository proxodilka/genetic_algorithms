import numpy as np
from .utils import normalize, np_array, SEED

# np.random.seed(SEED.value)
import random


def briding(population, threshold=2, npairs=None, consider_weights=False, how=None):
    """
    Common function to do in/out-briding

    Parameters
    ----------
    population: list of individuals,
        List to make pairs from.
    threshold: int (default 2),
        Threshold value for in/out-briding.
    npairs: int (optional),
        Number of pairs to generate, if not specified create
        pairs for half of population.
    consider_weights: bool (default False),
        Consider individual weights when randomly picking first
        component of pairs.
    how: {'in', 'out'},
        How to make pairs.

    Returns
    -------
    List of pairs
    """
    comparators = {
        "in": (lambda x, y: x < y, float("inf")),
        "out": (lambda x, y: x > y, -float("inf")),
    }
    comparator, inf = comparators[how]
    population = np_array(population)

    if npairs is None:
        npairs = len(population) // 2

    weights = None
    if consider_weights:
        weights = normalize([o.score for o in population], reverse=True)
    pair_candidates = np.random.choice(population, npairs, p=weights, replace=False)

    pairs = [None] * npairs
    for i, f_cand in enumerate(pair_candidates):
        best_candidate, best_threshold = None, inf
        for s_cand in population:
            dist = f_cand.distance(s_cand)
            if comparator(dist, threshold):
                best_candidate = f_cand
                best_threshold = dist
                break
            elif comparator(dist, best_threshold):
                best_candidate = f_cand
                best_threshold = dist
        pairs[i] = (f_cand, best_candidate)
    return pairs


def inbriding(population, threshold=2, npairs=None, consider_weights=False, **kwargs):
    return briding(population, threshold, npairs, consider_weights, how="in")


def outbriding(population, threshold=2, npairs=None, consider_weights=False, **kwargs):
    return briding(population, threshold, npairs, consider_weights, how="out")


def tournament(population, nfighters=4, npairs=None, consider_weights=False, **kwargs):
    assert nfighters > 1, f"Number of figthers must be at least 2, got {nfighters}."
    population = np_array(population)

    if npairs is None:
        npairs = len(population) // 2

    pairs = [None] * npairs
    weights = None
    if consider_weights:
        weights = [o.score for o in population]
    else:
        weights = np.ones(len(population))
    weights = normalize(weights, reverse=True)
    for i in range(npairs):
        # breakpoint()
        # weights = normalize(weights, reverse=True)
        fighters_idx = np.random.choice(
            len(population), nfighters, p=weights, replace=False
        )
        fighters = population[fighters_idx]
        winners_idx = np.argsort([o.score for o in fighters])[:2]
        # weights[winners_idx] = float("inf")
        pairs[i] = tuple(fighters[winners_idx])
    return pairs


def get_pairs(population, methods, npairs=None, **kwargs):
    if npairs is None:
        npairs = len(population) // 2

    pairs = []
    for method in methods.items(npairs):
        npair = int(npairs * method.probability)
        if npair < 1:
            continue
        new_pairs = method.variant(population, npairs=npair, **kwargs)
        pairs.extend(new_pairs)
    return pairs
