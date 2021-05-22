import numpy as np
from .utils import SEED, np_array
import ray

# np.random.seed(SEED.value)


def swap_mutation(offspring, **kwargs):
    f_pos, s_pos = np.random.choice(len(offspring), 2, replace=False)
    offspring = offspring.copy()
    offspring[f_pos], offspring[s_pos] = (
        offspring[s_pos].copy(),
        offspring[f_pos].copy(),
    )
    return offspring


def reverse_mutation(offspring, **kwargs):
    f_pos, s_pos = np.random.choice(len(offspring), 2, replace=False)
    f_pos, s_pos = min(f_pos, s_pos), max(f_pos, s_pos)
    offspring = offspring.copy()
    offspring[f_pos:s_pos] = offspring[f_pos:s_pos][::-1]
    return offspring


def opt_mutation(offspring, nswaps=None, **kwargs):
    def opt_swap(o, i, j):
        obj = o.copy()
        obj[i:j] = obj[i:j][::-1]
        return obj

    def opt_swap2(o, i, j):
        obj = o.copy()
        obj[i], obj[j] = obj[j], obj[i]
        return obj

    result = offspring
    if nswaps is None:
        nswaps = 52
    points = np.random.choice(range(len(offspring)), nswaps)
    for i in points:
        for j in range(i, min(i + nswaps, len(result))):
            if np.random.randint(2) == 0:
                new_route = opt_swap(result, i, j)
            else:
                new_route = opt_swap2(result, i, j)
            if new_route.score < result.score:
                result = new_route
    return result
    # offspring[:] = result[:]


# @ray.remote
def mutate(offspring, mutation_rate, method, consider_incest=False):
    if isinstance(offspring, tuple):
        assert (
            len(offspring) == 2
        ), f"Incorrect tuple of offspring and parrents likeness. Excepted lenght 2, got {len(offspring)}."
        offspring, parrents_likeness = offspring
    else:
        parrents_likeness = 0

    incest_rate = parrents_likeness / (len(offspring) * 2)
    mutation_rate = (
        mutation_rate if not consider_incest else (mutation_rate + incest_rate)
    )

    if np.random.rand() < mutation_rate:
        offspring = method(offspring, incest_rate=incest_rate)
    return offspring


def get_mutated(offsprings, methods, __hack__sort=False, **kwargs):
    if __hack__sort:
        offsprings = np.sort(offsprings)

    cumind = np.cumsum(
        [0] + [len(offsprings) * m.probability for m in methods.items(len(offsprings))],
        dtype=np.int,
    )
    bins = [slice(cumind[i - 1], cumind[i]) for i in range(1, len(cumind))]

    futures = []

    queue = []
    num_cpus = 12
    queue = [
        (offspring, methods[i].variant, kwargs)
        for i, bin_slice in enumerate(bins)
        for offspring in offsprings[bin_slice]
    ]
    step = len(queue) // num_cpus
    # breakpoint()
    # futures = [drain_queue.remote(queue[i:min(i+step, len(queue))]) for i in range(0, len(queue), step)]
    # for i, bin_slice in enumerate(bins):
    #     for offspring in offsprings[bin_slice]:
    #         futures.append(mutate.remote(offspring, method=methods[i].variant, **kwargs))
    # new_offsprings = ray.get(futures)
    new_offsprings = [drain_queue(queue)]
    # breakpoint()
    # new_offsprings[0].extend(new_offsprings[1:])
    # breakpoint()
    return np.concatenate([np_array(o) for o in new_offsprings])


# @ray.remote
def drain_queue(queue):
    # breakpoint()
    return [mutate(o, method=m, **k) for o, m, k in queue]
