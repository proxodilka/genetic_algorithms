import numpy as np
from .utils import SEED

# np.random.seed(SEED)

DEBUG = True


def pmx_crossover(pair):
    f_parrent, s_parrent = pair
    f_offspring, s_offspring = f_parrent.copy(), s_parrent.copy()
    f_cut, s_cut = np.random.choice(len(f_parrent), 2, replace=False)

    f_cut, s_cut = min(f_cut, s_cut), max(f_cut, s_cut)
    f_offspring[f_cut:s_cut], s_offspring[f_cut:s_cut] = (
        s_offspring[f_cut:s_cut].copy(),
        f_offspring[f_cut:s_cut].copy(),
    )

    replacement_map = {
        key: value
        for key, value in zip(f_offspring[f_cut:s_cut], s_offspring[f_cut:s_cut])
    }

    def replace(idx):
        if idx >= f_cut and idx < s_cut:
            return
        start_value = f_offspring[idx]
        end_value = start_value
        while end_value in replacement_map:
            end_value = replacement_map[end_value]
        if start_value != end_value:
            f_offspring[idx] = end_value

            fs_slice = s_offspring[:f_cut]
            fs_slice[fs_slice == end_value] = start_value
            ss_slice = s_offspring[s_cut:]
            ss_slice[ss_slice == end_value] = start_value

            s_offspring._recompute_score()

    for i in range(len(f_offspring)):
        replace(i)
    consistent_condition = all(
        o.score != float("inf") for o in [f_offspring, s_offspring]
    )
    if not consistent_condition:
        if DEBUG:
            f_temp, s_temp = f_parrent.copy(), s_parrent.copy()
            f_temp[f_cut:s_cut], s_temp[f_cut:s_cut] = (
                s_temp[f_cut:s_cut].copy(),
                f_temp[f_cut:s_cut].copy(),
            )
            breakpoint()
            for i in range(len(f_offspring)):
                replace(i)
        else:
            raise RuntimeError
    parrents_likeness = f_parrent.distance(s_parrent)
    return [(f_offspring, parrents_likeness), (s_offspring, parrents_likeness)]


def cycle_crossover(pair):
    f_parrent, s_parrent = pair
    cycles = [-1] * len(f_parrent)
    replacement_map = {
        key: (i, value) for i, (key, value) in enumerate(zip(f_parrent, s_parrent))
    }

    def mark_cycle(idx, ncycle):
        start_value = f_parrent[idx]
        i, end_value = replacement_map[start_value]
        cycles[idx] = ncycle
        cycles[i] = ncycle
        while end_value != start_value:
            i, end_value = replacement_map[end_value]
            cycles[i] = ncycle

    ncycle = 0
    for i in range(len(f_parrent)):
        if cycles[i] == -1:
            mark_cycle(i, ncycle)
            ncycle += 1

    parrents_likeness = f_parrent.distance(s_parrent)
    offsprings = []
    for i in range(ncycle):
        offspring = [None] * len(f_parrent)
        for j in range(len(f_parrent)):
            offspring[j] = f_parrent[j] if cycles[j] == i else s_parrent[j]
        offspring = type(f_parrent)(
            gens=np.array(offspring), score_fn=f_parrent._score_fn
        )
        offsprings.append((offspring, parrents_likeness))
    return offsprings


def get_offsprings(pairs, methods, **kwargs):
    cumind = np.cumsum(
        [0] + [max(len(pairs) * m.probability, 1) for m in methods.items(len(pairs))],
        dtype=np.int,
    )
    bins = [slice(cumind[i - 1], cumind[i]) for i in range(1, len(cumind))]
    offsprings = []
    for i, bin_slice in enumerate(bins):
        offsprings
        for pair in pairs[bin_slice]:
            offsprings_ = methods[i].variant(pair, **kwargs)
            offsprings.extend(offsprings_)
    return offsprings
