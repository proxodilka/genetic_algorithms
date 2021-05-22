from dataclasses import dataclass
import ray

from algorithms.evolution.island import IslandsManager, get_statictic
from algorithms.evolution.algorithm import BaseEvolutionAlgo

from algorithms.individual import PermutationIndividual, init_individ
from algorithms.genetic_politics.genetic_politics import genetic_politics
from algorithms.utils import np_array

from utils import TravellingSalesmanDataSets, calc_path_length, Enumerator
import numpy as np
from pathlib import Path
import os
import cloudpickle


def solve_task(
    dataset_name: str,
    nIslands: int = 1,
    politic_name: str = "Basic",
    population_name: str = "",
    population_size: int = 100,
    migration_rule: str = "poly-3",
):
    np.random.seed(0)
    weights: np.ndarray = getattr(TravellingSalesmanDataSets, dataset_name)
    IndividualType: PermutationIndividual = init_individ(PermutationIndividual, weights)

    def score_fn(x):
        return calc_path_length(weights, x)

    islands: IslandsManager = IslandsManager(nIslands, migration_rule=migration_rule)

    algo_kwargs = {
        "individual_type": IndividualType.none(),
        "score_fn": score_fn,
        "genetic_politics": Enumerator(
            [genetic_politics[politic_name], genetic_politics["MA"], genetic_politics[politic_name], genetic_politics[politic_name]] * 4
        ),
        "stuck_determiner": Enumerator(
            [
                lambda x: False,
                BaseEvolutionAlgo.default_stuck_determiner,
                BaseEvolutionAlgo.diversity_stuck_determiner,
                BaseEvolutionAlgo.default_stuck_determiner,
            ]
            * 4
        ),
        "seed": Enumerator(np.random.randint(32000) for _ in range(nIslands)),
        "log": True,
    }

    base_population = get_population(
        dataset_name,
        politic_name,
        nIslands,
        population_size,
        population_name,
        base_population_size=population_size * nIslands,
        **algo_kwargs,
    )

    algo_kwargs["base_population"] = Enumerator(
        [
            base_population[population_size * i : population_size * (i + 1)]
            for i in range(nIslands)
        ]
    )

    result = islands.run(n=600, sync_freq=50, **algo_kwargs)

    return result


def get_population(*args, force_new=False, **kwargs):
    """
    Parameters
    ----------
    *args : iterable
        Components of the file name possibly containing the population dump.
    force_new : bool, default: False
        Generate new population even if there is suitable dump.
    **kwargs : dict
        Parameters for BaseEvolutionAlgo to generate population with.
    """
    populations_path = Path(
        os.environ.get("POPULATIONS_PATH", os.path.abspath("population_dumps"))
    ).joinpath("_".join(map(str, args)))

    if not force_new and populations_path.is_file():
        with open(populations_path, "rb") as file:
            return cloudpickle.loads(file.read())

    spreaded_kwargs = {}
    for k, v in kwargs.items():
        spreaded_kwargs[k] = v[0] if isinstance(v, Enumerator) else v

    algo: BaseEvolutionAlgo = BaseEvolutionAlgo(**spreaded_kwargs)
    base_population = algo.generate_population()

    dump = cloudpickle.dumps(base_population)
    with open(populations_path, "wb") as file:
        file.write(dump)

    return base_population


if __name__ == "__main__":
    res = solve_task("att48", nIslands=16, population_size=100, migration_rule="ring")

    breakpoint()
