import ray
import numpy as np

from .algorithm import BaseEvolutionAlgo
from utils import Enumerator
from algorithms.utils import StatisticLogger, np_array
import pandas


@ray.remote
class IslandWorker:
    def __init__(self, **kwargs):
        self.island: BaseEvolutionAlgo = BaseEvolutionAlgo(**kwargs)

    @ray.method(num_returns=2)
    def run(self, n):
        self.island.execute(n)
        return (self.island.population, self.island.best_individ)

    def set_population(self, new_population):
        self.island.base_population = new_population

    def get_statistic(self, field=None):
        if isinstance(field, int):
            return self.island.logger.storage[field]
        elif isinstance(field, str):
            return getattr(self.island.logger, field)
        return self.island.logger.storage

    def write_logs(self):
        self.island.logger.write()


class MigrationManager:
    """
    Manage migration of the given population.

    Parameters
    ----------
    population : np.ndarray of individuals
    """

    def __init__(self, population):
        self.population = np.sort(population)

    def to_migrate(self, n, update_self=True):
        """
        Parameters
        ----------
        n : int
            Number of individuals to migrate from the source population.
        update_self : bool, default: True
            Whether to remove migrated individuals from the source population.

        Returns
        -------
        tuple of np.ndarrays of individuals
            First element is the individuals to migrate.
            Second element is the source population without migrated invididuals.
        """
        nbest = int(n / 2 + 0.5)
        nrand = n - nbest
        random_indices = np.random.randint(
            low=nbest, high=len(self.population), size=nrand
        )

        migrated = np.concatenate(
            [self.population[:nbest], self.population[random_indices]]
        )
        migrated_indices = np.concatenate([np.arange(nrand), random_indices])

        new_source = np.delete(self.population, migrated_indices)
        if update_self:
            self.population = new_source

        return (migrated, new_source)

    def append(self, individuals):
        """
        Append new individuals to the source population.

        Parameters
        ----------
        individuals : np.ndarray of individuals.
        """
        self.population = np.concatenate([self.population, individuals])

    def round_merge(self, populations):
        n_populations = len(populations)
        global_population = np.concatenate(populations)

        normalized_populations = np.empty(n_populations, dtype=object)
        for i in range(len(normalized_populations)):
            normalized_populations[i] = np.empty(
                len(global_population) // len(normalized_populations), dtype=object
            )

        counted_individs = pandas.Series(global_population).value_counts(sort=False)

        def individuals_yielder(source):
            while True:
                has_yield = False
                for individ, count in source.items():
                    if count > 0:
                        has_yield = True
                        yield individ
                if not has_yield:
                    break
                source -= 1

        def append_individual(population_idx, individ_idx, value):
            if len(normalized_populations[population_idx]) <= individ_idx:
                # breakpoint()
                normalized_populations[population_idx] = np.concatenate(
                    [normalized_populations[population_idx], np_array([value])]
                )
                return
            normalized_populations[population_idx][individ_idx] = value

        yielder = individuals_yielder(counted_individs)
        for i in range(counted_individs.sum()):
            population_idx = i % n_populations
            individ_idx = i // n_populations
            next_individ = next(yielder)
            append_individual(population_idx, individ_idx, next_individ)

        return normalized_populations

    def squeeze_round_merge(self, populations):
        global_population = np.concatenate(populations)
        counted_individs = pandas.Series(global_population).value_counts(sort=False).clip(upper=30)
        nindivids = counted_individs.sum()

        n_populations = max(1, nindivids // len(populations[0]))

        normalized_populations = np.empty(n_populations, dtype=object)
        for i in range(len(normalized_populations)):
            normalized_populations[i] = np.empty(1, dtype=object)

        def individuals_yielder(source):
            while True:
                has_yield = False
                for individ, count in source.items():
                    if count > 0:
                        has_yield = True
                        yield individ
                if not has_yield:
                    break
                source -= 1

        def append_individual(population_idx, individ_idx, value):
            if len(normalized_populations[population_idx]) <= individ_idx:
                # breakpoint()
                normalized_populations[population_idx] = np.concatenate(
                    [normalized_populations[population_idx], np_array([value])]
                )
                return
            normalized_populations[population_idx][individ_idx] = value

        yielder = individuals_yielder(counted_individs)
        for i in range(counted_individs.sum()):
            population_idx = i % n_populations
            individ_idx = i // n_populations
            next_individ = next(yielder)
            append_individual(population_idx, individ_idx, next_individ)

        return normalized_populations


class IslandsManager:
    def __init__(self, nIslands: int, migration_rule: str = "poly-3"):
        self.nIslands: int = nIslands
        self.islands: list = []
        self._rule = migration_rule
        self._update_migration_rules()
        self.logger = StatisticLogger()

    def _update_migration_rules(self):
        if len(self.islands) != 0:
            self.nIslands = len(self.islands)
        self.migration_rule = self._build_migration_rules(self._rule)

    def _build_ring(self):
        return {i: [(i + 1) % self.nIslands] for i in range(self.nIslands)}

    def _build_poly(self, n):
        npairs = int(self.nIslands / n + 0.5)
        to_connect = [
            list(range(i * n, min((i + 1) * n, self.nIslands))) for i in range(npairs)
        ]

        res_dict = {}

        for group in to_connect:
            for i in group:
                res_dict[i] = [j for j in group if j != i]
        # breakpoint()
        return res_dict

    def _build_tree(self, nchilds=2):
        res_dict = {i: [] for i in range(self.nIslands)}
        for i in range(self.nIslands):
            res_dict[i] += [
                nchilds * i + j
                for j in range(1, nchilds + 1)
                if nchilds * i + j < self.nIslands
            ]
            for j in range(1, nchilds + 1):
                if nchilds * i + j >= self.nIslands:
                    break
                res_dict[nchilds * i + j] += [i]
        # breakpoint()
        return res_dict

    def _build_migration_rules(self, name):
        components = name.split("-")
        if components[0] == "poly":
            shape = int(components[1]) if len(components) > 1 else 3
            return self._build_poly(shape)
        return self.MIGRATION_RULES[name](self)

    MIGRATION_RULES = {"ring": _build_ring, "tree": _build_tree}

    def run(self, n: int = 100, sync_freq: int = 50, **kwargs):
        """
        Parameters
        ----------
        n : int, default: 100
            Number of evolution steps.
        sync_freq : int, default: 50
            Synchronization frequency of islands. Synchronization occurs in
            each `sync_freq` evolution step.
        **kwargs : dict
            Evolution algorithm parameters.

        Returns
        -------
        list of tuples
        """
        if not ray.is_initialized():
            ray.init()

        islands_kwargs = self._compute_kwargs(**kwargs)
        self.islands: list = [
            IslandWorker.remote(**arguments) for arguments in islands_kwargs
        ]
        for i in range(int(n / sync_freq + 0.5)):
            print(f"sync-step {i}")
            futures = [x.run.remote(sync_freq) for x in self.islands]
            island_populations = [ray.get(ft) for ft in futures]
            self.log(island_populations, (i + 1) * sync_freq)
            # print_statistic(island_populations)
            if i > 0 and (i % 2) == 0:
                new_populations = self.round_merge(island_populations)
            else:
                new_populations = self.merge(island_populations)
            if len(self.islands) != len(new_populations):
                self.islands = self.islands[:len(new_populations)]
                self._update_migration_rules()
            ray.get(
                [
                    i.set_population.remote(p)
                    for (i, p) in zip(self.islands, new_populations)
                ]
            )
        futures = [i.write_logs.remote() for i in self.islands]
        self.logger.write()
        ray.get(futures)
        return island_populations

    def round_merge(self, populations):
        return MigrationManager.squeeze_round_merge(None, [p[0] for p in populations])

    def merge(self, populations):
        populations: list[MigrationManager] = [
            MigrationManager(p[0]) for p in populations
        ]

        nmigrants = 10
        migrants = {i: np.array([]) for i in range(self.nIslands)}

        for src_island, to_migrate_islands in self.migration_rule.items():
            for idx in to_migrate_islands:
                to_migrate, _ = populations[src_island].to_migrate(nmigrants)
                # breakpoint()
                migrants[idx] = np.concatenate([migrants[idx], to_migrate])

        for island_idx in migrants.keys():
            populations[island_idx].append(migrants[island_idx])

        return [wrapper.population for wrapper in populations]

    def _compute_kwargs(self, **kwargs):
        """
        Compute arguments for each island.

        Returns
        -------
        list of dict
        """
        result = [{} for _ in range(self.nIslands)]

        for key, value in kwargs.items():
            if key == "genetic_politics":
                # breakpoint()
                pass
            if isinstance(value, Enumerator):
                for island_idx, island_value in zip(range(self.nIslands), value):
                    result[island_idx][key] = island_value
            else:
                for island_idx in range(self.nIslands):
                    result[island_idx][key] = value

        return result

    def log(self, islands, generation):
        global_population = np.concatenate([island[0] for island in islands])
        # breakpoint()
        best_individ = np.min(np_array([island[1] for island in islands]))
        # breakpoint()
        self.logger.log(
            generation=generation,
            population=global_population,
            best_individ_score=best_individ.score,
        )
        global_stat = self.logger.storage[-1]

        per_island_stat = [
            ray.get(island.get_statistic.remote(-1)) for island in self.islands
        ]
        global_unique = np.concatenate(
            [ray.get(island.get_statistic.remote("unique")) for island in self.islands]
        )
        global_unique = np.unique(global_unique, axis=0)
        global_stat["exploration"] = global_unique.shape[0]
        # breakpoint()
        print(
            f"Global diversity: {global_stat['diversity']} | Best score: {global_stat['best_individ_score']} | Exploration: {global_stat['exploration']}"
        )
        print("Per island statistic:")
        for i, island in enumerate(per_island_stat):
            print(
                f"\t{i + 1}. Diversity: {island['diversity']} | Best score: {island['best_individ_score']} | Exploration: {island['exploration']}"
            )


def get_statictic(populations):
    island_statictic = [
        {
            "diversity": BaseEvolutionAlgo.get_diversity(island[0]),
            "solution": island[1].score,
        }
        for island in populations
    ]

    global_population = np.concatenate([island[0] for island in populations])
    global_optimums = np.sort(np_array([island[1] for island in populations]))
    global_statistic = {
        "diversity": BaseEvolutionAlgo.get_diversity(global_population),
        "solution": global_optimums[0].score,
    }
    return global_statistic, island_statictic


def lol(populations):
    island_statictic = [
        {"diversity": BaseEvolutionAlgo.get_diversity(island), "solution": None}
        for island in populations
    ]

    global_population = np.concatenate([island for island in populations])
    global_statistic = {
        "diversity": BaseEvolutionAlgo.get_diversity(global_population),
        "solution": None,
    }
    return global_statistic, island_statictic


def print_statistic(res, is_lol=False):
    if is_lol:
        global_stat, island_stat = lol(res)
    else:
        global_stat, island_stat = get_statictic(res)

    print(
        f"Global diversity: {global_stat['diversity']} | Best score: {global_stat['solution']}"
    )
    print("Per island statistic:")
    for i, island in enumerate(island_stat):
        print(
            f"\t{i + 1}. Diversity: {island['diversity']} | Best score: {island['solution']}"
        )
