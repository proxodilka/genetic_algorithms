from ..population import BasePopulation
from ..individual import BaseIdividual
from ..heuristic_methods import MonteCarlo

from ..pairs_generators import get_pairs
from ..crossover import get_offsprings
from ..mutations import get_mutated
from ..population_updaters import execute_selection

from ..utils import np_array, static_vars, Logger, DummyLogger, Picker, StatisticLogger
from pandas.core.dtypes.common import is_list_like
from ..utils import SEED

import numpy as np
from timeit import default_timer as timer

# np.random.seed(SEED)


class BaseEvolutionAlgo:
    def __init__(
        self,
        base_population_size=100,
        individual_type=None,
        score_fn=None,
        base_population=None,
        genetic_politics=None,
        stuck_determiner=None,
        log=False,
        logger_kwargs=None,
        verbose_level=0,
        **kwargs,
    ):
        if kwargs.get("seed") is not None:
            np.random.seed(kwargs["seed"])

        if base_population is None:
            self.base_population_size = base_population_size
            self.individual_type = individual_type
            self.base_population = None
        else:
            assert is_list_like(
                base_population
            ), f"Excepted list-like type of base population, got: {type(base_population)}"
            self.base_population = np_array(base_population)
            self.individual_type = type(self.base_population[0])
            self.base_population_size = len(self.base_population)

        self.score_fn = score_fn
        self.i = 0
        self._kwargs = kwargs.copy()

        self.genetic_politics = self.build_genetic_politics(genetic_politics)

        self.best_solution = self.individual_type.none()
        self.best_individ = self.individual_type.none()
        self.worst_score, self.median_score = float("inf"), float("inf")

        self.stuck_determiner = (
            self.diversity_stuck_determiner
            if stuck_determiner is None
            else stuck_determiner
        )
        self.stuck_counter = 0
        self.stuck_state = "normal"
        self.verbose_level = verbose_level
        if logger_kwargs is None:
            logger_kwargs = {}
        self.logger = (
            StatisticLogger(
                **logger_kwargs,
            )
            if log
            else DummyLogger()
        )

    @classmethod
    def build_genetic_politics(cls, genetic_politics):
        if "normal" in genetic_politics and "stuck" in genetic_politics:
            normal_gp = genetic_politics["normal"].copy()
            normal_gp.update(**genetic_politics["stuck"])
            result = {
                "normal": genetic_politics["normal"],
                "stuck": normal_gp,
            }
        else:
            result = {
                "normal": genetic_politics,
                "stuck": genetic_politics.copy(),
            }
        return result

    def execute(self, n):
        self.population = (
            self.generate_population()
            if self.base_population is None
            else self.base_population
        )
        self.try_to_update_optimum()
        for _ in range(n):
            # print(
            #     f"SCORE: {self.best_individ.score} | DIVERSITY: {self.get_diversity()}"
            # )
            t1 = timer()
            self.do_step()
            self.try_to_update_optimum()
            t2 = timer()
            # print(t2 - t1)

    def generate_population(self, n=None):
        h_politics = self.get_politics("heuristic_politics")
        methods: list = h_politics.pop("methods")

        n = n if n else self.base_population_size
        result = []
        for variant in methods.items(n):
            method = variant.variant
            nsolutions = int(n * variant.probability)
            result.extend(
                method(
                    self.individual_type, self.score_fn, **self._kwargs
                ).get_solutions(nsolutions)
            )
        return np_array(result)

    def expand_population(self):
        pairs = self.generate_pairs()
        # breakpoint()
        return self.mate(pairs)

    def log(self):
        self.logger.log(
            generation=self.i,
            best_score=self.best_solution.score,
            best_individ_score=self.best_individ.score,
            worst_individ_score=self.worst_score,
            median_individ_score=self.median_score,
            population=self.population,
        )

    def do_step(self):
        self.log()
        if self.verbose_level > 1:
            self.full_verbose()
        self.are_we_stuck()
        new_individs = self.expand_population()
        # breakpoint()
        mutated_individs = self.mutate(new_individs)
        # breakpoint()
        self.selection(mutated_individs)
        self.i += 1

    def mutate(self, individs):
        # breakpoint()
        kwargs = self.get_politics("mutation_politics")
        return np_array(get_mutated(individs, **kwargs))

    def selection(self, new_individs):
        kwargs = self.get_politics("selection_politics")
        self.population = execute_selection(new_individs, self.population, **kwargs)

    def generate_pairs(self):
        kwargs = self.get_politics("pairs_generation_politics")
        return get_pairs(self.population, **kwargs)

    def mate(self, pairs):
        kwargs = self.get_politics("crossover_politics")
        return np_array(get_offsprings(pairs, **kwargs))

    def try_to_update_optimum(self):
        is_optimum_opdated = False
        scores = [o.score for o in self.population]
        self.best_individ = self.population[np.argmin(scores)]
        self.worst_score = np.max(scores)
        self.median_score = np.median(scores)
        if self.best_individ.score < self.best_solution.score:
            self.best_solution = self.best_individ.copy()
            is_optimum_opdated = True
        if is_optimum_opdated and self.verbose_level > 0:
            print(f"Optimum updated: {self.best_solution}: {self.best_solution.score}")
        return is_optimum_opdated

    def are_we_stuck(self):
        are_we_stuck = self.stuck_determiner(self)
        if are_we_stuck and self.verbose_level > 0:
            print(
                f"We're stuck at {self.i} iteration, trying to do something with it..."
            )
        self.stuck_state = "stuck" if are_we_stuck else "normal"

    def get_diversity(self):
        population = self.population if isinstance(self, BaseEvolutionAlgo) else self
        nunique = np.unique([individ.gens for individ in population], axis=0).shape[0]
        return nunique / population.shape[0]

    @staticmethod
    def diversity_stuck_determiner(self):
        epsilon = 0.1
        is_stuck = self.get_diversity() < epsilon
        if is_stuck:
            #print("STUCK!")
            pass
        return is_stuck

    @staticmethod
    def default_stuck_determiner(self):
        epsilon = 0.99
        min_s, max_s = self.worst_score, self.best_individ.score
        if max_s / min_s > epsilon:
            # breakpoint()
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 49:
            # print("STUCK!")
            self.stuck_counter = 0
            return True
        return False

    def get_politics(self, key):
        # breakpoint()
        politics_template = self.genetic_politics[self.stuck_state][key]
        return {
            k: v if not callable(v) else v(self) for k, v in politics_template.items()
        }

    def full_verbose(self):
        best_individ = min(self.population, key=lambda x: x.score)
        print(
            f"\n========== Generation {self.i} ==========\n\tPopulation:\n{self.population}\n\t"
            + f"Best solution: {self.best_solution}: {self.best_solution.score}\n\t"
            + f"Best individ:  {best_individ}: {best_individ.score}\n"
        )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.logger.write()
