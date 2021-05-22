from abc import abstractmethod
import numpy as np
from .utils import SEED

# np.random.seed(SEED.value)


class BaseSolver:
    def __init__(self, individual_type, score_fn, **kwargs):
        self.type = individual_type
        self.scorer = score_fn
        self.i = 0

    def get_solutions(self, n, steps=None):
        if steps is None:
            steps = min(2 * self.type.shape(), 100)

        return [self.get_solution(steps) for _ in range(n)]

    @abstractmethod
    def get_solution(self, steps, return_score=False):
        pass


class MonteCarlo(BaseSolver):
    def get_solution(self, steps, return_score=False):
        print(self.i)
        self.i += 1
        generator = self.type.build_random

        best_score = float("inf")
        solution = None
        for _ in range(steps):
            candidate = generator()
            c_score = self.scorer(candidate)

            if c_score < best_score:
                solution = candidate
                best_score = c_score

        if return_score:
            return solution, best_score
        else:
            return solution


class DFS(BaseSolver):
    def get_solution(self, steps, return_score=False):
        generator = self.type.build_random
        hood_generator = self.type.hood

        solution = generator()
        best_score = self.scorer(solution)

        for _ in range(steps):
            hood = hood_generator(solution)
            solution_was_updated = False
            for candidate in hood:
                c_score = self.scorer(candidate)
                if c_score < best_score:
                    solution = candidate
                    best_score = c_score
                    solution_was_updated = True
            if not solution_was_updated:
                break

        if return_score:
            return solution, best_score
        else:
            return solution


class NearestCity(BaseSolver):
    def __init__(self, individual_type, score_fn, weights=None, **kwargs):
        assert weights is not None or hasattr(
            self, "weights"
        ), "City weights matrix is not provided, pass it to the constructor, or use decorator to bind it to the new class."
        super().__init__(individual_type, score_fn)
        if not hasattr(self, "weights"):
            self.weights = weights

    def get_solution(self, *args, **kwargs):
        return_score = kwargs.get("return_score", False)
        solution = min(
            [self.solve(self.weights, i) for i in range(len(self.weights))],
            key=lambda x: x[1],
        )
        individ, score = solution
        individ = self.type(gens=individ, score_fn=self.scorer)
        if return_score:
            return individ, score
        return individ

    def solve(self, weights, start_city):
        n_cities = weights.shape[0]
        result = np.zeros(n_cities, dtype=np.int)
        visited_mask = ~np.zeros(n_cities, dtype=np.bool)
        result[0] = start_city
        visited_mask[start_city] = False

        for i in range(1, len(result)):
            city_from = result[i - 1]
            nearest_city = self.get_nearest_city(weights, city_from, visited_mask)
            result[i] = nearest_city
            visited_mask[nearest_city] = False

        score = self.scorer(result)
        return result, score

    @staticmethod
    def get_nearest_city(weights, city, visited_mask, return_weight=False):
        masked_min_idx = weights[city][visited_mask].argmin()
        nearest_city = np.where(visited_mask)[0][masked_min_idx]
        if return_weight:
            return nearest_city, weights[city, nearest_city]
        else:
            return nearest_city
