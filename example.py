from algorithms.evolution.island import IslandsManager, print_statistic
from algorithms.individual import PermutationIndividual, init_individ
from algorithms.genetic_politics.genetic_politics import genetic_politics
from utils import TravellingSalesmanDataSets, calc_path_length

TSP_task = TravellingSalesmanDataSets.berlin52  # Getting sample TSP weights
IndividualType: PermutationIndividual = init_individ(
    PermutationIndividual, TSP_task
)  # Registering new individual type based on our task


def score_fn(
    individual: PermutationIndividual,
):  # Defining score function for our individual type
    return calc_path_length(weights=TSP_task, path=individual)


islands = IslandsManager(nIslands=6, migration_rule="ring")
solution = islands.run(
    n=500,
    individual_type=IndividualType.none(),
    score_fn=score_fn,
    genetic_politics=genetic_politics["Basic"],
    log=True,
) # Executing genetic algorithm

print_statistic(solution)
# Possible output:
# Global diversity: 0.31833333333333336 | Best score: 7870.674211298587
# Per island statistic:
#         1. Diversity: 0.13 | Best score: 7935.150646930246
#         2. Diversity: 0.1  | Best score: 7935.150646930246
#         3. Diversity: 0.86 | Best score: 7870.674211298587
#         4. Diversity: 0.05 | Best score: 7870.674211298587
#         5. Diversity: 0.8  | Best score: 7870.674211298587
#         6. Diversity: 0.03 | Best score: 7870.674211298587
