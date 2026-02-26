import numpy as np
from typing import Optional, List
import time

from ga_functions import FitnessFn, CrossoverType
from helper_functions import *

from matplotlib import pyplot as plt



def ga_iteration_extra(P: np.ndarray, fitness_fn: FitnessFn, crossover: CrossoverType = "UX", rng: Optional[np.random.Generator] = None, debug_ties: bool = False) -> Tuple[np.ndarray, bool, int]:
    improvement = False

    if rng is None:
        rng = np.random.default_rng()

    P = np.asarray(P)
    if P.ndim != 2:
        raise ValueError(f"P is wrong shape. Current shape {P.shape}")
    N, L = P.shape
    if N % 2 != 0:
        raise ValueError(f"Population size should be even. Current N={N}")

    P_shuf = P[rng.permutation(N)]

    if crossover == "UX":
        recombination = uniform_crossover
    elif crossover == "2X":
        recombination = two_point_crossover
    else:
        raise ValueError("crossover must be 'UX' or '2X'")

    P_next = np.empty_like(P_shuf)
    is_child = np.array([0, 0, 1, 1], dtype=np.int8)


    total_fitness_evaluations = 0   # this stores the total number of solutions that are evaluated = total number of strings the fitness functions is applied to
    out = 0
    for i in range(0, N, 2):
        p1 = P_shuf[i]
        p2 = P_shuf[i + 1]
        c1, c2 = recombination(p1, p2, rng)

        family = np.stack([p1, p2, c1, c2], axis=0)

        total_fitness_evaluations += family.shape[0]    # total evaluations for this family -> number of family members that are evaluated

        fit = np.asarray(fitness_fn(family))

        if fit.shape != (4,):
            raise ValueError(f"fitness_fn should return proper shape, got {fit.shape}")

        order = np.lexsort((-is_child, -fit))
        winners_idx = order[:2]
        winners = family[winners_idx]

        if not improvement:
            parents_min = fit[:2].min()  # indices 0,1
            children_max = fit[2:].max()  # indices 2,3
            improvement = children_max > parents_min

        if debug_ties:
            # Rule to pick children in case of a tie
            if np.all(fit == fit[0]):
                print(f"Family all-tied fit={fit[0]} winners_idx={winners_idx}")

        P_next[out] = winners[0]
        P_next[out + 1] = winners[1]
        out += 2

    return P_next, improvement, total_fitness_evaluations


"""
This Function is based on the function 'run_ga' in file 'ga_functions.py', with minor tweaks to cover the extra questions

Additions:
    - The stop condition is that all population members should converge to global optimum '111...11' for the algorithm to stop 
    - Added return value prop(t) (List[float]) where prop(t) is the percentage of '1's within the population, for each generation t. 
      It is calculated by taking the mean fitness value of each generation (since fitness == Count Ones) and dividing by the string length l
      
 
"""
def run_ga_extra(N:int, l:int, fitness_fn: FitnessFn, crossover: CrossoverType = "UX", max_failures = 20) -> Tuple[np.ndarray, bool, int, int, float, List[float]]:
    # Step 1: initiate random population of size N, length l
    rng = np.random.default_rng()
    P_old = rng.integers(0, 2, size=(N, l), dtype=np.int8)
    consecutive_failures = 0
    total_generations = 0
    total_fitness_evaluations = 0
    t0 = time.perf_counter()

    # Extra Question 1: prop(t) -> percentage of '1' in total population
    prop_ones : List[float] = [np.mean(fitness_fn(P_old)) / l]

    # Step 2: run 'ga_iteration' iteratively to generate new populations
    while consecutive_failures < max_failures:
        P_new, improve_flag, iter_evaluations = ga_iteration_extra(P_old, fitness_fn, crossover=crossover, rng=rng)
        P_old = P_new
        total_generations += 1
        total_fitness_evaluations += iter_evaluations

        # Calculating prop(t): In this case total percentage of ones = (sum(count_ones(Population))) / (N*l) => avg(fitness(Population)) / l [since fitness function for this experiment is 'Count Ones']
        mean_fitness = np.mean(fitness_fn(P_new))
        prop_ones.append(mean_fitness/l)


        # New stop condition -> all population converges to '1111...111'
        stop_condition = np.all(np.all(P_new == 1, axis=1))
        print(
            f"Generation: {total_generations}\n\tPopulation Converged: {stop_condition}\n\tMean Fitness: {mean_fitness:.2f}\n\tConsecutive Failures: {consecutive_failures}\n")

        # Step 3: Check for global optimum in new population, or if no improvement has been made for the past 20 generations
        if stop_condition:
            total_time = time.perf_counter() - t0

            return P_new, True, total_generations, total_fitness_evaluations, total_time, prop_ones
        else:
            if not improve_flag:
                consecutive_failures += 1
            else:
                consecutive_failures = 0


    total_time = time.perf_counter() - t0
    return P_new, False, total_generations, total_fitness_evaluations, total_time, prop_ones

def plot_metrics(gens:int, prop_t: List[float], corr_t:List[int], err_t:List[int]):
    # x axis: number of generations (starting at 0)
    x = np.arange(gens+1)

    # 1. Plot prop(t)
    plt.plot(x, prop_t)
    plt.xticks(x)
    plt.title("Proportion of bits-1 in total population")
    plt.xlabel("Generations")
    plt.ylabel("ptop(t)")

    plt.savefig(r"plots/1_prop_t.png", dpi=300, bbox_inches="tight")
    plt.show()


    # 2. err(t) vs corr(t)


    # 3. schema 1#### vs 0####


    return


def main() -> None:
    # Settings are fixed to the following values for this experiment
    N = 200
    L = 40
    fitness_fn : FitnessFn = fitness_counting_ones
    crossover : CrossoverType = "UX"
    # max_failures = 50     # just in case

    P_final, success, gens, evals, time, prop_t = run_ga_extra(N=N, l=L, fitness_fn=fitness_fn, crossover=crossover)

    if success:
        plot_metrics(gens=gens, prop_t=prop_t, err_t=[], corr_t=[])

    else:
        print("Population failed to converge, no plots available")

if __name__ == "__main__":
    main()