from typing import Callable, Literal, Optional, Tuple
import numpy as np
from functools import partial
import time

# import Crossover functions
from helper_functions import CrossoverType, uniform_crossover, two_point_crossover
# Import Fitnes functions
from helper_functions import fitness_counting_ones, fitness_helper_B, trap_function_tightly_linked, trap_function_non_tightly_linked

FitnessFn = Callable[[np.ndarray], np.ndarray]

def ga_iteration(P: np.ndarray, fitness_fn: FitnessFn, crossover: CrossoverType = "UX", rng: Optional[np.random.Generator] = None, debug_ties: bool = False) -> Tuple[np.ndarray, bool, int]:
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

        # if not improvement:
        #     if 2 in winners_idx or 3 in winners_idx:
        #         improvement = True

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
This is the function that runs the genetic algorithm until it either converges or fails.

Inputs
    - N: population size
    - l: solution length 
    - k: sub-function length (for non-
    - fitness_fn: fitness function to be used
    - crossover: crossover type

Steps
    1. initialises a random population of size N, where each individual is a binary vector of length l
    2. runs ga_iteration repeatedly to generate the next population P(t+1)
    3. stops if:
        - '111..11' is part of P(t+1), or
        - there has been no improvement for 20 consecutive generations (based on the "improve_flag" from ga_iteration)
    4. returns the final population and a true/false flag to indicate success/failure

Output
    - final population
    - True / False flag to indicate if optimum found (found -> True, 20 failures -> False)
    - total generations
"""
def run_ga(N:int, l:int, fitness_fn: FitnessFn, crossover: CrossoverType = "UX", max_failures = 20) -> Tuple[np.ndarray, bool, int, int, float]:
    # Step 1: initiate random population of size N, length l
    rng = np.random.default_rng()
    P_old = rng.integers(0, 2, size=(N, l), dtype=np.int8)
    consecutive_failures = 0
    total_generations = 0
    total_fitness_evaluations = 0
    t0 = time.perf_counter()

    # Step 2: run 'ga_iteration' iteratively to generate new populations
    while consecutive_failures < max_failures:
        P_new, improve_flag, iter_evaluations = ga_iteration(P_old, fitness_fn, crossover=crossover, rng=rng)
        P_old = P_new
        total_generations += 1
        total_fitness_evaluations += iter_evaluations


        global_optimum_found = np.any(np.all(P_new == 1, axis=1))

        # Step 3: Check for global optimum in new population, or if no improvement has been made for the past 20 generations
        if global_optimum_found:
            total_time = time.perf_counter() - t0
            print(f"global optimum found: True, total generations: {total_generations}")
            return P_new, True, total_generations, total_fitness_evaluations, total_time
        else:
            if not improve_flag:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
        print(f"global optimum found: False, total generations: {total_generations}, consecutive failures: {consecutive_failures}")

    total_time = time.perf_counter() - t0
    return P_new, False, total_generations, total_fitness_evaluations, total_time





if __name__ == "__main__":
    # Testing
    # rng = np.random.default_rng(42)
    # N, L = 10, 40
    # P0 = rng.integers(0, 2, size=(N, L), dtype=np.int8)
    # rng_test = np.random.default_rng(123)
    #
    # p = rng_test.integers(0, 2, size=(1, 40), dtype=np.int8)
    # P_tie = np.repeat(p, repeats=10, axis=0)
    # P1_ux, imp1_ux = ga_iteration(P0, fitness_counting_ones, crossover="UX", rng=rng)
    # P1_2x, imp2_ux = ga_iteration(P0, fitness_counting_ones, crossover="2X", rng=rng)
    # P_next, imp_next = ga_iteration(P_tie, fitness_counting_ones, crossover="UX", rng=rng_test, debug_ties=True)
    #
    # print("P0 shape:", P0.shape)
    # print("P1 (UX) shape:", P1_ux.shape, "unique values:", np.unique(P1_ux))
    # print("P1 (2X) shape:", P1_2x.shape, "unique values:", np.unique(P1_2x))
    # print("Mean fitness P0:", fitness_counting_ones(P0).mean())
    # print("Mean fitness P1 (UX):", fitness_counting_ones(P1_ux).mean())
    # print("Mean fitness P1 (2X):", fitness_counting_ones(P1_2x).mean())
    # print("Tie test passed (children selected on equal fitness).")
    #
    # k = 4
    # d = 2.5
    #
    # fitness_tight = partial(trap_function_tightly_linked, k=k, d=d)
    # fitness_nontight = partial(trap_function_non_tightly_linked, k=k, d=d)
    #
    # final_population, ok, gens, evals, total_time = run_ga(N=10, l=12, fitness_fn=fitness_nontight, crossover="UX")
    print()

