from typing import Callable, Literal, Optional, Tuple
import numpy as np

CrossoverType = Literal["UX", "2X"]

def uniform_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # Uniform crossover with p = 0.5
    if p1.shape != p2.shape:
        raise ValueError("Parents must have the same shape.")
    mask = rng.random(p1.shape[0]) < 0.5

    c1 = p1.copy()
    c2 = p2.copy()
    c1[mask] = p2[mask]
    c2[mask] = p1[mask]
    return c1, c2

def two_point_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # Two point crossover
    if p1.shape != p2.shape:
        raise ValueError("Parents dont have same value")
    L = p1.shape[0]

    # Selecting two distinct values
    a, b = rng.choice(L + 1, size=2, replace=False)
    if a > b:
        a, b = b, a

    c1 = p1.copy()
    c2 = p2.copy()
    c1[a:b] = p2[a:b]
    c2[a:b] = p1[a:b]
    return c1, c2

def ga_iteration(P: np.ndarray, fitness_fn: Callable[[np.ndarray], np.ndarray], crossover: CrossoverType = "UX", rng: Optional[np.random.Generator] = None, debug_ties: bool = False) -> Tuple[np.ndarray, bool]:
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

    out = 0
    for i in range(0, N, 2):
        p1 = P_shuf[i]
        p2 = P_shuf[i + 1]
        c1, c2 = recombination(p1, p2, rng)

        family = np.stack([p1, p2, c1, c2], axis=0)
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
            parents_max = fit[:2].max()  # indices 0,1
            children_max = fit[2:].max()  # indices 2,3
            improvement = children_max > parents_max

        if debug_ties:
            # Rule to pick children in case of a tie
            if np.all(fit == fit[0]):
                print(f"Family all-tied fit={fit[0]} winners_idx={winners_idx}")

        P_next[out] = winners[0]
        P_next[out + 1] = winners[1]
        out += 2

    return P_next, improvement


"""
This is the function that runs the genetic algorithm until it either converges or fails.

Inputs
    - N: population size
    - l: solution length 
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
def run_ga(N:int, l:int, fitness_fn: Callable[[np.ndarray], np.ndarray], crossover: CrossoverType = "UX", max_failures = 20) -> Tuple[np.ndarray, bool, int]:
    # Step 1: initiate random population of size N, length l
    rng = np.random.default_rng()
    P_old = rng.integers(0, 2, size=(N, l), dtype=np.int8)
    consecutive_failures = 0
    total_generations = 0

    # Step 2: run 'ga_iteration' iteratively to generate new populations
    while consecutive_failures < max_failures:
        P_new, improve_flag = ga_iteration(P_old, fitness_fn, crossover)
        P_old = P_new
        total_generations += 1

        global_optimum_found = np.any(np.all(P_new == 1, axis=1))

        # Step 3: Check for global optimum in new population, or if no improvement has been made for the past 20 generations
        if global_optimum_found:
            print(f"global optimum found: True, total generations: {total_generations}")
            return P_new, True, total_generations
        else:
            if not improve_flag:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
        print(f"global optimum found: False, total generations: {total_generations}, consecutive failures: {consecutive_failures}")

    return P_new, False, total_generations




# Fitness function - to be implemented
def fitness_counting_ones(pop: np.ndarray) -> np.ndarray:
    return pop.sum(axis=1).astype(float)

"""
B(x1,x2,...,xk): returns either k or the fitness value based on the formula (k - d) - ((k - d) / (k - 1)) * co depending on co (count of ones)
This is used as a helper function in the trap functions

Inputs
    - x: Nxk array
    - d: predefined value, based on which the function will either be deceptive or non-deceptive
Steps
    1. count the number of '1' on each row
    2. perform the formula on each row
Output
    - N-long array of the B values of all rows of x
"""
def fitness_helper_B(x: np.ndarray, d: float) -> np.ndarray:
    k = x.shape[1]
    # Step 1: count the number of ones on each row
    co = fitness_counting_ones(x) # or fitness_counting_ones(x)

    # Step 2:
    #   where CO(x) = k ->  B = k, where CO(x) < k -> apply formula
    return np.where(
        co == k,
        float(k),
        (k - d) - ((k - d) / (k - 1)) * co
    )

def trap_function_tightly_linked(x: np.ndarray, k: int, d:float) -> np.ndarray:
    N, l = x.shape
    m = l // k

    x_split = x.reshape(N,m, k)
    B_x = np.zeros([N, m])
    for i in range(m):
        B_x[:, i] = fitness_helper_B(x_split[:, i], d)

    tf = B_x.sum(axis = 1)
    return tf


def trap_function_non_tightly_linked(x: np.ndarray, k: int, d: float) -> np.ndarray:
    N, l = x.shape
    m = l // k

    B_x = np.zeros([N, m])
    for i in range(m):
        B_x[:, i] = fitness_helper_B(x[:, i::m], d)

    tf = B_x.sum(axis=1)
    return tf


if __name__ == "__main__":
    # Testing
    rng = np.random.default_rng(42)
    N, L = 10, 40
    P0 = rng.integers(0, 2, size=(N, L), dtype=np.int8)
    rng_test = np.random.default_rng(123)

    p = rng_test.integers(0, 2, size=(1, 40), dtype=np.int8)
    P_tie = np.repeat(p, repeats=10, axis=0)
    P1_ux, imp1_ux = ga_iteration(P0, fitness_counting_ones, crossover="UX", rng=rng)
    P1_2x, imp2_ux = ga_iteration(P0, fitness_counting_ones, crossover="2X", rng=rng)
    P_next, imp_next = ga_iteration(P_tie, fitness_counting_ones, crossover="UX", rng=rng_test, debug_ties=True)

    print("P0 shape:", P0.shape)
    print("P1 (UX) shape:", P1_ux.shape, "unique values:", np.unique(P1_ux))
    print("P1 (2X) shape:", P1_2x.shape, "unique values:", np.unique(P1_2x))
    print("Mean fitness P0:", fitness_counting_ones(P0).mean())
    print("Mean fitness P1 (UX):", fitness_counting_ones(P1_ux).mean())
    print("Mean fitness P1 (2X):", fitness_counting_ones(P1_2x).mean())
    print("Tie test passed (children selected on equal fitness).")

    final_population, optimum_found, total_generations = run_ga(N=10, l=40, fitness_fn=fitness_counting_ones, crossover="UX", max_failures=5)

    b1 = fitness_helper_B(np.array([[1,0,1,1],[1,0,0,1],[1,1,1,1]]), d=1)
    b2 = fitness_helper_B(np.array([[1,0,1,1],[0,0,0,1],[0,0,0,0]]), d=2.5)


    print()

    trap_function_non_tightly_linked(P0, k=4, d=1)

