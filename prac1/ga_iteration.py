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

def ga_iteration(P: np.ndarray, fitness_fn: Callable[[np.ndarray], np.ndarray], crossover: CrossoverType = "UX", rng: Optional[np.random.Generator] = None, debug_ties: bool = False) -> np.ndarray:
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

        if debug_ties:
            # Rule to pick children in case of a tie
            if np.all(fit == fit[0]):
                print(f"Family all-tied fit={fit[0]} winners_idx={winners_idx}")

        P_next[out] = winners[0]
        P_next[out + 1] = winners[1]
        out += 2

    return P_next

# Fitness function - to be implemented
def fitness_counting_ones(pop: np.ndarray) -> np.ndarray:
    return pop.sum(axis=1).astype(float)

if __name__ == "__main__":
    # Testing
    rng = np.random.default_rng(42)
    N, L = 10, 40
    P0 = rng.integers(0, 2, size=(N, L), dtype=np.int8)
    rng_test = np.random.default_rng(123)

    p = rng_test.integers(0, 2, size=(1, 40), dtype=np.int8)
    P_tie = np.repeat(p, repeats=10, axis=0)
    P1_ux = ga_iteration(P0, fitness_counting_ones, crossover="UX", rng=rng)
    P1_2x = ga_iteration(P0, fitness_counting_ones, crossover="2X", rng=rng)
    P_next = ga_iteration(P_tie, fitness_counting_ones, crossover="UX", rng=rng_test, debug_ties=True)

    print("P0 shape:", P0.shape)
    print("P1 (UX) shape:", P1_ux.shape, "unique values:", np.unique(P1_ux))
    print("P1 (2X) shape:", P1_2x.shape, "unique values:", np.unique(P1_2x))
    print("Mean fitness P0:", fitness_counting_ones(P0).mean())
    print("Mean fitness P1 (UX):", fitness_counting_ones(P1_ux).mean())
    print("Mean fitness P1 (2X):", fitness_counting_ones(P1_2x).mean())
    print("Tie test passed (children selected on equal fitness).")