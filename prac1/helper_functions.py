import numpy as np
from typing import Literal, Tuple


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

