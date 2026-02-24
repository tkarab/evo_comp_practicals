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


# Fitness function - count of ones
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

"""
Trap Function (tightly liked). Takes an input of N solutions of length l. Divides each one in m = l/k sub-functions of length k (in a tightly linked way). Applies Fitness function B to each sub-function and sums up all sub-function values of each solution to find final solution fitness

Inputs
    - x: Nxl array -> N solutions of length l
    - k: the length of each sub-function
    - d: the d value to use when applying fitness function B -> determines whether trap function is deceptive or not
Steps
    1. Find m = l/k 
    2. divide each of the N solutions into m sub-functions of length l
    3. Apply B() to each sub-function
    4. Sum all sub-functions of each solution
Output
    - N scalar fitness values, one for each solution
    
"""
def trap_function_tightly_linked(x: np.ndarray, k: int, d:float) -> np.ndarray:
    N, l = x.shape
    m = l // k

    # Splitting each row of x (meaning each of the N solutions of length l) into m sub-functions of length k using reshaping
    x_split = x.reshape(N,m, k)
    B_x = np.zeros([N, m])
    # Applying function B to each sub-function for all N solutions
    for i in range(m):
        B_x[:, i] = fitness_helper_B(x_split[:, i], d)

    # Summing up all subfunction values for each solution
    tf = B_x.sum(axis = 1)

    return tf


"""
Trap Function (non-tightly liked). Takes an input of N solutions of length l. Divides each one in m = l/k sub-functions of length k (in a non-tightly linked way). Applies Fitness function B to each sub-function and sums up all sub-function values of each solution to find final solution fitness

Inputs
    - x: Nxl array -> N solutions of length l
    - k: the length of each sub-function
    - d: the d value to use when applying fitness function B -> determines whether trap function is deceptive or not
Steps
    1. Find m = l/k 
    2. divide each of the N solutions into m sub-functions of length l, using stride m and starting point i=0,1,...,m-1 to achieve non-tight linkage
    3. Apply B() to each sub-function
    4. Sum all sub-functions of each solution
Output
    - N scalar fitness values, one for each solution

"""
def trap_function_non_tightly_linked(x: np.ndarray, k: int, d: float) -> np.ndarray:
    N, l = x.shape
    m = l // k

    B_x = np.zeros([N, m])

    for i in range(m):
        # Dividing each solution into sub-functions of length k, (using stride m, starting from 0,1,....m-1) and applying B
        B_x[:, i] = fitness_helper_B(x[:, i::m], d)

    # Summing up all subfunction values for each solution
    tf = B_x.sum(axis=1)

    return tf

