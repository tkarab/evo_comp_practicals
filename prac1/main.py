from functools import partial

from helper_functions import trap_function_tightly_linked, trap_function_non_tightly_linked
from ga_functions import run_ga

k = 4
d = 2.5

fitness_tight = partial(trap_function_tightly_linked, k=k, d=d)
fitness_nontight = partial(trap_function_non_tightly_linked, k=k, d=d)

final_population, ok, gens = run_ga(N=10, l=12, fitness_fn=fitness_nontight, crossover="UX")