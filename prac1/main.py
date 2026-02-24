"""
from functools import partial

from helper_functions import trap_function_tightly_linked, trap_function_non_tightly_linked
from ga_functions import run_ga

k = 4
d = 2.5

fitness_tight = partial(trap_function_tightly_linked, k=k, d=d)
fitness_nontight = partial(trap_function_non_tightly_linked, k=k, d=d)

final_population, ok, gens = run_ga(N=10, l=12, fitness_fn=fitness_nontight, crossover="UX")
"""

from __future__ import annotations
from functools import partial
import io
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from ga_functions import run_ga
from helper_functions import CrossoverType, fitness_counting_ones, trap_function_tightly_linked, trap_function_non_tightly_linked


@dataclass
class TrialSummary:
    N: int
    success_count: int
    generations_success: List[int]
    wall_time_s: float

    @property
    def solved(self) -> bool:
        return self.success_count >= 9


def run_ga_silent(N: int, l: int, fitness_fn: Callable[[np.ndarray], np.ndarray], crossover: CrossoverType) -> Tuple[bool, int]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _, ok, gens = run_ga(N=N, l=l, fitness_fn=fitness_fn, crossover=crossover, max_failures=20)
    return bool(ok), int(gens)


def run_10_trials(
    N: int,
    l: int,
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    crossover: CrossoverType,
) -> TrialSummary:
    t0 = time.perf_counter()
    success_count = 0
    generations_success: List[int] = []

    for _ in range(10):
        ok, gens = run_ga_silent(N, l, fitness_fn, crossover)
        if ok:
            success_count += 1
            generations_success.append(gens)

    t1 = time.perf_counter()
    return TrialSummary(N=N, success_count=success_count, generations_success=generations_success, wall_time_s=(t1 - t0))


def as_multiple_of_10(x: int) -> int:
    return (x // 10) * 10


def search_optimal_N(
    l: int,
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    crossover: CrossoverType,
    N_start: int = 10,
    N_cap: int = 1280,
) -> Tuple[int | None, List[TrialSummary]]:
    results: List[TrialSummary] = []

    # initial run for N=10
    N = N_start
    summary = run_10_trials(N, l, fitness_fn, crossover)
    results.append(summary)

    if summary.solved:
        return N, results

    # Double until find N_high (first success) or hit cap
    N_low = N
    N_high = None

    while True:
        N = N * 2
        if N > N_cap:
            return None, results  # Fail
        summary = run_10_trials(N, l, fitness_fn, crossover)
        results.append(summary)
        if summary.solved:
            N_high = N
            break
        N_low = N

    # By flowchart: N_low = N_high / 2
    N_low = N_high // 2

    # Bisection until N_final
    # Maintain invariant: N_low fails, N_high succeeds
    while (N_high - N_low) > 10:
        N_new_raw = (N_low + N_high) // 2
        N_new = as_multiple_of_10(N_new_raw)

        if N_new <= N_low:
            N_new = N_low + 10
        if N_new >= N_high:
            N_new = N_high - 10

        summary = run_10_trials(N_new, l, fitness_fn, crossover)
        results.append(summary)

        if summary.solved:
            N_high = N_new
        else:
            N_low = N_new

    return N_high, results


def print_results(results: List[TrialSummary]) -> None:
    print("\n main: search for optimal N ")
    for r in results:
        gens_list = ", ".join(str(g) for g in r.generations_success)
        print(f"N={r.N:4d}  success={r.success_count}/10  solved={r.solved}  "
              f"time={r.wall_time_s:.3f}s  gens(successes)=[{gens_list}]")


def main() -> None:
    N = 10
    l = 12
    crossover = "UX"
    k = 4
    d = 2.5

    fitness_function = [
        ("CountingOnes", fitness_counting_ones),
        ("TrapTight", partial(trap_function_tightly_linked, k=k, d=d)),
        ("TrapNonTight", partial(trap_function_non_tightly_linked, k=k, d=d)),
    ]

    for name, fitness_fn in fitness_function:
        final_population, ok, gens = run_ga(
            N=N,
            l=l,
            fitness_fn=fitness_fn,
            crossover=crossover
        )
        print(f"{name}: ok={ok}, gens={gens}, best_fitness={fitness_fn(final_population).max()}")


"""
def main() -> None:
    l = 40
    crossover = "UX"
    k = 4
    d = 2.5

    fitness_function = [
        ("CountingOnes", fitness_counting_ones),
        ("TrapTight", partial(trap_function_tightly_linked, k=k, d=d)),
        ("TrapNonTight", partial(trap_function_non_tightly_linked, k=k, d=d)),
    ]

    for name, fitness_fn in fitness_function:
        print(f"\n--- {name} ---")
        N_final, results = search_optimal_N(
            l=l,
            fitness_fn=fitness_fn,
            crossover=crossover,
            N_start=10,
            N_cap=1280
        )
        print_results(results)
        print("N_final:", N_final)
"""


if __name__ == "__main__":
    main()
