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

import os
from functools import partial
import io
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Callable, List, Tuple, Literal, Optional
import statistics

import numpy as np
import pandas as pd

from ga_functions import run_ga
from helper_functions import CrossoverType, fitness_counting_ones, trap_function_tightly_linked, trap_function_non_tightly_linked

FitnessFnName = Literal["CountingOnes","TrapTight","TrapNonTight"]

@dataclass(frozen=True)
class ExperimentConfig:
    fitness_fn_name: FitnessFnName
    max_failures: int = 20
    k: Optional[int] = None
    d: Optional[float] = None

ExperimentName = Literal["Ex1", "Ex2", "Ex3", "Ex4", "Ex5"]
CHOSEN_EX : ExperimentName = "Ex1"

EXPERIMENTS = {
    "Ex1" : ExperimentConfig(fitness_fn_name="CountingOnes"),
    "Ex2" : ExperimentConfig(fitness_fn_name="TrapTight", k=4, d=1),
    "Ex3" : ExperimentConfig(fitness_fn_name="TrapTight", k=4, d=2.5),
    "Ex4" : ExperimentConfig(fitness_fn_name="TrapNonTight", k=4, d=1),
    "Ex5" : ExperimentConfig(fitness_fn_name="TrapNonTight", k=4, d=2.5),
}


@dataclass
class TrialSummary:
    N: int
    success_count: int
    generations_success: List[int]  # List of the total generations it took for each successful trial
    evaluations_list : List[int]     # List of total fitness evaluations in the run
    trial_times: List[float]    # List of the time it took to run each successful trial

    @property
    def solved(self) -> bool:
        return self.success_count >= 9


def run_ga_silent(N: int, l: int, fitness_fn: Callable[[np.ndarray], np.ndarray], crossover: CrossoverType) -> Tuple[bool, int, int, float]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _, ok, gens, fit_evals, total_time = run_ga(N=N, l=l, fitness_fn=fitness_fn, crossover=crossover, max_failures=20)
    return bool(ok), int(gens), int(fit_evals), float(total_time)


def run_10_trials(
    N: int,
    l: int,
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    crossover: CrossoverType,
) -> TrialSummary:
    success_count = 0
    generations_success: List[int] = []
    evaluations: List[int] = []
    trial_times: List[float] = []

    for _ in range(10):
        ok, gens, fit_evals, total_time = run_ga_silent(N, l, fitness_fn, crossover)
        if ok:
            success_count += 1
            generations_success.append(gens)
            evaluations.append(fit_evals)
            trial_times.append(1000*total_time)


    return TrialSummary(N=N, success_count=success_count, generations_success=generations_success, evaluations_list=evaluations, trial_times=trial_times)


def as_multiple_of_10(x: int) -> int:
    return (x // 10) * 10


def search_optimal_N(
    l: int,
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    crossover: CrossoverType,
    N_start: int = 10,
    N_cap: int = 1280,
) -> Tuple[int | None, List[TrialSummary], TrialSummary]:
    results: List[TrialSummary] = []
    n_final_summary : TrialSummary = None

    # initial run for N=10
    N = N_start
    summary = run_10_trials(N, l, fitness_fn, crossover)
    results.append(summary)

    if summary.solved:
        n_final_summary = summary
        return N, results, n_final_summary

    # Double until find N_high (first success) or hit cap
    N_low = N
    N_high = None

    while True:
        print(f"N={N} failed, trying {2*N}")
        N = N * 2
        if N > N_cap:
            return None, results, None  # Fail
        summary = run_10_trials(N, l, fitness_fn, crossover)
        results.append(summary)
        if summary.solved:
            n_final_summary = summary
            N_high = N
            break
        N_low = N

    # By flowchart: N_low = N_high / 2
    N_low = N_high // 2

    # Bisection until N_final
    # Maintain invariant: N_low fails, N_high succeeds
    print("Bisection search")
    while (N_high - N_low) > 10:
        N_new_raw = (N_low + N_high) // 2
        N_new = as_multiple_of_10(N_new_raw)
        print(f"Trying {N_new}")

        if N_new <= N_low:
            N_new = N_low + 10
        if N_new >= N_high:
            N_new = N_high - 10

        summary = run_10_trials(N_new, l, fitness_fn, crossover)
        results.append(summary)

        if summary.solved:
            n_final_summary = summary
            N_high = N_new
            print(f"New best: {N_new}")
        else:
            N_low = N_new

    return N_high, results, n_final_summary


def save_N_final_summary_results(n_final:int, n_final_summary:TrialSummary, crossover_method) -> None:
    results_df = pd.DataFrame({
        "Generations": n_final_summary.generations_success,
        "Evaluations": n_final_summary.evaluations_list,
        "Trial Times": n_final_summary.trial_times,
    })

    filename = f"Results_{CHOSEN_EX}_N_{n_final}_crossover_{crossover_method}.csv"
    full_path = os.path.join("results", filename)

    results_df.to_csv(full_path, index=False)

    print(f"Results saved to {full_path}")

    return None


def print_results(results: List[TrialSummary]) -> None:
    print("\n main: search for optimal N ")
    for r in results:
        gens_list = ", ".join(str(g) for g in r.generations_success)
        print(f"N={r.N:4d}  success={r.success_count}/10  solved={r.solved}  "
              f"average time={(statistics.mean(r.trial_times) if r.trial_times else 0):.3f}ms  gens(successes)=[{gens_list}]")


# def main() -> None:
#     N = 10
#     l = 12
#     crossover = "UX"
#     k = 4
#     d = 2.5
#
#     fitness_function = [
#         ("CountingOnes", fitness_counting_ones),
#         ("TrapTight", partial(trap_function_tightly_linked, k=k, d=d)),
#         ("TrapNonTight", partial(trap_function_non_tightly_linked, k=k, d=d)),
#     ]
#
#     for name, fitness_fn in fitness_function:
#         final_population, ok, gens, evals, total_time = run_ga(
#             N=N,
#             l=l,
#             fitness_fn=fitness_fn,
#             crossover=crossover
#         )
#         print(f"{name}: ok={ok}, gens={gens}, best_fitness={fitness_fn(final_population).max()}")


def main() -> None:
    l = 40

    Ex_config : ExperimentConfig = EXPERIMENTS[CHOSEN_EX]

    k = Ex_config.k
    d = Ex_config.d

    fitness_function = {
        "CountingOnes": fitness_counting_ones,
        "TrapTight": partial(trap_function_tightly_linked, k=k, d=d),
        "TrapNonTight": partial(trap_function_non_tightly_linked, k=k, d=d),
    }

    fitness_fn = fitness_function[Ex_config.fitness_fn_name]


    # for name, fitness_fn in fitness_function:
    print(f"Experiment: {CHOSEN_EX}\nConfigurations:"
          f"\n\tFitness: {Ex_config.fitness_fn_name}"
          f"\n\tk: {Ex_config.k}"
          f"\n\td: {Ex_config.d}")

    N_start = 10
    N_max = 1280

    for crossover in ["UX", "2X"]:
        print(f"\n\nCrossover: {crossover}")

        N_final, results, n_final_summary = search_optimal_N(
            l=l,
            fitness_fn=fitness_fn,
            crossover=crossover,
            N_start=N_start,
            N_cap=N_max
        )
        if N_final is not None:
            save_N_final_summary_results(n_final=N_final, n_final_summary=n_final_summary, crossover_method=crossover)
            print_results(results)
            print(f"N_final ({crossover}):", N_final)
        else:
            print(f"Experiment {CHOSEN_EX} failed for Crossover: {crossover}, N exceeded {N_max}")



if __name__ == "__main__":
    main()
