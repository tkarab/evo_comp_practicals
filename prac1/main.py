from __future__ import annotations

import io
import os
import statistics
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ga_functions import run_ga
from helper_functions import (
    CrossoverType,
    fitness_counting_ones,
    trap_function_tightly_linked,
    trap_function_non_tightly_linked,
)

#configs

FitnessFnName = Literal["CountingOnes", "TrapTight", "TrapNonTight"]
ExperimentName = Literal["Ex1", "Ex2", "Ex3", "Ex4", "Ex5"]

L_DEFAULT = 40
N_START_DEFAULT = 10
N_CAP_DEFAULT = 1280
N_RUNS_DEFAULT = 10

@dataclass(frozen=True)
class ExperimentConfig:
    fitness_fn_name: FitnessFnName
    max_failures: int = 20
    k: Optional[int] = None
    d: Optional[float] = None

EXPERIMENTS: Dict[ExperimentName, ExperimentConfig] = {
    "Ex1": ExperimentConfig(fitness_fn_name="CountingOnes"),
    "Ex2": ExperimentConfig(fitness_fn_name="TrapTight", k=4, d=1),
    "Ex3": ExperimentConfig(fitness_fn_name="TrapTight", k=4, d=2.5),
    "Ex4": ExperimentConfig(fitness_fn_name="TrapNonTight", k=4, d=1),
    "Ex5": ExperimentConfig(fitness_fn_name="TrapNonTight", k=4, d=2.5),
}

@dataclass
class TrialResult:
    trial_idx: int
    ok: bool
    generations: int
    evaluations: int
    time_ms: float

@dataclass
class TrialSummary:
    N: int
    results: List[TrialResult]

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def solved(self) -> bool:
        return self.success_count >= 9

def mean_sd(xs: List[Union[int, float]]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))

def as_multiple_of_10(x: int) -> int:
    return (x // 10) * 10

def make_fitness_fn(ex_config: ExperimentConfig) -> Callable[[np.ndarray], np.ndarray]:
    if ex_config.fitness_fn_name == "CountingOnes":
        return fitness_counting_ones

    if ex_config.k is None or ex_config.d is None:
        raise ValueError(f"Trap fitness requires k and d, got k={ex_config.k}, d={ex_config.d}")

    if ex_config.fitness_fn_name == "TrapTight":
        return partial(trap_function_tightly_linked, k=ex_config.k, d=ex_config.d)

    if ex_config.fitness_fn_name == "TrapNonTight":
        return partial(trap_function_non_tightly_linked, k=ex_config.k, d=ex_config.d)

    raise ValueError(f"Unknown fitness_fn_name")

# GA RUNNERS

def run_ga_silent(
    N: int,
    l: int,
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    crossover: CrossoverType,
    max_failures: int,
) -> Tuple[bool, int, int, float]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _, ok, gens, fit_evals, total_time = run_ga(
            N=N,
            l=l,
            fitness_fn=fitness_fn,
            crossover=crossover,
            max_failures=max_failures,
        )
    return bool(ok), int(gens), int(fit_evals), float(total_time)

def run_trials(
    N: int,
    l: int,
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    crossover: CrossoverType,
    max_failures: int,
    n_runs: int = N_RUNS_DEFAULT,
) -> TrialSummary:
    results: List[TrialResult] = []
    for t in range(1, n_runs + 1):
        ok, gens, fit_evals, total_time_sec = run_ga_silent(
            N=N,
            l=l,
            fitness_fn=fitness_fn,
            crossover=crossover,
            max_failures=max_failures,
        )
        results.append(
            TrialResult(
                trial_idx=t,
                ok=ok,
                generations=gens,
                evaluations=fit_evals,
                time_ms=1000.0 * total_time_sec,
            )
        )
    return TrialSummary(N=N, results=results)

def search_optimal_N(
    l: int,
    fitness_fn: Callable[[np.ndarray], np.ndarray],
    crossover: CrossoverType,
    max_failures: int,
    n_runs: int = N_RUNS_DEFAULT,
    N_start: int = N_START_DEFAULT,
    N_cap: int = N_CAP_DEFAULT,
) -> Tuple[Optional[int], List[TrialSummary], Optional[TrialSummary]]:
    tried: List[TrialSummary] = []

    N = N_start
    summary = run_trials(N, l, fitness_fn, crossover, max_failures=max_failures, n_runs=n_runs)
    tried.append(summary)
    if summary.solved:
        return N, tried, summary
    N_low = N
    N_high: Optional[int] = None
    final_summary: Optional[TrialSummary] = None

    while True:
        print(f"N={N} failed ({summary.success_count}/{n_runs}), trying {2 * N}")
        N = 2 * N
        if N > N_cap:
            return None, tried, None
        summary = run_trials(N, l, fitness_fn, crossover, max_failures=max_failures, n_runs=n_runs)
        tried.append(summary)
        if summary.solved:
            N_high = N
            final_summary = summary
            break
        N_low = N

    N_low = N_high // 2
    print("Bisection search")

    while (N_high - N_low) > 10:
        N_new_raw = (N_low + N_high) // 2
        N_new = as_multiple_of_10(N_new_raw)

        if N_new <= N_low:
            N_new = N_low + 10
        if N_new >= N_high:
            N_new = N_high - 10

        print(f"Trying N={N_new}")
        summary = run_trials(N_new, l, fitness_fn, crossover, max_failures=max_failures, n_runs=n_runs)
        tried.append(summary)

        if summary.solved:
            N_high = N_new
            final_summary = summary
            print(f"New best: N={N_new} ({summary.success_count}/{n_runs})")
        else:
            N_low = N_new

    return N_high, tried, final_summary

# Report saving

def print_search_trace(tried: List[TrialSummary], n_runs: int) -> None:
    print("\nTrace:")
    for s in tried:
        ok = "SOLVED" if s.solved else "FAIL"
        tms = [r.time_ms for r in s.results]
        t_mean, t_sd = mean_sd(tms)
        print(f"  N={s.N:4d}  success={s.success_count}/{n_runs}  {ok}  time_mean={t_mean:.2f}ms  time_sd={t_sd:.2f}ms")

def save_run_details_csv(
    ex_name: str,
    crossover: CrossoverType,
    n_final: int,
    n_final_summary: TrialSummary,
    out_dir: str = "results",
) -> str:
    rows = [
        {
            "Trial": r.trial_idx,
            "OK": int(r.ok),
            "Generations": r.generations,
            "Evaluations": r.evaluations,
            "Time_ms": r.time_ms,
        }
        for r in n_final_summary.results
    ]
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"Results_{ex_name}_N_{n_final}_crossover_{crossover}.csv")
    df.to_csv(path, index=False)
    return path

def summarize_for_report(n_final_summary: TrialSummary) -> Dict[str, float]:
    gens = [r.generations for r in n_final_summary.results]
    evals = [r.evaluations for r in n_final_summary.results]
    times = [r.time_ms for r in n_final_summary.results]

    gen_mean, gen_sd = mean_sd([float(x) for x in gens])
    eval_mean, eval_sd = mean_sd([float(x) for x in evals])
    time_mean, time_sd = mean_sd([float(x) for x in times])

    return {
        "Gen_mean": gen_mean,
        "Gen_sd": gen_sd,
        "Eval_mean": eval_mean,
        "Eval_sd": eval_sd,
        "Time_ms_mean": time_mean,
        "Time_ms_sd": time_sd,
    }

# Main: running all combinations

def main() -> None:
    os.makedirs("results", exist_ok=True)

    l = L_DEFAULT
    n_runs = N_RUNS_DEFAULT
    N_start = N_START_DEFAULT
    N_cap = N_CAP_DEFAULT

    crossovers: List[CrossoverType] = ["UX", "2X"]
    summary_rows: List[dict] = []

    for ex_name, ex_config in EXPERIMENTS.items():
        fitness_fn = make_fitness_fn(ex_config)

        print(f"Experiment: {ex_name}")
        print(f"Fitness: {ex_config.fitness_fn_name}")
        print(f"k={ex_config.k}  d={ex_config.d}")
        print(f"max_failures={ex_config.max_failures}  runs={n_runs}")

        for crossover in crossovers:
            print(f"\n-- Crossover: {crossover} --")

            n_final, tried, n_final_summary = search_optimal_N(
                l=l,
                fitness_fn=fitness_fn,
                crossover=crossover,
                max_failures=ex_config.max_failures,
                n_runs=n_runs,
                N_start=N_start,
                N_cap=N_cap,
            )

            if n_final is None or n_final_summary is None:
                print_search_trace(tried, n_runs)
                print(f"RESULT: FAIL (exceeded N_cap={N_cap})")

                summary_rows.append(
                    {
                        "Experiment": ex_name,
                        "Crossover": crossover,
                        "FitnessFn": ex_config.fitness_fn_name,
                        "k": ex_config.k,
                        "d": ex_config.d,
                        "N_final": "FAIL",
                        "Successes": "",
                        "Gen_mean": "",
                        "Gen_sd": "",
                        "Eval_mean": "",
                        "Eval_sd": "",
                        "Time_ms_mean": "",
                        "Time_ms_sd": "",
                    }
                )
                continue

            details_csv = save_run_details_csv(
                ex_name=ex_name,
                crossover=crossover,
                n_final=n_final,
                n_final_summary=n_final_summary,
                out_dir="results",
            )
            stats = summarize_for_report(n_final_summary)

            print_search_trace(tried, n_runs)
            print(f"RESULT: N_final={n_final}  success={n_final_summary.success_count}/{n_runs}")
            print(f"Saved: {details_csv}")

            summary_rows.append(
                {
                    "Experiment": ex_name,
                    "Crossover": crossover,
                    "FitnessFn": ex_config.fitness_fn_name,
                    "k": ex_config.k,
                    "d": ex_config.d,
                    "N_final": n_final,
                    "Successes": f"{n_final_summary.success_count}/{n_runs}",
                    **stats,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join("results", "SUMMARY_all_experiments.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path}")

if __name__ == "__main__":
    main()