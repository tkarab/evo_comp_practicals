from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from helper_functions import (
    CrossoverType,
    fitness_counting_ones,
    two_point_crossover,
    uniform_crossover,
)

FitnessFn = Callable[[np.ndarray], np.ndarray]


# Per-generation metrics container

@dataclass
class GenMetrics:
    prop_ones: float
    err: int
    corr: int
    schema1_count: int
    schema0_count: int
    schema1_mean: float
    schema1_sd: float
    schema0_mean: float
    schema0_sd: float


# One GA iteration

def ga_iteration_extra(
        P: np.ndarray,
        fitness_fn: FitnessFn,
        crossover: CrossoverType = "UX",
        rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, bool, int, int, int]:
    if rng is None:
        rng = np.random.default_rng()

    P = np.asarray(P)
    if P.ndim != 2:
        raise ValueError(f"P must be 2D")
    N, L = P.shape
    if N % 2 != 0:
        raise ValueError(f"Population size must be even")

    P_shuf = P[rng.permutation(N)]

    if crossover == "UX":
        recombination = uniform_crossover
    elif crossover == "2X":
        recombination = two_point_crossover
    else:
        raise ValueError("crossover must be UX or 2X")

    P_next = np.empty_like(P_shuf)
    is_child = np.array([0, 0, 1, 1], dtype=np.int8)

    fitness_evals = 0
    improvement = False
    err_t = 0
    corr_t = 0

    out = 0
    for i in range(0, N, 2):
        p1 = P_shuf[i]
        p2 = P_shuf[i + 1]
        c1, c2 = recombination(p1, p2, rng)

        family = np.stack([p1, p2, c1, c2], axis=0)
        fitness_evals += 4

        fit = np.asarray(fitness_fn(family))
        if fit.shape != (4,):
            raise ValueError(f"fitness wrong shape")

        order = np.lexsort((-is_child, -fit))
        winners_idx = order[:2]
        winners = family[winners_idx]

        if not improvement:
            parents_max = float(fit[:2].max())
            children_max = float(fit[2:].max())
            if children_max > parents_max:
                improvement = True

        # Err/Corr for family
        disagree = (p1 != p2)
        if np.any(disagree):
            wbits = winners[:, disagree]
            err_t += int(np.sum(np.all(wbits == 0, axis=0)))
            corr_t += int(np.sum(np.all(wbits == 1, axis=0)))

        P_next[out] = winners[0]
        P_next[out + 1] = winners[1]
        out += 2

    return P_next, improvement, fitness_evals, err_t, corr_t


# Schema stats

def _mean_sd(arr: np.ndarray) -> Tuple[float, float]:
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def schema_stats(P: np.ndarray, fitness_vals: np.ndarray) -> Tuple[int, int, float, float, float, float]:
    first = P[:, 0]
    idx1 = (first == 1)
    idx0 = ~idx1

    count1 = int(np.sum(idx1))
    count0 = int(np.sum(idx0))

    mean1, sd1 = _mean_sd(fitness_vals[idx1])
    mean0, sd0 = _mean_sd(fitness_vals[idx0])

    return count1, count0, mean1, sd1, mean0, sd0


# Run of GA

def run_ga_extra(
        N: int,
        L: int,
        fitness_fn: FitnessFn,
        crossover: CrossoverType = "UX",
        max_failures: int = 20,
        seed: Optional[int] = 42,
) -> Tuple[bool, int, int, float, List[GenMetrics]]:
    rng = np.random.default_rng(seed)

    P = rng.integers(0, 2, size=(N, L), dtype=np.int8)

    total_generations = 0
    total_evals = 0
    consecutive_failures = 0
    metrics: List[GenMetrics] = []

    t0 = time.perf_counter()

    fit0 = np.asarray(fitness_fn(P))
    prop0 = float(np.mean(fit0) / L)
    c1, c0, m1, s1, m0, s0 = schema_stats(P, fit0)
    metrics.append(GenMetrics(prop0, 0, 0, c1, c0, m1, s1, m0, s0))

    while consecutive_failures < max_failures:
        P, improved, evals, err_t, corr_t = ga_iteration_extra(P, fitness_fn, crossover=crossover, rng=rng)
        total_generations += 1
        total_evals += evals

        fit = np.asarray(fitness_fn(P))
        prop = float(np.mean(fit) / L)

        c1, c0, m1, s1, m0, s0 = schema_stats(P, fit)
        metrics.append(GenMetrics(prop, err_t, corr_t, c1, c0, m1, s1, m0, s0))

        if bool(np.all(P == 1)):
            total_time = time.perf_counter() - t0
            return True, total_generations, total_evals, total_time, metrics

        if not improved:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

    total_time = time.perf_counter() - t0
    return False, total_generations, total_evals, total_time, metrics


# Plotting

def plot_all(metrics: List[GenMetrics], out_dir: str = "plots") -> None:
    os.makedirs(out_dir, exist_ok=True)

    t = np.arange(len(metrics))

    prop = np.array([m.prop_ones for m in metrics], dtype=float)
    err = np.array([m.err for m in metrics], dtype=int)
    corr = np.array([m.corr for m in metrics], dtype=int)

    s1c = np.array([m.schema1_count for m in metrics], dtype=int)
    s0c = np.array([m.schema0_count for m in metrics], dtype=int)

    s1m = np.array([m.schema1_mean for m in metrics], dtype=float)
    s1s = np.array([m.schema1_sd for m in metrics], dtype=float)
    s0m = np.array([m.schema0_mean for m in metrics], dtype=float)
    s0s = np.array([m.schema0_sd for m in metrics], dtype=float)

    # 1) prop(t)
    plt.figure()
    plt.plot(t, prop)
    plt.title("prop(t): proportion of 1-bits in the population")
    plt.xlabel("Generation t")
    plt.ylabel("prop(t)")
    plt.ylim(0.0, 1.05)
    plt.savefig(os.path.join(out_dir, "1_prop_t.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Err(t) and Correct(t)
    plt.figure()
    plt.plot(t, err, label="Err(t)")
    plt.plot(t, corr, label="Correct(t)")
    plt.title("Selection decisions per generation")
    plt.xlabel("Generation t")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "2_err_corr_t.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3) Schema counts
    plt.figure()
    plt.plot(t, s1c, label="Schema 1******** (bit1=1)")
    plt.plot(t, s0c, label="Schema 0******** (bit1=0)")
    plt.title("Schema membership counts")
    plt.xlabel("Generation t")
    plt.ylabel("Count in population")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "3_schema_counts_t.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4) Schema mean fitness
    plt.figure()
    plt.plot(t, s1m, label="Schema1 mean")
    plt.plot(t, s0m, label="Schema0 mean")
    plt.title("Schema mean fitness")
    plt.xlabel("Generation t")
    plt.ylabel("Mean fitness")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "4_schema_mean_fitness_t.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 5) Schema fitness sd
    plt.figure()
    plt.plot(t, s1s, label="Schema1 sd")
    plt.plot(t, s0s, label="Schema0 sd")
    plt.title("Schema fitness standard deviation")
    plt.xlabel("Generation t")
    plt.ylabel("SD of fitness")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "5_schema_sd_fitness_t.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to: {out_dir}/")
    print("1_prop_t.png")
    print("2_err_corr_t.png")
    print("3_schema_counts_t.png")
    print("4_schema_mean_fitness_t.png")
    print("5_schema_sd_fitness_t.png")


# Main

def main() -> None:
    N = 200
    L = 40
    crossover: CrossoverType = "UX"
    fitness_fn = fitness_counting_ones

    ok, gens, evals, total_time, metrics = run_ga_extra(
        N=N,
        L=L,
        fitness_fn=fitness_fn,
        crossover=crossover,
        max_failures=20,
        seed=42,
    )

    print(f"Done. ok={ok} gens={gens} evals={evals} time={total_time:.3f}s")

    if not ok:
        print("population did not fully converge to all-ones before stopping")

    plot_all(metrics, out_dir="prac1/plots")


if __name__ == "__main__":
    main()