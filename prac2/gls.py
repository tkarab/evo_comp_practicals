import csv
import os
import random
import time
from dataclasses import dataclass, field
from gpx import greedy_partition_crossover
from graph_utils import Graph, GraphColoring, get_conflict_count, vertex_descent

@dataclass
class GLSResult:
    solved: bool
    best_conflicts: int
    crossovers: int
    elapsed: float
    k: int
    P: int
    L: int
    restarts: int = 0

@dataclass
class RunStats:
    k: int
    P: int
    L: int
    n_runs: int
    n_solved: int
    success_rate: float
    mean_crossovers: float
    mean_elapsed: float
    mean_best_conflicts: float
    results: list = field(repr=False, default_factory=list)

def _init_population(graph, k, P, L, verbose=False):
    if verbose:
        print(f"  Initialising population (P={P}, k={k}, L={L}) ...")
    population = []
    for _ in range(P):
        coloring = GraphColoring(k=k, graph=graph)
        coloring, solved = vertex_descent(graph, coloring, L)
        if solved:
            return [coloring], [0], True
        population.append(coloring)
    conflicts = [get_conflict_count(graph, c) for c in population]
    if verbose:
        print(f"  Population ready. Best: {min(conflicts)}, Worst: {max(conflicts)}")
    return population, conflicts, False

def gls(
    graph: Graph,
    k: int,
    P: int,
    L: int,
    max_time: float,
    restart_after: int = 200,
    verbose: bool = False,
) -> GLSResult:
    t_start = time.time()

    population, conflicts, solved = _init_population(graph, k, P, L, verbose)
    if solved:
        elapsed = time.time() - t_start
        if verbose:
            print(f"Solved during initialisation ({elapsed:.2f}s)")
        return GLSResult(solved=True, best_conflicts=0, crossovers=0,
                         elapsed=elapsed, k=k, P=P, L=L, restarts=0)

    crossovers  = 0
    global_best = min(conflicts)
    stagnation  = 0
    restarts    = 0

    while time.time() - t_start < max_time:
        i, j = random.sample(range(P), 2)
        child = greedy_partition_crossover(population[i], population[j])

        child, solved = vertex_descent(graph, child, L)
        crossovers += 1
        child_conflicts = get_conflict_count(graph, child)

        if solved:
            elapsed = time.time() - t_start
            if verbose:
                print(f"Solved at crossover {crossovers} ({elapsed:.2f}s)")
            return GLSResult(solved=True, best_conflicts=0,
                             crossovers=crossovers, elapsed=elapsed,
                             k=k, P=P, L=L, restarts=restarts)

        worst_idx = max(range(P), key=lambda idx: conflicts[idx])
        if child_conflicts <= conflicts[worst_idx]:
            population[worst_idx] = child
            conflicts[worst_idx] = child_conflicts

        current_best = min(conflicts)
        if current_best < global_best:
            global_best = current_best
            stagnation  = 0
        else:
            stagnation += 1

        if stagnation >= restart_after:
            restarts += 1
            if verbose:
                print(f"  [restart #{restarts} at crossover {crossovers} | "
                      f"t={time.time()-t_start:.1f}s | global_best={global_best}]")
            population, conflicts, solved = _init_population(graph, k, P, L, verbose=False)
            if solved:
                elapsed = time.time() - t_start
                return GLSResult(solved=True, best_conflicts=0,
                                 crossovers=crossovers, elapsed=elapsed,
                                 k=k, P=P, L=L, restarts=restarts)
            stagnation = 0

        if verbose and crossovers % 500 == 0:
            print(f"  [{crossovers} crossovers | {time.time()-t_start:.1f}s] "
                  f"global_best={global_best} current_best={current_best} "
                  f"restarts={restarts}")

    elapsed = time.time() - t_start
    if verbose:
        print(f"Time limit reached. Global best: {global_best}")

    return GLSResult(solved=False, best_conflicts=global_best,
                     crossovers=crossovers, elapsed=elapsed,
                     k=k, P=P, L=L, restarts=restarts)

CSV_HEADER = ["k", "P", "L", "run", "solved", "best_conflicts",
              "crossovers", "elapsed", "restarts"]

def init_csv(filepath: str):
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def append_csv(filepath: str, result: GLSResult, run_index: int):
    with open(filepath, "a", newline="") as f:
        csv.writer(f).writerow([
            result.k, result.P, result.L, run_index,
            int(result.solved), result.best_conflicts,
            result.crossovers, f"{result.elapsed:.2f}", result.restarts,
        ])

def run_experiment(
    graph: Graph,
    k: int,
    P: int,
    L: int,
    max_time: float,
    restart_after: int = 200,
    n_runs: int = 10,
    csv_path: str = None,
    verbose: bool = False,
) -> RunStats:
    results = []
    for run in range(n_runs):
        result = gls(graph=graph, k=k, P=P, L=L,
                     max_time=max_time, restart_after=restart_after,
                     verbose=verbose)
        results.append(result)
        status = "SOLVED" if result.solved else f"best={result.best_conflicts}"
        print(f"  [k={k} P={P} L={L} run {run+1}/{n_runs}] "
              f"{status} | crossovers={result.crossovers} | "
              f"restarts={result.restarts} | t={result.elapsed:.1f}s")
        if csv_path:
            append_csv(csv_path, result, run + 1)

    n_solved       = sum(r.solved for r in results)
    solved_results = [r for r in results if r.solved]
    return RunStats(
        k=k, P=P, L=L, n_runs=n_runs,
        n_solved=n_solved,
        success_rate=n_solved / n_runs,
        mean_crossovers=(sum(r.crossovers for r in solved_results) / len(solved_results)
                         if solved_results else float("inf")),
        mean_elapsed=sum(r.elapsed for r in results) / n_runs,
        mean_best_conflicts=sum(r.best_conflicts for r in results) / n_runs,
        results=results,
    )

def print_stats(stats: RunStats):
    print(f"  k={stats.k} P={stats.P} L={stats.L} | "
          f"solved={stats.n_solved}/{stats.n_runs} ({stats.success_rate*100:.0f}%) | "
          f"mean_crossovers={stats.mean_crossovers:.0f} | "
          f"mean_elapsed={stats.mean_elapsed:.1f}s | "
          f"mean_best_conflicts={stats.mean_best_conflicts:.2f}")

if __name__ == "__main__":

    RESTART_AFTER = 200

    graph_small = Graph(filename="flat300_26_0.col.rtf.doc")
    graph_large = Graph(filename="flat1000_76_0.col.rtf.doc")

    CSV_2 = "results_task2_baseline.csv"
    CSV_3 = "results_task3_flat1000.csv"
    init_csv(CSV_2)
    init_csv(CSV_3)

    # TASK 2

    print("=" * 60)
    print("TASK 2a — flat300, k=28, P=50, L=100  [15 min]")
    print("=" * 60)
    result = gls(graph=graph_small, k=28, P=50, L=100,
                 max_time=900, restart_after=RESTART_AFTER, verbose=True)
    append_csv(CSV_2, result, run_index=1)
    print(f"Result: solved={result.solved} | best_conflicts={result.best_conflicts} "
          f"| crossovers={result.crossovers} | restarts={result.restarts} "
          f"| elapsed={result.elapsed:.1f}s\n")

    print("=" * 60)
    print("TASK 2b — flat300, k=26, P=50, L=100  [15 min]")
    print("=" * 60)
    result = gls(graph=graph_small, k=26, P=50, L=100,
                 max_time=900, restart_after=RESTART_AFTER, verbose=True)
    append_csv(CSV_2, result, run_index=2)
    print(f"Result: solved={result.solved} | best_conflicts={result.best_conflicts} "
          f"| crossovers={result.crossovers} | restarts={result.restarts} "
          f"| elapsed={result.elapsed:.1f}s\n")

    print(f"Task 2 results saved to {CSV_2}\n")

    # TASK 3

    print("=" * 60)
    print("TASK 3a — flat1000, k=100, P=100, L=200  [5 min]")
    print("=" * 60)
    result = gls(graph=graph_large, k=100, P=100, L=200,
                 max_time=300, restart_after=RESTART_AFTER, verbose=True)
    append_csv(CSV_3, result, run_index=1)
    print(f"Result: solved={result.solved} | best_conflicts={result.best_conflicts} "
          f"| crossovers={result.crossovers} | restarts={result.restarts} "
          f"| elapsed={result.elapsed:.1f}s\n")

    print("=" * 60)
    print("TASK 3b — flat1000, k=83, P=100, L=200  [60 min]")
    print("=" * 60)
    result = gls(graph=graph_large, k=83, P=100, L=200,
                 max_time=3600, restart_after=RESTART_AFTER, verbose=True)
    append_csv(CSV_3, result, run_index=2)
    print(f"Result: solved={result.solved} | best_conflicts={result.best_conflicts} "
          f"| crossovers={result.crossovers} | restarts={result.restarts} "
          f"| elapsed={result.elapsed:.1f}s\n")

    print(f"Task 3 results saved to {CSV_3}\n")

    # TASK 4.1

    P_VALUES = [10, 25, 50, 100]
    L_VALUES = [10, 50, 100, 200]
    CSV_41   = "results_task41_grid.csv"

    init_csv(CSV_41)
    print("=" * 60)
    print("TASK 4.1 — Grid search on flat300-26  [~8 hours]")
    print("=" * 60)
    for P in P_VALUES:
        for L in L_VALUES:
            print(f"\n--- P={P}, L={L} ---")
            stats = run_experiment(
                graph=graph_small, k=26, P=P, L=L,
                max_time=180, restart_after=RESTART_AFTER,
                n_runs=10, csv_path=CSV_41,
            )
            print_stats(stats)
    print(f"\nTask 4.1 complete. Results saved to {CSV_41}\n")

    # TASK 4.2

    CSV_42 = "results_task42_k83.csv"
    init_csv(CSV_42)
    print("=" * 60)
    print("TASK 4.2 — flat300, k=83, P=50, L=100  [~50 min]")
    print("=" * 60)
    stats = run_experiment(
        graph=graph_small, k=83, P=50, L=100,
        max_time=300, restart_after=RESTART_AFTER,
        n_runs=10, csv_path=CSV_42,
    )
    print_stats(stats)
    print(f"Results saved to {CSV_42}\n")

    # TASK 4.4

    CSV_44 = "results_task44_restarts.csv"
    init_csv(CSV_44)
    print("=" * 60)
    print("TASK 4.4 — GLS with vs without restarts on flat300-26  [~60 min]")
    print("=" * 60)

    print("\n--- GLS WITHOUT restarts ---")
    stats_no_restart = run_experiment(
        graph=graph_small, k=26, P=50, L=100,
        max_time=180, restart_after=999999999,
        n_runs=10, csv_path=CSV_44,
    )
    print_stats(stats_no_restart)

    print("\n--- GLS WITH restarts ---")
    stats_with_restart = run_experiment(
        graph=graph_small, k=26, P=50, L=100,
        max_time=180, restart_after=200,
        n_runs=10, csv_path=CSV_44,
    )
    print_stats(stats_with_restart)
    print(f"\nTask 4.4 results saved to {CSV_44}\n")
    print("=" * 60)
    print("ALL TASKS COMPLETE")
    print("=" * 60)