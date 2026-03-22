import csv
import os
import random
import time
from dataclasses import dataclass, field
from graph_utils import Graph, GraphColoring, get_conflict_count, vertex_descent

@dataclass
class ILSResult:
    solved: bool
    best_conflicts: int
    iterations: int
    elapsed: float
    k: int
    L: int
    perturbation: float

@dataclass
class ILSRunStats:
    k: int
    L: int
    perturbation: float
    n_runs: int
    n_solved: int
    success_rate: float
    mean_iterations: float
    mean_elapsed: float
    mean_best_conflicts: float
    results: list = field(repr=False, default_factory=list)

def perturb(coloring: GraphColoring, fraction: float) -> GraphColoring:
    n = coloring.graph.vertex_number
    k = coloring.k
    new_assignment = coloring.assignment.copy()
    n_perturb = max(1, int(fraction * n))
    for v in random.sample(range(n), n_perturb):
        new_assignment[v] = random.randrange(k)
    return GraphColoring(k=k, graph=coloring.graph, assignment=new_assignment)

def ils(
    graph: Graph,
    k: int,
    L: int,
    perturbation: float,
    max_time: float,
    verbose: bool = False,
) -> ILSResult:
    t_start = time.time()

    current = GraphColoring(k=k, graph=graph)
    current, solved = vertex_descent(graph, current, L)
    iterations = 1

    if solved:
        elapsed = time.time() - t_start
        if verbose:
            print(f"Solved on initial VD ({elapsed:.2f}s)")
        return ILSResult(solved=True, best_conflicts=0, iterations=iterations,
                         elapsed=elapsed, k=k, L=L, perturbation=perturbation)

    best_conflicts = get_conflict_count(graph, current)
    best = current

    if verbose:
        print(f"Initial conflicts: {best_conflicts}")

    while time.time() - t_start < max_time:
        candidate = perturb(best, perturbation)
        candidate, solved = vertex_descent(graph, candidate, L)
        iterations += 1
        candidate_conflicts = get_conflict_count(graph, candidate)

        if solved:
            elapsed = time.time() - t_start
            if verbose:
                print(f"Solved at iteration {iterations} ({elapsed:.2f}s)")
            return ILSResult(solved=True, best_conflicts=0, iterations=iterations,
                             elapsed=elapsed, k=k, L=L, perturbation=perturbation)

        if candidate_conflicts <= best_conflicts:
            best = candidate
            best_conflicts = candidate_conflicts

        if verbose and iterations % 100 == 0:
            print(f"  [iter {iterations} | {time.time()-t_start:.1f}s] "
                  f"best={best_conflicts}")

    elapsed = time.time() - t_start
    if verbose:
        print(f"Time limit reached. Best conflicts: {best_conflicts}")

    return ILSResult(solved=False, best_conflicts=best_conflicts,
                     iterations=iterations, elapsed=elapsed,
                     k=k, L=L, perturbation=perturbation)


CSV_HEADER = ["k", "L", "perturbation", "run", "solved", "best_conflicts",
              "iterations", "elapsed"]

def init_csv(filepath: str):
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

def append_csv(filepath: str, result: ILSResult, run_index: int):
    with open(filepath, "a", newline="") as f:
        csv.writer(f).writerow([
            result.k, result.L, result.perturbation, run_index,
            int(result.solved), result.best_conflicts,
            result.iterations, f"{result.elapsed:.2f}",
        ])

def run_ils_experiment(
    graph: Graph,
    k: int,
    L: int,
    perturbation: float,
    max_time: float,
    n_runs: int = 10,
    csv_path: str = None,
    verbose: bool = False,
) -> ILSRunStats:
    results = []
    for run in range(n_runs):
        result = ils(graph=graph, k=k, L=L, perturbation=perturbation,
                     max_time=max_time, verbose=verbose)
        results.append(result)
        status = "SOLVED" if result.solved else f"best={result.best_conflicts}"
        print(f"  [k={k} L={L} pert={perturbation:.0%} run {run+1}/{n_runs}] "
              f"{status} | iters={result.iterations} | t={result.elapsed:.1f}s")
        if csv_path:
            append_csv(csv_path, result, run + 1)

    n_solved       = sum(r.solved for r in results)
    solved_results = [r for r in results if r.solved]
    return ILSRunStats(
        k=k, L=L, perturbation=perturbation, n_runs=n_runs,
        n_solved=n_solved,
        success_rate=n_solved / n_runs,
        mean_iterations=(sum(r.iterations for r in solved_results) / len(solved_results)
                         if solved_results else float("inf")),
        mean_elapsed=sum(r.elapsed for r in results) / n_runs,
        mean_best_conflicts=sum(r.best_conflicts for r in results) / n_runs,
        results=results,
    )

def print_ils_stats(stats: ILSRunStats):
    print(f"  k={stats.k} L={stats.L} pert={stats.perturbation:.0%} | "
          f"solved={stats.n_solved}/{stats.n_runs} ({stats.success_rate*100:.0f}%) | "
          f"mean_iters={stats.mean_iterations:.0f} | "
          f"mean_elapsed={stats.mean_elapsed:.1f}s | "
          f"mean_best_conflicts={stats.mean_best_conflicts:.2f}")

if __name__ == "__main__":

    graph_small = Graph(filename="flat300_26_0.col.rtf.doc")
    graph_large = Graph(filename="flat1000_76_0.col.rtf.doc")

    # TASK 4.3

    PERTURBATIONS = [0.05, 0.10, 0.20, 0.40]
    CSV_43        = "results_task43_ils.csv"

    init_csv(CSV_43)

    print("=" * 60)
    print("TASK 4.3 — ILS on flat300-26, varying perturbation  [~2 hours]")
    print("=" * 60)

    for pert in PERTURBATIONS:
        print(f"\n--- perturbation={pert:.0%} ---")
        stats = run_ils_experiment(
            graph=graph_small, k=26, L=100, perturbation=pert,
            max_time=180, n_runs=10, csv_path=CSV_43,
        )
        print_ils_stats(stats)

    print(f"\nTask 4.3 (flat300) complete. Results saved to {CSV_43}")

    print("\n--- TASK 4.3 — ILS on flat1000-83, perturbation=20%  [15 min] ---")
    result = ils(graph=graph_large, k=83, L=200, perturbation=0.20,
                 max_time=900, verbose=True)
    append_csv(CSV_43, result, run_index=99)
    print(f"flat1000 result: solved={result.solved} | "
          f"best_conflicts={result.best_conflicts} | "
          f"iterations={result.iterations} | elapsed={result.elapsed:.1f}s\n")

    print("=" * 60)
    print("TASK 4.3 COMPLETE")
    print("=" * 60)