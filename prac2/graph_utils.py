import os
import random
from typing import List
import time
import math

import numpy as np


class Graph:
    def __init__(self, vertex_number=None, edges=None, filename=None):
        if filename is not None:
            vertex_number, edges = self._read_graph_data(filename)

        if vertex_number is None or edges is None:
            raise ValueError("Provide either filename or (vertex_number, edges).")

        self.vertex_number = vertex_number
        self.vertices = set(range(vertex_number))
        self.edges: List[List[int]] = [list(nbrs) for nbrs in edges]

    @staticmethod
    def _read_graph_data(filename: str):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "graph_data", filename)

        edge_list = []
        n = None

        with open(path) as f:
            for line in f:
                if line.startswith("p "):
                    n = int(line.strip().split(" ")[-2])
                    edge_list = [[] for _ in range(n)]
                elif line.startswith("e "):
                    i, j = [int(x) for x in line.strip().replace("\\", "").split(" ")[-2:]]
                    edge_list[i - 1].append(j - 1)
                    edge_list[j - 1].append(i - 1)

        if n is None:
            raise ValueError(f"Could not parse graph file: {filename}")

        return n, edge_list

    def are_neighbors(self, i, j):
        return j in self.edges[i]


class GraphColoring:
    def __init__(self, k: int, graph: Graph, assignment: List[int] = None, partition: List[List[int]] = None):
        if k is None or graph is None:
            raise ValueError("Must provide k and graph.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        self.k = k
        self.graph = graph

        if assignment is not None:
            ok, msg = self._check_assignment(assignment)
            if not ok:
                raise ValueError(msg)
            self.assignment = assignment.copy()
            self._build_partition()

        elif partition is not None:
            ok, msg = self._check_partition(partition)
            if not ok:
                raise ValueError(msg)
            self.partition = [cls.copy() for cls in partition]
            self._build_assignment()

        else:
            self.assignment = [random.randrange(k) for _ in range(graph.vertex_number)]
            self._build_partition()

    def _check_assignment(self, assignment):
        if len(assignment) != self.graph.vertex_number:
            return False, "Assignment length does not match number of vertices."

        for c in assignment:
            if c < 0 or c >= self.k:
                return False, "Assignment contains an invalid color."

        return True, None

    def _check_partition(self, partition):
        if len(partition) != self.k:
            return False, "Partition length does not match k."

        seen = set()
        for cls in partition:
            for v in cls:
                if v < 0 or v >= self.graph.vertex_number:
                    return False, "Partition contains an invalid vertex."
                if v in seen:
                    return False, "A vertex appears more than once in the partition."
                seen.add(v)

        if seen != self.graph.vertices:
            return False, "Partition must cover all vertices exactly once."

        return True, None

    def _build_partition(self):
        self.partition = [[] for _ in range(self.k)]
        for v, c in enumerate(self.assignment):
            self.partition[c].append(v)

    def _build_assignment(self):
        self.assignment = [None] * self.graph.vertex_number
        for c, cls in enumerate(self.partition):
            for v in cls:
                self.assignment[v] = c

    def change_color(self, vertex, new_color):
        old_color = self.assignment[vertex]
        if old_color == new_color:
            return

        self.assignment[vertex] = new_color
        self.partition[old_color].remove(vertex)
        self.partition[new_color].append(vertex)


def get_conflict_count(graph: Graph, coloring: GraphColoring) -> int:
    conflicts = 0
    for v in range(graph.vertex_number):
        vc = coloring.assignment[v]
        for nb in graph.edges[v]:
            if coloring.assignment[nb] == vc:
                conflicts += 1
    return conflicts // 2


def build_cost_matrix(graph: Graph, coloring: GraphColoring) -> List[List[int]]:
    k = coloring.k
    n = graph.vertex_number
    c = [[0] * k for _ in range(n)]

    for v in range(n):
        for nb in graph.edges[v]:
            c[v][coloring.assignment[nb]] += 1

    return c


def vertex_descent(
    graph: Graph,
    coloring: GraphColoring,
    L: int,
) -> tuple:
    n = graph.vertex_number

    conflicts = get_conflict_count(graph, coloring)
    c = build_cost_matrix(graph, coloring)

    descent_cycles = 0
    vertices = list(range(n))
    while descent_cycles < L:

        improved = False
        random.shuffle(vertices)

        for v in vertices:
            old_color = coloring.assignment[v]
            old_cost = c[v][old_color]
            min_cost = min(c[v])

            if min_cost <= old_cost:
                best = [color for color, cost in enumerate(c[v]) if cost == min_cost]
                new_color = random.choice(best)

                coloring.assignment[v] = new_color
                coloring.partition[old_color].remove(v)
                coloring.partition[new_color].append(v)

                conflicts = conflicts - (old_cost - min_cost)

                for nb in graph.edges[v]:
                    c[nb][old_color] -= 1
                    c[nb][new_color] += 1

                if min_cost < old_cost:
                    improved = True

                if c[v][new_color] == 0 and conflicts == 0:
                    return coloring, True

        if improved:
            descent_cycles += 1
        else:
            return coloring, False

    return coloring, False


def vertex_descent_full_run(graph, coloring, L, debug=False):
    return vertex_descent(graph, coloring, L)

def dsatur_initialization(graph: Graph, k: int, break_ties_by_degree: bool = False):
    n = graph.vertex_number
    c = [[0 for _ in range(k)] for _ in range(n)]
    assignment = [-1 for _ in range(n)]

    while True:
        allowed_counts = [row.count(0) for row in c]

        unassigned_allowed_counts = [count for v, count in enumerate(allowed_counts) if assignment[v] == -1 and count > 0]

        if not unassigned_allowed_counts:
            break

        min_allowed = min(unassigned_allowed_counts)

        candidate_vertices = [v for v in range(n) if assignment[v] == -1 and allowed_counts[v] == min_allowed]

        if break_ties_by_degree:
            max_degree = max(len(graph.edges[v]) for v in candidate_vertices)
            candidate_vertices = [v for v in candidate_vertices if len(graph.edges[v]) == max_degree]

        chosen_vertex = random.choice(candidate_vertices)
        chosen_color = c[chosen_vertex].index(0)

        assignment[chosen_vertex] = chosen_color

        for neighbor in graph.edges[chosen_vertex]:
            c[neighbor][chosen_color] += 1

    for v, color in enumerate(assignment):
        if color == -1:
            assignment[v] = random.randrange(k)

    coloring = GraphColoring(graph=graph, k=k, assignment=assignment)

    return coloring

def tabu_search(
    graph: Graph,
    coloring: GraphColoring,
    max_iterations: int,
    A:int = 10,
    alpha: float = 0.6,
):
    n = graph.vertex_number
    k=coloring.k
    c = build_cost_matrix(graph, coloring)
    current_conflicts = get_conflict_count(graph, coloring)
    best_conflicts = current_conflicts
    best_assignment = coloring.assignment.copy()

    conflicting_vertices = [i for i in range(n) if c[i][coloring.assignment[i]] > 0]

    iter = 0

    tabu_moves = [[] for _ in range(n)]

    while iter < max_iterations and current_conflicts > 0:
        possible_moves = []
        for v in conflicting_vertices:
            current_color = coloring.assignment[v]
            active_tabu_colors = [color for color, tenure in tabu_moves[v] if tenure >= iter]

            allowed = []
            for candidate in range(k):
                if candidate == current_color:
                    continue

                candidate_conflicts = current_conflicts - (c[v][current_color] - c[v][candidate])
                aspiration = candidate_conflicts < best_conflicts

                if aspiration or candidate not in active_tabu_colors:
                    allowed.append((candidate, candidate_conflicts))

            if allowed:
                best_vertex_cost = min(conf for _, conf in allowed)
                best_candidates = [cand for cand, conf in allowed if conf == best_vertex_cost]

                for candidate in best_candidates:
                    possible_moves.append((v, current_color, candidate, best_vertex_cost))

        if not possible_moves:
            break

        # Choose best move
        best_cost = min(move[3] for move in possible_moves)
        best_moves = [move for move in possible_moves if move[3] == best_cost]
        best_move = random.choice(best_moves)

        v = best_move[0]
        old_color = best_move[1]
        new_color = best_move[2]
        new_conflicts = best_move[3]
        current_conflicts = new_conflicts

        # Apply new move
        coloring.change_color(v, new_color)
        for neighbor in graph.edges[v]:
            c[neighbor][new_color] += 1
            c[neighbor][old_color] -= 1

        # Check for new best assignment
        if current_conflicts < best_conflicts:
            best_conflicts = current_conflicts
            best_assignment = coloring.assignment.copy()

        nbCFL = len(conflicting_vertices)
        tl = random.randint(0, A - 1) + alpha * nbCFL
        expire_iter = iter + math.ceil(tl)

        # Add move to tabu moves
        tabu_moves[v].append([old_color, expire_iter])

        # Remove tabu moves past their expiration
        for i in range(n):
            tabu_moves[i] = [move for move in tabu_moves[i] if move[1] >= iter]


        # recalculate conflicting vertices
        conflicting_vertices = [i for i in range(n) if c[i][coloring.assignment[i]] > 0]
        iter+=1

    return GraphColoring(graph=graph, k=k, assignment=best_assignment)







if __name__ == "__main__":
    import sys
    import time

    CONFIGS = {
        "test": ("debug10.col.doc", 3, 10),
        "small": ("flat300_26_0.col.rtf.doc", 28, 100),
        "large": ("flat1000_76_0.col.rtf.doc", 83, 200),
    }

    name = sys.argv[1] if len(sys.argv) > 1 else "small"
    filename, k, L = CONFIGS[name]

    graph = Graph(filename=filename)
    print(f"Graph: {name} | n={graph.vertex_number} | k={k} | L={L}")

    # coloring = GraphColoring(k=k, graph=graph)
    # Initiailize coloring using DSATUR
    coloring = dsatur_initialization(graph, k=k)

    import copy
    col_tabu = copy.deepcopy(coloring)

    col_tabu, conflicts_tabu = tabu_search(graph, col_tabu, 500)

    print(f"Initial conflicts: {get_conflict_count(graph, coloring)}")

    t = time.time()
    coloring, solved = vertex_descent(graph, coloring, L=L)
    elapsed = time.time() - t

    print(f"Final conflicts:   {get_conflict_count(graph, coloring)}")
    print(f"Solved: {solved} | Time: {elapsed:.3f}s")
