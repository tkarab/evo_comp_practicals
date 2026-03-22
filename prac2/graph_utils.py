import os
from typing import List
import random

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
        path = os.path.join(os.getcwd(), "graph_data", filename)
        edge_list = []
        with open(path) as f:
            for line in f:
                if line.startswith("p "):
                    n = int(line.strip().split(" ")[-2])
                    edge_list = [[] for _ in range(n)]
                elif line.startswith("e "):
                    (i, j) = [int(x) for x in line.strip().replace("\\", "").split(" ")[-2:]]
                    edge_list[i - 1].append(j - 1)
                    edge_list[j - 1].append(i - 1)
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
            return False
        for c in assignment:
            if c < 0 or c >= self.k:
                return False
        return True, None

    def _check_partition(self, partition):
        if len(partition) != self.k:
            return False
        seen = set()
        for cls in partition:
            for v in cls:
                if v < 0 or v >= self.graph.vertex_number:
                    return False
                if v in seen:
                    return False
                seen.add(v)
        if seen != self.graph.vertices:
            return False
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

    k = coloring.k
    n = graph.vertex_number

    # Build incremental cost matrix
    c = build_cost_matrix(graph, coloring)

    no_improve_count = 0
    vertices = list(range(n))

    while no_improve_count < L:
        improved = False
        random.shuffle(vertices)

        for v in vertices:
            old_color = coloring.assignment[v]
            old_cost  = c[v][old_color]

            min_cost = min(c[v])

            if min_cost < old_cost:
                best = [color for color, cost in enumerate(c[v]) if cost == min_cost]
                new_color = random.choice(best)

                coloring.assignment[v] = new_color
                coloring.partition[old_color].remove(v)
                coloring.partition[new_color].append(v)

                for nb in graph.edges[v]:
                    c[nb][old_color] -= 1
                    c[nb][new_color] += 1

                improved = True

                if c[v][new_color] == 0 and get_conflict_count(graph, coloring) == 0:
                    return coloring, True

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

    return coloring, False

# Convenience alias used by gls.py

def vertex_descent_full_run(graph, coloring, L, debug=False):
    """Alias for vertex_descent, kept for API compatibility with gls.py."""
    return vertex_descent(graph, coloring, L)

# Quick test when run directly

if __name__ == "__main__":
    import sys, time

    CONFIGS = {
        "test":  ("debug10.col.doc",           3,  10),
        "small": ("flat300_26_0.col.rtf.doc",  26, 100),
        "large": ("flat1000_76_0.col.rtf.doc", 83, 200),
    }

    name = sys.argv[1] if len(sys.argv) > 1 else "small"
    filename, k, L = CONFIGS[name]

    graph = Graph(filename=filename)
    print(f"Graph: {name} | n={graph.vertex_number} | k={k} | L={L}")

    coloring = GraphColoring(k=k, graph=graph)
    print(f"Initial conflicts: {get_conflict_count(graph, coloring)}")

    t = time.time()
    coloring, solved = vertex_descent(graph, coloring, L=L)
    elapsed = time.time() - t

    print(f"Final conflicts:   {get_conflict_count(graph, coloring)}")
    print(f"Solved: {solved} | Time: {elapsed:.3f}s")