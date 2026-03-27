import os
import random
from typing import List
import time

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


def dsatur(graph:Graph, k:int):
    n = graph.vertex_number

    # initially empty cost matrix
    c = [[0 for _ in range(k)] for _ in range(n)]

    # Initialize assignment -> -1 == not assigned yet
    assignment = [-1 for _ in range(n)]

    # Allowed color classes =  number of '0' in cost matrix of vertex
    allowed_color_classes_per_vertex = [color_counts.count(0) for color_counts in c]

    # Minimum number of allowed classes in any (available) vertex (that are >0, and assignment of vertex is still -1) -> those with this number of allowed classes are the candidates
    min_allowed_classes = min([classes for i,classes in enumerate(allowed_color_classes_per_vertex) if classes > 0 and assignment[i] == -1])

    # Candidate vertices == those where # of allowed color classes == min_allowed_color_classes
    candidate_vertices = [i for i in range(n) if allowed_color_classes_per_vertex[i] == min_allowed_classes and assignment[i] == -1]


    # if not eve one vertex has a single allowed color class -> saturation
    while len(candidate_vertices) > 0:

        # Choose randomly
        chosen_vertex = random.choice(candidate_vertices)
        # color with lowest index from list of allowed ones (therefore all indices j where c[chosen_vertex][j] == 0) -> with .index(0_ you get the first appearence a.k.a. the allowed color with the lowest index
        chosen_color = c[chosen_vertex].index(0)
        # Assign coloring
        assignment[chosen_vertex] = chosen_color
        # Update neighbors in cos matrix
        for neighbor in graph.edges[chosen_vertex]:
            # Each Neighbor has one more adjacent vertex of color *chosen_color*
            c[neighbor][chosen_color] += 1

        # recalculate allowed classes per vertex and min allowed classes
        allowed_color_classes_per_vertex = [color_counts.count(0) for color_counts in c]

        if allowed_color_classes_per_vertex.count(0) == n:
            print("print no vertex has any allowed color classes")

        allowed_classes_for_non_assigned_vertices = [classes for i,classes in enumerate(allowed_color_classes_per_vertex) if classes > 0 and assignment[i] == -1]
        min_allowed_classes = min(allowed_classes_for_non_assigned_vertices) if len(allowed_classes_for_non_assigned_vertices)>0 else -1
        # Candidate vertices == those where # of allowed color classes == min_allowed_color_classes
        candidate_vertices = [i for i in range(n) if allowed_color_classes_per_vertex[i] == min_allowed_classes and assignment[i] == -1]

        if len(candidate_vertices) == 0:
            print()


    # Remaining vertices -> those that color is still '-1'
    remaining_vertices = [i for i,vertex_color in enumerate(assignment) if vertex_color == -1]
    for vertex in remaining_vertices:
        # Assign color randomly
        assignment[vertex] = random.randrange(k)

    coloring = GraphColoring(graph=graph, k=k, assignment=assignment)

    conflicts = get_conflict_count(graph, coloring)

    return coloring

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

    coloring = GraphColoring(k=k, graph=graph)
    print(f"Initial conflicts: {get_conflict_count(graph, coloring)}")

    dsatur(graph, coloring.k)

    t = time.time()
    coloring, solved = vertex_descent(graph, coloring, L=L)
    elapsed = time.time() - t

    print(f"Final conflicts:   {get_conflict_count(graph, coloring)}")
    print(f"Solved: {solved} | Time: {elapsed:.3f}s")