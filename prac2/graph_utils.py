import os
from typing import List
import random
import numpy as np
import time


class Graph:
    def __init__(self, vertex_number=None, edges=None, filename=None):
        if filename is not None:
            vertex_number, edges = self._read_graph_data(filename)

        if vertex_number is None or edges is None:
            raise ValueError("Provide either filename or (vertex_number, edges).")

        self.vertex_number = vertex_number
        self.edges = edges

    @staticmethod
    def _read_graph_data(filename: str) -> (int, List[List[int]]):
        path = os.path.join(os.getcwd(), "graph_data", filename)

        edge_list = []

        with open(path) as f:
            for line in f:
                if line.startswith("p "):
                    n = int(line.strip().split(" ")[-2])
                    edge_list = [[] for i in range(n)]

                elif line.startswith("e "):
                    (i, j) = [int(num) for num in line.strip().replace("\\","").split(" ")[-2:]]
                    edge_list[i - 1].append(j - 1)
                    edge_list[j - 1].append(i - 1)

        return n, edge_list

    def are_neighbors(self, i, j):
        return j in self.edges[i]


class GraphColoring:
    def __init__(self, k: int, assignment: List[int]):
        self.number_of_colors: int = k
        self.assignment: List[int] = assignment
        self.partition: List[List[int]] = [[] for i in range(k)]

        self.get_partition()

    def get_partition(self):
        for color_index in range(self.number_of_colors):
            self.partition[color_index] = (
                np.where(np.array(self.assignment) == color_index)[0]
            ).tolist()

    def change_color(self, vertex, new_color):
        old_color = self.assignment[vertex]
        if old_color == new_color:
            return

        self.assignment[vertex] = new_color
        self.partition[old_color].remove(vertex)
        self.partition[new_color].append(vertex)


def get_conflict_count(graph: Graph, coloring: GraphColoring) -> int:
    num_colors = coloring.number_of_colors
    conflicts = 0

    # For each color class in the partition representation, check all pair combinations of vertices for potential neighbors
    # # of neighbors in color class -> conflicts
    for color_index in range(num_colors):
        color_class = coloring.partition[color_index]

        for i, vertex in enumerate(color_class):
            for other_vertex in color_class[i + 1:]:
                if graph.are_neighbors(vertex, other_vertex):
                    conflicts += 1

    return conflicts

# Alternative method for conflict count (probably inferior to get_conflict_count, ignore for now)
# def get_conflict_count_2(graph: Graph, coloring: GraphColoring) -> int:
#     conflicts = 0
#
#     for vertex in range(graph.vertex_number):
#         vertex_color = coloring.assignment[vertex]
#
#         for neighbor in graph.edges[vertex]:
#             if coloring.assignment[neighbor] == vertex_color:
#                 conflicts += 1
#
#     return conflicts // 2

"""
Returns:
    - final coloring
    - solved (True/False)
    - improvement (True/False)
"""
def vertex_descent_iteration(
    graph: Graph,
    coloring: GraphColoring,
    debug: bool = False
) -> (GraphColoring, bool, bool):
    k = coloring.number_of_colors
    n = graph.vertex_number
    vertices_random_order = random.sample(range(n), n)
    improvement = False

    # Count initial number of conflicts
    current_conflicts = get_conflict_count(graph, coloring)

    for vertex in vertices_random_order:
        old_color = coloring.assignment[vertex]

        # color_counts: a list of length k, with the counts of total neighbors from each color that the vertex has.
        #   e.g. color_counts[4] == 5 -> 5 neighbors with the color code '4'
        neighbors = graph.edges[vertex]
        color_counts = [0] * k

        # Filling in the counts of each color
        for neighbor in neighbors:
            neighbor_color = coloring.assignment[neighbor]
            color_counts[neighbor_color] += 1

        # best recoloring candidates: the color codes with the least counts amongst the color_counts list
        min_conflicts_for_vertex = min(color_counts)
        best_color_candidates = [
            color for color, count in enumerate(color_counts)
            if count == min_conflicts_for_vertex
        ]

        # Tie-break: pick new color randomly
        new_color = random.choice(best_color_candidates)
        coloring.change_color(vertex, new_color)

        # Total number of conflicts is only affected by the difference in conflicts due to recoloring current vertex
        new_conflicts = current_conflicts - (
                color_counts[old_color] - color_counts[new_color]
        )

        # If the last recoloring causes 0 conflicts -> problem solved
        if new_conflicts == 0:
            return coloring, True, True

        # If the last recoloring causes fewer conflicts than before -> set improvement flag to True
        if new_conflicts < current_conflicts:
            improvement = True

        current_conflicts = new_conflicts

    return coloring, False, improvement


def vertex_descent_full_run(
    graph: Graph,
    coloring: GraphColoring,
    L: int,
    debug: bool = False
) -> tuple[GraphColoring, bool]:
    descent_cycles = 0

    if debug:
        print(f"\nStarting full Vertex Descent run (max {L} cycles)")
        print(f"Initial Conflicts: {get_conflict_count(graph, coloring)}")

    # Runs vertex descent algorithm until one of the following happens:
    #   - No improvement is achieved (conflicts remain unchanged)
    #   - Problem is solved
    #   - Completed L descent cycles
    while descent_cycles < L:
        if debug:
            print(f"\n--- Descent cycle {descent_cycles + 1} ---")

        coloring, solved, improvement = vertex_descent_iteration(
            graph, coloring, debug=debug
        )

        if debug:
            print(f"Conflicts: {get_conflict_count(graph, coloring)}, Improvement: {improvement}")

        if solved:
            if debug:
                print(f"Stopped: solved at cycle {descent_cycles + 1}")
            return coloring, True

        if not improvement:
            if debug:
                print(f"Stopped: no improvement at cycle {descent_cycles + 1}")
            return coloring, False

        descent_cycles += 1

    if debug:
        print("Stopped: reached maximum number of descent cycles")

    return coloring, False


config = {
    "test_graph" : {
        "filename" : "debug10.col.doc",
        "k": 3,
        "L": 10
    },
    "small_graph":{
        "filename" : "flat300_26_0.col.rtf.doc",
        "k": 28,
        "L": 100
    },
    "big_graph":{
        "filename" : "flat1000_76_0.col.rtf.doc",
        "k": 83,
        "L": 200
    }
}

if __name__ == "__main__":
    # random.seed(42)

    graph = "big_graph"
    filename = config[graph]["filename"]
    k = config[graph]["k"]
    L = config[graph]["L"]

    graph = Graph(filename=filename)

    initial_assignment = [random.randrange(k) for _ in range(graph.vertex_number)]
    coloring = GraphColoring(k=k, assignment=initial_assignment)

    print("Initial random assignment:", coloring.assignment)
    print("Initial conflicts:", get_conflict_count(graph, coloring))

    final_coloring, solved = vertex_descent_full_run(
        graph=graph,
        coloring=coloring,
        L=L,
        debug=False
    )

    print("\n" + "=" * 60)
    print("Final assignment:", final_coloring.assignment)
    print("Final partition :", final_coloring.partition)
    print("Final conflicts :", get_conflict_count(graph, final_coloring))
    print("Solved          :", solved)