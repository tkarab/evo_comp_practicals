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
        self.vertices = set(range(vertex_number))
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
    def __init__(self, k: int, graph:Graph, assignment: List[int] = None, partition: List[List[int]] = None):

        # Check if Graph and k are properly provided
        if k is None or graph is None:
            raise ValueError("Must provide k and graph when creating GraphColoring instance.")
        elif k <= 0:
            raise ValueError("k must be greater than 0.")

        # Assign k and graph
        self.k: int = k
        self.graph: Graph = graph

        # Case 1: Assignment provided
        if assignment is not None:
            is_ok, message = self.assignment_sanity_check(assignment)
            if not is_ok:
                raise ValueError(message)

            self.assignment: List[int] = assignment.copy()
            self.create_partition_representation_from_assignment()

        # Case 2: Partition provided
        elif assignment is None and partition is not None:
            is_ok, message = self.partition_sanity_check(partition)
            if not is_ok:
                raise ValueError(message)
            self.partition: List[List[int]] = [color_class.copy() for color_class in partition]
            self.create_assignment_representation_from_partition()

        # Case 3: No representation provided
        elif assignment is None and partition is None:
            self.create_randomized_assignment()
            self.create_partition_representation_from_assignment()

        return

    def assignment_sanity_check(self, assignment: List[int]) -> (bool, str):
        if len(assignment) != self.graph.vertex_number:
            return False, f"Assignment representation must have as many elements as total graph vertices ({self.graph.vertex_number})."

        for color in assignment:
            if color < 0 or color >= self.k:
                return False, f"Each color code must be in the range [0, {self.k - 1}]."

        return True, None

    def partition_sanity_check(self, partition: List[List[int]]) -> (bool, str):
        if len(partition) != self.k:
            return False, f"Partition representation must have as many elements as total number of colors ({self.k})."

        vertices_assigned = set()
        for color_class in partition:
            for vertex_number in color_class:
                if vertex_number < 0 or vertex_number >= self.graph.vertex_number:
                    return False, f"Each vertex must be in the range [0, {self.graph.vertex_number - 1}]."
                elif vertex_number in vertices_assigned:
                    return False, f"Each vertex must appear exactly once in partition representation. (found {vertex_number} multiple times)"
                vertices_assigned.add(vertex_number)
        if vertices_assigned != self.graph.vertices:
            return False, f"All vertices must appear exactly once in partition representation."

        return True, None

    def create_partition_representation_from_assignment(self):
        self.partition: List[List[int]] = [[] for i in range(k)]
        for color_index in range(self.k):
            self.partition[color_index] = (
                np.where(np.array(self.assignment) == color_index)[0]
            ).tolist()

        return

    def create_assignment_representation_from_partition(self):
        self.assignment: List[int] = [None] * self.graph.vertex_number
        for color_code, color_class in enumerate(self.partition):
            for vertex_number in color_class:
                self.assignment[vertex_number] = color_code

        return


    def create_randomized_assignment(self):
        self.assignment: List[int] = [random.randrange(self.k) for _ in range(self.graph.vertex_number)]



    def change_color(self, vertex, new_color):
        old_color = self.assignment[vertex]
        if old_color == new_color:
            return

        self.assignment[vertex] = new_color
        self.partition[old_color].remove(vertex)
        self.partition[new_color].append(vertex)


def get_conflict_count(graph: Graph, coloring: GraphColoring) -> int:
    num_colors = coloring.k
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
    k = coloring.k
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
    coloring = GraphColoring(k=k, graph=graph, assignment=initial_assignment)
    coloring2 = GraphColoring(k=k, graph=graph, assignment=initial_assignment)



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