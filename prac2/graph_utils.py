import os
from typing import List
import random
import numpy as np


class Graph:
    def __init__(self, vertex_number=None, edges=None, filename=None):
        if filename is not None:
            vertex_number, edges = self._read_graph_data(filename)

        if vertex_number is None or edges is None:
            raise ValueError("Provide either filename or (vertex_number, edges).")

        self.vertex_number = vertex_number
        self.edges = edges

    @staticmethod
    def _read_graph_data(filename:str) -> (int, List[List[int]]):
        path = os.path.join(os.getcwd(), "graph_data", filename)

        edge_list = []

        with open(path) as f:
            for line in f:
                if line.startswith("p "):
                    n = int(line.strip().split(" ")[-2])
                    edge_list = [[] for i in range(n)]

                elif line.startswith("e "):
                    (i, j) = [int(num) for num in line.strip().split(" ")[-2:]]
                    edge_list[i - 1].append(j - 1)
                    edge_list[j - 1].append(i - 1)

        return n, edge_list

    def are_neighbors(self, i,j):
        return j in self.edges[i]


class GraphColoring:
    def __init__(self, k:int, assignment:List[int]):
        self.number_of_colors: int = k
        self.assignment: List[int] = assignment
        self.partition: List[List[int]] = [[] for i in range(k)]

        self.get_partition()

    def get_partition(self):
        for color_index in range(self.number_of_colors):
            self.partition[color_index] = (np.where(np.array(self.assignment) == color_index)[0]).tolist()

        return

    def change_color(self, vertex, new_color):
        old_color = self.assignment[vertex]
        if old_color == new_color:
            return

        self.assignment[vertex] = new_color
        self.partition[old_color].remove(vertex)
        self.partition[new_color].append(vertex)


def get_conflicts_1(graph:Graph, coloring: GraphColoring) -> int:
    num_colors = coloring.number_of_colors
    conflicts = 0

    for color_index in range(num_colors):
        color_class = coloring.partition[color_index]
        # m = len(color_class)

        for i, vertex in enumerate(color_class):
            for other_vertex in color_class[i+1:]:
                if graph.are_neighbors(vertex, other_vertex):
                    conflicts += 1

    return conflicts


def get_conflicts_2(graph: Graph, coloring: GraphColoring) -> int:
    conflicts = 0

    for vertex in range(graph.vertex_number):
        vertex_color = coloring.assignment[vertex]

        for neighbor in graph.edges[vertex]:


            if coloring.assignment[neighbor] == vertex_color:
                conflicts += 1

    return conflicts // 2


def vertex_descent_iteration(graph: Graph, coloring: GraphColoring) -> (GraphColoring, bool, bool):
    k = coloring.number_of_colors
    n = graph.vertex_number
    vertices_random_order = random.sample(range(n), n)
    improvement = False

    for vertex in vertices_random_order:
        vertex_color = coloring.assignment[vertex]
        best_color = vertex_color
        current_conflicts = get_conflicts_1(graph, coloring)

        for color_code in range(k):
            if not color_code == vertex_color:
                coloring.change_color(vertex, color_code)
                new_conflicts = get_conflicts_1(graph, coloring)

                # in case it is solved mid-way through
                if new_conflicts == 0:
                    return coloring, True, True

                # Case 1: less conflicts -> better coloring
                if new_conflicts < current_conflicts:
                    current_conflicts = new_conflicts
                    best_color = color_code
                    improvement = True
                # Case 2: Same conflicts -> tie; choosing randomly between current and new color
                elif new_conflicts == current_conflicts:
                    best_color = random.choice([best_color, color_code])
                    coloring.change_color(vertex, best_color)
                # Case 3: More conflicts -> keep the current coloring
                else:
                    coloring.change_color(vertex, best_color)

    return coloring, False, improvement

def vertex_descent_full_run(graph: Graph, coloring: GraphColoring, L: int) -> tuple[GraphColoring, bool]:
    descent_cycles = 0
    while descent_cycles < L:
        coloring, solved, improvement = vertex_descent_iteration(graph, coloring)

        if solved:
            return coloring, True

        if not improvement:
            return coloring, False

        descent_cycles += 1

    return coloring, False



n = 10
k = 3
colors_assignment = [0,1,0,2,1]

coloring = GraphColoring(k, colors_assignment)
graph = Graph(filename = "test_graph.col.doc")

conflict1 = get_conflicts_1(graph, coloring)
conflict2 = get_conflicts_2(graph, coloring)

new_coloring, solved, improved = vertex_descent_iteration(graph, coloring)

print()

