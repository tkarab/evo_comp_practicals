import random
from graph_utils import GraphColoring


def greedy_partition_crossover(parent1: GraphColoring, parent2: GraphColoring, debug=False):
    if parent1.graph != parent2.graph:
        raise ValueError("Parents must be on the same graph.")
    if parent1.k != parent2.k:
        raise ValueError("Parents must have the same k.")

    graph = parent1.graph
    k = parent1.k

    p1 = [set(group) for group in parent1.partition]
    p2 = [set(group) for group in parent2.partition]
    child_partition = [set() for _ in range(k)]

    assigned = set()

    for i in range(k):
        current_parent = p1 if i % 2 == 0 else p2

        max_size = max(len(group) for group in current_parent)
        candidate_indices = [idx for idx, group in enumerate(current_parent) if len(group) == max_size]
        chosen_idx = random.choice(candidate_indices)
        chosen_group = set(current_parent[chosen_idx])

        child_partition[i].update(chosen_group)
        assigned.update(chosen_group)

        if debug:
            print(f"Iteration {i}: choose {'p1' if i % 2 == 0 else 'p2'}")
            print(f"Chosen group index: {chosen_idx}, size: {len(chosen_group)}")

        for v in chosen_group:
            for group in p1:
                group.discard(v)
            for group in p2:
                group.discard(v)

    remaining = list(set(graph.vertices) - assigned)
    for v in remaining:
        child_partition[random.randrange(k)].add(v)

    return GraphColoring(graph, k, partition=child_partition)