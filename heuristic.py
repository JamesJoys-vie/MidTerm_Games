from collections import deque
from itertools import permutations
from typing import Dict, Tuple, FrozenSet, Set
from map_parser import State, identify_corner_deadlocks


_GOAL_DISTANCE_CACHE: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
_DEADLOCK_CACHE: Dict[Tuple[Tuple[str, ...], ...], Set[Tuple[int, int]]] = {}


def precompute_maze_distances(
    grid: Tuple[Tuple[str, ...], ...],
    goals: FrozenSet[Tuple[int, int]]
) -> Dict[Tuple[int, int], Dict[Tuple[int, int], int]]:
    """
    Precompute BFS maze distances from every floor cell to each goal position.
    Uses backward BFS from each goal, avoiding wall cells ('%').
    Returns a mapping: distance_map[goal][(x, y)] -> minimum step count.
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    distance_map = {}

    for goal in goals:
        distance_map[goal] = {}
        queue = deque([(goal[0], goal[1], 0)])
        visited = {goal}
        distance_map[goal][goal] = 0

        while queue:
            x, y, dist = queue.popleft()

            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny][nx] != '%' and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        distance_map[goal][(nx, ny)] = dist + 1
                        queue.append((nx, ny, dist + 1))

    return distance_map


def get_precomputed_distances(state: State) -> Dict[Tuple[int, int], Dict[Tuple[int, int], int]]:
    """Return (or initialize) the cached maze distance table for the current state."""
    global _GOAL_DISTANCE_CACHE
    sample_goal = next(iter(state.goals)) if state.goals else None
    if sample_goal is None or sample_goal not in _GOAL_DISTANCE_CACHE:
        _GOAL_DISTANCE_CACHE = precompute_maze_distances(state.grid, state.goals)
    return _GOAL_DISTANCE_CACHE


def get_deadlock_set(state: State) -> Set[Tuple[int, int]]:
    """Return (or initialize) the set of static corner deadlock positions for the current state."""
    global _DEADLOCK_CACHE
    if state.grid not in _DEADLOCK_CACHE:
        _DEADLOCK_CACHE[state.grid] = identify_corner_deadlocks(state.grid, state.goals)
    return _DEADLOCK_CACHE[state.grid]


def heuristic_maze_min_matching(state: State) -> int:
    """
    Primary heuristic: BFS Maze Distance with Optimal Bipartite Goal Assignment.
    Returns the minimum total maze distance over all box-to-goal assignments.
    Applies deadlock detection and falls back to greedy matching for more than 4 boxes.
    """
    deadlocks = get_deadlock_set(state)
    for box in state.boxes:
        if box in deadlocks:
            return 100_000

    if state.is_goal_state():
        return 0

    dist_cache = get_precomputed_distances(state)
    boxes = list(state.boxes)
    goals = list(state.goals)

    if len(boxes) <= 4 and len(boxes) == len(goals):
        min_total = float('inf')
        for perm in permutations(goals):
            total = 0
            for b, g in zip(boxes, perm):
                d = dist_cache.get(g, {}).get(b, float('inf'))
                total += d
            if total < min_total:
                min_total = total
        return min_total if min_total != float('inf') else 100_000

    unassigned_goals = set(goals)
    total_distance = 0

    for box in boxes:
        if box in unassigned_goals:
            unassigned_goals.remove(box)
            continue

        best_goal = None
        best_dist = float('inf')
        for goal in unassigned_goals:
            d = dist_cache.get(goal, {}).get(box, float('inf'))
            if d < best_dist:
                best_dist = d
                best_goal = goal

        if best_goal is not None:
            total_distance += best_dist
            unassigned_goals.remove(best_goal)
        else:
            total_distance += 50

    return total_distance


def heuristic_maze_nearest_goal(state: State) -> int:
    """
    Simple heuristic: sum of BFS distances from each box to its nearest goal.
    Admissible and consistent; runs in O(|Boxes| * |Goals|).
    """
    deadlocks = get_deadlock_set(state)
    for box in state.boxes:
        if box in deadlocks:
            return 100_000

    if state.is_goal_state():
        return 0

    dist_cache = get_precomputed_distances(state)
    total = 0

    for box in state.boxes:
        min_d = float('inf')
        for goal in state.goals:
            d = dist_cache.get(goal, {}).get(box, float('inf'))
            if d < min_d:
                min_d = d
        if min_d == float('inf'):
            return 100_000
        total += min_d

    return total


def heuristic_zero(state: State) -> int:
    """Trivial heuristic h(n) = 0; reduces A* to Uniform Cost Search. Used as a baseline."""
    return 0


def verify_admissibility(
    h_value: int,
    optimal_cost_to_goal: int
) -> Tuple[bool, str]:
    """
    Check the admissibility condition: h(n) <= h*(n).
    Returns (is_admissible, message).
    """
    is_valid = h_value <= optimal_cost_to_goal
    msg = (f"h(n)={h_value} <= h*(n)={optimal_cost_to_goal} [PASS - Admissible]"
           if is_valid else
           f"h(n)={h_value} > h*(n)={optimal_cost_to_goal} [VIOLATION]")
    return is_valid, msg


def verify_consistency(
    h_current: int,
    step_cost: int,
    h_next: int
) -> Tuple[bool, str]:
    """
    Check the consistency condition (triangle inequality): h(n) <= c(n, n') + h(n').
    Returns (is_consistent, message).
    """
    is_valid = h_current <= step_cost + h_next
    msg = (f"h(n)={h_current} <= c(n,n')={step_cost} + h(n')={h_next} [PASS - Consistent]"
           if is_valid else
           f"h(n)={h_current} > {step_cost + h_next} [VIOLATION]")
    return is_valid, msg


if __name__ == "__main__":
    print("Module heuristic.py is ready.")
    print("Default heuristic: heuristic_maze_min_matching (no Manhattan/Euclidean distance used).")
