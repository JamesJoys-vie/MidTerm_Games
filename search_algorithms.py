import sys
import time
import heapq
from collections import deque
from typing import List, Tuple, Dict, Set, Optional, Callable, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from map_parser import State
from heuristic import heuristic_maze_min_matching


DIRECTIONS = {
    'North': (0, -1),
    'South': (0, 1),
    'East':  (1, 0),
    'West':  (-1, 0)
}

ACTION_NAMES = ['North', 'South', 'East', 'West']


def get_possible_actions(state: State) -> List[str]:
    """
    Return all valid actions from the current state using Sokoban movement rules.
    A box can only be pushed if the cell behind it is free (not a wall, not another box).
    """
    valid_actions = []
    ax, ay = state.agent_pos
    grid = state.grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    for action, (dx, dy) in DIRECTIONS.items():
        next_ax = ax + dx
        next_ay = ay + dy

        if next_ax < 0 or next_ax >= width or next_ay < 0 or next_ay >= height:
            continue

        if grid[next_ay][next_ax] == '%':
            continue

        next_pos = (next_ax, next_ay)
        if next_pos in state.boxes:
            pushed_box_x = next_ax + dx
            pushed_box_y = next_ay + dy

            if pushed_box_x < 0 or pushed_box_x >= width or pushed_box_y < 0 or pushed_box_y >= height:
                continue

            if grid[pushed_box_y][pushed_box_x] == '%':
                continue

            if (pushed_box_x, pushed_box_y) in state.boxes:
                continue

        valid_actions.append(action)

    return valid_actions


def execute_action(state: State, action: str) -> Tuple[State, int]:
    """
    Apply an action to a state and return the resulting successor state with step cost 1.
    If a box occupies the target cell, it is shifted one step in the same direction.
    """
    dx, dy = DIRECTIONS[action]
    ax, ay = state.agent_pos
    new_ax = ax + dx
    new_ay = ay + dy
    new_agent_pos = (new_ax, new_ay)

    if new_agent_pos in state.boxes:
        new_box_pos = (new_ax + dx, new_ay + dy)
        new_boxes = set(state.boxes)
        new_boxes.remove(new_agent_pos)
        new_boxes.add(new_box_pos)
        boxes_frozenset = frozenset(new_boxes)
    else:
        boxes_frozenset = state.boxes

    step_cost = 1
    next_state = State(
        agent_pos=new_agent_pos,
        boxes=boxes_frozenset,
        goals=state.goals,
        grid=state.grid,
        cost=state.cost + step_cost,
        heuristic_value=0
    )

    return next_state, step_cost


def ucs_search(
    initial_state: State,
    goal_test: Optional[Callable[[State], bool]] = None,
    time_limit: float = 60.0
) -> Dict[str, Any]:
    """
    Uniform Cost Search (UCS): expands nodes in order of increasing path cost g(n).
    Guaranteed to find the optimal solution when all step costs are positive.
    """
    if goal_test is None:
        goal_test = lambda s: s.is_goal_state()

    start_time = time.time()
    nodes_explored = 0
    max_queue_size = 0

    if goal_test(initial_state):
        return {
            'algorithm': 'UCS',
            'success': True,
            'path': [],
            'cost': 0,
            'nodes_explored': 0,
            'max_queue_size': 0,
            'execution_time': time.time() - start_time,
            'memory_used_mb': 0.0
        }

    counter = 0
    pq = []
    heapq.heappush(pq, (0, counter, initial_state, []))

    best_g: Dict[Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]], int] = {}
    best_g[(initial_state.agent_pos, initial_state.boxes)] = 0

    while pq:
        max_queue_size = max(max_queue_size, len(pq))

        if time.time() - start_time > time_limit:
            break

        g, _, current_state, path = heapq.heappop(pq)
        state_key = (current_state.agent_pos, current_state.boxes)

        if g > best_g.get(state_key, float('inf')):
            continue

        nodes_explored += 1

        if goal_test(current_state):
            elapsed_time = time.time() - start_time
            mem_estimate = (sys.getsizeof(best_g) + sys.getsizeof(pq)) / (1024 * 1024)
            return {
                'algorithm': 'UCS',
                'success': True,
                'path': path,
                'cost': g,
                'nodes_explored': nodes_explored,
                'max_queue_size': max_queue_size,
                'execution_time': elapsed_time,
                'memory_used_mb': mem_estimate
            }

        for action in get_possible_actions(current_state):
            next_state, step_cost = execute_action(current_state, action)
            next_g = g + step_cost
            next_key = (next_state.agent_pos, next_state.boxes)

            if next_g < best_g.get(next_key, float('inf')):
                best_g[next_key] = next_g
                counter += 1
                heapq.heappush(pq, (next_g, counter, next_state, path + [action]))

    elapsed_time = time.time() - start_time
    mem_estimate = (sys.getsizeof(best_g) + sys.getsizeof(pq)) / (1024 * 1024)
    return {
        'algorithm': 'UCS',
        'success': False,
        'path': None,
        'cost': float('inf'),
        'nodes_explored': nodes_explored,
        'max_queue_size': max_queue_size,
        'execution_time': elapsed_time,
        'memory_used_mb': mem_estimate
    }


def astar_search(
    initial_state: State,
    goal_test: Optional[Callable[[State], bool]] = None,
    heuristic: Optional[Callable[[State], int]] = None,
    time_limit: float = 60.0
) -> Dict[str, Any]:
    """
    A* Search: expands nodes by f(n) = g(n) + h(n).
    With an admissible and consistent heuristic, finds the optimal solution while
    exploring fewer nodes than UCS. Deadlock branches (h >= 100,000) are pruned.
    """
    if goal_test is None:
        goal_test = lambda s: s.is_goal_state()
    if heuristic is None:
        heuristic = heuristic_maze_min_matching

    start_time = time.time()
    nodes_explored = 0
    max_queue_size = 0

    if goal_test(initial_state):
        return {
            'algorithm': 'A*',
            'success': True,
            'path': [],
            'cost': 0,
            'nodes_explored': 0,
            'max_queue_size': 0,
            'execution_time': time.time() - start_time,
            'memory_used_mb': 0.0
        }

    initial_h = heuristic(initial_state)
    counter = 0
    open_set = []
    heapq.heappush(open_set, (initial_h, counter, 0, initial_state, []))

    best_g: Dict[Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]], int] = {}
    best_g[(initial_state.agent_pos, initial_state.boxes)] = 0

    while open_set:
        max_queue_size = max(max_queue_size, len(open_set))

        if time.time() - start_time > time_limit:
            break

        f, _, g, current_state, path = heapq.heappop(open_set)
        state_key = (current_state.agent_pos, current_state.boxes)

        if g > best_g.get(state_key, float('inf')):
            continue

        nodes_explored += 1

        if goal_test(current_state):
            elapsed_time = time.time() - start_time
            mem_estimate = (sys.getsizeof(best_g) + sys.getsizeof(open_set)) / (1024 * 1024)
            return {
                'algorithm': 'A*',
                'success': True,
                'path': path,
                'cost': g,
                'nodes_explored': nodes_explored,
                'max_queue_size': max_queue_size,
                'execution_time': elapsed_time,
                'memory_used_mb': mem_estimate
            }

        for action in get_possible_actions(current_state):
            next_state, step_cost = execute_action(current_state, action)
            next_g = g + step_cost
            next_key = (next_state.agent_pos, next_state.boxes)

            if next_g < best_g.get(next_key, float('inf')):
                best_g[next_key] = next_g
                h_val = heuristic(next_state)
                if h_val >= 100_000:
                    continue
                next_f = next_g + h_val
                counter += 1
                heapq.heappush(open_set, (next_f, counter, next_g, next_state, path + [action]))

    elapsed_time = time.time() - start_time
    mem_estimate = (sys.getsizeof(best_g) + sys.getsizeof(open_set)) / (1024 * 1024)
    return {
        'algorithm': 'A*',
        'success': False,
        'path': None,
        'cost': float('inf'),
        'nodes_explored': nodes_explored,
        'max_queue_size': max_queue_size,
        'execution_time': elapsed_time,
        'memory_used_mb': mem_estimate
    }


def bfs_search(
    initial_state: State,
    goal_test: Optional[Callable[[State], bool]] = None,
    time_limit: float = 1.0
) -> Dict[str, Any]:
    """
    Breadth-First Search (BFS): explores states layer by layer using a FIFO queue.
    Finds the solution with the fewest number of actions.
    """
    if goal_test is None:
        goal_test = lambda s: s.is_goal_state()

    start_time = time.time()
    queue = deque([(initial_state, [])])
    visited = {(initial_state.agent_pos, initial_state.boxes)}
    nodes_explored = 0

    while queue:
        if time.time() - start_time > time_limit:
            break

        current_state, path = queue.popleft()
        nodes_explored += 1

        if goal_test(current_state):
            return {
                'algorithm': 'BFS',
                'success': True,
                'path': path,
                'cost': len(path),
                'nodes_explored': nodes_explored,
                'execution_time': time.time() - start_time
            }

        for action in get_possible_actions(current_state):
            next_state, _ = execute_action(current_state, action)
            key = (next_state.agent_pos, next_state.boxes)
            if key not in visited:
                visited.add(key)
                queue.append((next_state, path + [action]))

    return {
        'algorithm': 'BFS',
        'success': False,
        'path': None,
        'cost': float('inf'),
        'nodes_explored': nodes_explored,
        'execution_time': time.time() - start_time
    }


def dfs_search(
    initial_state: State,
    goal_test: Optional[Callable[[State], bool]] = None,
    max_depth: int = 50,
    time_limit: float = 1.0
) -> Dict[str, Any]:
    """
    Depth-Limited Depth-First Search (DFS): explores states using a LIFO stack
    with a configurable maximum depth bound.
    """
    if goal_test is None:
        goal_test = lambda s: s.is_goal_state()

    start_time = time.time()
    stack = [(initial_state, [], 0)]
    visited = {(initial_state.agent_pos, initial_state.boxes): 0}
    nodes_explored = 0

    while stack:
        if time.time() - start_time > time_limit:
            break

        current_state, path, depth = stack.pop()
        nodes_explored += 1

        if goal_test(current_state):
            return {
                'algorithm': 'DFS',
                'success': True,
                'path': path,
                'cost': len(path),
                'nodes_explored': nodes_explored,
                'execution_time': time.time() - start_time
            }

        if depth < max_depth:
            for action in get_possible_actions(current_state):
                next_state, _ = execute_action(current_state, action)
                key = (next_state.agent_pos, next_state.boxes)
                if key not in visited or depth + 1 < visited[key]:
                    visited[key] = depth + 1
                    stack.append((next_state, path + [action], depth + 1))

    return {
        'algorithm': 'DFS',
        'success': False,
        'path': None,
        'cost': float('inf'),
        'nodes_explored': nodes_explored,
        'execution_time': time.time() - start_time
    }


def gbfs_search(
    initial_state: State,
    goal_test: Optional[Callable[[State], bool]] = None,
    heuristic: Optional[Callable[[State], int]] = None,
    time_limit: float = 1.0
) -> Dict[str, Any]:
    """
    Greedy Best-First Search (GBFS): selects nodes purely by heuristic value f(n) = h(n).
    Fast and suitable for agent decision-making within tight time limits (<= 1000ms).
    """
    if goal_test is None:
        goal_test = lambda s: s.is_goal_state()
    if heuristic is None:
        heuristic = heuristic_maze_min_matching

    start_time = time.time()
    counter = 0
    pq = [(heuristic(initial_state), counter, initial_state, [])]
    visited = {(initial_state.agent_pos, initial_state.boxes)}
    nodes_explored = 0

    while pq:
        if time.time() - start_time > time_limit:
            break

        h_val, _, current_state, path = heapq.heappop(pq)
        nodes_explored += 1

        if goal_test(current_state):
            return {
                'algorithm': 'GBFS',
                'success': True,
                'path': path,
                'cost': len(path),
                'nodes_explored': nodes_explored,
                'execution_time': time.time() - start_time
            }

        for action in get_possible_actions(current_state):
            next_state, _ = execute_action(current_state, action)
            key = (next_state.agent_pos, next_state.boxes)
            if key not in visited:
                visited.add(key)
                h_next = heuristic(next_state)
                if h_next >= 100_000:
                    continue
                counter += 1
                heapq.heappush(pq, (h_next, counter, next_state, path + [action]))

    return {
        'algorithm': 'GBFS',
        'success': False,
        'path': None,
        'cost': float('inf'),
        'nodes_explored': nodes_explored,
        'execution_time': time.time() - start_time
    }


if __name__ == "__main__":
    from map_parser import MapParser
    map_path = sys.argv[1] if len(sys.argv) > 1 else "maps/example_map.txt"
    try:
        p = MapParser(map_path)
        init_st = p.get_initial_state()
        print(f"--- Running solver test: {p.filename} ---")

        res_ucs = ucs_search(init_st)
        print(f"[UCS] Success={res_ucs['success']}, Cost={res_ucs['cost']}, Nodes={res_ucs['nodes_explored']}, Time={res_ucs['execution_time']:.4f}s, Path={res_ucs['path']}")

        res_astar = astar_search(init_st)
        print(f"[A* ] Success={res_astar['success']}, Cost={res_astar['cost']}, Nodes={res_astar['nodes_explored']}, Time={res_astar['execution_time']:.4f}s, Path={res_astar['path']}")
    except Exception as err:
        print(f"Error: {err}")
