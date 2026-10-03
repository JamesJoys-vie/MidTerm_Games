import time
import heapq
from collections import deque
from typing import Tuple, List, Set, FrozenSet, Optional, Dict

class Agent1Controller:
    """Independent controller for Agent 1 using A* Search."""

    def __init__(self, name: str = "Agent 1 (A* Solver)"):
        self.name = name
        self.algorithm_name = "A* Search / Maze Heuristic"
        self.action_history: List[str] = []
        self.directions = {
            'North': (0, -1),
            'South': (0, 1),
            'East':  (1, 0),
            'West':  (-1, 0)
        }
        self._deadlocks_cache: Optional[Set[Tuple[int, int]]] = None
        self._maze_dist_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}

    def _get_corner_deadlocks(self, grid: Tuple[Tuple[str, ...], ...], goals: FrozenSet[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        if self._deadlocks_cache is not None:
            return self._deadlocks_cache

        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        deadlocks = set()

        for y in range(height):
            for x in range(width):
                if grid[y][x] == '%' or (x, y) in goals:
                    continue
                wn = (y - 1 < 0) or (grid[y - 1][x] == '%')
                ws = (y + 1 >= height) or (grid[y + 1][x] == '%')
                ww = (x - 1 < 0) or (grid[y][x - 1] == '%')
                we = (x + 1 >= width) or (grid[y][x + 1] == '%')
                if (wn and ww) or (wn and we) or (ws and ww) or (ws and we):
                    deadlocks.add((x, y))

        self._deadlocks_cache = deadlocks
        return deadlocks

    def _get_maze_distances(self, grid: Tuple[Tuple[str, ...], ...], goal: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        """BFS walking distance from `goal` to every reachable floor cell (cached per target)."""
        if goal in self._maze_dist_cache:
            return self._maze_dist_cache[goal]

        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        dist_map = {goal: 0}
        q = deque([goal])

        while q:
            curr = q.popleft()
            d = dist_map[curr]
            for dx, dy in self.directions.values():
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] != '%':
                    if (nx, ny) not in dist_map:
                        dist_map[(nx, ny)] = d + 1
                        q.append((nx, ny))

        self._maze_dist_cache[goal] = dist_map
        return dist_map

    def _walk_dist(self, grid: Tuple[Tuple[str, ...], ...], src: Tuple[int, int], dst: Tuple[int, int]) -> int:
        """Walking (BFS) distance between two cells. The grid is undirected, so one BFS from dst suffices."""
        if src == dst:
            return 0
        return self._get_maze_distances(grid, dst).get(src, 999)

    def _astar_plan_box(
        self,
        agent_start: Tuple[int, int],
        box_start: Tuple[int, int],
        target_goal: Tuple[int, int],
        other_boxes: Set[Tuple[int, int]],
        opponent_pos: Tuple[int, int],
        grid: Tuple[Tuple[str, ...], ...],
        deadlocks: Set[Tuple[int, int]],
        time_limit: float
    ) -> Optional[List[str]]:
        start_time = time.time()
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        dist_map = self._get_maze_distances(grid, target_goal)

        def heuristic(bpos: Tuple[int, int], apos: Tuple[int, int]) -> int:
            base_dist = dist_map.get(bpos, 999)
            box_to_agent = self._walk_dist(grid, apos, bpos)
            agent_to_box = max(0, box_to_agent - 1)
            return base_dist * 2 + agent_to_box

        init_h = heuristic(box_start, agent_start)
        counter = 0
        pq = [(init_h, 0, counter, agent_start, box_start, [])]
        visited: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {(agent_start, box_start): 0}

        while pq:
            if time.time() - start_time > time_limit:
                break

            f, g, _, apos, bpos, path = heapq.heappop(pq)

            if bpos == target_goal:
                return path

            if g > visited.get((apos, bpos), g):
                continue

            for act, (dx, dy) in self.directions.items():
                nax, nay = apos[0] + dx, apos[1] + dy
                na = (nax, nay)

                if not (0 <= nax < width and 0 <= nay < height) or grid[nay][nax] == '%':
                    continue
                if na in other_boxes or na == opponent_pos:
                    continue

                if na == bpos:
                    nbx, nby = bpos[0] + dx, bpos[1] + dy
                    nb = (nbx, nby)
                    if not (0 <= nbx < width and 0 <= nby < height) or grid[nby][nbx] == '%':
                        continue
                    if nb in other_boxes or nb == opponent_pos:
                        continue
                    if nb in deadlocks and nb != target_goal:
                        continue
                    next_state = (na, nb)
                else:
                    next_state = (na, bpos)

                new_g = g + 1
                if next_state not in visited or new_g < visited[next_state]:
                    visited[next_state] = new_g
                    new_h = heuristic(next_state[1], next_state[0])
                    counter += 1
                    heapq.heappush(pq, (new_g + new_h, new_g, counter, next_state[0], next_state[1], path + [act]))

        return None

    def get_action(
        self,
        my_pos: Tuple[int, int],
        opponent_pos: Tuple[int, int],
        my_boxes: FrozenSet[Tuple[int, int]],
        opponent_boxes: FrozenSet[Tuple[int, int]],
        goals: FrozenSet[Tuple[int, int]],
        grid: Tuple[Tuple[str, ...], ...],
        time_limit: float = 0.95
    ) -> str:
        start_time = time.time()
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        all_boxes = set(my_boxes) | set(opponent_boxes)

        uncompleted_boxes = [b for b in my_boxes if b not in goals]

        if not uncompleted_boxes:
            return 'Stay'

        deadlocks = self._get_corner_deadlocks(grid, goals)

        empty_goals = [g for g in goals if g not in all_boxes]
        target_goals = empty_goals if empty_goals else list(goals)

        best_plan: Optional[List[str]] = None
        remaining_time = time_limit - (time.time() - start_time)

        # Sort boxes by walking distance from the agent
        agent_dist = self._get_maze_distances(grid, my_pos)
        sorted_boxes = sorted(uncompleted_boxes, key=lambda b: agent_dist.get(b, 999))

        for b in sorted_boxes:
            other_boxes = all_boxes - {b}
            # Sort goals by walking distance from the box
            box_dist = self._get_maze_distances(grid, b)
            sorted_goals = sorted(target_goals, key=lambda g: box_dist.get(g, 999))
            for g in sorted_goals:
                budget = max(0.1, (remaining_time - (time.time() - start_time)) / 2)
                plan = self._astar_plan_box(
                    my_pos, b, g, other_boxes, opponent_pos, grid, deadlocks, budget
                )
                if plan:
                    if best_plan is None or len(plan) < len(best_plan):
                        best_plan = plan
                    break
            if best_plan and len(best_plan) <= 5:
                break

        if best_plan:
            action = best_plan[0]
            self.action_history.append(action)
            return action

        for act, (dx, dy) in self.directions.items():
            nx = my_pos[0] + dx
            ny = my_pos[1] + dy
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny][nx] != '%' and (nx, ny) != opponent_pos and (nx, ny) not in all_boxes:
                    return act

        return 'Stay'