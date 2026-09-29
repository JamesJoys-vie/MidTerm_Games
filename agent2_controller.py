import time
import heapq
from collections import deque
from typing import Tuple, List, Set, FrozenSet, Optional, Dict

class Agent2Controller:
    """Independent controller for Agent 2 using Greedy Best-First Search (GBFS)."""

    def __init__(self, name: str = "Agent 2 (Greedy Best-First Solver)"):
        self.name = name
        self.algorithm_name = "Greedy Best-First Search / Competitive"
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

    def _get_maze_distances(self, grid: Tuple[Tuple[str, ...], ...], target: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        if target in self._maze_dist_cache:
            return self._maze_dist_cache[target]

        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        dist_map = {target: 0}
        q = deque([target])

        while q:
            curr = q.popleft()
            d = dist_map[curr]
            for dx, dy in self.directions.values():
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] != '%':
                    if (nx, ny) not in dist_map:
                        dist_map[(nx, ny)] = d + 1
                        q.append((nx, ny))

        self._maze_dist_cache[target] = dist_map
        return dist_map

    def _gbfs_plan_box(
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
            agent_to_box = abs(apos[0] - bpos[0]) + abs(apos[1] - bpos[1])
            return base_dist * 3 + agent_to_box

        init_h = heuristic(box_start, agent_start)
        counter = 0
        pq = [(init_h, counter, agent_start, box_start, [])]
        visited: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = {(agent_start, box_start)}

        while pq:
            if time.time() - start_time > time_limit:
                break

            _, _, apos, bpos, path = heapq.heappop(pq)

            if bpos == target_goal:
                return path

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

                if next_state not in visited:
                    visited.add(next_state)
                    new_h = heuristic(next_state[1], next_state[0])
                    counter += 1
                    heapq.heappush(pq, (new_h, counter, next_state[0], next_state[1], path + [act]))

        return None

    def _bfs_navigate(
        self,
        start_pos: Tuple[int, int],
        target_positions: Set[Tuple[int, int]],
        obstacles: Set[Tuple[int, int]],
        grid: Tuple[Tuple[str, ...], ...]
    ) -> Optional[str]:
        if start_pos in target_positions:
            return None
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        q = deque([(start_pos, [])])
        visited = {start_pos}

        while q:
            curr, path = q.popleft()
            if curr in target_positions:
                return path[0] if path else None

            for act, (dx, dy) in self.directions.items():
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] != '%':
                    if (nx, ny) not in obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append(((nx, ny), path + [act]))
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
        deadlocks = self._get_corner_deadlocks(grid, goals)

        uncompleted_my_boxes = [b for b in my_boxes if b not in goals]

        if not uncompleted_my_boxes:
            opp_on_goal = [b for b in opponent_boxes if b in goals]
            if opp_on_goal:
                target_b = min(opp_on_goal, key=lambda b: abs(b[0] - my_pos[0]) + abs(b[1] - my_pos[1]))
                other_b = all_boxes - {target_b}
                for act, (dx, dy) in self.directions.items():
                    off_goal = (target_b[0] + dx, target_b[1] + dy)
                    if 0 <= off_goal[0] < width and 0 <= off_goal[1] < height:
                        if off_goal not in goals and off_goal not in other_b and grid[off_goal[1]][off_goal[0]] != '%':
                            plan = self._gbfs_plan_box(
                                my_pos, target_b, off_goal, other_b, opponent_pos, grid, deadlocks, 0.4
                            )
                            if plan:
                                action = plan[0]
                                self.action_history.append(action)
                                return action

            center_spots = {(width // 2, height // 2), (width // 2 - 1, height // 2), (width // 2 + 1, height // 2)}
            valid_center = {s for s in center_spots if 0 <= s[0] < width and 0 <= s[1] < height and grid[s[1]][s[0]] != '%' and s not in all_boxes and s != opponent_pos}
            if valid_center and my_pos not in valid_center:
                move = self._bfs_navigate(my_pos, valid_center, all_boxes | {opponent_pos}, grid)
                if move:
                    self.action_history.append(move)
                    return move

            return 'Stay'

        empty_goals = [g for g in goals if g not in all_boxes]
        target_goals = empty_goals if empty_goals else list(goals)

        best_plan: Optional[List[str]] = None
        remaining_time = time_limit - (time.time() - start_time)

        sorted_boxes = sorted(uncompleted_my_boxes, key=lambda b: abs(b[0] - my_pos[0]) + abs(b[1] - my_pos[1]))

        for b in sorted_boxes:
            other_boxes = all_boxes - {b}
            sorted_goals = sorted(target_goals, key=lambda g: abs(g[0] - b[0]) + abs(g[1] - b[1]))
            for g in sorted_goals:
                budget = max(0.1, (remaining_time - (time.time() - start_time)) / 2)
                plan = self._gbfs_plan_box(
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
