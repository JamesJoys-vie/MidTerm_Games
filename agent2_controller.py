import time
import heapq
from collections import deque
from typing import Tuple, List, Set, FrozenSet, Optional, Dict, Deque

class Agent2Controller:
    """Independent controller for Agent 2 using Greedy Best-First Search"""

    # Aggressive tuning: steals are worth far more than knock-offs
    KNOCK_OFF_VALUE = 1.5
    STEAL_VALUE = 3.0

    # Face-offs: impatient, replans around the contested cell on the very next step
    PATIENCE = 0
    STALL_WINDOW = 4

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
        self._recent_pos: Deque[Tuple[int, int]] = deque(maxlen=self.STALL_WINDOW)
        self._last_pos: Optional[Tuple[int, int]] = None
        self._last_action = 'Stay'
        self._face_off_cell: Optional[Tuple[int, int]] = None
        self._waited = 0
        self._avoid: Set[Tuple[int, int]] = set()  # Extra cells to plan around this tick

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
        """BFS walking distance from `target` to every reachable floor cell (cached per target)."""
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

    def _walk_dist(self, grid: Tuple[Tuple[str, ...], ...], src: Tuple[int, int], dst: Tuple[int, int]) -> int:
        """Walking (BFS) distance between two cells. The grid is undirected, so one BFS from dst suffices."""
        if src == dst:
            return 0
        return self._get_maze_distances(grid, dst).get(src, 999)

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
            agent_to_box = self._walk_dist(grid, apos, bpos)
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

    def _is_free(self, pos: Tuple[int, int], grid: Tuple[Tuple[str, ...], ...], blocked: Set[Tuple[int, int]]) -> bool:
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        x, y = pos
        return 0 <= x < width and 0 <= y < height and grid[y][x] != '%' and pos not in blocked

    def _bfs_paths(
        self,
        start: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
        grid: Tuple[Tuple[str, ...], ...]
    ) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], str]]:
        """BFS around obstacles. Returns walking distance and first action from `start` for every reachable cell."""
        dist = {start: 0}
        first: Dict[Tuple[int, int], str] = {}
        q = deque([start])
        while q:
            curr = q.popleft()
            for act, (dx, dy) in self.directions.items():
                nxt = (curr[0] + dx, curr[1] + dy)
                if nxt not in dist and self._is_free(nxt, grid, obstacles):
                    dist[nxt] = dist[curr] + 1
                    first[nxt] = first.get(curr, act)
                    q.append(nxt)
        return dist, first

    def _best_box_plan(
        self,
        my_pos: Tuple[int, int],
        box: Tuple[int, int],
        targets: List[Tuple[int, int]],
        all_boxes: Set[Tuple[int, int]],
        opponent_pos: Tuple[int, int],
        grid: Tuple[Tuple[str, ...], ...],
        deadlocks: Set[Tuple[int, int]],
        deadline: float
    ) -> Optional[List[str]]:
        """Plans `box` onto the first reachable target, trying targets nearest to the box first."""
        other_boxes = (all_boxes | self._avoid) - {box}
        box_dist = self._get_maze_distances(grid, box)
        for t in sorted(targets, key=lambda t: box_dist.get(t, 999)):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            plan = self._gbfs_plan_box(
                my_pos, box, t, other_boxes, opponent_pos, grid, deadlocks, max(0.05, remaining / 2)
            )
            if plan:
                return plan
        return None

    def _knock_off_cells(
        self,
        box: Tuple[int, int],
        goals: FrozenSet[Tuple[int, int]],
        all_boxes: Set[Tuple[int, int]],
        grid: Tuple[Tuple[str, ...], ...]
    ) -> List[Tuple[int, int]]:
        """Non-goal neighbours a box on a goal can be pushed onto, which turns it neutral."""
        cells = []
        for dx, dy in self.directions.values():
            off = (box[0] + dx, box[1] + dy)
            if off not in goals and self._is_free(off, grid, all_boxes):
                cells.append(off)
        return cells

    def get_action(
        self,
        my_pos: Tuple[int, int],
        opponent_pos: Tuple[int, int],
        my_boxes: FrozenSet[Tuple[int, int]],
        opponent_boxes: FrozenSet[Tuple[int, int]],
        neutral_boxes: FrozenSet[Tuple[int, int]],
        goals: FrozenSet[Tuple[int, int]],
        grid: Tuple[Tuple[str, ...], ...],
        time_limit: float = 0.95
    ) -> str:
        all_boxes = set(my_boxes) | set(opponent_boxes) | set(neutral_boxes)
        self._avoid = set()
        self._recent_pos.append(my_pos)
        action = None

        # A move that left us in place was cancelled by the resolver: both agents went for
        # the same cell (or the same push). That cell is now contested.
        if self._last_action in self.directions and my_pos == self._last_pos:
            self._face_off_cell = self._contested_cell(my_pos, self._last_action, all_boxes)
            self._waited = 0

        if self._face_off_cell is not None:
            cell = self._face_off_cell
            if abs(opponent_pos[0] - cell[0]) + abs(opponent_pos[1] - cell[1]) > 1:
                self._face_off_cell = None  # The opponent cleared out: carry on with the plan
            elif self._waited < self.PATIENCE:
                self._waited += 1
                action = 'Stay'
            else:
                self._avoid = {cell}  # Done waiting: find another way around the contested cell
                self._face_off_cell = None
        elif len(self._recent_pos) == self.STALL_WINDOW and len(set(self._recent_pos)) == 2:
            # Bouncing between two cells while the opponent mirrors us: route away from it this tick
            self._recent_pos.clear()
            self._avoid = {(opponent_pos[0] + dx, opponent_pos[1] + dy) for dx, dy in self.directions.values()}

        if action is None:
            action = self._choose_action(
                my_pos, opponent_pos, my_boxes, opponent_boxes, neutral_boxes, goals, grid, time_limit
            )
        self._last_pos, self._last_action = my_pos, action
        return action

    def _contested_cell(
        self,
        my_pos: Tuple[int, int],
        action: str,
        all_boxes: Set[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """The cell our cancelled move fought over: the step target, or where a pushed box was headed."""
        dx, dy = self.directions[action]
        cell = (my_pos[0] + dx, my_pos[1] + dy)
        if cell in all_boxes:
            cell = (cell[0] + dx, cell[1] + dy)
        return cell

    def _choose_action(
        self,
        my_pos: Tuple[int, int],
        opponent_pos: Tuple[int, int],
        my_boxes: FrozenSet[Tuple[int, int]],
        opponent_boxes: FrozenSet[Tuple[int, int]],
        neutral_boxes: FrozenSet[Tuple[int, int]],
        goals: FrozenSet[Tuple[int, int]],
        grid: Tuple[Tuple[str, ...], ...],
        time_limit: float = 0.95
    ) -> str:
        """
        Aggressive policy:
        1. Claim the neutral box with the shortest plan onto an empty goal. An unclaimed box
           already on a goal is pushed onto another goal, or nudged off so it can be pushed back.
        2. Once no neutral box can be claimed, steal or knock off opponent boxes, preferring
           ones the opponent is not closer to, with steals weighted well above knock-offs.
        3. If every attack is blocked for now, close in on the opponent's boxes, otherwise hold the center.
        """
        deadline = time.time() + time_limit * 0.9
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        all_boxes = set(my_boxes) | set(opponent_boxes) | set(neutral_boxes)
        deadlocks = self._get_corner_deadlocks(grid, goals)
        empty_goals = [g for g in goals if g not in all_boxes]
        agent_dist = self._get_maze_distances(grid, my_pos)

        # Every neutral box is up for grabs, including unclaimed ones already on a goal ('C' cells)
        claimables = sorted(neutral_boxes, key=lambda b: agent_dist.get(b, 999))
        claims = []
        if empty_goals:
            for b in claimables:
                plan = self._best_box_plan(my_pos, b, empty_goals, all_boxes, opponent_pos, grid, deadlocks, deadline)
                if plan:
                    claims.append(plan)
        if not claims:
            # No goal to move it to: nudge an unclaimed goal box off its goal so it can be pushed back as ours
            for b in claimables:
                if b in goals:
                    cells = [c for c in self._knock_off_cells(b, goals, all_boxes, grid) if c not in deadlocks]
                    plan = self._best_box_plan(my_pos, b, cells, all_boxes, opponent_pos, grid, deadlocks, deadline)
                    if plan:
                        claims.append(plan)
        if claims:
            action = min(claims, key=len)[0]
            self.action_history.append(action)
            return action

        # Contest any box the opponent is not strictly closer to first; guarded ones only lure us around
        targets = sorted(opponent_boxes, key=lambda b: agent_dist.get(b, 999))
        unguarded = [b for b in targets if self._walk_dist(grid, opponent_pos, b) >= agent_dist.get(b, 999)]
        guarded = [b for b in targets if b not in unguarded]
        for group in (unguarded, guarded):
            options = []  # (value per step, -plan length, plan)
            for b in group:
                for value, cells in ((self.STEAL_VALUE, empty_goals),
                                     (self.KNOCK_OFF_VALUE, self._knock_off_cells(b, goals, all_boxes, grid))):
                    plan = self._best_box_plan(my_pos, b, cells, all_boxes, opponent_pos, grid, deadlocks, deadline)
                    if plan:
                        options.append((value / (len(plan) + 1), -len(plan), plan))
            if options:
                action = max(options)[2][0]
                self.action_history.append(action)
                return action

        # Every attack is blocked for now: get next to an opponent box to keep up the pressure
        obstacles = all_boxes | {opponent_pos} | self._avoid
        attack_spots = {
            (b[0] + dx, b[1] + dy)
            for b in opponent_boxes for dx, dy in self.directions.values()
            if self._is_free((b[0] + dx, b[1] + dy), grid, obstacles)
        }
        if attack_spots and my_pos not in attack_spots:
            move = self._bfs_navigate(my_pos, attack_spots, obstacles, grid)
            if move:
                self.action_history.append(move)
                return move

        center_spots = {(width // 2, height // 2), (width // 2 - 1, height // 2), (width // 2 + 1, height // 2)}
        valid_center = {s for s in center_spots if self._is_free(s, grid, obstacles)}
        if not attack_spots and valid_center and my_pos not in valid_center:
            move = self._bfs_navigate(my_pos, valid_center, obstacles, grid)
            if move:
                self.action_history.append(move)
                return move

        return 'Stay'
