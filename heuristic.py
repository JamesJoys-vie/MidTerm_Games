from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from map_parser import State

Cell = Tuple[int, int]
Grid = Tuple[Tuple[str, ...], ...]

DEADLOCK_VALUE = 100_000 # prunes h >= 100_000
UNREACHABLE = DEADLOCK_VALUE
USE_PUSH_DISTANCE = True # False -> plain maze (walking) BFS distance
_MEMO_LIMIT = 1_000_000

_DIRS = ((0, -1), (0, 1), (1, 0), (-1, 0))

def precompute_maze_distances(
    grid: Grid,
    goals: FrozenSet[Cell],
) -> Dict[Cell, Dict[Cell, int]]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    distance_map: Dict[Cell, Dict[Cell, int]] = {}

    for goal in goals:
        dist = {goal: 0}
        frontier = [goal]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for x, y in frontier:
                for dx, dy in _DIRS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] != '%':
                        c = (nx, ny)
                        if c not in dist:
                            dist[c] = d
                            nxt.append(c)
            frontier = nxt
        distance_map[goal] = dist

    return distance_map


def precompute_push_distances(
    grid: Grid,
    goals: FrozenSet[Cell],
) -> Dict[Cell, Dict[Cell, int]]:
    
    height = len(grid)
    width = len(grid[0]) if height else 0
    floor = [[grid[y][x] != '%' for x in range(width)] for y in range(height)]
    push_map: Dict[Cell, Dict[Cell, int]] = {}

    for goal in goals:
        dist = {goal: 0}
        frontier = [goal]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for x, y in frontier:
                for dx, dy in _DIRS:
                    ax, ay = x - 2 * dx, y - 2 * dy          # agent cell (bounds imply c in bounds)
                    if 0 <= ax < width and 0 <= ay < height:
                        cx, cy = x - dx, y - dy
                        if floor[cy][cx] and floor[ay][ax]:
                            c = (cx, cy)
                            if c not in dist:
                                dist[c] = d
                                nxt.append(c)
            frontier = nxt
        push_map[goal] = dist

    return push_map

class _Context:
    __slots__ = ("grid", "goals", "goal_list", "dist_by_goal", "rows", "dead", "memo_match", "memo_nearest")

    def __init__(self, grid: Grid, goals: FrozenSet[Cell]):
        self.grid = grid
        self.goals = goals
        self.goal_list: List[Cell] = sorted(goals)

        push = precompute_push_distances(grid, goals)
        alive: Set[Cell] = set()
        for g in self.goal_list:
            alive.update(push[g])

        height = len(grid)
        width = len(grid[0]) if height else 0
        self.dead: Set[Cell] = {
            (x, y)
            for y in range(height) for x in range(width)
            if grid[y][x] != '%' and (x, y) not in alive
        }

        self.dist_by_goal = push if USE_PUSH_DISTANCE else precompute_maze_distances(grid, goals)

        cells: Set[Cell] = set()
        for g in self.goal_list:
            cells.update(self.dist_by_goal[g])
        tables = [self.dist_by_goal[g] for g in self.goal_list]
        self.rows: Dict[Cell, Tuple[int, ...]] = {
            c: tuple(t.get(c, UNREACHABLE) for t in tables) for c in cells
        }

        self.memo_match: Dict[FrozenSet[Cell], int] = {}
        self.memo_nearest: Dict[FrozenSet[Cell], int] = {}


_CONTEXTS: Dict[Tuple[Grid, FrozenSet[Cell]], _Context] = {}
_last: Optional[_Context] = None


def clear_heuristic_caches() -> None:
    global _last
    _CONTEXTS.clear()
    _last = None


def _get_context(state: State) -> _Context:
    global _last
    ctx = _last
    if ctx is not None and ctx.grid is state.grid and ctx.goals is state.goals:
        return ctx
    key = (state.grid, state.goals)
    ctx = _CONTEXTS.get(key)
    if ctx is None:
        ctx = _Context(state.grid, state.goals)
        _CONTEXTS[key] = ctx
    _last = ctx
    return ctx


def get_precomputed_distances(state: State) -> Dict[Cell, Dict[Cell, int]]:
    return _get_context(state).dist_by_goal


def get_deadlock_set(state: State) -> Set[Cell]:
    return _get_context(state).dead

def _hungarian(cost: List[List[int]], n: int, m: int) -> int:
    big = 1 << 60
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [big] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            row = cost[i0 - 1]
            ui0 = u[i0]
            delta = big
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - ui0 - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return -v[0]

def heuristic_maze_min_matching(state: State) -> int:
    ctx = _get_context(state)
    boxes = state.boxes

    cached = ctx.memo_match.get(boxes)
    if cached is not None:
        return cached

    result = _min_matching(ctx, boxes)
    if len(ctx.memo_match) >= _MEMO_LIMIT:
        ctx.memo_match.clear()
    ctx.memo_match[boxes] = result
    return result


def _min_matching(ctx: _Context, boxes: FrozenSet[Cell]) -> int:
    dead = ctx.dead
    rows = ctx.rows
    goals = ctx.goals

    free: List[Tuple[int, ...]] = []
    for b in boxes:
        if b in dead:
            return DEADLOCK_VALUE
        if b in goals:
            continue
        row = rows.get(b)
        if row is None:
            return DEADLOCK_VALUE
        free.append(row)

    n = len(free)
    if n == 0:
        return 0

    gidx = [i for i, g in enumerate(ctx.goal_list) if g not in boxes]
    m = len(gidx)
    if n > m:
        total = sum(min(r) for r in free)
        return DEADLOCK_VALUE if total >= DEADLOCK_VALUE else total

    matrix = [[r[j] for j in gidx] for r in free]

    if n == 1:
        total = min(matrix[0])
    elif n == 2 and m == 2:
        a, b = matrix
        total = min(a[0] + b[1], a[1] + b[0])
    elif n == 3 and m == 3:
        a, b, c = matrix
        total = min(a[0] + b[1] + c[2], a[0] + b[2] + c[1],
                    a[1] + b[0] + c[2], a[1] + b[2] + c[0],
                    a[2] + b[0] + c[1], a[2] + b[1] + c[0])
    else:
        total = _hungarian(matrix, n, m)

    return DEADLOCK_VALUE if total >= DEADLOCK_VALUE else total


def heuristic_maze_nearest_goal(state: State) -> int:
    ctx = _get_context(state)
    boxes = state.boxes

    cached = ctx.memo_nearest.get(boxes)
    if cached is not None:
        return cached

    dead = ctx.dead
    rows = ctx.rows
    total = 0
    for b in boxes:
        if b in dead:
            total = DEADLOCK_VALUE
            break
        row = rows.get(b)
        if row is None:
            total = DEADLOCK_VALUE
            break
        total += min(row)
        if total >= DEADLOCK_VALUE:
            total = DEADLOCK_VALUE
            break

    if len(ctx.memo_nearest) >= _MEMO_LIMIT:
        ctx.memo_nearest.clear()
    ctx.memo_nearest[boxes] = total
    return total


def heuristic_zero(state: State) -> int:
    """h(n) = 0; reduces A* to Uniform Cost Search (baseline)."""
    return 0


# Property checks
def verify_admissibility(h_value: int, optimal_cost_to_goal: int) -> Tuple[bool, str]:
    """Check h(n) <= h*(n)."""
    ok = h_value <= optimal_cost_to_goal
    msg = (f"h(n)={h_value} <= h*(n)={optimal_cost_to_goal} [PASS - Admissible]"
           if ok else
           f"h(n)={h_value} > h*(n)={optimal_cost_to_goal} [VIOLATION]")
    return ok, msg


def verify_consistency(h_current: int, step_cost: int, h_next: int) -> Tuple[bool, str]:
    """Check h(n) <= c(n, n') + h(n')."""
    ok = h_current <= step_cost + h_next
    msg = (f"h(n)={h_current} <= c(n,n')={step_cost} + h(n')={h_next} [PASS - Consistent]"
           if ok else
           f"h(n)={h_current} > {step_cost + h_next} [VIOLATION]")
    return ok, msg


if __name__ == "__main__":
    mode = "push distance" if USE_PUSH_DISTANCE else "maze (walking) distance"
    print(f"Module heuristic.py is ready. Default heuristic: heuristic_maze_min_matching ({mode}).")