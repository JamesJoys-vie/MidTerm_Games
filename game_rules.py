from typing import List, Tuple

from map_parser import State

DIRECTIONS = {
    'North': (0, -1),
    'South': (0, 1),
    'East':  (1, 0),
    'West':  (-1, 0)
}
ACTION_NAMES = ['North', 'South', 'East', 'West']

def get_possible_actions(state: State) -> List[str]:
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