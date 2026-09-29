import os
import sys
from typing import List, Tuple, Set, Optional, FrozenSet
from dataclasses import dataclass, field

@dataclass
class State:
    """
    Represents a single search state in the Sokoban state space.
    Stores agent position, box positions (immutable/hashable), goal positions, grid, and cost.
    """
    agent_pos: Tuple[int, int]
    boxes: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset)
    goals: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset)
    grid: Tuple[Tuple[str, ...], ...] = ()
    cost: int = 0
    heuristic_value: int = 0

    def __hash__(self) -> int:
        """Hashes only (agent_pos, boxes) since grid and goals are constant."""
        return hash((self.agent_pos, self.boxes))

    def __eq__(self, other: object) -> bool:
        """Two states are equal if the agent and all boxes share the same positions."""
        if not isinstance(other, State):
            return False
        return self.agent_pos == other.agent_pos and self.boxes == other.boxes

    def __lt__(self, other: 'State') -> bool:
        """Supports comparison in a priority queue when f(n) values are tied."""
        return self.cost < other.cost

    def is_goal_state(self) -> bool:
        """Returns True when all boxes are on goal positions."""
        return self.boxes == self.goals


@dataclass
class TwoAgentState:
    """
    Represents a search state for the competitive two-agent Sokoban mode.
    Tracks both agents' positions and each agent's assigned box set separately.
    """
    agent1_pos: Tuple[int, int]
    agent2_pos: Tuple[int, int]
    boxes_agent1: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset)
    boxes_agent2: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset)
    goals: FrozenSet[Tuple[int, int]] = field(default_factory=frozenset)
    grid: Tuple[Tuple[str, ...], ...] = ()
    step_count: int = 0

    @property
    def all_boxes(self) -> FrozenSet[Tuple[int, int]]:
        """Returns the combined set of all boxes from both agents."""
        return self.boxes_agent1 | self.boxes_agent2

    def __hash__(self) -> int:
        return hash((self.agent1_pos, self.agent2_pos, self.boxes_agent1, self.boxes_agent2))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TwoAgentState):
            return False
        return (self.agent1_pos == other.agent1_pos and
                self.agent2_pos == other.agent2_pos and
                self.boxes_agent1 == other.boxes_agent1 and
                self.boxes_agent2 == other.boxes_agent2)


def identify_corner_deadlocks(
    grid: Tuple[Tuple[str, ...], ...],
    goals: FrozenSet[Tuple[int, int]]
) -> Set[Tuple[int, int]]:
    """
    Identifies all non-goal floor cells that form static corner deadlocks.
    A cell is a corner deadlock if two adjacent perpendicular walls trap a box permanently.
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    deadlocks = set()

    for y in range(height):
        for x in range(width):
            if grid[y][x] == '%' or (x, y) in goals:
                continue

            wall_north = (y - 1 < 0) or (grid[y - 1][x] == '%')
            wall_south = (y + 1 >= height) or (grid[y + 1][x] == '%')
            wall_west = (x - 1 < 0) or (grid[y][x - 1] == '%')
            wall_east = (x + 1 >= width) or (grid[y][x + 1] == '%')

            if (wall_north and wall_west) or \
               (wall_north and wall_east) or \
               (wall_south and wall_west) or \
               (wall_south and wall_east):
                deadlocks.add((x, y))

    return deadlocks


class MapParser:
    """Reads, validates, and preprocesses a Sokoban map from a text file."""

    def __init__(self, filename: str):
        self.filename = self._resolve_path(filename)
        self.grid: List[List[str]] = []
        self.width = 0
        self.height = 0
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.agent1_pos: Optional[Tuple[int, int]] = None
        self.agent2_pos: Optional[Tuple[int, int]] = None
        self.boxes: Set[Tuple[int, int]] = set()
        self.goals: Set[Tuple[int, int]] = set()
        self.is_competitive = False

        self._read_file()
        self._validate()
        self._create_initial_state()

        self.deadlocks = identify_corner_deadlocks(
            self.initial_state.grid,
            self.initial_state.goals
        )

    def _resolve_path(self, path: str) -> str:
        """Locates the map file in the current directory or inside a maps/ subdirectory."""
        if os.path.exists(path):
            return path
        candidate = os.path.join("maps", os.path.basename(path))
        if os.path.exists(candidate):
            return candidate
        base_name = os.path.basename(path)
        if os.path.exists(base_name):
            return base_name
        return path

    def _read_file(self):
        """Reads the map file and parses all cell symbols into internal structures."""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Map file not found: '{self.filename}'")

        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\r\n') for line in f.readlines()]

        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        self.height = len(lines)
        self.width = max(len(line) for line in lines) if self.height > 0 else 0

        for row_idx, line in enumerate(lines):
            row = list(line.ljust(self.width, ' '))
            self.grid.append(row)

            for col_idx, char in enumerate(row):
                pos = (col_idx, row_idx)
                if char == 'A':
                    self.agent_pos = pos
                elif char == '1':
                    self.agent1_pos = pos
                    self.is_competitive = True
                elif char == '2':
                    self.agent2_pos = pos
                    self.is_competitive = True
                elif char == 'B':
                    self.boxes.add(pos)
                elif char == 'D':
                    self.goals.add(pos)
                elif char == 'C':
                    self.boxes.add(pos)
                    self.goals.add(pos)

        if self.agent_pos and not self.agent1_pos:
            self.agent1_pos = self.agent_pos

    def _validate(self):
        """Validates the map structure to ensure the problem is well-formed."""
        if not self.agent_pos and not self.agent1_pos:
            raise ValueError(f"Map '{self.filename}' has no agent position ('A' or '1')!")

        if len(self.boxes) == 0:
            raise ValueError(f"Map '{self.filename}' contains no boxes ('B')!")

        if len(self.goals) == 0:
            raise ValueError(f"Map '{self.filename}' contains no goal positions ('D')!")

        if not self.is_competitive and len(self.boxes) != len(self.goals):
            raise ValueError(
                f"Single-agent map is unbalanced: {len(self.boxes)} boxes != {len(self.goals)} goals!"
            )

    def _create_initial_state(self):
        """Builds the immutable initial State object from the parsed map data."""
        grid_tuple = tuple(tuple(row) for row in self.grid)
        self.initial_state = State(
            agent_pos=self.agent_pos if self.agent_pos else self.agent1_pos,
            boxes=frozenset(self.boxes),
            goals=frozenset(self.goals),
            grid=grid_tuple,
            cost=0,
            heuristic_value=0
        )

    def get_initial_state(self) -> State:
        """Returns the initial state for single-agent search algorithms."""
        return self.initial_state

    def get_two_agent_initial_state(self) -> TwoAgentState:
        """
        Returns the initial state for competitive two-agent mode.
        Splits the box list evenly between the two agents.
        """
        sorted_boxes = sorted(list(self.boxes))
        half = len(sorted_boxes) // 2
        boxes1 = frozenset(sorted_boxes[:half])
        boxes2 = frozenset(sorted_boxes[half:])

        return TwoAgentState(
            agent1_pos=self.agent1_pos if self.agent1_pos else (1, 1),
            agent2_pos=self.agent2_pos if self.agent2_pos else (self.width - 2, self.height - 2),
            boxes_agent1=boxes1,
            boxes_agent2=boxes2,
            goals=frozenset(self.goals),
            grid=self.initial_state.grid,
            step_count=0
        )

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Returns True if the position is within bounds and not a wall ('%')."""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] != '%'
        return False

    def print_map(self):
        """Prints the map grid and key statistics to the console."""
        print("\n" + "=" * 55)
        print(f"MAP INFO: {os.path.basename(self.filename)}")
        print("=" * 55)
        for row in self.grid:
            print("".join(row))
        print("-" * 55)
        print(f"Size       : {self.width} cols x {self.height} rows")
        print(f"Agent      : {self.agent_pos or (self.agent1_pos, self.agent2_pos)}")
        print(f"Boxes (B)  : {len(self.boxes)}")
        print(f"Goals (D)  : {len(self.goals)}")
        print(f"Deadlocks  : {len(self.deadlocks)} corner cells")
        print("=" * 55 + "\n")

if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "maps/example_map.txt"
    try:
        parser = MapParser(test_file)
        parser.print_map()
        st = parser.get_initial_state()
        print(f"Initial State: Agent={st.agent_pos}, Boxes={list(st.boxes)}, IsGoal={st.is_goal_state()}")
    except Exception as e:
        print(f"Error: {e}")
