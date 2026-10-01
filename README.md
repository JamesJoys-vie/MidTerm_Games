# Sokoban with AI Solving

Midterm Project, Introduction to AI (TDTU).

A Sokoban (push-the-box) project with two modes:

- **Single-agent mode:** solves a whole map with classic search (BFS, DFS, UCS, GBFS, A\*), benchmarks UCS against A\*, and plays the solution back in a Pygame GUI.
- **Competitive mode:** two AI agents share one map, race to push their own boxes onto the goals for a fixed number of steps, and the one with more boxes on goals wins.

---

## Table of contents

1. [Project structure](#project-structure)
2. [Map format](#map-format)
3. [Algorithms](#algorithms)
4. [Competitive gameplay](#competitive-gameplay)
5. [Installation](#installation)
6. [Running the project](#running-the-project)
7. [GUI controls](#gui-controls)
8. [Experimental results](#experimental-results)
9. [Group](#group-n04)
---

## Project structure

```
sokoban/
├── main.py                 # Text menu that launches everything
├── map_parser.py           # Map loading, State / TwoAgentState, corner-deadlock detection
├── heuristic.py            # Push-distance heuristic + admissibility/consistency checkers
├── search_algorithms.py    # BFS, DFS, UCS, GBFS, A* over the full state
├── experiment.py           # UCS vs A* benchmark + heuristic verification
├── gui_game.py             # Pygame single-agent solver viewer
├── competitive_game.py     # Pygame 2-agent arena + simultaneous-move rules
├── agent1_controller.py    # Agent 1: weighted A* planner
├── agent2_controller.py    # Agent 2: greedy best-first planner (+ sabotage)
├── experiment_results.csv  # Output of the benchmark
└── maps/
    ├── example_map.txt
    ├── test_small.txt
    ├── test_medium.txt
    ├── test_hard.txt
    └── competitive_map.txt
```

---

## Map format

Maps are plain text files.

| Symbol | Meaning |
|:---:|---|
| `%` | Wall |
| `A` | Agent (single-agent maps) |
| `1`, `2` | Agent 1 and Agent 2 (competitive maps) |
| `B` | Box |
| `D` | Destination (goal) |
| `C` | Box already standing on a goal |
| space | Empty floor |

Single-agent maps must have exactly as many boxes as goals. Competitive maps may have unequal counts. `competitive_map.txt` has 6 boxes and 6 goals.

Example (`maps/test_small.txt`):

```
%%%%%%%
%     %
% A B %
%   D %
%%%%%%%
```

---

## Algorithms

### Single-agent search (`search_algorithms.py`)

The solvers search the **full state** `(agent position, set of all box positions)`. The actions are `North`, `South`, `East` and `West`, and every step costs 1. Moving into a box pushes it if the cell behind it is free (not a wall and not another box).

| Algorithm | Expands nodes by | Optimal? | Notes |
|---|---|:---:|---|
| **BFS** | FIFO order | Yes (unit costs) | Baseline, memory-hungry |
| **DFS** | LIFO, depth ≤ 50 | No | Depth-limited, can miss solutions |
| **UCS** | path cost `g(n)` | Yes | Corner-deadlock pruning, parent-pointer path rebuild, compact sorted-tuple state keys |
| **GBFS** | heuristic `h(n)` only | No | Fast, but paths can be long |
| **A\*** | `f(n) = g(n) + h(n)` | Yes (admissible `h`) | Keeps the best `g` per state, prunes states with `h ≥ 100,000` |

Every search takes a `time_limit` and returns `success=False` if it runs out of time. Results come back as a dictionary with `success`, `path`, `cost`, `nodes_explored`, `execution_time` and, for UCS and A\*, `max_queue_size` and `memory_used_mb`.

### Heuristic (`heuristic.py`)

`heuristic_maze_min_matching` is the default heuristic for A\* and GBFS.

1. **Push distances.** For each goal, a reverse BFS computes how many pushes a box at each cell needs to reach that goal. A cell only counts if the agent could stand behind the box to push it.
2. **Dead squares.** Cells from which no goal is reachable get `h = 100,000` and are pruned. This is stronger than simple corner detection.
3. **Min-cost matching.** The heuristic is the minimum total push distance over all box-to-goal assignments, computed by brute force for 3 boxes or fewer and by the Hungarian algorithm beyond that.

One action moves at most one box by one cell, and push distance counts only pushes, so the heuristic never overestimates the true cost (admissible) and satisfies the triangle inequality (consistent). `experiment.py` checks both properties empirically on a sample of reachable states.

### Competitive agents (`agent1_controller.py`, `agent2_controller.py`)

The controllers do not search the full state. Each turn they plan **one box to one goal** over `(agent position, box position)`.

- Other boxes and the opponent are treated as static obstacles.
- The distance to the goal is a plain BFS walking distance that ignores boxes.
- Pushes into non-goal corner cells are skipped.

| | Agent 1 (cyan) | Agent 2 (magenta) |
|---|---|---|
| Search | Weighted A\*: `h = 2·dist + max(0, manhattan(agent, box) − 1)` | Greedy best-first: `h = 3·dist + manhattan(agent, box)` |
| Behaviour | Shorter plans, slower to compute | Faster, less optimal |
| After finishing its own boxes | Returns `Stay` | Tries to push an opponent's box off a goal, otherwise walks to the map center to block |

Per turn, both agents:

1. Replan from scratch within a time limit of about 1 second.
2. Try the boxes nearest to them first, and for each box the goals nearest to it first.
3. Prefer empty goals, falling back to any goal.
4. Stop early once a plan has 5 steps or fewer.
5. Execute only the first action of the best plan.
6. Take any free neighbouring cell if no plan is found.

There is no minimax or opponent model. Adversarial behaviour comes from replanning every turn, from contested goals, and from Agent 2's sabotage.

---

## Competitive gameplay

**Setup**

- All boxes are sorted by `(x, y)` and split in half. Agent 1 owns the first half and Agent 2 owns the rest.
- Both agents start at `1` and `2` on the map, and all goals are shared.

**Rules** (`resolve_simultaneous_step` in `competitive_game.py`)

- Both agents choose an action at the same time: `North`, `South`, `East`, `West` or `Stay`. Neither gets priority.
- Walls, map edges and the other agent's cell block movement.
- Either agent can push **any** box, including the opponent's. A push fails if the cell behind the box is a wall, another box, or the other agent.
- If the agents swap cells, enter the same cell, push the same box, or push boxes into the same destination, neither moves that tick.
- Boxes keep their owner. Only your own boxes standing on goals count toward your score.

**End condition**

The game ends after `n` steps (default 60, adjustable from 20 to 200). The agent with more of its own boxes on goals wins, and ties are possible.

---

## Installation

### Prerequisites

- **Python 3.8+**
- **pygame 2.0+**, needed only for the two GUIs. The console tools and the benchmark use only the standard library.

```bash
python --version        # on macOS/Linux use python3; on Windows try py -3
```

### Setup

1. Put all `.py` files in one folder and the map files in a `maps/` subfolder (see [Project structure](#project-structure)).
2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. Install pygame:

   ```bash
   pip install "pygame>=2.0"
   python -c "import pygame; print(pygame.version.ver)"
   ```

Run every command below from the project folder.

---

## Running the project

### Option A: interactive menu

```bash
python main.py
```

| Choice | Action |
|:---:|---|
| `1` | Benchmark: UCS vs A\* on `example_map`, `test_small` and `test_medium`, prints a Markdown table, writes `experiment_results.csv`, and checks heuristic admissibility and consistency |
| `2` | Single-agent GUI: pick a map (1–4, Enter for `example_map`) |
| `3` | 2-agent arena: enter the number of steps (Enter for 60), uses `maps/competitive_map.txt` |
| `4` | Console test: runs UCS, A\*, BFS and GBFS on a chosen map and prints each path |
| `0` | Exit |

### Option B: run modules directly

```bash
# Benchmark and heuristic verification
python experiment.py

# Console solver (UCS then A*); the map argument is optional
python search_algorithms.py maps/test_small.txt

# Single-agent GUI
python gui_game.py maps/test_medium.txt

# Competitive arena (Only `competitive_map.txt` is suitable): <map> <steps>
python competitive_game.py maps/competitive_map.txt 80

# Print map info (size, boxes, goals, deadlock cells)
python map_parser.py maps/example_map.txt
```

---

## GUI controls

**Single-agent solver**

| Input | Action |
|---|---|
| A\* / UCS buttons | Choose the algorithm |
| `S` or SOLVE PUZZLE | Run the search (30 s limit), then play the solution |
| `Space` | Pause / resume |
| `←` / `→` | Step back / forward |
| `R` | Reset |
| `Esc` | Quit |

**Competitive arena**

| Input | Action |
|---|---|
| `Space` | Run / pause |
| `→` | Advance one step |
| `↑` / `↓` | Increase / decrease the step limit by 10 |
| `R` | Reset |
| `Esc` | Quit |

Cyan is Agent 1 and magenta is Agent 2. The side panel shows each agent's score, its last action and the step counter.

---

## Experimental results

From `experiment_results.csv` (UCS vs A\* with `heuristic_maze_min_matching`):

| Map | Optimal cost | UCS nodes | A\* nodes | Fewer nodes | UCS time | A\* time |
|---|:---:|---:|---:|---:|---:|---:|
| `test_small` | 4 | 16 | 12 | 25% | 0.0001 s | 0.0001 s |
| `example_map` | 34 | 44,111 | 6,561 | 85% | 0.161 s | 0.045 s |
| `test_medium` | 52 | 929,611 | 126,668 | 86% | 3.59 s | 1.07 s |

On `test_medium`, A\* used about 3.4× less time and about 10× less memory (5 MB vs 49 MB). On the tiny map the heuristic overhead makes A\* slightly slower than UCS.

---

## Group N04
| Student ID | Name               |
|:----------:|:-------------------|
| 524H0081   | Phan Huy Binh      |
| 524H0093   | Trinh Hai Huy      |
| 524H0086   | Tran Hong Nhat Duy |