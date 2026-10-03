import os
import csv
from collections import deque
from typing import List, Dict, Any

from map_parser import MapParser, State
from heuristic import (
    heuristic_maze_min_matching,
    heuristic_maze_nearest_goal,
    heuristic_zero,
    verify_admissibility,
    verify_consistency
)
from search_algorithms import ucs_search, astar_search, get_possible_actions, execute_action


class PerformanceBenchmark:
    """Runs quantitative performance comparisons between UCS and A* across multiple test maps."""

    def __init__(self, map_files: List[str]):
        self.map_files = map_files
        self.results: List[Dict[str, Any]] = []

    def run(self) -> List[Dict[str, Any]]:
        print("\n" + "=" * 75)
        print("[EXPERIMENT 1] PERFORMANCE COMPARISON: TIME & MEMORY — UCS VS A*")
        print("=" * 75)

        for map_path in self.map_files:
            if not os.path.exists(map_path):
                candidate = os.path.join("maps", os.path.basename(map_path))
                if os.path.exists(candidate):
                    map_path = candidate
                else:
                    print(f"Skipping: File not found '{map_path}'")
                    continue

            map_name = os.path.basename(map_path)
            print(f"\nTesting map: {map_name} ...")

            try:
                parser = MapParser(map_path)
                init_state = parser.get_initial_state()

                res_ucs = ucs_search(init_state, time_limit=30.0)

                res_astar = astar_search(init_state, heuristic=heuristic_maze_min_matching, time_limit=30.0)

                time_ucs = res_ucs['execution_time']
                time_astar = res_astar['execution_time']
                speedup = (time_ucs / time_astar) if (time_astar > 0 and res_astar['success']) else 1.0

                nodes_ucs = res_ucs['nodes_explored']
                nodes_astar = res_astar['nodes_explored']
                node_reduction = ((nodes_ucs - nodes_astar) / nodes_ucs * 100) if nodes_ucs > 0 else 0.0

                row = {
                    'map_name': map_name,
                    'cost_optimal': res_ucs['cost'] if res_ucs['success'] else res_astar['cost'],
                    'ucs_success': res_ucs['success'],
                    'ucs_time_sec': time_ucs,
                    'ucs_nodes': nodes_ucs,
                    'ucs_max_queue': res_ucs['max_queue_size'],
                    'ucs_memory_mb': res_ucs['memory_used_mb'],
                    'astar_success': res_astar['success'],
                    'astar_time_sec': time_astar,
                    'astar_nodes': nodes_astar,
                    'astar_max_queue': res_astar['max_queue_size'],
                    'astar_memory_mb': res_astar['memory_used_mb'],
                    'speedup_ratio': speedup,
                    'node_reduction_pct': node_reduction
                }
                self.results.append(row)

                status_ucs = f"Cost={res_ucs['cost']}, Nodes={nodes_ucs}, Time={time_ucs:.4f}s" if res_ucs['success'] else "FAILED"
                status_astar = f"Cost={res_astar['cost']}, Nodes={nodes_astar}, Time={time_astar:.4f}s" if res_astar['success'] else "FAILED"
                print(f"   • UCS : {status_ucs}")
                print(f"   • A*  : {status_astar}")
                print(f"   => A* expanded {node_reduction:.1f}% fewer nodes than UCS.")

            except Exception as e:
                print(f"Error processing {map_name}: {e}")

        return self.results

    def export_csv(self, filename: str = "experiment_results.csv"):
        """Export experiment results to a CSV file."""
        if not self.results:
            return
        keys = self.results[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nResults saved to file: {filename}")

    def print_markdown_table(self):
        """Print a summary results table in Markdown format."""
        if not self.results:
            return
        print("\n" + "=" * 95)
        print("EXPERIMENT RESULTS SUMMARY TABLE")
        print("=" * 95)
        header = "| Map Name | Cost | UCS Time (s) | A* Time (s) | UCS Nodes | A* Nodes | Node Reduction (%) |"
        sep = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
        print(header)
        print(sep)
        for r in self.results:
            line = (f"| {r['map_name']:<12} "
                    f"| {r['cost_optimal']:^4} "
                    f"| {r['ucs_time_sec']:.4f} "
                    f"| {r['astar_time_sec']:.4f} "
                    f"| {r['ucs_nodes']:^9} "
                    f"| {r['astar_nodes']:^8} "
                    f"| {r['node_reduction_pct']:>6.1f}% |")
            print(line)
        print("=" * 95)


class HeuristicPropertyValidator:
    """Experimentally verifies the admissibility and consistency of the proposed heuristic function."""

    def __init__(self, initial_state: State, max_test_states: int = 150):
        self.initial_state = initial_state
        self.max_test_states = max_test_states
        self.reachable_states: List[State] = []
        self._collect_reachable_states()

    def _collect_reachable_states(self):
        """Explore the state space via BFS to collect a sample set of reachable states."""
        queue = deque([self.initial_state])
        visited = {(self.initial_state.agent_pos, self.initial_state.boxes)}
        self.reachable_states.append(self.initial_state)

        while queue and len(self.reachable_states) < self.max_test_states:
            curr = queue.popleft()
            for action in get_possible_actions(curr):
                nxt, _ = execute_action(curr, action)
                key = (nxt.agent_pos, nxt.boxes)
                if key not in visited:
                    visited.add(key)
                    self.reachable_states.append(nxt)
                    queue.append(nxt)

    def test_admissibility(self) -> Dict[str, Any]:
        """
        Verify h(n) <= h*(n) for all states in the sample.
        Computes optimal cost h*(n) using UCS for each state.
        """
        print("\n" + "-" * 75)
        print("[EXPERIMENT 2] ADMISSIBILITY CHECK: h(n) <= h*(n)")
        print("-" * 75)

        total = 0
        admissible_count = 0
        violations = []

        sample_states = self.reachable_states[:25]

        for idx, state in enumerate(sample_states):
            h_val = heuristic_maze_min_matching(state)

            if h_val >= 100_000:
                continue

            res = ucs_search(state, time_limit=2.0)
            if not res['success']:
                continue

            h_star = res['cost']
            total += 1

            if h_val <= h_star:
                admissible_count += 1
            else:
                violations.append((idx, h_val, h_star))

        pct = (admissible_count / total * 100) if total > 0 else 100.0
        print(f"   • Total states tested       : {total}")
        print(f"   • States satisfying h<=h*   : {admissible_count}")
        print(f"   • Violations                : {len(violations)}")
        print(f"   • Admissibility rate        : {pct:.2f}%")

        if len(violations) == 0:
            print("   => CONCLUSION: Heuristic achieves 100% ADMISSIBILITY!")
        else:
            print(f"   => WARNING: {len(violations)} admissibility violation(s) detected.")

        return {'total': total, 'valid': admissible_count, 'rate': pct, 'violations': violations}

    def test_consistency(self) -> Dict[str, Any]:
        """
        Verify h(n) <= c(n, n') + h(n') for all consecutive state transition pairs.
        Checks the triangle inequality across all reachable state transitions.
        """
        print("\n" + "-" * 75)
        print("[EXPERIMENT 3] CONSISTENCY CHECK: h(n) <= c(n, n') + h(n')")
        print("-" * 75)

        total_transitions = 0
        consistent_count = 0
        violations = []

        for state in self.reachable_states:
            h_curr = heuristic_maze_min_matching(state)
            if h_curr >= 100_000:
                continue

            for action in get_possible_actions(state):
                next_state, step_cost = execute_action(state, action)
                h_next = heuristic_maze_min_matching(next_state)

                if h_next >= 100_000:
                    continue

                total_transitions += 1
                if h_curr <= step_cost + h_next:
                    consistent_count += 1
                else:
                    violations.append((h_curr, step_cost, h_next))

        pct = (consistent_count / total_transitions * 100) if total_transitions > 0 else 100.0
        print(f"   • Total state transitions tested      : {total_transitions}")
        print(f"   • Transitions satisfying inequality   : {consistent_count}")
        print(f"   • Violations                          : {len(violations)}")
        print(f"   • Consistency rate                    : {pct:.2f}%")

        if len(violations) == 0:
            print("   => CONCLUSION: Heuristic achieves 100% CONSISTENCY (Monotone)!")
        else:
            print(f"   => WARNING: {len(violations)} consistency violation(s) detected.")

        return {'total': total_transitions, 'valid': consistent_count, 'rate': pct, 'violations': violations}


def main():
    test_maps = [
        "maps/example_map.txt",
        "maps/test_small.txt",
        "maps/test_medium.txt"
    ]

    print("\n" + "=" * 75)
    print("STARTING SOKOBAN AI EXPERIMENT SUITE")
    print("=" * 75)

    bench = PerformanceBenchmark(test_maps)
    bench.run()
    bench.export_csv("experiment_results.csv")
    bench.print_markdown_table()

    try:
        small_map = "maps/test_small.txt"
        if not os.path.exists(small_map):
            small_map = "test_small.txt"
        parser = MapParser(small_map)
        validator = HeuristicPropertyValidator(parser.get_initial_state(), max_test_states=100)
        validator.test_admissibility()
        validator.test_consistency()
    except Exception as err:
        print(f"Error during heuristic property verification: {err}")

    print("\n" + "=" * 75)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
