def print_banner():
    print("\n" + "=" * 70)
    print("=========SOKOBAN AI SOLVER & TWO-AGENT ARENA=========")
    print("=========Midterm Project - Introduction to AI (TDTU)=======")
    print("=" * 70)
    print("1. [Benchmark]   Run UCS vs A* experiments & Heuristic verification")
    print("2. [Single GUI]  Open Pygame 1-Agent Solver (UCS & A*, Step by Step)")
    print("3. [2-Agent GUI] Open Pygame 2-Agent Competitive Arena")
    print("4. [Quick Test]  Run console search test on sample maps")
    print("5. [Tutorial]    Open system tutorial file TUTORIAL_HUONG_DAN_HE_THONG.md")
    print("0. [Exit]        Exit the program")
    print("=" * 70)


def run_benchmark():
    print("\nRunning experiment suite: python experiment.py ...\n")
    import experiment
    experiment.main()


def run_single_gui():
    print("\nOpening Pygame Single-Agent Solver...")
    print("   Select map:")
    print("   1. maps/example_map.txt (Default)")
    print("   2. maps/test_small.txt")
    print("   3. maps/test_medium.txt")
    print("   4. maps/test_hard.txt")

    choice = input("Enter choice: ").strip()
    map_file = "maps/example_map.txt"
    if choice == "1":
        map_file = "maps/example_map.txt"
    elif choice == "2":
        map_file = "maps/test_small.txt"
    elif choice == "3":
        map_file = "maps/test_medium.txt"
    elif choice == "4":
        map_file = "maps/test_hard.txt"

    import gui_game
    app = gui_game.SokobanGameGUI(map_file)
    app.run()


def run_competitive_gui():
    print("\nOpening Pygame 2-Agent Competitive Arena...")
    steps_input = input("   Enter number of steps n (default 60): ").strip()
    steps_n = int(steps_input) if steps_input.isdigit() else 60

    import competitive_game
    app = competitive_game.CompetitiveGameGUI("maps/competitive_map.txt", steps_n)
    app.run()


def run_quick_test():
    print("\nRunning console search test...")
    from map_parser import MapParser
    from search_algorithms import ucs_search, astar_search, bfs_search, gbfs_search
    from heuristic import heuristic_maze_min_matching

    p = MapParser("maps/test_hard.txt")
    st = p.get_initial_state()

    print(f"Map: {p.filename}")
    print("1. UCS  :", ucs_search(st)['path'])
    print("2. A*   :", astar_search(st, heuristic=heuristic_maze_min_matching)['path'])
    print("3. BFS  :", bfs_search(st)['path'])
    print("4. GBFS :", gbfs_search(st, heuristic=heuristic_maze_min_matching)['path'])


def main():
    while True:
        print_banner()
        choice = input("Enter your choice (0-5): ").strip()

        if choice == "1":
            run_benchmark()
        elif choice == "2":
            run_single_gui()
        elif choice == "3":
            run_competitive_gui()
        elif choice == "4":
            run_quick_test()
        elif choice == "5":
            print("\n📖 Detailed tutorial is located at: TUTORIAL_HUONG_DAN_HE_THONG.md")
        elif choice == "0":
            print("\nThank you for using the program!")
            break
        else:
            print("Invalid choice, please try again!")

        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()
