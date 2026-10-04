import os
import sys
import time
import pygame
from typing import List, Dict, Any

from map_parser import MapParser, State
from search_algorithms import ucs_search, astar_search
from game_rules import execute_action
from heuristic import heuristic_maze_min_matching


COLOR_BG = (30, 34, 42)
COLOR_PANEL = (40, 44, 52)
COLOR_PANEL_BORDER = (60, 66, 78)

COLOR_WALL = (75, 85, 99)
COLOR_WALL_EDGE = (55, 65, 81)
COLOR_FLOOR = (229, 231, 235)
COLOR_FLOOR_ALT = (215, 218, 224)

COLOR_GOAL = (245, 158, 11)
COLOR_GOAL_INNER = (251, 191, 36)

COLOR_BOX = (217, 119, 6)
COLOR_BOX_BORDER = (180, 83, 9)
COLOR_BOX_GOAL = (16, 185, 129)
COLOR_BOX_GOAL_BORDER = (5, 150, 105)

COLOR_AGENT = (59, 130, 246)
COLOR_AGENT_BORDER = (29, 78, 216)

COLOR_TEXT_MAIN = (243, 244, 246)
COLOR_TEXT_MUTED = (156, 163, 175)
COLOR_ACCENT = (14, 165, 233)

COLOR_BTN_IDLE = (55, 65, 81)
COLOR_BTN_HOVER = (75, 85, 99)
COLOR_BTN_ACTIVE = (14, 165, 233)


class Button:
    """Represents an interactive graphical button with hover and click feedback."""

    def __init__(self, x: int, y: int, width: int, height: int, text: str, action_id: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action_id = action_id
        self.is_hovered = False
        self.is_active = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if the button was left-clicked."""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        """Draws the button onto the given surface."""
        if self.is_active:
            bg_color = COLOR_BTN_ACTIVE
        elif self.is_hovered:
            bg_color = COLOR_BTN_HOVER
        else:
            bg_color = COLOR_BTN_IDLE

        pygame.draw.rect(surface, bg_color, self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, self.rect, width=2, border_radius=6)

        text_surf = font.render(self.text, True, COLOR_TEXT_MAIN)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)


class MapRenderer:
    """Handles rendering of the Sokoban map including the agent, boxes, goals, and walls."""

    def __init__(self, tile_size: int = 56):
        self.tile_size = tile_size

    def render(self, surface: pygame.Surface, state: State, offset_x: int, offset_y: int):
        """Draws the full map for the given state at the specified pixel offset."""
        grid = state.grid
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        ts = self.tile_size

        for y in range(height):
            for x in range(width):
                px = offset_x + x * ts
                py = offset_y + y * ts
                cell_rect = pygame.Rect(px, py, ts, ts)

                char = grid[y][x]
                if char == '%':
                    pygame.draw.rect(surface, COLOR_WALL, cell_rect)
                    pygame.draw.rect(surface, COLOR_WALL_EDGE, cell_rect, width=3)
                    pygame.draw.line(surface, COLOR_WALL_EDGE, (px, py), (px + ts, py + ts), 1)
                else:
                    floor_color = COLOR_FLOOR if (x + y) % 2 == 0 else COLOR_FLOOR_ALT
                    pygame.draw.rect(surface, floor_color, cell_rect)
                    pygame.draw.rect(surface, (200, 204, 210), cell_rect, width=1)

        for gx, gy in state.goals:
            center = (offset_x + gx * ts + ts // 2, offset_y + gy * ts + ts // 2)
            pygame.draw.circle(surface, COLOR_GOAL, center, ts // 3, width=3)
            pygame.draw.circle(surface, COLOR_GOAL_INNER, center, ts // 6)

        for bx, by in state.boxes:
            px = offset_x + bx * ts + 5
            py = offset_y + by * ts + 5
            box_rect = pygame.Rect(px, py, ts - 10, ts - 10)

            is_on_goal = (bx, by) in state.goals
            box_bg = COLOR_BOX_GOAL if is_on_goal else COLOR_BOX
            box_border = COLOR_BOX_GOAL_BORDER if is_on_goal else COLOR_BOX_BORDER

            pygame.draw.rect(surface, box_bg, box_rect, border_radius=8)
            pygame.draw.rect(surface, box_border, box_rect, width=3, border_radius=8)

            pygame.draw.line(surface, box_border, (px + 6, py + 6), (px + ts - 16, py + ts - 16), 2)
            pygame.draw.line(surface, box_border, (px + ts - 16, py + 6), (px + 6, py + ts - 16), 2)

        ax, ay = state.agent_pos
        agent_center = (offset_x + ax * ts + ts // 2, offset_y + ay * ts + ts // 2)
        radius = ts // 2 - 6

        pygame.draw.circle(surface, COLOR_AGENT, agent_center, radius)
        pygame.draw.circle(surface, COLOR_AGENT_BORDER, agent_center, radius, width=3)

        eye_radius = max(2, ts // 14)
        eye_offset_x = ts // 8
        eye_offset_y = ts // 10
        pygame.draw.circle(surface, (255, 255, 255), (agent_center[0] - eye_offset_x, agent_center[1] - eye_offset_y), eye_radius)
        pygame.draw.circle(surface, (255, 255, 255), (agent_center[0] + eye_offset_x, agent_center[1] - eye_offset_y), eye_radius)
        pygame.draw.circle(surface, (0, 0, 0), (agent_center[0] - eye_offset_x, agent_center[1] - eye_offset_y), max(1, eye_radius // 2))
        pygame.draw.circle(surface, (0, 0, 0), (agent_center[0] + eye_offset_x, agent_center[1] - eye_offset_y), max(1, eye_radius // 2))


class SokobanGameGUI:
    """Manages the main game loop, user input handling, and overall game state control."""

    def __init__(self, map_path: str = "maps/example_map.txt"):
        pygame.init()
        pygame.font.init()

        self.parser = MapParser(map_path)
        self.initial_state = self.parser.get_initial_state()

        self.tile_size = 56
        self.board_width = self.parser.width * self.tile_size
        self.board_height = self.parser.height * self.tile_size

        self.panel_width = 320
        self.window_width = max(800, self.board_width + self.panel_width + 40)
        self.window_height = max(520, self.board_height + 40)

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(f"Sokoban AI Solver - {os.path.basename(self.parser.filename)}")

        self.clock = pygame.time.Clock()
        self.renderer = MapRenderer(self.tile_size)

        self.font_title = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_main = pygame.font.SysFont("Arial", 14)
        self.font_bold = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 12)

        self.history: List[State] = [self.initial_state]
        self.current_step = 0
        self.solution_path: List[str] = []
        self.solution_info: Dict[str, Any] = {}

        self.selected_algo = "A*"
        self.is_playing = False
        self.status_message = "Ready (Press Solve or Space)"
        self.last_step_time = 0.0
        self.step_delay = 0.35

        self._init_buttons()

    def _init_buttons(self):
        """Initializes the interactive buttons on the right control panel."""
        panel_x = self.window_width - self.panel_width + 20
        btn_w = 130
        btn_h = 36

        self.btn_algo_astar = Button(panel_x, 80, btn_w, btn_h, "A* Search", "ALGO_ASTAR")
        self.btn_algo_astar.is_active = True
        self.btn_algo_ucs = Button(panel_x + 140, 80, btn_w, btn_h, "UCS Search", "ALGO_UCS")

        self.btn_solve = Button(panel_x, 130, 270, btn_h, "SOLVE PUZZLE", "SOLVE")
        self.btn_pause = Button(panel_x, 175, btn_w, btn_h, "Pause", "PAUSE")
        self.btn_reset = Button(panel_x + 140, 175, btn_w, btn_h, "Reset", "RESET")

        self.btn_prev = Button(panel_x, 220, btn_w, btn_h, "Step Back", "PREV")
        self.btn_next = Button(panel_x + 140, 220, btn_w, btn_h, "Step Forward", "NEXT")

        self.buttons = [
            self.btn_algo_astar, self.btn_algo_ucs,
            self.btn_solve, self.btn_pause, self.btn_reset,
            self.btn_prev, self.btn_next
        ]

    def solve_puzzle(self):
        """Runs the selected search algorithm to find a solution for the current map."""
        self.status_message = f"Searching with {self.selected_algo}..."
        self._draw_frame()
        pygame.display.flip()

        if self.selected_algo == "A*":
            res = astar_search(self.initial_state, heuristic=heuristic_maze_min_matching, time_limit=30.0)
        else:
            res = ucs_search(self.initial_state, time_limit=30.0)

        self.solution_info = res

        if res['success']:
            self.solution_path = res['path']
            self.history = [self.initial_state]
            curr = self.initial_state
            for act in self.solution_path:
                curr, _ = execute_action(curr, act)
                self.history.append(curr)

            self.current_step = 0
            self.is_playing = True
            self.status_message = f"Solution found! ({len(self.solution_path)} steps)"
        else:
            self.status_message = "No solution found or search timed out!"

    def handle_events(self) -> bool:
        """Processes keyboard and mouse events. Returns False to signal quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.is_playing = not self.is_playing
                    self.status_message = "Playing..." if self.is_playing else "Paused"
                elif event.key == pygame.K_LEFT:
                    self.is_playing = False
                    if self.current_step > 0:
                        self.current_step -= 1
                        self.status_message = f"Stepped back to step {self.current_step}"
                elif event.key == pygame.K_RIGHT:
                    self.is_playing = False
                    if self.current_step < len(self.history) - 1:
                        self.current_step += 1
                        self.status_message = f"Advanced to step {self.current_step}"
                elif event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_s:
                    self.solve_puzzle()

            for btn in self.buttons:
                if btn.handle_event(event):
                    if btn.action_id == "ALGO_ASTAR":
                        self.selected_algo = "A*"
                        self.btn_algo_astar.is_active = True
                        self.btn_algo_ucs.is_active = False
                    elif btn.action_id == "ALGO_UCS":
                        self.selected_algo = "UCS"
                        self.btn_algo_astar.is_active = False
                        self.btn_algo_ucs.is_active = True
                    elif btn.action_id == "SOLVE":
                        self.solve_puzzle()
                    elif btn.action_id == "PAUSE":
                        self.is_playing = not self.is_playing
                        btn.text = "Resume" if not self.is_playing else "Pause"
                    elif btn.action_id == "RESET":
                        self.reset_game()
                    elif btn.action_id == "PREV":
                        self.is_playing = False
                        if self.current_step > 0:
                            self.current_step -= 1
                    elif btn.action_id == "NEXT":
                        self.is_playing = False
                        if self.current_step < len(self.history) - 1:
                            self.current_step += 1

        return True

    def reset_game(self):
        """Resets the game to its initial state."""
        self.history = [self.initial_state]
        self.current_step = 0
        self.is_playing = False
        self.solution_path = []
        self.solution_info = {}
        self.status_message = "Reset to initial state"

    def update(self):
        """Advances playback by one step when auto-play is active and the delay has elapsed."""
        if self.is_playing:
            now = time.time()
            if now - self.last_step_time >= self.step_delay:
                self.last_step_time = now
                if self.current_step < len(self.history) - 1:
                    self.current_step += 1
                else:
                    self.is_playing = False
                    self.status_message = " Goal Achieved!"

    def _draw_frame(self):
        """Draws the full frame: Sokoban map on the left, HUD control panel on the right."""
        self.screen.fill(COLOR_BG)

        board_area_width = self.window_width - self.panel_width
        offset_x = max(20, (board_area_width - self.board_width) // 2)
        offset_y = max(20, (self.window_height - self.board_height) // 2)

        current_state = self.history[self.current_step]
        self.renderer.render(self.screen, current_state, offset_x, offset_y)

        panel_rect = pygame.Rect(self.window_width - self.panel_width, 0, self.panel_width, self.window_height)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (panel_rect.left, 0), (panel_rect.left, self.window_height), 2)

        px = panel_rect.left + 20

        title_surf = self.font_title.render("🎮 SOKOBAN AI SOLVER", True, COLOR_ACCENT)
        self.screen.blit(title_surf, (px, 25))

        subtitle_surf = self.font_small.render(f"Map: {os.path.basename(self.parser.filename)}", True, COLOR_TEXT_MUTED)
        self.screen.blit(subtitle_surf, (px, 50))

        for btn in self.buttons:
            btn.draw(self.screen, self.font_bold)

        info_y = 275
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (px, info_y), (panel_rect.right - 20, info_y), 1)
        info_y += 15

        label_stat = self.font_bold.render("PERFORMANCE STATS:", True, COLOR_TEXT_MAIN)
        self.screen.blit(label_stat, (px, info_y))
        info_y += 24

        total_actions = len(self.solution_path)
        cost_val = self.solution_info.get('cost', self.current_step)
        time_val = self.solution_info.get('execution_time', 0.0)
        nodes_val = self.solution_info.get('nodes_explored', 0)

        lines = [
            f"• Algorithm      : {self.selected_algo}",
            f"• Step           : {self.current_step} / {total_actions}",
            f"• Total Cost     : {cost_val}",
            f"• Search Time    : {time_val:.4f} s",
            f"• Nodes Explored : {nodes_val:,}",
            f"• Status         : {self.status_message}"
        ]

        for line in lines:
            txt_surf = self.font_main.render(line, True, COLOR_TEXT_MAIN)
            self.screen.blit(txt_surf, (px, info_y))
            info_y += 22

        info_y += 10
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (px, info_y), (panel_rect.right - 20, info_y), 1)
        info_y += 12

        label_ctrl = self.font_bold.render("KEYBOARD SHORTCUTS:", True, COLOR_ACCENT)
        self.screen.blit(label_ctrl, (px, info_y))
        info_y += 22

        controls = [
            "• [Space]      : Pause / Resume",
            "• [Left Arrow] : Step Backward",
            "• [Right Arrow]: Step Forward",
            "• [R]          : Reset",
            "• [S]          : Solve"
        ]

        for ctrl in controls:
            ctrl_surf = self.font_small.render(ctrl, True, COLOR_TEXT_MUTED)
            self.screen.blit(ctrl_surf, (px, info_y))
            info_y += 18

    def run(self):
        """Runs the main application loop."""
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self._draw_frame()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


def main():
    map_arg = sys.argv[1] if len(sys.argv) > 1 else "maps/example_map.txt"
    app = SokobanGameGUI(map_arg)
    app.run()


if __name__ == "__main__":
    main()
