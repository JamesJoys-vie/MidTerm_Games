import os
import sys
import time
import pygame
from typing import Tuple, Set

from map_parser import MapParser, TwoAgentState
from agent1_controller import Agent1Controller
from agent2_controller import Agent2Controller

COLOR_BG = (24, 26, 32)
COLOR_PANEL = (33, 37, 46)
COLOR_PANEL_BORDER = (55, 62, 78)

COLOR_WALL = (75, 85, 99)
COLOR_WALL_EDGE = (55, 65, 81)
COLOR_FLOOR = (235, 237, 240)
COLOR_FLOOR_ALT = (220, 224, 230)

COLOR_GOAL = (245, 158, 11)
COLOR_GOAL_INNER = (251, 191, 36)

COLOR_AGENT1 = (14, 165, 233)
COLOR_AGENT1_BORDER = (2, 132, 199)
COLOR_BOX_AGENT1 = (16, 185, 129)
COLOR_BOX_AGENT1_BORDER = (5, 150, 105)

COLOR_AGENT2 = (244, 63, 94)
COLOR_AGENT2_BORDER = (225, 29, 72)
COLOR_BOX_AGENT2 = (168, 85, 247)
COLOR_BOX_AGENT2_BORDER = (147, 51, 234)

COLOR_BOX_NEUTRAL = (156, 163, 175)
COLOR_BOX_NEUTRAL_BORDER = (107, 114, 128)
COLOR_BOX_UNCLAIMED_GOAL = (250, 204, 21)
COLOR_BOX_UNCLAIMED_GOAL_BORDER = (202, 138, 4)

COLOR_BOX_DONE_BORDER = (245, 158, 11)

COLOR_TEXT_MAIN = (243, 244, 246)
COLOR_TEXT_MUTED = (156, 163, 175)
COLOR_TEXT_GOLD = (251, 191, 36)

COLOR_BTN_IDLE = (55, 65, 81)
COLOR_BTN_HOVER = (75, 85, 99)
COLOR_BTN_ACTIVE = (14, 165, 233)


def resolve_simultaneous_step(
    agent1_pos: Tuple[int, int],
    agent2_pos: Tuple[int, int],
    action1: str,
    action2: str,
    boxes_neutral: Set[Tuple[int, int]],
    boxes1: Set[Tuple[int, int]],
    boxes2: Set[Tuple[int, int]],
    goals: Set[Tuple[int, int]],
    grid: Tuple[Tuple[str, ...], ...]
) -> Tuple[Tuple[int, int], Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Resolves one simultaneous tick. A box pushed onto a goal is claimed by the pusher;
    a box pushed onto a non-goal cell becomes neutral.
    Returns (agent1_pos, agent2_pos, boxes_neutral, boxes1, boxes2).
    """

    dirs = {
        'North': (0, -1), 'South': (0, 1),
        'East': (1, 0),  'West': (-1, 0),
        'Stay': (0, 0)
    }
    dx1, dy1 = dirs.get(action1, (0, 0))
    dx2, dy2 = dirs.get(action2, (0, 0))

    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    all_boxes = set(boxes_neutral) | set(boxes1) | set(boxes2)
    unchanged = (agent1_pos, agent2_pos, boxes_neutral, boxes1, boxes2)

    cand1 = (agent1_pos[0] + dx1, agent1_pos[1] + dy1)
    cand2 = (agent2_pos[0] + dx2, agent2_pos[1] + dy2)

    if cand1 == agent2_pos and cand2 == agent1_pos:
        return unchanged
    if cand1 == cand2 and cand1 != agent1_pos and cand2 != agent2_pos:
        return unchanged

    def eval_agent(pos, cand, dx, dy, other_pos):
        new_pos = pos
        push_from = None
        push_to = None
        if 0 <= cand[0] < width and 0 <= cand[1] < height:
            if grid[cand[1]][cand[0]] != '%' and cand != other_pos:
                if cand in all_boxes:
                    push_pos = (cand[0] + dx, cand[1] + dy)
                    if 0 <= push_pos[0] < width and 0 <= push_pos[1] < height:
                        if grid[push_pos[1]][push_pos[0]] != '%' and push_pos not in all_boxes and push_pos != other_pos:
                            new_pos = cand
                            push_from = cand
                            push_to = push_pos
                else:
                    new_pos = cand
        return new_pos, push_from, push_to

    new_a1, push1_from, push1_to = eval_agent(agent1_pos, cand1, dx1, dy1, agent2_pos)
    new_a2, push2_from, push2_to = eval_agent(agent2_pos, cand2, dx2, dy2, agent1_pos)

    if push1_from and push1_from == push2_from:
        return unchanged
    if push1_to and push2_to and push1_to == push2_to:
        return unchanged
    if push1_to and new_a2 == push1_to:
        return unchanged
    if push2_to and new_a1 == push2_to:
        return unchanged
    new_neutral = set(boxes_neutral)
    new_boxes1 = set(boxes1)
    new_boxes2 = set(boxes2)

    def apply_push(push_from, push_to, pusher_boxes):
        for box_set in (new_neutral, new_boxes1, new_boxes2):
            box_set.discard(push_from)
        if push_to in goals:
            pusher_boxes.add(push_to)
        else:
            new_neutral.add(push_to)

    if push1_from:
        apply_push(push1_from, push1_to, new_boxes1)
    if push2_from:
        apply_push(push2_from, push2_to, new_boxes2)
    return new_a1, new_a2, new_neutral, new_boxes1, new_boxes2


class CompetitiveGameGUI:
    def __init__(self, map_path: str = "maps/competitive_map.txt", total_steps_n: int = 60):
        pygame.init()
        pygame.font.init()

        self.map_parser = MapParser(map_path)
        self.state = self.map_parser.get_two_agent_initial_state()

        self.controller1 = Agent1Controller("Agent 1 (Cyan)")
        self.controller2 = Agent2Controller("Agent 2 (Magenta)")

        self.max_steps_n = total_steps_n
        self.current_step = 0
        self.is_running = False
        self.is_game_over = False
        self.winner_text = ""

        self.tile_size = 52
        self.board_width = self.map_parser.width * self.tile_size
        self.board_height = self.map_parser.height * self.tile_size
        self.panel_width = 330

        self.window_width = max(860, self.board_width + self.panel_width + 40)
        self.window_height = max(560, self.board_height + 40)

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Sokoban 2-Agent Competitive Arena")

        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_main = pygame.font.SysFont("Arial", 14)
        self.font_bold = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_huge = pygame.font.SysFont("Arial", 26, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 12)

        self.last_step_time = 0.0
        self.step_interval = 0.28

        self.action1_last = "Ready"
        self.action2_last = "Ready"
        self.history = []

    def step(self):
        if self.is_game_over or self.current_step >= self.max_steps_n:
            self._evaluate_winner()
            return

        act1 = self.controller1.get_action(
            my_pos=self.state.agent1_pos,
            opponent_pos=self.state.agent2_pos,
            my_boxes=self.state.boxes_agent1,
            opponent_boxes=self.state.boxes_agent2,
            neutral_boxes=self.state.boxes_neutral,
            goals=self.state.goals,
            grid=self.state.grid,
            time_limit=0.9
        )
        act2 = self.controller2.get_action(
            my_pos=self.state.agent2_pos,
            opponent_pos=self.state.agent1_pos,
            my_boxes=self.state.boxes_agent2,
            opponent_boxes=self.state.boxes_agent1,
            neutral_boxes=self.state.boxes_neutral,
            goals=self.state.goals,
            grid=self.state.grid,
            time_limit=0.9
        )

        self.history.append((self.state, self.action1_last, self.action2_last))
        self.action1_last = act1
        self.action2_last = act2

        new_a1, new_a2, new_bn, new_b1, new_b2 = resolve_simultaneous_step(
            self.state.agent1_pos,
            self.state.agent2_pos,
            act1, act2,
            set(self.state.boxes_neutral),
            set(self.state.boxes_agent1),
            set(self.state.boxes_agent2),
            self.state.goals,
            self.state.grid
        )

        self.current_step += 1
        self.state = TwoAgentState(
            agent1_pos=new_a1,
            agent2_pos=new_a2,
            boxes_neutral=frozenset(new_bn),
            boxes_agent1=frozenset(new_b1),
            boxes_agent2=frozenset(new_b2),
            goals=self.state.goals,
            grid=self.state.grid,
            step_count=self.current_step
        )

        if self.current_step >= self.max_steps_n:
            self._evaluate_winner()

    def _evaluate_winner(self):
        self.is_running = False
        self.is_game_over = True

        score1 = sum(1 for b in self.state.boxes_agent1 if b in self.state.goals)
        score2 = sum(1 for b in self.state.boxes_agent2 if b in self.state.goals)

        if score1 > score2:
            self.winner_text = f"AGENT 1 Win! ({score1} vs {score2})"
        elif score2 > score1:
            self.winner_text = f"AGENT 2 Win! ({score2} vs {score1})"
        else:
            self.winner_text = f"Tied! ({score1} - {score2})"

    def reset(self):
        self.state = self.map_parser.get_two_agent_initial_state()
        self.current_step = 0
        self.is_running = False
        self.is_game_over = False
        self.winner_text = ""
        self.action1_last = "Ready"
        self.action2_last = "Ready"
        self.history = []

    def step_back(self):
        if not self.history:
            return
        self.is_running = False
        self.is_game_over = False
        self.winner_text = ""
        self.state, self.action1_last, self.action2_last = self.history.pop()
        self.current_step = self.state.step_count

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    if not self.is_game_over:
                        self.is_running = not self.is_running
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_RIGHT:
                    if not self.is_game_over:
                        self.step()
                elif event.key == pygame.K_LEFT:
                    self.step_back()
                elif event.key == pygame.K_UP:
                    self.max_steps_n = min(200, self.max_steps_n + 10)
                elif event.key == pygame.K_DOWN:
                    self.max_steps_n = max(20, self.max_steps_n - 10)

        return True

    def update(self):
        if self.is_running and not self.is_game_over:
            now = time.time()
            if now - self.last_step_time >= self.step_interval:
                self.last_step_time = now
                self.step()

    def _draw_board(self, offset_x: int, offset_y: int):
        ts = self.tile_size
        grid = self.state.grid
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        for y in range(height):
            for x in range(width):
                px = offset_x + x * ts
                py = offset_y + y * ts
                rect = pygame.Rect(px, py, ts, ts)

                if grid[y][x] == '%':
                    pygame.draw.rect(self.screen, COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, COLOR_WALL_EDGE, rect, width=2)
                else:
                    color = COLOR_FLOOR if (x + y) % 2 == 0 else COLOR_FLOOR_ALT
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (210, 214, 220), rect, width=1)

        for gx, gy in self.state.goals:
            center = (offset_x + gx * ts + ts // 2, offset_y + gy * ts + ts // 2)
            pygame.draw.circle(self.screen, COLOR_GOAL, center, ts // 3, width=3)
            pygame.draw.circle(self.screen, COLOR_GOAL_INNER, center, ts // 6)

        for bx, by in self.state.boxes_neutral:
            px = offset_x + bx * ts + 5
            py = offset_y + by * ts + 5
            rect = pygame.Rect(px, py, ts - 10, ts - 10)

            # A neutral box on a goal (a 'C' cell) scores for nobody until an agent re-occupies it
            if (bx, by) in self.state.goals:
                fill, border = COLOR_BOX_UNCLAIMED_GOAL, COLOR_BOX_UNCLAIMED_GOAL_BORDER
            else:
                fill, border = COLOR_BOX_NEUTRAL, COLOR_BOX_NEUTRAL_BORDER
            pygame.draw.rect(self.screen, fill, rect, border_radius=6)
            pygame.draw.rect(self.screen, border, rect, width=2, border_radius=6)

        for bx, by in self.state.boxes_agent1:
            px = offset_x + bx * ts + 5
            py = offset_y + by * ts + 5
            rect = pygame.Rect(px, py, ts - 10, ts - 10)
            is_goal = (bx, by) in self.state.goals

            pygame.draw.rect(self.screen, COLOR_BOX_AGENT1, rect, border_radius=6)
            border_color = COLOR_BOX_DONE_BORDER if is_goal else COLOR_BOX_AGENT1_BORDER
            pygame.draw.rect(self.screen, border_color, rect, width=4 if is_goal else 2, border_radius=6)

            txt = self.font_bold.render("1", True, (255, 255, 255))
            self.screen.blit(txt, txt.get_rect(center=rect.center))

        for bx, by in self.state.boxes_agent2:
            px = offset_x + bx * ts + 5
            py = offset_y + by * ts + 5
            rect = pygame.Rect(px, py, ts - 10, ts - 10)
            is_goal = (bx, by) in self.state.goals

            pygame.draw.rect(self.screen, COLOR_BOX_AGENT2, rect, border_radius=6)
            border_color = COLOR_BOX_DONE_BORDER if is_goal else COLOR_BOX_AGENT2_BORDER
            pygame.draw.rect(self.screen, border_color, rect, width=4 if is_goal else 2, border_radius=6)

            txt = self.font_bold.render("2", True, (255, 255, 255))
            self.screen.blit(txt, txt.get_rect(center=rect.center))

        a1x, a1y = self.state.agent1_pos
        c1 = (offset_x + a1x * ts + ts // 2, offset_y + a1y * ts + ts // 2)
        pygame.draw.circle(self.screen, COLOR_AGENT1, c1, ts // 2 - 5)
        pygame.draw.circle(self.screen, COLOR_AGENT1_BORDER, c1, ts // 2 - 5, width=3)
        lbl1 = self.font_bold.render("A1", True, (255, 255, 255))
        self.screen.blit(lbl1, lbl1.get_rect(center=c1))

        a2x, a2y = self.state.agent2_pos
        c2 = (offset_x + a2x * ts + ts // 2, offset_y + a2y * ts + ts // 2)
        pygame.draw.circle(self.screen, COLOR_AGENT2, c2, ts // 2 - 5)
        pygame.draw.circle(self.screen, COLOR_AGENT2_BORDER, c2, ts // 2 - 5, width=3)
        lbl2 = self.font_bold.render("A2", True, (255, 255, 255))
        self.screen.blit(lbl2, lbl2.get_rect(center=c2))

    def _draw_panel(self):
        panel_rect = pygame.Rect(self.window_width - self.panel_width, 0, self.panel_width, self.window_height)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (panel_rect.left, 0), (panel_rect.left, self.window_height), 2)

        px = panel_rect.left + 20
        y = 20

        title = self.font_title.render("2-AGENT ARENA", True, COLOR_TEXT_GOLD)
        self.screen.blit(title, (px, y))
        y += 30

        sub = self.font_small.render(f"MAP: {os.path.basename(self.map_parser.filename)}", True, COLOR_TEXT_MUTED)
        self.screen.blit(sub, (px, y))
        y += 25

        score1 = sum(1 for b in self.state.boxes_agent1 if b in self.state.goals)
        score2 = sum(1 for b in self.state.boxes_agent2 if b in self.state.goals)

        pygame.draw.rect(self.screen, (20, 24, 30), (px, y, 290, 85), border_radius=8)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (px, y, 290, 85), width=2, border_radius=8)

        txt_a1 = self.font_bold.render(f"AGENT 1: {score1} Goals", True, COLOR_AGENT1)
        self.screen.blit(txt_a1, (px + 15, y + 15))
        txt_act1 = self.font_small.render(f"Action: {self.action1_last}", True, COLOR_TEXT_MUTED)
        self.screen.blit(txt_act1, (px + 15, y + 35))

        txt_a2 = self.font_bold.render(f"AGENT 2: {score2} Goals", True, COLOR_AGENT2)
        self.screen.blit(txt_a2, (px + 155, y + 15))
        txt_act2 = self.font_small.render(f"Action: {self.action2_last}", True, COLOR_TEXT_MUTED)
        self.screen.blit(txt_act2, (px + 155, y + 35))

        y += 105

        rem_steps = max(0, self.max_steps_n - self.current_step)
        step_txt = self.font_bold.render(f"Steps: {self.current_step} / {self.max_steps_n}", True, COLOR_TEXT_MAIN)
        self.screen.blit(step_txt, (px, y))
        y += 22

        rem_txt = self.font_main.render(f"Remaining: {rem_steps} steps", True, COLOR_TEXT_MUTED)
        self.screen.blit(rem_txt, (px, y))
        y += 30

        if self.is_game_over:
            win_surf = self.font_bold.render(self.winner_text, True, COLOR_TEXT_GOLD)
            self.screen.blit(win_surf, (px, y))
        else:
            status_str = "IN ACTION..." if self.is_running else "PAUSED (Space to continue)"
            status_surf = self.font_main.render(status_str, True, COLOR_AGENT1 if self.is_running else COLOR_TEXT_MUTED)
            self.screen.blit(status_surf, (px, y))

        y += 35
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (px, y), (panel_rect.right - 20, y), 1)
        y += 15

        label_ctrl = self.font_bold.render("Keybindings:", True, COLOR_TEXT_MAIN)
        self.screen.blit(label_ctrl, (px, y))
        y += 22

        guide = [
            "• [Space]      : Run / Pause",
            "• [Right Arrow]: Forward a step",
            "• [Left Arrow] : Backward a step",
            "• [Up / Down]  : Increase / Decrease (10-step)",
            "• [R]          : Reset",
            "• [Esc]        : Quit"
        ]
        for g in guide:
            surf = self.font_small.render(g, True, COLOR_TEXT_MUTED)
            self.screen.blit(surf, (px, y))
            y += 20

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()

            self.screen.fill(COLOR_BG)
            board_area_w = self.window_width - self.panel_width
            off_x = max(20, (board_area_w - self.board_width) // 2)
            off_y = max(20, (self.window_height - self.board_height) // 2)

            self._draw_board(off_x, off_y)
            self._draw_panel()

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


def main():
    map_arg = sys.argv[1] if len(sys.argv) > 1 else "maps/competitive_map.txt"
    steps_n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    app = CompetitiveGameGUI(map_arg, steps_n)
    app.run()


if __name__ == "__main__":
    main()
