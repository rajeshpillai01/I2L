import os
import time
import numpy as np
from core import I2LSystem, Executor


class I2LDiscoveryGame:
    def __init__(self):
        self.system = I2LSystem()
        self.executor = Executor()

        # 1. Game Dimensions
        self.width = 30
        self.height = 12

        # 2. Physics State
        self.ball_pos = [5.0, 5.0]  # Use floats for smooth math
        self.ball_vel = [1.0, 1.0]

        # 3. Load Discovered Laws from knowledge_graph.json
        # We define fallbacks in case the JSON is missing specific keys
        self.laws = {
            "move": self.system.memory.graph.get("learned_macros", {}).get("LINEAR_MOVEMENT", ["V_ADD"]),
            "bounce_x": self.system.memory.graph.get("learned_macros", {}).get("WALL_BOUNCE_X", ["V_REFLECT_X"]),
            "bounce_y": self.system.memory.graph.get("learned_macros", {}).get("WALL_BOUNCE_Y", ["V_REFLECT_Y"])
        }

    def update(self):
        """
        The Core Loop: Translates Symbolic Logic into Physical Motion.
        """
        # --- STEP A: MOVEMENT ---
        # We MUST pass a history list so V_ADD can find the 'Velocity' vector.
        # History = [Current Position, Current Velocity]
        context_history = [self.ball_pos, self.ball_vel]

        move_trace = self.executor.run_sequence(
            self.ball_pos,
            self.laws["move"],
            memory=context_history
        )

        if move_trace and len(move_trace) > 1:
            # Successfully moved using AI logic
            self.ball_pos = list(move_trace[-1])
        else:
            # Fallback: Manual Newtonian integration if logic fails
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]

        # --- STEP B: X-AXIS BOUNDARIES (Left/Right Walls) ---
        if self.ball_pos[0] <= 0 or self.ball_pos[0] >= self.width - 1:
            bounce_trace = self.executor.run_sequence(
                self.ball_vel,
                self.laws["bounce_x"],
                memory=self.system.memory
            )
            if bounce_trace:
                self.ball_vel = list(bounce_trace[-1])

            # Anti-Stuck: Reset position slightly inside the wall
            self.ball_pos[0] = 1.0 if self.ball_pos[0] <= 0 else float(self.width - 2)

        # --- STEP C: Y-AXIS BOUNDARIES (Ceiling/Floor) ---
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.height - 1:
            bounce_trace = self.executor.run_sequence(
                self.ball_vel,
                self.laws["bounce_y"],
                memory=self.system.memory
            )
            if bounce_trace:
                self.ball_vel = list(bounce_trace[-1])

            self.ball_pos[1] = 1.0 if self.ball_pos[1] <= 0 else float(self.height - 2)

    def render(self):
        """ASCII Renderer for the terminal."""
        os.system('cls' if os.name == 'nt' else 'clear')

        header = "--- I2L-v5 NEURO-SYMBOLIC GAME ENGINE ---"
        print(header.center(self.width + 2))

        print("+" + "-" * self.width + "+")
        for y in range(self.height):
            row = "|"
            for x in range(self.width):
                # Rounding position to nearest integer for display
                if [x, y] == [int(round(self.ball_pos[0])), int(round(self.ball_pos[1]))]:
                    row += "⚽"
                else:
                    row += " "
            print(row + "|")
        print("+" + "-" * self.width + "+")

        print(f" Pos: [{self.ball_pos[0]:.1f}, {self.ball_pos[1]:.1f}]")
        print(f" Vel: {self.ball_vel}")
        print(f" Active Laws: {self.laws['bounce_x']} | {self.laws['bounce_y']}")

    def play(self):
        print("🚀 Booting Engine...")
        time.sleep(1)
        try:
            while True:
                self.update()
                self.render()
                time.sleep(0.08)  # ~12 FPS for readability
        except KeyboardInterrupt:
            print("\nExiting Engine.")


if __name__ == "__main__":
    game = I2LDiscoveryGame()
    game.play()