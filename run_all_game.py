import os
from core import I2LSystem

def run_pipeline():
    system = I2LSystem()
    print("\n🚀 I2L-v5: Initializing Ephemeral Workspace Pipeline...")

    # --- PART 1: CLASSIC LOGIC (The Foundation) ---
    print("\n[Task 1: Simple Square Plus One]")
    system.evolve_ai("SquarePlusOne", 5, 26)

    print("\n[Task 2: Recursive Discovery]")
    system.evolve_ai("Fibonacci_Alpha", [1, 1], 5)

    print("\n[Task 3: Deep Abstraction]")
    system.evolve_ai("Deep_Abstraction", 3, 101)

    # --- PART 2: GAME PHYSICS SYNTHESIS (The Million Dollar Addition) ---
    print("\n[Task 4: Physics Synthesis - Linear Movement]")
    # Input: Position [2,2], Context (Velocity): [1,1] -> Target: [3,3]
    # The AI must discover that V_ADD is the law governing this shift.
    res_move, _ = system.evolve_ai("LINEAR_MOVEMENT", [2, 2], [3, 3])
    if res_move: print(f"✅ Physics Hardened: {res_move}")

    print("\n[Task 5: Physics Synthesis - Wall Bounce X]")
    # Input: Velocity [1,1] -> Target: [-1,1]
    # The AI must discover V_REFLECT_X reverses the horizontal vector.
    res_bounce, _ = system.evolve_ai("WALL_BOUNCE_X", [1, 1], [-1, 1])
    if res_bounce: print(f"✅ Reflection Logic Hardened: {res_bounce}")

    print("\n[Task 6: Physics Synthesis - Wall Bounce Y]")
    # Input: Velocity [1,1] -> Target: [1,-1]
    res_bounce_y, _ = system.evolve_ai("WALL_BOUNCE_Y", [1, 1], [1, -1])

    # --- PART 3: REINFORCEMENT ---
    print("\n[Task 7: The Coincidence Trap]")
    system.evolve_ai("Ambiguous_Rule", 2, 4)
    system.evolve_ai("Ambiguous_Rule", 3, 6)

    print("\n--- 🏁 Pipeline Complete ---")
    print("📂 Long-Term Memory (Physics & Logic) updated in 'knowledge_graph.json'.")

if __name__ == "__main__":
    # Clear old memory to ensure a clean 'Learning' session
    if os.path.exists("knowledge_graph.json"):
        os.remove("knowledge_graph.json")
    run_pipeline()