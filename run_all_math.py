import os
from core import I2LSystem

def run_pipeline():
    system = I2LSystem()
    print("\n🚀 I2L-v5: Initializing Ephemeral Workspace Pipeline...")

    # TASK 1: Base Synthesis
    print("\n[Task 1: Simple Square Plus One]")
    # 5 -> 26 via [SQUARE, ADD_ONE]
    system.evolve_ai("SquarePlusOne", 5, 26)

    # TASK 2: Recursive Discovery (Calibrated to 3 steps)
    print("\n[Task 2: The Fibonacci Baseline]")
    # [1, 1] -> 2 -> 3 -> 5. Logic: [SUM_PREV, SUM_PREV, SUM_PREV]
    res_fib, _ = system.evolve_ai("Fibonacci_Alpha", [1, 1], 5)
    if res_fib: print(f"✅ Fibonacci Alpha Hardened: {res_fib}")

    # TASK 3: Contextual Recall (The Wide Trigger)
    print("\n[Task 4: Wide Trigger Test - Fibonacci Beta]")
    # [3, 3] -> 6 -> 9 -> 15. Should trigger instant recall of Task 2
    res_beta, _ = system.evolve_ai("Fibonacci_Beta", [3, 3], 15)
    if res_beta: print(f"💡 RECALL SUCCESS: Applied {res_beta}")

    # TASK 4: Deep Abstraction (Macro Nesting)
    print("\n[Task 5: Deep Abstraction]")
    # 3 -> SquarePlusOne -> 10 -> SquarePlusOne -> 101
    # Logic: [SquarePlusOne, SquarePlusOne]
    res_deep, _ = system.evolve_ai("Deep_Abstraction", 3, 101)
    if res_deep: print(f"🔥 NESTED SUCCESS: {res_deep}")

    # TASK 5: The Coincidence Trap
    print("\n[Task 6: The Coincidence Trap]")
    # Input 2, Target 4. Must prefer DOUBLE over ADD_ONE repeats.
    res_stress, _ = system.evolve_ai("Ambiguous_Rule", 2, 4)
    if res_stress: print(f"🏆 V5 VERIFIED: {res_stress}")

    print("\n--- 🏁 Pipeline Complete ---")
    print("📂 Long-Term Memory updated in 'knowledge_graph.json'.")

if __name__ == "__main__":
    if os.path.exists("knowledge_graph.json"): os.remove("knowledge_graph.json")
    run_pipeline()