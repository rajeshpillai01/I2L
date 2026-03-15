import os
import torch
from core import I2LSystem, Primitives
from train_artwork import train


def run_pipeline():
    # 1. INITIALIZE SYSTEM
    # This loads the Memory (JSON) and the Executor
    system = I2LSystem()

    # 2. GENERATE DATA (The Textbook)
    # We generate 2000 samples of logic patterns
    if not os.path.exists("neural_artwork_data.csv"):
        print("\n🔨 Step 1: Generating Logic Data...")
        system.generator.export_to_csv(samples=2000)
    else:
        print("\n✅ Step 1: Logic Data already exists. Skipping.")

    # 3. TRAIN MODEL (The Learning)
    # This creates 'logic_artwork.pth' (The AI's Intuition)
    if not os.path.exists("logic_artwork.pth"):
        print("\n🧠 Step 2: Training Neural Artwork (this may take a minute)...")
        train()
    else:
        print("\n✅ Step 2: Neural Artwork already trained. Skipping.")

    print("\n🚀 Step 3: Running Inference & Evolution...")

    # --- TASK 1: BASIC ARITHMETIC ---
    print("\n[Task 1: Simple Square Plus One]")
    res1 = system.evolve_ai("Simple Square Plus One", 5, 26)
    if res1 and res1[0]:
        print(f"✅ Success! Trace: {res1[1]}")

    # --- TASK 2: GENERALIZATION ---
    print("\n[Task 2: Double then Subtract]")
    res2 = system.evolve_ai("Double then Subtract", 10, 19)
    if res2 and res2[0]:
        print(f"✅ Success! Trace: {res2[1]}")

    # --- TASK 3: DEEP ABSTRACTION ---
    print("\n[Task 3: Using Learned Macros]")
    res3 = system.evolve_ai("Deep Abstraction", 3, 101)
    if res3 and res3[0]:
        print(f"✅ Success! Trace: {res3[1]}")

    # --- TASK 4: COMPOSITIONAL LOGIC ---
    print("\n[Task 4: Cross-Macro Composition]")
    res4 = system.evolve_ai("Compositional Logic", 2, 9)
    if res4 and res4[0]:
        print(f"🔥 TOTAL SUCCESS! Trace: {res4[1]}")
        print(f"   Logic used: {res4[0]}")

    # --- TASK 5: STATEFUL SEQUENCES (FIBONACCI) ---
    print("\n[Task 5: Fibonacci Recursive Test]")
    # We use 21 to ensure the 3-step neural search fails and forces
    # the RECURSIVE loop to take over!
    target_fib = 21
    res_fib = system.evolve_ai("Fibonacci Sequence", [1, 1], target_fib)

    if res_fib and res_fib[0]:
        print(f"🧬 FIBONACCI SUCCESS! Target {target_fib} reached.")
        print(f"   Trace: {res_fib[1]}")
        print(f"   Logic: {res_fib[0]}")
    else:
        print("❌ Fibonacci failed. Ensure SUM_PREV has the \"\"\"STATEFUL\"\"\" docstring.")

    print("\n--- Pipeline Complete ---")
    print("📂 Check 'knowledge_graph.json' to see your AI's new memories!")

    # v2 STRESS TEST: The Coincidence Trap
    # Input: 2, Target: 4.
    # Potential rules: SQUARE or DOUBLE.
    # The Validator must check if the rule works for Input: 3.
    print("\n[v2 Stress Test: The Coincidence Trap]")
    res_stress = system.evolve_ai("Ambiguous Rule", 2, 4)

    if res_stress and res_stress[0]:
        print(f"🏆 V2 SUCCESS: The system found and VERIFIED the rule: {res_stress[0]}")
    else:
        print("❌ V2 failure: The system couldn't verify a universal rule.")

    # Starting with [1, 2] instead of [1, 1]
    # and a target that isn't just +1 away
    logic_fib, trace_fib = system.evolve_ai("True Fibonacci", [1, 2], 8)
    print(f"   Trace: {trace_fib}")
    print(f"   Logic: {logic_fib}")

if __name__ == "__main__":
    run_pipeline()