I2L-v1: Intuition-to-Logic Engine 🧠🔢
I2L-v1 is a Neuro-Symbolic framework that bridges the gap between neural "intuition" and symbolic "logic." It allows an AI to discover mathematical patterns, verify them through execution, and store them as reusable "Macros" in a persistent Knowledge Graph.

🚀 Key Features
Neural-Guided Search: Uses a PyTorch MLP to prune the logical search space.

Symbolic Execution: Validates all neural hypotheses in a safe, deterministic sandbox.

Hierarchical Memory: Successfully nests learned logic (Macros) inside new problems.

Recursive Fallback: Automatically switches from neural guessing to algorithmic iteration for sequences like Fibonacci.

📂 Project Structure
core.py: The orchestrator containing the I2LSystem, Executor, and LogicMemory.

train_artwork.py: Training script for the Neural Intuition model.

inference.py: Logic for search and macro-composition.

run_all.py: The main pipeline to generate data, train, and evolve logic.

knowledge_graph.json: The persistent memory of the AI.

🛠️ Installation & Usage
Clone the repository:

Bash
git clone https://github.com/your-username/I2L_v1.git
cd I2L_v1
Install Dependencies:

Bash
pip install torch pandas
Run the Engine:

Bash
python run_all.py
📈 Evolutionary Logic Trace
The engine follows a tiered learning path:

Primitive Learning: [SQUARE, ADD_ONE]

Abstraction: [Macro1, Macro1]

Composition: [Macro1, Macro2]

Recursive Discovery: Iterative application of SUM_PREV.
