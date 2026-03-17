import torch
import itertools
import numpy as np
from core import Executor, Primitives, LogicMemory, FitnessScorer, LogicValidator


def solve_with_artwork(input_val, target_val, temp_model, available_atoms=None):
    # 1. Vocabulary & Memory Setup
    base_atoms = Primitives.get_all_atoms()
    mem = LogicMemory()
    all_macros = mem.graph.get("learned_macros", {})
    macro_names = list(all_macros.keys())

    # Union of Primitives and Learned Knowledge
    current_atoms = sorted(list(set(base_atoms + macro_names)))
    atom_map = {i + 1: atom for i, atom in enumerate(current_atoms)}
    atom_map[0] = None

    # 2. Neural Hunch (Ephemeral Inference)
    indices = []
    if temp_model is not None:
        temp_model.eval()
        with torch.no_grad():
            # --- THE FIX: Linearized Flattener ---
            # Converts [2,2] or 5 into a clean list of floats for the Tensor
            def flatten(val):
                if isinstance(val, (list, tuple, np.ndarray)):
                    return [float(x) for x in val]
                return [float(val)]

            # Construct a fixed-width feature vector
            # We pad/clip to ensure the model always sees the expected input size
            raw_features = flatten(input_val) + flatten(target_val)

            # Ensure we have at least 4 features (2 for in, 2 for target) + 1 context
            while len(raw_features) < 4:
                raw_features.append(0.0)

            flat_in = [0.0] + raw_features[:4]  # [Context, In_X, In_Y, Target_X, Target_Y]

            try:
                test_input = torch.tensor([flat_in], dtype=torch.float32)
                raw_output = temp_model(test_input)
                _, top_indices = torch.topk(raw_output, k=min(10, len(current_atoms) + 1), dim=2)
                indices = top_indices.squeeze(0).tolist()
            except Exception as e:
                # print(f"Neural Hunch Warning: {e}")
                indices = [[0] * 5 for _ in range(3)]

    if not indices:
        indices = [[0] * 5 for _ in range(3)]

    # 3. Search Space Prioritization (Dynamic Discovery)
    # This now calls the Introspective method from core.py
    stateful_names = Primitives.get_important_atoms()
    is_seq = isinstance(input_val, (list, tuple, np.ndarray))

    # Identify indices of Macros and Stateful Atoms (like V_ADD)
    important_indices = [idx for idx, atom in atom_map.items()
                         if atom in macro_names or (is_seq and atom in stateful_names)]

    step_options = []
    for step in range(3):
        # Merge Neural guesses with Important atoms
        options = list(set(indices[step] + important_indices + [0]))

        def get_prio(idx):
            atom = atom_map.get(idx)
            if atom in macro_names: return 3
            if atom in stateful_names: return 2
            return 1 if idx != 0 else 0

        step_options.append(sorted(options, key=get_prio, reverse=True))

    # 4. Combinatorial Execution
    executor = Executor()
    podium = []

    # We increase search depth slightly for Physics tasks
    for combo in itertools.product(*step_options):
        logic_chain = [atom_map[idx] for idx in combo if idx != 0 and atom_map.get(idx)]
        if not logic_chain or any(p["logic"] == logic_chain for p in podium): continue

        # The Executor now handles the 'history' (velocity) via parametric injection
        trace = executor.run_sequence(input_val, logic_chain, memory=mem)

        if trace and trace[-1] == target_val:
            # Verify the logic holds across different randomized seeds
            if LogicValidator.verify(executor, logic_chain, input_val, memory=mem):
                score = FitnessScorer.score(logic_chain, input_val, all_macros)
                podium.append({"logic": logic_chain, "trace": trace, "score": score})
                print(f"📍 Found Verified Candidate: {logic_chain}")

        if len(podium) >= 3: break

    if podium:
        podium.sort(key=lambda x: x["score"], reverse=True)
        return podium
    return None