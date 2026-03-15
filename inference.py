import torch
from core import Executor, Primitives, LogicMemory
from train_artwork import NeuralArtwork


def solve_with_artwork(input_val, target_val, available_atoms=None):
    # 1. Vocabulary Setup
    base_atoms = Primitives.get_all_atoms()
    neural_vocab_size = len(base_atoms) + 1

    mem = LogicMemory()
    macro_names = list(mem.graph.get("learned_macros", {}).keys())

    # The full search space (Primitives + Learned Macros)
    current_atoms = base_atoms + macro_names if available_atoms is None else available_atoms

    atom_map = {i + 1: atom for i, atom in enumerate(current_atoms)}
    atom_map[0] = None

    # 2. Neural Input Preparation
    if isinstance(input_val, list):
        flat_input = [input_val[0], input_val[1], target_val]
    else:
        flat_input = [0, input_val, target_val]

    # 3. Model Loading
    model = NeuralArtwork(input_dim=3, hidden_dim=64, output_dim=3, vocab_size=neural_vocab_size)
    try:
        model.load_state_dict(torch.load("logic_artwork.pth"))
        model.eval()
    except Exception as e:
        print(f"⚠️ Model load failed: {e}. Proceeding with Symbolic Search only.")

    # 4. Neural Inference & Confidence Capture
    with torch.no_grad():
        test_input = torch.tensor([flat_input], dtype=torch.float32)
        raw_output = model(test_input)

        avg_probs = torch.mean(raw_output.squeeze(0), dim=0).tolist()
        confidence_map = {"NONE": avg_probs[0]}
        for i, atom in enumerate(base_atoms):
            if i + 1 < len(avg_probs):
                confidence_map[atom] = avg_probs[i + 1]

        try:
            import streamlit as st
            if hasattr(st, "runtime") and st.runtime.exists():
                st.session_state.confidence_map = confidence_map
        except:
            pass

        probs, indices = torch.topk(raw_output, k=min(5, neural_vocab_size), dim=2)
        indices = indices.squeeze(0).tolist()

    executor = Executor()

    # --- THE QUICK WIN CHECK (Anti-Bloat Priority) ---
    for atom in current_atoms:
        trace = executor.run_sequence(input_val, [atom], memory=mem)
        if trace and trace[-1] == target_val:
            print(f"✅ INSTANT PRIMITIVE SUCCESS: {atom}")
            return [atom], trace

    # 5. Advanced Search Injection
    stateful_names = Primitives.get_important_atoms()
    important_indices = []

    for idx, atom in atom_map.items():
        if atom in macro_names or atom in stateful_names:
            important_indices.append(idx)

    # Combine Neural Guesses with Stateful/Macro overrides
    step_options = [sorted(list(set(indices[step] + important_indices)),
                           key=lambda x: 20 if atom_map.get(x) in macro_names else 1)
                    for step in range(3)]

    # 6. Combinatorial Search
    for i in step_options[0]:
        for j in step_options[1]:
            for k in step_options[2]:
                raw_chain = [atom_map.get(idx) for idx in [i, j, k]]
                logic_chain = [a for a in raw_chain if a is not None]
                if not logic_chain: continue

                trace = executor.run_sequence(input_val, logic_chain, memory=mem)

                if trace and trace[-1] == target_val:
                    used_macro = any(a in macro_names for a in logic_chain)
                    prefix = "🚀 MACRO-FIRST SUCCESS" if used_macro else "✅ PRIMITIVE SUCCESS"
                    print(f"{prefix}: {logic_chain}")
                    return logic_chain, trace

    return None, None