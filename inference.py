import torch
from core import Executor, Primitives
from train_artwork import NeuralArtwork


def solve_with_artwork(input_val, target_val, available_atoms=None):
    from core import Primitives, Executor

    # 1. The 'Neural' vocabulary (must match what was used in train_artwork.py)
    base_atoms = Primitives.get_all_atoms()
    neural_vocab_size = len(base_atoms) + 1

    # 2. The 'Search' vocabulary (includes Macros)
    if available_atoms is None:
        available_atoms = base_atoms

    # This map stays dynamic for the search phase
    atom_map = {i + 1: atom for i, atom in enumerate(available_atoms)}
    atom_map[0] = None


    if isinstance(input_val, list):
        flat_input = [input_val[0], input_val[1], target_val]
    else:
        flat_input = [0, input_val, target_val]

    # Load the model using the NEURAL vocab size
    model = NeuralArtwork(input_dim=3, hidden_dim=64, output_dim=3, vocab_size=neural_vocab_size)

    # Now it will load perfectly because it matches the .pth file!
    model.load_state_dict(torch.load("logic_artwork.pth"))
    model.eval()


    with torch.no_grad():
        test_input = torch.tensor([flat_input], dtype=torch.float32)
        raw_output = model(test_input)
        # 1. Get Neural Guesses
        probs, indices = torch.topk(raw_output, k=5, dim=2)
        indices = indices.squeeze(0).tolist()

    executor = Executor()

    # We'll need a memory instance to pass to the executor for macros
    from core import LogicMemory
    mem = LogicMemory()

    # 2. INJECTION: Add Macro Indices to the search space
    scored_candidates = []
    macro_names = list(mem.graph.get("learned_macros", {}).keys())
    stateful_names = Primitives.get_important_atoms()
    macro_indices = []  # This needs to be populated!

    important_indices = []

    for idx, atom in atom_map.items():
        if atom in macro_names or atom in stateful_names:
            important_indices.append(idx)
            scored_candidates.append((idx, 5))
        elif atom is not None:
            scored_candidates.append((idx, 10))

    # --- UN-INDENT THESE ---
    # Now we build the options ONCE using the full important_indices list
    step_options = [sorted(list(set(indices[step] + important_indices)),
                           key=lambda x: 5 if atom_map.get(x) in macro_names else 10)
                    for step in range(3)]

    for i in step_options[0]:
        for j in step_options[1]:
            for k in step_options[2]:
                raw_chain = [atom_map.get(idx) for idx in [i, j, k]]
                logic_chain = [a for a in raw_chain if a is not None]
                if not logic_chain: continue

                trace = executor.run_sequence(input_val, logic_chain, memory=mem)

                if trace and trace[-1] == target_val:
                    # Determine if we used a macro or just primitives
                    used_macro = any(a in macro_names for a in logic_chain)
                    prefix = "🚀 MACRO-FIRST SUCCESS" if used_macro else "✅ PRIMITIVE SUCCESS"
                    print(f"{prefix}: {logic_chain}")
                    return logic_chain, trace

    return None, None


# --- TEST IT OUT ---
if __name__ == "__main__":
    I, T = 5, 26
    instructions, result_trace = solve_with_artwork(I, T)

    print(f"\n🎯 Input: {I} | Target: {T}")
    print(f"🤖 Neural Pathfinding Suggestion: {' -> '.join(instructions)}")

    if result_trace and result_trace[-1] == T:
        print(f"✅ VERIFIED: The logic is mathematically sound.")
        print(f"📈 Execution Trace: {result_trace}")
    else:
        print(f"❌ FAILED: The Neural Artwork suggested an invalid path.")