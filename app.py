import streamlit as st
import time
import json
import os
import pandas as pd
import torch
from core import I2LSystem, Primitives, FitnessScorer
from train_artwork import NeuralArtwork

# --- UI Setup ---
st.set_page_config(page_title="I2L-v4: The Infinite Loop", layout="wide", page_icon="🧠")
st.title("🧠 I2L-v4: The Infinite Loop")
st.markdown("🔍 *Self-Correcting Neuro-Symbolic Discovery*")
st.markdown("---")

# --- Persistent Neural Brain Setup ---
if 'system' not in st.session_state:
    # 1. Setup Model Dimensions
    base_atoms = Primitives.get_all_atoms()
    neural_vocab_size = len(base_atoms) + 1

    # 2. Initialize and Load the Neural Brain
    model = NeuralArtwork(input_dim=3, hidden_dim=64, output_dim=3, vocab_size=neural_vocab_size)
    try:
        if os.path.exists("logic_artwork.pth"):
            model.load_state_dict(torch.load("logic_artwork.pth"))
            st.toast("🧠 Neural Brain Loaded & Persistent!")
        model.eval()
    except Exception as e:
        st.error(f"Failed to load brain weights: {e}")

    # 3. Initialize Orchestrator with the LIVE model
    st.session_state.system = I2LSystem(model=model)

# --- Sidebar: Logic Library (The Leaderboard) ---
with st.sidebar:
    st.header("📂 Logic Library")

    if st.button("🗑️ Purge All Memories", use_container_width=True):
        st.session_state.system.memory.graph = {"learned_macros": {}}
        with open(st.session_state.system.memory.file_path, 'w') as f:
            json.dump({"learned_macros": {}}, f, indent=4)
        st.success("Library Wiped!")
        st.rerun()

    st.divider()

    memory_data = st.session_state.system.memory.graph["learned_macros"]
    if memory_data:
        for task_name, candidates in memory_data.items():
            # Sort candidates by score so the UI shows the current preference
            # We use the length of the chain as a fallback for the UI sort
            sorted_cands = sorted(candidates, key=lambda c: FitnessScorer.score(c, memory_data), reverse=True)

            with st.expander(f"📚 {task_name} ({len(candidates)} versions)"):
                for i, chain in enumerate(sorted_cands):
                    score = FitnessScorer.score(chain, memory_data)
                    is_champ = (i == 0)

                    st.write(f"**{'🏆 Champion' if is_champ else f'⚖️ Candidate {i + 1}'}**")
                    st.caption(f"Fitness Score: {score:.1f}")
                    st.code(chain)
                    if is_champ:
                        st.markdown("---")
    else:
        st.info("No laws discovered yet.")

# --- Main Stage: Evolution ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🚀 Trigger Evolution")
    with st.container(border=True):
        task_name = st.text_input("Task Label", "Fibonacci Discovery")
        input_val = st.text_input("Input (e.g. 5 or [1, 1])", "[1, 1]")
        target_val = st.number_input("Target Value", value=21)
        evolve_btn = st.button("🧬 Evolve & Distill", type="primary", use_container_width=True)

    if evolve_btn:
        try:
            parsed_input = eval(input_val)
            with st.status("🧠 Neuro-Symbolic Loop Running...", expanded=True) as status:
                # ... [Keep your evolve_ai call] ...
                logic_chain, tournament_results = st.session_state.system.evolve_ai(task_name, parsed_input, target_val)

                # Store results in session state so they persist after the button click
                st.session_state.last_results = tournament_results
                st.session_state.last_logic = logic_chain
                st.session_state.last_input_type = "Recursive" if isinstance(parsed_input, list) else "Arithmetic"

                if logic_chain:
                    status.update(label="✅ Discovery Successful!", state="complete")
                    st.balloons()
                else:
                    status.update(label="❌ No Solution Found", state="error")
        except Exception as e:
            st.error(f"Input Error: {e}")

        # --- Persisted Results Display (Outside the button block) ---
    if 'last_results' in st.session_state and st.session_state.last_results:
        results = st.session_state.last_results

        st.divider()
        st.subheader("🏁 Logic Tournament Results")

        if results["survivors"]:
            # 1. Logic Selector
            options = [f"Rank {i + 1}: {s['logic']}" for i, s in enumerate(results["survivors"])]
            selected_idx = st.selectbox("🔍 Inspect Verified Theory:", range(len(options)),
                                        format_func=lambda x: options[x])

            survivor = results["survivors"][selected_idx]
            steps = survivor['logic']
            trace = survivor['trace']

            # 2. Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Steps", len(steps))
            m2.metric("Fitness", f"{survivor.get('score', 0.0):.1f}")
            m3.metric("Type", st.session_state.last_input_type)

            # 3. Execution Table
            st.write("#### ⚗️ Step-by-Step Execution")
            history = [{"Step": i + 1, "Input": trace[i], "Operation": steps[i], "Result": trace[i + 1]} for i in
                       range(len(steps))]
            st.table(pd.DataFrame(history))

            # 4. Visualization
            st.line_chart(trace)

        if results.get("debunked"):
            with st.expander("💀 Show Debunked Theories", expanded=False):
                for debunked in results.get("debunked"):
                    st.error(f"Logic: `{debunked['logic']}` → Got `{debunked['got']}`")

with col2:
    st.subheader("🎯 Neural Confidence")
    if 'confidence_map' in st.session_state:
        df_probs = pd.DataFrame([
            {"Atom": k, "Confidence": v}
            for k, v in st.session_state.confidence_map.items()
        ]).sort_values(by="Confidence", ascending=False)

        st.bar_chart(df_probs, x="Atom", y="Confidence", color="#00d4ff")
        st.caption("Live probability distribution. Watch this shift as the AI learns!")
    else:
        st.info("Run an evolution to visualize neural weights.")

# --- Footer ---
st.markdown("---")
st.caption("I2L-v4 Framework | Mode: Autonomous Learning | 'Occam's Razor' Enabled")