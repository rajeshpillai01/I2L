import streamlit as st
import time
import json
import os
import pandas as pd
from core import I2LSystem, Primitives, FitnessScorer

# --- UI Setup ---
st.set_page_config(page_title="I2L-v3: Neuro-Symbolic Engine", layout="wide", page_icon="🧠")
st.title("🧠 I2L-v3: Intuition-to-Logic Dashboard")
st.markdown("🔍 *Bridging the gap between Neural Intuition and Symbolic Rigor*")
st.markdown("---")

# --- Session State Management ---
if 'system' not in st.session_state:
    st.session_state.system = I2LSystem()

# --- Sidebar: Knowledge Graph & Management ---
with st.sidebar:
    st.header("📂 Knowledge Graph")

    if st.button("🗑️ Purge All Memories", use_container_width=True):
        st.session_state.system.memory.graph = {"learned_macros": {}}
        # Manual save to clear the file
        with open(st.session_state.system.memory.file_path, 'w') as f:
            json.dump({"learned_macros": {}}, f, indent=4)
        st.success("Memory Wiped!")
        st.rerun()

    st.divider()

    memory_data = st.session_state.system.memory.graph["learned_macros"]
    if memory_data:
        for macro, logic in memory_data.items():
            # Calculate a quick 'Efficiency' score for the UI
            efficiency = FitnessScorer.score(logic, memory_data)
            with st.expander(f"✨ {macro} (Score: {efficiency:.1f})"):
                st.code(f"Chain: {logic}")
    else:
        st.info("No memories learned yet. Run an evolution to start.")

# --- Main Stage: Evolution ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🚀 Trigger New Evolution")
    with st.container(border=True):
        task_name = st.text_input("Task Label", "Fibonacci Discovery")
        input_val = st.text_input("Input (e.g. 5 or [1, 1])", "[1, 1]")
        target_val = st.number_input("Target Value", value=21)

        evolve_btn = st.button("🧬 Evolve Logic", type="primary", use_container_width=True)

    if evolve_btn:
        try:
            parsed_input = eval(input_val)

            with st.status("🧠 Processing I2L Loop...", expanded=True) as status:
                st.write("1. Scanning Neural Artwork for Intuition...")
                time.sleep(0.4)
                st.write("2. Running Competitive Symbolic Sandbox...")

                # Execute I2L Core
                logic_chain, trace = st.session_state.system.evolve_ai(task_name, parsed_input, target_val)

                if logic_chain:
                    status.update(label="✅ Evolution Successful!", state="complete")
                else:
                    status.update(label="❌ Evolution Failed", state="error")

            if logic_chain:
                st.balloons()

                # Calculate V3 Metrics
                macros = st.session_state.system.memory.graph["learned_macros"]
                f_score = FitnessScorer.score(logic_chain, macros)

                st.success(f"**Winner:** {logic_chain}")

                # Metric display
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Steps", len(logic_chain))
                m_col2.metric("Fitness", f"{f_score:.1f}")
                m_col3.metric("Type", "Recursive" if isinstance(parsed_input, list) else "Arithmetic")

                # Visualizing the Trace
                st.write("**Growth Curve:**")
                st.line_chart(trace)

        except Exception as e:
            st.error(f"Input Error: {e}")

with col2:
    st.subheader("🎯 Neural Confidence Map")
    if 'confidence_map' in st.session_state:
        # Create a dataframe for the bar chart
        df_probs = pd.DataFrame([
            {"Atom": k, "Confidence": v}
            for k, v in st.session_state.confidence_map.items()
        ]).sort_values(by="Confidence", ascending=False)

        # Display the chart
        st.bar_chart(df_probs, x="Atom", y="Confidence", color="#00d4ff")
        st.caption("Average probability assigned to each atom during the neural search phase.")
    else:
        st.info("The neural distribution will appear here after evolution.")

# --- Footer ---
st.markdown("---")
st.caption("I2L-v3 Framework | Architecture: Neuro-Symbolic | Mode: Professional Dashboard")