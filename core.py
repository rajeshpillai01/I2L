import itertools
import inspect
import random
import json
import os
import torch
import torch.nn as nn
import numpy as np


# --- 1. THE ATOMS (Universal Interface) ---
class Primitives:
    # --- MATH ---
    @staticmethod
    def SQUARE(n, history=None):
        return n ** 2 if isinstance(n, (int, float)) else n

    @staticmethod
    def ADD_ONE(n, history=None):
        return n + 1 if isinstance(n, (int, float)) else n

    @staticmethod
    def SUM_PREV(n, history=None):
        if history and isinstance(history, list) and len(history) >= 2:
            # Handle both scalar and vector summation
            if isinstance(history[-1], (int, float)) and isinstance(history[-2], (int, float)):
                return history[-1] + history[-2]
            elif isinstance(history[-1], list) and isinstance(history[-2], list):
                return [history[-1][0] + history[-2][0], history[-1][1] + history[-2][1]]
        return n

    # --- PHYSICS ---
    @staticmethod
    def V_ADD(n, history=None):
        """
        The Semantic Guard: Returns None if a partner vector is missing.
        This prevents V_ADD from polluting non-vector logic chains.
        """
        if isinstance(n, list) and history:
            for item in reversed(history):
                if isinstance(item, list) and item != n and len(item) == len(n):
                    return [n[0] + item[0], n[1] + item[1]]
        return None

    @staticmethod
    def V_REFLECT_X(n, history=None):
        return [-n[0], n[1]] if isinstance(n, list) else n

    @staticmethod
    def V_REFLECT_Y(n, history=None):
        return [n[0], -n[1]] if isinstance(n, list) else n

    # --- DYNAMIC DISCOVERY ---
    @classmethod
    def get_all_atoms(cls):
        return [name for name, func in inspect.getmembers(cls, predicate=inspect.isfunction) if name.isupper()]

    @classmethod
    def get_important_atoms(cls):
        important = []
        for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.isupper(): continue
            params = inspect.signature(func).parameters
            if "history" in params or "context" in params:
                important.append(name)
        return important


# --- 2. THE FITNESS SCORER ---
class FitnessScorer:
    @staticmethod
    def score(logic_chain, input_val, learned_macros=None):
        if not logic_chain: return 0
        score = 100 / len(logic_chain)

        is_vector = isinstance(input_val, list)
        for atom in logic_chain:
            # Type Alignment Bonuses
            if is_vector and atom.startswith("V_"): score += 25
            if not is_vector and not atom.startswith("V_"): score += 15
            if learned_macros and atom in learned_macros: score += 30

        # Repetition Penalty
        if len(logic_chain) > 2 and len(set(logic_chain)) == 1:
            score *= 0.2
        return round(score, 2)


# --- 3. THE LOGIC VALIDATOR ---
class LogicValidator:
    @staticmethod
    def verify(executor, logic, input_val, memory=None):
        """
        Universal Validator: Checks for functional consistency and type stability.
        """
        # Test 1: Original Task
        trace1 = executor.run_sequence(input_val, logic, memory=memory)
        if not trace1 or trace1[-1] is None: return False

        # Test 2: Shifted Stress Test (Generalization)
        if isinstance(input_val, (int, float)):
            test_val = input_val + 5.0
        else:
            test_val = [input_val[0] + 5.0, input_val[1] + 5.0]

        trace2 = executor.run_sequence(test_val, logic, memory=memory)
        if not trace2 or trace2[-1] is None: return False

        # Type Stability: Ensure logic doesn't morph data types
        if type(trace1[-1]) != type(input_val): return False

        return True


# --- 4. THE LONG-TERM MEMORY ---
class LogicMemory:
    def __init__(self, file_path="knowledge_graph.json"):
        self.file_path = file_path
        self.graph = self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"learned_macros": {}}

    def store(self, task_name, logic_chain):
        if "learned_macros" not in self.graph: self.graph["learned_macros"] = {}
        self.graph["learned_macros"][task_name] = logic_chain
        with open(self.file_path, 'w') as f: json.dump(self.graph, f, indent=4)

    def contextual_recall(self, input_val, target_val, executor):
        for task, chain in self.graph.get("learned_macros", {}).items():
            test_trace = executor.run_sequence(input_val, chain, memory=self)
            if test_trace and test_trace[-1] == target_val:
                return chain, task
        return None, None


# --- 5. THE EXECUTION ENGINE ---
class Executor:
    def run_sequence(self, initial, instructions, memory=None):
        current = initial if not isinstance(initial, list) else list(initial)
        history = [current]

        try:
            for instr in instructions:
                # Resolve Macro
                if memory:
                    graph = memory.graph if hasattr(memory, 'graph') else memory
                    macro = graph.get("learned_macros", {}).get(instr)
                    if macro:
                        res_trace = self.run_sequence(current, macro, memory)
                        if res_trace is None: return None
                        current = res_trace[-1]
                        history.append(current)
                        continue

                # Resolve Primitive
                func = getattr(Primitives, instr, None)
                if not func: return None

                next_val = func(current, history)
                if next_val is None: return None

                current = next_val
                history.append(current)

            return history
        except:
            return None


# --- 6. THE V5 ORCHESTRATOR ---
class I2LSystem:
    def __init__(self):
        self.executor = Executor()
        self.memory = LogicMemory()
        self.atoms = Primitives.get_all_atoms()

    def evolve_ai(self, task_label, input_val, target_val):
        existing_logic, _ = self.memory.contextual_recall(input_val, target_val, self.executor)
        if existing_logic: return existing_logic, {}

        from inference import solve_with_artwork
        from train_artwork import NeuralArtwork

        temp_model = NeuralArtwork(input_dim=5, hidden_dim=64, output_dim=3, vocab_size=len(self.atoms) + 1)
        podium = solve_with_artwork(input_val, target_val, temp_model, self.atoms)
        del temp_model

        if podium:
            best = podium[0]
            self.memory.store(task_label, best['logic'])
            return best['logic'], {"survivors": podium}

        print(f"⚠️ Task '{task_label}' failed to find valid logic.")
        return None, None