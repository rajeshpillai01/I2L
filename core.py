import itertools
import inspect
import random
import csv
import json
import os
import torch




# --- 1. THE ATOMS (Logic Primitives) ---
class Primitives:
    """The Library of Atoms. These are the 'Roads' in our Neural Artwork."""

    @staticmethod
    def ADD_ONE(n,history=None): return n + 1

    @staticmethod
    def DOUBLE(n,history=None): return n * 2

    @staticmethod
    def SQUARE(n,history=None): return n ** 2

    @staticmethod
    def SUBTRACT_ONE(n,history=None): return n - 1

    @staticmethod
    def SUM_PREV(n, history):
        """STATEFUL: Sums the last two numbers."""
        if len(history) >= 2:
            return history[-1] + history[-2]
        return n

    @classmethod
    def get_all_atoms(cls):
        """Self-scanning: Automatically finds all uppercase methods."""
        return [name for name, func in inspect.getmembers(cls, predicate=inspect.isfunction)
                if name.isupper()]

    @classmethod
    def get_important_atoms(cls):
        """Generic Scanner: Finds any atom that requires 'history'."""
        important = []
        for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
            # If it takes more than 1 argument, it's stateful/important
            sig = inspect.signature(func)
            if len(sig.parameters) > 1:
                important.append(name)
        return important


# --- 2. THE SANDBOX (Symbolic Executor) ---
class Executor:
    def run_sequence(self, initial_data, instructions, memory=None):
        if instructions is None: return None
        current = [initial_data] if not isinstance(initial_data, list) else initial_data
        try:
            for instr in instructions:
                # If the instruction is in our Knowledge Graph, execute the sub-logic
                if memory and instr in memory.graph.get("learned_macros", {}):
                    macro_logic = memory.recall(instr)
                    # Recursive call: run the macro's steps starting at current value
                    current = self.run_sequence(current, macro_logic, memory=memory)

                else:
                    # Otherwise, it's a standard Primitive atom
                    func = getattr(Primitives, instr)
                    next_val = func(current[-1], current)
                    if next_val > 1000000 or next_val < -1000000:
                        return None
                    current.append(next_val)
            return current
        except Exception as e:
            return None


# --- 3. THE MEMORY (Knowledge Graph) ---
class LogicMemory:
    """The Storage: Saves successful logic chains as reusable macros."""

    def __init__(self, file_path="knowledge_graph.json"):
        self.file_path = file_path
        self.graph = self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {"learned_macros": {}}

    def store(self, task_name, logic_chain):
        self.graph["learned_macros"][task_name] = logic_chain
        with open(self.file_path, 'w') as f:
            json.dump(self.graph, f, indent=4)

    def recall(self, task_name):
        return self.graph["learned_macros"].get(task_name)


# --- 4. THE SCHOOL (Data Generator) ---
class LogicDataGenerator:
    """The Generator: Now with Balanced Logic Sampling."""

    def __init__(self):
        self.atoms = Primitives.get_all_atoms()
        self.executor = Executor()

    def generate_training_set(self, samples=2000, max_depth=3):
        dataset = []

        # 1. Standard Math Logic (90% of data)
        # Change the limit here to samples * 0.9
        while len(dataset) < int(samples * 0.9):
            history_val = 0
            start_val = random.randint(1, 30)
            depth = random.randint(1, max_depth)
            logic_chain = [random.choice(self.atoms) for _ in range(depth)]

            trace = self.executor.run_sequence(start_val, logic_chain)

            if trace and trace[-1] < 1000000:
                dataset.append({
                    "context": history_val,
                    "input": start_val,
                    "target": trace[-1],
                    "logic_chain": ",".join(logic_chain)
                })

        # 2. SEED: Stateful Logic (Remaining 10%)
        # This loop should be OUTSIDE the first one
        while len(dataset) < samples:
            a, b = random.randint(1, 20), random.randint(1, 20)
            dataset.append({
                "context": a,
                "input": b,
                "target": a + b,
                "logic_chain": "SUM_PREV"
            })

        # Important: Shuffle so the model doesn't just see SUM_PREV at the very end
        random.shuffle(dataset)
        return dataset

    def export_to_csv(self, filename="neural_artwork_data.csv", samples=2000):
        data = self.generate_training_set(samples)
        if not data: return
        keys = data[0].keys()
        with open(filename, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
        print(f"✅ Balanced Data exported to {filename}")

# --- 5. THE ORCHESTRATOR (I2L System) ---
class I2LSystem:
    """The CNS: Connects Memory, Intuition, and Execution."""

    def __init__(self):
        self.executor = Executor()
        self.memory = LogicMemory()
        self.generator = LogicDataGenerator()
        self.atoms = Primitives.get_all_atoms()

    def evolve_ai(self, task_label, input_val, target_val):
        from inference import solve_with_artwork

        # 1. MEMORY FIRST
        existing_logic = self.memory.recall(task_label)
        if existing_logic:
            print(f"🧠 Memory Hit! Using learned macro: {existing_logic}")
            return existing_logic, ["Restored from memory"]

        # 2. SEQUENCE REPEATER (The Fibonacci Fix)
        # If input is a list, try repeating every atom we have to see if a pattern emerges.
        if isinstance(input_val, list):
            for atom_name in self.atoms:
                temp_trace = list(input_val)
                chain = []
                for _ in range(20):
                    func = getattr(Primitives, atom_name)
                    try:
                        next_val = func(temp_trace[-1], temp_trace)
                    except TypeError:
                        next_val = func(temp_trace[-1])

                    temp_trace.append(next_val)
                    chain.append(atom_name)
                    if next_val == target_val:
                        print(f"🚀 PATTERN SUCCESS: Found sequence using {atom_name}!")
                        self.memory.store(task_label, chain)
                        return chain, temp_trace
                    if next_val > target_val: break

        # 3. NEURAL SEARCH (The Math Fix)
        # If it's not a repeating sequence, use the Neural Artwork's intuition.
        learned_macro_names = list(self.memory.graph["learned_macros"].keys())
        current_atoms = self.atoms + learned_macro_names
        logic_chain, trace = solve_with_artwork(input_val, target_val, available_atoms=current_atoms)

        if trace and trace[-1] == target_val:
            print(f"✨ New Logic Learned! Saving to Knowledge Graph.")
            self.memory.store(task_label, logic_chain)
            return logic_chain, trace

        print(f"❌ Evolution failed for {task_label}")
        return None, None # Returning two Nones prevents the "non-iterable" error
