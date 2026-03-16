import itertools
import inspect
import random
import csv
import json
import os
import torch
import torch.optim as optim
import torch.nn as nn


# --- 1. THE ATOMS (Logic Primitives) ---
class Primitives:
    """The Library of Atoms. Unified signature for all: (n, history=None)."""

    @staticmethod
    def ADD_ONE(n, history=None):
        return n + 1

    @staticmethod
    def DOUBLE(n, history=None):
        return n * 2

    @staticmethod
    def SQUARE(n, history=None):
        return n ** 2

    @staticmethod
    def SUBTRACT_ONE(n, history=None):
        return n - 1

    @staticmethod
    def SUM_PREV(n, history=None):
        """STATEFUL: Sums the last two numbers in the history."""
        if history is not None and isinstance(history, list) and len(history) >= 2:
            return history[-1] + history[-2]
        return n

    @classmethod
    def get_all_atoms(cls):
        return [name for name, func in inspect.getmembers(cls, predicate=inspect.isfunction)
                if name.isupper()]

    @classmethod
    def get_important_atoms(cls):
        important = []
        for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
            sig = inspect.signature(func)
            if len(sig.parameters) > 1:
                important.append(name)
        return important


# --- 2. THE SANDBOX (Symbolic Executor) ---
class Executor:
    def run_sequence(self, initial_data, instructions, memory=None):
        """Executes a chain of instructions. Handles recursive atoms via history."""
        if not instructions or not isinstance(instructions, list):
            return None

        # Ensure we work with a list copy to avoid mutating the original input
        current = [initial_data] if not isinstance(initial_data, list) else list(initial_data)

        try:
            for instr in instructions:
                # Resolve Macros (Tasks learned previously)
                if memory and instr in memory.graph.get("learned_macros", {}):
                    candidates = memory.recall(instr)
                    macro_logic = candidates[0] if isinstance(candidates, list) else candidates
                    current = self.run_sequence(current, macro_logic, memory=memory)
                else:
                    # Execute Primitives
                    func = getattr(Primitives, instr)
                    # Always pass the current tail and the full history for stateful atoms
                    next_val = func(current[-1], current)
                    current.append(next_val)
            return current
        except Exception:
            return None


# --- 3. THE MEMORY (Knowledge Graph) ---
class LogicMemory:
    def __init__(self, file_path="knowledge_graph.json"):
        self.file_path = file_path
        self.graph = self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {"learned_macros": {}}

    def store(self, task_name, logic_chain):
        if task_name not in self.graph["learned_macros"]:
            self.graph["learned_macros"][task_name] = []
        if logic_chain not in self.graph["learned_macros"][task_name]:
            self.graph["learned_macros"][task_name].append(logic_chain)
        with open(self.file_path, 'w') as f:
            json.dump(self.graph, f, indent=4)

    def recall(self, task_name):
        return self.graph["learned_macros"].get(task_name)


# --- 4. THE SCHOOL (Data Generator) ---
class LogicDataGenerator:
    def __init__(self):
        self.atoms = Primitives.get_all_atoms()
        self.executor = Executor()

    def generate_training_set(self, samples=2000, max_depth=3):
        dataset = []
        # Standard logic (90%)
        while len(dataset) < int(samples * 0.9):
            start_val = random.randint(1, 30)
            depth = random.randint(1, max_depth)
            logic_chain = [random.choice(self.atoms) for _ in range(depth)]
            trace = self.executor.run_sequence(start_val, logic_chain)
            if trace and trace[-1] < 1000000:
                dataset.append({
                    "context": 0, "input": start_val, "target": trace[-1],
                    "logic_chain": ",".join(logic_chain)
                })
        # Seed logic (10% - specifically SUM_PREV)
        while len(dataset) < samples:
            a, b = random.randint(1, 20), random.randint(1, 20)
            dataset.append({
                "context": a, "input": b, "target": a + b, "logic_chain": "SUM_PREV"
            })
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


# --- 5. LOGIC HELPERS (Scoring, Compression, Validation) ---
class LogicCompressor:
    def compress(self, logic_chain):
        if not logic_chain or len(logic_chain) < 2: return logic_chain
        compressed = []
        i = 0
        while i < len(logic_chain):
            count = 1
            while i + count < len(logic_chain) and logic_chain[i] == logic_chain[i + count]:
                count += 1
            if count > 2:
                compressed.append(f"{count}x_{logic_chain[i]}")
            else:
                compressed.extend([logic_chain[i]] * count)
            i += count
        return compressed


class FitnessScorer:
    @staticmethod
    def score(logic_chain, learned_macros=None):
        if not logic_chain: return 0
        # Efficiency penalty
        score = 100 / len(logic_chain)
        # Macro bloat penalty
        if learned_macros:
            macro_count = sum(1 for atom in logic_chain if atom in learned_macros)
            score -= (macro_count * 40)
        # Diversity bonus
        return score + (len(set(logic_chain)) * 5)


class LogicValidator:
    def __init__(self, executor):
        self.executor = executor

    def verify(self, logic_chain, original_input, memory):
        # Universal test: check if it works for other random numbers
        test_inputs = [random.randint(1, 20) for _ in range(3)]
        for test_in in test_inputs:
            actual_trace = self.executor.run_sequence(test_in, logic_chain, memory=memory)
            if actual_trace is None: return False
        return True


# --- 6. NEURAL HANDLERS ---
class OnlineTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def distill(self, input_val, target_val, logic_chain, base_atoms):
        if not self.model: return
        self.model.train()
        first_atom = logic_chain[0]
        try:
            target_idx = base_atoms.index(first_atom) + 1
            flat_in = [input_val[0], input_val[1], target_val] if isinstance(input_val, list) else [0, input_val,
                                                                                                    target_val]
            input_tensor = torch.tensor([flat_in], dtype=torch.float32)
            label_tensor = torch.tensor([target_idx], dtype=torch.long)
            self.optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = self.criterion(output[:, 0, :], label_tensor)
            loss.backward()
            self.optimizer.step()
        except:
            pass


# --- 7. THE ORCHESTRATOR (I2L System) ---
class I2LSystem:
    def __init__(self, model=None):
        self.executor = Executor()
        self.memory = LogicMemory()
        self.generator = LogicDataGenerator()
        self.atoms = Primitives.get_all_atoms()
        self.model = model
        self.trainer = OnlineTrainer(self.model) if model else None

    def evolve_ai(self, task_label, input_val, target_val):
        # Local import to prevent circular locks
        from inference import solve_with_artwork

        # 1. ANALOGY & TOURNAMENT
        all_learned = self.memory.graph.get("learned_macros", {})
        input_is_list = isinstance(input_val, list)
        stateful_prims = Primitives.get_important_atoms()

        # Check existing knowledge first (Analogy Scan)
        for task_name, versions in all_learned.items():
            for chain in versions:
                is_stateful = any(a in stateful_prims for a in chain)
                if input_is_list != is_stateful: continue

                trace = self.executor.run_sequence(input_val, chain, memory=self.memory)
                if trace and trace[-1] == target_val:
                    print(f"💡 Recognized Pattern: Linked to '{task_name}'")
                    return chain, {"survivors": [{"logic": chain, "trace": trace}]}

        # 2. DISCOVERY
        print(f"🔍 Discovery for {task_label}...")

        # 2.1. SEQUENCE DISCOVERY (Optimized for Recursive Logic)
        if input_is_list:
            priority_atoms = stateful_prims + [a for a in self.atoms if a not in stateful_prims]
            for atom_name in priority_atoms:
                temp_trace, chain, last_val = list(input_val), [], input_val[-1]

                for _ in range(30):  # Allow enough depth for Fibonacci
                    func = getattr(Primitives, atom_name)
                    next_val = func(temp_trace[-1], temp_trace)

                    # Hardened Safety Breaks
                    if next_val == last_val and len(temp_trace) > len(input_val): break
                    if next_val > target_val: break
                    if next_val > 1_000_000: break

                    temp_trace.append(next_val)
                    chain.append(atom_name)
                    last_val = next_val

                    if next_val == target_val:
                        print(f"🧬 Recursive Sequence Discovered: {atom_name}")
                        self.memory.store(task_label, chain)
                        return chain, {"survivors": [{"logic": chain, "trace": temp_trace}]}
            return None, {"survivors": []}

        # 2.2. NEURAL SEARCH (Arithmetic Discovery)
        podium = solve_with_artwork(input_val, target_val, available_atoms=self.atoms + list(all_learned.keys()))
        if podium and isinstance(podium, list):
            validator = LogicValidator(self.executor)
            compressor = LogicCompressor()
            # Store all valid candidates
            for cand in podium:
                if validator.verify(cand['logic'], input_val, self.memory):
                    self.memory.store(task_label, compressor.compress(cand['logic']))
            # Distill the best into neural weights
            if self.trainer: self.trainer.distill(input_val, target_val, podium[0]['logic'], self.atoms)
            return podium[0]['logic'], {"survivors": podium}

        return None, {"survivors": []}