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
        """The Dumb Pipe: Executes exactly what is provided."""
        if not instructions or not isinstance(instructions, list):
            return None

        current = [initial_data] if not isinstance(initial_data, list) else initial_data

        try:
            for instr in instructions:
                # If it's a known macro, we resolve it to its CHAMPION logic first
                if memory and instr in memory.graph.get("learned_macros", {}):
                    candidates = memory.recall(instr)
                    # We strictly take the first candidate (The Champion)
                    macro_logic = candidates[0] if isinstance(candidates, list) else candidates
                    current = self.run_sequence(current, macro_logic, memory=memory)
                else:
                    # Execute Primitive
                    func = getattr(Primitives, instr)
                    # Note: We pass current[-1] for value and current for history
                    next_val = func(current[-1], current)
                    current.append(next_val)
            return current
        except Exception:
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
        """Stores new logic as a candidate in the library."""
        if task_name not in self.graph["learned_macros"]:
            self.graph["learned_macros"][task_name] = []

        # Add only if this exact chain isn't already known
        if logic_chain not in self.graph["learned_macros"][task_name]:
            self.graph["learned_macros"][task_name].append(logic_chain)

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


import collections


class LogicCompressor:
    """The Architect: Simplifies redundant logic into efficient macros."""

    def compress(self, logic_chain):
        if not logic_chain or len(logic_chain) < 2:
            return logic_chain

        compressed = []
        i = 0
        while i < len(logic_chain):
            count = 1
            # Look ahead for repeating identical atoms
            while i + count < len(logic_chain) and logic_chain[i] == logic_chain[i + count]:
                count += 1

            if count > 2:
                # If an atom repeats more than twice, represent it as a 'Loop'
                # Example: ['ADD_ONE'] * 5 -> ['5x_ADD_ONE']
                compressed.append(f"{count}x_{logic_chain[i]}")
            else:
                # Otherwise, keep as is
                compressed.extend([logic_chain[i]] * count)
            i += count

        return compressed


class FitnessScorer:
    """The Judge: Ranks logic chains by efficiency, density, and simplicity."""

    @staticmethod
    def score(logic_chain, learned_macros=None):
        if not logic_chain:
            return 0

        # 1. Length Penalty: Shorter chains are more 'intelligent'
        # A 1-step solution is ALWAYS better than a 2-step solution.
        score = 100 / len(logic_chain)

        # 2. SIMPLICITY BIAS (V3.1): Penalize Macro-bloat
        # If the AI uses a learned macro, we subtract points.
        # This forces the AI to prefer Primitives (like SQUARE) over Macros
        # unless the Macro is truly the only efficient way.
        if learned_macros:
            macro_count = sum(1 for atom in logic_chain if atom in learned_macros)
            score -= (macro_count * 40) # 40 point penalty per macro used

        # 3. Diversity Bonus: Reward varied atoms (prevents lazy repetition)
        unique_atoms = len(set(logic_chain))
        diversity_bonus = unique_atoms * 5

        return score + diversity_bonus


# --- 5. THE LOGIC VALIDATOR ---
class LogicValidator:
    """The Skeptic: Verifies that a logic chain is a universal rule."""

    def __init__(self, executor):
        self.executor = executor

    def verify(self, logic_chain, original_input, memory):
        # 1. First, run standard robustness tests
        test_inputs = [random.randint(1, 20) for _ in range(3)]
        for test_in in test_inputs:
            actual_trace = self.executor.run_sequence(test_in, logic_chain, memory=memory)
            if actual_trace is None: return False

        # 2. THE ANTI-BLOAT SHIELD (Improved for v3.2)
        if not isinstance(original_input, list) and len(logic_chain) > 1:
            # Test over a range of numbers to ensure they are truly identical
            test_points = [2, 5, 10]
            for atom in Primitives.get_all_atoms():
                is_identical = True
                for val in test_points:
                    prim_res = self.executor.run_sequence(val, [atom])
                    logi_res = self.executor.run_sequence(val, logic_chain, memory=memory)

                    if not prim_res or not logi_res or prim_res[-1] != logi_res[-1]:
                        is_identical = False
                        break

                if is_identical:
                    print(f"⚠️ TRUE BLOAT: {logic_chain} is just {atom}")
                    return False # Reject the complex version

        return True


class OnlineTrainer:
    """The Teacher: Performs real-time distillation of symbolic successes."""

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def distill(self, input_val, target_val, logic_chain, base_atoms):
        """Fine-tunes the model on a single successful discovery."""
        self.model.train()

        # 1. Prepare the 'Experience'
        # We simplify the logic chain to its first atom for neural intuition training
        # (Teaching the model: 'When you see this input/target, START with this atom')
        first_atom = logic_chain[0]
        try:
            target_idx = base_atoms.index(first_atom) + 1  # +1 for NONE offset
        except ValueError:
            return  # Skip if it's a nested macro we haven't vectorized yet

        # 2. Vectorize
        if isinstance(input_val, list):
            flat_in = [input_val[0], input_val[1], target_val]
        else:
            flat_in = [0, input_val, target_val]

        input_tensor = torch.tensor([flat_in], dtype=torch.float32)
        label_tensor = torch.tensor([target_idx], dtype=torch.long)

        # 3. Backprop (The Infinite Loop)
        self.optimizer.zero_grad()
        output = self.model(input_tensor)
        # We only train on the first step's prediction for simplicity in v4
        loss = self.criterion(output[:, 0, :], label_tensor)
        loss.backward()
        self.optimizer.step()

        print(f"🧠 Online Training Complete: Loss {loss.item():.4f} | Logic {first_atom} absorbed.")


# --- 6. THE ORCHESTRATOR (I2L System) ---
class I2LSystem:
    """The CNS: Connects Memory, Intuition, and Execution."""

    def __init__(self,model=None):
        self.executor = Executor()
        self.memory = LogicMemory()
        self.generator = LogicDataGenerator()
        self.atoms = Primitives.get_all_atoms()
        self.model = model  # Pass the loaded model here
        self.trainer = OnlineTrainer(self.model) if model else None

    def evolve_ai(self, task_label, input_val, target_val):
        from inference import solve_with_artwork

        # 1. RETRIEVE CANDIDATES & INIT TOURNAMENT
        candidates = self.memory.graph["learned_macros"].get(task_label, [])
        tournament_results = {"survivors": [], "debunked": []}

        if candidates:
            # Score them to establish the "Initial Hierarchy"
            scored = sorted(
                [(FitnessScorer.score(c, self.memory.graph["learned_macros"]), c) for c in candidates],
                key=lambda x: x[0], reverse=True
            )

            # --- THE EXPERIMENT PHASE ---
            for score, chain in scored:
                trace = self.executor.run_sequence(input_val, chain, memory=self.memory)

                if trace and trace[-1] == target_val:
                    # Candidate survives the new data point
                    tournament_results["survivors"].append({"logic": chain, "score": score, "trace": trace})
                else:
                    # Candidate is debunked by the new data point
                    actual_result = trace[-1] if trace else "Execution Error"
                    tournament_results["debunked"].append({
                        "logic": chain,
                        "got": actual_result,
                        "expected": target_val
                    })

            # If we have survivors, pick the Champion and return
            if tournament_results["survivors"]:
                # Survivors are already sorted because 'scored' was sorted
                champion = tournament_results["survivors"][0]
                print(f"🏆 Tournament Winner: {champion['logic']} (Fitness: {champion['score']:.1f})")
                return champion['logic'], tournament_results

        # 2. DISCOVERY: Only if no memories survived or existed
        print(f"🔍 No valid memories for {task_label}. Starting discovery...")

        # 2.1. SEQUENCE DISCOVERY (Lists/Recursive)
        if isinstance(input_val, list):
            seq_candidates = []
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
                        s = FitnessScorer.score(chain, self.memory.graph["learned_macros"])
                        seq_candidates.append((s, chain, temp_trace))
                        break
                    if next_val > target_val: break

            if seq_candidates:
                seq_candidates.sort(key=lambda x: x[0], reverse=True)
                _, best_chain, best_trace = seq_candidates[0]
                self.memory.store(task_label, best_chain)
                return best_chain, {"survivors": [{"logic": best_chain, "trace": best_trace}], "debunked": []}

        # --- MOVED OUTSIDE THE IF BLOCK ---
        # 2.2. NEURAL SEARCH (Podium Discovery)
        learned_macro_names = list(self.memory.graph["learned_macros"].keys())
        current_atoms = self.atoms + learned_macro_names

        # Catch the full Podium List (list of dictionaries)
        podium_results = solve_with_artwork(input_val, target_val, available_atoms=current_atoms)

        # 3. SUCCESS PATH: MULTI-STORE & CHAMPION SELECTION
        if podium_results and isinstance(podium_results, list):
            validator = LogicValidator(self.executor)
            compressor = LogicCompressor()

            # The first item is our Champion (highest score)
            champion = podium_results[0]
            best_logic = champion['logic']
            best_trace = champion['trace']

            # Iterate through the podium to store ALL valid unique theories
            for candidate in podium_results:
                c_logic = candidate['logic']

                # Verify that this isn't a "lucky guess" for just one number
                if validator.verify(c_logic, input_val, self.memory):
                    clean_c = compressor.compress(c_logic)
                    self.memory.store(task_label, clean_c)
                    print(f"💾 Stored Candidate: {clean_c}")
                else:
                    print(f"⚠️ Candidate failed validation (not universal): {c_logic}")

            # Neural training on the best-performing logic
            if self.trainer:
                self.trainer.distill(input_val, target_val, compressor.compress(best_logic), self.atoms)

            print(f"✨ Primary Logic Discovered: {best_logic}")

            return best_logic, {
                "survivors": podium_results,
                "debunked": tournament_results["debunked"]
            }

        print(f"❌ Evolution failed for {task_label}")
        return None, {"survivors": [], "debunked": tournament_results["debunked"]} # Returning two Nones prevents the "non-iterable" error
