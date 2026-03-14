import json
import os

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
        """Save a verified logic sequence as a reusable Macro."""
        self.graph["learned_macros"][task_name] = logic_chain
        with open(self.file_path, 'w') as f:
            json.dump(self.graph, f, indent=4)

    def recall(self, task_name):
        return self.graph["learned_macros"].get(task_name)
