import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# 1. THE DATASET: Now loading 3 columns
class LogicDataset(Dataset):
    def __init__(self, csv_file, atom_list):
        df = pd.read_csv(csv_file)

        # FIX: Include 'context' in the values
        # This makes the shape [Batch, 3] instead of [Batch, 2]
        self.inputs = torch.tensor(df[['context', 'input', 'target']].values, dtype=torch.float32)

        self.atom_to_idx = {atom: i + 1 for i, atom in enumerate(atom_list)}
        self.atom_to_idx['<PAD>'] = 0

        self.labels = []
        for chain in df['logic_chain']:
            tokens = [self.atom_to_idx[a] for a in chain.split(',')]
            while len(tokens) < 3: tokens.append(0)
            self.labels.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# 2. THE MODEL (No changes needed here, just ensuring input_dim stays 3)
class NeuralArtwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocab_size):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, output_dim * vocab_size)
        self.network = nn.Sequential(self.encoder,self.relu,self.decoder)
        self.output_dim = output_dim
        self.vocab_size = vocab_size

    def forward(self, x):
        x = self.network(x)
        # Reshape to [Batch, Steps, Atoms]
        return x.view(-1, self.output_dim, self.vocab_size)

# 3. THE TRAINING LOOP
def train():
    from core import Primitives
    atoms = Primitives.get_all_atoms()

    # Ensure the CSV exists before loading
    dataset = LogicDataset("neural_artwork_data.csv", atoms)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # input_dim=3 matches our new [context, input, target] tensor
    model = NeuralArtwork(input_dim=3, hidden_dim=64, output_dim=3, vocab_size=len(atoms) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("🚀 Training the Neural Artwork...")
    for epoch in range(100):
        total_loss = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # 'inputs' is now [16, 3]
            loss = criterion(outputs.view(-1, len(atoms) + 1), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss = loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Logic Clarity (Loss): {total_loss:.4f}")

    torch.save(model.state_dict(), "logic_artwork.pth")
    print("🎨 Artwork complete and saved as 'logic_artwork.pth'")