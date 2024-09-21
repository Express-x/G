import torch
import torch.nn as nn
import torch.optim as optim
from model import XLSTMModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextDataset(Dataset):
  def __init__(self, lines, max_len):
    self.lines = lines
    self.max_len = max_len

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, item):
    text = self.lines[item]
    input_ids = np.zeros((self.max_len))
    input_ids[:len(text)] = text
    return {
      'input_ids': torch.tensor(input_ids).long(),
      'attention_mask': torch.ones(self.max_len),
    }

def load_data(file_path):
    lines = []
    with open(file_path, 'r') as file:
      for line in file:
        lines.append([ord(c) for c in line.strip()])
    return lines

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)

      optimizer.zero_grad()

      outputs = model(input_ids, attention_mask=attention_mask)
      loss = criterion(outputs, outputs)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
  
    return total_loss / len(loader)

def main():
    lines = load_data('data.txt')
    dataset = TextDataset(lines, max_len=200)
   
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = XLSTMModel(input_dim=100, hidden_dim=128, output_dim=100)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        loss = train(model, device, loader, optimizer, criterion)
        print(f'Epoch {epoch+1}, Loss: {loss}' ) 

if __name__ == "__main__":
    main()
