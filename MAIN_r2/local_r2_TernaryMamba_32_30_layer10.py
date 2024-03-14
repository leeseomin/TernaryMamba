import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from mamba_ssm import Mamba
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from datasets import load_dataset
import csv
import pandas as pd

# Hyperparams
epochs = 100
lr = 1e-4
batch_size = 32
block_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = 25000
print_iters = 100
eval_iters = 10
eval_interval = 300
n_embed = 384
n_layers = 10
dropout = 0.2
num_proc_load_dataset = 8
expansion_factor = 4
Qb = 1


# Load and process data
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", split='train[:30%]', num_proc=num_proc_load_dataset)
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc_load_dataset,
    )
    
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(data_dir, f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        idx = 0
        for example in tqdm(dset, desc=f'writing {filename}'):
            arr[idx : idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()
    
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # Convert uint16 to int64
    train_data = train_data.astype(np.int64)
    val_data = val_data.astype(np.int64)

    train_data = torch.tensor(train_data, dtype=torch.long)
    val_data = torch.tensor(val_data, dtype=torch.long)

# Quantization function
def quantize_weights(weights):
    abs_mean = torch.mean(torch.abs(weights))
    scaled_weights = weights / (abs_mean + 1e-8)
    return torch.round(torch.clamp(scaled_weights, -1, 1))

# Mamba Block
class MambaBlock(nn.Module):
    def __init__(self, d_model, expansion_factor, SSM_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.swiglue = nn.Sequential(
            nn.Linear(d_model, expansion_factor*d_model),
            nn.SiLU(),
            nn.Linear(expansion_factor*d_model, d_model)
        )
        self.swiglue[0].weight.data = quantize_weights(self.swiglue[0].weight.data)
        self.swiglue[2].weight.data = quantize_weights(self.swiglue[2].weight.data)
        self.SSM = Mamba(d_model=d_model, d_state=SSM_dim, d_conv=4, expand=expansion_factor).to(device)

    def forward(self, x):
        x = x + self.swiglue(self.ln1(x))
        x = x + self.SSM(self.ln2(x))
        return x

# Language Model
class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, expansion_factor, SSM_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight.data = quantize_weights(self.embed.weight.data)
        self.layers = nn.ModuleList([MambaBlock(d_model, expansion_factor, SSM_dim)
                                     for _ in range(num_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight.data = quantize_weights(self.proj.weight.data)

    def forward(self, x, targets=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)

        # Scale activations to [-Qb, Qb] per token
        x_abs_max = torch.amax(torch.abs(x), dim=-1, keepdim=True)
        x_scale = Qb / (x_abs_max + 1e-8)
        x = x * x_scale

        x = self.ln_out(x)
        logits = self.proj(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training and evaluation
vocab_size = enc.max_token_value + 1
model = MambaLM(vocab_size, n_embed, n_layers, expansion_factor=expansion_factor, SSM_dim=16).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss(model, data, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Initialize lists to store loss data
train_losses = []
val_losses = []

for iter in tqdm(range(max_iters)):
    xb, yb = get_batch(train_data)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        train_loss = estimate_loss(model, train_data, eval_iters)
        val_loss = estimate_loss(model, val_data, eval_iters)
        print(f"Step {iter} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save loss data to CSV file
        with open('loss_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iter, train_loss, val_loss])

# Plot the loss graph using the saved CSV data
loss_data = pd.read_csv('loss_data.csv', header=None, names=['Iteration', 'Train Loss', 'Val Loss'])

plt.figure(figsize=(10, 6))
plt.plot(loss_data['Iteration'], loss_data['Train Loss'], label='Training Loss')
plt.plot(loss_data['Iteration'], loss_data['Val Loss'], label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss_graph.png')  # Save the loss graph as an image file
plt.show()

# Generate text
model.eval()
sample_idx = model.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=1000)
print(enc.decode(sample_idx[0].tolist()))

