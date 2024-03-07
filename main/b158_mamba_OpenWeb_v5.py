import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from mamba_ssm import Mamba
import matplotlib.pyplot as plt
from safetensors.torch import save_file
import numpy as np
import tiktoken
from datasets import load_dataset

# Hyperparams
epochs = 100
lr = 1e-3
batch_size = 64
block_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = 10000
print_iters = 100
eval_iters = 10
eval_interval = 300
n_embed = 384
n_layers = 6
dropout = 0.2
num_proc = 8
num_proc_load_dataset = num_proc
expansion_factor = 4

# Load and process data
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
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
        self.linear1 = nn.Linear(d_model, expansion_factor*d_model)
        self.linear2 = nn.Linear(expansion_factor*d_model, d_model)
        self.linear1.weight.data = quantize_weights(self.linear1.weight.data)
        self.linear2.weight.data = quantize_weights(self.linear2.weight.data)
        self.activation = nn.Hardtanh(-1, 1)
        self.SSM = Mamba(d_model=d_model, d_state=SSM_dim, d_conv=4, expand=expansion_factor).to(device)

    def forward(self, x):
        gated = self.activation(self.linear1(self.ln1(x)))
        x = x + self.linear2(gated)
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
        self.proj = nn.Linear(d_model, vocab_size)
        self.proj.weight.data = quantize_weights(self.proj.weight.data)

    def forward(self, x, targets=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
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
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        losses[split] = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data)
            _, loss = model(X, Y)
            losses[split][k] = loss.item()
    model.train()
    return losses

def get_batch(split, data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Initialize the figure and axes for plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
train_losses = []
val_losses = []

for iter in tqdm(range(max_iters)):
    xb, yb = get_batch('train', train_data)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter} | train loss {losses['train'].mean():.4f} | val loss {losses['val'].mean():.4f}")
        train_losses.append(losses['train'].mean())
        val_losses.append(losses['val'].mean())

        ax.clear()
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Val Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        plt.savefig('training_progress.png')

    if iter % print_iters == 0:
        model.eval()
        sample_idx = model.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=200)
        print(enc.decode(sample_idx[0].tolist()))
        model.train()

    # Save model every 5000 iterations
    if iter % 5000 == 0 and iter != 0:
        save_file(model.state_dict(), f"mamba_model_iter_{iter}.safetensors")

# Generate text
model.eval()
sample_idx = model.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=1000)
print(enc.decode(sample_idx[0].tolist()))


