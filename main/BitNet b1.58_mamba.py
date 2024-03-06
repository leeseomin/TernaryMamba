import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from mamba_ssm import Mamba

# hyperparams
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
n_heads = 6
n_layers = 6
dropout = 0.2

# Load and process data
with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda xx: [stoi[x] for x in xx]
decode = lambda xx: ''.join([itos[x] for x in xx])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

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
model = MambaLM(vocab_size, n_embed, n_layers, expansion_factor=4, SSM_dim=16).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        losses[split] = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[split][k] = loss.item()
    model.train()
    return losses

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

for iter in tqdm(range(max_iters)):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}  |  train loss {losses['train'].mean():.4f}  |  val loss {losses['val'].mean():.4f}")
        
    if iter % print_iters == 0:
        model.eval()
        sample_idx = model.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=200)
        print(decode(sample_idx[0].tolist()))
        model.train()

# Generate text
model.eval()
sample_idx = model.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=1000)
print(decode(sample_idx[0].tolist()))

